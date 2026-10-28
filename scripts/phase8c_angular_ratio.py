"""ANALYSIS_PLAN A6.1 / A6.2: angular ratio Delta_phi_W / HPBW_TX and analytical standoffs per held-out geometry.

Evaluator side (uses true geometry). HPBW_TX is measured on the primary method's
wall beam, rebuilt through the unchanged pipeline (`phase7._plan_for`) for the
target world at sigma 0.5, records 0-7, twin r mod 2 (the stress grid). The
rebuilt beam direction must equal the main run's `beam_phi_deg` on the records
whose twin draw matches (0, 1, 4, 5).

phi_W(R) = atan(2d / R) (BRIEF §5.3 image method: departure angle of the mixed path).
u(R) = sqrt(R^2 + 4 d^2) - R; delta_tau_W = (u(R_A) - u(R_T)) / c (A5 operational note).

Usage (GPU, held-out approved): python scripts/phase8c_angular_ratio.py MAIN_TRIALS.parquet OUT.csv
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import torch

from isac_verify.beams import ArrayGeometry, BeamResult, beampattern, dbm_to_watt, direct_beam
from isac_verify.config import PilotConfig
from isac_verify.phase7 import _acquire_record, _plan_for, _twin_offsets
from isac_verify.phase8a import HELDOUT, accepted_geometry, approve_heldout, geometry_seed_streams, twin_draws
from isac_verify.receiver import ProcessingChoices, noise_calibration

C_LIGHT = 299_792_458.0
SIGMA, RECORDS, TWINS, METHOD = 0.5, 8, 2, "imperfect_robust"


def hpbw_deg(weights: torch.Tensor, array: ArrayGeometry) -> tuple[float, float]:
    """Full -3 dB width of the lobe holding the pattern maximum, and the maximum's angle."""
    grid = np.arange(-60.0, 60.0 + 1e-9, 0.01)
    gain = beampattern(grid.tolist(), BeamResult(weights=weights, status="ok", phi_deg=0.0), array).cpu().numpy()
    peak = int(np.argmax(gain))
    above = gain >= gain[peak] / 2.0
    lo, hi = peak, peak
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < len(grid) - 1 and above[hi + 1]:
        hi += 1
    return float(grid[hi] - grid[lo]), float(grid[peak])


def standoff_tau(r_t: float, d: float, psf_m: float) -> float:
    """Delta_R with u(R_T - Delta_R) - u(R_T) = psf_m (closed form of sqrt(R^2+4d^2) = R + k)."""
    k = math.sqrt(r_t**2 + 4 * d**2) - r_t + psf_m
    return r_t - (4 * d**2 - k**2) / (2 * k)


def standoff_phi(r_t: float, d: float, hpbw: float) -> float:
    """Delta_R with phi_W(R_T - Delta_R) - phi_W(R_T) = HPBW_TX."""
    angle = math.atan2(2 * d, r_t) + math.radians(hpbw)
    return r_t - 2 * d / math.tan(angle) if angle < math.pi / 2 else r_t


def main() -> None:
    main_trials, out_path = pd.read_parquet(sys.argv[1]), sys.argv[2]
    approve_heldout()
    config = PilotConfig.load("configs/pilot.yaml")
    choices = ProcessingChoices.from_config(config)
    noise_full = noise_calibration(choices=choices, config=config,
                                   records=int(config.get("phase6", "processing", "noise_calibration_records")),
                                   seed=int(config.get("phase6", "processing", "noise_calibration_seed")))
    array = ArrayGeometry.from_config(config)
    direct_weights = direct_beam(float(config.get("experiment_a", "reference_beam_phi_deg")),
                                 dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")), array).weights
    reference = main_trials[(main_trials["method"] == METHOD) & (main_trials["world"] == "target")
                            & (main_trials["sigma_map_m"] == SIGMA)].set_index(["geometry_id", "record_index"])
    rows = []
    for slot in range(int(config.get("phase8", "geometry", "heldout_geometries"))):
        gid = f"ho_{slot:02d}"
        geometry, config_g, prepared, _ = accepted_geometry(config, slot, HELDOUT)
        seeds = {"streams": geometry_seed_streams(config, HELDOUT, slot, RECORDS)}
        twin_z = twin_draws(config, HELDOUT, slot, TWINS)
        widths, checked = [], 0
        for record in range(RECORDS):
            base = _acquire_record(config=config_g, choices=choices, world="target", record_index=record,
                                   direct_weights=direct_weights, links=prepared["links"],
                                   calibration=prepared["calibration"], noise_full=noise_full, timing={}, seeds=seeds)
            z = twin_z[record % TWINS]
            plan, _, _ = _plan_for(config=config_g, choices=choices, base=base, world="target", method=METHOD,
                                   sigma_m=SIGMA, twin_offsets=_twin_offsets(config_g, SIGMA, z["positive"], z["negative"]),
                                   direct_weights=direct_weights, timing={})
            width, peak_deg = hpbw_deg(plan.beam.weights, array)
            widths.append(width)
            if record % 4 == record % TWINS:
                assert float(plan.beam.phi_deg) == float(reference.loc[(gid, record), "beam_phi_deg"]), (gid, record)
                checked += 1
        r_t, r_a, d = geometry["candidate_distance_m"], geometry["relay_distance_m"], geometry["half_width_m"]
        phi_t = math.degrees(math.atan2(2 * d, r_t))
        hpbw = float(np.median(widths))
        psf_ns = float(main_trials[main_trials["geometry_id"] == gid]["point_spread_delay_ns"].median())
        rows.append({
            "geometry_id": gid, "r_t_m": r_t, "r_a_m": r_a, "half_width_m": d,
            "phi_w_rt_deg": phi_t, "beam_phi_deg_record0": float(reference.loc[(gid, 0), "beam_phi_deg"]),
            "delta_phi_w_deg": math.degrees(math.atan2(2 * d, r_a)) - phi_t,
            "hpbw_tx_deg_median": hpbw, "hpbw_tx_deg_min": min(widths), "hpbw_tx_deg_max": max(widths),
            "beam_direction_checks": checked,
            "psf_baseline_ns": psf_ns,
            "delta_r_tau_baseline_m": standoff_tau(r_t, d, C_LIGHT * psf_ns * 1e-9),
            "delta_r_phi_m": standoff_phi(r_t, d, hpbw),
        })
        print(gid, f"hpbw={hpbw:.3f}", f"checks={checked}", flush=True)
    out = pd.DataFrame(rows)
    out["angular_ratio"] = out["delta_phi_w_deg"] / out["hpbw_tx_deg_median"]
    # Self-check: the closed-form standoffs reproduce their defining equations.
    for _, g in out.iterrows():
        u = lambda r: math.sqrt(r**2 + 4 * g.half_width_m**2) - r  # noqa: E731
        ra = g.r_t_m - g.delta_r_tau_baseline_m
        assert abs(u(ra) - u(g.r_t_m) - C_LIGHT * g.psf_baseline_ns * 1e-9) < 1e-6
    out.to_csv(out_path, index=False)
    print(out[["geometry_id", "delta_phi_w_deg", "hpbw_tx_deg_median", "angular_ratio",
               "delta_r_tau_baseline_m", "delta_r_phi_m"]].sort_values("angular_ratio").to_string())


if __name__ == "__main__":
    main()
