"""EXPLORATORY, post hoc (D3): relay energy in the active wall gate vs the passive gate, by path pair.

Noiseless, no new trials. For one geometry and level, replay world, record 0, sigma 0.5 (twin = 0), the direct
action and the imperfect_robust plan are rebuilt through the unchanged pipeline (`phase7._acquire_record` /
`_plan_for`); the rebuilt beam direction must equal the run's `beam_phi_deg`. The passive gate is the A11.1 one
(`wpc_passive.passive_gate`). Each relay pair (downlink i x uplink j) is observed alone, noiseless:
  active  = wall beam, active wall gate;   passive = direct beam, passive gate (from the direct action).
Per pair: TX gain of the wall beam and of the direct beam at the downlink departure angle (rel. to each beam's
peak), and the receive response at the uplink arrival angle when the receiver is steered to the gate centre,
with the configured AoA taper (Hann) and with a rectangular one (rel. to the response at the gate centre).
Noiseless scores: S0 = relay gate energy / direct gate energy / twin ratio, for both methods.

Usage (GPU, held-out approved): python scripts/wpd_mechanism.py OUT.csv GEOMETRY_ID:LEVEL:TRIALS.parquet [...]
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import torch

from isac_verify.beams import ArrayGeometry, BeamResult, array_response, beam_gain, dbm_to_watt, direct_beam
from isac_verify.config import PilotConfig
from isac_verify.phase7 import _acquire_record, _map, _observe, _plan_for, _twin_offsets
from isac_verify.phase8a import (HELDOUT, accepted_geometry, approve_heldout, geometry_config,
                                 geometry_seed_streams, prepare_geometry, twin_draws)
from isac_verify.receiver import ProcessingChoices, _aoa_window, noise_calibration
from isac_verify.waveform import qpsk_symbols
from phase8c_pair_probe import level_config
from wpc_passive import energy as gate_e
from wpc_passive import passive_gate

SIGMA, METHOD, RECORD = 0.5, "imperfect_robust", 0
GRID = np.arange(-90.0, 90.0 + 1e-9, 0.05)


def rx_response_db(phi_deg: float, steer_deg: float, array: ArrayGeometry, window: torch.Tensor) -> float:
    """Response of the receiver's angle map, steered to map angle `steer`, to a path arriving at link angle `phi`.

    `angular_delay_map` combines sum_n a_n(steer) w_n h_n (steering_vector(steer).conj() = a(steer)), so a path
    with h ~ a(phi) peaks at map angle -phi; the probe asserts this mirror on the measured maps. Relative to peak.
    """
    a, s = array_response(phi_deg, array), array_response(steer_deg, array)
    w = window.to(a.dtype)
    value = (w * s * a).sum().abs().square() / (w * s * s.conj()).sum().abs().square()
    return 10 * math.log10(max(float(value), 1e-30))


def tx_db(phi_deg: float, beam: BeamResult, array: ArrayGeometry) -> float:
    peak = max(beam_gain(float(a), beam, array) for a in GRID)
    return 10 * math.log10(max(beam_gain(phi_deg, beam, array), 1e-30) / peak)


def probe(config: PilotConfig, gid: str, level: str, trials: pd.DataFrame) -> list[dict]:
    cfg = level_config(config, level)
    slot = int(gid.split("_")[1])
    geometry, config_g, prepared, _ = accepted_geometry(cfg, slot, HELDOUT)
    standoff = cfg.section("phase8").get("stress_runtime", {}).get("relay_standoff_m")
    if standoff is not None:  # same relay placement as phase8a._run_slots
        geometry = {**geometry, "relay_distance_m": geometry["candidate_distance_m"] - float(standoff)}
        config_g = geometry_config(cfg, geometry)
        prepared = prepare_geometry(config_g)
    choices = ProcessingChoices.from_config(cfg)
    noise_full = noise_calibration(choices=choices, config=cfg,
                                   records=int(cfg.get("phase6", "processing", "noise_calibration_records")),
                                   seed=int(cfg.get("phase6", "processing", "noise_calibration_seed")))
    array = ArrayGeometry.from_config(cfg)
    direct_weights = direct_beam(float(cfg.get("experiment_a", "reference_beam_phi_deg")),
                                 dbm_to_watt(cfg.get("waveform", "nominal_total_power_dbm")), array).weights
    seeds = {"streams": geometry_seed_streams(cfg, HELDOUT, slot, RECORD + 1)}
    twins = int(cfg.section("phase8", "grid")["twin_realizations_per_sigma"])
    z = twin_draws(cfg, HELDOUT, slot, twins)[RECORD % twins]
    links, calibration = prepared["links"], prepared["calibration"]
    base = _acquire_record(config=config_g, choices=choices, world="replay", record_index=RECORD,
                           direct_weights=direct_weights, links=links, calibration=calibration,
                           noise_full=noise_full, timing={}, seeds=seeds)
    offsets = _twin_offsets(config_g, SIGMA, z["positive"], z["negative"])
    plan, direct_twin, _ = _plan_for(config=config_g, choices=choices, base=base, world="replay", method=METHOD,
                                     sigma_m=SIGMA, twin_offsets=offsets, direct_weights=direct_weights, timing={})
    row = trials[(trials["geometry_id"] == gid) & (trials["world"] == "replay") & (trials["method"] == METHOD)
                 & (trials["sigma_map_m"] == SIGMA) & (trials["record_index"] == RECORD)].iloc[0]
    assert float(plan.beam.phi_deg) == float(row["beam_phi_deg"]), (gid, level, plan.beam.phi_deg, row["beam_phi_deg"])
    pgate, agate = passive_gate(config_g, plan, base["candidate"], offsets, SIGMA), plan.wall_gate
    wall_beam = BeamResult(weights=plan.beam.weights, status="ok", phi_deg=0.0)
    d_beam = BeamResult(weights=direct_weights, status="ok", phi_deg=0.0)
    hann = _aoa_window(choices)
    rect = torch.ones_like(hann)
    rx_array = choices.receiver_array
    streams = seeds["streams"]
    syms = {"active": int(streams["wall_action"][METHOD]["probe_symbols"][RECORD]),
            "passive": int(streams["direct"]["probe_symbols"][RECORD])}
    weights = {"active": plan.beam.weights, "passive": direct_weights}
    gates = {"active": agate, "passive": pgate}

    def gate_energy(method: str, gate, *, down=(), up=(), peak: bool = False):
        obs = _observe(config=config_g, world="replay", beam_weights=weights[method],
                       symbols=qpsk_symbols(choices.profile, syms[method]), profile=choices.profile,
                       target_links=(), background_links=(), downlink_links=down, uplink_links=up,
                       calibration=calibration, probe_seed=syms[method], noise_seed=0, relay_noise_seed=0,
                       noiseless=True)
        m = _map(obs, choices, noise_full, {})
        if peak:  # map AoA of the strongest cell over the whole map
            idx = int(torch.argmax(m.power_w).item())
            return gate_e(m, gate).energy_w, float(m.aoa_deg[idx % m.aoa_deg.numel()].item())
        return gate_e(m, gate).energy_w

    # twin ratios as the scores use them: active r from the plan, passive r_pass from the twin map
    direct_total = gate_energy("passive", plan.direct_gate, down=links["downlink"], up=links["uplink"])
    totals = {m: gate_energy(m, gates[m], down=links["downlink"], up=links["uplink"]) for m in gates}
    r_pass = gate_e(direct_twin, pgate).energy_w / gate_e(direct_twin, plan.direct_gate).energy_w
    common = {"label": "exploratory, post hoc", "geometry_id": gid, "level": level,
              "run_active_supported": bool(row["supported"]), "run_S_c1": float(row["S_c1"]),
              "wall_beam_phi_deg": float(plan.beam.phi_deg), "twin_wall_angle_deg": float(plan.predicted_aod_deg),
              "active_gate_aoa_deg": [agate.aoa_min_deg, agate.aoa_max_deg],
              "passive_gate_aoa_deg": [pgate.aoa_min_deg, pgate.aoa_max_deg],
              "gate_delay_ns": [agate.delay_min_s * 1e9, agate.delay_max_s * 1e9],
              "relay_gate_energy_active_w": totals["active"], "relay_gate_energy_passive_w": totals["passive"],
              "direct_gate_energy_noiseless_w": direct_total, "r_pass": r_pass,
              "S0_passive": totals["passive"] / direct_total / r_pass,
              "relay_distance_m": geometry["relay_distance_m"], "candidate_distance_m": geometry["candidate_distance_m"]}
    a_center, p_center = 0.5 * (agate.aoa_min_deg + agate.aoa_max_deg), plan.predicted_aod_deg
    out, checks = [], []
    for d in links["downlink"]:
        for u in links["uplink"]:
            e = {m: gate_energy(m, gates[m], down=(d,), up=(u,)) for m in gates}
            if d.label.split(":", 1)[1].startswith("los") and not u.label.split(":", 1)[1].startswith("los"):
                _, map_peak = gate_energy("passive", pgate, down=(d,), up=(u,), peak=True)
                assert abs(map_peak + u.aoa_deg) < 2.0, (gid, u.label, u.aoa_deg, map_peak)  # mirrored map axis
                checks.append((u.label, u.aoa_deg, map_peak))
            out.append({**common, "downlink": d.label.split(":", 1)[1], "uplink": u.label.split(":", 1)[1],
                        "aod_deg": d.aod_deg, "aoa_deg": u.aoa_deg, "map_aoa_deg": -u.aoa_deg,
                        "map_aoa_in_passive_gate": bool(pgate.aoa_min_deg <= -u.aoa_deg <= pgate.aoa_max_deg),
                        "arrival_delay_ns": (d.delay_s + u.delay_s + calibration.measured_added_delay_s) * 1e9,
                        "tx_wall_beam_db": tx_db(d.aod_deg, wall_beam, array),
                        "tx_direct_beam_db": tx_db(d.aod_deg, d_beam, array),
                        "rx_hann_at_passive_center_db": rx_response_db(u.aoa_deg, p_center, rx_array, hann),
                        "rx_rect_at_passive_center_db": rx_response_db(u.aoa_deg, p_center, rx_array, rect),
                        "rx_hann_at_active_center_db": rx_response_db(u.aoa_deg, a_center, rx_array, hann),
                        "energy_active_w": e["active"], "energy_passive_w": e["passive"],
                        "share_active": e["active"] / totals["active"] if totals["active"] > 0 else math.nan,
                        "share_passive": e["passive"] / totals["passive"] if totals["passive"] > 0 else math.nan})
    print(f"{gid} {level}: S_c1(run)={row['S_c1']:.4f} S0_passive={common['S0_passive']:.4f} "
          f"twin_wall={p_center:.2f} beam={plan.beam.phi_deg:.2f} active_gate_aoa={common['active_gate_aoa_deg']} "
          f"passive_gate_aoa={common['passive_gate_aoa_deg']} map_peak_checks={checks}", flush=True)
    return out


def main() -> None:
    approve_heldout()
    hann_check = torch.hann_window(8, periodic=False, dtype=torch.float64)
    assert hann_check[0] == 0 and hann_check[-1] == 0  # the configured taper is a symmetric Hann
    config = PilotConfig.load("configs/pilot.yaml")
    rows = []
    for spec in sys.argv[2:]:
        gid, level, trials_path = spec.split(":", 2)
        rows += probe(config, gid, level, pd.read_parquet(trials_path))
    out = pd.DataFrame(rows)
    out.to_csv(sys.argv[1], index=False)
    pd.set_option("display.width", 250)
    cols = ["geometry_id", "level", "downlink", "uplink", "aod_deg", "aoa_deg", "map_aoa_in_passive_gate",
            "tx_wall_beam_db", "tx_direct_beam_db", "rx_hann_at_passive_center_db", "rx_rect_at_passive_center_db",
            "rx_hann_at_active_center_db", "share_active", "share_passive", "energy_active_w", "energy_passive_w"]
    print(out[cols].round(3).to_string())


if __name__ == "__main__":
    main()
