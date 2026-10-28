"""Phase 8C exploratory (post hoc) diagnostic: replay-world wall-gate energy by relay path pair.

Evaluator side, noiseless, no new trials. For one geometry and level, the
primary method's (`imperfect_robust`) wall beam is rebuilt through the
unchanged pipeline for the replay world, record 0, sigma 0.5, twin 0
(`phase7._acquire_record` / `_plan_for`). The rebuilt beam direction must equal
the run's `beam_phi_deg` for that row. Each relay path pair (downlink path i x
uplink path j) is then observed alone, noiseless, through the same receiver map,
and its energy in the planned wall gate is reported, together with the
departure angle, the beam gain at it, the arrival delay and the AoA at the gNB.
Pair energies do not add exactly (cross terms); the sum is reported.

Usage (GPU, held-out approved):
  python scripts/phase8c_pair_probe.py OUT.csv GEOMETRY_ID:LEVEL:TRIALS.parquet [...]
LEVEL is `main` (no stress) or a declared stress level (e.g. standoff_20m, bw246).
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import torch

from isac_verify.beams import ArrayGeometry, BeamResult, beam_gain, dbm_to_watt, direct_beam
from isac_verify.config import PilotConfig
from isac_verify.phase7 import _acquire_record, _map, _observe, _plan_for, _twin_offsets
from isac_verify.phase8a import (HELDOUT, accepted_geometry, approve_heldout, geometry_config,
                                 geometry_seed_streams, prepare_geometry, twin_draws)
from isac_verify.receiver import ProcessingChoices, gate_energy, noise_calibration
from isac_verify.stress import stress_config
from isac_verify.waveform import qpsk_symbols

SIGMA, METHOD, RECORD = 0.5, "imperfect_robust", 0


def level_config(config: PilotConfig, level: str) -> PilotConfig:
    return config if level == "main" else stress_config(config, level)


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
    z = twin_draws(cfg, HELDOUT, slot, 2)[0]
    links, calibration = prepared["links"], prepared["calibration"]
    base = _acquire_record(config=config_g, choices=choices, world="replay", record_index=RECORD,
                           direct_weights=direct_weights, links=links, calibration=calibration,
                           noise_full=noise_full, timing={}, seeds=seeds)
    plan, _, _ = _plan_for(config=config_g, choices=choices, base=base, world="replay", method=METHOD,
                           sigma_m=SIGMA, twin_offsets=_twin_offsets(config_g, SIGMA, z["positive"], z["negative"]),
                           direct_weights=direct_weights, timing={})
    row = trials[(trials["geometry_id"] == gid) & (trials["world"] == "replay") & (trials["method"] == METHOD)
                 & (trials["sigma_map_m"] == SIGMA) & (trials["record_index"] == RECORD)].iloc[0]
    assert float(plan.beam.phi_deg) == float(row["beam_phi_deg"]), (gid, level, plan.beam.phi_deg, row["beam_phi_deg"])
    weights, gate = plan.beam.weights, plan.wall_gate
    beam = BeamResult(weights=weights, status="ok", phi_deg=0.0)
    peak = max(beam_gain(float(a), beam, array) for a in np.arange(-60.0, 60.0 + 1e-9, 0.05))
    wall_symbols = int(seeds["streams"]["wall_action"][METHOD]["probe_symbols"][RECORD])

    def energy(*, background=(), down=(), up=()) -> float:
        obs = _observe(config=config_g, world="replay" if down else "background", beam_weights=weights,
                       symbols=qpsk_symbols(choices.profile, wall_symbols),
                       profile=choices.profile, target_links=(), background_links=background,
                       downlink_links=down, uplink_links=up, calibration=calibration,
                       probe_seed=wall_symbols, noise_seed=0, relay_noise_seed=0, noiseless=True)
        return gate_energy(_map(obs, choices, noise_full, {}), delay_min_s=gate.delay_min_s,
                           delay_max_s=gate.delay_max_s, aoa_min_deg=gate.aoa_min_deg,
                           aoa_max_deg=gate.aoa_max_deg).energy_w

    relay_total = energy(down=links["downlink"], up=links["uplink"])
    background_total = energy(background=links["background"])
    common = {
        "geometry_id": gid, "level": level, "run_status": row["status"], "run_supported": bool(row["supported"]),
        "beam_phi_deg": float(plan.beam.phi_deg), "wall_side": plan.wall_side,
        "predicted_wall_delay_ns": plan.predicted_delay_s * 1e9,
        "gate_delay_min_ns": gate.delay_min_s * 1e9, "gate_delay_max_ns": gate.delay_max_s * 1e9,
        "gate_aoa_min_deg": gate.aoa_min_deg, "gate_aoa_max_deg": gate.aoa_max_deg,
        "candidate_delay_ns": base["candidate"].delay_s * 1e9,
        "relay_added_delay_ns": calibration.measured_added_delay_s * 1e9,
        "relay_distance_m": geometry["relay_distance_m"], "candidate_distance_m": geometry["candidate_distance_m"],
        "gate_energy_relay_all_w": relay_total, "gate_energy_background_w": background_total,
    }
    out = []
    for d in links["downlink"]:
        for u in links["uplink"]:
            e = energy(down=(d,), up=(u,))
            out.append({**common, "downlink": d.label.split(":", 1)[1], "uplink": u.label.split(":", 1)[1],
                        "aod_deg": d.aod_deg,
                        "beam_gain_rel_peak_db": 10 * math.log10(max(beam_gain(d.aod_deg, beam, array), 1e-30) / peak),
                        "arrival_delay_ns": (d.delay_s + u.delay_s + calibration.measured_added_delay_s) * 1e9,
                        "aoa_deg": u.aoa_deg, "gate_energy_w": e,
                        "share_of_relay_gate_energy": e / relay_total if relay_total > 0 else math.nan})
    pairs = pd.DataFrame(out)
    print(f"{gid} {level}: status={row['status']} gate=[{common['gate_delay_min_ns']:.1f},"
          f"{common['gate_delay_max_ns']:.1f}] ns x [{gate.aoa_min_deg:.1f},{gate.aoa_max_deg:.1f}] deg "
          f"relay_all={relay_total:.3e} background={background_total:.3e} "
          f"sum_pairs/relay_all={pairs['gate_energy_w'].sum() / relay_total:.3f}", flush=True)
    return out


def main() -> None:
    out_path = sys.argv[1]
    approve_heldout()
    config = PilotConfig.load("configs/pilot.yaml")
    rows = []
    for spec in sys.argv[2:]:
        gid, level, trials_path = spec.split(":", 2)
        rows += probe(config, gid, level, pd.read_parquet(trials_path))
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    cols = ["geometry_id", "level", "downlink", "uplink", "aod_deg", "beam_gain_rel_peak_db", "arrival_delay_ns",
            "aoa_deg", "share_of_relay_gate_energy"]
    print(out.sort_values(["geometry_id", "level", "share_of_relay_gate_energy"], ascending=[True, True, False])
          .groupby(["geometry_id", "level"]).head(6)[cols].round(3).to_string())


if __name__ == "__main__":
    main()
