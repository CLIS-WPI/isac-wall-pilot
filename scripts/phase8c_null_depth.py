"""ANALYSIS_PLAN A7.1: realized null depth toward the relay under D11 array calibration error.

Evaluator side (uses the true relay azimuth and the error draws). For each
held-out geometry, both worlds, records 0-7, the primary method's wall beam is
rebuilt through the unchanged pipeline on the perturbed links of realization
r mod 4 (`stress.perturb_links`), sigma 0.5, twin r mod 2. Self-checks: the
rebuilt `beam_phi_deg` equals the run's, and the ideal-array null depth of the
rebuilt beam equals the run's `null_depth_db`.

Realized null depth = 10 log10(G(phi_relay) / max G) with G the `beam_gain` of
w * e_TX on the 481-point grid over [-60, 60] deg (`phase7._null_depth`).

Usage (GPU, held-out approved): python scripts/phase8c_null_depth.py RUN_TRIALS.parquet OUT.csv
"""

from __future__ import annotations

import math
import sys

import pandas as pd
import torch

from isac_verify.beams import ArrayGeometry, BeamResult, beam_gain, dbm_to_watt, direct_beam
from isac_verify.config import PilotConfig
from isac_verify.phase7 import _acquire_record, _null_depth, _plan_for, _twin_offsets
from isac_verify.phase8a import HELDOUT, accepted_geometry, approve_heldout, geometry_seed_streams, twin_draws
from isac_verify.receiver import ProcessingChoices, noise_calibration
from isac_verify.stress import ARRAY_ERROR_LEVEL, array_error_factors, perturb_links, stress_config

SIGMA, METHOD = 0.5, "imperfect_robust"


def realized_null_depth_db(weights: torch.Tensor, e_tx: torch.Tensor, relay_deg: float, array: ArrayGeometry) -> float:
    beam = BeamResult(weights=weights * e_tx, status="ok", phi_deg=0.0)
    peak = max(beam_gain(float(a), beam, array) for a in torch.linspace(-60.0, 60.0, 481))
    return 10.0 * math.log10(beam_gain(relay_deg, beam, array) / peak)


def main() -> None:
    trials, out_path = pd.read_parquet(sys.argv[1]), sys.argv[2]
    approve_heldout()
    config = stress_config(PilotConfig.load("configs/pilot.yaml"), ARRAY_ERROR_LEVEL)
    spec = config.section("phase8", "stress_runtime", "array_error")
    grid = config.section("phase8", "grid")
    records, twins = int(grid["records_per_world"]), int(grid["twin_realizations_per_sigma"])
    choices = ProcessingChoices.from_config(config)
    noise_full = noise_calibration(choices=choices, config=config,
                                   records=int(config.get("phase6", "processing", "noise_calibration_records")),
                                   seed=int(config.get("phase6", "processing", "noise_calibration_seed")))
    array = ArrayGeometry.from_config(config)
    elements = int(config.get("arrays", "gnb_tx", "num_elements"))
    direct_weights = direct_beam(float(config.get("experiment_a", "reference_beam_phi_deg")),
                                 dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")), array).weights
    run = trials[(trials["method"] == METHOD) & (trials["sigma_map_m"] == SIGMA)].set_index(
        ["geometry_id", "world", "record_index"])
    rows = []
    for slot in range(int(config.get("phase8", "geometry", "heldout_geometries"))):
        gid = f"ho_{slot:02d}"
        _, config_g, prepared, _ = accepted_geometry(config, slot, HELDOUT)
        factors = [array_error_factors(spec, slot, k, elements) for k in range(int(spec["realizations"]))]
        seeds = {"streams": geometry_seed_streams(config, HELDOUT, slot, records)}
        twin_z = twin_draws(config, HELDOUT, slot, twins)
        for world in ("target", "replay"):
            for record in range(records):
                k = record % len(factors)
                e_tx, e_rx = factors[k]
                ref = run.loc[(gid, world, record)]
                base = _acquire_record(config=config_g, choices=choices, world=world, record_index=record,
                                       direct_weights=direct_weights, links=perturb_links(prepared["links"], e_tx, e_rx),
                                       calibration=prepared["calibration"], noise_full=noise_full, timing={},
                                       seeds=seeds)
                if base["point_spread"] is None:
                    rows.append({"geometry_id": gid, "world": world, "record_index": record, "realization": k,
                                 "status": "no_beam"})
                    continue
                z = twin_z[record % twins]
                plan, _, _ = _plan_for(config=config_g, choices=choices, base=base, world=world, method=METHOD,
                                       sigma_m=SIGMA,
                                       twin_offsets=_twin_offsets(config_g, SIGMA, z["positive"], z["negative"]),
                                       direct_weights=direct_weights, timing={})
                if plan.beam is None or plan.beam.weights is None:
                    rows.append({"geometry_id": gid, "world": world, "record_index": record, "realization": k,
                                 "status": "no_beam"})
                    continue
                w = plan.beam.weights
                assert float(plan.beam.phi_deg) == float(ref["beam_phi_deg"]), (gid, world, record)
                ideal = _null_depth(config=config_g, weights=w, true_relay_deg=prepared["true_relay_deg"])
                assert abs(ideal - float(ref["null_depth_db"])) < 1e-9, (gid, world, record, ideal, ref["null_depth_db"])
                realized = realized_null_depth_db(w, torch.as_tensor(e_tx, dtype=w.dtype, device=w.device),
                                                  prepared["true_relay_deg"], array)
                rows.append({"geometry_id": gid, "world": world, "record_index": record, "realization": k,
                             "status": "ok", "beam_phi_deg": float(plan.beam.phi_deg),
                             "true_relay_deg": float(prepared["true_relay_deg"]),
                             "ideal_null_depth_db": ideal, "realized_null_depth_db": realized})
        print(gid, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    ok = out[out["status"] == "ok"]
    print(f"rows={len(out)} ok={len(ok)} realized median={ok['realized_null_depth_db'].median():.2f} dB "
          f"ideal median={ok['ideal_null_depth_db'].median():.2f} dB")


if __name__ == "__main__":
    main()
