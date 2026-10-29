"""ANALYSIS_PLAN A11.1 (C1): passive twin baseline from the SAME direct action, no extra probe.

For one geometry slot of one run (`calibration`, `heldout` = main held-out run, or a standoff level), every
record's direct action is recomputed with the stored seeds (`phase7._acquire_record`). The imperfect_robust
plan for each sigma gives the twin wall path, the wall-gate delay window and the twin map under the direct
beam (`phase7._plan_for`). The passive gate keeps that delay window and moves the angle window to the
reciprocal path's arrival angle (A11 notes). Planner side uses only measured quantities and the twin; the
world label and the stored rows are used only for the self-check and the output labels.

Self-check: recomputed E_D and (where the robust beam exists) beam_phi_deg equal the stored robust row.

Usage (GPU): python scripts/wpc_passive.py RUN SLOT REFERENCE_TRIALS.parquet OUT_DIR [a12]
  RUN = calibration | heldout | standoff_03m ... standoff_30m
  a12: ANALYSIS_PLAN A12.4 re-run with the fixed receiver; the stored rows predate A12.1, so their E_D and
       beam_phi_deg are recorded next to the recomputed values (E_D_stored, beam_phi_stored) instead of asserted.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from isac_verify.beams import ArrayGeometry, dbm_to_watt, direct_beam, uncertainty_wall_angles_deg
from isac_verify.config import PilotConfig
from isac_verify.phase7 import (OUTSIDE_TWIN_FREE_SPACE, TwinFreeSpaceError, _acquire_record, _noise_only_map,
                                _plan_for, _twin_map, _twin_offsets, _twin_paths)
from isac_verify.phase8a import (HELDOUT, SPLIT, accepted_geometry, approve_heldout, geometry_config,
                                 geometry_seed_streams, noise_only_seed_fn, prepare_geometry, twin_draws)
from isac_verify.receiver import ProcessingChoices, gate_energy, noise_calibration
from isac_verify.stress import stress_config
from isac_verify.verification import GateWindow, _twin_interval

METHOD = "imperfect_robust"


def energy(m, gate: GateWindow):
    return gate_energy(m, delay_min_s=gate.delay_min_s, delay_max_s=gate.delay_max_s,
                       aoa_min_deg=gate.aoa_min_deg, aoa_max_deg=gate.aoa_max_deg)


def passive_gate(config: PilotConfig, plan, candidate, offsets: dict, sigma_m: float) -> GateWindow:
    """Reciprocal mixed path: same delay window as the active wall gate; arrival angle at the wall angle."""
    quantiles = tuple(float(v) for v in config.section("phase6", "gates")["wall_interval_quantiles"])
    sigma_range = max(0.5 * config.speed_of_light * candidate.delay_sigma_s,
                      float(config.get("phase6", "diagnostic", "candidate_range_uncertainty_floor_m")))
    angles, _ = uncertainty_wall_angles_deg(candidate_range_m=candidate.range_m,
                                            wall_offset_m=float(offsets[plan.wall_side]), sigma_map_m=sigma_m,
                                            sigma_candidate_range_m=sigma_range,
                                            num_samples=int(config.get("beam", "num_robust_samples")))
    low, high = _twin_interval(angles.numpy(), quantiles=quantiles, center=plan.predicted_aod_deg)
    return GateWindow(delay_min_s=plan.wall_gate.delay_min_s, delay_max_s=plan.wall_gate.delay_max_s,
                      aoa_min_deg=low - plan.gate_expansion_aoa_deg, aoa_max_deg=high + plan.gate_expansion_aoa_deg,
                      center_delay_s=plan.wall_gate.center_delay_s, center_aoa_deg=plan.predicted_aod_deg)


def main() -> None:
    run, slot, ref_path, out_dir = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    a12 = sys.argv[5:] == ["a12"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run}_{slot:02d}.parquet"
    assert not out_path.exists(), out_path
    base_config = PilotConfig.load("configs/pilot.yaml")
    split = SPLIT if run == "calibration" else HELDOUT
    if split == HELDOUT:
        approve_heldout()
    config = stress_config(base_config, run) if run.startswith("standoff_") else base_config
    grid = config.section("phase8", "grid")
    sigmas = [float(v) for v in grid["sigma_map_m"]] if run == "calibration" else [0.5]
    records, twins = int(grid["records_per_world"]), int(grid["twin_realizations_per_sigma"])
    declared = [float(v) for v in config.section("phase8").get("stress_runtime", {}).get(
        "declared_sigma_map_m", grid["sigma_map_m"])]
    noise_seed = noise_only_seed_fn(config, split, slot, len(declared))
    choices = ProcessingChoices.from_config(config)
    noise_full = noise_calibration(choices=choices, config=config,
                                   records=int(config.get("phase6", "processing", "noise_calibration_records")),
                                   seed=int(config.get("phase6", "processing", "noise_calibration_seed")))
    direct_weights = direct_beam(float(config.get("experiment_a", "reference_beam_phi_deg")),
                                 dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
                                 ArrayGeometry.from_config(config)).weights
    geometry, config_g, prepared, _ = accepted_geometry(config, slot, split)
    standoff = config.section("phase8").get("stress_runtime", {}).get("relay_standoff_m")
    if standoff is not None:  # same placement as phase8a._run_slots (A5.2)
        geometry = {**geometry, "relay_distance_m": geometry["candidate_distance_m"] - float(standoff)}
        config_g = geometry_config(config, geometry)
        prepared = prepare_geometry(config_g)
        assert prepared["rejection"] is None
    gid = f"{'ho' if split == HELDOUT else 'cal'}_{slot:02d}"
    seeds = {"streams": geometry_seed_streams(config, split, slot, records)}
    twin_z = twin_draws(config, split, slot, twins)
    ref = pd.read_parquet(ref_path)
    ref = ref[(ref["geometry_id"] == gid) & (ref["method"] == METHOD)].set_index(
        ["world", "record_index", "sigma_map_m"])
    eps = float(config.get("phase6", "evidence", "epsilon_w"))
    floor = 10.0 ** (float(config.get("phase6", "evidence", "reference_floor_snr_db")) / 10.0)
    rows, checks = [], 0
    for world in ("target", "replay"):
        for record in range(records):
            base = _acquire_record(config=config_g, choices=choices, world=world, record_index=record,
                                   direct_weights=direct_weights, links=prepared["links"],
                                   calibration=prepared["calibration"], noise_full=noise_full, timing={},
                                   seeds=seeds)
            for sigma in sigmas:
                twin_idx = record % twins
                stored = ref.loc[(world, record, sigma)]
                assert int(stored["twin_index"]) == twin_idx
                row = {"run": run, "geometry_id": gid, "world": world, "record_index": record, "sigma_map_m": sigma,
                       "twin_index": twin_idx, "passive_status": "computed"}
                if base.get(OUTSIDE_TWIN_FREE_SPACE):
                    rows.append({**row, "passive_status": OUTSIDE_TWIN_FREE_SPACE})
                    continue
                if base["point_spread"] is None:
                    rows.append({**row, "passive_status": "acquisition_failed"})
                    continue
                direct = base["direct_energy"]
                if a12:
                    row |= {"E_D_stored": float(stored["E_D"]), "beam_phi_stored": float(stored["beam_phi_deg"])}
                else:
                    assert float(direct.energy_w) == float(stored["E_D"]), (gid, world, record, sigma)
                offsets = _twin_offsets(config_g, sigma, twin_z[twin_idx]["positive"], twin_z[twin_idx]["negative"])
                try:
                    plan, direct_twin, _ = _plan_for(config=config_g, choices=choices, base=base, world=world,
                                                     method=METHOD, sigma_m=sigma, twin_offsets=offsets,
                                                     direct_weights=direct_weights, timing={})
                except TwinFreeSpaceError:
                    rows.append({**row, "passive_status": OUTSIDE_TWIN_FREE_SPACE})
                    continue
                if plan.wall_gate is None:
                    rows.append({**row, "passive_status": "no_twin_wall_path"})
                    continue
                if plan.beam is not None and plan.beam.weights is not None:
                    if a12:
                        row["beam_phi_deg"] = float(plan.beam.phi_deg)
                    else:
                        assert float(plan.beam.phi_deg) == float(stored["beam_phi_deg"]), (gid, world, record, sigma)
                    checks += 1
                else:  # robust beam unusable: the passive score still needs the twin map under the direct beam
                    direct_twin = _twin_map(config=config_g, choices=choices, weights=direct_weights,
                                            twin_paths=_twin_paths(config_g, base["candidate"], offsets, {}),
                                            timing={})
                gate = passive_gate(config_g, plan, base["candidate"], offsets, sigma)
                measured = energy(base["direct_map"], gate)
                twin_pass, twin_direct = energy(direct_twin, gate), energy(direct_twin, plan.direct_gate)
                r_pass = twin_pass.energy_w / twin_direct.energy_w if twin_direct.energy_w > 0 else float("nan")
                score = (measured.net_energy_w / max(direct.net_energy_w, eps) / r_pass
                         if math.isfinite(r_pass) and r_pass > 0 else float("nan"))
                row |= {"E_D": direct.energy_w, "N_D": direct.noise_w, "E_pass": measured.energy_w,
                        "N_pass": measured.noise_w, "r_pass": r_pass, "S_pass": score,
                        "rho_pass": (measured.net_energy_w / measured.noise_w if measured.noise_w > 0 else float("nan")),
                        "direct_reliable": bool(direct.net_energy_w > direct.noise_w * floor and math.isfinite(r_pass)
                                                and r_pass > 0),
                        "ref_det_pass": bool(twin_pass.energy_w > measured.noise_w * floor
                                             and twin_direct.energy_w > direct.noise_w * floor),
                        "wall_side": plan.wall_side, "gate_delay_min_ns": gate.delay_min_s * 1e9,
                        "gate_delay_max_ns": gate.delay_max_s * 1e9, "gate_aoa_min_deg": gate.aoa_min_deg,
                        "gate_aoa_max_deg": gate.aoa_max_deg, "gate_cells": measured.cells,
                        "direct_gate_aoa_min_deg": plan.direct_gate.aoa_min_deg,
                        "direct_gate_aoa_max_deg": plan.direct_gate.aoa_max_deg}
                if run == "calibration":
                    noise_map = _noise_only_map(config=config_g, choices=choices, direct_weights=direct_weights,
                                                noise_full=noise_full,
                                                seed=noise_seed(world, declared.index(sigma), record), timing={})
                    n = energy(noise_map, gate)
                    row["floor_excess_pass"] = (n.energy_w - n.noise_w) / n.noise_w
                rows.append(row)
            del base
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"{run} {gid}: rows={len(rows)} beam_checks={checks} "
          f"computed={sum(r['passive_status'] == 'computed' for r in rows)}", flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
