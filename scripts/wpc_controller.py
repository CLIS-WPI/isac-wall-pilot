"""ANALYSIS_PLAN A11.2 (C2): closed-loop controller (no LLM) on the held-out run and the 15 m standoff level.

First probe = the stored imperfect_robust row at sigma 0.5 (frozen decision). SUPPORTED with S >= s_esc:
authorize. SUPPORTED with S < s_esc: run a FRESH 245.76 MHz imperfect_robust trial (new GPU trial; same
geometry, hypothesis, record index and twin draw; probe-symbol, receiver-noise and relay-noise seed bases
replaced by new disjoint bases) and authorize only if it is SUPPORTED. Otherwise hold.

Usage:
  python scripts/wpc_controller.py trials RUN SLOT FIRST_TRIALS.parquet OUT_DIR    (GPU; RUN = heldout | standoff_15m)
  python scripts/wpc_controller.py analyze OUT_DIR MAIN_TRIALS.parquet STANDOFF15_TRIALS.parquet GEOMETRIES.csv
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S_ESC = 0.17839032918135744  # docs/results/paper/e2_escalation_README.json (calibration only)
FRESH_BASES = {"probe_symbols": 33201, "receiver_noise": 34201, "relay_noise": 35201}  # A11 notes
TWINS = {"heldout": 4, "standoff_15m": 2}


def first_probes(trials: pd.DataFrame, config) -> pd.DataFrame:
    from phase8b_heldout_analysis import PRIMARY, classify

    f = classify(trials, config)
    f = f[(f["method"] == PRIMARY) & (f["sigma_map_m"] == 0.5)].copy()
    f["escalate"] = f["supported"] & (f["S_c1"] < S_ESC)
    return f


def fresh_config(config, run: str):
    from isac_verify.config import PilotConfig
    from isac_verify.stress import stress_config

    data = copy.deepcopy(stress_config(config, "bw246").data)
    for kind, base in FRESH_BASES.items():
        data["seeds"]["streams"][kind]["heldout"] = [base + i for i in range(24)]
    used = {v for name, s in config.section("seeds", "streams").items() for v in s["heldout"] + s["calibration"]}
    assert not used & {v for b in FRESH_BASES.values() for v in range(b, b + 24)}
    if run == "standoff_15m":
        data["phase8"].setdefault("stress_runtime", {})["relay_standoff_m"] = float(
            stress_config(config, run).section("phase8", "stress_runtime")["relay_standoff_m"])
    return PilotConfig(path=config.path, data=data)


def trials(run: str, slot: int, first_path: Path, out_dir: Path) -> None:
    from isac_verify.beams import ArrayGeometry, dbm_to_watt, direct_beam
    from isac_verify.config import PilotConfig
    from isac_verify.phase7 import _acquire_record, evaluate_record
    from isac_verify.phase8a import (HELDOUT, accepted_geometry, approve_heldout, geometry_config,
                                     geometry_seed_streams, prepare_geometry, twin_draws)
    from isac_verify.receiver import ProcessingChoices, noise_calibration
    from phase8b_heldout_analysis import classify

    out_path = out_dir / f"{run}_{slot:02d}.parquet"
    assert not out_path.exists(), out_path
    approve_heldout()
    base_config = PilotConfig.load("configs/pilot.yaml")
    gid = f"ho_{slot:02d}"
    first = first_probes(pd.read_parquet(first_path), base_config)
    first = first[first["geometry_id"] == gid]
    todo = first[first["escalate"]][["world", "record_index"]].to_records(index=False).tolist()
    rows = []
    if todo:
        config = fresh_config(base_config, run)
        choices = ProcessingChoices.from_config(config)
        assert choices.profile.fft_size == 4096
        noise_full = noise_calibration(choices=choices, config=config,
                                       records=int(config.get("phase6", "processing", "noise_calibration_records")),
                                       seed=int(config.get("phase6", "processing", "noise_calibration_seed")))
        direct_weights = direct_beam(float(config.get("experiment_a", "reference_beam_phi_deg")),
                                     dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
                                     ArrayGeometry.from_config(config)).weights
        geometry, config_g, prepared, _ = accepted_geometry(config, slot, HELDOUT)
        standoff = config.section("phase8").get("stress_runtime", {}).get("relay_standoff_m")
        if standoff is not None:  # same placement as phase8a._run_slots (A5.2)
            geometry = {**geometry, "relay_distance_m": geometry["candidate_distance_m"] - float(standoff)}
            config_g = geometry_config(config, geometry)
            prepared = prepare_geometry(config_g)
            assert prepared["rejection"] is None
        ref = first.iloc[0]
        for col in ("candidate_distance_m", "relay_distance_m", "half_width_m"):
            assert float(geometry[col]) == float(ref[col]), (gid, col)
        records = int(first["record_index"].max()) + 1
        seeds = {"streams": geometry_seed_streams(config, HELDOUT, slot, records)}
        twin_z = twin_draws(config, HELDOUT, slot, TWINS[run])  # same twin draw as the first probe
        for world, record in todo:
            base = _acquire_record(config=config_g, choices=choices, world=world, record_index=int(record),
                                   direct_weights=direct_weights, links=prepared["links"],
                                   calibration=prepared["calibration"], noise_full=noise_full, timing={}, seeds=seeds)
            out = evaluate_record(config=config_g, choices=choices, world=world, record_index=int(record), base=base,
                                  sigmas=[0.5], twin_z=twin_z, seeds=seeds, direct_weights=direct_weights,
                                  links=prepared["links"], calibration=prepared["calibration"], noise_full=noise_full,
                                  noise_by_symbols={}, timing={}, representative={},
                                  true_candidate_deg=prepared["true_candidate_deg"],
                                  true_relay_deg=prepared["true_relay_deg"], methods=("imperfect_robust",),
                                  geometry_id=gid, split=HELDOUT, inflations=(1.0,), true_wall=prepared["true_wall"])
            rows += out
        fresh = classify(pd.DataFrame(rows), base_config)
        fresh["run"] = run
        chk = fresh.merge(first[["world", "record_index", "twin_index"]], on=["world", "record_index"],
                          suffixes=("", "_first"), validate="one_to_one")
        assert len(chk) == len(fresh) and (chk["twin_index"] == chk["twin_index_first"]).all()
        fresh.to_parquet(out_path, index=False)
    else:
        pd.DataFrame().to_parquet(out_path, index=False)
    print(f"{run} {gid}: escalated={len(todo)} fresh_trials={len(rows)}", flush=True)


def analyze(out_dir: Path, main_path: Path, s15_path: Path, geometries_path: Path) -> None:
    from isac_verify.config import PilotConfig
    from phase8b_heldout_analysis import Bootstrap
    from wpb_reanalysis import entry

    config = PilotConfig.load("configs/pilot.yaml")
    boot = Bootstrap(sorted(pd.read_csv(geometries_path)["geometry_id"]), int(config.get("phase8", "heldout",
                                                                                         "bootstrap_draws")))
    key = ["geometry_id", "world", "record_index"]
    rows = []
    for run, path in (("heldout", main_path), ("standoff_15m", s15_path)):
        first = first_probes(pd.read_parquet(path), config)
        parts = [pd.read_parquet(p) for p in sorted(out_dir.glob(f"{run}_*.parquet"))]
        fresh = pd.concat([p for p in parts if len(p)], ignore_index=True)
        m = first.merge(fresh[key + ["supported", "S_c1", "status"]].rename(
            columns={"supported": "fresh_supported", "S_c1": "fresh_S", "status": "fresh_status"}), on=key, how="left")
        assert int(m["escalate"].sum()) == len(fresh) == int(m["fresh_supported"].notna().sum()), run
        m["authorized"] = m["supported"] & (~m["escalate"] | m["fresh_supported"].fillna(False).astype(bool))
        m["held"] = ~m["authorized"]
        m["cost_wall"] = 1.0 + 2.0 * m["escalate"]
        m["cost_with_direct"] = 1.0 + 4.0 * m["escalate"]
        m.to_parquet(out_dir / f"c2_{run}_decisions.parquet", index=False)
        for world in ("target", "replay"):
            w = m[m["world"] == world]
            col = "authorized" if world == "replay" else "held"
            rows.append({"label": "A11.2 pre-registered", "run": run, "world": world,
                         "reported": "authorized actions" if world == "replay" else "held actions",
                         "count": int(w[col].sum()), "candidates": len(w), **entry(boot, w, col),
                         "first_probe_supported": float(w["supported"].mean()), "escalated": int(w["escalate"].sum()),
                         "fresh_supported": int(w["fresh_supported"].fillna(False).astype(bool).sum()),
                         "mean_cost_wall_probes": float(w["cost_wall"].mean()),
                         "mean_cost_incl_fresh_direct": float(w["cost_with_direct"].mean())})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "c2_controller.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.drop(columns=["label"]).round(4).to_string())


def main() -> None:
    np.seterr(all="ignore")
    if sys.argv[1] == "trials":
        trials(sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
    else:
        analyze(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))


if __name__ == "__main__":
    main()
