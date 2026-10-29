"""ANALYSIS_PLAN A11.1 (C1) analysis: calibrate the passive thresholds, then evaluate against the active probe.

calibrate: rule 3 (k_pass, pooled over sigma, noise-only passive-gate excess), rule 4 (s_hi,pass = 0.995
  quantile of floor-passing target S_pass, pooled over sigma) and A2.2 (s_lo,pass = largest per-sigma-cell
  value with replay rate <= 5%, all replay rows of the cell as denominator). Calibration rows only.
evaluate: passive SUPPORTED vs the frozen active imperfect_robust SUPPORTED on the same records (held-out
  sigma 0.5 and the 6 standoff levels), A4.1 geometry bootstrap, paired differences, A11 matched rules.

Usage:
  python scripts/wpc_passive_analysis.py calibrate PASSIVE_DIR OUT_DIR
  python scripts/wpc_passive_analysis.py evaluate PASSIVE_DIR OUT_DIR MAIN_TRIALS.parquet GEOMETRIES.csv STRESS_ROOT
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from isac_verify.calibration import floor_k, s_hi_value, s_lo_value
from isac_verify.config import PilotConfig
from isac_verify.verification import classify_evidence
from phase8b_heldout_analysis import PRIMARY, Bootstrap, classify, diff_entry
from wpb_reanalysis import entry

STANDOFFS = [f"standoff_{m:02d}m" for m in (3, 6, 10, 15, 20, 30)]


def load(passive_dir: Path, run: str) -> pd.DataFrame:
    frame = pd.concat([pd.read_parquet(p) for p in sorted(passive_dir.glob(f"{run}_*.parquet"))], ignore_index=True)
    for c in ("direct_reliable", "ref_det_pass"):
        frame[c] = frame[c].fillna(False).astype(bool)
    return frame


def fit(cal: pd.DataFrame) -> dict:
    ok = cal["passive_status"] == "computed"
    k = floor_k(cal.loc[ok, "floor_excess_pass"].dropna().to_numpy(), 0.01)
    floor_pass = ok & (cal["rho_pass"] > k["value"])
    s = cal["S_pass"]
    hi = s_hi_value(s[floor_pass & (cal["world"] == "target") & s.notna()].to_numpy(), 0.995)
    cells = {}
    for sigma, part in cal.groupby("sigma_map_m"):
        replay = part["world"] == "replay"
        eligible = (replay & part["direct_reliable"] & part["ref_det_pass"] & floor_pass.loc[part.index]
                    & part["S_pass"].notna() & (part["S_pass"] <= hi["value"]))
        cells[f"{sigma:g}"] = s_lo_value(part.loc[eligible, "S_pass"].to_numpy(), int(replay.sum()), 0.05)
    binding = max(cells, key=lambda c: cells[c]["value"])
    return {"floor_k": k, "s_hi": hi, "s_lo": {"value": cells[binding]["value"], "binding_sigma": binding,
                                              "cells": cells}}


def classify_passive(frame: pd.DataFrame, thresholds: dict) -> pd.Series:
    t = {"noise_sigmas": thresholds["floor_k"]["value"], "supported_min": thresholds["s_lo"]["value"],
         "supported_max": thresholds["s_hi"]["value"]}
    out = []
    for r in frame.itertuples():
        if r.passive_status != "computed" or not math.isfinite(r.S_pass):
            out.append(False)
            continue
        label, _ = classify_evidence(score=float(r.S_pass), net_wall_w=float(r.E_pass - r.N_pass),
                                     wall_noise_w=float(r.N_pass), direct_reliable=bool(r.direct_reliable),
                                     reference_detectable=bool(r.ref_det_pass), thresholds=t)
        out.append(label == "SUPPORTED")
    return pd.Series(out, index=frame.index)


def calibrate(passive_dir: Path, out: Path) -> None:
    cal = load(passive_dir, "calibration")
    assert cal["geometry_id"].nunique() == 12 and len(cal) == 12 * 2 * 32 * 5, len(cal)
    thresholds = fit(cal)
    cal["supported_pass"] = classify_passive(cal, thresholds)
    rates = cal.groupby(["sigma_map_m", "world"])["supported_pass"].mean().unstack("world")
    (out / "c1_passive_thresholds.json").write_text(json.dumps({
        "label": "A11.1 pre-registered calibration (calibration geometries only)", "rows": len(cal),
        "status_counts": cal["passive_status"].value_counts().to_dict(), **thresholds,
        "calibration_rates": {f"{s:g}": r for s, r in rates.to_dict("index").items()}}, indent=2, default=float))
    print(json.dumps(thresholds, indent=1, default=float))
    print(rates.round(4))


def evaluate(passive_dir: Path, out: Path, main_path: Path, geometries_path: Path, stress_root: Path) -> None:
    config = PilotConfig.load("configs/pilot.yaml")
    thresholds = json.loads((out / "c1_passive_thresholds.json").read_text())
    boot = Bootstrap(sorted(pd.read_csv(geometries_path)["geometry_id"]), int(config.get("phase8", "heldout",
                                                                                         "bootstrap_draws")))
    key = ["geometry_id", "world", "record_index", "sigma_map_m"]
    rows, merged_all = [], []
    for run in ["heldout", *STANDOFFS]:
        p = load(passive_dir, run)
        p["supported_pass"] = classify_passive(p, thresholds)
        active = classify(pd.read_parquet(main_path if run == "heldout" else stress_root / run / "heldout_trials.parquet"),
                          config)
        active = active[(active["method"] == PRIMARY) & (active["sigma_map_m"] == 0.5)]
        m = p.merge(active[key + ["supported", "S_c1"]], on=key, validate="one_to_one")
        assert len(m) == len(p) == len(active), (run, len(m), len(p), len(active))
        merged_all.append(m.assign(run=run))
        for world in ("target", "replay"):
            w = m[m["world"] == world]
            pas, act = entry(boot, w, "supported_pass"), entry(boot, w, "supported")
            d = diff_entry(boot, (w, "supported_pass"), (w, "supported"))
            rows.append({"label": "A11.1 pre-registered", "run": run, "world": world,
                         **{f"passive_{k}": v for k, v in pas.items()}, **{f"active_{k}": v for k, v in act.items()},
                         "passive_minus_active": d["point"], "diff_ci95_lo": d["ci95"][0], "diff_ci95_hi": d["ci95"][1]})
    table = pd.DataFrame(rows)
    get = lambda run, world, col: float(table[(table["run"] == run) & (table["world"] == world)][col].iloc[0])
    verdict = {
        "target_similar_heldout": abs(get("heldout", "target", "passive_minus_active")) <= 0.05,
        "replay_higher_15m": get("standoff_15m", "replay", "passive_point") > get("standoff_15m", "replay", "active_point"),
        "replay_higher_20m": get("standoff_20m", "replay", "passive_point") > get("standoff_20m", "replay", "active_point"),
        "both_near_zero_30m": get("standoff_30m", "replay", "passive_point") <= 0.05
        and get("standoff_30m", "replay", "active_point") <= 0.05,
    }
    merged = pd.concat(merged_all, ignore_index=True)
    merged.to_parquet(out / "c1_passive_rows.parquet", index=False)
    table.to_csv(out / "c1_passive_vs_active.csv", index=False)
    (out / "c1_verdicts.json").write_text(json.dumps({"label": "A11.1 matched rules", **verdict,
                                                      "thresholds": thresholds}, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(table[["run", "world", "passive_point", "passive_ci95_lo", "passive_ci95_hi", "active_point", "active_ci95_lo",
                 "active_ci95_hi", "passive_minus_active", "diff_ci95_lo", "diff_ci95_hi",
                 "passive_geometries_with_any_supported", "active_geometries_with_any_supported"]].round(3).to_string())
    print(json.dumps(verdict, indent=1))


def main() -> None:
    mode, passive_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    np.seterr(all="ignore")
    if mode == "calibrate":
        calibrate(passive_dir, out)
    else:
        evaluate(passive_dir, out, Path(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6]))


if __name__ == "__main__":
    main()
