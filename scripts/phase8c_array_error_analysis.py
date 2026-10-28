"""Phase 8C block 3 (D11 array calibration error) analysis: ANALYSIS_PLAN A7.1 predictions.

Evaluator side. Decisions come only from configs `phase8.frozen`; intervals are
the A4.1 geometry-level bootstrap. Baseline is the main held-out run's sigma 0.5
cell (matched records {0,1,4,5} secondary). Strata use each geometry's median
delta_tau_W / PSF at this level. Null depths come from scripts/phase8c_null_depth.py.

Usage: python scripts/phase8c_array_error_analysis.py BASELINE_TRIALS.parquet GEOMETRIES.csv RUN_DIR NULL.csv OUT_DIR
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

from isac_verify.config import PilotConfig
from phase8b_heldout_analysis import PRIMARY, Bootstrap, classify, diff_entry, rate_entry, strata
from phase8c_bandwidth_analysis import MATCHED_RECORDS, SIGMA, replay_rows, stratified_rates, validity

LEVEL = "array_error_d11"


def distribution(values: pd.Series) -> dict:
    q = values.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_list()
    return {"n": int(values.notna().sum()), "min": q[0], "q25": q[1], "median": q[2], "q75": q[3], "max": q[4]}


def main() -> None:
    baseline_path, geometries_path, run_dir, null_path, out_dir = (Path(a) for a in sys.argv[1:6])
    out_dir.mkdir(parents=True, exist_ok=True)
    config = PilotConfig.load("configs/pilot.yaml")
    draws = int(config.get("phase8", "heldout", "bootstrap_draws"))
    geometries = pd.read_csv(geometries_path)
    boot = Bootstrap(sorted(geometries["geometry_id"]), draws)
    baseline = pd.read_parquet(baseline_path)
    trials = pd.read_parquet(run_dir / "heldout_trials.parquet")
    frame = classify(trials, config)
    s = strata(frame, geometries)
    frame["stratum"] = frame["geometry_id"].map(s["stratum"])

    check = validity(frame, trials, run_dir, "baseline", config)  # level_label here checks the profile is baseline
    check["realization_is_record_mod_4"] = bool((frame["array_error_realization"] == frame["record_index"] % 4).all())
    check["realizations_per_geometry"] = sorted(frame.groupby("geometry_id")["array_error_realization"].nunique().unique().tolist())

    rows = replay_rows(frame)
    ge1 = rows[rows["stratum"] == "ge1"]
    diff = diff_entry(boot, (rows, "supported"), (replay_rows(baseline), "supported"))
    target = rate_entry(boot, frame[(frame["method"] == PRIMARY) & (frame["world"] == "target")], "supported")
    ge1_rate = rate_entry(Bootstrap(sorted(ge1["geometry_id"].unique()), draws), ge1, "supported") if len(ge1) else None

    null = pd.read_csv(null_path)
    ok = null[null["status"] == "ok"]
    base_null = baseline[(baseline["method"] == PRIMARY) & (baseline["sigma_map_m"] == SIGMA)]["null_depth_db"]
    sigma2 = (0.5 * math.log(10) / 20) ** 2 + math.radians(5.0) ** 2
    result = {
        "validity": check,
        "null_depth": {
            "analytical_db": 10 * math.log10(sigma2 / int(config.get("arrays", "gnb_tx", "num_elements"))),
            "realized_db": distribution(ok["realized_null_depth_db"]),
            "realized_db_by_world": {w: distribution(g["realized_null_depth_db"]) for w, g in ok.groupby("world")},
            "realized_db_by_realization": {int(k): distribution(g["realized_null_depth_db"])
                                           for k, g in ok.groupby("realization")},
            "ideal_array_same_beams_db": distribution(ok["ideal_null_depth_db"]),
            "ideal_array_baseline_run_db": distribution(base_null),
            "rows_without_beam": int((null["status"] != "ok").sum()),
        },
        "replay_supported": rate_entry(boot, rows, "supported"),
        "baseline_replay_supported": rate_entry(boot, replay_rows(baseline), "supported"),
        "replay_minus_baseline": diff,
        "secondary_matched_records_minus_baseline": diff_entry(
            boot, (replay_rows(frame, MATCHED_RECORDS), "supported"), (replay_rows(baseline, MATCHED_RECORDS), "supported")),
        "n_geometries_ge1": int((s["stratum"] == "ge1").sum()),
        "replay_supported_ge1": ge1_rate,
        "a7_1_replay_matched": bool(abs(diff["point"]) <= 0.05 and (ge1_rate is None or ge1_rate["point"] <= 0.05)),
        "target_supported_primary": target,
        "a7_1_target_matched": bool(target["point"] >= 0.95),
        "stratified": stratified_rates(frame, draws),
    }
    per_geometry = frame[(frame["method"] == PRIMARY)].groupby(["geometry_id", "world"])["supported"].mean().unstack(
        "world").add_prefix("supported_").join(s[["delta_tau_w_ns", "ratio_median", "stratum"]]).join(
        ok.groupby("geometry_id")["realized_null_depth_db"].median().rename("realized_null_depth_db_median"))
    (out_dir / "array_error_results.json").write_text(json.dumps(result, indent=2, default=float))
    per_geometry.reset_index().to_csv(out_dir / "per_geometry.csv", index=False)
    pd.DataFrame(result["stratified"]).to_csv(out_dir / "rates_by_stratum.csv", index=False)
    print(json.dumps({k: v for k, v in result.items() if k != "stratified"}, indent=1, default=float))
    print(per_geometry.round(3).to_string())
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
