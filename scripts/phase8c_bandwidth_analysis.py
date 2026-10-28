"""Phase 8C block 1 (bandwidth) analysis: ANALYSIS_PLAN A5.1 predictions and A3.1 strata per level.

Evaluator side. Decisions come only from configs `phase8.frozen`; the bootstrap
is the A4.1 geometry-level bootstrap (same seed and draws). Baseline is the main
held-out run's sigma 0.5 cell; the matched comparator keeps records {0,1,4,5}.

Usage: python scripts/phase8c_bandwidth_analysis.py BASELINE_TRIALS.parquet GEOMETRIES.csv OUT_DIR LEVEL=RUN_DIR [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from isac_verify.config import PilotConfig
from phase8b_heldout_analysis import PRIMARY, Bootstrap, classify, diff_entry, rate_entry, strata

SIGMA = 0.5
MATCHED_RECORDS = [0, 1, 4, 5]  # same seeds and twin draw (r mod 2 == r mod 4) in stress and main run
# A5.1 predicted direction of each level relative to baseline (-1 decrease, +1 increase).
PREDICTED = {"bw246": -1, "bw61": +1}


def replay_rows(frame: pd.DataFrame, records: list[int] | None = None) -> pd.DataFrame:
    rows = frame[(frame["method"] == PRIMARY) & (frame["world"] == "replay") & (frame["sigma_map_m"] == SIGMA)]
    return rows if records is None else rows[rows["record_index"].isin(records)]


def validity(frame: pd.DataFrame, trials: pd.DataFrame, run_dir: Path, level: str, config: PilotConfig) -> dict:
    grid = config.section("phase8", "stress", "grid")
    summary = json.loads((run_dir / "heldout_summary.json").read_text())
    environment = json.loads((run_dir / "environment.json").read_text())
    expected = int(config.get("phase8", "geometry", "heldout_geometries")) * len(grid["sigma_map_m"]) * 2 * int(grid["records_per_world"]) * len(grid["methods"])
    return {
        "rows_expected": len(frame) == expected and bool(summary["rows_match"]),
        "split_is_heldout": set(frame["split"]) == {"heldout"},
        "level_label": set(frame["bandwidth_profile"]) == {level},
        "methods": set(frame["method"]) == set(grid["methods"]),
        "decision_source_frozen": summary["decision_source"] == "configs phase8.frozen",
        "reclassification_matches_run": bool(
            (trials["supported"].to_numpy() == frame["supported"].to_numpy()).all()
            and (trials["supported_common"].to_numpy() == frame["supported_common"].to_numpy()).all()
        ),
        "preflight_pass": "status: PASS" in (run_dir / "preflight.log").read_text(),
        "source_commit_clean": environment.get("source_commit", {}).get("dirty_paths") == 0,
        "source_commit": environment.get("source_commit", {}).get("commit"),
        "acquisition_failed_rows": int((frame["status"] == "ACQUISITION_FAILED").sum()),
    }


def stratified_rates(frame: pd.DataFrame, draws: int) -> list[dict]:
    out = []
    for (stratum, method, world), rows in frame.groupby(["stratum", "method", "world"]):
        boot = Bootstrap(sorted(rows["geometry_id"].unique()), draws)
        for column in ("supported", "supported_common"):
            if rows[column].notna().any():
                out.append({"stratum": stratum, "method": method, "world": world, "rate": column,
                            **rate_entry(boot, rows, column)})
    return out


def main() -> None:
    baseline_path, geometries_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    levels = dict(arg.split("=", 1) for arg in sys.argv[4:])
    out_dir.mkdir(parents=True, exist_ok=True)
    config = PilotConfig.load("configs/pilot.yaml")
    draws = int(config.get("phase8", "heldout", "bootstrap_draws"))
    geometries = pd.read_csv(geometries_path)
    ids = sorted(geometries["geometry_id"])
    boot = Bootstrap(ids, draws)

    baseline = pd.read_parquet(baseline_path)
    base_strata = strata(baseline[baseline["sigma_map_m"] == SIGMA], geometries)
    result = {"baseline": {
        "psf_delay_ns_median": float(baseline["point_spread_delay_ns"].median()),
        "n_geometries_ratio_lt1": int((base_strata["stratum"] == "lt1").sum()),
        "replay_supported": rate_entry(boot, replay_rows(baseline), "supported"),
        "replay_supported_matched_records": rate_entry(boot, replay_rows(baseline, MATCHED_RECORDS), "supported"),
    }, "levels": {}}
    strata_rows = [base_strata.assign(level="baseline")]
    for level, run in levels.items():
        run_dir = Path(run)
        trials = pd.read_parquet(run_dir / "heldout_trials.parquet")
        frame = classify(trials, config)
        s = strata(frame, geometries)
        frame["stratum"] = frame["geometry_id"].map(s["stratum"])
        strata_rows.append(s.assign(level=level))
        n_lt1 = int((s["stratum"] == "lt1").sum())
        rows = replay_rows(frame)
        vs_base = diff_entry(boot, (rows, "supported"), (replay_rows(baseline), "supported"))
        vs_matched = diff_entry(boot, (replay_rows(frame, MATCHED_RECORDS), "supported"),
                                (replay_rows(baseline, MATCHED_RECORDS), "supported"))
        sign = PREDICTED[level]
        result["levels"][level] = {
            "validity": validity(frame, trials, run_dir, level, config),
            "psf_delay_ns_median": float(frame["point_spread_delay_ns"].median()),
            "a5_1a_n_geometries_ratio_lt1": n_lt1,
            "a5_1a_matched": bool(sign * (n_lt1 - result["baseline"]["n_geometries_ratio_lt1"]) > 0),
            "replay_supported": rate_entry(boot, rows, "supported"),
            "a5_1b_minus_baseline": vs_base,
            "a5_1b_matched": bool(sign * vs_base["point"] > 0),
            "secondary_matched_records_minus_baseline": vs_matched,
            "target_supported_primary": rate_entry(
                boot, frame[(frame["method"] == PRIMARY) & (frame["world"] == "target")], "supported"),
            "stratified": stratified_rates(frame, draws),
        }
    (out_dir / "bandwidth_results.json").write_text(json.dumps(result, indent=2, default=float))
    pd.concat(strata_rows).reset_index().to_csv(out_dir / "geometry_strata_by_level.csv", index=False)
    pd.DataFrame([{"level": lv, **r} for lv, v in result["levels"].items() for r in v["stratified"]]).to_csv(
        out_dir / "rates_by_level_stratum.csv", index=False)
    print(json.dumps({k: v for k, v in result["baseline"].items()}, default=float))
    for level, v in result["levels"].items():
        print(level, json.dumps({k: v[k] for k in v if k != "stratified"}, default=float))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
