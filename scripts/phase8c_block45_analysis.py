"""Phase 8C blocks 4 (probe power) and 5 (wall thickness D8, target RCS): ANALYSIS_PLAN A8.1 / A8.2.

Evaluator side. Decisions come only from configs `phase8.frozen`; intervals are
the A4.1 geometry-level bootstrap. Baseline is the main held-out run's sigma 0.5
cell (10 dBm, 0.10 m, constant RCS); matched records {0,1,4,5} are secondary.
Strata use each geometry's median delta_tau_W / PSF at that level (A5 notes).

Usage: python scripts/phase8c_block45_analysis.py BASELINE_TRIALS.parquet GEOMETRIES.csv OUT_DIR LEVEL=RUN_DIR [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from isac_verify.config import PilotConfig
from phase8b_heldout_analysis import COMPARATOR, PRIMARY, Bootstrap, classify, diff_entry, rate_entry, strata
from phase8c_bandwidth_analysis import MATCHED_RECORDS, SIGMA, stratified_rates, validity

POWER_LEVELS = {"probe_power_m20dbm": -20.0, "probe_power_m10dbm": -10.0, "probe_power_0dbm": 0.0,
                "probe_power_p20dbm": 20.0}
BASELINE_POWER_DBM = 10.0
OUTSIDE = "candidate_outside_twin_free_space"  # A9.1 reason
KEY = ["geometry_id", "world", "record_index", "method", "sigma_map_m"]


def cell(frame: pd.DataFrame, method: str, world: str) -> pd.DataFrame:
    return frame[(frame["method"] == method) & (frame["world"] == world) & (frame["sigma_map_m"] == SIGMA)]


def sub_rate(rows: pd.DataFrame, column: str, draws: int) -> dict | None:
    if rows.empty:
        return None
    return rate_entry(Bootstrap(sorted(rows["geometry_id"].unique()), draws), rows, column)


def crossing_power(rates: dict[float, float]) -> float | None:
    """A8 notes: highest tested power whose target rate is < 0.5 (None if never)."""
    below = [p for p, r in rates.items() if r < 0.5]
    return max(below) if below else None


def a8_1b(p_robust: float | None, p_c2f: float | None) -> str:
    if p_c2f is None:
        return "not matched"
    if p_robust is None:
        return "matched" if p_c2f >= -10.0 else "indeterminate (not matched)"
    return "matched" if p_c2f >= p_robust + 10.0 else "not matched"


def _check_a8_1b() -> None:
    assert crossing_power({-20.0: 0.1, -10.0: 0.4, 0.0: 0.9, 10.0: 1.0}) == -10.0
    assert crossing_power({-20.0: 0.9}) is None
    assert a8_1b(-20.0, -10.0) == "matched" and a8_1b(-10.0, -10.0) == "not matched"
    assert a8_1b(None, -10.0) == "matched" and a8_1b(None, -20.0).startswith("indeterminate")
    assert a8_1b(-20.0, None) == "not matched"


def acquisition_counts(frame: pd.DataFrame) -> dict:
    """A9.3 (evaluator side), per world, in (geometry, world, record) units: detected records, false
    acquisitions (measured candidate > 2 m from the true claimed position), outside-free-space records."""
    unit = ["geometry_id", "world", "record_index"]
    rec = frame.groupby(unit).agg(
        r=("candidate_range_m", "first"), phi=("candidate_aoa_deg", "first"), r_true=("candidate_distance_m", "first"),
        phi_true=("true_candidate_azimuth_deg", "first"),
        failed=("status", lambda v: bool((v == "ACQUISITION_FAILED").any())),
        outside=("status_reason", lambda v: bool((v == OUTSIDE).any())))
    dphi = np.radians(rec["phi"] - rec["phi_true"])
    error = np.sqrt(rec["r"] ** 2 + rec["r_true"] ** 2 - 2 * rec["r"] * rec["r_true"] * np.cos(dphi))
    detected = ~rec["failed"] & np.isfinite(rec["r"])
    out = {}
    for world, idx in rec.groupby(level="world").groups.items():
        d = detected.loc[idx]
        out[world] = {"records": int(len(idx)), "acquisition_failed": int(rec.loc[idx, "failed"].sum()),
                      "detected": int(d.sum()), "false_acquisitions_gt_2m": int((error.loc[idx][d] > 2.0).sum()),
                      "outside_twin_free_space": int(rec.loc[idx, "outside"].sum()),
                      "median_position_error_m": float(error.loc[idx][d].median()) if d.any() else float("nan")}
    return out


def replay_identical(frame: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    """RCS validity: replay rows on matched records equal the main run's in every common non-timing column."""
    a = frame[(frame["world"] == "replay") & frame["record_index"].isin(MATCHED_RECORDS)]
    b = baseline[(baseline["world"] == "replay") & baseline["record_index"].isin(MATCHED_RECORDS)
                 & baseline["method"].isin(a["method"].unique()) & (baseline["sigma_map_m"] == SIGMA)]
    a, b = (f.sort_values(KEY).reset_index(drop=True) for f in (a, b))
    cols = [c for c in a.columns if c in b.columns and c != "trial_runtime_s"]
    differing = [c for c in cols if not a[c].astype(str).equals(b[c].astype(str))] if len(a) == len(b) else ["row_count"]
    return {"rows": int(len(a)), "baseline_rows": int(len(b)), "compared_columns": len(cols), "differing": differing}


def main() -> None:
    baseline_path, geometries_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    levels = dict(arg.split("=", 1) for arg in sys.argv[4:])
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_a8_1b()
    config = PilotConfig.load("configs/pilot.yaml")
    draws = int(config.get("phase8", "heldout", "bootstrap_draws"))
    s_hi = float(config.get("phase8", "frozen", "thresholds", PRIMARY, "supported_max"))
    geometries = pd.read_csv(geometries_path)
    boot = Bootstrap(sorted(geometries["geometry_id"]), draws)
    baseline = pd.read_parquet(baseline_path)
    base_strata = strata(baseline[baseline["sigma_map_m"] == SIGMA], geometries)
    baseline["stratum"] = baseline["geometry_id"].map(base_strata["stratum"])
    result: dict = {"baseline": {
        "robust_target": rate_entry(boot, cell(baseline, PRIMARY, "target"), "supported"),
        "robust_replay": rate_entry(boot, cell(baseline, PRIMARY, "replay"), "supported"),
        "c2f_target_common": rate_entry(boot, cell(baseline, COMPARATOR, "target"), "supported_common"),
    }, "levels": {}}
    strata_rows = [base_strata.assign(level="baseline")]
    for level, run in levels.items():
        run_dir = Path(run)
        trials = pd.read_parquet(run_dir / "heldout_trials.parquet")
        frame = classify(trials, config)
        s = strata(frame, geometries)
        frame["stratum"] = frame["geometry_id"].map(s["stratum"])
        strata_rows.append(s.assign(level=level))
        check = validity(frame, trials, run_dir, "baseline", config)  # level_label: profile is baseline
        run_geometries = pd.read_csv(run_dir / "geometries_heldout.csv").set_index("geometry_id").sort_index()
        main_geometries = geometries.set_index("geometry_id").sort_index()
        cols = ["candidate_distance_m", "relay_distance_m", "half_width_m"]
        check["geometries_equal_main"] = bool(np.array_equal(run_geometries[cols].to_numpy(),
                                                             main_geometries[cols].to_numpy()))
        check["wall_thickness_m"] = sorted(run_geometries["wall_thickness_m"].unique().tolist())
        check["power_dbm"] = sorted(frame["power_dbm"].unique().tolist())
        rt, rr = cell(frame, PRIMARY, "target"), cell(frame, PRIMARY, "replay")
        matched = rt[rt["record_index"].isin(MATCHED_RECORDS)].merge(
            cell(baseline, PRIMARY, "target")[KEY + ["S_c1"]], on=KEY, suffixes=("", "_baseline"))
        ratio = (matched["S_c1"] / matched["S_c1_baseline"]).replace([np.inf, -np.inf], np.nan)
        lt1 = rr[rr["stratum"] == "lt1"]
        entry = {
            "validity": check,
            "robust_target": rate_entry(boot, rt, "supported"),
            "robust_target_no_s_hi": rate_entry(boot, rt, "supported_no_s_hi"),
            "robust_replay": rate_entry(boot, rr, "supported"),
            "robust_replay_minus_baseline": diff_entry(boot, (rr, "supported"),
                                                       (cell(baseline, PRIMARY, "replay"), "supported")),
            "robust_target_minus_baseline": diff_entry(boot, (rt, "supported"),
                                                       (cell(baseline, PRIMARY, "target"), "supported")),
            "n_geometries_ge1": int((s["stratum"] == "ge1").sum()),
            "robust_replay_ge1": sub_rate(rr[rr["stratum"] == "ge1"], "supported", draws),
            "robust_replay_lt1": sub_rate(lt1, "supported", draws),
            "baseline_robust_replay_same_lt1_geometries": sub_rate(
                cell(baseline, PRIMARY, "replay")[cell(baseline, PRIMARY, "replay")["geometry_id"].isin(
                    lt1["geometry_id"].unique())], "supported", draws),
            "narrow_target": rate_entry(boot, cell(frame, "imperfect_narrow", "target"), "supported"),
            "narrow_replay": rate_entry(boot, cell(frame, "imperfect_narrow", "replay"), "supported"),
            "c2f_target_common": rate_entry(boot, cell(frame, COMPARATOR, "target"), "supported_common"),
            "c2f_replay_common": rate_entry(boot, cell(frame, COMPARATOR, "replay"), "supported_common"),
            "acquisition_failed_rows_by_world": {w: int(((frame["world"] == w)
                                                         & (frame["status"] == "ACQUISITION_FAILED")).sum())
                                                 for w in ("target", "replay")},
            "direct_gate_snr_median": float(((rt["E_D"] - rt["N_D"]) / rt["N_D"]).median()),
            "robust_target_S_ratio_to_baseline_matched_median": float(ratio.median()),
            "robust_target_S_ratio_n": int(ratio.notna().sum()),
            "robust_target_fraction_S_above_s_hi": float((rt["S_c1"] > s_hi).mean()),
            "a9_3_acquisition_counts": acquisition_counts(frame),
            "stratified": stratified_rates(frame, draws),
        }
        if level.startswith("rcs_"):
            entry["validity"]["replay_rows_identical_to_main_matched"] = replay_identical(frame, baseline)
        result["levels"][level] = entry

    lv = result["levels"]
    predictions: dict = {}
    if all(level in lv for level in POWER_LEVELS):
        power = {POWER_LEVELS[k]: v for k, v in lv.items() if k in POWER_LEVELS}
        robust = {p: v["robust_target"]["point"] for p, v in power.items()}
        robust[BASELINE_POWER_DBM] = result["baseline"]["robust_target"]["point"]
        c2f = {p: v["c2f_target_common"]["point"] for p, v in power.items()}
        c2f[BASELINE_POWER_DBM] = result["baseline"]["c2f_target_common"]["point"]
        p_r, p_c = crossing_power(robust), crossing_power(c2f)
        predictions["A8.1(a)"] = bool(robust[-10.0] >= 0.95 and robust[-20.0] < robust[-10.0]
                                      and sum(power[-20.0]["acquisition_failed_rows_by_world"].values()) > 0)
        predictions["A8.1(b)"] = {"P_robust": p_r, "P_c2f": p_c, "verdict": a8_1b(p_r, p_c)}
        predictions["A8.1(c)"] = bool(all(v["robust_replay_ge1"] is None or v["robust_replay_ge1"]["point"] <= 0.05
                                          for v in power.values()))
    thick = [k for k in ("wall_thickness_50mm", "wall_thickness_200mm") if k in lv]
    if len(thick) == 2:
        t50 = lv["wall_thickness_50mm"]
        predictions["A8.2(a) ge1"] = bool(all(lv[k]["robust_replay_ge1"] is None
                                              or lv[k]["robust_replay_ge1"]["point"] <= 0.05 for k in thick))
        predictions["A8.2(a) lt1 at 50 mm > baseline"] = bool(
            t50["robust_replay_lt1"] is not None
            and t50["robust_replay_lt1"]["point"] > t50["baseline_robust_replay_same_lt1_geometries"]["point"])
    if "rcs_p6db" in lv and "rcs_m6db" in lv:
        predictions["A8.2(b) +6 dB"] = bool(lv["rcs_p6db"]["robust_target"]["point"] < 0.2
                                            and lv["rcs_p6db"]["robust_target_no_s_hi"]["point"] >= 0.95)
        predictions["A8.2(b) -6 dB"] = bool(lv["rcs_m6db"]["robust_target"]["point"] >= 0.95)
    result["predictions"] = predictions

    (out_dir / "block45_results.json").write_text(json.dumps(result, indent=2, default=float))
    pd.concat(strata_rows).reset_index().to_csv(out_dir / "geometry_strata_by_level.csv", index=False)
    pd.DataFrame([{"level": k, **r} for k, v in lv.items() for r in v["stratified"]]).to_csv(
        out_dir / "rates_by_level_stratum.csv", index=False)
    print(json.dumps(result["baseline"], default=float))
    for level, v in lv.items():
        print(level, json.dumps({k: v[k] for k in v if k != "stratified"}, default=float))
    print("predictions", json.dumps(predictions, default=str))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
