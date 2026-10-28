"""Phase 8 held-out analysis: rule 8 (primary), Amendments 2-3 (secondary), BRIEF §16 decision.

Evaluator side. Decisions come only from configs `phase8.frozen` (via
`phase8a.frozen_decision`); nothing is fitted. Geometry-level bootstrap per
BRIEF §13.3.

Usage: python scripts/phase8b_heldout_analysis.py TRIALS.parquet GEOMETRIES.csv OUT_DIR [RUN_DIR]
(RUN_DIR holds heldout_summary.json and environment.json for the run-level validity checks.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from isac_verify.calibration import MAPPED_METHODS, WALL_PROBING_METHODS, classify_trials
from isac_verify.config import PilotConfig
from isac_verify.phase8a import frozen_decision

C_LIGHT = 299_792_458.0
BOOTSTRAP_SEED = 20261027  # declared here, before any held-out result was seen
PRIMARY, SECONDARY, COMPARATOR = "imperfect_robust", "imperfect_narrow", "mapfree_coarse_to_fine"


def classify(trials: pd.DataFrame, config: PilotConfig) -> pd.DataFrame:
    factor, fitted = frozen_decision(config)
    frame = classify_trials(trials, fitted, factor)
    # A3.2: same conditions with the upper bound s_hi removed.
    no_hi = json.loads(json.dumps(fitted))
    for method in MAPPED_METHODS:
        no_hi["mapped"][method]["s_hi"]["value"] = float("inf")
    frame["supported_no_s_hi"] = classify_trials(trials, no_hi, factor)["supported"].to_numpy()
    return frame


def strata(frame: pd.DataFrame, geometries: pd.DataFrame) -> pd.DataFrame:
    """A3.1: delta_tau_W (BRIEF §5.3 image method) over the measured PSF, per geometry."""
    g = geometries.set_index("geometry_id")
    r_t, r_a, d = g["candidate_distance_m"], g["relay_distance_m"], g["half_width_m"]
    tau_wall = (np.sqrt(r_t**2 + 4 * d**2) + r_t) / C_LIGHT
    tau_relay = (np.sqrt(r_a**2 + 4 * d**2) + r_a) / C_LIGHT + 2 * (r_t - r_a) / C_LIGHT
    ratio = frame.assign(r=frame["geometry_id"].map((tau_relay - tau_wall) * 1e9) / frame["point_spread_delay_ns"])
    by_geometry = ratio.groupby("geometry_id")["r"].agg(["median", "min", "max"])
    out = pd.DataFrame({"delta_tau_w_ns": (tau_relay - tau_wall) * 1e9, "two_d_over_r_t": 2 * d / r_t,
                        "r_a_over_r_t": r_a / r_t}).join(by_geometry.add_prefix("ratio_"))
    out["stratum"] = np.where(out["ratio_median"] < 1.0, "lt1", "ge1")
    out["stratum_split_within_geometry"] = (out["ratio_min"] < 1.0) & (out["ratio_max"] >= 1.0)
    return out


class Bootstrap:
    """Geometry-level resampling; every statistic uses the same draws (paired differences)."""

    def __init__(self, geometry_ids: list[str], draws: int):
        self.ids = list(geometry_ids)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        picks = rng.integers(0, len(self.ids), size=(draws, len(self.ids)))
        self.weights = np.stack([np.bincount(row, minlength=len(self.ids)) for row in picks])

    def rate(self, rows: pd.DataFrame, column: str) -> tuple[float, np.ndarray]:
        grouped = rows.groupby("geometry_id")[column].agg(["sum", "count"]).reindex(self.ids, fill_value=0)
        s, n = grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)
        if n.sum() == 0:
            return float("nan"), np.full(len(self.weights), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(s.sum() / n.sum()), (self.weights @ s) / (self.weights @ n)


def ci(samples: np.ndarray) -> list[float]:
    return [float(np.nanquantile(samples, 0.025)), float(np.nanquantile(samples, 0.975))]


def cell(frame: pd.DataFrame, method: str, world: str, sigmas: list[float] | None) -> pd.DataFrame:
    rows = frame[(frame["method"] == method) & (frame["world"] == world)]
    return rows if sigmas is None else rows[rows["sigma_map_m"].isin(sigmas)]


def rate_entry(boot: Bootstrap, rows: pd.DataFrame, column: str) -> dict:
    point, samples = boot.rate(rows, column)
    return {"point": point, "ci95": ci(samples), "n_rows": int(len(rows)),
            "n_geometries": int(rows["geometry_id"].nunique())}


def diff_entry(boot: Bootstrap, a: tuple[pd.DataFrame, str], b: tuple[pd.DataFrame, str]) -> dict:
    pa, sa = boot.rate(*a)
    pb, sb = boot.rate(*b)
    interval = ci(sa - sb)
    return {"point": pa - pb, "ci95": interval, "ci_lower_gt_0": bool(interval[0] > 0.0)}


def analyze(frame: pd.DataFrame, geometry_ids: list[str], config: PilotConfig) -> dict:
    h = config.section("phase8", "heldout")
    boot = Bootstrap(geometry_ids, int(h["bootstrap_draws"]))
    sigmas = sorted(float(s) for s in frame["sigma_map_m"].unique())
    rmax, umax = float(h["replay_supported_rate_max"]), float(h["replay_supported_rate_upper_ci_max"])
    genuine = [s for s in sigmas if s <= float(h["genuine_target_sigma_max_m"])]
    out: dict = {"primary": {}, "secondary": {}}

    # Rule 8 / A2.3: replay criterion, pooled and at each advantage sigma; all cells reported.
    for method in (PRIMARY, SECONDARY):
        cells = {"pooled": None, **{str(s): [s] for s in sigmas}}
        entries = {}
        for label, subset in cells.items():
            e = rate_entry(boot, cell(frame, method, "replay", subset), "supported")
            e["meets"] = bool(e["point"] <= rmax and e["ci95"][1] <= umax)
            entries[label] = e
        gating = ["pooled"] + [str(float(s)) for s in h["advantage_sigma_m"]]
        out["primary" if method == PRIMARY else "secondary"][f"replay_supported_{method}"] = {
            "cells": entries, "gating_cells": gating, "meets_all_gating": all(entries[c]["meets"] for c in gating)}

    # Rule 8: genuine-target SUPPORTED at sigma <= 0.5 (useful = point >= 0.5), pooled and per sigma.
    for method in (PRIMARY, SECONDARY):
        e = rate_entry(boot, cell(frame, method, "target", genuine), "supported")
        e["useful"] = bool(e["point"] >= float(h["useful_target_supported_rate_min"]))
        e["per_sigma"] = {str(s): rate_entry(boot, cell(frame, method, "target", [s]), "supported") for s in genuine}
        out["primary" if method == PRIMARY else "secondary"][f"target_supported_sigma_le_0p5_{method}"] = e

    # Rule 8: advantage (primary SUPPORTED - comparator SUPPORTED_common) and A2.1 like-for-like.
    adv, like = {}, {}
    for s in sigmas:
        adv[str(s)] = diff_entry(boot, (cell(frame, PRIMARY, "target", [s]), "supported"),
                                 (cell(frame, COMPARATOR, "target", [s]), "supported_common"))
        like[str(s)] = diff_entry(boot, (cell(frame, PRIMARY, "target", [s]), "supported_common"),
                                  (cell(frame, COMPARATOR, "target", [s]), "supported_common"))
    adv_sigmas = [str(float(s)) for s in h["advantage_sigma_m"]]
    out["primary"]["advantage"] = {"by_sigma": adv, "decision_sigmas": adv_sigmas,
                                   "holds": any(adv[s]["ci_lower_gt_0"] for s in adv_sigmas)}
    out["secondary"]["a2_1_like_for_like"] = {"by_sigma": like,
                                              "holds_at_decision_sigmas": any(like[s]["ci_lower_gt_0"] for s in adv_sigmas)}

    # BRIEF §16 provisional engineering targets (baseline has no blocker: every geometry is in the unblocked subset).
    t = h["brief16_targets"]
    target = out["primary"][f"target_supported_sigma_le_0p5_{PRIMARY}"]
    attack = out["primary"][f"replay_supported_{PRIMARY}"]["cells"]["pooled"]
    primary_rows = frame[frame["method"] == PRIMARY].assign(
        not_observable=lambda f: f["path_availability"].eq("NOT_OBSERVABLE"))
    nob = rate_entry(boot, primary_rows, "not_observable")
    out["primary"]["brief16"] = {
        "target_support": {**target, "target_min": t["target_support_min"],
                           "point_meets": bool(target["point"] >= t["target_support_min"])},
        "attack_support": {**attack, "target_max": t["attack_support_max"],
                           "point_meets": bool(attack["point"] <= t["attack_support_max"])},
        "not_observable": {**nob, "target_max": t["not_observable_max"],
                           "point_meets": bool(nob["point"] <= t["not_observable_max"])},
    }
    return out


def criteria_table(result: dict, contrast: str = "advantage") -> list[dict]:
    """A4.2: every §16 / rule-8 criterion with point, interval and the A4.1 tests, checkable by hand.

    ``contrast`` is "advantage" (primary, rule 8) or "a2_1_like_for_like" (secondary, A2.1).
    """
    p, b16 = result["primary"], result["primary"]["brief16"]
    replay = p[f"replay_supported_{PRIMARY}"]
    useful = p[f"target_supported_sigma_le_0p5_{PRIMARY}"]
    rows = []

    def add(name, entry, threshold, side, interval_holds=None):
        lo, hi = entry["ci95"]
        point = entry["point"]
        ok = point >= threshold if side == ">=" else point <= threshold
        # whole interval on the failing side
        fails = hi < threshold if side == ">=" else lo > threshold
        rows.append({"criterion": name, "role": "criterion", "point": point, "ci95_lo": lo, "ci95_hi": hi, "threshold": threshold,
                     "pass_if": f"{side} {threshold:g}", "point_meets": bool(ok),
                     "interval_criterion_holds": interval_holds, "whole_interval_failing": bool(fails)})

    add("target SUPPORT, sigma<=0.5 (§16)", b16["target_support"], float(b16["target_support"]["target_min"]), ">=")
    add("useful target SUPPORT, sigma<=0.5 (rule 8)", useful, 0.5, ">=")
    add("attack SUPPORT, pooled (§16)", b16["attack_support"], float(b16["attack_support"]["target_max"]), "<=")
    add("NOT_OBSERVABLE (§16)", b16["not_observable"], float(b16["not_observable"]["target_max"]), "<=")
    for cell_name in replay["gating_cells"]:
        e = replay["cells"][cell_name]
        add(f"replay SUPPORTED, {cell_name} (rule 8 / A2.3)", e, 0.05, "<=", bool(e["ci95"][1] <= 0.10))
    by_sigma = (p["advantage"] if contrast == "advantage" else result["secondary"]["a2_1_like_for_like"])["by_sigma"]
    sigmas = p["advantage"]["decision_sigmas"]
    entries = [by_sigma[s] for s in sigmas]
    for s, e in zip(sigmas, entries):
        rows.append({"criterion": f"  component: {contrast}, sigma={s}", "role": "component", "point": e["point"],
                     "ci95_lo": e["ci95"][0], "ci95_hi": e["ci95"][1], "threshold": 0.0, "pass_if": "> 0",
                     "point_meets": bool(e["point"] > 0), "interval_criterion_holds": bool(e["ci95"][0] > 0),
                     "whole_interval_failing": bool(e["ci95"][1] <= 0)})
    rows.append({
        "criterion": f"{contrast} at sigma 0.5 or 1.0 (rule 8{'' if contrast == 'advantage' else ', A2.1'})",
        "role": "criterion", "point": None, "ci95_lo": None, "ci95_hi": None, "threshold": 0.0,
        "pass_if": "> 0 at either sigma (components above)",
        "point_meets": any(e["point"] > 0 for e in entries),
        "interval_criterion_holds": any(e["ci95"][0] > 0 for e in entries),
        "whole_interval_failing": all(e["ci95"][1] <= 0 for e in entries),
    })
    return rows


def decide(result: dict, validity: dict, contrast: str = "advantage") -> dict:
    """A4.1 label: INVALID > NO-GO (some whole interval failing) > CONDITIONAL GO (all points and
    interval criteria) > INCONCLUSIVE."""
    table = [r for r in criteria_table(result, contrast) if r.get("role", "criterion") == "criterion"]
    if not validity["all_pass"]:
        label = "INVALID"
    elif any(r["whole_interval_failing"] for r in table):
        label = "NO-GO"
    elif all(r["point_meets"] for r in table) and all(
        r["interval_criterion_holds"] for r in table if r["interval_criterion_holds"] is not None
    ):
        label = "CONDITIONAL GO"
    else:
        label = "INCONCLUSIVE"
    return {"label": label, "contrast": contrast, "criteria": criteria_table(result, contrast)}


def validity_checks(frame: pd.DataFrame, run_dir: Path | None, geometry_count: int) -> dict:
    """A4.1 INVALID conditions that this script can check; run-level files only when a run dir is given."""
    checks = {
        "decision_columns_complete": bool(frame[["supported", "supported_common"]].notna().all().all()),
        "every_geometry_has_full_grid": bool(frame.groupby("geometry_id").size().nunique() == 1),
        "geometry_count": int(frame["geometry_id"].nunique()) == geometry_count,
    }
    if run_dir is not None:
        summary = json.loads((run_dir / "heldout_summary.json").read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        checks.update({
            "rows_match_expected": bool(summary["rows_match"]),
            "split_is_heldout": bool(set(frame["split"]) == {"heldout"}),
            "decision_source_frozen": summary["decision_source"] == "configs phase8.frozen",
            "preflight_pass": environment["status"] == "PASS",
            "gpu_is_h100": "H100" in environment["gpu"]["device"]["name"],
            "source_commit_clean": environment.get("source_commit", {}).get("dirty_paths") == 0,
        })
    return {"checks": checks, "all_pass": all(checks.values()), "run_level_checked": run_dir is not None}


def rate_tables(frame: pd.DataFrame, draws: int, by: list[str]) -> pd.DataFrame:
    cache: dict[tuple, Bootstrap] = {}
    rows = []
    for keys, group in frame.groupby(["method", "sigma_map_m", "world"] + by):
        keys = dict(zip(["method", "sigma_map_m", "world"] + by, keys))
        ids = tuple(sorted(group["geometry_id"].unique()))
        if ids not in cache:
            cache[ids] = Bootstrap(list(ids), draws)
        sub = cache[ids]
        row = {**keys, "n": int(len(group))}
        for column in ("supported", "supported_no_s_hi", "supported_common"):
            applies = keys["method"] in (WALL_PROBING_METHODS if column == "supported_common" else MAPPED_METHODS)
            if not applies:  # no such rule for this method (as in calibration.rate_table)
                row[column] = row[f"{column}_lo"] = row[f"{column}_hi"] = float("nan")
                continue
            point, samples = sub.rate(group, column)
            row[column], (row[f"{column}_lo"], row[f"{column}_hi"]) = point, ci(samples)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    trials_path, geometries_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    run_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    out_dir.mkdir(parents=True, exist_ok=True)
    config = PilotConfig.load("configs/pilot.yaml")
    trials = pd.read_parquet(trials_path)
    geometries = pd.read_csv(geometries_path)
    frame = classify(trials, config)
    if "supported" in trials.columns and set(trials["split"]) == {"heldout"}:
        # The run's analyze_heldout applied the same frozen block; the two must agree exactly.
        assert (trials["supported"].to_numpy() == frame["supported"].to_numpy()).all()
        assert (trials["supported_common"].to_numpy() == frame["supported_common"].to_numpy()).all()
    s = strata(frame, geometries)
    frame["stratum"] = frame["geometry_id"].map(s["stratum"])
    ids = sorted(frame["geometry_id"].unique())
    draws = int(config.get("phase8", "heldout", "bootstrap_draws"))
    result = {"population": analyze(frame, ids, config), "strata": {}}
    for name in ("lt1", "ge1"):
        part = frame[frame["stratum"] == name]
        if part["geometry_id"].nunique():
            result["strata"][name] = analyze(part, sorted(part["geometry_id"].unique()), config)
    result["validity"] = validity_checks(frame, run_dir, len(geometries))
    result["decision"] = decide(result["population"], result["validity"])
    result["decision_like_for_like_secondary"] = decide(result["population"], result["validity"], "a2_1_like_for_like")
    pd.DataFrame(result["decision"]["criteria"]).to_csv(out_dir / "criteria_primary.csv", index=False)
    pd.DataFrame(result["decision_like_for_like_secondary"]["criteria"]).to_csv(
        out_dir / "criteria_like_for_like.csv", index=False)
    result["geometry_strata"] = s.reset_index().to_dict(orient="records")
    result["bootstrap"] = {"seed": BOOTSTRAP_SEED, "draws": draws, "unit": "geometry", "interval": "percentile 95%"}
    (out_dir / "heldout_results.json").write_text(json.dumps(result, indent=2, default=float))
    rate_tables(frame, draws, []).to_csv(out_dir / "rates_ci.csv", index=False)
    rate_tables(frame, draws, ["stratum"]).to_csv(out_dir / "rates_ci_by_stratum.csv", index=False)
    per_geometry = frame.groupby(["geometry_id", "method", "sigma_map_m", "world"])[
        ["supported", "supported_no_s_hi", "supported_common"]].mean().reset_index()
    per_geometry.to_csv(out_dir / "rates_per_geometry.csv", index=False)
    s.to_csv(out_dir / "geometry_strata.csv")
    print("validity", json.dumps(result["validity"]))
    for key in ("decision", "decision_like_for_like_secondary"):
        print(key, result[key]["label"])
        print(pd.DataFrame(result[key]["criteria"]).to_string())
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
