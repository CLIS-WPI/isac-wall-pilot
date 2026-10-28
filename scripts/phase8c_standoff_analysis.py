"""Phase 8C block 2 (relay standoff sweep) analysis: ANALYSIS_PLAN A5.2 and the A6.2 two-ratio prediction.

Evaluator side. Decisions come only from configs `phase8.frozen`; intervals are
the A4.1 geometry-level bootstrap, resampling geometries with all their rows in
a cell. Per (geometry, level): delay ratio delta_tau_W / PSF (that level's
measured PSF) and angular ratio Delta_phi_W / HPBW_TX (HPBW_TX from
scripts/phase8c_angular_ratio.py); a ratio >= 1 is "resolved" (A6 operational notes).

Usage: python scripts/phase8c_standoff_analysis.py ANGULAR.csv OUT_DIR LEVEL=RUN_DIR [...]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from isac_verify.config import PilotConfig
from phase8b_heldout_analysis import Bootstrap, classify, rate_entry

C_LIGHT = 299_792_458.0
SIGMA = 0.5
MAPPED = ("exact_oracle", "imperfect_robust")
COMPARATOR = "mapfree_coarse_to_fine"
RATE = {"exact_oracle": "supported", "imperfect_robust": "supported", COMPARATOR: "supported_common"}
TAU_BINS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, np.inf]


def u(r: pd.Series, d: pd.Series) -> pd.Series:
    return np.sqrt(r**2 + 4 * d**2) - r


def level_frame(run_dir: Path, level: str, angular: pd.DataFrame, config: PilotConfig) -> tuple[pd.DataFrame, dict]:
    trials = pd.read_parquet(run_dir / "heldout_trials.parquet")
    frame = classify(trials, config)
    geometries = pd.read_csv(run_dir / "geometries_heldout.csv").set_index("geometry_id")
    summary = json.loads((run_dir / "heldout_summary.json").read_text())
    environment = json.loads((run_dir / "environment.json").read_text())
    infeasible = (geometries["standoff_rejection"].dropna().to_dict()
                  if "standoff_rejection" in geometries else {})
    g = geometries.loc[sorted(frame["geometry_id"].unique())]
    a = angular.set_index("geometry_id").loc[g.index]
    psf = frame.groupby("geometry_id")["point_spread_delay_ns"].median()
    per_geometry = pd.DataFrame({
        "level": level,
        "delta_r_m": g["candidate_distance_m"] - g["relay_distance_m"],
        "r_t_m": g["candidate_distance_m"], "r_a_m": g["relay_distance_m"], "half_width_m": g["half_width_m"],
        "delta_tau_w_ns": (u(g["relay_distance_m"], g["half_width_m"]) - u(g["candidate_distance_m"], g["half_width_m"]))
        / C_LIGHT * 1e9,
        "psf_ns": psf,
        "delta_phi_w_deg": np.degrees(np.arctan2(2 * g["half_width_m"], g["relay_distance_m"])
                                      - np.arctan2(2 * g["half_width_m"], g["candidate_distance_m"])),
        "hpbw_tx_deg": a["hpbw_tx_deg_median"],
    })
    per_geometry["ratio_tau"] = per_geometry["delta_tau_w_ns"] / per_geometry["psf_ns"]
    per_geometry["ratio_phi"] = per_geometry["delta_phi_w_deg"] / per_geometry["hpbw_tx_deg"]
    per_geometry["delay_resolved"] = per_geometry["ratio_tau"] >= 1.0
    per_geometry["angle_resolved"] = per_geometry["ratio_phi"] >= 1.0
    frame = frame.join(per_geometry.drop(columns=["level", *frame.columns.intersection(per_geometry.columns)]),
                       on="geometry_id")
    frame["level"] = level
    grid = config.section("phase8", "stress")
    expected = (int(config.get("phase8", "geometry", "heldout_geometries")) - len(infeasible)) * len(
        grid["grid"]["sigma_map_m"]) * 2 * int(grid["grid"]["records_per_world"]) * len(grid["standoff"]["methods"])
    validity = {
        "rows_expected": len(frame) == expected,
        "split_is_heldout": set(frame["split"]) == {"heldout"},
        "methods": set(frame["method"]) == set(grid["standoff"]["methods"]),
        "single_delta_r": bool(np.allclose(per_geometry["delta_r_m"], per_geometry["delta_r_m"].iloc[0])),
        "decision_source_frozen": summary["decision_source"] == "configs phase8.frozen",
        "reclassification_matches_run": bool(
            (trials["supported"].to_numpy() == frame["supported"].to_numpy()).all()
            and (trials["supported_common"].to_numpy() == frame["supported_common"].to_numpy()).all()),
        "preflight_pass": "status: PASS" in (run_dir / "preflight.log").read_text(),
        "source_commit_clean": environment.get("source_commit", {}).get("dirty_paths") == 0,
        "source_commit": environment.get("source_commit", {}).get("commit"),
        "acquisition_failed_rows": int((frame["status"] == "ACQUISITION_FAILED").sum()),
        "infeasible_pairs": infeasible,
    }
    return frame, {"validity": validity, "per_geometry": per_geometry}


def cell_rates(rows: pd.DataFrame, by: list[str], draws: int) -> list[dict]:
    out = []
    for method, column in RATE.items():
        part = rows[rows["method"] == method]
        for key, cell in part.groupby(by, observed=True):
            boot = Bootstrap(sorted(cell["geometry_id"].unique()), draws)
            pairs = cell[["geometry_id", "level"]].drop_duplicates()
            out.append({"method": method, "rate": column, **dict(zip(by, key if isinstance(key, tuple) else (key,))),
                        "n_pairs": int(len(pairs)), **rate_entry(boot, cell, column)})
    return out


def main() -> None:
    angular, out_dir = pd.read_csv(sys.argv[1]), Path(sys.argv[2])
    levels = dict(arg.split("=", 1) for arg in sys.argv[3:])
    out_dir.mkdir(parents=True, exist_ok=True)
    config = PilotConfig.load("configs/pilot.yaml")
    draws = int(config.get("phase8", "heldout", "bootstrap_draws"))
    frames, meta = [], {}
    for level, run in levels.items():
        frame, info = level_frame(Path(run), level, angular, config)
        frames.append(frame)
        meta[level] = info
    trials = pd.concat(frames, ignore_index=True)
    replay = trials[(trials["world"] == "replay") & (trials["sigma_map_m"] == SIGMA)].copy()
    target = trials[(trials["world"] == "target") & (trials["sigma_map_m"] == SIGMA)]
    replay["tau_bin"] = pd.cut(replay["ratio_tau"], TAU_BINS, right=False).astype(str)

    per_level = pd.DataFrame(cell_rates(replay, ["level"], draws))
    per_level_target = pd.DataFrame(cell_rates(target, ["level"], draws))
    two_by_two = pd.DataFrame(cell_rates(replay, ["delay_resolved", "angle_resolved"], draws))
    two_by_two_20 = pd.DataFrame(cell_rates(replay[replay["level"] == "standoff_20m"],
                                            ["delay_resolved", "angle_resolved"], draws)) if "standoff_20m" in levels else None
    tau_bins = pd.DataFrame(cell_rates(replay, ["tau_bin"], draws))
    per_geometry = pd.concat([info["per_geometry"] for info in meta.values()]).reset_index()
    scored = replay[replay["method"].isin(list(RATE))].copy()
    scored["value"] = np.where(scored["method"] == COMPARATOR, scored["supported_common"], scored["supported"])
    rates = scored.groupby(["geometry_id", "level", "method"])["value"].mean().unstack("method")
    per_geometry = per_geometry.join(rates.add_prefix("replay_"), on=["geometry_id", "level"])
    standoffs = angular[["geometry_id", "r_t_m", "half_width_m", "hpbw_tx_deg_median", "psf_baseline_ns",
                         "delta_r_tau_baseline_m", "delta_r_phi_m"]].copy()
    standoffs["predicted_verification_radius_m"] = standoffs[["delta_r_tau_baseline_m", "delta_r_phi_m"]].min(axis=1)
    standoffs["delta_r_min_baseline_m"] = standoffs["delta_r_tau_baseline_m"]  # A5.2 Delta_R_min(B)
    # Self-check: each standoff reproduces its defining equation.
    for _, s in standoffs.iterrows():
        d, r = s.half_width_m, s.r_t_m
        ra = r - s.delta_r_tau_baseline_m
        assert abs((math.sqrt(ra**2 + 4 * d**2) - ra) - (math.sqrt(r**2 + 4 * d**2) - r)
                   - C_LIGHT * s.psf_baseline_ns * 1e-9) < 1e-6
        rp = r - s.delta_r_phi_m
        assert abs(math.degrees(math.atan2(2 * d, rp) - math.atan2(2 * d, r)) - s.hpbw_tx_deg_median) < 1e-6

    per_level.to_csv(out_dir / "replay_by_level.csv", index=False)
    per_level_target.to_csv(out_dir / "target_by_level.csv", index=False)
    two_by_two.to_csv(out_dir / "two_by_two_pooled.csv", index=False)
    if two_by_two_20 is not None:
        two_by_two_20.to_csv(out_dir / "two_by_two_standoff_20m.csv", index=False)
    tau_bins.to_csv(out_dir / "replay_by_delay_ratio_bin.csv", index=False)
    per_geometry.to_csv(out_dir / "per_geometry_level.csv", index=False)
    standoffs.to_csv(out_dir / "standoffs.csv", index=False)
    (out_dir / "validity.json").write_text(json.dumps({lv: m["validity"] for lv, m in meta.items()}, indent=2,
                                                      default=str))
    cols = ["method", "point", "ci95", "n_pairs", "n_geometries"]
    print("validity", json.dumps({lv: m["validity"] for lv, m in meta.items()}, default=str))
    print(per_level[["level"] + cols].to_string())
    print(two_by_two[["delay_resolved", "angle_resolved"] + cols].to_string())
    if two_by_two_20 is not None:
        print("standoff_20m\n" + two_by_two_20[["delay_resolved", "angle_resolved"] + cols].to_string())
    print(tau_bins[["tau_bin"] + cols].to_string())
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
