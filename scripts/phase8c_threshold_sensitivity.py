"""EXPLORATORY, post hoc: sensitivity of the main held-out result to the lower score bound s_lo.

Changes no decision or label. On the main held-out run, the mapped methods
`imperfect_robust` and `imperfect_narrow` are reclassified with `s_lo` replaced
(each method's frozen value, 0.0218, 0.05, 0.1, 0.2, 0.3, 0.5). Every other
frozen condition (floor k, s_hi, direct_reliable, reference_detectable,
acquisition) is unchanged. Replay and target SUPPORTED are reported pooled over
sigma and per sigma with the A4.1 geometry-level bootstrap.

AUC of the score S (higher = target-like), target vs replay, per A3.1
delta_tau_W / PSF stratum, method and sigma, over rows with finite S. Interval:
the same geometry-level bootstrap (weighted Mann-Whitney with geometry
multiplicities).

Usage: python scripts/phase8c_threshold_sensitivity.py MAIN_TRIALS.parquet GEOMETRIES.csv OUT_DIR
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from isac_verify.calibration import classify_trials
from isac_verify.config import PilotConfig
from isac_verify.phase8a import frozen_decision
from phase8b_heldout_analysis import Bootstrap, ci, rate_entry, strata

METHODS = ("imperfect_robust", "imperfect_narrow")
S_LO_GRID = (0.0218, 0.05, 0.1, 0.2, 0.3, 0.5)


def weighted_auc(target: np.ndarray, replay: np.ndarray, w_target: np.ndarray, w_replay: np.ndarray) -> np.ndarray:
    """P(S_target > S_replay) + 0.5 P(tie) for each row of weights (draws x rows)."""
    scores = np.concatenate([target, replay])
    order = np.argsort(scores, kind="mergesort")
    is_target = np.concatenate([np.ones(len(target), bool), np.zeros(len(replay), bool)])[order]
    weights = np.concatenate([w_target, w_replay], axis=1)[:, order]
    replay_w = np.where(is_target, 0.0, weights)
    target_w = np.where(is_target, weights, 0.0)
    _, first, counts = np.unique(scores[order], return_index=True, return_counts=True)
    group = np.repeat(np.arange(len(first)), counts)
    below = (np.cumsum(replay_w, axis=1) - replay_w)[:, first][:, group]  # replay weight at strictly lower S
    ties = np.add.reduceat(replay_w, first, axis=1)[:, group]  # replay weight at equal S
    return (target_w * (below + 0.5 * ties)).sum(axis=1) / (target_w.sum(axis=1) * replay_w.sum(axis=1))


def _check_weighted_auc() -> None:
    """Self-check against the brute-force pairwise definition, with ties and weights."""
    rng = np.random.default_rng(1)
    t, r = np.round(rng.normal(1, 1, 50), 1), np.round(rng.normal(0, 1, 40), 1)
    wt, wr = rng.integers(0, 3, 50).astype(float), rng.integers(0, 3, 40).astype(float)
    pair = (t[:, None] > r[None, :]) + 0.5 * (t[:, None] == r[None, :])
    assert abs(weighted_auc(t, r, wt[None], wr[None])[0] - (wt @ pair @ wr) / (wt.sum() * wr.sum())) < 1e-12


def auc_entry(rows: pd.DataFrame, boot: Bootstrap) -> dict:
    finite = rows[np.isfinite(rows["S_c1"].astype(float))]
    t, r = finite[finite["world"] == "target"], finite[finite["world"] == "replay"]
    if len(t) == 0 or len(r) == 0:
        return {"auc": float("nan"), "ci95": [float("nan")] * 2, "n_target": len(t), "n_replay": len(r)}
    index = {g: i for i, g in enumerate(boot.ids)}
    gt, gr = t["geometry_id"].map(index).to_numpy(), r["geometry_id"].map(index).to_numpy()
    point = weighted_auc(t["S_c1"].to_numpy(float), r["S_c1"].to_numpy(float),
                         np.ones((1, len(t))), np.ones((1, len(r))))[0]
    samples = weighted_auc(t["S_c1"].to_numpy(float), r["S_c1"].to_numpy(float),
                           boot.weights[:, gt].astype(float), boot.weights[:, gr].astype(float))
    return {"auc": float(point), "ci95": ci(samples), "n_target": int(len(t)), "n_replay": int(len(r)),
            "rows_without_finite_S": int(len(rows) - len(finite)),
            "n_geometries": int(finite["geometry_id"].nunique())}


def main() -> None:
    trials_path, geometries_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_weighted_auc()
    config = PilotConfig.load("configs/pilot.yaml")
    draws = int(config.get("phase8", "heldout", "bootstrap_draws"))
    geometries = pd.read_csv(geometries_path)
    boot = Bootstrap(sorted(geometries["geometry_id"]), draws)
    trials = pd.read_parquet(trials_path)
    trials = trials[trials["method"].isin(METHODS)].reset_index(drop=True)
    factor, fitted = frozen_decision(config)
    assert factor == 1.0  # S_c1 is the decision score
    frozen_run = classify_trials(trials, fitted, factor)
    assert (frozen_run["supported"].to_numpy() == trials["supported"].to_numpy()).all(), "frozen reclassification"
    sigmas = sorted(float(s) for s in trials["sigma_map_m"].unique())

    rows = []
    for method in METHODS:
        frozen_s_lo = float(fitted["mapped"][method]["s_lo"]["value"])
        # 0.0218 is the rounded frozen robust value; a grid value within 1e-4 of the frozen one is the frozen one.
        for s_lo in sorted({frozen_s_lo, *(v for v in S_LO_GRID if abs(v - frozen_s_lo) > 1e-4)}):
            changed = copy.deepcopy(fitted)
            changed["mapped"][method]["s_lo"]["value"] = s_lo
            frame = classify_trials(trials[trials["method"] == method], changed, factor)
            for world in ("replay", "target"):
                part = frame[frame["world"] == world]
                for label, cell in [("pooled", part)] + [(f"{s:g}", part[part["sigma_map_m"] == s]) for s in sigmas]:
                    rows.append({"exploratory": True, "method": method, "s_lo": s_lo,
                                 "s_lo_is_frozen": s_lo == frozen_s_lo, "world": world, "sigma_map_m": label,
                                 **rate_entry(boot, cell, "supported")})
    rates = pd.DataFrame(rows)

    s = strata(trials, geometries)
    trials["stratum"] = trials["geometry_id"].map(s["stratum"])
    aucs = []
    for method in METHODS:
        part = trials[trials["method"] == method]
        for stratum in ("lt1", "ge1", "all"):
            st = part if stratum == "all" else part[part["stratum"] == stratum]
            ids = sorted(st["geometry_id"].unique())
            sboot = Bootstrap(ids, draws)
            for label, cell in [("pooled", st)] + [(f"{x:g}", st[st["sigma_map_m"] == x]) for x in sigmas]:
                aucs.append({"exploratory": True, "method": method, "stratum": stratum, "sigma_map_m": label,
                             **auc_entry(cell, sboot)})
    auc = pd.DataFrame(aucs)

    rates.to_csv(out_dir / "supported_vs_s_lo.csv", index=False)
    auc.to_csv(out_dir / "auc_by_stratum.csv", index=False)
    s.reset_index().to_csv(out_dir / "strata.csv", index=False)
    (out_dir / "README.json").write_text(json.dumps({
        "label": "EXPLORATORY, post hoc threshold sensitivity; changes no decision or label",
        "source": str(trials_path), "frozen_s_lo": {m: fitted["mapped"][m]["s_lo"]["value"] for m in METHODS},
        "s_hi": {m: fitted["mapped"][m]["s_hi"]["value"] for m in METHODS},
        "bootstrap_draws": draws, "score_column": "S_c1",
    }, indent=2))
    show = rates.copy()
    show["ci95"] = show["ci95"].map(lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]")
    print(show[["method", "s_lo", "world", "sigma_map_m", "point", "ci95"]].round(4).to_string())
    a = auc.copy()
    a["ci95"] = a["ci95"].map(lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]")
    print(a[["method", "stratum", "sigma_map_m", "auc", "ci95", "n_target", "n_replay", "n_geometries",
             "rows_without_finite_S"]].round(4).to_string())
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
