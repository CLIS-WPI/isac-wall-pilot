"""EXPLORATORY, post hoc (D1, D2): why the passive twin baseline beat the active probe on replay. No GPU, no trials.

Rows: the A11.1 passive rows (`wpc_passive.py`) and the frozen active imperfect_robust rows on the same records.
Held-out run (passive exists at sigma 0.5 only), the 6 standoff levels (sigma 0.5), and the calibration
geometries per sigma (secondary: the thresholds were fitted there).

D1 threshold-free: AUC (target vs replay) of active S_c1 and passive S_pass; a row without a finite score ranks
   lowest (it can never be SUPPORTED). Geometry-bootstrap CI (A4.1 draws). Oracle s_lo per level and score: every
   other frozen condition kept (floor k, s_hi, direct_reliable, reference_detectable); s_lo set to the largest
   value that keeps target SUPPORTED >= 0.99 on that level's own rows; replay SUPPORTED reported there.
D2 combination rules with the frozen thresholds: passive AND active, passive OR active.

Usage: python scripts/wpd_score_analysis.py PASSIVE_DIR WPC_DIR MAIN_TRIALS.parquet GEOMETRIES.csv STRESS_ROOT
         CALIBRATION_TRIALS.parquet OUT_DIR
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from isac_verify.calibration import classify_trials
from isac_verify.config import PilotConfig
from isac_verify.phase8a import frozen_decision
from phase8b_heldout_analysis import PRIMARY, Bootstrap, ci
from phase8c_threshold_sensitivity import _check_weighted_auc, weighted_auc
from wpb_reanalysis import entry
from wpc_passive_analysis import STANDOFFS, classify_passive, load

LABEL = "exploratory, post hoc"
KEY = ["geometry_id", "world", "record_index", "sigma_map_m"]
TARGET_KEEP = 0.99


def active_rows(trials: pd.DataFrame, config: PilotConfig, sigma: float | None) -> pd.DataFrame:
    """Frozen classification plus `eligible` = SUPPORTED with s_lo removed (every other condition kept)."""
    factor, fitted = frozen_decision(config)
    assert factor == 1.0
    t = trials[trials["method"] == PRIMARY]
    if sigma is not None:
        t = t[t["sigma_map_m"] == sigma]
    t = t.reset_index(drop=True)
    frozen = classify_trials(t, fitted, factor)
    if sigma is not None:  # held-out runs stored their frozen decision; calibration rows predate the freeze
        assert (frozen["supported"].to_numpy() == t["supported"].to_numpy()).all(), "frozen reclassification"
    no_lo = copy.deepcopy(fitted)
    no_lo["mapped"][PRIMARY]["s_lo"]["value"] = -math.inf
    frozen["eligible"] = classify_trials(t, no_lo, factor)["supported"].to_numpy()
    return frozen[KEY + ["supported", "eligible", "S_c1"]]


def passive_rows(frame: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    no_lo = copy.deepcopy(thresholds)
    no_lo["s_lo"]["value"] = -math.inf
    out = frame.assign(supported_pass=classify_passive(frame, thresholds), eligible_pass=classify_passive(frame, no_lo))
    return out[KEY + ["supported_pass", "eligible_pass", "S_pass"]]


def auc(boot: Bootstrap, rows: pd.DataFrame, score: str) -> dict:
    s = rows[score].astype(float).to_numpy()
    s = np.where(np.isfinite(s), s, -np.inf)
    index = {g: i for i, g in enumerate(boot.ids)}
    is_t = (rows["world"] == "target").to_numpy()
    g = rows["geometry_id"].map(index).to_numpy()
    point = weighted_auc(s[is_t], s[~is_t], np.ones((1, is_t.sum())), np.ones((1, (~is_t).sum())))[0]
    samples = weighted_auc(s[is_t], s[~is_t], boot.weights[:, g[is_t]].astype(float),
                           boot.weights[:, g[~is_t]].astype(float))
    return {"auc": float(point), "auc_ci95_lo": ci(samples)[0], "auc_ci95_hi": ci(samples)[1],
            "rows_without_finite_score": int((~np.isfinite(rows[score].astype(float))).sum())}


def oracle(boot: Bootstrap, rows: pd.DataFrame, score: str, eligible: str) -> dict:
    t, r = rows[rows["world"] == "target"], rows[rows["world"] == "replay"]
    need = math.ceil(TARGET_KEEP * len(t))
    s_t = np.sort(t.loc[t[eligible], score].astype(float).to_numpy())[::-1]
    if len(s_t) < need:  # unreachable even with s_lo = -inf
        return {"oracle_s_lo": float("nan"), "oracle_target": float(t[eligible].mean()),
                "oracle_replay": float("nan"), "oracle_replay_ci95_lo": float("nan"),
                "oracle_replay_ci95_hi": float("nan"), "oracle_replay_geometries": -1}
    s_lo = float(s_t[need - 1])
    hit = lambda part: part[eligible] & (part[score].astype(float) >= s_lo)
    rr = entry(boot, r.assign(_hit=hit(r)), "_hit")
    return {"oracle_s_lo": s_lo, "oracle_target": float(hit(t).mean()), "oracle_replay": rr["point"],
            "oracle_replay_ci95_lo": rr["ci95_lo"], "oracle_replay_ci95_hi": rr["ci95_hi"],
            "oracle_replay_geometries": rr["geometries_with_any_supported"]}


def _check_oracle() -> None:
    rows = pd.DataFrame({"geometry_id": ["a"] * 200, "world": ["target"] * 100 + ["replay"] * 100,
                         "s": np.r_[np.arange(100.0), np.arange(100.0) - 50.0], "e": True})
    o = oracle(Bootstrap(["a"], 3), rows, "s", "e")
    assert o["oracle_s_lo"] == 1.0 and o["oracle_target"] == 0.99 and o["oracle_replay"] == 0.49, o


def main() -> None:
    passive_dir, wpc_dir, main_path, geometries_path, stress_root, cal_path, out = (Path(a) for a in sys.argv[1:8])
    out.mkdir(parents=True, exist_ok=True)
    _check_weighted_auc()
    _check_oracle()
    config = PilotConfig.load("configs/pilot.yaml")
    draws = int(config.get("phase8", "heldout", "bootstrap_draws"))
    thresholds = json.loads((wpc_dir / "c1_passive_thresholds.json").read_text())
    ho_boot = Bootstrap(sorted(pd.read_csv(geometries_path)["geometry_id"]), draws)
    cal_active = pd.read_parquet(cal_path)
    cal_boot = Bootstrap(sorted(cal_active["geometry_id"].unique()), draws)

    cells = []  # (run, sigma label, merged rows, bootstrap)
    for run in ["heldout", *STANDOFFS]:
        trials = pd.read_parquet(main_path if run == "heldout" else stress_root / run / "heldout_trials.parquet")
        m = passive_rows(load(passive_dir, run), thresholds).merge(active_rows(trials, config, 0.5), on=KEY,
                                                                   validate="one_to_one")
        assert len(m) == 2 * 32 * (24 if run == "heldout" else 6), (run, len(m))
        cells.append((run, "0.5", m, ho_boot))
    cal = passive_rows(load(passive_dir, "calibration"), thresholds).merge(active_rows(cal_active, config, None),
                                                                           on=KEY, validate="one_to_one")
    assert len(cal) == 12 * 2 * 32 * 5
    assert (cal.groupby("sigma_map_m")["supported_pass"].sum() > 0).all()
    cells += [("calibration", f"{s:g}", part, cal_boot) for s, part in cal.groupby("sigma_map_m")]
    cells.append(("calibration", "pooled", cal, cal_boot))

    d1, d2 = [], []
    for run, sigma, m, boot in cells:
        m = m.assign(both=m["supported_pass"] & m["supported"], either=m["supported_pass"] | m["supported"])
        base = {"label": LABEL, "run": run, "sigma_map_m": sigma,
                "secondary": run == "calibration" or run.startswith("standoff_")}
        for name, score, elig, sup in (("active", "S_c1", "eligible", "supported"),
                                       ("passive", "S_pass", "eligible_pass", "supported_pass")):
            frozen = {w: float(m.loc[m["world"] == w, sup].mean()) for w in ("target", "replay")}
            d1.append({**base, "score": name, **auc(boot, m, score), "frozen_target": frozen["target"],
                       "frozen_replay": frozen["replay"], **oracle(boot, m, score, elig)})
        for rule in ("supported", "supported_pass", "both", "either"):
            row = {**base, "rule": {"supported": "active", "supported_pass": "passive", "both": "passive AND active",
                                    "either": "passive OR active"}[rule]}
            for w in ("target", "replay"):
                row |= {f"{w}_{k}": v for k, v in entry(boot, m[m["world"] == w], rule).items()}
            d2.append(row)
    d1, d2 = pd.DataFrame(d1), pd.DataFrame(d2)
    d1.to_csv(out / "d1_threshold_free.csv", index=False)
    d2.to_csv(out / "d2_combination_rules.csv", index=False)
    pd.set_option("display.width", 250)
    print(d1[["run", "sigma_map_m", "score", "auc", "auc_ci95_lo", "auc_ci95_hi", "rows_without_finite_score",
              "frozen_target", "frozen_replay", "oracle_s_lo", "oracle_target", "oracle_replay",
              "oracle_replay_ci95_lo", "oracle_replay_ci95_hi", "oracle_replay_geometries"]].round(4).to_string())
    print(d2[["run", "sigma_map_m", "rule", "target_point", "replay_point", "replay_ci95_lo", "replay_ci95_hi",
              "replay_geometries_with_any_supported", "replay_cp95_upper_if_zero"]].round(4).to_string())


if __name__ == "__main__":
    main()
