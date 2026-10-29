"""ANALYSIS_PLAN A12.4 matched rules (AUC unchanged; replay similar to the mirrored A11.1 run) and the record of the
stored-vs-recomputed E_D / beam_phi_deg differences (they measure A12.1).

Usage: python scripts/a12_c1_verdicts.py PASSIVE_DIR OUT_DIR   (reads docs/results/{wpc,wpd,wpc_a12,wpd_a12})
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from wpc_passive_analysis import load

R = Path("docs/results")


def main() -> None:
    passive_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    new_auc, old_auc = (pd.read_csv(R / d / "d1_threshold_free.csv") for d in ("wpd_a12", "wpd"))
    new_c1, old_c1 = (pd.read_csv(R / d / "c1_passive_vs_active.csv") for d in ("wpc_a12", "wpc"))
    runs = ["heldout", "standoff_03m", "standoff_06m", "standoff_10m", "standoff_15m", "standoff_20m", "standoff_30m"]
    get = lambda f, run, col, **kw: float(f[(f["run"] == run) & np.logical_and.reduce(
        [f[k] == v for k, v in kw.items()])][col].iloc[0])
    auc = {r: {"new": get(new_auc, r, "auc", score="passive"), "mirrored": get(old_auc, r, "auc", score="passive")}
           for r in runs}
    replay = {r: {"new": get(new_c1, r, "passive_point", world="replay"),
                  "mirrored": get(old_c1, r, "passive_point", world="replay")} for r in runs}
    auc_runs = ["heldout", "standoff_10m", "standoff_15m", "standoff_20m", "standoff_30m"]
    rows = pd.concat([load(passive_dir, r) for r in ["calibration", *runs]], ignore_index=True)
    ok = rows["E_D"].notna()
    rel = ((rows.loc[ok, "E_D"] - rows.loc[ok, "E_D_stored"]).abs() / rows.loc[ok, "E_D_stored"].abs())
    both = rows["beam_phi_deg"].notna() & rows["beam_phi_stored"].notna()
    dphi = rows.loc[both, "beam_phi_deg"] - rows.loc[both, "beam_phi_stored"]
    flipped = (np.sign(rows.loc[both, "beam_phi_deg"]) != np.sign(rows.loc[both, "beam_phi_stored"]))
    result = {
        "label": "A12.4 matched rules (pre-registered) and self-check record",
        "auc_passive": auc,
        "auc_unchanged_1000": {r: auc[r]["new"] == 1.0 for r in auc_runs},
        "replay_passive": replay,
        "replay_similar_within_0.10": {r: abs(replay[r]["new"] - replay[r]["mirrored"]) <= 0.10 for r in runs},
        "self_check": {"rows": int(len(rows)), "E_D_rows": int(ok.sum()), "E_D_rel_diff_max": float(rel.max()),
                       "E_D_rel_diff_median": float(rel.median()), "beam_rows": int(both.sum()),
                       "beam_phi_sign_flips": int(flipped.sum()),
                       "beam_phi_abs_diff_max_same_sign": float(dphi[~flipped].abs().max()),
                       "beam_phi_abs_diff_median_same_sign": float(dphi[~flipped].abs().median())},
    }
    (out / "a12_4_verdicts.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
