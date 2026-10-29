"""ANALYSIS_PLAN A12.3: impact of the A12.1 receiver sign fix on the frozen probe results.

run:     one held-out slot, records 0-3, all sigma, both worlds, all methods, through the unchanged
         `phase8a._run_slots` (same seeds) with the fixed receiver, into OUT_DIR/shards.
compare: the new rows against the frozen run on (geometry_id, world, record_index, sigma_map_m, method).
         Decisions (`status`, frozen `supported`, `supported_common`) and S_c1 (relative difference); every other
         non-timing column that differs is listed with its largest absolute and relative difference.

Usage (GPU, held-out approved):
  python scripts/a12_impact.py run OUT_DIR SLOT
  python scripts/a12_impact.py compare OUT_DIR FROZEN_TRIALS.parquet RESULT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["geometry_id", "world", "record_index", "sigma_map_m", "method"]
RECORDS = 4
TIMING = ("runtime", "elapsed", "_s_", "time")


def run(out: Path, slot: int) -> None:
    from isac_verify.config import PilotConfig
    from isac_verify.phase8a import HELDOUT, _run_slots, approve_heldout

    approve_heldout()
    (out / "shards").mkdir(parents=True, exist_ok=True)
    _run_slots(PilotConfig.load("configs/pilot.yaml"), out, [slot], RECORDS, HELDOUT)


def compare(out: Path, frozen_path: Path, result: Path) -> None:
    from isac_verify.config import PilotConfig
    from phase8b_heldout_analysis import classify

    config = PilotConfig.load("configs/pilot.yaml")
    new = pd.concat([pd.read_parquet(p) for p in sorted((out / "shards").glob("geometry_*.parquet"))],
                    ignore_index=True)
    frozen = pd.read_parquet(frozen_path)
    frozen = frozen[frozen["geometry_id"].isin(new["geometry_id"].unique()) & (frozen["record_index"] < RECORDS)]
    new_c, old_c = classify(new, config), classify(frozen, config)
    assert (old_c["supported"].to_numpy() == frozen["supported"].to_numpy()).all(), "frozen reclassification"
    m = old_c.merge(new_c, on=KEY, suffixes=("_old", "_new"), validate="one_to_one")
    assert len(m) == len(new) == len(frozen) == 2 * 5 * RECORDS * 6 * new["geometry_id"].nunique(), len(m)

    def differs(col: str) -> pd.Series:
        a, b = m[f"{col}_old"], m[f"{col}_new"]
        if a.dtype == object:  # list-valued columns compare by their text form
            a, b = a.astype(str), b.astype(str)
        return ~((a == b) | (a.isna() & b.isna()))

    decisions = {c: int(differs(c).sum()) for c in ("status", "supported", "supported_common")}
    s_old, s_new = m["S_c1_old"].astype(float), m["S_c1_new"].astype(float)
    finite = np.isfinite(s_old) & np.isfinite(s_new)
    rel = ((s_new - s_old).abs() / s_old.abs())[finite]
    columns = {}
    for col in [c for c in new.columns if c not in KEY and not any(t in c for t in TIMING)]:
        if f"{col}_old" not in m or not differs(col).any():
            continue
        a, b = m[f"{col}_old"], m[f"{col}_new"]
        entry = {"rows_differing": int(differs(col).sum())}
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            d = (b.astype(float) - a.astype(float)).abs()
            entry |= {"max_abs_diff": float(d.max()),
                      "max_rel_diff": float((d / a.astype(float).abs()).replace(np.inf, np.nan).max())}
        columns[col] = entry
    verdict = {"identical_decisions": all(v == 0 for v in decisions.values()),
               "s_rel_below_1e-3": bool((rel < 1e-3).all())}
    result.mkdir(parents=True, exist_ok=True)
    flagged = m[differs("status") | differs("supported") | differs("supported_common")
                | (finite & ((s_new - s_old).abs() / s_old.abs() >= 1e-3))]
    flagged[KEY + ["status_old", "status_new", "supported_old", "supported_new", "S_c1_old", "S_c1_new",
                   "candidate_aoa_deg_old", "candidate_aoa_deg_new", "beam_phi_deg_old", "beam_phi_deg_new"]].to_csv(
        result / "a12_3_flagged_rows.csv", index=False)
    summary = {"label": "A12.3 pre-registered", "rows": len(m), "geometries": sorted(new["geometry_id"].unique()),
               "records": list(range(RECORDS)), "decision_rows_differing": decisions,
               "S_c1_finite_rows": int(finite.sum()), "S_c1_rel_diff_max": float(rel.max()),
               "S_c1_rel_diff_median": float(rel.median()), "S_c1_rows_rel_ge_1e-3": int((rel >= 1e-3).sum()),
               "nonfinite_S_c1_mismatch": int((np.isfinite(s_old) != np.isfinite(s_new)).sum()),
               "verdict": verdict, "differing_columns": columns}
    (result / "a12_3_impact.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "differing_columns"}, indent=1))
    print(pd.DataFrame(columns).T.to_string())


def main() -> None:
    np.seterr(all="ignore")
    if sys.argv[1] == "run":
        run(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        compare(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))


if __name__ == "__main__":
    main()
