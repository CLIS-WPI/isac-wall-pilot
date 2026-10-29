"""EXPLORATORY, post hoc: offline escalation policy from existing rows only (no new trials, no new thresholds).

Rows: primary method `imperfect_robust`, sigma 0.5, in the main held-out run (baseline profile, 122.88 MHz)
and the bw246 stress run (245.76 MHz). A row is paired across runs by (geometry_id, world, record_index).
Primary pairing: records {0, 1, 4, 5}, where both runs used the same twin realization (main: twin = r mod 4,
stress: twin = r mod 2). Secondary: all 8 stress records (twins differ for records 2, 3, 6, 7).

Policies (decisions are the frozen ones each run already recorded):
  P0 one wall probe at 122.88 MHz;  P1 one wall probe at 245.76 MHz;
  P2 probe at 122.88 MHz, and if SUPPORTED a second probe at 245.76 MHz; SUPPORTED only if both are.
Cost per candidate in baseline-probe units (same probe duration; a 245.76 MHz probe costs 2):
  P0 = 1, P1 = 2, P2 = 1 + 2 * [first probe SUPPORTED].
Intervals: A4.1 geometry-level bootstrap (`phase8b_heldout_analysis.Bootstrap`).

Usage: python scripts/paper_e2_escalation.py MAIN_RUN_DIR BW246_RUN_DIR OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from isac_verify.config import PilotConfig
from phase8b_heldout_analysis import PRIMARY, Bootstrap, classify, rate_entry
from phase8c_bandwidth_analysis import MATCHED_RECORDS, SIGMA

LABEL = "exploratory, post hoc"
KEY = ["geometry_id", "world", "record_index"]


def streams_shared(main_dir: Path, stress_dir: Path) -> bool:
    """True when every seed stream of every stress record equals the main run's for the same slot and record."""
    m, s = (json.loads((d / "seed_manifest_heldout.json").read_text())["slots"] for d in (main_dir, stress_dir))

    def same(a, b) -> bool:
        if isinstance(b, dict):
            return all(same(a[k], v) for k, v in b.items())
        return a[: len(b)] == b if isinstance(b, list) else a == b

    return all(x["physical_geometry"] == y["physical_geometry"] and same(x["streams"], y["streams"])
               for x, y in zip(m, s, strict=False)) and len(s) <= len(m)


def main() -> None:
    main_dir, stress_dir, out_dir = (Path(a) for a in sys.argv[1:4])
    out_dir.mkdir(parents=True, exist_ok=True)
    config = PilotConfig.load("configs/pilot.yaml")
    draws = int(config.get("phase8", "heldout", "bootstrap_draws"))
    base = pd.read_parquet(main_dir / "heldout_trials.parquet")
    base = base[(base["method"] == PRIMARY) & (base["sigma_map_m"] == SIGMA)]
    wide = classify(pd.read_parquet(stress_dir / "heldout_trials.parquet"), config)
    wide = wide[(wide["method"] == PRIMARY) & (wide["sigma_map_m"] == SIGMA)]
    assert set(base["bandwidth_profile"]) == {"baseline"} and set(wide["bandwidth_profile"]) == {"bw246"}
    pairs = base[KEY + ["twin_index", "supported"]].merge(
        wide[KEY + ["twin_index", "supported"]], on=KEY, suffixes=("_122", "_246"), validate="one_to_one")
    assert len(pairs) == len(wide)
    same_twin = pairs["twin_index_122"] == pairs["twin_index_246"]
    assert (same_twin == pairs["record_index"].isin(MATCHED_RECORDS)).all()
    pairs["P0"] = pairs["supported_122"].astype(bool)
    pairs["P1"] = pairs["supported_246"].astype(bool)
    pairs["P2"] = pairs["P0"] & pairs["P1"]
    pairs["cost_P0"], pairs["cost_P1"] = 1.0, 2.0
    pairs["cost_P2"] = 1.0 + 2.0 * pairs["P0"]
    boot = Bootstrap(sorted(pairs["geometry_id"].unique()), draws)

    rows = []
    for pairing, part in (("matched_records_0_1_4_5", pairs[same_twin]), ("all_8_records", pairs)):
        for policy in ("P0", "P1", "P2"):
            entry = {"label": LABEL, "pairing": pairing, "policy": policy}
            for world in ("target", "replay"):
                w = part[part["world"] == world]
                r, c = rate_entry(boot, w, policy), rate_entry(boot, w, f"cost_{policy}")
                entry |= {f"{world}_supported": r["point"], f"{world}_ci95_lo": r["ci95"][0],
                          f"{world}_ci95_hi": r["ci95"][1], f"{world}_cost": c["point"],
                          f"{world}_cost_ci95_lo": c["ci95"][0], f"{world}_cost_ci95_hi": c["ci95"][1],
                          f"{world}_n_records": r["n_rows"], "n_geometries": r["n_geometries"]}
            rows.append(entry)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "e2_escalation.csv", index=False)
    (out_dir / "e2_escalation_README.json").write_text(json.dumps({
        "label": LABEL, "method": PRIMARY, "sigma_map_m": SIGMA,
        "pairing": "same (geometry_id, world, record_index) in both runs; primary: records 0,1,4,5 (same twin)",
        "seed_streams_shared": streams_shared(main_dir, stress_dir),
        "note": "Seed streams (probe symbols, receiver noise, relay noise, twin error, geometry) are shared "
                "across the two runs, so the 245.76 MHz probe is not an independent noise realization of "
                "the 122.88 MHz probe (the arrays differ in size, so the draws are not identical either; their "
                "dependence was not characterized). P2's rates are not those of an independently re-drawn "
                "second probe.",
        "cost_unit": "one 122.88 MHz wall probe; a 245.76 MHz probe of the same duration costs 2",
    }, indent=2))
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
