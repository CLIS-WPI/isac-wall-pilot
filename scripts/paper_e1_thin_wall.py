"""EXPLORATORY, post hoc: 50 mm wall level, replay SUPPORTED in the delta_tau_W/PSF >= 1 stratum.

Lists the imperfect_robust replay SUPPORTED rows (sigma 0.5, frozen decisions) per geometry and stratum, with the
same rows' S in the main held-out run where the twin realization matches (records 0, 1, 4, 5). Also labels the
noiseless pair decomposition made by scripts/phase8c_pair_probe.py for one >= 1 geometry.

Usage: python scripts/paper_e1_thin_wall.py RUN_DIR MAIN_TRIALS.parquet STRATA.csv PAIR_PROBE.csv OUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from isac_verify.config import PilotConfig
from phase8b_heldout_analysis import PRIMARY, classify
from phase8c_bandwidth_analysis import SIGMA

LABEL = "exploratory, post hoc"
LEVEL = "wall_thickness_50mm"


def main() -> None:
    run_dir, main_path, strata_path, probe_path, out = (Path(a) for a in sys.argv[1:6])
    out.mkdir(parents=True, exist_ok=True)
    strata = pd.read_csv(strata_path)
    strata = strata[strata["level"] == LEVEL].set_index("geometry_id")
    frame = classify(pd.read_parquet(run_dir / "heldout_trials.parquet"), PilotConfig.load("configs/pilot.yaml"))
    rows = frame[(frame["method"] == PRIMARY) & (frame["world"] == "replay") & (frame["sigma_map_m"] == SIGMA)]
    base = pd.read_parquet(main_path)
    base = base[(base["method"] == PRIMARY) & (base["world"] == "replay") & (base["sigma_map_m"] == SIGMA)]
    key = ["geometry_id", "record_index", "twin_index"]
    hits = rows[rows["supported"]][key + ["S_c1"]].merge(
        base[key + ["S_c1", "supported"]], on=key, how="left", suffixes=("", "_main_10cm"))
    hits = hits.join(strata[["stratum", "ratio_median", "delta_tau_w_ns"]], on="geometry_id")
    hits.insert(0, "label", LABEL)
    hits.to_csv(out / "e1_thin_wall_replay_supported.csv", index=False)
    per_geometry = hits.groupby(["geometry_id", "stratum"]).agg(
        supported_rows=("S_c1", "size"), ratio_median=("ratio_median", "first")).reset_index()
    per_geometry["rows_in_cell"] = per_geometry["geometry_id"].map(rows.groupby("geometry_id").size())
    per_geometry.insert(0, "label", LABEL)
    per_geometry.to_csv(out / "e1_thin_wall_geometries.csv", index=False)
    probe = pd.read_csv(probe_path)
    probe.insert(0, "label", LABEL)
    probe.sort_values("share_of_relay_gate_energy", ascending=False).to_csv(out / probe_path.name, index=False)
    print(per_geometry.to_string())
    print(hits[hits["stratum"] == "ge1"].round(4).to_string())


if __name__ == "__main__":
    main()
