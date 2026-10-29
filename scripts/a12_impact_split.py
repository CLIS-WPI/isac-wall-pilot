"""EXPLORATORY, post hoc: A12.3 differences split by whether the planned wall side flipped; decision flips."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from isac_verify.config import PilotConfig
from phase8b_heldout_analysis import classify
cfg = PilotConfig.load("configs/pilot.yaml")
K = ["geometry_id", "world", "record_index", "sigma_map_m", "method"]
new = classify(pd.concat([pd.read_parquet(p) for p in sorted(__import__("glob").glob("results/a12_impact/shards/geometry_*.parquet"))]), cfg)
old = pd.read_parquet(".worktrees/heldout/results/phase8b_heldout/heldout_trials.parquet")
old = classify(old[old.geometry_id.isin(new.geometry_id.unique()) & (old.record_index < 4)], cfg)
m = old.merge(new, on=K, suffixes=("_old", "_new"))
m["side_same"] = m.wall_side_old.astype(str) == m.wall_side_new.astype(str)
f = np.isfinite(m.S_c1_old) & np.isfinite(m.S_c1_new)
m["rel"] = (m.S_c1_new - m.S_c1_old).abs() / m.S_c1_old.abs()
out = {}
for same, part in m[f].groupby("side_same"):
    out[f"side_same={same}"] = {"rows": len(part), "rel_max": float(part.rel.max()), "rel_median": float(part.rel.median()),
                                "rows_rel_ge_1e-3": int((part.rel >= 1e-3).sum())}
out["side_same_rel_median_by_world_method"] = {f"{w}/{k}": float(v) for (w, k), v in
                                               m[f & m.side_same].groupby(["world", "method"]).rel.median().items()}
out["side_flips_by_method"] = m[~m.side_same & m.wall_side_old.notna()].groupby("method").size().to_dict()
out["side_flips_by_sigma"] = {f"{k:g}": v for k, v in m[~m.side_same & m.wall_side_old.notna()].groupby("sigma_map_m").size().items()}
out["side_flips_by_geometry"] = m[~m.side_same & m.wall_side_old.notna()].groupby("geometry_id").size().to_dict()
out["candidate_aoa_old_vs_new_sum_max_abs"] = float((m.candidate_aoa_deg_old + m.candidate_aoa_deg_new).abs().max())
flip = m[m.supported_old != m.supported_new]
out["decision_flips"] = flip[K + ["supported_old", "supported_new", "S_c1_old", "S_c1_new", "wall_side_old", "wall_side_new", "beam_phi_deg_old", "beam_phi_deg_new"]].to_dict("records")
json.dump(out, open("docs/results/a12/a12_3_by_wall_side.json", "w"), indent=2, default=str)
print(json.dumps(out, indent=1, default=str))
