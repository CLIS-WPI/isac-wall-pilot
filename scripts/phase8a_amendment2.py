"""ANALYSIS_PLAN Amendment 2: per-sigma s_lo / t_m from the existing calibration trials.

Evaluator side, calibration data only (no new trials, no held-out data).
Usage: python scripts/phase8a_amendment2.py [results/phase8a]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from isac_verify.calibration import (
    MAPPED_METHODS,
    WALL_PROBING_METHODS,
    _col,
    _numeric,
    classify_trials,
    rate_table,
    s_lo_value,
)

C_LIGHT = 299_792_458.0
MAX_RATE = 0.05


def fitted_from_thresholds(thresholds: dict) -> dict:
    return {
        "mapped": {
            m: {
                "floor_k": thresholds["floor_k"][m],
                "s_hi": thresholds["s_hi"][m],
                "s_lo": thresholds["s_lo"][m],
            }
            for m in MAPPED_METHODS
        },
        "common": {m: thresholds["t_common"][m] for m in WALL_PROBING_METHODS},
    }


def per_sigma_minimum(values_by_cell: dict[float, tuple[np.ndarray, int]]) -> dict:
    """Smallest value whose replay rate is <= MAX_RATE in every sigma cell.

    The per-cell rate is non-increasing in the threshold, so the answer is the
    largest of the per-cell rule-4 minima (``s_lo_value`` applied per cell).
    """
    cells = {
        float(sigma): s_lo_value(values, rows, MAX_RATE)
        for sigma, (values, rows) in values_by_cell.items()
    }
    binding = max(cells, key=lambda sigma: cells[sigma]["value"])
    return {
        "value": cells[binding]["value"],
        "binding_sigma_m": binding,
        "per_sigma": {str(s): cell for s, cell in cells.items()},
        "rule": "amendment 2 (A2.2): smallest value with replay-world rate <= 0.05 in every sigma cell",
    }


def amend(trials: pd.DataFrame, fitted: dict, factor: float) -> dict:
    new = json.loads(json.dumps(fitted))
    for method in MAPPED_METHODS:
        rows = trials[trials["method"] == method]
        k = fitted["mapped"][method]["floor_k"]["value"]
        s_hi = fitted["mapped"][method]["s_hi"]["value"]
        score = _numeric(rows, _col("S", factor))
        eligible = (
            (rows["world"] == "replay")
            & (_numeric(rows, "direct_reliable") == 1.0)
            & (_numeric(rows, _col("ref_det", factor)) == 1.0)
            & (_numeric(rows, _col("wall_excess_snr", factor)) > k)
            & score.notna()
            & (score <= s_hi)
        )
        replay = rows["world"] == "replay"
        new["mapped"][method]["s_lo"] = per_sigma_minimum(
            {
                sigma: (
                    score[eligible & (rows["sigma_map_m"] == sigma)].to_numpy(),
                    int((replay & (rows["sigma_map_m"] == sigma)).sum()),
                )
                for sigma in sorted(rows["sigma_map_m"].unique())
            }
        )
    for method in WALL_PROBING_METHODS:
        rows = trials[(trials["method"] == method) & (trials["world"] == "replay")]
        name = _col("wall_excess_snr", factor) if method in MAPPED_METHODS else "wall_excess_snr"
        excess = _numeric(rows, name)
        new["common"][method] = per_sigma_minimum(
            {
                sigma: (
                    excess[(rows["sigma_map_m"] == sigma) & excess.notna()].to_numpy(),
                    int((rows["sigma_map_m"] == sigma).sum()),
                )
                for sigma in sorted(rows["sigma_map_m"].unique())
            }
        )
    return new


def pooled(classified: pd.DataFrame) -> dict:
    out = {}
    for (method, world), group in classified.groupby(["method", "world"]):
        out[f"{method}/{world}"] = {
            "supported": float(group["supported"].mean()) if method in MAPPED_METHODS else None,
            "supported_common": float(group["supported_common"].mean())
            if method in WALL_PROBING_METHODS
            else None,
        }
    return out


def geometry_diagnostics(classified: pd.DataFrame, geometries: pd.DataFrame, label: str) -> pd.DataFrame:
    g = geometries.set_index("geometry_id")
    r_t, r_a, d = g["candidate_distance_m"], g["relay_distance_m"], g["half_width_m"]
    # BRIEF §5.3 image method: outgoing wall leg plus direct return, target vs causal relay.
    tau_wall = (np.sqrt(r_t**2 + 4 * d**2) + r_t) / C_LIGHT
    tau_relay_wall = (np.sqrt(r_a**2 + 4 * d**2) + r_a) / C_LIGHT + 2 * (r_t - r_a) / C_LIGHT
    psf_ns = classified.groupby("geometry_id")["point_spread_delay_ns"].median()
    out = pd.DataFrame(
        {
            "true_wall_delay_ns": tau_wall * 1e9,
            "relay_wall_delay_ns": tau_relay_wall * 1e9,
            "delta_tau_w_ns": (tau_relay_wall - tau_wall) * 1e9,
            "psf_delay_ns": psf_ns,
            "two_d_over_r_t": 2 * d / r_t,
            "r_a_over_r_t": r_a / r_t,
        }
    )
    out["delta_tau_w_over_psf"] = out["delta_tau_w_ns"] / out["psf_delay_ns"]
    replay = classified[classified["world"] == "replay"]
    for method in MAPPED_METHODS:
        rows = replay[replay["method"] == method]
        out[f"{label}_supported_{method}"] = rows.groupby("geometry_id")["supported"].mean()
    for method in WALL_PROBING_METHODS:
        rows = replay[replay["method"] == method]
        out[f"{label}_common_{method}"] = rows.groupby("geometry_id")["supported_common"].mean()
    return out


def main() -> None:
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase8a")
    out_dir = base / "amendment2"
    out_dir.mkdir(exist_ok=True)
    trials = pd.read_parquet(base / "calibration_trials.parquet")
    thresholds = json.loads((base / "thresholds_pooled_preA2.json").read_text())
    geometries = pd.read_csv(base / "geometries_calibration.csv")
    factor = float(thresholds["gate_inflation_c"]["value"])
    # The image-method delay must match the solver's true wall delay (self-check of geometry_diagnostics' model).
    truth = trials.groupby("geometry_id")["true_wall_delay_ns"].first()
    g = geometries.set_index("geometry_id")
    model = (np.sqrt(g["candidate_distance_m"] ** 2 + 4 * g["half_width_m"] ** 2) + g["candidate_distance_m"]) / C_LIGHT * 1e9
    assert np.allclose(model.loc[truth.index], truth, rtol=0, atol=1e-3), (model - truth).abs().max()

    old = fitted_from_thresholds(thresholds)
    old_classified = classify_trials(trials, old, factor)
    old_rates = rate_table(old_classified)
    # Self-check: re-classification reproduces the committed v5 rates exactly.
    reference = pd.read_csv(base / "rates.csv")
    merged = reference.merge(old_rates, on=["method", "sigma_map_m", "world"], suffixes=("_ref", "_re"))
    assert len(merged) == len(reference) == len(old_rates)
    for column in ("supported_rate", "supported_common_rate"):
        a, b = merged[f"{column}_ref"].to_numpy(float), merged[f"{column}_re"].to_numpy(float)
        assert np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-12), column

    new = amend(trials, old, factor)
    new_classified = classify_trials(trials, new, factor)
    new_rates = rate_table(new_classified)
    replay = new_rates[new_rates["world"] == "replay"]
    for column in ("supported_rate", "supported_common_rate"):
        values = replay[column].dropna()
        assert (values <= MAX_RATE + 1e-12).all(), (column, values.max())

    summary = {
        "source": str(base / "calibration_trials.parquet"),
        "gate_inflation_c": factor,
        "unchanged": {"floor_k": {m: old["mapped"][m]["floor_k"]["value"] for m in MAPPED_METHODS},
                      "s_hi": {m: old["mapped"][m]["s_hi"]["value"] for m in MAPPED_METHODS}},
        "s_lo": {m: {"old": old["mapped"][m]["s_lo"]["value"], "new": new["mapped"][m]["s_lo"]["value"],
                     "binding_sigma_m": new["mapped"][m]["s_lo"]["binding_sigma_m"],
                     "per_sigma": new["mapped"][m]["s_lo"]["per_sigma"]} for m in MAPPED_METHODS},
        "t_common": {m: {"old": old["common"][m]["value"], "new": new["common"][m]["value"],
                         "binding_sigma_m": new["common"][m]["binding_sigma_m"],
                         "per_sigma": new["common"][m]["per_sigma"]} for m in WALL_PROBING_METHODS},
        "pooled_rates_old": pooled(old_classified),
        "pooled_rates_new": pooled(new_classified),
        "config_block": {
            "gate_inflation_c": factor,
            "thresholds": {
                m: {"noise_sigmas": old["mapped"][m]["floor_k"]["value"],
                    "supported_min": new["mapped"][m]["s_lo"]["value"],
                    "supported_max": old["mapped"][m]["s_hi"]["value"]}
                for m in MAPPED_METHODS
            },
            "t_common": {m: new["common"][m]["value"] for m in WALL_PROBING_METHODS},
        },
    }
    (out_dir / "thresholds_amendment2.json").write_text(json.dumps(summary, indent=2))
    old_rates.merge(new_rates, on=["method", "sigma_map_m", "world", "n"], suffixes=("_old", "_new")).to_csv(
        out_dir / "rates_old_vs_new.csv", index=False)
    diag = geometry_diagnostics(old_classified, geometries, "old").join(
        geometry_diagnostics(new_classified, geometries, "new").filter(like="new_"))
    diag.to_csv(out_dir / "per_geometry_replay.csv")
    print(json.dumps({k: summary[k] for k in ("s_lo", "t_common")}, indent=1, default=str)[:4000])
    print("config_block", json.dumps(summary["config_block"], indent=1))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
