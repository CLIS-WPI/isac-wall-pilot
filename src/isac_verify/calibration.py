"""Calibration-threshold computation for Phase 8A (evaluator side).

These functions implement ``docs/ANALYSIS_PLAN.md`` rules 3-6 on a table of
calibration trials. They use world labels and therefore belong with scoring,
not with planner or decision code. The decision functions that consume the
frozen values live in ``verification.classify_evidence``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .verification import classify_evidence

MAPPED_METHODS = ("exact_oracle", "imperfect_narrow", "imperfect_robust")
MAPFREE_METHODS = ("mapfree_two_beam", "mapfree_coarse_to_fine")
WALL_PROBING_METHODS = MAPPED_METHODS + MAPFREE_METHODS


class CalibrationError(ValueError):
    """Raised when calibration data cannot support a rule."""


def c_tag(factor: float) -> str:
    return "c" + f"{float(factor):g}".replace(".", "p")


def _col(name: str, factor: float) -> str:
    return f"{name}_{c_tag(factor)}"


def floor_k(excess: Sequence[float], false_rate: float = 0.01) -> dict[str, Any]:
    """Rule 3: smallest k with P(excess > k) <= false_rate.

    ``k`` is the order statistic ``x_(ceil((1 - false_rate) n))``.
    """
    values = np.sort(np.asarray(excess, dtype=float))
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count == 0:
        raise CalibrationError("no noise-only exceedance samples")
    index = int(math.ceil((1.0 - float(false_rate)) * count - 1e-12)) - 1
    index = min(max(index, 0), count - 1)
    k = float(values[index])
    realized = float(np.mean(values > k))
    return {
        "value": k,
        "n": count,
        "order_statistic_index_1based": index + 1,
        "false_exceedance_rate_target": float(false_rate),
        "false_exceedance_rate_realized": realized,
    }


def s_hi_value(
    target_s: Sequence[float], quantile: float = 0.995
) -> dict[str, Any]:
    """Rule 4: ``quantile`` of target-world S among floor-passing rows."""
    values = np.asarray(target_s, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise CalibrationError("no floor-passing target-world rows for s_hi")
    return {
        "value": float(np.quantile(values, quantile)),
        "n": int(values.size),
        "quantile": float(quantile),
    }


def s_lo_value(
    eligible_replay_s: Sequence[float],
    replay_rows: int,
    max_rate: float = 0.05,
) -> dict[str, Any]:
    """Rule 4: smallest S such that the replay SUPPORTED rate is <= max_rate.

    ``eligible_replay_s`` holds S for replay rows that already satisfy the
    other SUPPORTED conditions; ``replay_rows`` counts every replay row.
    """
    values = np.sort(np.asarray(eligible_replay_s, dtype=float))
    values = values[np.isfinite(values)]
    allowed = int(math.floor(float(max_rate) * int(replay_rows) + 1e-12))
    chosen: float | None = None
    for candidate in np.unique(values):
        if int(np.sum(values >= candidate)) <= allowed:
            chosen = float(candidate)
            break
    if chosen is None:
        chosen = (
            float(np.nextafter(values.max(), np.inf)) if values.size else 0.0
        )
    return {
        "value": chosen,
        "replay_rows": int(replay_rows),
        "n_allowed": allowed,
        "n_eligible_replay_rows": int(values.size),
        "max_rate": float(max_rate),
    }


def t_common_value(
    replay_excess: Sequence[float], quantile: float = 0.95
) -> dict[str, Any]:
    """Rule 5: replay-world quantile of wall_excess_snr."""
    values = np.asarray(replay_excess, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise CalibrationError("no replay-world wall_excess_snr samples")
    return {
        "value": float(np.quantile(values, quantile)),
        "n": int(values.size),
        "quantile": float(quantile),
    }


def coverage_table(
    trials: pd.DataFrame, inflations: Iterable[float]
) -> pd.DataFrame:
    """Rule 6: target-world gate coverage per method, sigma, and c."""
    rows: list[dict[str, Any]] = []
    target = trials[trials["world"] == "target"]
    for method in MAPPED_METHODS:
        method_rows = target[target["method"] == method]
        for factor in inflations:
            delay = method_rows[_col("cover_delay", factor)]
            aoa = method_rows[_col("cover_aoa", factor)]
            has_gate = delay.notna() & aoa.notna()
            for sigma, group in list(
                method_rows[has_gate].groupby("sigma_map_m")
            ) + [("pooled", method_rows[has_gate])]:
                covered_delay = group[_col("cover_delay", factor)] == 1.0
                covered_aoa = group[_col("cover_aoa", factor)] == 1.0
                rows.append(
                    {
                        "method": method,
                        "sigma_map_m": sigma,
                        "c": float(factor),
                        "n_target_trials_with_gate": int(len(group)),
                        "n_target_trials_total": int(
                            len(method_rows)
                            if sigma == "pooled"
                            else (method_rows["sigma_map_m"] == sigma).sum()
                        ),
                        "coverage_delay": float(covered_delay.mean())
                        if len(group)
                        else float("nan"),
                        "coverage_aoa": float(covered_aoa.mean())
                        if len(group)
                        else float("nan"),
                        "coverage": float(
                            (covered_delay & covered_aoa).mean()
                        )
                        if len(group)
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def select_inflation(
    coverage: pd.DataFrame,
    *,
    method: str,
    required: float,
    candidates: Sequence[float],
) -> dict[str, Any]:
    """Rule 6: smallest c whose pooled coverage for ``method`` meets ``required``."""
    pooled = coverage[
        (coverage["method"] == method)
        & (coverage["sigma_map_m"].astype(str) == "pooled")
    ]
    by_c = {
        float(row["c"]): float(row["coverage"]) for _, row in pooled.iterrows()
    }
    for factor in sorted(float(value) for value in candidates):
        if by_c.get(factor, float("nan")) >= float(required):
            return {
                "value": factor,
                "met": True,
                "required": float(required),
                "method": method,
                "pooled_coverage_by_c": by_c,
            }
    return {
        "value": float(max(candidates)),
        "met": False,
        "required": float(required),
        "method": method,
        "pooled_coverage_by_c": by_c,
    }


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce")


def fit_mapped_method(
    trials: pd.DataFrame,
    method: str,
    factor: float,
    *,
    false_rate: float,
    replay_rate_max: float,
    s_hi_quantile: float,
) -> dict[str, Any]:
    """Rules 3 and 4 for one mapped method at inflation ``factor``."""
    rows = trials[trials["method"] == method]
    excess = _numeric(rows, _col("floor_excess", factor))
    floor = floor_k(excess.dropna().to_numpy(), false_rate)
    k = floor["value"]
    wall_excess = _numeric(rows, _col("wall_excess_snr", factor))
    score = _numeric(rows, _col("S", factor))
    reliable = _numeric(rows, "direct_reliable") == 1.0
    detectable = _numeric(rows, _col("ref_det", factor)) == 1.0
    floor_pass = wall_excess > k
    target = rows["world"] == "target"
    replay = rows["world"] == "replay"
    hi = s_hi_value(
        score[target & floor_pass & score.notna()].to_numpy(), s_hi_quantile
    )
    s_hi = hi["value"]
    eligible = (
        replay
        & reliable
        & detectable
        & floor_pass
        & score.notna()
        & (score <= s_hi)
    )
    lo = s_lo_value(
        score[eligible].to_numpy(), int(replay.sum()), replay_rate_max
    )
    return {
        "floor_k": floor,
        "s_hi": hi,
        "s_lo": lo,
    }


def fit_thresholds(
    trials: pd.DataFrame,
    factor: float,
    *,
    false_rate: float = 0.01,
    replay_rate_max: float = 0.05,
    s_hi_quantile: float = 0.995,
    common_quantile: float = 0.95,
) -> dict[str, Any]:
    """Compute rules 3-5 for every method at inflation ``factor``."""
    mapped = {
        method: fit_mapped_method(
            trials,
            method,
            factor,
            false_rate=false_rate,
            replay_rate_max=replay_rate_max,
            s_hi_quantile=s_hi_quantile,
        )
        for method in MAPPED_METHODS
    }
    common: dict[str, Any] = {}
    for method in WALL_PROBING_METHODS:
        rows = trials[
            (trials["method"] == method) & (trials["world"] == "replay")
        ]
        name = (
            _col("wall_excess_snr", factor)
            if method in MAPPED_METHODS
            else "wall_excess_snr"
        )
        common[method] = t_common_value(
            _numeric(rows, name).to_numpy(), common_quantile
        )
    return {"mapped": mapped, "common": common}


def thresholds_for_decision(fitted: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Frozen per-method decision thresholds in ``classify_evidence`` form."""
    return {
        method: {
            "noise_sigmas": values["floor_k"]["value"],
            "supported_min": values["s_lo"]["value"],
            "supported_max": values["s_hi"]["value"],
        }
        for method, values in fitted["mapped"].items()
    }


def classify_trials(
    trials: pd.DataFrame,
    fitted: dict[str, Any],
    factor: float,
) -> pd.DataFrame:
    """Add ``supported`` and ``supported_common`` boolean columns.

    Mapped-method SUPPORTED uses ``verification.classify_evidence`` with the
    per-method frozen thresholds. SUPPORTED_common uses ``wall_excess_snr >=
    t_m`` for every wall-probing method. Rows that are ACQUISITION_FAILED are
    never supported.
    """
    decision = thresholds_for_decision(fitted)
    frame = trials.copy()
    supported = np.zeros(len(frame), dtype=bool)
    common = np.zeros(len(frame), dtype=bool)
    labels: list[str] = ["UNRESOLVED"] * len(frame)
    reasons: list[str] = [""] * len(frame)
    for position, (_, row) in enumerate(frame.iterrows()):
        method = row["method"]
        failed = row["status"] == "ACQUISITION_FAILED"
        if failed:
            labels[position] = "ACQUISITION_FAILED"
            reasons[position] = "direct_acquisition_failed"
            continue
        if method in decision:
            score = row[_col("S", factor)]
            wall_noise = row[_col("N_W", factor)]
            net = row[_col("E_W", factor)] - wall_noise
            if pd.notna(score) and pd.notna(wall_noise):
                label, reason = classify_evidence(
                    score=float(score),
                    net_wall_w=float(net),
                    wall_noise_w=float(wall_noise),
                    direct_reliable=bool(row["direct_reliable"] == 1.0)
                    and bool(math.isfinite(float(score))),
                    reference_detectable=bool(
                        row[_col("ref_det", factor)] == 1.0
                    ),
                    thresholds=decision[method],
                )
                labels[position] = label
                reasons[position] = reason
                supported[position] = label == "SUPPORTED"
            else:
                reasons[position] = "twin_path_not_observable"
        if method in fitted["common"]:
            name = (
                _col("wall_excess_snr", factor)
                if method in MAPPED_METHODS
                else "wall_excess_snr"
            )
            value = row[name]
            common[position] = bool(
                pd.notna(value)
                and float(value) >= fitted["common"][method]["value"]
            )
    frame["decision"] = labels
    frame["decision_reason"] = reasons
    frame["supported"] = supported
    frame["supported_common"] = common
    return frame


def rate_table(classified: pd.DataFrame) -> pd.DataFrame:
    """SUPPORTED and SUPPORTED_common rates per method x sigma x world."""
    rows: list[dict[str, Any]] = []
    for (method, sigma, world), group in classified.groupby(
        ["method", "sigma_map_m", "world"], sort=False
    ):
        rows.append(
            {
                "method": method,
                "sigma_map_m": float(sigma),
                "world": world,
                "n": int(len(group)),
                "supported_rate": float(group["supported"].mean())
                if method in MAPPED_METHODS
                else float("nan"),
                "supported_common_rate": float(
                    group["supported_common"].mean()
                )
                if method in WALL_PROBING_METHODS
                else float("nan"),
                "acquisition_failed": int(
                    (group["decision"] == "ACQUISITION_FAILED").sum()
                ),
            }
        )
    return pd.DataFrame(rows)
