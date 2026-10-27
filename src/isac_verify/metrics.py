"""Aggregation, paired summaries, and integrity checks for Phase 7."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

METHOD_ORDER = (
    "direct_only",
    "exact_oracle",
    "imperfect_narrow",
    "imperfect_robust",
    "mapfree_two_beam",
    "mapfree_coarse_to_fine",
)

METHOD_LABELS = {
    "direct_only": "Direct only",
    "exact_oracle": "Exact-map oracle",
    "imperfect_narrow": "Imperfect narrow",
    "imperfect_robust": "Imperfect robust",
    "mapfree_two_beam": "Map-free two-beam",
    "mapfree_coarse_to_fine": "Map-free coarse-to-fine",
}

def expected_trial_count(
    methods: Iterable[str], sigmas: Iterable[float], worlds: Iterable[str], records: int
) -> int:
    return len(list(methods)) * len(list(sigmas)) * len(list(worlds)) * int(records)


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Return one row per method, sigma, and world."""
    rows: list[dict[str, Any]] = []
    group_columns = ["method", "sigma_map_m", "world"]
    for keys, group in trials.groupby(group_columns, sort=False):
        method, sigma, world = keys
        s_available = group["S_available"].astype(bool)
        excess_available = group["wall_excess_snr_available"].astype(bool)
        s_values = pd.to_numeric(
            group.loc[s_available, "S"], errors="coerce"
        )
        excess_values = pd.to_numeric(
            group.loc[excess_available, "wall_excess_snr"], errors="coerce"
        )
        null = pd.to_numeric(group["null_depth_db"], errors="coerce")
        aoa_error = pd.to_numeric(
            group["candidate_aoa_error_deg"], errors="coerce"
        )
        runtime = pd.to_numeric(group["trial_runtime_s"], errors="coerce")
        beam_attempted = (
            ~group["method"].eq("direct_only")
            & ~group["path_availability"].eq("NOT_ATTEMPTED")
        )
        rows.append(
            {
                "method": method,
                "sigma_map_m": float(sigma),
                "world": world,
                "records": int(len(group)),
                "acquisition_rate": float(
                    np.mean(group["acquisition_status"].eq("ACQUIRED"))
                ),
                "acquisition_failure_rate": float(
                    np.mean(group["status"].eq("ACQUISITION_FAILED"))
                ),
                "not_observable_rate": float(
                    np.mean(group["path_availability"].eq("NOT_OBSERVABLE"))
                ),
                "not_attempted_rate": float(
                    np.mean(group["path_availability"].eq("NOT_ATTEMPTED"))
                ),
                "not_applicable_no_twin_rate": float(
                    np.mean(
                        group["path_availability"].eq("NOT_APPLICABLE_NO_TWIN")
                    )
                ),
                "available_path_rate": float(
                    np.mean(group["path_availability"].eq("AVAILABLE"))
                ),
                "unusable_beam_rate": (
                    float(
                        np.mean(
                            ~group.loc[beam_attempted, "beam_usable"].astype(bool)
                        )
                    )
                    if beam_attempted.any()
                    else None
                ),
                "S_available_count": int(s_available.sum()),
                "S_available_rate": float(np.mean(s_available)),
                "S_median": float(s_values.median()) if not s_values.empty else None,
                "S_q05": float(s_values.quantile(0.05)) if not s_values.empty else None,
                "S_q95": float(s_values.quantile(0.95)) if not s_values.empty else None,
                "wall_excess_snr_available_count": int(excess_available.sum()),
                "wall_excess_snr_available_rate": float(np.mean(excess_available)),
                "wall_excess_snr_median": (
                    float(excess_values.median()) if not excess_values.empty else None
                ),
                "wall_excess_snr_q05": (
                    float(excess_values.quantile(0.05))
                    if not excess_values.empty
                    else None
                ),
                "wall_excess_snr_q95": (
                    float(excess_values.quantile(0.95))
                    if not excess_values.empty
                    else None
                ),
                "candidate_aoa_error_deg_median": (
                    float(aoa_error.median())
                    if aoa_error.notna().any()
                    else None
                ),
                "candidate_aoa_error_deg_q05": (
                    float(aoa_error.quantile(0.05))
                    if aoa_error.notna().any()
                    else None
                ),
                "candidate_aoa_error_deg_q95": (
                    float(aoa_error.quantile(0.95))
                    if aoa_error.notna().any()
                    else None
                ),
                "null_depth_db_median": (
                    float(null.median()) if null.notna().any() else None
                ),
                "null_depth_db_q05": (
                    float(null.quantile(0.05))
                    if null.notna().any()
                    else None
                ),
                "null_depth_db_q95": (
                    float(null.quantile(0.95))
                    if null.notna().any()
                    else None
                ),
                "runtime_s_median": (
                    float(runtime.median()) if runtime.notna().any() else None
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["_method_order"] = frame["method"].map(
            {name: index for index, name in enumerate(METHOD_ORDER)}
        )
        frame = frame.sort_values(
            ["_method_order", "world", "sigma_map_m"]
        ).drop(columns="_method_order")
    return frame.reset_index(drop=True)


def integrity_checks(
    trials: pd.DataFrame,
    *,
    expected_rows: int,
    methods: Iterable[str],
    sigmas: Iterable[float],
    worlds: Iterable[str],
    records: int,
) -> dict[str, Any]:
    methods = list(methods)
    sigmas = [float(value) for value in sigmas]
    worlds = list(worlds)
    cells = trials.groupby(["method", "sigma_map_m", "world"]).size()
    expected_cells = {
        (method, sigma, world)
        for method in methods
        for sigma in sigmas
        for world in worlds
    }
    actual_cells = {tuple(key) for key in cells.index}
    s_available = trials["S_available"].astype(bool)
    s_values = pd.to_numeric(trials["S"], errors="coerce")
    s_valid = bool(
        np.isfinite(s_values[s_available]).all()
        and s_values[~s_available].isna().all()
    )
    excess_available = trials["wall_excess_snr_available"].astype(bool)
    excess_values = pd.to_numeric(trials["wall_excess_snr"], errors="coerce")
    excess_valid = bool(
        np.isfinite(excess_values[excess_available]).all()
        and excess_values[~excess_available].isna().all()
    )
    paired = trials.groupby(["world", "sigma_map_m", "record_index"])[
        ["probe_seed", "receiver_noise_seed", "relay_noise_seed"]
    ].nunique()
    paired_ok = bool((paired <= 1).all().all())
    statuses = set(trials["status"].astype(str))
    direct_rows = trials[trials["method"] == "direct_only"]
    mapfree_rows = trials[
        trials["method"].isin(["mapfree_two_beam", "mapfree_coarse_to_fine"])
    ]
    mapped_rows = trials[
        ~trials["method"].isin(
            ["direct_only", "mapfree_two_beam", "mapfree_coarse_to_fine"]
        )
    ]
    return {
        "expected_rows": int(expected_rows),
        "actual_rows": int(len(trials)),
        "row_count_matches": int(len(trials)) == int(expected_rows),
        "expected_cells": len(expected_cells),
        "actual_cells": len(actual_cells),
        "cells_complete": expected_cells == actual_cells,
        "cell_counts": {
            "minimum": int(cells.min()) if not cells.empty else 0,
            "maximum": int(cells.max()) if not cells.empty else 0,
            "expected_per_cell": int(records),
        },
        "S_availability_matches_finiteness": s_valid,
        "S_finite_where_available": int(
            np.isfinite(s_values[s_available]).sum()
        ),
        "S_nan_where_unavailable": int(s_values[~s_available].isna().sum()),
        "wall_excess_snr_availability_matches_finiteness": excess_valid,
        "wall_excess_snr_finite_where_available": int(
            np.isfinite(excess_values[excess_available]).sum()
        ),
        "wall_excess_snr_nan_where_unavailable": int(
            excess_values[~excess_available].isna().sum()
        ),
        "paired_seed_agreement": paired_ok,
        "paired_seed_groups_checked": int(len(paired)),
        "status_labels_allowed": bool(
            statuses <= {"SUPPORTED", "UNRESOLVED", "ACQUISITION_FAILED"}
        ),
        "direct_only_path_not_attempted": bool(
            direct_rows["path_availability"].eq("NOT_ATTEMPTED").all()
        ),
        "mapfree_path_not_applicable": bool(
            mapfree_rows["path_availability"].eq("NOT_APPLICABLE_NO_TWIN").all()
        ),
        "not_observable_only_from_twin": bool(
            not trials.loc[
                ~trials.index.isin(mapped_rows.index), "path_availability"
            ].eq("NOT_OBSERVABLE").any()
        ),
        "acquisition_failure_rows": int(
            trials["status"].eq("ACQUISITION_FAILED").sum()
        ),
        "S_available_by_method": {
            str(method): bool(group["S_available"].any())
            for method, group in trials.groupby("method", sort=False)
        },
        "wall_excess_snr_available_by_method": {
            str(method): bool(group["wall_excess_snr_available"].any())
            for method, group in trials.groupby("method", sort=False)
        },
    }
