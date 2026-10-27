from __future__ import annotations

import numpy as np
import pandas as pd

from isac_verify.calibration import (
    MAPFREE_METHODS,
    MAPPED_METHODS,
    classify_trials,
    coverage_table,
    fit_thresholds,
    floor_k,
    rate_table,
    s_hi_value,
    s_lo_value,
    select_inflation,
    t_common_value,
)


def _synthetic_trials(seed: int = 0, per_cell: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for method in MAPPED_METHODS + MAPFREE_METHODS + ("direct_only",):
        for sigma in (0.0, 0.5):
            for world in ("target", "replay"):
                for _ in range(per_cell):
                    target = world == "target"
                    noise = 5.0
                    excess = (
                        float(rng.normal(900.0, 50.0))
                        if target
                        else float(rng.normal(0.0, 0.4))
                    )
                    score = (
                        float(rng.normal(0.9, 0.05))
                        if target
                        else float(abs(rng.normal(0.0, 0.002)))
                    )
                    row = {
                        "method": method,
                        "world": world,
                        "sigma_map_m": sigma,
                        "status": "UNRESOLVED",
                        "direct_reliable": 1.0,
                        "wall_excess_snr": excess,
                    }
                    for tag in ("c1", "c1p25"):
                        row[f"E_W_{tag}"] = noise * (1.0 + excess)
                        row[f"N_W_{tag}"] = noise
                        row[f"S_{tag}"] = score
                        row[f"ref_det_{tag}"] = 1.0
                        row[f"wall_excess_snr_{tag}"] = excess
                        row[f"floor_excess_{tag}"] = float(rng.normal(0.0, 0.3))
                        covered = rng.random() < (0.9 if tag == "c1" else 0.998)
                        row[f"cover_delay_{tag}"] = float(covered)
                        row[f"cover_aoa_{tag}"] = 1.0
                    rows.append(row)
    return pd.DataFrame(rows)


def test_floor_k_is_smallest_value_meeting_false_rate() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 5000)
    result = floor_k(x, 0.01)
    assert result["false_exceedance_rate_realized"] <= 0.01
    ordered = np.sort(x)
    below = ordered[result["order_statistic_index_1based"] - 2]
    assert np.mean(x > below) > 0.01


def test_s_hi_is_the_requested_quantile() -> None:
    values = np.linspace(0.0, 1.0, 1001)
    assert abs(s_hi_value(values, 0.995)["value"] - 0.995) < 1e-12


def test_s_lo_is_smallest_value_with_replay_rate_at_most_five_percent() -> None:
    rng = np.random.default_rng(2)
    replay = np.abs(rng.normal(0.0, 1.0, 400))
    result = s_lo_value(replay, replay_rows=400, max_rate=0.05)
    count = int(np.sum(replay >= result["value"]))
    assert count / 400 <= 0.05
    smaller = replay[replay < result["value"]].max()
    assert np.sum(replay >= smaller) / 400 > 0.05


def test_s_lo_counts_ineligible_replay_rows_in_the_denominator() -> None:
    eligible = np.array([1.0, 2.0, 3.0])
    result = s_lo_value(eligible, replay_rows=100, max_rate=0.05)
    assert result["n_allowed"] == 5
    assert result["value"] == 1.0


def test_s_lo_exceeds_max_when_no_rows_are_allowed() -> None:
    eligible = np.array([1.0, 2.0, 3.0])
    result = s_lo_value(eligible, replay_rows=10, max_rate=0.05)
    assert result["n_allowed"] == 0
    assert result["value"] > 3.0


def test_t_common_is_replay_quantile() -> None:
    values = np.linspace(-1.0, 1.0, 2001)
    result = t_common_value(values, 0.95)
    assert abs(result["value"] - 0.9) < 1e-9


def test_inflation_selects_smallest_candidate_meeting_coverage() -> None:
    trials = _synthetic_trials()
    table = coverage_table(trials, (1.0, 1.25))
    choice = select_inflation(
        table,
        method="imperfect_robust",
        required=0.99,
        candidates=(1.0, 1.25),
    )
    assert choice["met"] is True
    assert choice["value"] == 1.25
    keep = select_inflation(
        table,
        method="imperfect_robust",
        required=0.5,
        candidates=(1.0, 1.25),
    )
    assert keep["value"] == 1.0


def test_inflation_reports_failure_when_unmet() -> None:
    trials = _synthetic_trials()
    table = coverage_table(trials, (1.0, 1.25))
    choice = select_inflation(
        table,
        method="imperfect_robust",
        required=1.01,
        candidates=(1.0, 1.25),
    )
    assert choice["met"] is False
    assert choice["value"] == 1.25


def test_frozen_rules_hold_on_the_calibration_sample() -> None:
    trials = _synthetic_trials()
    fitted = fit_thresholds(trials, 1.0)
    classified = classify_trials(trials, fitted, 1.0)
    rates = rate_table(classified)
    for method in MAPPED_METHODS:
        replay = classified[
            (classified["method"] == method) & (classified["world"] == "replay")
        ]
        assert replay["supported"].mean() <= 0.05
        target = classified[
            (classified["method"] == method) & (classified["world"] == "target")
        ]
        assert target["supported"].mean() > 0.9
    for method in MAPPED_METHODS + MAPFREE_METHODS:
        replay = classified[
            (classified["method"] == method) & (classified["world"] == "replay")
        ]
        assert 0.04 <= replay["supported_common"].mean() <= 0.06
    direct = classified[classified["method"] == "direct_only"]
    assert not direct["supported_common"].any()
    assert set(rates["method"]) == set(trials["method"])


def test_acquisition_failures_are_never_supported() -> None:
    trials = _synthetic_trials()
    fitted = fit_thresholds(trials, 1.0)
    failed = trials.copy()
    failed["status"] = "ACQUISITION_FAILED"
    classified = classify_trials(failed, fitted, 1.0)
    assert not classified["supported"].any()
    assert not classified["supported_common"].any()
    assert set(classified["decision"]) == {"ACQUISITION_FAILED"}
