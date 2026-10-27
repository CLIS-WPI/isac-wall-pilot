from __future__ import annotations

import pytest

from isac_verify.config import PilotConfig
from isac_verify.phase6 import phase6_diagnostics
from isac_verify.verification import classify_evidence, planner_field_names


@pytest.fixture(scope="module")
def phase6_summary(config: PilotConfig, tmp_path_factory) -> dict:
    output = tmp_path_factory.mktemp("phase6-tests")
    return phase6_diagnostics(config, output)


def test_twin_plan_uses_fixed_gates_and_allowed_fields(
    phase6_summary: dict,
) -> None:
    assert phase6_summary["checks"]["planner_excludes_truth"] is True
    assert planner_field_names() == (
        "candidate",
        "twin_positive_wall_m",
        "twin_negative_wall_m",
        "sigma_map_m",
        "sigma_candidate_range_m",
        "gnb_position_m",
        "tx_power_w",
        "robust_num_samples",
        "min_departure_separation_deg",
        "speed_of_light_m_per_s",
    )
    exact = phase6_summary["worlds"]["exact"]
    robust = phase6_summary["worlds"]["robust"]
    assert exact["status"] == "AVAILABLE"
    assert robust["status"] == "AVAILABLE"
    assert exact["plan"]["direct_gate"] == robust["plan"]["direct_gate"]
    assert exact["plan"]["wall_gate"] == robust["plan"]["wall_gate"]
    assert exact["plan"]["beam_phi_deg"] == pytest.approx(
        robust["plan"]["beam_phi_deg"], abs=0.02
    )
    assert exact["plan"]["wall_gate"]["delay_min_ns"] > (
        exact["plan"]["direct_gate"]["delay_max_ns"]
    )


def test_target_and_replay_separate_under_identical_processing(
    phase6_summary: dict,
) -> None:
    for mode in ("exact", "robust"):
        target = phase6_summary["worlds"][mode]["worlds"]["target"]["evidence"]
        replay = phase6_summary["worlds"][mode]["worlds"]["replay"]["evidence"]
        assert target["status"] == "UNRESOLVED"
        assert replay["status"] == "UNRESOLVED"
        assert target["E_D"] == pytest.approx(replay["E_D"], rel=1e-3)
        assert target["E_W"] > 100.0 * replay["E_W"]
        assert target["score"] > 0.5
        assert replay["score"] < 0.02
        assert target["score"] > 50.0 * replay["score"]


@pytest.fixture
def test_only_thresholds() -> dict:
    return {
        "supported_min": 0.5,
        "rejected_max": 0.05,
        "noise_sigmas": 3.0,
    }


def test_blocked_genuine_wall_path_matches_noise(
    phase6_summary: dict, test_only_thresholds: dict,
) -> None:
    blocked = phase6_summary["blocked"]
    target = phase6_summary["worlds"]["exact"]["worlds"]["target"]["evidence"]
    assert blocked["blocked_path_count"] > 0
    assert blocked["status"] == "UNRESOLVED"
    assert blocked["status"] != "SUPPORTED"
    assert blocked["E_W"] - blocked["N_W"] <= 3.0 * blocked["N_W"] + 1e-30
    assert blocked["E_W"] < 0.01 * target["E_W"]
    assert blocked["score"] < 0.02
    assert blocked["score"] < 0.01 * target["score"]
    blocked_status, blocked_reason = classify_evidence(
        score=blocked["score"],
        net_wall_w=blocked["E_W"] - blocked["N_W"],
        wall_noise_w=blocked["N_W"],
        direct_reliable=blocked["direct_reliable"],
        reference_detectable=blocked["reference_detectable"],
        thresholds=test_only_thresholds,
    )
    target_status, _ = classify_evidence(
        score=target["score"],
        net_wall_w=target["E_W"] - target["N_W"],
        wall_noise_w=target["N_W"],
        direct_reliable=target["direct_reliable"],
        reference_detectable=target["reference_detectable"],
        thresholds=test_only_thresholds,
    )
    assert blocked_status == "UNRESOLVED"
    assert blocked_reason == "wall_energy_consistent_with_noise"
    assert blocked_status not in {"SUPPORTED", "ATTACK"}
    assert target_status == "SUPPORTED"

    assert phase6_summary["checks"]["guard_noise_within_bounds"] is True
    assert phase6_summary["checks"]["guard_disjoint_from_gates"] is True
    uncertainty = phase6_summary["candidate_uncertainty"]
    assert uncertainty["used_range_sigma_m"] == pytest.approx(
        max(
            uncertainty["measured_range_sigma_m"],
            uncertainty["range_floor_m"],
        )
    )
