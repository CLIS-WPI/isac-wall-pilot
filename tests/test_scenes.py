from __future__ import annotations

import math

import pytest

from isac_verify.config import PilotConfig
from isac_verify.material import build_material_reference, build_wall_thickness_stress
from isac_verify.wall_paths import (
    blocked_scene_diagnostics,
    find_reciprocal_wall_path,
    nominal_path_diagnostics,
    twin_diagnostics,
)


@pytest.fixture(scope="module")
def nominal(config: PilotConfig):
    return nominal_path_diagnostics(config)


@pytest.fixture(scope="module")
def twin(config: PilotConfig):
    return twin_diagnostics(config)


@pytest.fixture(scope="module")
def blocked(config: PilotConfig):
    return blocked_scene_diagnostics(config)


def test_finite_wall_reflection_matches_image_method(
    config: PilotConfig, nominal
) -> None:
    diagnostics, records = nominal
    core = diagnostics["core_paths"]
    assert set(core) == {
        "direct",
        "outgoing_wall_positive_wall",
        "outgoing_wall_negative_wall",
        "reciprocal_wall_positive_wall",
        "reciprocal_wall_negative_wall",
    }
    direct = core["direct"][0]
    expected_direct_ns = (
        2.0 * float(config.get("physical", "candidate_position_m")[0])
        / config.speed_of_light
        * 1e9
    )
    assert direct["delay_ns"] == pytest.approx(expected_direct_ns, abs=1e-3)
    assert direct["aod_deg"] == pytest.approx(0.0, abs=1e-6)
    assert direct["aoa_deg"] == pytest.approx(0.0, abs=1e-6)

    for comparison in diagnostics["outgoing_wall_comparisons"]:
        side = comparison["wall"]
        measured = comparison["measured"]
        expected = comparison["image_method"]
        assert measured["delay_ns"] == pytest.approx(
            expected["delay_ns"], abs=1e-3
        )
        assert measured["aod_deg"] == pytest.approx(
            expected["aod_deg"], abs=1e-2
        )
        assert measured["aoa_deg"] == pytest.approx(0.0, abs=1e-6)
        assert comparison["reflection_point_error_m"] <= 1e-3
        assert abs(comparison["amplitude_relative_error"]) <= 1e-3
        assert measured["vertex_points_m"][0] == pytest.approx(
            expected["reflection_point_m"], abs=1e-3
        )
        reciprocal = find_reciprocal_wall_path(records, side)
        assert reciprocal["delay_ns"] == pytest.approx(
            measured["delay_ns"], abs=1e-6
        )
        assert reciprocal["aod_deg"] == pytest.approx(0.0, abs=1e-6)
        assert reciprocal["aoa_deg"] == pytest.approx(
            expected["aod_deg"], abs=1e-2
        )
        assert reciprocal["amplitude"] == pytest.approx(
            measured["amplitude"], rel=1e-4
        )


def test_wall_mesh_normals_and_finite_extent(nominal) -> None:
    diagnostics, _ = nominal
    walls = diagnostics["true_wall_offsets_m"]
    assert walls == {"positive": 15.0, "negative": -15.0}
    assert diagnostics["validity"]["inner_half_width_m"] == pytest.approx(15.0)
    for side, expected_offset in walls.items():
        geometry = diagnostics["wall_geometry"][side]
        assert geometry["offset_m"] == expected_offset
        assert geometry["normal_alignment_min"] == pytest.approx(1.0)
        assert geometry["planarity_y_spread_m"] <= 1e-5
        assert geometry["bbox_min_m"][0] == pytest.approx(-10.0, abs=1e-3)
        assert geometry["bbox_max_m"][0] == pytest.approx(130.0, abs=1e-3)
        assert geometry["bbox_min_m"][2] == pytest.approx(0.0, abs=1e-6)
        assert geometry["bbox_max_m"][2] == pytest.approx(12.0, abs=1e-6)
        assert geometry["bbox_min_m"][1] == pytest.approx(expected_offset, abs=1e-6)
        assert geometry["bbox_max_m"][1] == pytest.approx(expected_offset, abs=1e-6)
        assert geometry["width_m"] == pytest.approx(140.0, abs=1e-3)
        assert geometry["height_m"] == pytest.approx(12.0, abs=1e-3)
        assert geometry["surface_area_m2"] == pytest.approx(1680.0, abs=1e-2)
    for record in diagnostics["outgoing_wall_comparisons"]:
        side = record["wall"].split("_")[0]
        reflection = record["measured"]["vertex_points_m"][0]
        assert reflection[1] == pytest.approx(walls[side], abs=1e-6)


def test_twin_translates_wall_rigidly(twin) -> None:
    assert twin["epsilon_plus_m"] == pytest.approx(0.5)
    assert twin["epsilon_minus_m"] == pytest.approx(-0.5)
    assert twin["wall_offsets_m"] == {
        "positive": 15.5,
        "negative": -15.5,
    }
    for side, expected_offset in twin["wall_offsets_m"].items():
        geometry = twin["wall_geometry"][side]
        assert geometry["offset_m"] == expected_offset
        assert geometry["bbox_min_m"][1] == pytest.approx(expected_offset, abs=1e-6)
        assert geometry["bbox_max_m"][1] == pytest.approx(expected_offset, abs=1e-6)
        assert geometry["normal_alignment_min"] == pytest.approx(1.0)
        assert geometry["planarity_y_spread_m"] <= 1e-5
        assert geometry["width_m"] == pytest.approx(140.0, abs=1e-3)
        assert geometry["height_m"] == pytest.approx(12.0, abs=1e-3)
    for row in twin["rows"]:
        assert row["image_delay_shift_ns"] > 0.0
        assert row["rt_delay_shift_ns"] == pytest.approx(
            row["image_delay_shift_ns"], abs=1e-3
        )
        assert row["rt_aod_shift_deg"] == pytest.approx(
            row["image_aod_shift_deg"], abs=1e-2
        )
        assert row["image_twin_vs_rt_twin_delay_ns"] == pytest.approx(
            0.0, abs=1e-3
        )
        assert row["image_twin_vs_rt_twin_aod_deg"] == pytest.approx(
            0.0, abs=1e-2
        )


def test_absorber_blocker_removes_positive_wall_path(
    config: PilotConfig, blocked
) -> None:
    assert blocked["blocker_scene_metadata"]["center_m"] == pytest.approx(
        [45.0, 7.5, 2.0]
    )
    assert blocked["positive_outgoing_present"] is False
    assert blocked["positive_reciprocal_present"] is False
    assert blocked["negative_outgoing_present"] is True
    assert set(blocked["removed_families"]) == {
        "outgoing_wall_positive_wall",
        "reciprocal_wall_positive_wall",
    }
    assert blocked["blocked_path_count"] < blocked["unblocked_path_count"]
    assert blocked["new_paths"] == []
    assert any(
        record["family"] == "direct" for record in blocked["blocked_records"]
    )


def test_d8_wall_thickness_stress_is_labeled_follow_up(
    config: PilotConfig,
) -> None:
    reference = build_wall_thickness_stress(config)
    assert reference["label"].startswith("follow-up wall-thickness stress")
    assert reference["baseline_thickness_m"] == pytest.approx(
        float(config.get("wall", "thickness_m"))
    )
    assert reference["thicknesses_m"] == [0.05, 0.10, 0.20]
    assert reference["angles_deg"] == [63.435, 45.0]
    assert len(reference["entries"]) == 6
    baseline = {
        (entry["thickness_m"], entry["incidence_angle_deg"]): entry
        for entry in reference["entries"]
    }[(0.10, 63.435)]
    material = build_material_reference(config)["coefficients"][-1]
    assert baseline["r_te"]["real"] == pytest.approx(
        material["r_te"]["real"], abs=1e-12
    )
    assert baseline["r_tm"]["imag"] == pytest.approx(
        material["r_tm"]["imag"], abs=1e-12
    )
    for entry in reference["entries"]:
        assert math.isfinite(entry["r_te"]["magnitude"])
        assert math.isfinite(entry["r_tm"]["magnitude"])
