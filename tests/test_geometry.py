from __future__ import annotations

import math

import numpy as np
import pytest

from isac_verify.config import PilotConfig
from isac_verify.geometry import (
    LatencyFloorError,
    Wall,
    app_to_sionna_angles,
    delay_bin_spacing_s,
    direct_round_trip_delay_s,
    fraunhofer_distance_m,
    image_method_reference,
    reflection_geometry,
    replay_added_delay_s,
    require_latency_floor,
    sionna_to_app_angles,
    wall_departure_angle_deg,
    wall_reflection_point,
    wall_visibility,
)


def test_image_method_matches_brief_table(config: PilotConfig) -> None:
    reference = image_method_reference(
        candidate_range_m=float(config.get("geometry_reference", "candidate_range_m")),
        relay_range_m=float(config.get("geometry_reference", "relay_range_m")),
        wall_offset_m=float(config.get("geometry_reference", "wall_offset_m")),
        speed_of_light_m_per_s=config.speed_of_light,
        latency_floor_s=float(config.get("geometry_reference", "latency_floor_ns")) * 1e-9,
    )
    expected_ns = {
        "candidate_direct_round_trip_delay_s": float(
            config.get("geometry_reference", "candidate_direct_round_trip_delay_ns")
        ),
        "relay_added_delay_s": float(
            config.get("geometry_reference", "relay_added_delay_ns")
        ),
        "target_wall_direct_return_delay_s": float(
            config.get("geometry_reference", "target_wall_direct_return_delay_ns")
        ),
        "relay_wall_plus_replay_delay_s": float(
            config.get("geometry_reference", "relay_wall_plus_replay_delay_ns")
        ),
        "wall_delay_difference_s": float(
            config.get("geometry_reference", "wall_delay_difference_ns")
        ),
    }
    for key, expected in expected_ns.items():
        assert f"{reference[key] * 1e9:.6f}" == f"{expected:.6f}", key
    expected_angles = {
        "wall_departure_candidate_deg": float(
            config.get("geometry_reference", "wall_departure_candidate_deg")
        ),
        "wall_departure_relay_deg": float(
            config.get("geometry_reference", "wall_departure_relay_deg")
        ),
    }
    for key, expected in expected_angles.items():
        assert f"{reference[key]:.6f}" == f"{expected:.6f}", key

    wavelength = config.wavelength_m
    fraunhofer = fraunhofer_distance_m(
        int(config.get("arrays", "gnb_tx", "num_elements")),
        float(config.get("arrays", "gnb_tx", "spacing_wavelengths")),
        wavelength,
    )
    assert f"{fraunhofer:.2f}" == "9.64"
    profile = config.profile("baseline")
    bin_spacing = delay_bin_spacing_s(
        int(profile["fft_size"]), float(profile["subcarrier_spacing_hz"])
    )
    assert f"{bin_spacing * 1e9:.6f}" == "8.138021"


def test_reflection_point_and_finite_wall_visibility(config: PilotConfig) -> None:
    wall = Wall.from_mapping(config.section("physical", "walls", "positive"))
    point, length = reflection_geometry(
        config.get("physical", "gnb_position_m"),
        config.get("physical", "candidate_position_m"),
        wall,
    )
    assert f"{point[0]:.6f}" == "30.000000"
    assert f"{point[1]:.6f}" == "15.000000"
    assert f"{point[2]:.6f}" == "2.000000"
    assert wall_visibility(point, wall)
    assert not wall_visibility(
        np.array([200.0, wall.offset_m, 2.0]), wall
    )
    assert not wall_visibility(
        np.array([30.0, wall.offset_m, 12.1]), wall
    )
    expected_outgoing = math.hypot(60.0, 30.0)
    assert length == pytest.approx(expected_outgoing, abs=1e-12)


def test_latency_floor_rejects_short_replay(config: PilotConfig) -> None:
    floor = float(config.get("geometry_reference", "latency_floor_ns")) * 1e-9
    short_delay = replay_added_delay_s(50.0, 49.5, config.speed_of_light)
    assert short_delay < floor
    with pytest.raises(LatencyFloorError):
        require_latency_floor(short_delay, floor)
    long_delay = replay_added_delay_s(60.0, 30.0, config.speed_of_light)
    assert require_latency_floor(long_delay, floor) == long_delay


def test_angle_conversion_round_trip() -> None:
    for azimuth in [0.0, 26.565, 90.0, 180.0, 270.0, 359.0]:
        for elevation in [-60.0, -10.0, 0.0, 26.565, 60.0]:
            sionna_azimuth, zenith = app_to_sionna_angles(azimuth, elevation)
            recovered_azimuth, recovered_elevation = sionna_to_app_angles(
                sionna_azimuth, zenith
            )
            assert recovered_azimuth == pytest.approx(azimuth, abs=1e-12)
            assert recovered_elevation == pytest.approx(elevation, abs=1e-12)
