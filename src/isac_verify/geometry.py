"""Image-method geometry and application/Sionna angle conventions.

Application frame: x down the street, y lateral, z height, azimuth from +x
toward +y, elevation from the horizontal plane toward +z.

Sionna frame: azimuth from +x toward +y, zenith from +z toward the horizontal
plane.  The conversion is therefore ``zenith = 90 deg - elevation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin, sqrt
from typing import Iterable, Sequence

import numpy as np


class GeometryError(ValueError):
    """Raised when a geometry request is outside the declared model."""


class LatencyFloorError(GeometryError):
    """Raised when a replay delay is below the configured causal floor."""


@dataclass(frozen=True)
class Wall:
    offset_m: float
    inward_normal: tuple[float, float, float]
    x_min_m: float
    x_max_m: float
    z_min_m: float
    z_max_m: float

    @classmethod
    def from_mapping(cls, values: dict) -> "Wall":
        return cls(
            offset_m=float(values["offset_m"]),
            inward_normal=tuple(float(v) for v in values["inward_normal"]),
            x_min_m=float(values["x_min_m"]),
            x_max_m=float(values["x_max_m"]),
            z_min_m=float(values["z_min_m"]),
            z_max_m=float(values["z_max_m"]),
        )

    def contains(self, point: Sequence[float]) -> bool:
        x, y, z = (float(v) for v in point)
        return (
            abs(y - self.offset_m) <= 1e-9
            and self.x_min_m <= x <= self.x_max_m
            and self.z_min_m <= z <= self.z_max_m
        )


def _canonical_azimuth_deg(azimuth_deg: float) -> float:
    return float(azimuth_deg) % 360.0


def app_to_sionna_angles(
    azimuth_deg: float, elevation_deg: float
) -> tuple[float, float]:
    """Convert application azimuth/elevation to Sionna azimuth/zenith."""
    if not -90.0 <= elevation_deg <= 90.0:
        raise GeometryError(f"elevation {elevation_deg} deg is outside [-90, 90]")
    return _canonical_azimuth_deg(azimuth_deg), 90.0 - float(elevation_deg)


def sionna_to_app_angles(
    azimuth_deg: float, zenith_deg: float
) -> tuple[float, float]:
    """Convert Sionna azimuth/zenith to application azimuth/elevation."""
    if not 0.0 <= zenith_deg <= 180.0:
        raise GeometryError(f"zenith {zenith_deg} deg is outside [0, 180]")
    return _canonical_azimuth_deg(azimuth_deg), 90.0 - float(zenith_deg)


def app_direction_vector(
    azimuth_deg: float, elevation_deg: float
) -> np.ndarray:
    azimuth = radians(azimuth_deg)
    elevation = radians(elevation_deg)
    return np.array(
        [
            cos(elevation) * cos(azimuth),
            cos(elevation) * sin(azimuth),
            sin(elevation),
        ],
        dtype=np.float64,
    )


def sionna_direction_vector(
    azimuth_deg: float, zenith_deg: float
) -> np.ndarray:
    azimuth = radians(azimuth_deg)
    zenith = radians(zenith_deg)
    return np.array(
        [
            sin(zenith) * cos(azimuth),
            sin(zenith) * sin(azimuth),
            cos(zenith),
        ],
        dtype=np.float64,
    )


def wall_departure_angle_deg(radius_m: float, wall_offset_m: float) -> float:
    if radius_m <= 0.0:
        raise GeometryError("radius must be positive")
    return degrees(atan2(2.0 * wall_offset_m, radius_m))


def wall_departure_angle_rad(radius_m: float, wall_offset_m: float) -> float:
    if radius_m <= 0.0:
        raise GeometryError("radius must be positive")
    return atan2(2.0 * wall_offset_m, radius_m)


def wall_reflection_point(
    tx_position_m: Sequence[float],
    rx_position_m: Sequence[float],
    wall_y_m: float,
) -> np.ndarray:
    tx = np.asarray(tx_position_m, dtype=np.float64)
    rx = np.asarray(rx_position_m, dtype=np.float64)
    denominator = tx[1] + rx[1] - 2.0 * wall_y_m
    if abs(denominator) <= 1e-15:
        raise GeometryError("transmitter and receiver are mirror-symmetric on the wall")
    t = (tx[1] - wall_y_m) / denominator
    if not 0.0 < t < 1.0:
        raise GeometryError(f"specular intersection parameter {t} is outside (0, 1)")
    point = tx + t * (rx - tx)
    point[1] = wall_y_m
    return point


def distance_m(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)))


def wall_path_length_m(
    tx_position_m: Sequence[float],
    rx_position_m: Sequence[float],
    wall_y_m: float,
) -> float:
    point = wall_reflection_point(tx_position_m, rx_position_m, wall_y_m)
    return distance_m(tx_position_m, point) + distance_m(point, rx_position_m)


def direct_round_trip_delay_s(
    candidate_range_m: float, speed_of_light_m_per_s: float
) -> float:
    if candidate_range_m < 0.0:
        raise GeometryError("candidate range must be non-negative")
    return 2.0 * candidate_range_m / speed_of_light_m_per_s


def replay_added_delay_s(
    candidate_range_m: float,
    relay_range_m: float,
    speed_of_light_m_per_s: float,
) -> float:
    if relay_range_m < 0.0:
        raise GeometryError("relay range must be non-negative")
    if candidate_range_m < relay_range_m:
        raise GeometryError("relay must be no farther than the candidate")
    return 2.0 * (candidate_range_m - relay_range_m) / speed_of_light_m_per_s


def wall_delay_s(
    candidate_range_m: float,
    wall_offset_m: float,
    speed_of_light_m_per_s: float,
) -> float:
    length = hypot(candidate_range_m, 2.0 * wall_offset_m) + candidate_range_m
    return length / speed_of_light_m_per_s


def relay_wall_total_delay_s(
    relay_range_m: float,
    wall_offset_m: float,
    added_delay_s: float,
    speed_of_light_m_per_s: float,
) -> float:
    length = hypot(relay_range_m, 2.0 * wall_offset_m) + relay_range_m
    return length / speed_of_light_m_per_s + added_delay_s


def wall_delay_difference_s(
    candidate_range_m: float,
    relay_range_m: float,
    wall_offset_m: float,
    added_delay_s: float,
    speed_of_light_m_per_s: float,
) -> float:
    candidate = wall_delay_s(
        candidate_range_m, wall_offset_m, speed_of_light_m_per_s
    )
    relay = relay_wall_total_delay_s(
        relay_range_m, wall_offset_m, added_delay_s, speed_of_light_m_per_s
    )
    return relay - candidate


def require_latency_floor(
    added_delay_s: float, latency_floor_s: float
) -> float:
    if added_delay_s < latency_floor_s:
        raise LatencyFloorError(
            f"added delay {added_delay_s * 1e9:.6f} ns is below "
            f"the {latency_floor_s * 1e9:.6f} ns floor"
        )
    return added_delay_s


def latency_floor_ok(added_delay_s: float, latency_floor_s: float) -> bool:
    return added_delay_s >= latency_floor_s


def fraunhofer_distance_m(
    num_elements: int,
    spacing_wavelengths: float,
    wavelength_m: float,
) -> float:
    if num_elements < 1:
        raise GeometryError("array must have at least one element")
    aperture = (num_elements - 1) * spacing_wavelengths * wavelength_m
    return 2.0 * aperture**2 / wavelength_m


def delay_bin_spacing_s(fft_size: int, subcarrier_spacing_hz: float) -> float:
    if fft_size <= 0 or subcarrier_spacing_hz <= 0.0:
        raise GeometryError("fft size and subcarrier spacing must be positive")
    return 1.0 / (fft_size * subcarrier_spacing_hz)


def wall_visibility(
    point_m: Sequence[float], wall: Wall
) -> bool:
    return wall.contains(point_m)


def reflection_geometry(
    tx_position_m: Sequence[float],
    rx_position_m: Sequence[float],
    wall: Wall,
) -> tuple[np.ndarray, float]:
    point = wall_reflection_point(tx_position_m, rx_position_m, wall.offset_m)
    if not wall_visibility(point, wall):
        raise GeometryError(f"reflection point {point.tolist()} is outside the finite wall")
    length = distance_m(tx_position_m, point) + distance_m(point, rx_position_m)
    return point, length


def image_method_reference(
    candidate_range_m: float,
    relay_range_m: float,
    wall_offset_m: float,
    speed_of_light_m_per_s: float,
    latency_floor_s: float,
) -> dict[str, float]:
    added = replay_added_delay_s(
        candidate_range_m, relay_range_m, speed_of_light_m_per_s
    )
    require_latency_floor(added, latency_floor_s)
    candidate_direct = direct_round_trip_delay_s(
        candidate_range_m, speed_of_light_m_per_s
    )
    candidate_wall = wall_delay_s(
        candidate_range_m, wall_offset_m, speed_of_light_m_per_s
    )
    relay_wall = relay_wall_total_delay_s(
        relay_range_m, wall_offset_m, added, speed_of_light_m_per_s
    )
    return {
        "wall_departure_candidate_deg": wall_departure_angle_deg(
            candidate_range_m, wall_offset_m
        ),
        "wall_departure_relay_deg": wall_departure_angle_deg(
            relay_range_m, wall_offset_m
        ),
        "candidate_direct_round_trip_delay_s": candidate_direct,
        "relay_added_delay_s": added,
        "target_wall_direct_return_delay_s": candidate_wall,
        "relay_wall_plus_replay_delay_s": relay_wall,
        "wall_delay_difference_s": relay_wall - candidate_wall,
    }


def horizontal_range_and_azimuth(
    point_m: Sequence[float], origin_m: Sequence[float]
) -> tuple[float, float]:
    delta = np.asarray(point_m, dtype=np.float64) - np.asarray(
        origin_m, dtype=np.float64
    )
    radius = float(hypot(delta[0], delta[1]))
    azimuth = degrees(atan2(delta[1], delta[0])) % 360.0
    return radius, azimuth


def finite_wall_points(
    wall: Wall, count: int
) -> Iterable[np.ndarray]:
    if count < 2:
        raise GeometryError("count must be at least 2")
    for x in np.linspace(wall.x_min_m, wall.x_max_m, count):
        yield np.array([x, wall.offset_m, 0.5 * (wall.z_min_m + wall.z_max_m)])
