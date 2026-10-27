"""Finite-wall true and twin scenes for Phase 4 path verification.

Walls are single planar rectangles with inward-facing normals.  Target cubes
are only added to scenes used by ``RCSSolver``; ``PathSolver`` scenes never
contain a sensing target (D4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import mitsuba as mi
import numpy as np
from sionna.rt import (
    AbsorberRadioMaterial,
    PlanarArray,
    RadioMaterial,
    Receiver,
    SceneObject,
    Transmitter,
    load_scene,
)
from sionna.rt.rcs import ConstantRCSSensingTarget

from .config import PilotConfig


class SceneError(ValueError):
    """Raised when a finite-wall scene violates a declared geometry rule."""


@dataclass(frozen=True)
class WallSpec:
    name: str
    offset_m: float
    inward_normal: tuple[float, float, float]
    x_min_m: float
    x_max_m: float
    z_min_m: float
    z_max_m: float

    @classmethod
    def from_config(
        cls, config: PilotConfig, side: str, offset_m: float | None = None
    ) -> "WallSpec":
        values = config.section("physical", "walls", side)
        return cls(
            name=f"{side}_wall",
            offset_m=(
                float(values["offset_m"]) if offset_m is None else float(offset_m)
            ),
            inward_normal=tuple(
                float(value) for value in values["inward_normal"]
            ),
            x_min_m=float(values["x_min_m"]),
            x_max_m=float(values["x_max_m"]),
            z_min_m=float(values["z_min_m"]),
            z_max_m=float(values["z_max_m"]),
        )

    @property
    def width_m(self) -> float:
        return self.x_max_m - self.x_min_m

    @property
    def height_m(self) -> float:
        return self.z_max_m - self.z_min_m


@dataclass(frozen=True)
class BlockerSpec:
    name: str
    center_m: tuple[float, float, float]
    along_azimuth_deg: float
    width_m: float
    height_m: float
    thickness_m: float = 0.05


@dataclass
class CorridorScene:
    scene: Any
    walls: dict[str, SceneObject]
    wall_specs: dict[str, WallSpec]
    material: RadioMaterial
    target: ConstantRCSSensingTarget | None
    blocker: SceneObject | None
    metadata: dict[str, Any]


def _wall_transform(spec: WallSpec) -> mi.ScalarTransform4f:
    center = mi.ScalarPoint3f(
        0.5 * (spec.x_min_m + spec.x_max_m),
        spec.offset_m,
        0.5 * (spec.z_min_m + spec.z_max_m),
    )
    angle = 90.0 if spec.inward_normal[1] < 0.0 else -90.0
    return (
        mi.ScalarTransform4f.translate(center)
        @ mi.ScalarTransform4f.rotate(mi.ScalarPoint3f(1.0, 0.0, 0.0), angle)
        @ mi.ScalarTransform4f.scale(
            mi.ScalarPoint3f(0.5 * spec.width_m, 0.5 * spec.height_m, 1.0)
        )
    )


def build_wall_mesh(spec: WallSpec) -> mi.Mesh:
    mesh = mi.load_dict(
        {"type": "rectangle", "to_world": _wall_transform(spec)}
    )
    if not isinstance(mesh, mi.Mesh):
        raise SceneError("wall rectangle did not produce a Mitsuba mesh")
    return mesh


def wall_geometry(
    spec: WallSpec,
    mesh: mi.Mesh,
    *,
    planarity_tolerance_m: float = 1e-5,
    dimension_tolerance_m: float = 1e-4,
) -> dict[str, Any]:
    vertices = np.asarray(
        [
            np.asarray(mesh.vertex_position(index)).reshape(3)
            for index in range(int(mesh.vertex_count()))
        ]
    )
    bbox = mesh.bbox()
    extents = bbox.max - bbox.min
    face_normals = np.asarray(
        [
            np.asarray(mesh.face_normal(index)).reshape(3)
            for index in range(int(mesh.face_count()))
        ]
    )
    expected = np.asarray(spec.inward_normal, dtype=np.float64)
    normalized = face_normals / np.linalg.norm(
        face_normals, axis=1, keepdims=True
    )
    alignment = normalized @ expected
    if alignment.min() < 1.0 - 1e-7:
        raise SceneError(
            f"{spec.name}: mesh normal does not point inward: {alignment.tolist()}"
        )
    y_spread = float(np.ptp(vertices[:, 1]))
    if y_spread > planarity_tolerance_m:
        raise SceneError(f"{spec.name}: wall is not planar, y spread {y_spread}")
    dimensions = (float(extents.x), float(extents.y), float(extents.z))
    expected_dimensions = (spec.width_m, 0.0, spec.height_m)
    for measured, expected_size in zip(dimensions, expected_dimensions):
        if expected_size > 0.0 and not np.isclose(
            measured, expected_size, rtol=0.0, atol=dimension_tolerance_m
        ):
            raise SceneError(
                f"{spec.name}: measured extent {measured} != {expected_size}"
            )
    return {
        "name": spec.name,
        "offset_m": spec.offset_m,
        "inward_normal_expected": expected.tolist(),
        "inward_normal_measured": normalized[0].tolist(),
        "normal_alignment_min": float(alignment.min()),
        "planarity_y_spread_m": y_spread,
        "bbox_min_m": np.asarray(bbox.min).reshape(3).tolist(),
        "bbox_max_m": np.asarray(bbox.max).reshape(3).tolist(),
        "width_m": float(extents.x),
        "height_m": float(extents.z),
        "surface_area_m2": float(mesh.surface_area()[0]),
    }


def build_wall_material(
    config: PilotConfig, name: str = "wall_material"
) -> RadioMaterial:
    wall = config.section("wall")
    return RadioMaterial(
        name=name,
        thickness=float(wall["thickness_m"]),
        relative_permittivity=float(wall["relative_permittivity"]),
        conductivity=float(wall["conductivity_s_per_m"]),
        scattering_coefficient=float(wall["scattering_coefficient"]),
        xpd_coefficient=float(wall["xpd_coefficient"]),
    )


def _planar_array(values: dict[str, Any]) -> PlanarArray:
    if "num_elements" in values:
        return PlanarArray(
            num_rows=1,
            num_cols=int(values["num_elements"]),
            vertical_spacing=float(values["spacing_wavelengths"]),
            horizontal_spacing=float(values["spacing_wavelengths"]),
            pattern=str(values["pattern"]),
            polarization=str(values["polarization"]),
        )
    return PlanarArray(
        num_rows=int(values["num_rows"]),
        num_cols=int(values["num_cols"]),
        vertical_spacing=float(values["vertical_spacing"]),
        horizontal_spacing=float(values["horizontal_spacing"]),
        pattern=str(values["pattern"]),
        polarization=str(values["polarization"]),
    )


def _blocker_mesh(spec: BlockerSpec) -> mi.Mesh:
    transform = (
        mi.ScalarTransform4f.translate(mi.ScalarPoint3f(*spec.center_m))
        @ mi.ScalarTransform4f.rotate(
            mi.ScalarPoint3f(0.0, 0.0, 1.0), spec.along_azimuth_deg
        )
        @ mi.ScalarTransform4f.scale(
            mi.ScalarPoint3f(
                0.5 * spec.thickness_m,
                0.5 * spec.width_m,
                0.5 * spec.height_m,
            )
        )
    )
    mesh = mi.load_dict({"type": "cube", "to_world": transform})
    if not isinstance(mesh, mi.Mesh):
        raise SceneError("blocker cube did not produce a Mitsuba mesh")
    return mesh


def validate_corridor_layout(
    config: PilotConfig,
    wall_specs: Sequence[WallSpec],
    *,
    points_m: Sequence[Sequence[float]],
) -> dict[str, float]:
    if len(wall_specs) != 2:
        raise SceneError("a corridor layout requires exactly two walls")
    positive = max(wall_specs, key=lambda wall: wall.offset_m)
    negative = min(wall_specs, key=lambda wall: wall.offset_m)
    half_width = 0.5 * (positive.offset_m - negative.offset_m)
    minimum = float(config.get("scene", "corridor_min_inner_half_width_m"))
    clearance = float(config.get("scene", "corridor_object_clearance_m"))
    if half_width < minimum:
        raise SceneError(
            f"corridor half-width {half_width} m is below the declared "
            f"{minimum} m validity floor"
        )
    for point in points_m:
        x, y, z = (float(value) for value in point)
        if not (
            negative.x_min_m <= x <= positive.x_max_m
            and positive.z_min_m <= z <= positive.z_max_m
        ):
            raise SceneError(f"point {point} is outside the finite wall extent")
        if not negative.offset_m + clearance < y < positive.offset_m - clearance:
            raise SceneError(
                f"point {point} lacks {clearance} m clearance from a wall"
            )
    return {
        "inner_half_width_m": half_width,
        "minimum_inner_half_width_m": minimum,
        "object_clearance_m": clearance,
    }


def build_corridor_scene(
    config: PilotConfig,
    *,
    wall_offsets_m: dict[str, float] | None = None,
    tx_name: str,
    tx_position_m: Sequence[float],
    tx_array_key: str,
    rx_name: str,
    rx_position_m: Sequence[float],
    rx_array_key: str,
    with_target: bool = False,
    target_position_m: Sequence[float] | None = None,
    blocker: BlockerSpec | None = None,
    additional_points_m: Sequence[Sequence[float]] = (),
) -> CorridorScene:
    offsets = wall_offsets_m or {}
    specs = {
        "positive": WallSpec.from_config(
            config, "positive", offsets.get("positive")
        ),
        "negative": WallSpec.from_config(
            config, "negative", offsets.get("negative")
        ),
    }
    points = [tx_position_m, rx_position_m, *additional_points_m]
    if with_target:
        if target_position_m is None:
            raise SceneError("target position is required with a sensing target")
        points.append(target_position_m)
    validity = validate_corridor_layout(
        config, list(specs.values()), points_m=points
    )

    scene = load_scene()
    scene.frequency = config.frequency_hz
    scene.tx_array = _planar_array(config.section("arrays", tx_array_key))
    scene.rx_array = _planar_array(config.section("arrays", rx_array_key))
    material = build_wall_material(config)
    scene.add(
        [
            material,
            Transmitter(tx_name, position=list(tx_position_m)),
            Receiver(rx_name, position=list(rx_position_m)),
        ]
    )
    walls: dict[str, SceneObject] = {}
    wall_metadata: dict[str, Any] = {}
    for side, spec in specs.items():
        mesh = build_wall_mesh(spec)
        walls[side] = SceneObject(
            mi_mesh=mesh, name=spec.name, radio_material=material
        )
        wall_metadata[side] = wall_geometry(
            spec,
            mesh,
            planarity_tolerance_m=float(
                config.get("scene", "planarity_tolerance_m")
            ),
            dimension_tolerance_m=float(
                config.get("scene", "wall_dimension_tolerance_m")
            ),
        )
    scene.add(list(walls.values()))

    target = None
    if with_target:
        target = ConstantRCSSensingTarget(
            name="candidate",
            sigma=float(config.get("target", "rcs_m2")),
            length=0.2,
            width=0.2,
            height=0.2,
            position=list(target_position_m),
        )
        scene.add(target)

    blocker_object = None
    blocker_metadata = None
    if blocker is not None:
        absorber = AbsorberRadioMaterial(name=f"{blocker.name}_absorber")
        blocker_mesh = _blocker_mesh(blocker)
        blocker_object = SceneObject(
            mi_mesh=blocker_mesh,
            name=blocker.name,
            radio_material=absorber,
        )
        scene.add([absorber, blocker_object])
        bbox = blocker_mesh.bbox()
        blocker_metadata = {
            "name": blocker.name,
            "center_m": list(blocker.center_m),
            "along_azimuth_deg": blocker.along_azimuth_deg,
            "width_m": blocker.width_m,
            "height_m": blocker.height_m,
            "thickness_m": blocker.thickness_m,
            "bbox_min_m": np.asarray(bbox.min).reshape(3).tolist(),
            "bbox_max_m": np.asarray(bbox.max).reshape(3).tolist(),
            "material": "AbsorberRadioMaterial (Sionna perfect absorber)",
        }

    metadata = {
        "wall_offsets_m": {
            side: spec.offset_m for side, spec in specs.items()
        },
        "walls": wall_metadata,
        "validity": validity,
        "tx": {"name": tx_name, "position_m": list(tx_position_m)},
        "rx": {"name": rx_name, "position_m": list(rx_position_m)},
        "with_target": with_target,
        "target_position_m": (
            list(target_position_m) if target_position_m is not None else None
        ),
        "blocker": blocker_metadata,
    }
    return CorridorScene(
        scene=scene,
        walls=walls,
        wall_specs=specs,
        material=material,
        target=target,
        blocker=blocker_object,
        metadata=metadata,
    )


def target_corridor_scene(
    config: PilotConfig,
    *,
    wall_offsets_m: dict[str, float] | None = None,
    with_target: bool = True,
    blocker: BlockerSpec | None = None,
) -> CorridorScene:
    gnb = config.get("physical", "gnb_position_m")
    candidate = config.get("physical", "candidate_position_m")
    return build_corridor_scene(
        config,
        wall_offsets_m=wall_offsets_m,
        tx_name="gnb",
        tx_position_m=gnb,
        tx_array_key="gnb_tx",
        rx_name="gnb-rx",
        rx_position_m=gnb,
        rx_array_key="gnb_rx",
        with_target=with_target,
        target_position_m=candidate,
        blocker=blocker,
        additional_points_m=[candidate],
    )


def relay_corridor_scene(
    config: PilotConfig,
    *,
    direction: str,
    wall_offsets_m: dict[str, float] | None = None,
) -> CorridorScene:
    gnb = config.get("physical", "gnb_position_m")
    relay = config.get("physical", "relay_position_m")
    if direction == "downlink":
        return build_corridor_scene(
            config,
            wall_offsets_m=wall_offsets_m,
            tx_name="gnb",
            tx_position_m=gnb,
            tx_array_key="gnb_tx",
            rx_name="relay",
            rx_position_m=relay,
            rx_array_key="relay",
            additional_points_m=[relay],
        )
    if direction == "uplink":
        return build_corridor_scene(
            config,
            wall_offsets_m=wall_offsets_m,
            tx_name="relay",
            tx_position_m=relay,
            tx_array_key="relay",
            rx_name="gnb-rx",
            rx_position_m=gnb,
            rx_array_key="gnb_rx",
            additional_points_m=[relay],
        )
    raise SceneError(f"unknown relay direction {direction!r}")


def monostatic_corridor_scene(
    config: PilotConfig,
    *,
    wall_offsets_m: dict[str, float] | None = None,
) -> CorridorScene:
    gnb = config.get("physical", "gnb_position_m")
    return build_corridor_scene(
        config,
        wall_offsets_m=wall_offsets_m,
        tx_name="gnb",
        tx_position_m=gnb,
        tx_array_key="gnb_tx",
        rx_name="gnb-rx",
        rx_position_m=gnb,
        rx_array_key="gnb_rx",
    )
