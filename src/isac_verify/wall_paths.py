"""Phase 4 finite-wall path diagnostics and scene-verification helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sionna.rt import PathSolver
from sionna.rt.constants import InteractionType
from sionna.rt.rcs import RCSSolver

from .config import PilotConfig
from .geometry import (
    distance_m,
    wall_delay_s,
    wall_departure_angle_deg,
    wall_reflection_point,
)
from .material import build_material_reference, slab_coefficients
from .scenes import (
    BlockerSpec,
    CorridorScene,
    monostatic_corridor_scene,
    relay_corridor_scene,
    target_corridor_scene,
)


class WallPathError(RuntimeError):
    """Raised when a required finite-wall path or geometric check fails."""


@dataclass(frozen=True)
class PathSet:
    scene: CorridorScene
    paths: Any
    max_depth: int
    object_names: dict[int, str]

    @property
    def path_count(self) -> int:
        return int(np.asarray(self.paths.valid).sum())


def _solver_flags(config: PilotConfig, key: str) -> dict[str, Any]:
    solver = config.section("solver", key)
    return {
        "deterministic": bool(solver["deterministic"]),
        "refraction": bool(solver["refraction"]),
        "seed": int(solver["seed"]),
    }


def solve_target_paths(
    config: PilotConfig,
    *,
    wall_offsets_m: dict[str, float] | None = None,
    blocker: BlockerSpec | None = None,
    max_depth: int | None = None,
) -> PathSet:
    scene = target_corridor_scene(
        config,
        wall_offsets_m=wall_offsets_m,
        with_target=True,
        blocker=blocker,
    )
    flags = _solver_flags(config, "rcs_solver")
    depth = int(
        config.get("scene", "nominal_depth") if max_depth is None else max_depth
    )
    if depth < 1:
        raise WallPathError("RCS max_depth must be at least one")
    paths = RCSSolver(deterministic=flags["deterministic"])(
        scene.scene,
        max_depth=depth,
        synthetic_array=True,
        los=True,
        specular_reflection=True,
        refraction=flags["refraction"],
        seed=flags["seed"],
    )
    return PathSet(
        scene=scene,
        paths=paths,
        max_depth=depth,
        object_names=_object_names(scene),
    )


def solve_background_paths(
    config: PilotConfig,
    *,
    direction: str,
    max_depth: int = 1,
    wall_offsets_m: dict[str, float] | None = None,
) -> PathSet:
    if direction == "monostatic":
        scene = monostatic_corridor_scene(
            config, wall_offsets_m=wall_offsets_m
        )
    else:
        scene = relay_corridor_scene(
            config, direction=direction, wall_offsets_m=wall_offsets_m
        )
    flags = _solver_flags(config, "path_solver")
    paths = PathSolver(deterministic=flags["deterministic"])(
        scene.scene,
        max_depth=max_depth,
        synthetic_array=True,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=flags["refraction"],
        diffraction=False,
        seed=flags["seed"],
    )
    return PathSet(
        scene=scene,
        paths=paths,
        max_depth=max_depth,
        object_names=_object_names(scene),
    )


def _object_names(scene: CorridorScene) -> dict[int, str]:
    names = {
        int(obj.object_id): name for name, obj in scene.scene.objects.items()
    }
    return names


_INTERACTION_NAMES = {
    int(InteractionType.NONE): "NONE",
    int(InteractionType.SPECULAR): "SPECULAR",
    int(InteractionType.DIFFUSE): "DIFFUSE",
    int(InteractionType.REFRACTION): "REFRACTION",
    int(InteractionType.DIFFRACTION): "DIFFRACTION",
    int(InteractionType.SENSING): "SENSING",
}


def _interaction_name(value: int) -> str:
    return _INTERACTION_NAMES.get(int(value), f"UNKNOWN_{int(value)}")


def _arrays(paths: Any) -> tuple[np.ndarray, ...]:
    a_real, a_imag = paths.a
    a = np.asarray(a_real) + 1j * np.asarray(a_imag)
    return (
        np.asarray(paths.valid),
        np.asarray(paths.tau),
        a,
        np.asarray(paths.interactions),
        np.asarray(paths.objects),
        np.asarray(paths.vertices),
        np.asarray(paths.phi_t),
        np.asarray(paths.theta_t),
        np.asarray(paths.phi_r),
        np.asarray(paths.theta_r),
    )


def path_records(path_set: PathSet, *, scene_label: str) -> list[dict[str, Any]]:
    valid, tau, a, interactions, objects, vertices, phi_t, theta_t, phi_r, theta_r = _arrays(
        path_set.paths
    )
    if valid.ndim != 3:
        raise WallPathError(f"unexpected valid shape {valid.shape}")
    records: list[dict[str, Any]] = []
    for rx in range(valid.shape[0]):
        for tx in range(valid.shape[1]):
            for path_index in range(valid.shape[-1]):
                if not bool(valid[rx, tx, path_index]):
                    continue
                raw_interactions = [
                    int(value)
                    for value in interactions[:, rx, tx, path_index]
                ]
                raw_objects = [
                    int(value) for value in objects[:, rx, tx, path_index]
                ]
                kept = [
                    index
                    for index, value in enumerate(raw_interactions)
                    if value != 0
                ]
                names = [_interaction_name(raw_interactions[i]) for i in kept]
                path_objects = [raw_objects[i] for i in kept]
                path_vertices = [
                    [
                        float(value)
                        for value in vertices[i, rx, tx, path_index, :]
                    ]
                    for i in kept
                ]
                wall_sides = []
                for name, object_id in zip(names, path_objects):
                    if name == "SPECULAR":
                        wall_sides.append(
                            path_set.object_names.get(object_id, f"object_{object_id}")
                        )
                    elif name == "SENSING":
                        wall_sides.append("candidate")
                    else:
                        wall_sides.append("none")
                coefficient = (
                    a[rx, 0, tx, 0, path_index, 0]
                    if a.ndim == 6
                    else a[rx, 0, tx, 0, path_index]
                )
                family = _path_family(names, wall_sides)
                records.append(
                    {
                        "scene": scene_label,
                        "depth": int(path_set.max_depth),
                        "rx_index": rx,
                        "tx_index": tx,
                        "path_index": path_index,
                        "delay_ns": float(tau[rx, tx, path_index]) * 1e9,
                        "amplitude": float(abs(coefficient)),
                        "coefficient_real": float(np.real(coefficient)),
                        "coefficient_imag": float(np.imag(coefficient)),
                        "aod_deg": float(np.rad2deg(phi_t[rx, tx, path_index])),
                        "aoa_deg": float(np.rad2deg(phi_r[rx, tx, path_index])),
                        "aod_zenith_deg": float(
                            np.rad2deg(theta_t[rx, tx, path_index])
                        ),
                        "aoa_zenith_deg": float(
                            np.rad2deg(theta_r[rx, tx, path_index])
                        ),
                        "interaction_types": [
                            raw_interactions[i] for i in kept
                        ],
                        "interaction_names": names,
                        "object_ids": path_objects,
                        "object_names": wall_sides,
                        "vertex_points_m": path_vertices,
                        "family": family,
                        "path_signature": _path_signature(names, wall_sides),
                    }
                )
    for record in records:
        record["path_power_w"] = float(record["amplitude"] ** 2)
    return records


def _path_family(names: Sequence[str], objects: Sequence[str]) -> str:
    compact = tuple(names)
    if compact == ("SENSING",):
        return "direct"
    if compact == ("SPECULAR", "SENSING"):
        return f"outgoing_wall_{objects[0]}"
    if compact == ("SENSING", "SPECULAR"):
        return f"reciprocal_wall_{objects[1]}"
    if len(compact) == 1 and compact[0] == "SPECULAR":
        return f"wall_clutter_{objects[0]}"
    if compact and all(name == "SPECULAR" for name in compact):
        return "multi_bounce_" + "_".join(objects)
    if not compact:
        return "los"
    return "other"


def _path_signature(
    names: Sequence[str], objects: Sequence[str]
) -> str:
    return ">".join(f"{name}:{obj}" for name, obj in zip(names, objects))


def nominal_core_paths(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["depth"] != 2:
            continue
        key = record["family"]
        if key in {
            "direct",
            "outgoing_wall_positive_wall",
            "outgoing_wall_negative_wall",
            "reciprocal_wall_positive_wall",
            "reciprocal_wall_negative_wall",
        }:
            selected.setdefault(key, []).append(record)
    return selected


def find_outgoing_wall_path(
    records: Iterable[dict[str, Any]], wall_name: str
) -> dict[str, Any]:
    for record in records:
        if record["family"] == f"outgoing_wall_{wall_name}":
            return record
    raise WallPathError(f"missing outgoing-wall path for {wall_name}")


def find_reciprocal_wall_path(
    records: Iterable[dict[str, Any]], wall_name: str
) -> dict[str, Any]:
    for record in records:
        if record["family"] == f"reciprocal_wall_{wall_name}":
            return record
    raise WallPathError(f"missing reciprocal-wall path for {wall_name}")


def expected_target_wall_path(
    config: PilotConfig, wall_offset_m: float
) -> dict[str, float]:
    candidate = config.get("physical", "candidate_position_m")
    gnb = config.get("physical", "gnb_position_m")
    point = wall_reflection_point(
        gnb, candidate, wall_offset_m
    )
    range_m = float(candidate[0])
    delay = wall_delay_s(
        range_m, wall_offset_m, config.speed_of_light
    )
    departure = wall_departure_angle_deg(range_m, wall_offset_m)
    incidence = math.degrees(
        math.atan2(range_m, 2.0 * abs(wall_offset_m))
    )
    coefficient = slab_coefficients(config, incidence)
    length_unfolded = math.hypot(range_m, 2.0 * wall_offset_m)
    expected_amplitude = (
        config.wavelength_m
        * math.sqrt(float(config.get("target", "rcs_m2")))
        * coefficient["r_te"]["magnitude"]
        / ((4.0 * math.pi) ** 1.5 * length_unfolded * range_m)
    )
    return {
        "wall_offset_m": wall_offset_m,
        "reflection_point_m": point.tolist(),
        "aod_deg": departure,
        "aoa_deg": 0.0,
        "delay_ns": delay * 1e9,
        "incidence_from_normal_deg": incidence,
        "r_te_magnitude": coefficient["r_te"]["magnitude"],
        "r_tm_magnitude": coefficient["r_tm"]["magnitude"],
        "expected_amplitude": expected_amplitude,
        "unfolded_outgoing_leg_m": length_unfolded,
        "return_leg_m": range_m,
    }


def expected_relay_wall_path(
    config: PilotConfig,
    wall_offset_m: float,
    *,
    direction: str = "downlink",
) -> dict[str, float]:
    gnb = config.get("physical", "gnb_position_m")
    relay = config.get("physical", "relay_position_m")
    point = wall_reflection_point(gnb, relay, wall_offset_m)
    range_m = float(relay[0])
    length = math.hypot(range_m, 2.0 * wall_offset_m)
    departure = math.degrees(
        math.atan2(2.0 * wall_offset_m, range_m)
    )
    if direction == "uplink":
        departure = (180.0 - departure + 180.0) % 360.0 - 180.0
    elif direction != "downlink":
        raise ValueError(f"unknown relay direction {direction!r}")
    return {
        "wall_offset_m": wall_offset_m,
        "reflection_point_m": point.tolist(),
        "aod_deg": departure,
        "delay_ns": length / config.speed_of_light * 1e9,
        "unfolded_outgoing_leg_m": length,
        "relay_leg_m": range_m,
    }


def expected_monostatic_wall_path(
    config: PilotConfig, wall_offset_m: float
) -> dict[str, float]:
    gnb = config.get("physical", "gnb_position_m")
    delay = 2.0 * abs(wall_offset_m) / config.speed_of_light
    sign = 90.0 if wall_offset_m > 0.0 else -90.0
    return {
        "wall_offset_m": wall_offset_m,
        "reflection_point_m": [float(gnb[0]), wall_offset_m, float(gnb[2])],
        "aod_deg": sign,
        "aoa_deg": sign,
        "delay_ns": delay * 1e9,
    }


def depth_convergence(config: PilotConfig) -> dict[str, Any]:
    depths = [int(value) for value in config.get("scene", "phase4_max_depth")]
    summaries = []
    previous_signatures: set[str] = set()
    for depth in depths:
        path_set = solve_target_paths(config, max_depth=depth)
        records = path_records(path_set, scene_label=f"target_true_depth_{depth}")
        signatures = {
            (
                record["path_signature"],
                tuple(round(value, 6) for point in record["vertex_points_m"] for value in point),
            )
            for record in records
        }
        new = sorted(
            value[0]
            for value in signatures
            if value not in previous_signatures
        )
        family_counts: dict[str, int] = {}
        for record in records:
            family_counts[record["family"]] = (
                family_counts.get(record["family"], 0) + 1
            )
        summaries.append(
            {
                "max_depth": depth,
                "path_count": len(records),
                "total_path_power_w": float(
                    sum(record["path_power_w"] for record in records)
                ),
                "family_counts": family_counts,
                "new_path_signatures": new,
            }
        )
        previous_signatures = signatures
    for index in range(1, len(summaries)):
        previous = summaries[index - 1]
        current = summaries[index]
        current["total_power_change_w"] = (
            current["total_path_power_w"] - previous["total_path_power_w"]
        )
        current["total_power_change_fraction"] = (
            current["total_power_change_w"] / previous["total_path_power_w"]
            if previous["total_path_power_w"] > 0.0
            else None
        )
    return {
        "depths": depths,
        "summaries": summaries,
        "note": (
            "total_path_power_w sums |a_tx0_rx0|^2 over every valid RCS path at "
            "the reported maximum depth; duplicated high-depth paths are "
            "retained because they are present in the solver output"
        ),
    }


def _same_vertex_set(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
    *,
    tolerance_m: float = 1.0e-3,
) -> bool:
    if len(first) != len(second):
        return False
    for point_a, point_b in zip(first, second):
        if len(point_a) != len(point_b):
            return False
        if any(
            abs(float(a) - float(b)) > tolerance_m
            for a, b in zip(point_a, point_b)
        ):
            return False
    return True


def duplicate_path_groups(
    records: Sequence[dict[str, Any]],
    *,
    tolerance_m: float = 1.0e-3,
) -> list[list[int]]:
    """Group records with identical interactions, objects, and 1 mm vertices."""
    buckets: dict[tuple[Any, ...], list[int]] = {}
    for index, record in enumerate(records):
        key = (
            tuple(record["interaction_types"]),
            tuple(record["object_ids"]),
        )
        buckets.setdefault(key, []).append(index)
    groups: list[list[int]] = []
    for indices in buckets.values():
        clusters: list[list[int]] = []
        for index in indices:
            for cluster in clusters:
                if _same_vertex_set(
                    records[index]["vertex_points_m"],
                    records[cluster[0]]["vertex_points_m"],
                    tolerance_m=tolerance_m,
                ):
                    cluster.append(index)
                    break
            else:
                clusters.append([index])
        groups.extend(cluster for cluster in clusters if len(cluster) > 1)
    return groups


def _duplicate_summary(
    label: str,
    path_set: PathSet,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    groups = duplicate_path_groups(records)
    duplicate_indices = {index for group in groups for index in group}
    duplicate_power = sum(
        records[index]["path_power_w"] for index in duplicate_indices
    )
    total_power = sum(record["path_power_w"] for record in records)
    return {
        "label": label,
        "max_depth": path_set.max_depth,
        "path_count": len(records),
        "unique_path_count": len(records) - sum(len(group) - 1 for group in groups),
        "duplicate_group_count": len(groups),
        "duplicate_path_count": len(duplicate_indices),
        "extra_copy_count": sum(len(group) - 1 for group in groups),
        "duplicate_power_w": float(duplicate_power),
        "total_power_w": float(total_power),
        "duplicate_power_fraction": (
            float(duplicate_power / total_power) if total_power > 0.0 else 0.0
        ),
        "groups": [
            [
                {
                    "path_index": records[index]["path_index"],
                    "delay_ns": records[index]["delay_ns"],
                    "amplitude": records[index]["amplitude"],
                    "interaction_names": records[index]["interaction_names"],
                    "object_names": records[index]["object_names"],
                }
                for index in group
            ]
            for group in groups
        ],
    }


def duplicate_path_diagnostics(config: PilotConfig) -> dict[str, Any]:
    """P1 duplicate-path audit for target, background, and relay links."""
    specs = (
        ("target", lambda depth: solve_target_paths(config, max_depth=depth)),
        (
            "background_monostatic",
            lambda depth: solve_background_paths(
                config, direction="monostatic", max_depth=depth
            ),
        ),
        (
            "downlink",
            lambda depth: solve_background_paths(
                config, direction="downlink", max_depth=depth
            ),
        ),
        (
            "uplink",
            lambda depth: solve_background_paths(
                config, direction="uplink", max_depth=depth
            ),
        ),
    )
    entries = []
    for label, solve in specs:
        for depth in (3, 4):
            path_set = solve(depth)
            records = path_records(
                path_set, scene_label=f"{label}_duplicate_depth_{depth}"
            )
            entries.append(_duplicate_summary(label, path_set, records))
    return {
        "vertex_tolerance_m": 1.0e-3,
        "depths": [3, 4],
        "entries": entries,
        "total_duplicate_groups": sum(
            entry["duplicate_group_count"] for entry in entries
        ),
        "source_finding": (
            "RCSSolver._pair_legs forms the Cartesian product of the "
            "transmitter-side and receiver-side legs, keeps pairs that share "
            "a scattering point and satisfy depth <= max_depth, and applies "
            "no vertex-set duplicate removal. PathsBuffer.chain then combines "
            "each selected TX-side/RX-side leg into one path and also performs "
            "no deduplication. Any duplicate geometry enumerated by the "
            "solver therefore remains a separate coefficient in Paths.a and "
            "in the CIR sum. This is the installed-source behavior, not an "
            "inferred intent."
        ),
    }


def _depth_table(
    config: PilotConfig,
    *,
    name: str,
    solve: Any,
) -> dict[str, Any]:
    depths = [int(value) for value in config.get("scene", "phase4_max_depth")]
    rows = []
    for depth in depths:
        path_set = solve(depth)
        records = path_records(path_set, scene_label=f"{name}_depth_{depth}")
        rows.append(
            {
                "max_depth": depth,
                "path_count": len(records),
                "total_path_power_w": float(
                    sum(record["path_power_w"] for record in records)
                ),
                "families": sorted({record["family"] for record in records}),
            }
        )
    for index in range(1, len(rows)):
        previous = rows[index - 1]["total_path_power_w"]
        current = rows[index]["total_path_power_w"]
        rows[index]["power_change_w"] = current - previous
        rows[index]["power_change_fraction"] = (
            (current - previous) / previous if previous > 0.0 else None
        )
    return {"name": name, "depths": depths, "rows": rows}


def depth_power_decision(config: PilotConfig) -> dict[str, Any]:
    """P2/D9 per-link depth decision based on the next depth's power change."""
    tolerance = float(config.get("scene", "d9_path_power_relative_tolerance"))
    specs = {
        "target": lambda depth: solve_target_paths(config, max_depth=depth),
        "background": lambda depth: solve_background_paths(
            config, direction="monostatic", max_depth=depth
        ),
        "downlink": lambda depth: solve_background_paths(
            config, direction="downlink", max_depth=depth
        ),
        "uplink": lambda depth: solve_background_paths(
            config, direction="uplink", max_depth=depth
        ),
    }
    tables = {
        name: _depth_table(config, name=name, solve=solve)
        for name, solve in specs.items()
    }
    selected: dict[str, int | None] = {}
    for name, table in tables.items():
        chosen = None
        for index in range(len(table["rows"]) - 1):
            fraction = table["rows"][index + 1]["power_change_fraction"]
            if fraction is not None and abs(fraction) < tolerance:
                chosen = int(table["rows"][index]["max_depth"])
                break
        selected[name] = chosen
    required_relay_double_bounce = bool(
        config.get("scene", "d9_require_relay_double_bounce")
    )
    if required_relay_double_bounce:
        for name in ("downlink", "uplink"):
            if selected[name] is None or selected[name] < 2:
                selected[name] = 2
    configured = {
        name: int(config.get("scene", "d9_depths", name))
        for name in ("target", "background", "downlink", "uplink")
    }
    if selected != configured:
        raise WallPathError(
            "computed D9 depths "
            f"{selected} do not match configured {configured}"
        )
    return {
        "tolerance": tolerance,
        "require_relay_double_bounce": required_relay_double_bounce,
        "selected_depths": selected,
        "tables": tables,
        "note": (
            "The relay D9 is raised to 2 when the strict first under-tolerance "
            "depth is 1, so that first-order wall paths and the two "
            "wall-wall double-bounce paths are all retained for Phase 5. "
            "The depth-2 to depth-3 power change remains below the declared "
            "tolerance."
        ),
    }


def nominal_path_diagnostics(
    config: PilotConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    true_set = solve_target_paths(
        config, max_depth=int(config.get("scene", "nominal_depth"))
    )
    true_records = path_records(true_set, scene_label="target_true_nominal")
    core = nominal_core_paths(true_records)
    expected = {
        "positive_wall": expected_target_wall_path(config, 15.0),
        "negative_wall": expected_target_wall_path(config, -15.0),
    }
    comparisons = []
    for side, wall_name in (
        ("positive", "positive_wall"),
        ("negative", "negative_wall"),
    ):
        measured = find_outgoing_wall_path(true_records, wall_name)
        predicted = expected[wall_name]
        comparisons.append(
            {
                "wall": wall_name,
                "measured": measured,
                "image_method": predicted,
                "delay_error_ns": measured["delay_ns"] - predicted["delay_ns"],
                "aod_error_deg": measured["aod_deg"] - predicted["aod_deg"],
                "aoa_error_deg": measured["aoa_deg"] - predicted["aoa_deg"],
                "reflection_point_error_m": float(
                    distance_m(
                        measured["vertex_points_m"][0],
                        predicted["reflection_point_m"],
                    )
                ),
                "amplitude_relative_error": (
                    measured["amplitude"] - predicted["expected_amplitude"]
                )
                / predicted["expected_amplitude"],
            }
        )
    return (
        {
            "true_wall_offsets_m": true_set.scene.metadata["wall_offsets_m"],
            "wall_geometry": true_set.scene.metadata["walls"],
            "validity": true_set.scene.metadata["validity"],
            "core_paths": core,
            "outgoing_wall_comparisons": comparisons,
            "all_path_records": true_records,
        },
        true_records,
    )


def background_diagnostics(config: PilotConfig) -> dict[str, Any]:
    monostatic = solve_background_paths(
        config, direction="monostatic", max_depth=2
    )
    monostatic_records = path_records(
        monostatic, scene_label="background_monostatic"
    )
    downlink = solve_background_paths(
        config, direction="downlink", max_depth=2
    )
    downlink_records = path_records(downlink, scene_label="background_downlink")
    uplink = solve_background_paths(
        config, direction="uplink", max_depth=2
    )
    uplink_records = path_records(uplink, scene_label="background_uplink")
    clutter = []
    for record in monostatic_records:
        if record["family"].startswith("wall_clutter_"):
            side = record["family"].removeprefix("wall_clutter_")
            predicted = expected_monostatic_wall_path(
                config, monostatic.scene.wall_specs[side.split("_")[0]].offset_m
            )
            clutter.append(
                {
                    **record,
                    "image_method": predicted,
                    "delay_error_ns": record["delay_ns"] - predicted["delay_ns"],
                    "aod_error_deg": record["aod_deg"] - predicted["aod_deg"],
                    "aoa_error_deg": record["aoa_deg"] - predicted["aoa_deg"],
                    "reflection_point_error_m": float(
                        distance_m(
                            record["vertex_points_m"][0],
                            predicted["reflection_point_m"],
                        )
                    ),
                }
            )
    relay_table = []
    for direction, records in (("downlink", downlink_records), ("uplink", uplink_records)):
        for record in records:
            if record["family"] == "los" or record["family"].startswith(
                "wall_clutter_"
            ):
                side = (
                    record["family"].removeprefix("wall_clutter_")
                    if record["family"].startswith("wall_clutter_")
                    else None
                )
                predicted = (
                    expected_relay_wall_path(
                        config,
                        config.get("physical", "walls", side.split("_")[0])[
                            "offset_m"
                        ],
                        direction=direction,
                    )
                    if side
                    else None
                )
                relay_table.append(
                    {
                        "direction": direction,
                        **record,
                        "image_method": predicted,
                        "delay_error_ns": (
                            record["delay_ns"] - predicted["delay_ns"]
                            if predicted
                            else None
                        ),
                        "aod_error_deg": (
                            record["aod_deg"] - predicted["aod_deg"]
                            if predicted
                            else None
                        ),
                        "reflection_point_error_m": (
                            float(
                                distance_m(
                                    record["vertex_points_m"][0],
                                    predicted["reflection_point_m"],
                                )
                            )
                            if predicted
                            else None
                        ),
                    }
                )
    return {
        "monostatic_wall_clutter": clutter,
        "relay_links": relay_table,
        "monostatic_all_paths": monostatic_records,
        "downlink_all_paths": downlink_records,
        "uplink_all_paths": uplink_records,
        "note": (
            "monostatic self-link is solved only for background diagnosis; "
            "the D6 ideal-isolation model still excludes direct gNB leakage "
            "from the experiment. No diffuse scattering, diffraction, ground, "
            "or roof was enabled."
        ),
    }


def blocker_spec(config: PilotConfig) -> BlockerSpec:
    values = config.section("scene", "blocker")
    candidate = config.get("physical", "candidate_position_m")
    gnb = config.get("physical", "gnb_position_m")
    wall_y = float(config.get("physical", "walls", "positive", "offset_m"))
    wall_point = wall_reflection_point(gnb, candidate, wall_y)
    midpoint = 0.5 * (wall_point + np.asarray(candidate, dtype=np.float64))
    delta = np.asarray(candidate, dtype=np.float64) - wall_point
    azimuth = math.degrees(math.atan2(delta[1], delta[0]))
    return BlockerSpec(
        name=str(values["name"]),
        center_m=tuple(float(value) for value in midpoint),
        along_azimuth_deg=azimuth,
        width_m=float(values["width_m"]),
        height_m=float(values["height_m"]),
        thickness_m=float(values["thickness_m"]),
    )


def blocked_scene_diagnostics(config: PilotConfig) -> dict[str, Any]:
    blocker = blocker_spec(config)
    unblocked = solve_target_paths(config, max_depth=2)
    unblocked_records = path_records(unblocked, scene_label="target_true_unblocked")
    blocked = solve_target_paths(config, max_depth=2, blocker=blocker)
    blocked_records = path_records(blocked, scene_label="target_true_blocked")
    unblocked_families = {
        record["family"] for record in unblocked_records
    }
    blocked_families = {record["family"] for record in blocked_records}
    new_records = [
        record
        for record in blocked_records
        if record["path_signature"] not in {
            other["path_signature"] for other in unblocked_records
        }
    ]
    return {
        "blocker": blocker.__dict__,
        "blocker_scene_metadata": blocked.scene.metadata["blocker"],
        "unblocked_path_count": len(unblocked_records),
        "blocked_path_count": len(blocked_records),
        "unblocked_families": sorted(unblocked_families),
        "blocked_families": sorted(blocked_families),
        "removed_families": sorted(unblocked_families - blocked_families),
        "new_paths": new_records,
        "positive_outgoing_present": any(
            record["family"] == "outgoing_wall_positive_wall"
            for record in blocked_records
        ),
        "positive_reciprocal_present": any(
            record["family"] == "reciprocal_wall_positive_wall"
            for record in blocked_records
        ),
        "negative_outgoing_present": any(
            record["family"] == "outgoing_wall_negative_wall"
            for record in blocked_records
        ),
        "blocked_records": blocked_records,
        "unblocked_records": unblocked_records,
    }


def twin_diagnostics(config: PilotConfig) -> dict[str, Any]:
    sigma = float(config.get("scene", "wall_twin_sigma_m"))
    offsets = {"positive": 15.0 + sigma, "negative": -15.0 - sigma}
    true_set = solve_target_paths(config, max_depth=2)
    twin_set = solve_target_paths(config, max_depth=2, wall_offsets_m=offsets)
    true_records = path_records(true_set, scene_label="target_true")
    twin_records = path_records(twin_set, scene_label="target_twin")
    rows = []
    for side, wall_name in (
        ("positive", "positive_wall"),
        ("negative", "negative_wall"),
    ):
        true_measured = find_outgoing_wall_path(true_records, wall_name)
        twin_measured = find_outgoing_wall_path(twin_records, wall_name)
        true_expected = expected_target_wall_path(
            config, true_set.scene.wall_specs[side].offset_m
        )
        twin_expected = expected_target_wall_path(
            config, twin_set.scene.wall_specs[side].offset_m
        )
        rows.append(
            {
                "wall": wall_name,
                "true_offset_m": true_set.scene.wall_specs[side].offset_m,
                "twin_offset_m": twin_set.scene.wall_specs[side].offset_m,
                "image_true_delay_ns": true_expected["delay_ns"],
                "image_twin_delay_ns": twin_expected["delay_ns"],
                "image_delay_shift_ns": (
                    twin_expected["delay_ns"] - true_expected["delay_ns"]
                ),
                "image_true_aod_deg": true_expected["aod_deg"],
                "image_twin_aod_deg": twin_expected["aod_deg"],
                "image_aod_shift_deg": (
                    twin_expected["aod_deg"] - true_expected["aod_deg"]
                ),
                "rt_true_delay_ns": true_measured["delay_ns"],
                "rt_twin_delay_ns": twin_measured["delay_ns"],
                "rt_delay_shift_ns": (
                    twin_measured["delay_ns"] - true_measured["delay_ns"]
                ),
                "rt_true_aod_deg": true_measured["aod_deg"],
                "rt_twin_aod_deg": twin_measured["aod_deg"],
                "rt_aod_shift_deg": (
                    twin_measured["aod_deg"] - true_measured["aod_deg"]
                ),
                "image_twin_vs_rt_twin_delay_ns": (
                    twin_measured["delay_ns"] - twin_expected["delay_ns"]
                ),
                "image_twin_vs_rt_twin_aod_deg": (
                    twin_measured["aod_deg"] - twin_expected["aod_deg"]
                ),
            }
        )
    return {
        "epsilon_plus_m": sigma,
        "epsilon_minus_m": -sigma,
        "wall_offsets_m": offsets,
        "wall_geometry": twin_set.scene.metadata["walls"],
        "rows": rows,
        "twin_records": twin_records,
        "true_records": true_records,
    }


def write_phase4_outputs(config: PilotConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    nominal, nominal_records = nominal_path_diagnostics(config)
    all_records: list[dict[str, Any]] = []
    for depth in [int(value) for value in config.get("scene", "phase4_max_depth")]:
        path_set = solve_target_paths(config, max_depth=depth)
        all_records.extend(
            path_records(path_set, scene_label=f"target_true_depth_{depth}")
        )
    all_records.extend(nominal_records)
    background = background_diagnostics(config)
    all_records.extend(background["monostatic_all_paths"])
    all_records.extend(background["downlink_all_paths"])
    all_records.extend(background["uplink_all_paths"])
    convergence = depth_convergence(config)
    duplicates = duplicate_path_diagnostics(config)
    depth_decision = depth_power_decision(config)
    blocked = blocked_scene_diagnostics(config)
    all_records.extend(blocked["unblocked_records"])
    all_records.extend(blocked["blocked_records"])
    twin = twin_diagnostics(config)
    all_records.extend(twin["true_records"])
    all_records.extend(twin["twin_records"])
    frame = pd.DataFrame(all_records)
    frame.to_parquet(output / "paths_diagnostic.parquet", index=False)
    material = build_material_reference(config)
    geometry = {
        "phase": 4,
        "walls": nominal["true_wall_offsets_m"],
        "target_nominal": nominal,
        "background": background,
        "depth_convergence": convergence,
        "duplicate_path_audit": duplicates,
        "d9_depth_decision": depth_decision,
        "delay_coincidence_note": (
            "At background depth 4, the four-bounce wall-wall monostatic "
            "clutter path is at 120 m and 400.276917 ns, delay-coincident "
            "with the direct target return (AoA +/-90 deg vs 0 deg). It is "
            "not present at the selected background D9=2; it appears only at "
            "depth 4."
        ),
        "blocked_scene": {
            key: value
            for key, value in blocked.items()
            if key not in ("blocked_records", "unblocked_records")
        },
        "twin": {
            key: value
            for key, value in twin.items()
            if key not in ("twin_records", "true_records")
        },
        "material_r_te_at_63_435": material["coefficients"][-1]["r_te"],
        "d8_wall_thickness": build_thickness_reference(config),
    }
    (output / "geometry_reference.json").write_text(
        _json_dump(geometry), encoding="utf-8"
    )
    return {
        "nominal": nominal,
        "background": background,
        "convergence": convergence,
        "blocked": blocked,
        "twin": twin,
        "material": material,
        "d8": geometry["d8_wall_thickness"],
        "duplicates": duplicates,
        "depth_decision": depth_decision,
        "records": all_records,
    }


def build_thickness_reference(config: PilotConfig) -> dict[str, Any]:
    from .material import build_wall_thickness_stress

    return build_wall_thickness_stress(config)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_dump(value: Any) -> str:
    import json

    return json.dumps(_jsonable(value), indent=2) + "\n"
