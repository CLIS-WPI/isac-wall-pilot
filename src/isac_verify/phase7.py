"""Phase 7 smoke experiment: paired worlds, map-error twin, and scan controls."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from .beams import (
    ArrayGeometry,
    array_response,
    dbm_to_watt,
    direct_beam,
    beam_gain,
)
from .config import PilotConfig
from .geometry import horizontal_range_and_azimuth
from .replay import calibrate_matched_relay
from .receiver import (
    AngularDelayMap,
    CandidateEstimate,
    LinkPath,
    Observation,
    PointSpread,
    ProcessingChoices,
    angular_delay_map,
    delay_doppler_map,
    estimate_candidate,
    gate_energy,
    link_paths,
    make_observation,
    measure_point_spread,
    noise_calibration,
)
from .verification import (
    PlannerInput,
    WallPlan,
    direct_gate_window,
    evidence_score,
    plan_wall_action,
)
from .waveform import qpsk_symbols
from .wall_paths import (
    PathSet,
    solve_background_paths,
    solve_target_paths,
    path_records,
)


class Phase7Error(RuntimeError):
    """Raised when a smoke-stage invariant is violated."""


METHODS = (
    "direct_only",
    "exact_oracle",
    "imperfect_narrow",
    "imperfect_robust",
    "mapfree_two_beam",
    "mapfree_coarse_to_fine",
)


def _require_gpu(config: PilotConfig) -> None:
    backend = str(config.get("device", "backend")).lower()
    if backend != "cuda":
        raise Phase7Error(f"scientific run refuses non-CUDA backend {backend!r}")
    if not torch.cuda.is_available():
        raise Phase7Error("scientific run requires CUDA; CPU fallback is forbidden")
    name = torch.cuda.get_device_name(0)
    required = str(config.get("device", "require_name_substring"))
    if required and required not in name:
        raise Phase7Error(f"GPU name {name!r} does not contain {required!r}")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _now() -> float:
    _sync()
    return time.perf_counter()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (complex, np.complexfloating)):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _seed_manifest(config: PilotConfig) -> dict[str, Any]:
    base = int(config.get("seeds", "master"))
    smoke = {
        "probe_symbols": [base + 7301 + i for i in range(16)],
        "receiver_noise": [base + 7401 + i for i in range(16)],
        "relay_noise": [base + 7501 + i for i in range(16)],
        "beam_selection": [base + 7601 + i for i in range(16)],
        "twin_error": [
            [base + 7801 + 10 * index + i for i in range(4)]
            for index in range(3)
        ],
        "physical_geometry": [base + 7701],
    }
    return {
        "policy": "explicit_per_stream_never_none",
        "master": base,
        "stage": "smoke",
        "records_per_case": 16,
        "twin_realizations_per_sigma": 4,
        "streams": smoke,
        "pairing": (
            "record_index is shared across methods, sigma settings, and worlds; "
            "probe, receiver-noise, and relay-noise seeds are identical within "
            "a (world, sigma, record) cell"
        ),
    }


def _twin_offsets(
    config: PilotConfig,
    sigma_m: float,
    seed: int,
) -> dict[str, float]:
    positive = float(config.get("physical", "walls", "positive", "offset_m"))
    negative = float(config.get("physical", "walls", "negative", "offset_m"))
    delta = 0.0
    if sigma_m > 0.0:
        delta = float(np.random.default_rng(int(seed)).normal(0.0, sigma_m))
    return {"positive": positive + delta, "negative": negative + delta}


def _true_paths(config: PilotConfig) -> dict[str, PathSet]:
    return {
        "target": solve_target_paths(
            config, max_depth=int(config.get("scene", "d9_depths", "target"))
        ),
        "background": solve_background_paths(
            config,
            direction="monostatic",
            max_depth=int(config.get("scene", "d9_depths", "background")),
        ),
        "downlink": solve_background_paths(
            config,
            direction="downlink",
            max_depth=int(config.get("scene", "d9_depths", "downlink")),
        ),
        "uplink": solve_background_paths(
            config,
            direction="uplink",
            max_depth=int(config.get("scene", "d9_depths", "uplink")),
        ),
    }


def _true_links(paths: dict[str, PathSet]) -> dict[str, tuple[LinkPath, ...]]:
    return {
        "target": link_paths(paths["target"], label="true_target"),
        "background": link_paths(
            paths["background"], label="true_background", exclude_los=True
        ),
        "downlink": link_paths(paths["downlink"], label="true_downlink"),
        "uplink": link_paths(paths["uplink"], label="true_uplink"),
    }


def _calibrate(config: PilotConfig, paths: dict[str, PathSet]) -> Any:
    target = link_paths(
        paths["target"], label="true_target", families=("direct",)
    )[0]
    downlink = link_paths(
        paths["downlink"], label="true_downlink", families=("los",)
    )[0]
    uplink = link_paths(
        paths["uplink"], label="true_uplink", families=("los",)
    )[0]
    array = ArrayGeometry.from_config(config)
    phi = float(config.get("experiment_a", "reference_beam_phi_deg"))
    reference = direct_beam(
        phi,
        dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
        array,
    )
    rx_values = config.section("arrays", "gnb_rx")
    rx_array = ArrayGeometry(
        num_elements=int(rx_values["num_elements"]),
        spacing_wavelengths=float(rx_values["spacing_wavelengths"]),
        wavelength_m=config.wavelength_m,
    )
    rx_reference = array_response(phi, rx_array, dtype=torch.complex64)
    return calibrate_matched_relay(
        target_coefficient=target.matrix,
        target_delay_s=target.delay_s,
        downlink_coefficient=downlink.matrix.reshape(-1),
        downlink_delay_s=downlink.delay_s,
        uplink_coefficient=uplink.matrix.reshape(-1),
        uplink_delay_s=uplink.delay_s,
        reference_weights=reference.weights,
        reference_rx_weights=rx_reference,
        config=config,
        reference_beam_phi_deg=phi,
    )


def _observe(
    *,
    config: PilotConfig,
    world: str,
    beam_weights: torch.Tensor,
    symbols: torch.Tensor,
    profile: Any,
    target_links: Sequence[LinkPath],
    background_links: Sequence[LinkPath],
    downlink_links: Sequence[LinkPath],
    uplink_links: Sequence[LinkPath],
    calibration: Any,
    probe_seed: int,
    noise_seed: int,
    relay_noise_seed: int,
    noiseless: bool = False,
) -> Observation:
    return make_observation(
        world=world,
        beam_name="smoke",
        beam_weights=beam_weights,
        symbols=symbols,
        target_paths=tuple(target_links) if world == "target" else (),
        background_paths=tuple(background_links),
        relay_downlink=tuple(downlink_links),
        relay_uplink=tuple(uplink_links),
        relay_gain=calibration.gain if calibration is not None else 0.0,
        relay_added_delay_s=(
            calibration.measured_added_delay_s
            if calibration is not None
            else 0.0
        ),
        include_relay=(world == "replay"),
        profile=profile,
        config=config,
        noise_seed=int(noise_seed),
        relay_noise_seed=int(relay_noise_seed),
        probe_seed=int(probe_seed),
        noiseless=noiseless,
    )


def _map(
    observation: Observation,
    choices: ProcessingChoices,
    noise_per_cell_w: float,
    timing: dict[str, float],
) -> AngularDelayMap:
    start = _now()
    dd = delay_doppler_map(observation.h_eff, choices)
    result = angular_delay_map(
        dd, choices, noise_per_cell_w=float(noise_per_cell_w)
    )
    elapsed = _now() - start
    timing["receiver_s"] = timing.get("receiver_s", 0.0) + elapsed
    if timing.get("cold_jit_s", 0.0) == 0.0:
        timing["cold_jit_s"] = elapsed
    return result


def _twin_paths(
    config: PilotConfig,
    candidate: CandidateEstimate,
    offsets: dict[str, float],
    timing: dict[str, float],
) -> tuple[LinkPath, ...]:
    start = _now()
    solved = solve_target_paths(
        config,
        wall_offsets_m=offsets,
        max_depth=int(config.get("scene", "d9_depths", "target")),
        target_position_m=candidate.position_m,
    )
    timing["twin_scene_solve_s"] = (
        timing.get("twin_scene_solve_s", 0.0) + _now() - start
    )
    return link_paths(solved, label="twin_candidate")


def _twin_map(
    *,
    config: PilotConfig,
    choices: ProcessingChoices,
    weights: torch.Tensor,
    twin_paths: Sequence[LinkPath],
    timing: dict[str, float],
) -> AngularDelayMap:
    ones = torch.ones(
        (choices.profile.num_symbols, choices.profile.fft_size),
        device="cuda",
        dtype=torch.complex64,
    )
    observation = _observe(
        config=config,
        world="target",
        beam_weights=weights,
        symbols=ones,
        profile=choices.profile,
        target_links=twin_paths,
        background_links=(),
        downlink_links=(),
        uplink_links=(),
        calibration=None,
        probe_seed=0,
        noise_seed=0,
        relay_noise_seed=0,
        noiseless=True,
    )
    return _map(observation, choices, 0.0, timing)


def _measure_psf(
    *,
    config: PilotConfig,
    choices: ProcessingChoices,
    twin_paths: Sequence[LinkPath],
    direct_weights: torch.Tensor,
    timing: dict[str, float],
) -> PointSpread:
    direct = [path for path in twin_paths if path.family == "direct"]
    if not direct:
        raise Phase7Error("twin candidate has no direct path")
    ones = torch.ones(
        (choices.profile.num_symbols, choices.profile.fft_size),
        device="cuda",
        dtype=torch.complex64,
    )
    observation = _observe(
        config=config,
        world="target",
        beam_weights=direct_weights,
        symbols=ones,
        profile=choices.profile,
        target_links=direct,
        background_links=(),
        downlink_links=(),
        uplink_links=(),
        calibration=None,
        probe_seed=0,
        noise_seed=0,
        relay_noise_seed=0,
        noiseless=True,
    )
    angular = _map(observation, choices, 0.0, timing)
    return measure_point_spread(
        angular, choices=choices, h_eff=observation.h_eff
    )


def _base_world(
    *,
    config: PilotConfig,
    choices: ProcessingChoices,
    world: str,
    direct_weights: torch.Tensor,
    links: dict[str, tuple[LinkPath, ...]],
    calibration: Any,
    noise_full: float,
    timing: dict[str, float],
    seeds: dict[str, Any],
) -> dict[str, Any]:
    profile = choices.profile
    symbols = qpsk_symbols(profile, int(seeds["streams"]["probe_symbols"][0]))
    start = _now()
    observation = _observe(
        config=config,
        world=world,
        beam_weights=direct_weights,
        symbols=symbols,
        profile=profile,
        target_links=links["target"],
        background_links=links["background"],
        downlink_links=links["downlink"],
        uplink_links=links["uplink"],
        calibration=calibration,
        probe_seed=int(seeds["streams"]["probe_symbols"][0]),
        noise_seed=int(seeds["streams"]["receiver_noise"][0]),
        relay_noise_seed=int(seeds["streams"]["relay_noise"][0]),
    )
    timing["propagation_s"] = timing.get("propagation_s", 0.0) + _now() - start
    direct_map = _map(observation, choices, noise_full, timing)
    candidate = estimate_candidate(
        direct_map,
        choices=choices,
        point_spread=None,
        gnb_position_m=config.get("physical", "gnb_position_m"),
        speed_of_light_m_per_s=config.speed_of_light,
        h_eff=observation.h_eff,
    )
    nominal_offsets = {
        "positive": float(config.get("physical", "walls", "positive", "offset_m")),
        "negative": float(config.get("physical", "walls", "negative", "offset_m")),
    }
    twin = _twin_paths(config, candidate, nominal_offsets, timing)
    psf = _measure_psf(
        config=config,
        choices=choices,
        twin_paths=twin,
        direct_weights=direct_weights,
        timing=timing,
    )
    candidate = estimate_candidate(
        direct_map,
        choices=choices,
        point_spread=psf,
        gnb_position_m=config.get("physical", "gnb_position_m"),
        speed_of_light_m_per_s=config.speed_of_light,
        h_eff=observation.h_eff,
    )
    direct_gate = direct_gate_window(candidate, psf)
    return {
        "candidate": candidate,
        "point_spread": psf,
        "direct_gate": direct_gate,
        "nominal_offsets": nominal_offsets,
        "calibration": calibration,
        "noise_full": noise_full,
    }


def _plan_for(
    *,
    config: PilotConfig,
    choices: ProcessingChoices,
    base: dict[str, Any],
    world: str,
    method: str,
    sigma_m: float,
    twin_offsets: dict[str, float],
    direct_weights: torch.Tensor,
    timing: dict[str, float],
) -> tuple[WallPlan, AngularDelayMap, AngularDelayMap]:
    candidate = base["candidate"]
    psf = base["point_spread"]
    if method == "exact_oracle":
        offsets = base["nominal_offsets"]
        plan_sigma = 0.0
        beam_mode = "exact"
    else:
        offsets = twin_offsets
        plan_sigma = 0.0 if method == "imperfect_narrow" else float(sigma_m)
        beam_mode = "exact" if method == "imperfect_narrow" else "robust"
    sigma_range = max(
        0.5 * config.speed_of_light * candidate.delay_sigma_s,
        float(
            config.get(
                "phase6", "diagnostic", "candidate_range_uncertainty_floor_m"
            )
        ),
    )
    plan_input = PlannerInput(
        candidate=candidate,
        twin_positive_wall_m=float(offsets["positive"]),
        twin_negative_wall_m=float(offsets["negative"]),
        sigma_map_m=float(plan_sigma),
        sigma_candidate_range_m=float(sigma_range),
        gnb_position_m=tuple(
            float(value) for value in config.get("physical", "gnb_position_m")
        ),
        tx_power_w=dbm_to_watt(
            config.get("waveform", "nominal_total_power_dbm")
        ),
        robust_num_samples=int(config.get("beam", "num_robust_samples")),
        min_departure_separation_deg=float(
            config.get("phase6", "processing", "min_departure_separation_deg")
        ),
        speed_of_light_m_per_s=config.speed_of_light,
    )
    start = _now()
    plan = plan_wall_action(
        plan_input,
        choices=choices,
        point_spread=psf,
        config=config,
        beam_mode=beam_mode,
    )
    timing["planning_s"] = timing.get("planning_s", 0.0) + _now() - start
    if plan.beam is None or plan.beam.weights is None:
        raise Phase7Error(f"{method}: plan has no usable wall beam")
    twin = _twin_paths(config, candidate, offsets, timing)
    direct_twin = _twin_map(
        config=config,
        choices=choices,
        weights=direct_weights,
        twin_paths=twin,
        timing=timing,
    )
    wall_twin = _twin_map(
        config=config,
        choices=choices,
        weights=plan.beam.weights,
        twin_paths=twin,
        timing=timing,
    )
    return plan, direct_twin, wall_twin


def _null_depth(
    *,
    config: PilotConfig,
    weights: torch.Tensor,
    true_relay_deg: float,
) -> float | None:
    array = ArrayGeometry.from_config(config)
    peak = max(
        beam_gain(float(angle), _weights_beam(weights), array)
        for angle in torch.linspace(-60.0, 60.0, 481)
    )
    relay_gain = beam_gain(
        true_relay_deg, _weights_beam(weights), array
    )
    if peak <= 0.0 or relay_gain <= 0.0:
        return None
    return float(10.0 * math.log10(relay_gain / peak))


def _weights_beam(weights: torch.Tensor) -> Any:
    from .beams import BeamResult

    return BeamResult(weights=weights, status="ok", phi_deg=0.0)


def _scan_looks(
    *,
    config: PilotConfig,
    choices: ProcessingChoices,
    base: dict[str, Any],
    world: str,
    links: dict[str, tuple[LinkPath, ...]],
    calibration: Any,
    look_plan: Sequence[tuple[float, int]],
    probe_seed: int,
    noise_seed: int,
    relay_noise_seed: int,
    scan_span_s: float,
    noise_by_symbols: dict[int, float],
    timing: dict[str, float],
) -> dict[str, Any]:
    array = ArrayGeometry.from_config(config)
    power = dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))
    direct_gate = base["direct_gate"]
    psf = base["point_spread"]
    attempts: list[dict[str, Any]] = []
    total_symbols = 0
    total_energy_j = 0.0
    for angle, num_symbols in look_plan:
        profile = replace(choices.profile, num_symbols=int(num_symbols))
        local_choices = replace(choices, profile=profile)
        symbols = qpsk_symbols(profile, int(probe_seed))
        weights = direct_beam(float(angle), power, array).weights
        start = _now()
        observation = _observe(
            config=config,
            world=world,
            beam_weights=weights,
            symbols=symbols,
            profile=profile,
            target_links=links["target"],
            background_links=links["background"],
            downlink_links=links["downlink"],
            uplink_links=links["uplink"],
            calibration=calibration,
            probe_seed=int(probe_seed),
            noise_seed=int(noise_seed),
            relay_noise_seed=int(relay_noise_seed),
        )
        timing["propagation_s"] = (
            timing.get("propagation_s", 0.0) + _now() - start
        )
        angular = _map(
            observation,
            local_choices,
            float(noise_by_symbols[int(num_symbols)]),
            timing,
        )
        energy = gate_energy(
            angular,
            delay_min_s=direct_gate.delay_max_s,
            delay_max_s=direct_gate.delay_max_s + float(scan_span_s),
            aoa_min_deg=float(angle) - psf.aoa_width_deg,
            aoa_max_deg=float(angle) + psf.aoa_width_deg,
        )
        snr = (
            energy.net_energy_w / energy.noise_w
            if energy.noise_w > 0.0
            else float("nan")
        )
        attempts.append(
            {
                "angle_deg": float(angle),
                "symbols": int(num_symbols),
                "E_W": energy.energy_w,
                "N_W": energy.noise_w,
                "net_energy_w": energy.net_energy_w,
                "cells": energy.cells,
                "excess_snr": snr,
                "weights": weights,
            }
        )
        total_symbols += int(num_symbols)
        total_energy_j += power * profile.symbol_duration_s * int(num_symbols)
    best = max(attempts, key=lambda row: float(row["excess_snr"]))
    return {
        "selected": best,
        "attempts": attempts,
        "total_symbols": total_symbols,
        "total_energy_j": total_energy_j,
    }


def _mapfree_look_plan(
    method: str,
    *,
    config: PilotConfig,
    beam_seed: int,
) -> Sequence[tuple[float, int]]:
    values = config.section("beam", "map_free")
    allowed = sorted(float(value) for value in values["angles_deg"])
    if method == "mapfree_two_beam":
        angle = float(random.Random(int(beam_seed)).choice(allowed))
        return [(angle, int(values["total_extra_symbols"]))]
    raise Phase7Error(f"unknown map-free method {method!r}")


def _coarse_to_fine(
    *,
    config: PilotConfig,
    choices: ProcessingChoices,
    base: dict[str, Any],
    world: str,
    links: dict[str, tuple[LinkPath, ...]],
    calibration: Any,
    probe_seed: int,
    noise_seed: int,
    relay_noise_seed: int,
    noise_by_symbols: dict[int, float],
    timing: dict[str, float],
) -> dict[str, Any]:
    values = config.section("beam", "map_free")
    coarse_angles = [float(value) for value in values["coarse_exploration_angles_deg"]]
    coarse_symbols = int(values["coarse_symbols"])
    refine_symbols = int(values["refinement_symbols"])
    step = float(values["refinement_step_deg"])
    scan_span = float(config.get("phase7", "scan", "post_direct_span_ns")) * 1e-9
    coarse_plan = [(angle, coarse_symbols) for angle in coarse_angles]
    coarse = _scan_looks(
        config=config,
        choices=choices,
        base=base,
        world=world,
        links=links,
        calibration=calibration,
        look_plan=coarse_plan,
        probe_seed=probe_seed,
        noise_seed=noise_seed,
        relay_noise_seed=relay_noise_seed,
        scan_span_s=scan_span,
        noise_by_symbols=noise_by_symbols,
        timing=timing,
    )
    best_angle = float(coarse["selected"]["angle_deg"])
    allowed = sorted(float(value) for value in values["angles_deg"])
    refine_angles = [best_angle - step, best_angle + step]
    refine_angles = [angle for angle in refine_angles if angle in allowed]
    while len(refine_angles) < 2:
        candidates = sorted(allowed, key=lambda angle: abs(angle - best_angle))
        for angle in candidates:
            if angle not in refine_angles and angle != best_angle:
                refine_angles.append(angle)
                break
        else:
            break
    refine_plan = [(angle, refine_symbols) for angle in refine_angles]
    refined = _scan_looks(
        config=config,
        choices=choices,
        base=base,
        world=world,
        links=links,
        calibration=calibration,
        look_plan=refine_plan,
        probe_seed=probe_seed,
        noise_seed=noise_seed,
        relay_noise_seed=relay_noise_seed,
        scan_span_s=scan_span,
        noise_by_symbols=noise_by_symbols,
        timing=timing,
    )
    attempts = coarse["attempts"] + refined["attempts"]
    best = max(attempts, key=lambda row: float(row["excess_snr"]))
    return {
        "selected": best,
        "attempts": attempts,
        "coarse_attempts": coarse["attempts"],
        "refined_attempts": refined["attempts"],
        "total_symbols": coarse["total_symbols"] + refined["total_symbols"],
        "total_energy_j": coarse["total_energy_j"] + refined["total_energy_j"],
    }


def _row_base(
    *,
    config: PilotConfig,
    world: str,
    method: str,
    sigma_m: float,
    record_index: int,
    seeds: dict[str, Any],
    twin_idx: int,
    twin_offsets: dict[str, float],
    base: dict[str, Any],
) -> dict[str, Any]:
    candidate = base["candidate"]
    psf = base["point_spread"]
    direct_gate = base["direct_gate"]
    true_positive = float(config.get("physical", "walls", "positive", "offset_m"))
    true_negative = float(config.get("physical", "walls", "negative", "offset_m"))
    return {
        "geometry_id": "smoke_nominal",
        "split": "smoke",
        "world": world,
        "method": method,
        "sigma_map_m": float(sigma_m),
        "record_index": int(record_index),
        "twin_index": int(twin_idx),
        "probe_seed": int(seeds["streams"]["probe_symbols"][record_index]),
        "receiver_noise_seed": int(seeds["streams"]["receiver_noise"][record_index]),
        "relay_noise_seed": int(seeds["streams"]["relay_noise"][record_index]),
        "beam_selection_seed": int(seeds["streams"]["beam_selection"][record_index]),
        "power_dbm": float(config.get("waveform", "nominal_total_power_dbm")),
        "bandwidth_profile": str(config.get("rf", "baseline_bandwidth_profile")),
        "relay_setting": "matched_fixed_relay_physical_noise",
        "acquisition_status": candidate.status,
        "candidate_delay_ns": candidate.delay_s * 1e9,
        "candidate_aoa_deg": candidate.azimuth_deg,
        "candidate_range_m": candidate.range_m,
        "point_spread_delay_ns": psf.delay_width_s * 1e9,
        "point_spread_aoa_deg": psf.aoa_width_deg,
        "direct_gate_delay_min_ns": direct_gate.delay_min_s * 1e9,
        "direct_gate_delay_max_ns": direct_gate.delay_max_s * 1e9,
        "direct_gate_aoa_min_deg": direct_gate.aoa_min_deg,
        "direct_gate_aoa_max_deg": direct_gate.aoa_max_deg,
        "true_positive_wall_m": true_positive,
        "true_negative_wall_m": true_negative,
        "twin_positive_wall_m": float(twin_offsets["positive"]),
        "twin_negative_wall_m": float(twin_offsets["negative"]),
        "twin_map_error_positive_m": float(twin_offsets["positive"]) - true_positive,
        "twin_map_error_negative_m": float(twin_offsets["negative"]) - true_negative,
        "notes": "shared base acquisition per world; plans use this candidate",
    }


def _finish_row(
    row: dict[str, Any],
    *,
    E_D: float,
    N_D: float,
    E_W: float,
    N_W: float,
    S: float,
    S_available: bool,
    wall_excess_snr: float,
    wall_excess_snr_available: bool,
    status: str,
    reason: str,
    path_availability: str,
    plan: WallPlan | None,
    selected_weights: torch.Tensor | None,
    config: PilotConfig,
    true_relay_deg: float,
    runtime_s: float,
    total_symbols: int,
    total_energy_j: float,
    discovery: Any = None,
) -> dict[str, Any]:
    row.update(
        {
            "E_D": float(E_D),
            "N_D": float(N_D),
            "E_W": float(E_W),
            "N_W": float(N_W),
            "S": float(S),
            "S_available": bool(S_available),
            "wall_excess_snr": float(wall_excess_snr),
            "wall_excess_snr_available": bool(wall_excess_snr_available),
            "status": status,
            "status_reason": reason,
            "path_availability": path_availability,
            "beam_phi_deg": (
                float(plan.beam.phi_deg)
                if plan is not None and plan.beam is not None
                else None
            ),
            "wall_gate_delay_min_ns": (
                plan.wall_gate.delay_min_s * 1e9
                if plan is not None and plan.wall_gate is not None
                else None
            ),
            "wall_gate_delay_max_ns": (
                plan.wall_gate.delay_max_s * 1e9
                if plan is not None and plan.wall_gate is not None
                else None
            ),
            "null_depth_db": (
                _null_depth(
                    config=config,
                    weights=selected_weights,
                    true_relay_deg=true_relay_deg,
                )
                if selected_weights is not None
                else None
            ),
            "trial_runtime_s": float(runtime_s),
            "energy_symbols": int(total_symbols),
            "transmit_energy_j": float(total_energy_j),
            "discovery": _jsonable(discovery),
        }
    )
    return row


def method_is_mapfree(method: str) -> bool:
    return method in {"mapfree_two_beam", "mapfree_coarse_to_fine"}


def _nvml_snapshot() -> dict[str, Any]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        pynvml.nvmlShutdown()
        return {
            "name": name.decode() if isinstance(name, bytes) else str(name),
            "total_bytes": int(memory.total),
            "used_bytes": int(memory.used),
            "free_bytes": int(memory.free),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_smoke(
    config_path: str | Path,
    output_path: str | Path = "results/phase7",
    *,
    require_h100: bool = True,
    record_limit: int | None = None,
) -> int:
    config = PilotConfig.load(config_path)
    _require_gpu(config)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    environment_path = output / "environment.json"
    from .environment import run_preflight

    if run_preflight(require_h100=require_h100, output_path=output) != 0:
        raise Phase7Error("preflight failed")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    seeds = _seed_manifest(config)
    (output / "seed_manifest.json").write_text(
        json.dumps(seeds, indent=2), encoding="utf-8"
    )
    prepared_records = int(
        record_limit
        if record_limit is not None
        else config.get("statistics", "smoke", "noise_records_per_case")
    )
    resolved = dict(config.data)
    resolved["phase7_runtime"] = {
        "stage": "smoke",
        "methods": list(METHODS),
        "records_per_case": prepared_records,
        "record_limit": record_limit,
        "twin_realizations_per_sigma": len(seeds["streams"]["twin_error"][0]),
        "thresholds": None,
    }
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    dependency_lines = [
        f"python={__import__('sys').version.split()[0]}",
        f"torch={torch.__version__}",
        f"sionna={__import__('sionna').__version__}",
        f"mitsuba={__import__('mitsuba').__version__}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
    ]
    (output / "dependency_versions.txt").write_text(
        "\n".join(dependency_lines) + "\n", encoding="utf-8"
    )
    timing: dict[str, float] = {}
    run_start = _now()
    nvml_start = _nvml_snapshot()
    choices = ProcessingChoices.from_config(config)
    noise_full = noise_calibration(
        choices=choices,
        config=config,
        records=int(config.get("phase6", "processing", "noise_calibration_records")),
        seed=int(config.get("phase6", "processing", "noise_calibration_seed")),
    )
    noise_by_symbols = {
        count: noise_calibration(
            choices=replace(
                choices, profile=replace(choices.profile, num_symbols=count)
            ),
            config=config,
            records=4,
            seed=7900 + count,
        )
        for count in (8, 16, 32, choices.profile.num_symbols)
    }
    start = _now()
    true_paths = _true_paths(config)
    timing["true_scene_solve_s"] = _now() - start
    links = _true_links(true_paths)
    calibration = _calibrate(config, true_paths)
    true_relay_deg = horizontal_range_and_azimuth(
        config.get("physical", "relay_position_m"),
        config.get("physical", "gnb_position_m"),
    )[1]
    direct_weights = direct_beam(
        float(config.get("experiment_a", "reference_beam_phi_deg")),
        dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
        ArrayGeometry.from_config(config),
    ).weights
    base_worlds = {
        world: _base_world(
            config=config,
            choices=choices,
            world=world,
            direct_weights=direct_weights,
            links=links,
            calibration=calibration,
            noise_full=noise_full,
            timing=timing,
            seeds=seeds,
        )
        for world in ("target", "replay")
    }
    sigmas = [float(value) for value in config.get("statistics", "smoke", "sigma_map_m")]
    worlds = [str(value) for value in config.get("statistics", "smoke", "worlds")]
    record_count = int(record_limit) if record_limit is not None else int(
        config.get("statistics", "smoke", "noise_records_per_case")
    )
    plan_cache: dict[Any, tuple[WallPlan, AngularDelayMap, AngularDelayMap]] = {}
    trials: list[dict[str, Any]] = []
    representative: dict[str, Any] = {}
    for sigma_m in sigmas:
        for world in worlds:
            base = base_worlds[world]
            offset_table = [
                _twin_offsets(config, sigma_m, seed)
                for seed in seeds["streams"]["twin_error"][sigmas.index(sigma_m)]
            ]
            for record_index in range(record_count):
                probe_seed = int(seeds["streams"]["probe_symbols"][record_index])
                noise_seed = int(seeds["streams"]["receiver_noise"][record_index])
                relay_seed = int(seeds["streams"]["relay_noise"][record_index])
                twin_idx = record_index % len(offset_table)
                twin_offsets = offset_table[twin_idx]
                profile = choices.profile
                symbols = qpsk_symbols(profile, probe_seed)
                start = _now()
                direct_observation = _observe(
                    config=config,
                    world=world,
                    beam_weights=direct_weights,
                    symbols=symbols,
                    profile=profile,
                    target_links=links["target"],
                    background_links=links["background"],
                    downlink_links=links["downlink"],
                    uplink_links=links["uplink"],
                    calibration=calibration,
                    probe_seed=probe_seed,
                    noise_seed=noise_seed,
                    relay_noise_seed=relay_seed,
                )
                timing["propagation_s"] = (
                    timing.get("propagation_s", 0.0) + _now() - start
                )
                direct_map = _map(direct_observation, choices, noise_full, timing)
                direct_energy = gate_energy(
                    direct_map,
                    delay_min_s=base["direct_gate"].delay_min_s,
                    delay_max_s=base["direct_gate"].delay_max_s,
                    aoa_min_deg=base["direct_gate"].aoa_min_deg,
                    aoa_max_deg=base["direct_gate"].aoa_max_deg,
                )
                if record_index == 0:
                    representative[f"{world}_direct_h_eff"] = direct_observation.h_eff
                    if world == "target":
                        representative["direct_weights"] = direct_weights
                for method in METHODS:
                    method_start = _now()
                    row = _row_base(
                        config=config,
                        world=world,
                        method=method,
                        sigma_m=sigma_m,
                        record_index=record_index,
                        seeds=seeds,
                        twin_idx=twin_idx,
                        twin_offsets=twin_offsets,
                        base=base,
                    )
                    if method == "direct_only":
                        row = _finish_row(
                            row,
                            E_D=direct_energy.energy_w,
                            N_D=direct_energy.noise_w,
                            E_W=float("nan"),
                            N_W=float("nan"),
                            S=float("nan"),
                            S_available=False,
                            wall_excess_snr=float("nan"),
                            wall_excess_snr_available=False,
                            status="NO_WALL_PROBE",
                            reason="direct_only",
                            path_availability="NOT_OBSERVABLE",
                            plan=None,
                            selected_weights=direct_weights,
                            config=config,
                            true_relay_deg=true_relay_deg,
                            runtime_s=_now() - method_start,
                            total_symbols=profile.num_symbols,
                            total_energy_j=(
                                dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))
                                * profile.symbol_duration_s
                                * profile.num_symbols
                            ),
                        )
                    elif method_is_mapfree(method):
                        if method == "mapfree_two_beam":
                            look_plan = _mapfree_look_plan(
                                method,
                                config=config,
                                beam_seed=int(seeds["streams"]["beam_selection"][record_index]),
                            )
                            scan = _scan_looks(
                                config=config,
                                choices=choices,
                                base=base,
                                world=world,
                                links=links,
                                calibration=calibration,
                                look_plan=look_plan,
                                probe_seed=probe_seed,
                                noise_seed=noise_seed,
                                relay_noise_seed=relay_seed,
                                scan_span_s=float(
                                    config.get("phase7", "scan", "post_direct_span_ns")
                                )
                                * 1e-9,
                                noise_by_symbols=noise_by_symbols,
                                timing=timing,
                            )
                        else:
                            scan = _coarse_to_fine(
                                config=config,
                                choices=choices,
                                base=base,
                                world=world,
                                links=links,
                                calibration=calibration,
                                probe_seed=probe_seed,
                                noise_seed=noise_seed,
                                relay_noise_seed=relay_seed,
                                noise_by_symbols=noise_by_symbols,
                                timing=timing,
                            )
                        best = scan["selected"]
                        snr = float(best["excess_snr"])
                        row = _finish_row(
                            row,
                            E_D=direct_energy.energy_w,
                            N_D=direct_energy.noise_w,
                            E_W=float(best["E_W"]),
                            N_W=float(best["N_W"]),
                            S=float("nan"),
                            S_available=False,
                            wall_excess_snr=snr,
                            wall_excess_snr_available=bool(
                                math.isfinite(snr) and float(best["N_W"]) > 0.0
                            ),
                            status="UNRESOLVED",
                            reason="mapfree_scan_no_map_reference",
                            path_availability=(
                                "AVAILABLE" if snr > 0.0 else "NOT_OBSERVABLE"
                            ),
                            plan=None,
                            selected_weights=best["weights"],
                            config=config,
                            true_relay_deg=true_relay_deg,
                            runtime_s=_now() - method_start,
                            total_symbols=profile.num_symbols + int(scan["total_symbols"]),
                            total_energy_j=(
                                dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))
                                * profile.symbol_duration_s
                                * (profile.num_symbols + int(scan["total_symbols"]))
                            ),
                            discovery={
                                "scan_kind": method,
                                "selected_angle_deg": best["angle_deg"],
                                "selected_symbols": best["symbols"],
                                "attempts": [
                                    {
                                        key: value
                                        for key, value in attempt.items()
                                        if key != "weights"
                                    }
                                    for attempt in scan["attempts"]
                                ],
                                "total_symbols": scan["total_symbols"],
                            },
                        )
                    else:
                        key = (world, method, sigma_m, tuple(round(value, 9) for value in twin_offsets.values()))
                        if key not in plan_cache:
                            plan_cache[key] = _plan_for(
                                config=config,
                                choices=choices,
                                base=base,
                                world=world,
                                method=method,
                                sigma_m=sigma_m,
                                twin_offsets=twin_offsets,
                                direct_weights=direct_weights,
                                timing=timing,
                            )
                        plan, direct_twin, wall_twin = plan_cache[key]
                        wall_observation = _observe(
                            config=config,
                            world=world,
                            beam_weights=plan.beam.weights,
                            symbols=symbols,
                            profile=profile,
                            target_links=links["target"],
                            background_links=links["background"],
                            downlink_links=links["downlink"],
                            uplink_links=links["uplink"],
                            calibration=calibration,
                            probe_seed=probe_seed,
                            noise_seed=noise_seed,
                            relay_noise_seed=relay_seed,
                        )
                        wall_map = _map(wall_observation, choices, noise_full, timing)
                        if record_index == 0 and method in {"exact_oracle", "imperfect_robust"}:
                            representative[f"{world}_{method}_wall_h_eff"] = wall_observation.h_eff
                            representative[f"{world}_{method}_weights"] = plan.beam.weights
                        evidence = evidence_score(
                            direct_measured=direct_map,
                            wall_measured=wall_map,
                            direct_twin=direct_twin,
                            wall_twin=wall_twin,
                            plan=plan,
                            config=config,
                        )
                        data = evidence.as_dict()
                        excess = (
                            (data["E_W"] - data["N_W"]) / data["N_W"]
                            if data["N_W"] > 0.0
                            else float("nan")
                        )
                        row = _finish_row(
                            row,
                            E_D=data["E_D"],
                            N_D=data["N_D"],
                            E_W=data["E_W"],
                            N_W=data["N_W"],
                            S=data["score"],
                            S_available=bool(math.isfinite(data["score"])),
                            wall_excess_snr=excess,
                            wall_excess_snr_available=bool(
                                math.isfinite(excess) and data["N_W"] > 0.0
                            ),
                            status=data["status"],
                            reason=data["reason"],
                            path_availability=plan.status,
                            plan=plan,
                            selected_weights=plan.beam.weights,
                            config=config,
                            true_relay_deg=true_relay_deg,
                            runtime_s=_now() - method_start,
                            total_symbols=profile.num_symbols + profile.num_symbols,
                            total_energy_j=(
                                dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))
                                * profile.symbol_duration_s
                                * (profile.num_symbols + profile.num_symbols)
                            ),
                            discovery=None,
                        )
                    trials.append(row)
    timing["end_to_end_s"] = _now() - run_start
    timing["nvml_start"] = nvml_start
    timing["nvml_end"] = _nvml_snapshot()
    timing["pytorch_peak_allocated_bytes"] = int(
        torch.cuda.max_memory_allocated()
    )
    timing["pytorch_peak_reserved_bytes"] = int(
        torch.cuda.max_memory_reserved()
    )
    timing["warm_steady_state_s"] = float(
        np.median([row["trial_runtime_s"] for row in trials[10:]])
        if len(trials) > 10
        else np.median([row["trial_runtime_s"] for row in trials])
    )
    timing["record_limit"] = record_count
    frame = pd.DataFrame(trials)
    expected = len(METHODS) * len(sigmas) * len(worlds) * record_count
    from .metrics import integrity_checks, summarize_trials

    checks = integrity_checks(
        frame,
        expected_rows=expected,
        methods=METHODS,
        sigmas=sigmas,
        worlds=worlds,
        records=record_count,
    )
    write_start = _now()
    frame.to_parquet(output / "trials.parquet", index=False)
    diagnostics = []
    for link_name, path_set in true_paths.items():
        for record in path_records(path_set, scene_label=link_name):
            row = dict(record)
            row["link"] = link_name
            row["world"] = "true"
            diagnostics.append(row)
    pd.DataFrame(diagnostics).to_parquet(
        output / "paths_diagnostic.parquet", index=False
    )
    summary = summarize_trials(frame)
    summary.to_csv(output / "aggregate_metrics.csv", index=False)
    geometry_reference = {
        "geometry_id": "smoke_nominal",
        "gnb_position_m": config.get("physical", "gnb_position_m"),
        "candidate_position_m": config.get("physical", "candidate_position_m"),
        "relay_position_m": config.get("physical", "relay_position_m"),
        "true_wall_offsets_m": {
            "positive": float(config.get("physical", "walls", "positive", "offset_m")),
            "negative": float(config.get("physical", "walls", "negative", "offset_m")),
        },
        "twin_wall_offsets_m": {
            f"{sigma:g}": [
                _twin_offsets(config, float(sigma), seed)
                for seed in seeds["streams"]["twin_error"][sigmas.index(float(sigma))]
            ]
            for sigma in sigmas
        },
        "true_path_counts": {
            name: int(path_set.path_count) for name, path_set in true_paths.items()
        },
        "relay_azimuth_deg": true_relay_deg,
        "base_candidate": {
            world: base["candidate"].as_dict() for world, base in base_worlds.items()
        },
        "point_spread": base_worlds["target"]["point_spread"].as_dict(),
    }
    (output / "geometry_reference.json").write_text(
        json.dumps(_jsonable(geometry_reference), indent=2), encoding="utf-8"
    )
    arrays = {
        key: value.detach().cpu().numpy()
        for key, value in representative.items()
        if isinstance(value, torch.Tensor)
    }
    np.savez_compressed(output / "representative_waveforms.npz", **arrays)
    timing["serialization_s"] = _now() - write_start
    (output / "timing.json").write_text(
        json.dumps(_jsonable(timing), indent=2), encoding="utf-8"
    )
    checks["timing_complete"] = all(
        key in timing
        for key in (
            "cold_jit_s",
            "true_scene_solve_s",
            "twin_scene_solve_s",
            "planning_s",
            "propagation_s",
            "receiver_s",
            "serialization_s",
            "warm_steady_state_s",
            "end_to_end_s",
        )
    )
    decision = {
        "stage": "smoke",
        "status": "SMOKE_COMPLETE" if all(
            value for key, value in checks.items() if isinstance(value, bool)
        ) else "FAIL",
        "implementation_validity": checks,
        "scientific_decision": "not issued at smoke stage",
        "thresholds": None,
        "supported_labels": 0,
        "environment": environment.get("status"),
        "methods": list(METHODS),
        "sigmas": sigmas,
        "worlds": worlds,
    }
    (output / "scientific_decision.json").write_text(
        json.dumps(_jsonable(decision), indent=2), encoding="utf-8"
    )
    from .report import build_report

    build_report(output, output / "report")
    print(
        "phase 7 smoke complete: "
        f"rows={len(frame)} expected={expected} "
        f"integrity={checks['row_count_matches'] and checks['cells_complete']} "
        f"status={decision['status']}"
    )
    print(
        f"wrote {output / 'trials.parquet'}, {output / 'report'}, "
        f"{output / 'scientific_decision.json'}"
    )
    return 0 if decision["status"] == "SMOKE_COMPLETE" else 2
