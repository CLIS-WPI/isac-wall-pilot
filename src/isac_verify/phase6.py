"""Phase 6 nominal measured acquisition, twin planning, and gates.

This module is the evaluator-side orchestration.  It generates the target and
replay observations from actual Sionna paths, calls the information-restricted
planner, and records evaluator-only diagnostics separately from planner inputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sionna.phy.channel import subcarrier_frequencies

from .beams import (
    ArrayGeometry,
    array_response,
    beam_gain,
    beampattern,
    dbm_to_watt,
    direct_beam,
    steering_vector,
)
from .config import PilotConfig
from .geometry import horizontal_range_and_azimuth
from .receiver import (
    AngularDelayMap,
    CandidateEstimate,
    ProcessingChoices,
    angular_delay_map,
    delay_doppler_map,
    estimate_candidate,
    gate_energy,
    guard_noise_power,
    link_paths,
    make_observation,
    measure_point_spread,
    noise_calibration,
)
from .replay import calibrate_matched_relay
from .verification import (
    EvidenceScore,
    PlannerInput,
    WallPlan,
    evidence_score,
    plan_wall_action,
    planner_field_names,
)
from .wall_paths import (
    blocker_spec,
    solve_background_paths,
    solve_target_paths,
)
from .waveform import qpsk_symbols


class Phase6Error(RuntimeError):
    """Raised when the nominal Phase 6 diagnostic cannot be formed."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _d9(config: PilotConfig, name: str) -> int:
    return int(config.get("scene", "d9_depths", name))


def _true_paths(config: PilotConfig) -> dict[str, Any]:
    target = solve_target_paths(config, max_depth=_d9(config, "target"))
    background = solve_background_paths(
        config,
        direction="monostatic",
        max_depth=_d9(config, "background"),
    )
    downlink = solve_background_paths(
        config,
        direction="downlink",
        max_depth=_d9(config, "downlink"),
    )
    uplink = solve_background_paths(
        config,
        direction="uplink",
        max_depth=_d9(config, "uplink"),
    )
    return {
        "target": target,
        "background": background,
        "downlink": downlink,
        "uplink": uplink,
    }


def _calibrate(config: PilotConfig, paths: dict[str, Any]) -> Any:
    target = link_paths(
        paths["target"], label="target", families=("direct",)
    )[0]
    downlink = link_paths(
        paths["downlink"], label="downlink", families=("los",)
    )[0]
    uplink = link_paths(
        paths["uplink"], label="uplink", families=("los",)
    )[0]
    array = ArrayGeometry.from_config(config)
    reference_phi = float(
        config.get("experiment_a", "reference_beam_phi_deg")
    )
    reference = direct_beam(
        reference_phi,
        dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
        array,
    )
    rx_values = config.section("arrays", "gnb_rx")
    rx_array = ArrayGeometry(
        num_elements=int(rx_values["num_elements"]),
        spacing_wavelengths=float(rx_values["spacing_wavelengths"]),
        wavelength_m=config.wavelength_m,
    )
    rx_reference = array_response(
        reference_phi, rx_array, dtype=torch.complex64
    )
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
        reference_beam_phi_deg=reference_phi,
    )


def _twin_target_paths(
    config: PilotConfig,
    candidate: CandidateEstimate,
    *,
    positive_wall_m: float,
    negative_wall_m: float,
) -> tuple[Any, ...]:
    twin = solve_target_paths(
        config,
        wall_offsets_m={
            "positive": positive_wall_m,
            "negative": negative_wall_m,
        },
        max_depth=_d9(config, "target"),
        target_position_m=candidate.position_m,
    )
    return link_paths(twin, label="twin_candidate")


def _all_noise_symbols(profile: Any) -> torch.Tensor:
    return torch.ones(
        (profile.num_symbols, profile.fft_size),
        device="cuda",
        dtype=torch.complex64,
    )


def _point_spread(
    config: PilotConfig,
    choices: ProcessingChoices,
    candidate: CandidateEstimate,
    twin_paths: Sequence[Any],
) -> Any:
    direct = [path for path in twin_paths if path.family == "direct"]
    if not direct:
        raise Phase6Error("twin candidate has no direct path")
    observation = make_observation(
        world="background",
        beam_name="direct",
        beam_weights=direct_beam(
            float(config.get("experiment_a", "reference_beam_phi_deg")),
            dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
            ArrayGeometry.from_config(config),
        ).weights,
        symbols=_all_noise_symbols(choices.profile),
        target_paths=direct,
        profile=choices.profile,
        config=config,
        noise_seed=0,
        relay_noise_seed=0,
        noiseless=True,
    )
    dd = delay_doppler_map(observation.h_eff, choices)
    angular = angular_delay_map(dd, choices, noise_per_cell_w=0.0)
    return measure_point_spread(angular, choices=choices)


def _observation_map(
    *,
    observation: Any,
    choices: ProcessingChoices,
    noise_per_cell_w: float,
) -> AngularDelayMap:
    dd = delay_doppler_map(observation.h_eff, choices)
    return angular_delay_map(
        dd, choices, noise_per_cell_w=noise_per_cell_w
    )


def _twin_map(
    *,
    config: PilotConfig,
    choices: ProcessingChoices,
    beam_weights: torch.Tensor,
    twin_paths: Sequence[Any],
    beam_name: str,
) -> AngularDelayMap:
    observation = make_observation(
        world="background",
        beam_name=beam_name,
        beam_weights=beam_weights,
        symbols=_all_noise_symbols(choices.profile),
        target_paths=twin_paths,
        profile=choices.profile,
        config=config,
        noise_seed=0,
        relay_noise_seed=0,
        noiseless=True,
    )
    dd = delay_doppler_map(observation.h_eff, choices)
    return angular_delay_map(dd, choices, noise_per_cell_w=0.0)


def _null_depth(
    *,
    plan: WallPlan,
    config: PilotConfig,
    true_relay_deg: float,
) -> dict[str, Any]:
    if plan.beam is None or plan.beam.weights is None:
        return {}
    array = ArrayGeometry.from_config(config)
    relay_gain = beam_gain(true_relay_deg, plan.beam, array)
    scan_angles = torch.linspace(
        -60.0, 60.0, 481, device="cuda", dtype=torch.float64
    )
    peak = max(
        beam_gain(float(angle), plan.beam, array)
        for angle in scan_angles
    )
    gains = {
        f"{angle:.1f}": (
            10.0 * math.log10(beam_gain(float(angle), plan.beam, array) / peak)
            if peak > 0.0
            else None
        )
        for angle in (-45.0, 0.0, 26.565, 45.0)
    }
    return {
        "true_relay_azimuth_deg": true_relay_deg,
        "relay_direction_gain_db_relative_peak": (
            10.0 * math.log10(relay_gain / peak)
            if relay_gain > 0.0 and peak > 0.0
            else None
        ),
        "gain_db_relative_peak_by_angle": gains,
    }


def _world_measurements(
    *,
    config: PilotConfig,
    choices: ProcessingChoices,
    point_spread: Any,
    candidate: CandidateEstimate,
    plan: WallPlan,
    beam_mode: str,
    paths: dict[str, Any],
    twin_paths: Sequence[Any],
    calibration: Any,
    noise_per_cell_w: float,
    probe_seed: int,
    noise_seed: int,
    relay_noise_seed: int,
    true_relay_deg: float,
    direct_weights: torch.Tensor,
) -> dict[str, Any]:
    if plan.beam is None or plan.beam.weights is None:
        raise Phase6Error("world measurement requires a usable wall beam")
    profile = choices.profile
    symbols = qpsk_symbols(profile, probe_seed)
    target_paths = tuple(
        link_paths(paths["target"], label="target")
    )
    background_paths = tuple(
        link_paths(
            paths["background"], label="background", exclude_los=True
        )
    )
    relay_downlink = tuple(
        link_paths(paths["downlink"], label="downlink")
    )
    relay_uplink = tuple(link_paths(paths["uplink"], label="uplink"))
    result: dict[str, Any] = {}
    direct_twin = _twin_map(
        config=config,
        choices=choices,
        beam_weights=direct_weights,
        twin_paths=twin_paths,
        beam_name="twin_direct",
    )
    wall_twin = _twin_map(
        config=config,
        choices=choices,
        beam_weights=plan.beam.weights,
        twin_paths=twin_paths,
        beam_name="twin_wall",
    )
    for world in ("target", "replay"):
        world_targets = target_paths if world == "target" else ()
        direct_observation = make_observation(
            world=world,
            beam_name="direct",
            beam_weights=direct_weights,
            symbols=symbols,
            target_paths=world_targets,
            background_paths=background_paths,
            relay_downlink=relay_downlink,
            relay_uplink=relay_uplink,
            relay_gain=calibration.gain,
            relay_added_delay_s=calibration.measured_added_delay_s,
            include_relay=(world == "replay"),
            profile=profile,
            config=config,
            noise_seed=noise_seed,
            relay_noise_seed=relay_noise_seed,
            probe_seed=probe_seed,
        )
        wall_observation = make_observation(
            world=world,
            beam_name=f"{beam_mode}_wall",
            beam_weights=plan.beam.weights,
            symbols=symbols,
            target_paths=world_targets,
            background_paths=background_paths,
            relay_downlink=relay_downlink,
            relay_uplink=relay_uplink,
            relay_gain=calibration.gain,
            relay_added_delay_s=calibration.measured_added_delay_s,
            include_relay=(world == "replay"),
            profile=profile,
            config=config,
            noise_seed=noise_seed,
            relay_noise_seed=relay_noise_seed,
            probe_seed=probe_seed,
        )
        direct_map = _observation_map(
            observation=direct_observation,
            choices=choices,
            noise_per_cell_w=noise_per_cell_w,
        )
        wall_map = _observation_map(
            observation=wall_observation,
            choices=choices,
            noise_per_cell_w=noise_per_cell_w,
        )
        evidence = evidence_score(
            direct_measured=direct_map,
            wall_measured=wall_map,
            direct_twin=direct_twin,
            wall_twin=wall_twin,
            plan=plan,
            config=config,
        )
        result[world] = {
            "direct_observation": direct_observation.as_dict(),
            "wall_observation": wall_observation.as_dict(),
            "evidence": evidence.as_dict(),
            "realized_null": _null_depth(
                plan=plan,
                config=config,
                true_relay_deg=true_relay_deg,
            ),
        }
    return result


def _blocked_diagnostic(
    *,
    config: PilotConfig,
    choices: ProcessingChoices,
    plan: WallPlan,
    paths: dict[str, Any],
    calibration: Any,
    noise_per_cell_w: float,
    twin_paths: Sequence[Any],
    probe_seed: int,
    noise_seed: int,
    direct_weights: torch.Tensor,
) -> dict[str, Any]:
    if plan.beam is None or plan.beam.weights is None:
        raise Phase6Error("blocked diagnostic requires a usable wall beam")
    blocked = solve_target_paths(
        config,
        max_depth=_d9(config, "target"),
        blocker=blocker_spec(config),
    )
    blocked_paths = tuple(link_paths(blocked, label="blocked_target"))
    background_paths = tuple(
        link_paths(
            paths["background"], label="background", exclude_los=True
        )
    )
    symbols = qpsk_symbols(choices.profile, probe_seed)
    direct_observation = make_observation(
        world="blocked",
        beam_name="direct",
        beam_weights=direct_weights,
        symbols=symbols,
        target_paths=blocked_paths,
        background_paths=background_paths,
        profile=choices.profile,
        config=config,
        noise_seed=noise_seed,
        relay_noise_seed=noise_seed,
        probe_seed=probe_seed,
    )
    wall_observation = make_observation(
        world="blocked",
        beam_name="exact_wall",
        beam_weights=plan.beam.weights,
        symbols=symbols,
        target_paths=blocked_paths,
        background_paths=background_paths,
        profile=choices.profile,
        config=config,
        noise_seed=noise_seed,
        relay_noise_seed=noise_seed,
        probe_seed=probe_seed,
    )
    direct_map = _observation_map(
        observation=direct_observation,
        choices=choices,
        noise_per_cell_w=noise_per_cell_w,
    )
    wall_map = _observation_map(
        observation=wall_observation,
        choices=choices,
        noise_per_cell_w=noise_per_cell_w,
    )
    twin_direct = _twin_map(
        config=config,
        choices=choices,
        beam_weights=direct_weights,
        twin_paths=twin_paths,
        beam_name="twin_direct",
    )
    twin_wall = _twin_map(
        config=config,
        choices=choices,
        beam_weights=plan.beam.weights,
        twin_paths=twin_paths,
        beam_name="twin_wall",
    )
    evidence = evidence_score(
        direct_measured=direct_map,
        wall_measured=wall_map,
        direct_twin=twin_direct,
        wall_twin=twin_wall,
        plan=plan,
        config=config,
    )
    data = evidence.as_dict()
    data["wall_to_noise_ratio"] = (
        data["E_W"] / data["N_W"] if data["N_W"] > 0.0 else None
    )
    data["wall_present_above_3x_noise"] = (
        data["E_W"] > 3.0 * data["N_W"]
        if data["N_W"] > 0.0
        else False
    )
    data["blocked_path_count"] = len(blocked_paths)
    return data


def _guard_disjoint_from_gates(
    plan: WallPlan, choices: ProcessingChoices
) -> bool:
    guard_low, guard_high = choices.guard_delay_ns
    for gate in (plan.direct_gate, plan.wall_gate):
        if gate is None:
            return False
        gate_low = gate.delay_min_s * 1e9
        gate_high = gate.delay_max_s * 1e9
        if guard_low <= gate_high and guard_high >= gate_low:
            return False
    return True


def phase6_diagnostics(
    config: PilotConfig,
    output_path: str | Path = "results/phase6",
) -> dict[str, Any]:
    """Run the single-seed nominal diagnostic for both worlds and wall beams."""
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    choices = ProcessingChoices.from_config(config)
    values = config.section("phase6", "diagnostic")
    noise_per_cell = noise_calibration(
        choices=choices,
        config=config,
        records=int(
            config.get("phase6", "processing", "noise_calibration_records")
        ),
        seed=int(
            config.get("phase6", "processing", "noise_calibration_seed")
        ),
    )
    paths = _true_paths(config)
    calibration = _calibrate(config, paths)
    if not calibration.feasible:
        raise Phase6Error("matched relay is infeasible at the power cap")
    gnb = tuple(float(value) for value in config.get("physical", "gnb_position_m"))
    relay = config.get("physical", "relay_position_m")
    relay_range, relay_azimuth = horizontal_range_and_azimuth(relay, gnb)
    del relay_range
    probe_seed = int(values["probe_seed"])
    noise_seed = int(values["receiver_noise_seed"])
    relay_noise_seed = int(values["relay_noise_seed"])
    symbols = qpsk_symbols(choices.profile, probe_seed)
    target_paths = tuple(link_paths(paths["target"], label="target"))
    background_paths = tuple(
        link_paths(paths["background"], label="background", exclude_los=True)
    )
    relay_downlink = tuple(link_paths(paths["downlink"], label="downlink"))
    relay_uplink = tuple(link_paths(paths["uplink"], label="uplink"))
    direct = direct_beam(
        float(config.get("experiment_a", "reference_beam_phi_deg")),
        dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
        ArrayGeometry.from_config(config),
    )
    direct_observation = make_observation(
        world="target",
        beam_name="direct",
        beam_weights=direct.weights,
        symbols=symbols,
        target_paths=target_paths,
        background_paths=background_paths,
        profile=choices.profile,
        config=config,
        noise_seed=noise_seed,
        relay_noise_seed=relay_noise_seed,
        probe_seed=probe_seed,
    )
    direct_map = _observation_map(
        observation=direct_observation,
        choices=choices,
        noise_per_cell_w=noise_per_cell,
    )
    guard_power_w = guard_noise_power(direct_observation.h_eff, choices)
    guard_ratio = guard_power_w / noise_per_cell
    guard_bounds = choices.guard_ratio_bounds
    guard_ok = guard_bounds[0] <= guard_ratio <= guard_bounds[1]
    direct_candidate = estimate_candidate(
        direct_map,
        choices=choices,
        point_spread=None,  # type: ignore[arg-type]
        gnb_position_m=gnb,
        speed_of_light_m_per_s=config.speed_of_light,
    )
    if not direct_candidate.detected:
        raise Phase6Error("nominal direct acquisition failed")
    twin_paths = _twin_target_paths(
        config,
        direct_candidate,
        positive_wall_m=float(
            config.get("physical", "walls", "positive", "offset_m")
        ),
        negative_wall_m=float(
            config.get("physical", "walls", "negative", "offset_m")
        ),
    )
    point_spread = _point_spread(
        config, choices, direct_candidate, twin_paths
    )
    direct_map = _observation_map(
        observation=direct_observation,
        choices=choices,
        noise_per_cell_w=noise_per_cell,
    )
    direct_candidate = estimate_candidate(
        direct_map,
        choices=choices,
        point_spread=point_spread,
        gnb_position_m=gnb,
        speed_of_light_m_per_s=config.speed_of_light,
    )
    measured_range_sigma_m = (
        0.5
        * config.speed_of_light
        * direct_candidate.delay_sigma_s
    )
    range_floor_m = float(
        values["candidate_range_uncertainty_floor_m"]
    )
    used_range_sigma_m = max(measured_range_sigma_m, range_floor_m)
    plan_input = PlannerInput(
        candidate=direct_candidate,
        twin_positive_wall_m=float(
            config.get("physical", "walls", "positive", "offset_m")
        ),
        twin_negative_wall_m=float(
            config.get("physical", "walls", "negative", "offset_m")
        ),
        sigma_map_m=float(values["sigma_map_m"]),
        sigma_candidate_range_m=used_range_sigma_m,
        gnb_position_m=(gnb[0], gnb[1], gnb[2]),
        tx_power_w=dbm_to_watt(
            config.get("waveform", "nominal_total_power_dbm")
        ),
        robust_num_samples=int(values["num_robust_samples"]),
        min_departure_separation_deg=float(
            config.get("phase6", "processing", "min_departure_separation_deg")
        ),
        speed_of_light_m_per_s=config.speed_of_light,
    )
    plan_results = {
        mode: plan_wall_action(
            plan_input,
            choices=choices,
            point_spread=point_spread,
            config=config,
            beam_mode=mode,
        )
        for mode in ("exact", "robust")
    }
    worlds: dict[str, Any] = {}
    for mode, plan in plan_results.items():
        if not plan.available:
            worlds[mode] = {"plan": plan.as_dict(), "status": plan.status}
            continue
        measurements = _world_measurements(
            config=config,
            choices=choices,
            point_spread=point_spread,
            candidate=direct_candidate,
            plan=plan,
            beam_mode=mode,
            paths=paths,
            twin_paths=twin_paths,
            calibration=calibration,
            noise_per_cell_w=noise_per_cell,
            probe_seed=probe_seed,
            noise_seed=noise_seed,
            relay_noise_seed=relay_noise_seed,
            true_relay_deg=relay_azimuth,
            direct_weights=direct.weights,
        )
        worlds[mode] = {
            "plan": plan.as_dict(),
            "status": plan.status,
            "worlds": measurements,
        }
    blocked = None
    exact_plan = plan_results["exact"]
    if exact_plan.available:
        blocked = _blocked_diagnostic(
            config=config,
            choices=choices,
            plan=exact_plan,
            paths=paths,
            calibration=calibration,
            noise_per_cell_w=noise_per_cell,
            twin_paths=twin_paths,
            probe_seed=probe_seed,
            noise_seed=int(values["blocked_target_noise_seed"]),
            direct_weights=direct.weights,
        )
    checks = {
        "direct_acquisition": direct_candidate.detected,
        "noise_calibration_positive": noise_per_cell > 0.0,
        "guard_noise_within_bounds": guard_ok,
        "guard_disjoint_from_gates": all(
            _guard_disjoint_from_gates(plan, choices)
            for plan in plan_results.values()
            if plan.available
        ),
        "candidate_uncertainty_positive": used_range_sigma_m > 0.0,
        "point_spread_positive": (
            point_spread.delay_width_s > 0.0
            and point_spread.aoa_width_deg > 0.0
        ),
        "relay_calibration_feasible": calibration.feasible,
        "exact_plan_available": plan_results["exact"].available,
        "robust_plan_available": plan_results["robust"].available,
        "blocked_not_supported": (
            blocked is not None and blocked["status"] != "SUPPORTED"
        ),
        "no_supported_label": all(
            row["evidence"]["status"] != "SUPPORTED"
            for entry in worlds.values()
            for row in entry.get("worlds", {}).values()
        ),
        "planner_excludes_truth": _planner_truth_free(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "status": status,
        "checks": checks,
        "processing": {
            "profile": choices.profile.name,
            "delay_search_ns": list(choices.delay_search_ns),
            "aoa_search_deg": list(choices.aoa_search_deg),
            "aoa_grid_step_deg": choices.aoa_grid_step_deg,
            "detection_floor_db": choices.detection_floor_db,
            "guard_delay_ns": list(choices.guard_delay_ns),
            "guard_ratio_bounds": list(choices.guard_ratio_bounds),
            "delay_window": choices.delay_window,
            "aoa_window": choices.aoa_window,
            "delay_window_scaling": "unit RMS",
            "aoa_window_scaling": "unit L2 norm",
            "doppler_bin": "zero",
            "delay_interpolation": config.get(
                "phase6", "processing", "delay_interpolation"
            ),
            "aoa_interpolation": config.get(
                "phase6", "processing", "aoa_interpolation"
            ),
            "psf_threshold_db": choices.psf_threshold_db,
            "uncertainty_rule": config.get(
                "phase6", "processing", "uncertainty_rule"
            ),
            "noise_source": config.get(
                "phase6", "evidence", "noise_source"
            ),
            "evidence_ratio_source": config.get(
                "phase6", "evidence", "ratio_source"
            ),
            "thresholds": config.get("phase6", "evidence", "thresholds"),
            "one_probe_and_noise_seed_per_world": bool(
                values["one_probe_and_noise_seed_per_world"]
            ),
        },
        "planner_input_fields": list(planner_field_names()),
        "candidate_uncertainty": {
            "rule": config.get(
                "phase6", "processing", "uncertainty_rule"
            ),
            "delay_sigma_ns": direct_candidate.delay_sigma_s * 1e9,
            "measured_range_sigma_m": measured_range_sigma_m,
            "range_floor_m": range_floor_m,
            "used_range_sigma_m": used_range_sigma_m,
        },
        "planner_input": plan_input.as_dict(),
        "point_spread": point_spread.as_dict(),
        "noise": {
            "calibration_per_cell_w": noise_per_cell,
            "guard_median_power_w": guard_power_w,
            "guard_ratio": guard_ratio,
            "guard_ratio_bounds": list(guard_bounds),
            "guard_ok": guard_ok,
            "calibration_records": int(
                config.get(
                    "phase6", "processing", "noise_calibration_records"
                )
            ),
            "receiver_variance_w": float(
                torch.tensor(
                    noise_per_cell
                    * choices.profile.num_symbols
                    * choices.profile.fft_size
                ).item()
            ),
        },
        "acquisition": direct_candidate.as_dict(),
        "wall_beams": {
            mode: plan.as_dict() for mode, plan in plan_results.items()
        },
        "worlds": worlds,
        "blocked": blocked,
        "relay_calibration": calibration.as_dict(),
        "evaluator_only": {
            "true_candidate_position_m": list(
                config.get("physical", "candidate_position_m")
            ),
            "true_relay_position_m": list(relay),
            "true_relay_azimuth_deg": relay_azimuth,
            "true_wall_offsets_m": {
                side: float(
                    config.get("physical", "walls", side, "offset_m")
                )
                for side in ("positive", "negative")
            },
        },
        "note": "single-seed nominal diagnostic, no statistical conclusion",
    }
    (output / "phase6_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _planner_truth_free() -> bool:
    forbidden = {
        "truth",
        "label",
        "labels",
        "true_position_m",
        "true_wall_offsets_m",
        "relay_gain",
        "relay_delay_s",
        "evaluator_seed",
        "hypothesis",
    }
    return not (set(planner_field_names()) & forbidden)


def print_phase6_tables(phase6: dict[str, Any]) -> None:
    print(f"phase 6 status: {phase6['status']}")
    acquisition = phase6["acquisition"]
    print(
        "phase 6 acquisition: "
        f"detected={acquisition['detected']} "
        f"tau_hat={acquisition['delay_ns']:.9f} ns "
        f"AoA_hat={acquisition['azimuth_deg']:.6f} deg "
        f"range_hat={acquisition['range_m']:.6f} m "
        f"p_hat={acquisition['position_m']} "
        f"peak/noise={acquisition['peak_to_noise_db']:.3f} dB"
    )
    spread = phase6["point_spread"]
    print(
        "phase 6 measured point spread: "
        f"delay full={spread['delay_width_ns']:.6f} ns "
        f"half={spread['delay_half_width_ns']:.6f} ns; "
        f"AoA full={spread['aoa_width_deg']:.6f} deg "
        f"half={spread['aoa_half_width_deg']:.6f} deg"
    )
    print(
        "phase 6 noise calibration: "
        f"{phase6['noise']['calibration_per_cell_w']:.6e} W/cell "
        f"from {phase6['noise']['calibration_records']} records"
    )
    print(
        "phase 6 guard noise: "
        f"median={phase6['noise']['guard_median_power_w']:.6e} W/cell "
        f"ratio={phase6['noise']['guard_ratio']:.6f} "
        f"bounds={phase6['noise']['guard_ratio_bounds']} "
        f"ok={phase6['noise']['guard_ok']}"
    )
    uncertainty = phase6["candidate_uncertainty"]
    print(
        "phase 6 candidate uncertainty: "
        f"measured_sigma_range={uncertainty['measured_range_sigma_m']:.6e} m "
        f"floor={uncertainty['range_floor_m']:.6e} m "
        f"used={uncertainty['used_range_sigma_m']:.6e} m"
    )
    for mode, plan in phase6["wall_beams"].items():
        gate = plan["wall_gate"]
        if gate is None:
            print(f"phase 6 {mode} wall: {plan['status']}")
            continue
        print(
            f"phase 6 {mode} wall plan: status={plan['status']} "
            f"side={plan['wall_side']} "
            f"AoD={plan['predicted_aod_deg']:.6f} deg "
            f"delay={plan['predicted_delay_ns']:.9f} ns "
            f"gate_delay=[{gate['delay_min_ns']:.6f}, "
            f"{gate['delay_max_ns']:.6f}] ns "
            f"gate_AoA=[{gate['aoa_min_deg']:.6f}, "
            f"{gate['aoa_max_deg']:.6f}] deg"
        )
    print(
        "phase 6 world/beam            E_D (W)      E_W (W)      "
        "N_D (W)      N_W (W)      r_DT        S        status"
    )
    for mode, entry in phase6["worlds"].items():
        if "worlds" not in entry:
            continue
        for world, row in entry["worlds"].items():
            evidence = row["evidence"]
            print(
                f"phase 6 {world:7s}/{mode:7s} "
                f"{evidence['E_D']:.6e} {evidence['E_W']:.6e} "
                f"{evidence['N_D']:.6e} {evidence['N_W']:.6e} "
                f"{evidence['r_dt']:.6e} {evidence['score']:.6e} "
                f"{evidence['status']}"
            )
            null = row["realized_null"].get(
                "relay_direction_gain_db_relative_peak"
            )
            print(
                f"phase 6 {world:7s}/{mode:7s} relay-direction null="
                f"{null if null is None else f'{null:.6f} dB'}"
            )
    blocked = phase6.get("blocked")
    if blocked is not None:
        print(
            "phase 6 blocked genuine wall path: "
            f"E_W={blocked['E_W']:.6e} W "
            f"N_W={blocked['N_W']:.6e} W "
            f"E_W/N_W={blocked['wall_to_noise_ratio']:.6f} "
            f"above_3x_noise={blocked['wall_present_above_3x_noise']} "
            f"status={blocked['status']}"
        )
