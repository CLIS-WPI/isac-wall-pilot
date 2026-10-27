"""Twin-only wall planning, gates, and the evidence score (BRIEF §8–§10).

``PlannerInput`` is the complete information boundary.  It contains the noisy
measured candidate, the declared twin map, fixed calibration values, and the
declared uncertainty model.  Evaluator truth is not a field and cannot be
smuggled through a keyword argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import norm

from .beams import (
    ArrayGeometry,
    BeamResult,
    projected_wall_beam,
    robust_wall_beam,
    sobol_uncertainty_samples,
)
from .config import PilotConfig
from .geometry import wall_delay_s
from .receiver import (
    AngularDelayMap,
    CandidateEstimate,
    GateEnergy,
    LinkPath,
    PointSpread,
    ProcessingChoices,
    gate_energy,
    link_paths,
)
from .wall_paths import WallPathError, path_records, solve_target_paths


class VerificationError(ValueError):
    """Raised when a planner request is outside the allowed information set."""


@dataclass(frozen=True)
class PlannerInput:
    candidate: CandidateEstimate
    twin_positive_wall_m: float
    twin_negative_wall_m: float
    sigma_map_m: float
    sigma_candidate_range_m: float
    gnb_position_m: tuple[float, float, float]
    tx_power_w: float
    robust_num_samples: int
    min_departure_separation_deg: float
    speed_of_light_m_per_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.as_dict(),
            "twin_positive_wall_m": self.twin_positive_wall_m,
            "twin_negative_wall_m": self.twin_negative_wall_m,
            "sigma_map_m": self.sigma_map_m,
            "sigma_candidate_range_m": self.sigma_candidate_range_m,
            "gnb_position_m": list(self.gnb_position_m),
            "tx_power_w": self.tx_power_w,
            "robust_num_samples": self.robust_num_samples,
            "min_departure_separation_deg": self.min_departure_separation_deg,
            "speed_of_light_m_per_s": self.speed_of_light_m_per_s,
        }


@dataclass(frozen=True)
class GateWindow:
    delay_min_s: float
    delay_max_s: float
    aoa_min_deg: float
    aoa_max_deg: float
    center_delay_s: float
    center_aoa_deg: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "delay_min_ns": self.delay_min_s * 1e9,
            "delay_max_ns": self.delay_max_s * 1e9,
            "aoa_min_deg": self.aoa_min_deg,
            "aoa_max_deg": self.aoa_max_deg,
            "center_delay_ns": self.center_delay_s * 1e9,
            "center_aoa_deg": self.center_aoa_deg,
        }


@dataclass(frozen=True)
class WallPlan:
    status: str
    beam_status: str
    wall_side: str | None
    predicted_delay_s: float
    predicted_aod_deg: float
    predicted_aoa_deg: float
    predicted_amplitude: float
    direct_gate: GateWindow | None
    wall_gate: GateWindow | None
    beam: BeamResult | None
    uncertainty_samples: int
    uncertainty_description: str

    @property
    def available(self) -> bool:
        return (
            self.status == "AVAILABLE"
            and self.beam is not None
            and self.beam.usable
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "beam_status": self.beam_status,
            "wall_side": self.wall_side,
            "predicted_delay_ns": self.predicted_delay_s * 1e9,
            "predicted_aod_deg": self.predicted_aod_deg,
            "predicted_aoa_deg": self.predicted_aoa_deg,
            "predicted_amplitude": self.predicted_amplitude,
            "direct_gate": (
                None if self.direct_gate is None else self.direct_gate.as_dict()
            ),
            "wall_gate": (
                None if self.wall_gate is None else self.wall_gate.as_dict()
            ),
            "beam_phi_deg": (
                None if self.beam is None else self.beam.phi_deg
            ),
            "projected_norm": (
                None if self.beam is None else self.beam.projected_norm
            ),
            "uncertainty_samples": self.uncertainty_samples,
            "uncertainty_description": self.uncertainty_description,
        }


@dataclass(frozen=True)
class EvidenceScore:
    direct: GateEnergy
    wall: GateEnergy
    twin_direct: GateEnergy
    twin_wall: GateEnergy
    r_dt: float
    score: float
    reference_detectable: bool
    direct_reliable: bool
    status: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "E_D": self.direct.energy_w,
            "N_D": self.direct.noise_w,
            "E_D_net": self.direct.net_energy_w,
            "E_W": self.wall.energy_w,
            "N_W": self.wall.noise_w,
            "E_W_net": self.wall.net_energy_w,
            "E_D_cells": self.direct.cells,
            "E_W_cells": self.wall.cells,
            "twin_E_D": self.twin_direct.energy_w,
            "twin_E_W": self.twin_wall.energy_w,
            "r_dt": self.r_dt,
            "score": self.score,
            "reference_detectable": self.reference_detectable,
            "direct_reliable": self.direct_reliable,
            "status": self.status,
            "reason": self.reason,
        }


def planner_field_names() -> tuple[str, ...]:
    return tuple(field.name for field in fields(PlannerInput))


def classify_evidence(
    *,
    score: float,
    net_wall_w: float,
    wall_noise_w: float,
    direct_reliable: bool,
    reference_detectable: bool,
    thresholds: dict[str, Any] | None,
) -> tuple[str, str]:
    """Map a score to a label without inventing an attack posterior."""
    if not direct_reliable:
        return "UNRESOLVED", "direct_denominator_below_floor"
    if not reference_detectable or not math.isfinite(score):
        return "UNRESOLVED", "twin_reference_undetectable"
    if thresholds is None:
        return "UNRESOLVED", "thresholds_null"
    noise_sigmas = float(thresholds.get("noise_sigmas", 0.0))
    if wall_noise_w > 0.0 and net_wall_w <= noise_sigmas * wall_noise_w:
        return "UNRESOLVED", "wall_energy_consistent_with_noise"
    supported_min = float(thresholds["supported_min"])
    rejected_max = float(thresholds["rejected_max"])
    if supported_min <= rejected_max:
        raise VerificationError("test thresholds must be ordered")
    if score >= supported_min:
        return "SUPPORTED", "above_supported_threshold"
    if score <= rejected_max:
        return "REJECTED", "below_rejected_threshold"
    return "UNRESOLVED", "between_thresholds"


def _wall_side(family: str) -> str:
    if family.endswith("positive_wall"):
        return "positive"
    if family.endswith("negative_wall"):
        return "negative"
    raise VerificationError(f"cannot infer wall side from {family!r}")


def _wall_offset(plan_input: PlannerInput, side: str) -> float:
    if side == "positive":
        return plan_input.twin_positive_wall_m
    if side == "negative":
        return plan_input.twin_negative_wall_m
    raise VerificationError(f"unknown wall side {side!r}")


def _select_twin_wall_path(
    plan_input: PlannerInput, config: PilotConfig
) -> tuple[Any, str]:
    twin = solve_target_paths(
        config,
        wall_offsets_m={
            "positive": plan_input.twin_positive_wall_m,
            "negative": plan_input.twin_negative_wall_m,
        },
        max_depth=int(config.get("scene", "d9_depths", "target")),
        target_position_m=plan_input.candidate.position_m,
    )
    records = path_records(twin, scene_label="twin_candidate")
    candidates = [
        record
        for record in records
        if str(record["family"]).startswith("outgoing_wall_")
        and abs(
            float(record["aod_deg"]) - plan_input.candidate.azimuth_deg
        )
        >= plan_input.min_departure_separation_deg
    ]
    if not candidates:
        raise WallPathError("twin has no eligible outgoing-wall path")
    selected = max(candidates, key=lambda row: float(row["path_power_w"]))
    return selected, _wall_side(str(selected["family"]))


def _gate_from_quantiles(
    values: Sequence[float],
    *,
    quantiles: tuple[float, float],
    expansion: float,
    center: float,
) -> tuple[float, float]:
    low = float(np.quantile(np.asarray(values), quantiles[0]))
    high = float(np.quantile(np.asarray(values), quantiles[1]))
    if center < low:
        low = center
    if center > high:
        high = center
    return low - expansion, high + expansion


def plan_wall_action(
    plan_input: PlannerInput,
    *,
    choices: ProcessingChoices,
    point_spread: PointSpread,
    config: PilotConfig,
    beam_mode: str,
) -> WallPlan:
    """Select a twin wall path, construct its beam, and freeze both gates."""
    if not plan_input.candidate.detected:
        raise VerificationError("cannot plan from an acquisition failure")
    values = config.section("phase6", "gates")
    quantiles = tuple(
        float(value) for value in values["wall_interval_quantiles"]
    )
    if len(quantiles) != 2 or not 0.0 < quantiles[0] < quantiles[1] < 1.0:
        raise VerificationError("wall interval quantiles are invalid")
    direct_gate = GateWindow(
        delay_min_s=(
            plan_input.candidate.delay_s
            - point_spread.delay_width_s
            - 3.0 * plan_input.candidate.delay_sigma_s
        ),
        delay_max_s=(
            plan_input.candidate.delay_s
            + point_spread.delay_width_s
            + 3.0 * plan_input.candidate.delay_sigma_s
        ),
        aoa_min_deg=(
            plan_input.candidate.azimuth_deg
            - point_spread.aoa_width_deg
            - 3.0 * plan_input.candidate.azimuth_sigma_deg
        ),
        aoa_max_deg=(
            plan_input.candidate.azimuth_deg
            + point_spread.aoa_width_deg
            + 3.0 * plan_input.candidate.azimuth_sigma_deg
        ),
        center_delay_s=plan_input.candidate.delay_s,
        center_aoa_deg=plan_input.candidate.azimuth_deg,
    )
    try:
        selected, side = _select_twin_wall_path(plan_input, config)
    except WallPathError:
        return WallPlan(
            status="NOT_OBSERVABLE",
            beam_status="not_observable",
            wall_side=None,
            predicted_delay_s=float("nan"),
            predicted_aod_deg=float("nan"),
            predicted_aoa_deg=float("nan"),
            predicted_amplitude=float("nan"),
            direct_gate=direct_gate,
            wall_gate=None,
            beam=None,
            uncertainty_samples=plan_input.robust_num_samples,
            uncertainty_description="no eligible twin wall path",
        )
    delay_s = float(selected["delay_ns"]) * 1e-9
    aod_deg = float(selected["aod_deg"])
    aoa_deg = float(plan_input.candidate.azimuth_deg)
    nominal_offset = _wall_offset(plan_input, side)
    quantile_samples, description = sobol_uncertainty_samples(
        plan_input.robust_num_samples
    )
    wall_errors = norm.ppf(quantile_samples[:, 0]) * plan_input.sigma_map_m
    range_errors = (
        norm.ppf(quantile_samples[:, 1])
        * plan_input.sigma_candidate_range_m
    )
    sampled_ranges = plan_input.candidate.range_m + range_errors
    if np.any(sampled_ranges <= 0.0):
        raise VerificationError("uncertainty ensemble produced a non-positive range")
    sampled_offsets = nominal_offset + wall_errors
    sampled_delays = [
        wall_delay_s(float(radius), float(offset), plan_input.speed_of_light_m_per_s)
        for radius, offset in zip(sampled_ranges, sampled_offsets)
    ]
    delay_low, delay_high = _gate_from_quantiles(
        sampled_delays,
        quantiles=quantiles,
        expansion=(
            point_spread.delay_width_s
            + 3.0 * plan_input.candidate.delay_sigma_s
        ),
        center=delay_s,
    )
    aoa_expansion = (
        point_spread.aoa_width_deg
        + 3.0 * plan_input.candidate.azimuth_sigma_deg
    )
    wall_gate = GateWindow(
        delay_min_s=delay_low,
        delay_max_s=delay_high,
        aoa_min_deg=aoa_deg - aoa_expansion,
        aoa_max_deg=aoa_deg + aoa_expansion,
        center_delay_s=delay_s,
        center_aoa_deg=aoa_deg,
    )
    array = ArrayGeometry.from_config(config)
    if beam_mode == "exact":
        beam = projected_wall_beam(
            aod_deg,
            plan_input.candidate.azimuth_deg,
            plan_input.tx_power_w,
            array,
            min_projected_norm=float(config.get("beam", "projected_norm_floor")),
        )
    elif beam_mode == "robust":
        beam = robust_wall_beam(
            candidate_range_m=plan_input.candidate.range_m,
            wall_offset_m=nominal_offset,
            sigma_map_m=plan_input.sigma_map_m,
            sigma_candidate_range_m=plan_input.sigma_candidate_range_m,
            phi_direct_deg=plan_input.candidate.azimuth_deg,
            tx_power_w=plan_input.tx_power_w,
            array=array,
            num_samples=plan_input.robust_num_samples,
            reference_index=int(
                config.get("steering", "reference_element_index")
            ),
            min_projected_norm=float(config.get("beam", "projected_norm_floor")),
        )
    else:
        raise VerificationError(f"unknown beam mode {beam_mode!r}")
    return WallPlan(
        status="AVAILABLE" if beam.usable else "UNUSABLE_BEAM",
        beam_status=beam.status,
        wall_side=side,
        predicted_delay_s=delay_s,
        predicted_aod_deg=aod_deg,
        predicted_aoa_deg=aoa_deg,
        predicted_amplitude=float(selected["amplitude"]),
        direct_gate=direct_gate,
        wall_gate=wall_gate,
        beam=beam,
        uncertainty_samples=plan_input.robust_num_samples,
        uncertainty_description=description,
    )


def evidence_score(
    *,
    direct_measured: AngularDelayMap,
    wall_measured: AngularDelayMap,
    direct_twin: AngularDelayMap,
    wall_twin: AngularDelayMap,
    plan: WallPlan,
    config: PilotConfig,
) -> EvidenceScore:
    """Compute the measured normalized evidence score with fixed gates."""
    if plan.direct_gate is None or plan.wall_gate is None:
        raise VerificationError("evidence requires both direct and wall gates")
    direct = gate_energy(
        direct_measured,
        delay_min_s=plan.direct_gate.delay_min_s,
        delay_max_s=plan.direct_gate.delay_max_s,
        aoa_min_deg=plan.direct_gate.aoa_min_deg,
        aoa_max_deg=plan.direct_gate.aoa_max_deg,
    )
    wall = gate_energy(
        wall_measured,
        delay_min_s=plan.wall_gate.delay_min_s,
        delay_max_s=plan.wall_gate.delay_max_s,
        aoa_min_deg=plan.wall_gate.aoa_min_deg,
        aoa_max_deg=plan.wall_gate.aoa_max_deg,
    )
    twin_direct = gate_energy(
        direct_twin,
        delay_min_s=plan.direct_gate.delay_min_s,
        delay_max_s=plan.direct_gate.delay_max_s,
        aoa_min_deg=plan.direct_gate.aoa_min_deg,
        aoa_max_deg=plan.direct_gate.aoa_max_deg,
    )
    twin_wall = gate_energy(
        wall_twin,
        delay_min_s=plan.wall_gate.delay_min_s,
        delay_max_s=plan.wall_gate.delay_max_s,
        aoa_min_deg=plan.wall_gate.aoa_min_deg,
        aoa_max_deg=plan.wall_gate.aoa_max_deg,
    )
    epsilon = float(config.get("phase6", "evidence", "epsilon_w"))
    floor_db = float(
        config.get("phase6", "evidence", "reference_floor_snr_db")
    )
    floor_linear = 10.0 ** (floor_db / 10.0)
    r_dt = (
        twin_wall.energy_w / twin_direct.energy_w
        if twin_direct.energy_w > 0.0
        else float("nan")
    )
    denominator = max(direct.net_energy_w, epsilon)
    score = (
        wall.net_energy_w / denominator / r_dt
        if math.isfinite(r_dt) and r_dt > 0.0
        else float("nan")
    )
    direct_reliable = direct.net_energy_w > direct.noise_w * floor_linear
    reference_detectable = (
        twin_direct.energy_w > direct.noise_w * floor_linear
    )
    thresholds = config.get("phase6", "evidence", "thresholds")
    status, reason = classify_evidence(
        score=score,
        net_wall_w=wall.net_energy_w,
        wall_noise_w=wall.noise_w,
        direct_reliable=direct_reliable and math.isfinite(r_dt) and r_dt > 0.0,
        reference_detectable=reference_detectable,
        thresholds=thresholds,
    )
    return EvidenceScore(
        direct=direct,
        wall=wall,
        twin_direct=twin_direct,
        twin_wall=twin_wall,
        r_dt=r_dt,
        score=score,
        reference_detectable=reference_detectable,
        direct_reliable=direct_reliable,
        status=status,
        reason=reason,
    )
