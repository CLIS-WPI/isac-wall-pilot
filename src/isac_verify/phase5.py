"""Phase 5 continuous-waveform replay reference and depth/duplicate diagnostics.

The diagonal OFDM model is the fast frequency-domain approximation.  This
module synthesizes one continuous CP-OFDM record at an integer oversampling
factor and propagates it through linear multipath channels with exact
fractional delays.  It is deliberately kept separate from the Phase 3
single-path Experiment A adapters.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .beams import (
    ArrayGeometry,
    array_response,
    dbm_to_watt,
    direct_beam,
    projected_wall_beam,
)
from .config import PilotConfig
from .geometry import replay_added_delay_s, require_latency_floor
from .replay import (
    RelayCalibration,
    calibrate_matched_relay,
    forwarded_relay_noise,
    noise_variance_w,
    relay_noise_variance_w,
)
from .wall_paths import (
    PathSet,
    WallPathError,
    depth_power_decision,
    duplicate_path_diagnostics,
    path_records,
    solve_background_paths,
    solve_target_paths,
)
from .waveform import OfdmProfile, profile_from_config, qpsk_symbols


class Phase5Error(ValueError):
    """Raised when a Phase 5 waveform world violates its declared regime."""


class DiagonalModelInvalid(Phase5Error):
    """Raised when CP or Doppler conditions invalidate the diagonal model."""


@dataclass(frozen=True)
class PathTerm:
    label: str
    family: str
    coefficient: complex
    delay_s: float
    power_w: float
    interaction_names: tuple[str, ...]
    object_names: tuple[str, ...]
    aod_deg: float = float("nan")
    aoa_deg: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "family": self.family,
            "coefficient": [self.coefficient.real, self.coefficient.imag],
            "delay_ns": self.delay_s * 1e9,
            "power_w": self.power_w,
            "interaction_names": list(self.interaction_names),
            "object_names": list(self.object_names),
            "aod_deg": self.aod_deg,
            "aoa_deg": self.aoa_deg,
        }


@dataclass(frozen=True)
class ContinuousRecord:
    profile: OfdmProfile
    oversampling_factor: int
    symbols: torch.Tensor
    record: torch.Tensor
    sample_rate_hz: float
    cp_samples: int
    symbol_samples: int

    @property
    def block_samples(self) -> int:
        return self.symbol_samples


@dataclass(frozen=True)
class AgreementProfile:
    max_relative_error: float
    median_relative_error: float
    edge_max_relative_error: float
    interior_max_relative_error: float
    edge_median_relative_error: float
    interior_median_relative_error: float
    tolerance: float
    edge_fraction: float
    per_subcarrier_max: torch.Tensor

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_relative_error": self.max_relative_error,
            "median_relative_error": self.median_relative_error,
            "edge_max_relative_error": self.edge_max_relative_error,
            "interior_max_relative_error": self.interior_max_relative_error,
            "edge_median_relative_error": self.edge_median_relative_error,
            "interior_median_relative_error": self.interior_median_relative_error,
            "tolerance": self.tolerance,
            "edge_fraction": self.edge_fraction,
            "per_subcarrier_max": self.per_subcarrier_max.detach().cpu(),
        }


def _next_power_of_two(value: int) -> int:
    return 1 << int(math.ceil(math.log2(max(1, value))))


def generate_continuous_record(
    profile: OfdmProfile,
    seed: int,
    *,
    oversampling_factor: int,
    device: str = "cuda",
) -> ContinuousRecord:
    """Synthesize one continuous CP-OFDM record on the oversampled grid."""
    if oversampling_factor < 1:
        raise Phase5Error("oversampling factor must be positive")
    symbols = qpsk_symbols(profile, seed, device=device)
    num = profile.fft_size
    high = num * oversampling_factor
    padded = torch.zeros(
        (profile.num_symbols, high), dtype=symbols.dtype, device=device
    )
    half = num // 2
    padded[:, :half] = symbols[:, :half]
    padded[:, high - half :] = symbols[:, half:]
    useful = torch.fft.ifft(padded, dim=-1, norm="ortho") * math.sqrt(
        oversampling_factor
    )
    cp = useful[..., -profile.cp_length * oversampling_factor :]
    record = torch.cat([cp, useful], dim=-1)
    return ContinuousRecord(
        profile=profile,
        oversampling_factor=oversampling_factor,
        symbols=symbols,
        record=record,
        sample_rate_hz=profile.sample_rate_hz * oversampling_factor,
        cp_samples=profile.cp_length * oversampling_factor,
        symbol_samples=high,
    )


def demodulate_continuous(
    record: torch.Tensor,
    profile: OfdmProfile,
    *,
    oversampling_factor: int,
    start_sample: int = 0,
) -> torch.Tensor:
    """Demodulate a continuous record at the original symbol boundaries."""
    block = (profile.cp_length + profile.fft_size) * oversampling_factor
    needed = start_sample + profile.num_symbols * block
    if record.numel() < needed:
        raise Phase5Error(
            f"record has {record.numel()} samples, need {needed}"
        )
    blocks = record[start_sample:needed].reshape(profile.num_symbols, block)
    useful = blocks[:, profile.cp_length * oversampling_factor :]
    spectrum = torch.fft.fft(useful, dim=-1, norm="ortho") / math.sqrt(
        oversampling_factor
    )
    num = profile.fft_size
    high = num * oversampling_factor
    half = num // 2
    return torch.cat(
        [spectrum[:, :half], spectrum[:, high - half :]], dim=-1
    )


def multipath_waveform(
    record: torch.Tensor,
    terms: Sequence[PathTerm],
    *,
    sample_rate_hz: float,
    guard_samples: int,
) -> torch.Tensor:
    """Apply a linear multipath channel with exact fractional delays."""
    if record.ndim != 1:
        raise Phase5Error("continuous record must be one-dimensional")
    if not terms:
        raise Phase5Error("multipath waveform requires at least one term")
    if sample_rate_hz <= 0.0:
        raise Phase5Error("sample rate must be positive")
    if guard_samples < 0:
        raise Phase5Error("guard samples must be non-negative")
    max_delay = max(term.delay_s for term in terms)
    if max_delay < 0.0:
        raise Phase5Error("path delays must be non-negative")
    needed = (
        int(record.numel())
        + guard_samples
        + int(math.ceil(max_delay * sample_rate_hz))
        + 16
    )
    nfft = _next_power_of_two(needed)
    spectrum = torch.fft.fft(record.to(torch.complex128), n=nfft)
    frequencies = torch.fft.fftfreq(
        nfft,
        d=1.0 / sample_rate_hz,
        device=record.device,
        dtype=torch.float64,
    )
    total = torch.zeros(nfft, dtype=torch.complex128, device=record.device)
    for term in terms:
        shift = (term.delay_s + guard_samples / sample_rate_hz)
        phase = torch.exp(-2j * math.pi * frequencies * shift)
        total += complex(term.coefficient) * spectrum * phase
    return torch.fft.ifft(total, n=nfft)


def pre_arrival_energy_fraction(
    waveform: torch.Tensor,
    *,
    guard_samples: int,
    delay_s: float,
    sample_rate_hz: float,
) -> float:
    arrival = guard_samples + int(math.floor(delay_s * sample_rate_hz))
    if arrival <= 0:
        return 0.0
    total = float(waveform.abs().square().sum().item())
    if total <= 0.0:
        raise Phase5Error("waveform has zero energy")
    before = float(waveform[:arrival].abs().square().sum().item())
    return before / total


def normalized_correlation_at_delay(
    reference: torch.Tensor,
    delayed: torch.Tensor,
    *,
    guard_samples: int,
    delay_s: float,
    sample_rate_hz: float,
    search_samples: int = 2,
) -> float:
    """Correlation of a delayed copy at its true delay, searching one bin."""
    if reference.ndim != 1 or delayed.ndim != 1:
        raise Phase5Error("correlation records must be one-dimensional")
    center = guard_samples + int(round(delay_s * sample_rate_hz))
    best = -1.0
    ref_norm = float(torch.linalg.vector_norm(reference).item())
    if ref_norm <= 0.0:
        raise Phase5Error("reference record has zero norm")
    for offset in range(-search_samples, search_samples + 1):
        start = center + offset
        if start < 0 or start + reference.numel() > delayed.numel():
            continue
        candidate = delayed[start : start + reference.numel()]
        candidate_norm = float(torch.linalg.vector_norm(candidate).item())
        if candidate_norm <= 0.0:
            continue
        correlation = abs(
            torch.vdot(reference.to(torch.complex128), candidate.to(torch.complex128)).item()
        ) / (ref_norm * candidate_norm)
        best = max(best, float(correlation))
    if best < 0.0:
        raise Phase5Error("no valid correlation window")
    return best


def _cir_arrays(path_set: PathSet) -> tuple[torch.Tensor, torch.Tensor]:
    paths = path_set.paths
    a, tau = paths.cir(out_type="torch", normalize_delays=False)
    return torch.as_tensor(a), torch.as_tensor(tau)


def _record_order(path_set: PathSet, label: str) -> list[dict[str, Any]]:
    return path_records(path_set, scene_label=label)


def path_terms(
    path_set: PathSet,
    *,
    tx_weights: torch.Tensor,
    rx_weights: torch.Tensor,
    mode: str,
    label: str,
) -> list[PathTerm]:
    """Return scalar effective path coefficients after beamforming."""
    a, tau = _cir_arrays(path_set)
    if a.ndim != 6:
        raise Phase5Error(f"{label}: unexpected CIR shape {tuple(a.shape)}")
    if tau.ndim != 3:
        raise Phase5Error(f"{label}: unexpected tau shape {tuple(tau.shape)}")
    records = _record_order(path_set, label)
    terms = []
    for record in records:
        path_index = int(record["path_index"])
        matrix = a[0, :, 0, :, path_index, 0]
        if mode == "target" or mode == "background":
            coefficient = torch.vdot(
                rx_weights.to(matrix.dtype), matrix @ tx_weights.to(matrix.dtype)
            )
        elif mode == "downlink":
            coefficient = matrix.reshape(-1) @ tx_weights.to(matrix.dtype)
        elif mode == "uplink":
            vector = matrix.reshape(-1)
            coefficient = torch.vdot(
                rx_weights.to(vector.dtype), vector
            )
        else:
            raise Phase5Error(f"unknown path mode {mode!r}")
        delay = float(tau[0, 0, path_index].item())
        value = complex(coefficient.item())
        terms.append(
            PathTerm(
                label=f"{label}:{record['family']}:{path_index}",
                family=str(record["family"]),
                coefficient=value,
                delay_s=delay,
                power_w=abs(value) ** 2,
                interaction_names=tuple(record["interaction_names"]),
                object_names=tuple(record["object_names"]),
                aod_deg=float(record["aod_deg"]),
                aoa_deg=float(record["aoa_deg"]),
            )
        )
    return terms


def first_path_matrix(
    path_set: PathSet,
    family: str,
    *,
    label: str,
) -> tuple[torch.Tensor, float]:
    a, tau = _cir_arrays(path_set)
    for record in _record_order(path_set, label):
        if record["family"] != family:
            continue
        path_index = int(record["path_index"])
        return a[0, :, 0, :, path_index, 0], float(
            tau[0, 0, path_index].item()
        )
    raise Phase5Error(f"{label}: missing direct family {family!r}")


def relay_pair_terms(
    downlink_terms: Sequence[PathTerm],
    uplink_terms: Sequence[PathTerm],
    calibration: RelayCalibration,
) -> list[PathTerm]:
    """Cascade every downlink path with every uplink path and the relay delay."""
    if not downlink_terms or not uplink_terms:
        raise Phase5Error("relay requires both downlink and uplink paths")
    if not calibration.feasible:
        raise Phase5Error("relay calibration is infeasible")
    pairs = []
    for down in downlink_terms:
        for up in uplink_terms:
            coefficient = (
                calibration.gain * down.coefficient * up.coefficient
            )
            delay = (
                down.delay_s
                + calibration.measured_added_delay_s
                + up.delay_s
            )
            pairs.append(
                PathTerm(
                    label=f"relay:{down.family}->{up.family}",
                    family=f"relay_{down.family}__{up.family}",
                    coefficient=coefficient,
                    delay_s=delay,
                    power_w=abs(coefficient) ** 2,
                    interaction_names=down.interaction_names + up.interaction_names,
                    object_names=down.object_names + up.object_names,
                    aod_deg=down.aod_deg,
                    aoa_deg=up.aoa_deg,
                )
            )
    return pairs


def check_diagonal_validity(
    terms: Sequence[PathTerm],
    profile: OfdmProfile,
    config: PilotConfig,
    *,
    frequency_shift_hz: float = 0.0,
) -> dict[str, Any]:
    """Raise when the CP or Doppler conditions invalidate the diagonal model."""
    if not terms:
        raise Phase5Error("validity check requires at least one path")
    cp_duration = profile.cp_length / profile.sample_rate_hz
    max_delay = max(term.delay_s for term in terms)
    min_delay = min(term.delay_s for term in terms)
    spread = max_delay - min_delay
    cp_fraction = float(
        config.get("phase5", "max_delay_spread_cp_fraction")
    )
    doppler_limit = float(
        config.get("phase5", "doppler_symbol_cycles_max")
    )
    doppler_cycles = abs(float(frequency_shift_hz)) * profile.symbol_duration_s
    checks = {
        "cp_duration_s": cp_duration,
        "max_delay_s": max_delay,
        "min_delay_s": min_delay,
        "delay_spread_s": spread,
        "cp_fraction": cp_fraction,
        "cp_valid": max_delay <= cp_fraction * cp_duration,
        "doppler_cycles": doppler_cycles,
        "doppler_limit": doppler_limit,
        "doppler_valid": doppler_cycles <= doppler_limit,
    }
    if not checks["cp_valid"] or not checks["doppler_valid"]:
        raise DiagonalModelInvalid(
            "diagonal OFDM model invalid: "
            f"max_delay={max_delay * 1e9:.3f} ns, "
            f"cp={cp_duration * 1e9:.3f} ns, "
            f"doppler_cycles={doppler_cycles:.6g}"
        )
    return checks


def diagonal_response(
    terms: Sequence[PathTerm],
    profile: OfdmProfile,
    *,
    device: str = "cuda",
) -> torch.Tensor:
    frequencies = torch.fft.fftfreq(
        profile.fft_size,
        d=1.0 / profile.sample_rate_hz,
        device=device,
    ).to(torch.float64)
    response = torch.zeros(
        profile.fft_size, dtype=torch.complex128, device=device
    )
    for term in terms:
        response += complex(term.coefficient) * torch.exp(
            -2j * math.pi * frequencies * term.delay_s
        )
    return response


def diagonal_symbols(
    terms: Sequence[PathTerm],
    symbols: torch.Tensor,
    profile: OfdmProfile,
) -> torch.Tensor:
    if tuple(symbols.shape) != (profile.num_symbols, profile.fft_size):
        raise Phase5Error("symbol grid shape does not match the profile")
    response = diagonal_response(terms, profile, device=str(symbols.device))
    return symbols.to(torch.complex128) * response[None, :]


def time_domain_symbols(
    record: ContinuousRecord,
    terms: Sequence[PathTerm],
    *,
    guard_symbols: int,
) -> tuple[torch.Tensor, int]:
    guard = guard_symbols * record.block_samples
    waveform = multipath_waveform(
        record.record.reshape(-1),
        terms,
        sample_rate_hz=record.sample_rate_hz,
        guard_samples=guard,
    )
    return (
        demodulate_continuous(
            waveform,
            record.profile,
            oversampling_factor=record.oversampling_factor,
            start_sample=guard,
        ),
        guard,
    )


def _agreement_profile(
    diagonal: torch.Tensor,
    time_domain: torch.Tensor,
    config: PilotConfig,
    *,
    label: str,
) -> AgreementProfile:
    if diagonal.shape != time_domain.shape:
        raise Phase5Error(
            f"{label}: diagonal {tuple(diagonal.shape)} != time "
            f"{tuple(time_domain.shape)}"
        )
    reference = diagonal.to(torch.complex128)
    magnitude = reference.abs()
    scale = float(magnitude.max().item())
    if scale <= 0.0:
        raise Phase5Error(f"{label}: zero diagonal response")
    floor = max(scale * 1e-12, torch.finfo(magnitude.dtype).tiny)
    error = (reference - time_domain.to(torch.complex128)).abs() / magnitude.clamp_min(floor)
    per_subcarrier = error.amax(dim=0)
    edge_fraction = float(config.get("phase5", "subcarrier_edge_fraction"))
    edge_count = max(1, int(math.ceil(edge_fraction * per_subcarrier.numel())))
    edge = per_subcarrier[:edge_count]
    interior = per_subcarrier[edge_count:-edge_count]
    edge_symbols = error[:, :edge_count]
    interior_symbols = error[:, edge_count:-edge_count]
    return AgreementProfile(
        max_relative_error=float(per_subcarrier.max().item()),
        median_relative_error=float(per_subcarrier.median().item()),
        edge_max_relative_error=float(edge.max().item()),
        interior_max_relative_error=float(interior.max().item()),
        edge_median_relative_error=float(edge_symbols.median().item()),
        interior_median_relative_error=float(interior_symbols.median().item()),
        tolerance=float(config.get("phase5", "relative_agreement_tolerance")),
        edge_fraction=edge_fraction,
        per_subcarrier_max=per_subcarrier.detach().cpu(),
    )


def _power_db(value_w: float) -> float | None:
    if value_w <= 0.0:
        return None
    return 10.0 * math.log10(value_w / 1e-3)


def _find_term(terms: Sequence[PathTerm], family: str) -> PathTerm | None:
    for term in terms:
        if term.family == family:
            return term
    return None


def _world_terms(
    target_set: PathSet,
    background_set: PathSet,
    downlink_set: PathSet,
    uplink_set: PathSet,
    *,
    tx_weights: torch.Tensor,
    rx_weights: torch.Tensor,
    label: str,
) -> dict[str, list[PathTerm]]:
    target = path_terms(
        target_set,
        tx_weights=tx_weights,
        rx_weights=rx_weights,
        mode="target",
        label=f"{label}_target",
    )
    background = path_terms(
        background_set,
        tx_weights=tx_weights,
        rx_weights=rx_weights,
        mode="background",
        label=f"{label}_background",
    )
    background = [term for term in background if term.family != "los"]
    downlink = path_terms(
        downlink_set,
        tx_weights=tx_weights,
        rx_weights=rx_weights,
        mode="downlink",
        label=f"{label}_downlink",
    )
    uplink = path_terms(
        uplink_set,
        tx_weights=tx_weights,
        rx_weights=rx_weights,
        mode="uplink",
        label=f"{label}_uplink",
    )
    return {
        "target": target,
        "background": background,
        "downlink": downlink,
        "uplink": uplink,
    }


def _beam_values(config: PilotConfig) -> tuple[Any, Any, Any, torch.Tensor, torch.Tensor]:
    array = ArrayGeometry.from_config(config)
    power = dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))
    reference_phi = float(config.get("experiment_a", "reference_beam_phi_deg"))
    direct = direct_beam(reference_phi, power, array)
    wall = projected_wall_beam(
        float(config.get("phase5", "vulnerability_beam_phi_deg")),
        reference_phi,
        power,
        array,
        min_projected_norm=float(config.get("beam", "projected_norm_floor")),
    )
    if not wall.usable:
        raise Phase5Error(f"wall beam is unusable: {wall.status}")
    rx_values = config.section("arrays", "gnb_rx")
    rx_array = ArrayGeometry(
        num_elements=int(rx_values["num_elements"]),
        spacing_wavelengths=float(rx_values["spacing_wavelengths"]),
        wavelength_m=config.wavelength_m,
    )
    rx_weights = array_response(
        reference_phi, rx_array, dtype=torch.complex64
    )
    return direct, wall, array, direct.weights, rx_weights


def _gain_sweep(
    config: PilotConfig,
    calibration: RelayCalibration,
    downlink_terms: Sequence[PathTerm],
    *,
    beam_name: str,
) -> list[dict[str, Any]]:
    max_power = float(calibration.max_transmit_power_w)
    down_power = sum(term.power_w for term in downlink_terms)
    rows = []
    for offset_db in config.get("phase5", "gain_sweep_db"):
        scaled_gain = calibration.gain * 10.0 ** (float(offset_db) / 20.0)
        tx_power = abs(scaled_gain) ** 2 * down_power
        rows.append(
            {
                "beam": beam_name,
                "gain_offset_db": float(offset_db),
                "scaled_gain_abs": abs(scaled_gain),
                "relay_transmit_power_w": tx_power,
                "relay_transmit_power_dbm": _power_db(tx_power),
                "max_transmit_power_w": max_power,
                "feasible": tx_power <= max_power * (1.0 + 1e-12),
            }
        )
    return rows


def _gate_terms(
    terms: Sequence[PathTerm],
    *,
    delay_center_s: float,
    aoa_center_deg: float,
    delay_half_width_s: float,
    aoa_tolerance_deg: float,
) -> list[PathTerm]:
    return [
        term
        for term in terms
        if abs(term.delay_s - delay_center_s) <= delay_half_width_s
        and abs(term.aoa_deg - aoa_center_deg) <= aoa_tolerance_deg
    ]


def _gate_power(terms: Sequence[PathTerm]) -> dict[str, Any]:
    coherent = sum((term.coefficient for term in terms), 0.0 + 0.0j)
    return {
        "gate_power_sum_w": float(sum(term.power_w for term in terms)),
        "gate_coherent_power_w": float(abs(coherent) ** 2),
        "coherent_coefficient": [coherent.real, coherent.imag],
        "path_count": len(terms),
        "families": sorted({term.family for term in terms}),
        "paths": [term.as_dict() for term in terms],
    }


def _vulnerability(
    config: PilotConfig,
    world: dict[str, list[PathTerm]],
    calibration: RelayCalibration,
) -> dict[str, Any]:
    target_walls = [
        term
        for term in world["target"]
        if term.family.startswith("outgoing_wall_")
    ]
    target_direct = _find_term(world["target"], "direct")
    if not target_walls or target_direct is None:
        return {
            "available": False,
            "adaptive_relay": "UNTESTED",
            "reason": "target direct or outgoing-wall path unavailable",
        }
    half_width = float(
        config.get("phase5", "vulnerability_gate_delay_half_width_ns")
    ) * 1e-9
    aoa_tolerance = float(
        config.get("phase5", "vulnerability_gate_aoa_tolerance_deg")
    )
    gate_aoa = float(target_direct.aoa_deg)
    target_wall = max(target_walls, key=lambda term: term.power_w)
    target_gate_delay = target_wall.delay_s
    target_gate = _gate_terms(
        world["target"],
        delay_center_s=target_gate_delay,
        aoa_center_deg=gate_aoa,
        delay_half_width_s=half_width,
        aoa_tolerance_deg=aoa_tolerance,
    )
    if not target_gate:
        return {
            "available": False,
            "adaptive_relay": "UNTESTED",
            "reason": "target wall gate is empty",
        }
    direct_uplinks = [
        term
        for term in world["uplink"]
        if abs(term.aoa_deg - gate_aoa) <= aoa_tolerance
    ]
    if not direct_uplinks:
        direct_uplinks = [
            min(world["uplink"], key=lambda term: abs(term.aoa_deg - gate_aoa))
        ]
    uplink = max(direct_uplinks, key=lambda term: term.power_w)
    anchor = max(world["downlink"], key=lambda term: term.power_w)
    required_delay = target_gate_delay - anchor.delay_s - uplink.delay_s
    downlink_gate = [
        term
        for term in world["downlink"]
        if abs(
            term.delay_s + required_delay + uplink.delay_s - target_gate_delay
        )
        <= half_width
    ]
    if not downlink_gate:
        return {
            "available": False,
            "adaptive_relay": "UNTESTED",
            "reason": "downlink gate group is empty",
        }
    target_sum = sum((term.coefficient for term in target_gate), 0.0 + 0.0j)
    downlink_sum = sum(
        (term.coefficient for term in downlink_gate), 0.0 + 0.0j
    )
    uplink_sum = sum(
        (term.coefficient for term in direct_uplinks), 0.0 + 0.0j
    )
    required_gain = target_sum / (downlink_sum * uplink_sum)
    total_input_power = float(
        sum(term.power_w for term in world["downlink"])
    )
    tx_power = abs(required_gain) ** 2 * total_input_power
    gate_energy = abs(required_gain * downlink_sum * uplink_sum) ** 2
    floor = calibration.latency_floor_s
    reference_target_delay_ns = float(
        config.get("phase5", "vulnerability_wall_delay_ns")
    )
    analytic_delay_ns = (
        reference_target_delay_ns
        - round(anchor.delay_s * 1e9, 6)
        - round(uplink.delay_s * 1e9, 6)
    )

    table = []
    for down in world["downlink"]:
        for up in world["uplink"]:
            pair_delay = down.delay_s + up.delay_s
            pair_required_delay = target_gate_delay - pair_delay
            aoa_consistent = (
                abs(up.aoa_deg - gate_aoa) <= aoa_tolerance
            )
            denominator = down.coefficient * up.coefficient
            pair_gain = (
                target_sum / denominator if denominator != 0.0 else complex(0.0)
            )
            pair_tx_power = abs(pair_gain) ** 2 * total_input_power
            table.append(
                {
                    "downlink_family": down.family,
                    "uplink_family": up.family,
                    "downlink_delay_ns": down.delay_s * 1e9,
                    "uplink_delay_ns": up.delay_s * 1e9,
                    "pair_echo_delay_ns": pair_delay * 1e9,
                    "arrival_aoa_deg": up.aoa_deg,
                    "gate_aoa_deg": gate_aoa,
                    "gate_aoa_consistent": aoa_consistent,
                    "gate_consistent": aoa_consistent,
                    "required_processing_delay_ns": pair_required_delay * 1e9,
                    "required_delay_shift_ns": (
                        pair_required_delay
                        - calibration.measured_added_delay_s
                    )
                    * 1e9,
                    "latency_floor_ns": floor * 1e9,
                    "causal_feasible": pair_required_delay >= floor,
                    "required_gain_abs": abs(pair_gain),
                    "required_gain_change_db": (
                        20.0 * math.log10(abs(pair_gain / calibration.gain))
                        if pair_gain != 0.0
                        else None
                    ),
                    "required_relay_transmit_power_w": pair_tx_power,
                    "required_relay_transmit_power_dbm": _power_db(
                        pair_tx_power
                    ),
                    "max_transmit_power_w": calibration.max_transmit_power_w,
                    "power_feasible": pair_tx_power
                    <= calibration.max_transmit_power_w * (1.0 + 1e-12),
                }
            )
    return {
        "available": True,
        "adaptive_relay": "UNTESTED",
        "gate_aoa_deg": gate_aoa,
        "gate_aoa_tolerance_deg": aoa_tolerance,
        "gate_delay_half_width_ns": half_width * 1e9,
        "gate_consistent_pair_count": sum(
            1 for row in table if row["gate_consistent"]
        ),
        "target_wall_delay_ns": target_gate_delay * 1e9,
        "relay_downlink_wall_delay_ns": anchor.delay_s * 1e9,
        "relay_uplink_direct_delay_ns": uplink.delay_s * 1e9,
        "reference_target_wall_delay_ns": reference_target_delay_ns,
        "analytic_required_processing_delay_ns": analytic_delay_ns,
        "analytic_required_delay_shift_ns": (
            analytic_delay_ns - calibration.measured_added_delay_s * 1e9
        ),
        "fixed_processing_delay_ns": (
            calibration.measured_added_delay_s * 1e9
        ),
        "required_processing_delay_ns": required_delay * 1e9,
        "required_delay_shift_ns": (
            required_delay - calibration.measured_added_delay_s
        )
        * 1e9,
        "latency_floor_ns": floor * 1e9,
        "causal_feasible": required_delay >= floor,
        "target_gate": _gate_power(target_gate),
        "downlink_gate": _gate_power(downlink_gate),
        "uplink_gate": _gate_power(direct_uplinks),
        "gate_energy_w": gate_energy,
        "total_downlink_input_power_w": total_input_power,
        "required_gain_abs": abs(required_gain),
        "required_gain_change_db": 20.0
        * math.log10(abs(required_gain / calibration.gain)),
        "required_relay_transmit_power_w": tx_power,
        "required_relay_transmit_power_dbm": _power_db(tx_power),
        "power_feasible": tx_power
        <= calibration.max_transmit_power_w * (1.0 + 1e-12),
        "pair_table": table,
        "note": (
            "Diagnostic only: the primary relay has one fixed gain and delay. "
            "A stronger adaptive relay is not simulated. Gate-consistent "
            "rows use the measured direct-return AoA. TX power uses the "
            "relay's total downlink input; gate energy uses all downlink "
            "paths inside the declared delay window."
        ),
    }


def _change_fraction(current: float, previous: float) -> float | None:
    if previous == 0.0:
        return 0.0 if current == 0.0 else float("inf")
    return (current - previous) / previous


def beam_weighted_depth_diagnostics(
    config: PilotConfig,
    *,
    direct_weights: torch.Tensor,
    wall_weights: torch.Tensor,
    rx_weights: torch.Tensor,
) -> dict[str, Any]:
    tolerance = float(
        config.get("scene", "d9_path_power_relative_tolerance")
    )
    half_width = float(
        config.get("phase5", "vulnerability_gate_delay_half_width_ns")
    ) * 1e-9
    aoa_tolerance = float(
        config.get("phase5", "vulnerability_gate_aoa_tolerance_deg")
    )
    target_gate_delay = float(
        config.get("phase5", "vulnerability_wall_delay_ns")
    ) * 1e-9
    direct_gate_delay = float(
        config.get("geometry_reference", "candidate_direct_round_trip_delay_ns")
    ) * 1e-9
    gate_aoa = float(
        config.get("experiment_a", "reference_beam_phi_deg")
    )
    depth_values = [
        int(value) for value in config.get("scene", "phase4_max_depth")
    ]
    target_depths = [depth for depth in depth_values if depth >= 2]
    beams = {"direct": direct_weights, "wall": wall_weights}
    target_rows: dict[str, list[dict[str, Any]]] = {
        beam: [] for beam in beams
    }
    downlink_rows: dict[str, list[dict[str, Any]]] = {
        beam: [] for beam in beams
    }
    background_rows: dict[str, list[dict[str, Any]]] = {
        beam: [] for beam in beams
    }
    for depth in target_depths:
        path_set = solve_target_paths(config, max_depth=depth)
        for beam, weights in beams.items():
            terms = path_terms(
                path_set,
                tx_weights=weights,
                rx_weights=rx_weights,
                mode="target",
                label=f"c2_target_{beam}_{depth}",
            )
            gate = _gate_terms(
                terms,
                delay_center_s=target_gate_delay,
                aoa_center_deg=gate_aoa,
                delay_half_width_s=half_width,
                aoa_tolerance_deg=aoa_tolerance,
            )
            target_rows[beam].append({"depth": depth, **_gate_power(gate)})
    for depth in depth_values:
        downlink_set = solve_background_paths(
            config, direction="downlink", max_depth=depth
        )
        background_set = solve_background_paths(
            config, direction="monostatic", max_depth=depth
        )
        for beam, weights in beams.items():
            down_terms = path_terms(
                downlink_set,
                tx_weights=weights,
                rx_weights=rx_weights,
                mode="downlink",
                label=f"c2_downlink_{beam}_{depth}",
            )
            downlink_rows[beam].append(
                {
                    "depth": depth,
                    "relay_input_power_w": float(
                        sum(term.power_w for term in down_terms)
                    ),
                    "families": sorted(
                        {term.family for term in down_terms}
                    ),
                }
            )
            background_terms = path_terms(
                background_set,
                tx_weights=weights,
                rx_weights=rx_weights,
                mode="background",
                label=f"c2_background_{beam}_{depth}",
            )
            direct_gate = _gate_terms(
                background_terms,
                delay_center_s=direct_gate_delay,
                aoa_center_deg=gate_aoa,
                delay_half_width_s=half_width,
                aoa_tolerance_deg=aoa_tolerance,
            )
            wall_gate = _gate_terms(
                background_terms,
                delay_center_s=target_gate_delay,
                aoa_center_deg=gate_aoa,
                delay_half_width_s=half_width,
                aoa_tolerance_deg=aoa_tolerance,
            )
            background_rows[beam].append(
                {
                    "depth": depth,
                    "direct_gate": _gate_power(direct_gate),
                    "wall_gate": _gate_power(wall_gate),
                }
            )
    for rows in (target_rows, downlink_rows):
        for beam_rows in rows.values():
            for index in range(1, len(beam_rows)):
                previous = beam_rows[index - 1]
                current = beam_rows[index]
                key = (
                    "gate_coherent_power_w"
                    if "gate_coherent_power_w" in current
                    else "relay_input_power_w"
                )
                current["power_change_fraction"] = _change_fraction(
                    float(current[key]), float(previous[key])
                )
    for beam_rows in background_rows.values():
        for index in range(1, len(beam_rows)):
            previous = beam_rows[index - 1]
            current = beam_rows[index]
            current["direct_gate_change_fraction"] = _change_fraction(
                float(current["direct_gate"]["gate_coherent_power_w"]),
                float(previous["direct_gate"]["gate_coherent_power_w"]),
            )
            current["wall_gate_change_fraction"] = _change_fraction(
                float(current["wall_gate"]["gate_coherent_power_w"]),
                float(previous["wall_gate"]["gate_coherent_power_w"]),
            )

    configured = {
        name: int(config.get("scene", "d9_depths", name))
        for name in ("target", "background", "downlink", "uplink")
    }
    decisions: dict[str, Any] = {}
    target_choice = configured["target"]
    target_index = target_depths.index(target_choice)
    if target_index + 1 < len(target_depths):
        changes = [
            abs(
                float(
                    target_rows[beam][target_index + 1][
                        "gate_coherent_power_w"
                    ]
                )
                - float(
                    target_rows[beam][target_index]["gate_coherent_power_w"]
                )
            )
            / max(
                float(target_rows[beam][target_index]["gate_coherent_power_w"]),
                np.finfo(float).tiny,
            )
            for beam in beams
        ]
        if max(changes) >= tolerance:
            target_choice = target_depths[target_index + 1]
    decisions["target"] = target_choice
    decisions["downlink"] = configured["downlink"]
    for index in range(len(depth_values) - 1):
        if depth_values[index] < configured["downlink"]:
            continue
        changes = [
            abs(
                _change_fraction(
                    float(downlink_rows[beam][index + 1]["relay_input_power_w"]),
                    float(downlink_rows[beam][index]["relay_input_power_w"]),
                )
            )
            for beam in beams
        ]
        if max(changes) < tolerance:
            break
        decisions["downlink"] = depth_values[index + 1]
    decisions["background"] = configured["background"]
    for index in range(len(depth_values) - 1):
        if depth_values[index] < configured["background"]:
            continue
        changes = []
        for beam in beams:
            row = background_rows[beam][index]
            next_row = background_rows[beam][index + 1]
            for gate in ("direct_gate", "wall_gate"):
                changes.append(
                    abs(
                        _change_fraction(
                            float(next_row[gate]["gate_coherent_power_w"]),
                            float(row[gate]["gate_coherent_power_w"]),
                        )
                    )
                )
        if max(changes) < tolerance:
            break
        decisions["background"] = depth_values[index + 1]
    decisions["uplink"] = configured["uplink"]
    selected = {
        name: int(decisions[name])
        for name in ("target", "background", "downlink", "uplink")
    }
    return {
        "tolerance": tolerance,
        "rule": (
            "start from the configured depth and raise only when the "
            "next-depth beam-weighted change is at least the tolerance"
        ),
        "gate_window": {
            "delay_half_width_ns": half_width * 1e9,
            "aoa_tolerance_deg": aoa_tolerance,
            "target_wall_gate_delay_ns": target_gate_delay * 1e9,
            "direct_gate_delay_ns": direct_gate_delay * 1e9,
            "gate_aoa_deg": gate_aoa,
        },
        "target_wall_gate": target_rows,
        "relay_input": downlink_rows,
        "background_gates": background_rows,
        "configured_depths": configured,
        "selected_depths": selected,
        "consistent": selected == configured,
        "note": (
            "Uplink depth is retained at the configured value because the "
            "declared beam-weighted checks cover target wall-gate echo, "
            "downlink relay input, and background gate energy."
        ),
    }


def phase5_diagnostics(
    config: PilotConfig,
    output_path: str | Path = "results/phase5",
) -> dict[str, Any]:
    profile = profile_from_config(config, "baseline")
    oversampling = int(config.get("phase5", "oversampling_factor"))
    guard_symbols = int(config.get("phase5", "guard_symbols"))
    seed = int(config.get("seeds", "streams", "probe_symbols", "calibration")[0])
    direct, wall, array, direct_weights, rx_weights = _beam_values(config)
    if not direct.usable:
        raise Phase5Error("direct beam is unusable")
    depths = {
        name: int(config.get("scene", "d9_depths", name))
        for name in ("target", "background", "downlink", "uplink")
    }
    target_set = solve_target_paths(config, max_depth=depths["target"])
    background_set = solve_background_paths(
        config, direction="monostatic", max_depth=depths["background"]
    )
    downlink_set = solve_background_paths(
        config, direction="downlink", max_depth=depths["downlink"]
    )
    uplink_set = solve_background_paths(
        config, direction="uplink", max_depth=depths["uplink"]
    )

    target_matrix, target_delay = first_path_matrix(
        target_set, "direct", label="phase5_target"
    )
    down_matrix, down_delay = first_path_matrix(
        downlink_set, "los", label="phase5_downlink"
    )
    up_matrix, up_delay = first_path_matrix(
        uplink_set, "los", label="phase5_uplink"
    )
    calibration = calibrate_matched_relay(
        target_coefficient=target_matrix,
        target_delay_s=target_delay,
        downlink_coefficient=down_matrix.reshape(-1),
        downlink_delay_s=down_delay,
        uplink_coefficient=up_matrix.reshape(-1),
        uplink_delay_s=up_delay,
        reference_weights=direct_weights,
        reference_rx_weights=rx_weights,
        config=config,
        reference_beam_phi_deg=float(direct.phi_deg),
    )
    require_latency_floor(
        calibration.measured_added_delay_s, calibration.latency_floor_s
    )

    beam_weighted_depth = beam_weighted_depth_diagnostics(
        config,
        direct_weights=direct.weights,
        wall_weights=wall.weights,
        rx_weights=rx_weights,
    )
    record = generate_continuous_record(
        profile,
        seed,
        oversampling_factor=oversampling,
    )
    beam_worlds = {}
    agreements = {}
    pair_tables = {}
    relay_power = {}
    gain_sweeps = []
    for beam_name, beam_weights in (
        ("direct", direct.weights),
        ("wall", wall.weights),
    ):
        world = _world_terms(
            target_set,
            background_set,
            downlink_set,
            uplink_set,
            tx_weights=beam_weights,
            rx_weights=rx_weights,
            label=f"phase5_{beam_name}",
        )
        pairs = relay_pair_terms(world["downlink"], world["uplink"], calibration)
        relay_terms = list(pairs) + list(world["background"])
        target_terms = list(world["target"]) + list(world["background"])
        validity_target = check_diagonal_validity(target_terms, profile, config)
        validity_relay = check_diagonal_validity(relay_terms, profile, config)
        target_td, _ = time_domain_symbols(
            record, target_terms, guard_symbols=guard_symbols
        )
        relay_td, _ = time_domain_symbols(
            record, relay_terms, guard_symbols=guard_symbols
        )
        target_diag = diagonal_symbols(target_terms, record.symbols, profile)
        relay_diag = diagonal_symbols(relay_terms, record.symbols, profile)
        target_agreement = _agreement_profile(
            target_diag, target_td, config, label=f"{beam_name}_target"
        )
        relay_agreement = _agreement_profile(
            relay_diag, relay_td, config, label=f"{beam_name}_relay"
        )
        beam_worlds[beam_name] = {
            "target_terms": [term.as_dict() for term in target_terms],
            "relay_terms": [term.as_dict() for term in relay_terms],
            "validity_target": validity_target,
            "validity_relay": validity_relay,
        }
        agreements[beam_name] = {
            "target": target_agreement.as_dict(),
            "relay": relay_agreement.as_dict(),
        }
        pair_tables[beam_name] = [
            term.as_dict() for term in pairs
        ]
        family_power: dict[str, float] = {}
        for term in world["downlink"]:
            family_power[term.family] = (
                family_power.get(term.family, 0.0) + term.power_w
            )
        relay_power[beam_name] = {
            "downlink_input_power_w": float(
                sum(term.power_w for term in world["downlink"])
            ),
            "family_power_w": family_power,
            "downlink_paths": [
                {
                    "family": term.family,
                    "delay_ns": term.delay_s * 1e9,
                    "power_w": term.power_w,
                    "power_dbm": _power_db(term.power_w),
                }
                for term in world["downlink"]
            ],
        }
        gain_sweeps.extend(
            _gain_sweep(
                config,
                calibration,
                world["downlink"],
                beam_name=beam_name,
            )
        )
        if beam_name == "wall":
            vulnerability = _vulnerability(config, world, calibration)

    # Physical relay noise is injected at the relay input before gain and
    # propagated through the full uplink; ideal mode remains noiseless.
    up_terms = _world_terms(
        target_set,
        background_set,
        downlink_set,
        uplink_set,
        tx_weights=direct.weights,
        rx_weights=rx_weights,
        label="phase5_noise",
    )["uplink"]
    up_response = diagonal_response(up_terms, profile)
    relay_variance = relay_noise_variance_w(config, profile.sample_rate_hz)
    forwarded = forwarded_relay_noise(
        up_response.reshape(1, -1),
        calibration,
        variance_w=relay_variance,
        seed=int(config.get("seeds", "streams", "relay_noise", "calibration")[0]),
        shape=(profile.num_symbols, 1, profile.fft_size),
        frequencies_hz=torch.fft.fftfreq(
            profile.fft_size,
            d=1.0 / profile.sample_rate_hz,
            device="cuda",
        ),
    )
    noise = {
        "relay_noise_figure_db": float(config.get("noise", "relay_noise_figure_db")),
        "receiver_noise_figure_db": float(config.get("noise", "receiver_noise_figure_db")),
        "relay_variance_w": relay_variance,
        "receiver_variance_w": noise_variance_w(config, profile.sample_rate_hz),
        "forwarded_relay_noise_mean_power_w": float(
            forwarded.abs().square().mean().item()
        ),
        "ideal_relay_note": "ideal mode is noiseless; physical mode injects relay noise before gain and uplink",
    }
    duplicate = duplicate_path_diagnostics(config)
    decision = depth_power_decision(config)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "profile": profile.name,
        "oversampling_factor": oversampling,
        "sample_rate_hz": record.sample_rate_hz,
        "cp_samples": record.cp_samples,
        "symbol_samples": record.symbol_samples,
        "seed": seed,
        "d9_depths": depths,
        "duplicates": duplicate,
        "depth_decision": decision,
        "beam_weighted_depth": beam_weighted_depth,
        "calibration": calibration.as_dict(),
        "agreements": agreements,
        "relay_power": relay_power,
        "gain_sweep": gain_sweeps,
        "vulnerability": vulnerability,
        "noise": noise,
        "validity": {
            beam: {
                "target": values["validity_target"],
                "relay": values["validity_relay"],
            }
            for beam, values in beam_worlds.items()
        },
        "worlds": beam_worlds,
        "pair_tables": pair_tables,
        "checks": {
            "target_wall_agreement": agreements["wall"]["target"]["max_relative_error"]
            <= agreements["wall"]["target"]["tolerance"],
            "relay_wall_agreement": agreements["wall"]["relay"]["max_relative_error"]
            <= agreements["wall"]["relay"]["tolerance"],
            "target_direct_agreement": agreements["direct"]["target"]["max_relative_error"]
            <= agreements["direct"]["target"]["tolerance"],
            "relay_direct_agreement": agreements["direct"]["relay"]["max_relative_error"]
            <= agreements["direct"]["relay"]["tolerance"],
            "no_duplicate_paths": duplicate["total_duplicate_groups"] == 0,
            "configured_d9_matches": decision["selected_depths"] == depths,
            "beam_weighted_d9_matches": (
                beam_weighted_depth["selected_depths"] == depths
            ),
        },
    }
    summary["status"] = "PASS" if all(summary["checks"].values()) else "FAIL"
    (output / "phase5_summary.json").write_text(
        _json_dump(summary), encoding="utf-8"
    )
    return summary


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _json_dump(value: Any) -> str:
    return json.dumps(_jsonable(value), indent=2) + "\n"
