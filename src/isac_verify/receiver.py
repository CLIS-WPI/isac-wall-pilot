"""Measured-signal processing and candidate estimation (BRIEF §10.1).

The receiver works only on the known probe and the measured effective channel.
It never imports the evaluator, metrics, or a true-scene object.  Delay-Doppler
processing uses Sionna's ``ofdm_to_delay_doppler_channel`` on the complete
time-frequency grid.  Angular processing uses the actual 16-element receive
steering vectors with the transmit dimension already collapsed by the beam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import torch
from sionna.phy.channel import (
    ofdm_to_delay_doppler_channel,
    subcarrier_frequencies,
)

from .beams import ArrayGeometry, steering_vector
from .config import PilotConfig
from .replay import noise_variance_w, proper_complex_noise
from .wall_paths import PathSet, WallPathError, path_records
from .waveform import OfdmProfile, profile_from_config, qpsk_symbols


class ReceiverError(ValueError):
    """Raised when measured processing inputs violate the declared model."""


@dataclass(frozen=True)
class LinkPath:
    label: str
    family: str
    matrix: torch.Tensor
    delay_s: float
    aod_deg: float
    aoa_deg: float
    vertex_points_m: tuple[tuple[float, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "family": self.family,
            "delay_ns": self.delay_s * 1e9,
            "aod_deg": self.aod_deg,
            "aoa_deg": self.aoa_deg,
        }


@dataclass(frozen=True)
class ProcessingChoices:
    profile: OfdmProfile
    delay_search_ns: tuple[float, float]
    aoa_search_deg: tuple[float, float]
    aoa_grid_step_deg: float
    detection_floor_db: float
    guard_delay_ns: tuple[float, float]
    guard_ratio_bounds: tuple[float, float]
    delay_window: str
    aoa_window: str
    psf_threshold_db: float
    min_departure_separation_deg: float
    receiver_array: ArrayGeometry

    @classmethod
    def from_config(cls, config: PilotConfig) -> "ProcessingChoices":
        values = config.section("phase6", "processing")
        delay_search = tuple(
            float(value) for value in values["delay_search_ns"]
        )
        aoa_search = tuple(
            float(value) for value in values["aoa_search_deg"]
        )
        guard = tuple(float(value) for value in values["guard_delay_ns"])
        guard_bounds = tuple(
            float(value) for value in values["guard_ratio_bounds"]
        )
        if len(delay_search) != 2 or delay_search[0] >= delay_search[1]:
            raise ReceiverError("delay search bounds are invalid")
        if len(aoa_search) != 2 or aoa_search[0] >= aoa_search[1]:
            raise ReceiverError("AoA search bounds are invalid")
        if len(guard) != 2 or guard[0] >= guard[1]:
            raise ReceiverError("guard-delay bounds are invalid")
        if guard[0] <= delay_search[1] and guard[1] >= delay_search[0]:
            raise ReceiverError(
                "guard delay must not overlap the delay search window"
            )
        if len(guard_bounds) != 2 or guard_bounds[0] >= guard_bounds[1]:
            raise ReceiverError("guard ratio bounds are invalid")
        delay_window = str(values["delay_window"]).lower()
        aoa_window = str(values["aoa_window"]).lower()
        if delay_window not in {"rectangular", "hann"}:
            raise ReceiverError(f"unsupported delay window {delay_window!r}")
        if aoa_window not in {"rectangular", "hann"}:
            raise ReceiverError(f"unsupported AoA window {aoa_window!r}")
        if float(values["aoa_grid_step_deg"]) <= 0.0:
            raise ReceiverError("AoA grid step must be positive")
        rx_values = config.section("arrays", "gnb_rx")
        return cls(
            profile=profile_from_config(config, str(values["profile"])),
            delay_search_ns=delay_search,
            aoa_search_deg=aoa_search,
            aoa_grid_step_deg=float(values["aoa_grid_step_deg"]),
            detection_floor_db=float(values["detection_floor_db"]),
            guard_delay_ns=guard,
            guard_ratio_bounds=guard_bounds,
            delay_window=delay_window,
            aoa_window=aoa_window,
            psf_threshold_db=float(values["psf_threshold_db"]),
            min_departure_separation_deg=float(
                values["min_departure_separation_deg"]
            ),
            receiver_array=ArrayGeometry(
                num_elements=int(rx_values["num_elements"]),
                spacing_wavelengths=float(rx_values["spacing_wavelengths"]),
                wavelength_m=config.wavelength_m,
            ),
        )

    @property
    def aoa_grid_deg(self) -> torch.Tensor:
        start, stop = self.aoa_search_deg
        step = self.aoa_grid_step_deg
        count = int(round((stop - start) / step)) + 1
        grid = start + step * torch.arange(
            count, device="cuda", dtype=torch.float64
        )
        if float(grid[-1]) < stop - 1e-9:
            grid = torch.cat(
                [
                    grid,
                    torch.tensor(
                        [stop], device="cuda", dtype=torch.float64
                    ),
                ]
            )
        return grid


@dataclass(frozen=True)
class Observation:
    world: str
    beam_name: str
    y: torch.Tensor
    h_eff: torch.Tensor
    symbols: torch.Tensor
    profile: OfdmProfile
    probe_seed: int
    noise_seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "world": self.world,
            "beam_name": self.beam_name,
            "y_shape": list(self.y.shape),
            "h_eff_shape": list(self.h_eff.shape),
            "probe_seed": self.probe_seed,
            "noise_seed": self.noise_seed,
        }


@dataclass(frozen=True)
class DelayDopplerMap:
    h_dd: torch.Tensor
    l_min: int
    l_max: int
    zero_doppler_index: int
    delay_s: torch.Tensor
    doppler_hz: torch.Tensor

    @property
    def zero_doppler(self) -> torch.Tensor:
        return self.h_dd[:, self.zero_doppler_index, :]


@dataclass(frozen=True)
class AngularDelayMap:
    power_w: torch.Tensor
    delay_s: torch.Tensor
    aoa_deg: torch.Tensor
    noise_per_cell_w: float
    delay_bin_s: float
    aoa_step_deg: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.power_w.shape),
            "delay_bins": int(self.delay_s.numel()),
            "aoa_bins": int(self.aoa_deg.numel()),
            "noise_per_cell_w": self.noise_per_cell_w,
            "delay_bin_ns": self.delay_bin_s * 1e9,
            "aoa_step_deg": self.aoa_step_deg,
        }


@dataclass(frozen=True)
class PointSpread:
    delay_width_s: float
    aoa_width_deg: float
    delay_half_width_s: float
    aoa_half_width_deg: float
    peak_delay_s: float
    peak_aoa_deg: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "delay_width_ns": self.delay_width_s * 1e9,
            "aoa_width_deg": self.aoa_width_deg,
            "delay_half_width_ns": self.delay_half_width_s * 1e9,
            "aoa_half_width_deg": self.aoa_half_width_deg,
            "peak_delay_ns": self.peak_delay_s * 1e9,
            "peak_aoa_deg": self.peak_aoa_deg,
        }


@dataclass(frozen=True)
class PointSpreadTheory:
    delay_width_s: float
    aoa_width_deg: float
    delay_bin_s: float
    aoa_step_deg: float
    delay_evaluation_bins: int
    aoa_evaluation_bins: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "delay_width_ns": self.delay_width_s * 1e9,
            "aoa_width_deg": self.aoa_width_deg,
            "delay_bin_ns": self.delay_bin_s * 1e9,
            "aoa_step_deg": self.aoa_step_deg,
            "delay_evaluation_bins": self.delay_evaluation_bins,
            "aoa_evaluation_bins": self.aoa_evaluation_bins,
        }


@dataclass(frozen=True)
class CandidateEstimate:
    detected: bool
    delay_s: float
    azimuth_deg: float
    range_m: float
    position_m: tuple[float, float, float]
    delay_sigma_s: float
    azimuth_sigma_deg: float
    peak_power_w: float
    noise_per_cell_w: float
    peak_to_noise_db: float
    detection_floor_db: float
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "status": self.status,
            "delay_ns": self.delay_s * 1e9,
            "azimuth_deg": self.azimuth_deg,
            "range_m": self.range_m,
            "position_m": list(self.position_m),
            "delay_sigma_ns": self.delay_sigma_s * 1e9,
            "azimuth_sigma_deg": self.azimuth_sigma_deg,
            "peak_power_w": self.peak_power_w,
            "noise_per_cell_w": self.noise_per_cell_w,
            "peak_to_noise_db": self.peak_to_noise_db,
            "detection_floor_db": self.detection_floor_db,
        }


@dataclass(frozen=True)
class GateEnergy:
    energy_w: float
    noise_w: float
    net_energy_w: float
    cells: int
    delay_bins: int
    aoa_bins: int


def link_paths(
    path_set: PathSet,
    *,
    label: str,
    families: Sequence[str] | None = None,
    exclude_los: bool = False,
) -> tuple[LinkPath, ...]:
    """Return per-path complex matrices with absolute delays."""
    a, tau = path_set.paths.cir(
        out_type="torch", normalize_delays=False
    )
    a = torch.as_tensor(a)
    tau = torch.as_tensor(tau)
    if a.ndim != 6 or tau.ndim != 3:
        raise ReceiverError(
            f"{label}: unexpected CIR shapes {tuple(a.shape)}, "
            f"{tuple(tau.shape)}"
        )
    wanted = None if families is None else set(str(f) for f in families)
    selected: list[LinkPath] = []
    for record in path_records(path_set, scene_label=label):
        family = str(record["family"])
        if wanted is not None and family not in wanted:
            continue
        if exclude_los and family == "los":
            continue
        path_index = int(record["path_index"])
        selected.append(
            LinkPath(
                label=f"{label}:{family}:{path_index}",
                family=family,
                matrix=a[0, :, 0, :, path_index, 0],
                delay_s=float(tau[0, 0, path_index].item()),
                aod_deg=float(record["aod_deg"]),
                aoa_deg=float(record["aoa_deg"]),
                vertex_points_m=tuple(
                    tuple(float(value) for value in point)
                    for point in record["vertex_points_m"]
                ),
            )
        )
    if not selected:
        raise WallPathError(f"{label}: no paths selected")
    return tuple(selected)


def _require_cuda(name: str, value: torch.Tensor) -> None:
    if not value.is_cuda:
        raise ReceiverError(f"{name} must stay on CUDA")


def cfr_from_paths(
    paths: Sequence[LinkPath],
    weights: torch.Tensor,
    profile: OfdmProfile,
) -> torch.Tensor:
    """Sum path matrices after transmit beamforming on the OFDM grid."""
    if not paths:
        raise ReceiverError("at least one path is required")
    frequencies = subcarrier_frequencies(
        profile.fft_size,
        profile.subcarrier_spacing_hz,
        device="cuda",
    ).to(torch.float64)
    weights = torch.as_tensor(weights)
    _require_cuda("beam weights", weights)
    response: torch.Tensor | None = None
    for path in paths:
        _require_cuda(path.label, path.matrix)
        matrix = path.matrix.to(torch.complex64)
        if weights.numel() != matrix.shape[1]:
            raise ReceiverError(
                f"{path.label}: beam dimension {weights.numel()} != "
                f"transmit antennas {matrix.shape[1]}"
            )
        beamed = matrix @ weights.to(matrix.dtype)
        phase = torch.exp(
            -2j * math.pi * frequencies * path.delay_s
        ).to(torch.complex64)
        contribution = beamed[:, None] * phase[None, :]
        response = (
            contribution
            if response is None
            else response + contribution
        )
    if response is None:
        raise ReceiverError("path response is empty")
    return response


def relay_transfer(
    *,
    downlink: Sequence[LinkPath],
    uplink: Sequence[LinkPath],
    weights: torch.Tensor,
    gain: complex,
    added_delay_s: float,
    profile: OfdmProfile,
) -> torch.Tensor:
    """Cascaded relay response for one beam, receive-by-frequency."""
    ones = torch.ones(1, device="cuda", dtype=torch.complex64)
    down = cfr_from_paths(downlink, weights, profile)[0]
    up = cfr_from_paths(uplink, ones, profile)
    frequencies = subcarrier_frequencies(
        profile.fft_size,
        profile.subcarrier_spacing_hz,
        device="cuda",
    ).to(torch.float64)
    phase = torch.exp(
        -2j * math.pi * frequencies * float(added_delay_s)
    ).to(torch.complex64)
    return up * (complex(gain) * phase * down)[None, :]


def relay_forwarded_noise(
    *,
    uplink: Sequence[LinkPath],
    gain: complex,
    added_delay_s: float,
    profile: OfdmProfile,
    variance_w: float,
    seed: int,
) -> torch.Tensor:
    """One relay-input noise record propagated through the uplink channel."""
    frequencies = subcarrier_frequencies(
        profile.fft_size,
        profile.subcarrier_spacing_hz,
        device="cuda",
    ).to(torch.float64)
    ones = torch.ones(1, device="cuda", dtype=torch.complex64)
    up = cfr_from_paths(uplink, ones, profile)
    relay_noise = proper_complex_noise(
        (profile.num_symbols, profile.fft_size),
        variance_w,
        seed,
    )
    phase = torch.exp(
        -2j * math.pi * frequencies * float(added_delay_s)
    ).to(torch.complex64)
    return (
        up[None, :, :]
        * (complex(gain) * phase)[None, None, :]
        * relay_noise[:, None, :]
    )


def make_observation(
    *,
    world: str,
    beam_name: str,
    beam_weights: torch.Tensor,
    symbols: torch.Tensor,
    target_paths: Sequence[LinkPath] = (),
    background_paths: Sequence[LinkPath] = (),
    relay_downlink: Sequence[LinkPath] = (),
    relay_uplink: Sequence[LinkPath] = (),
    relay_gain: complex | None = None,
    relay_added_delay_s: float | None = None,
    include_relay: bool = False,
    profile: OfdmProfile,
    config: PilotConfig,
    noise_seed: int,
    relay_noise_seed: int,
    probe_seed: int = -1,
    noiseless: bool = False,
) -> Observation:
    """Generate one measured frequency-domain action for a declared world."""
    if world not in {"target", "replay", "background", "blocked"}:
        raise ReceiverError(f"unknown world {world!r}")
    if tuple(symbols.shape) != (
        profile.num_symbols,
        profile.fft_size,
    ):
        raise ReceiverError("probe grid does not match the profile")
    if include_relay:
        if relay_gain is None or relay_added_delay_s is None:
            raise ReceiverError("relay gain and added delay are required")
        if not relay_downlink or not relay_uplink:
            raise ReceiverError("relay links are required")
    signal: torch.Tensor | None = None
    if target_paths:
        signal = cfr_from_paths(target_paths, beam_weights, profile)
    if background_paths:
        clutter = cfr_from_paths(
            background_paths, beam_weights, profile
        )
        signal = clutter if signal is None else signal + clutter
    if include_relay:
        signal = (
            relay_transfer(
                downlink=relay_downlink,
                uplink=relay_uplink,
                weights=beam_weights,
                gain=relay_gain,
                added_delay_s=relay_added_delay_s,
                profile=profile,
            )
            if signal is None
            else signal
            + relay_transfer(
                downlink=relay_downlink,
                uplink=relay_uplink,
                weights=beam_weights,
                gain=relay_gain,
                added_delay_s=relay_added_delay_s,
                profile=profile,
            )
        )
    if signal is None:
        num_rx = int(config.get("arrays", "gnb_rx", "num_elements"))
        signal = torch.zeros(
            (num_rx, profile.fft_size),
            device="cuda",
            dtype=torch.complex64,
        )
    y = (
        signal[None, :, :]
        * symbols.to(torch.complex64)[:, None, :]
    )
    if not noiseless:
        variance = noise_variance_w(config, profile.sample_rate_hz)
        y = y + proper_complex_noise(
            (profile.num_symbols, signal.shape[0], profile.fft_size),
            variance,
            noise_seed,
        )
        if include_relay:
            from .replay import relay_noise_variance_w

            y = y + relay_forwarded_noise(
                uplink=relay_uplink,
                gain=relay_gain,
                added_delay_s=relay_added_delay_s,
                profile=profile,
                variance_w=relay_noise_variance_w(
                    config, profile.sample_rate_hz
                ),
                seed=relay_noise_seed,
            )
    h_eff = y * symbols.to(torch.complex64).conj()[:, None, :]
    return Observation(
        world=world,
        beam_name=beam_name,
        y=y,
        h_eff=h_eff,
        symbols=symbols,
        profile=profile,
        probe_seed=probe_seed,
        noise_seed=noise_seed,
    )


def _delay_window(choices: ProcessingChoices) -> torch.Tensor:
    count = choices.profile.fft_size
    if choices.delay_window == "rectangular":
        return torch.ones(count, device="cuda", dtype=torch.float64)
    window = torch.hann_window(
        count, periodic=False, device="cuda", dtype=torch.float64
    )
    rms = torch.sqrt(torch.mean(window.square()))
    return window / rms


def _aoa_window(choices: ProcessingChoices) -> torch.Tensor:
    count = choices.receiver_array.num_elements
    if choices.aoa_window == "rectangular":
        return torch.ones(count, device="cuda", dtype=torch.float64)
    window = torch.hann_window(
        count, periodic=False, device="cuda", dtype=torch.float64
    )
    rms = torch.sqrt(torch.mean(window.square()))
    return window / rms


def delay_doppler_map(
    h_eff: torch.Tensor,
    choices: ProcessingChoices,
) -> DelayDopplerMap:
    """Transform an effective response into delay-Doppler coordinates."""
    if h_eff.ndim != 3:
        raise ReceiverError(
            "effective response must be symbols-by-rx-by-frequency"
        )
    _require_cuda("effective response", h_eff)
    profile = choices.profile
    if tuple(h_eff.shape[1:]) != (
        choices.receiver_array.num_elements,
        profile.fft_size,
    ):
        raise ReceiverError("effective response shape does not match arrays")
    sample_rate = profile.sample_rate_hz
    l_max = min(
        profile.fft_size - 1,
        int(math.ceil(choices.delay_search_ns[1] * 1e-9 * sample_rate)),
    )
    window = _delay_window(choices).to(h_eff.dtype)
    windowed = h_eff * window[None, None, :]
    h_dd = ofdm_to_delay_doppler_channel(
        windowed.permute(1, 0, 2), l_min=0, l_max=l_max
    )
    lag_count = l_max + 1
    lags = torch.arange(
        lag_count, device="cuda", dtype=torch.float64
    )
    delay_s = lags * profile.delay_bin_spacing_s
    doppler_hz = torch.fft.fftshift(
        torch.fft.fftfreq(
            profile.num_symbols,
            d=profile.symbol_duration_s,
            device="cuda",
            dtype=torch.float64,
        )
    )
    return DelayDopplerMap(
        h_dd=h_dd,
        l_min=0,
        l_max=l_max,
        zero_doppler_index=profile.num_symbols // 2,
        delay_s=delay_s,
        doppler_hz=doppler_hz,
    )


def angular_delay_map(
    delay_doppler: DelayDopplerMap,
    choices: ProcessingChoices,
    *,
    noise_per_cell_w: float,
) -> AngularDelayMap:
    """Scan receive steering vectors at zero Doppler and every delay lag."""
    if noise_per_cell_w < 0.0:
        raise ReceiverError("noise power must be non-negative")
    zero = delay_doppler.zero_doppler
    _require_cuda("delay-Doppler map", zero)
    angles = choices.aoa_grid_deg
    window = _aoa_window(choices).to(zero.dtype)[:, None]
    windowed = window * zero
    powers = []
    for angle in angles:
        vector = steering_vector(
            float(angle),
            choices.receiver_array,
            dtype=torch.complex64,
        )
        combined = (vector.conj()[:, None] * windowed).sum(dim=0)
        powers.append(combined.abs().square())
    power_w = torch.stack(powers, dim=1)
    return AngularDelayMap(
        power_w=power_w,
        delay_s=delay_doppler.delay_s,
        aoa_deg=angles,
        noise_per_cell_w=float(noise_per_cell_w),
        delay_bin_s=choices.profile.delay_bin_spacing_s,
        aoa_step_deg=choices.aoa_grid_step_deg,
    )


def noise_calibration(
    *,
    choices: ProcessingChoices,
    config: PilotConfig,
    records: int,
    seed: int,
) -> float:
    """Estimate mean delay-Doppler noise power per receive cell.

    The estimator uses the mean, not the median, so a gate containing many
    cells has the declared expected noise energy.
    """
    if records < 1:
        raise ReceiverError("noise calibration needs at least one record")
    variance = noise_variance_w(
        config, choices.profile.sample_rate_hz
    )
    sample_rate = choices.profile.sample_rate_hz
    l_max = min(
        choices.profile.fft_size - 1,
        int(math.ceil(choices.delay_search_ns[1] * 1e-9 * sample_rate)),
    )
    cells = []
    for index in range(records):
        noise = proper_complex_noise(
            (
                choices.profile.num_symbols,
                choices.receiver_array.num_elements,
                choices.profile.fft_size,
            ),
            variance,
            seed + index,
        )
        window = _delay_window(choices).to(noise.dtype)
        windowed = noise * window[None, None, :]
        h_dd = ofdm_to_delay_doppler_channel(
            windowed.permute(1, 0, 2), l_min=0, l_max=l_max
        )
        zero = h_dd[:, choices.profile.num_symbols // 2, :]
        cells.append(zero.abs().square().reshape(-1))
    measured = torch.cat(cells)
    estimate = float(measured.mean().item())
    if not math.isfinite(estimate) or estimate <= 0.0:
        raise ReceiverError("noise calibration returned a non-positive power")
    return estimate



def noise_gate_validation(
    *,
    config: PilotConfig,
    choices: ProcessingChoices,
    gates: Sequence[dict[str, Any]],
    records: int = 128,
    seed: int = 9201,
    calibration_records: int = 64,
) -> list[dict[str, Any]]:
    """Measure E/N through the actual no-target receiver pipeline.

    Each gate is evaluated on independent receiver-noise-only actions at its
    declared OFDM look length. The result is per gate variant; callers may
    aggregate variants for one method/gate.
    """
    if records < 64:
        raise ReceiverError("noise validation requires at least 64 records")
    if not gates:
        raise ReceiverError("noise validation requires at least one gate")
    array = choices.receiver_array
    weights = torch.ones(
        array.num_elements, device="cuda", dtype=torch.complex64
    ) / math.sqrt(array.num_elements)
    by_symbols: dict[int, list[dict[str, Any]]] = {}
    for gate in gates:
        by_symbols.setdefault(int(gate["symbols"]), []).append(dict(gate))
    results: list[dict[str, Any]] = []
    for symbols_count, local_gates in sorted(by_symbols.items()):
        local_choices = replace(
            choices,
            profile=replace(choices.profile, num_symbols=int(symbols_count)),
        )
        n_cell = noise_calibration(
            choices=local_choices,
            config=config,
            records=int(calibration_records),
            seed=int(seed),
        )
        symbols = qpsk_symbols(local_choices.profile, int(seed))
        ratios: dict[int, list[float]] = {index: [] for index in range(len(local_gates))}
        energies: dict[int, list[float]] = {index: [] for index in range(len(local_gates))}
        for record_index in range(int(records)):
            observation = make_observation(
                world="target",
                beam_name="noise-only",
                beam_weights=weights,
                symbols=symbols,
                target_paths=(),
                background_paths=(),
                relay_downlink=(),
                relay_uplink=(),
                include_relay=False,
                profile=local_choices.profile,
                config=config,
                noise_seed=int(seed) + 100000 + record_index,
                relay_noise_seed=int(seed) + 200000 + record_index,
                probe_seed=int(seed),
            )
            delay_doppler = delay_doppler_map(observation.h_eff, local_choices)
            angular = angular_delay_map(
                delay_doppler,
                local_choices,
                noise_per_cell_w=n_cell,
            )
            for index, gate in enumerate(local_gates):
                energy = gate_energy(
                    angular,
                    delay_min_s=float(gate["delay_min_ns"]) * 1e-9,
                    delay_max_s=float(gate["delay_max_ns"]) * 1e-9,
                    aoa_min_deg=float(gate["aoa_min_deg"]),
                    aoa_max_deg=float(gate["aoa_max_deg"]),
                )
                ratios[index].append(energy.energy_w / energy.noise_w)
                energies[index].append(energy.energy_w)
        for index, gate in enumerate(local_gates):
            values = np.asarray(ratios[index], dtype=float)
            gate_energies = np.asarray(energies[index], dtype=float)
            standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
            mean_ratio = float(values.mean())
            results.append(
                {
                    "method": str(gate.get("method", "")),
                    "gate": str(gate["gate"]),
                    "variant": str(gate.get("variant", "default")),
                    "symbols": int(symbols_count),
                    "records": int(records),
                    "cells": int(
                        gate_energy(
                            angular,
                            delay_min_s=float(gate["delay_min_ns"]) * 1e-9,
                            delay_max_s=float(gate["delay_max_ns"]) * 1e-9,
                            aoa_min_deg=float(gate["aoa_min_deg"]),
                            aoa_max_deg=float(gate["aoa_max_deg"]),
                        ).cells
                    ),
                    "noise_estimate_per_cell_w": float(n_cell),
                    "mean_energy_w": float(gate_energies.mean()),
                    "mean_e_over_n": mean_ratio,
                    "ci95_low": mean_ratio - 1.96 * standard_error,
                    "ci95_high": mean_ratio + 1.96 * standard_error,
                    "std_e_over_n": float(values.std(ddof=1)),
                }
            )
    return results

def guard_noise_power(
    h_eff: torch.Tensor,
    choices: ProcessingChoices,
) -> float:
    """Estimate delay-Doppler noise power in the disjoint guard interval."""
    sample_rate = choices.profile.sample_rate_hz
    l_min = int(
        math.ceil(choices.guard_delay_ns[0] * 1e-9 * sample_rate)
    )
    l_max = int(
        math.floor(choices.guard_delay_ns[1] * 1e-9 * sample_rate)
    )
    if l_min < 0 or l_max >= choices.profile.fft_size or l_min > l_max:
        raise ReceiverError("guard interval is outside the transform grid")
    window = _delay_window(choices).to(h_eff.dtype)
    windowed = h_eff * window[None, None, :]
    h_dd = ofdm_to_delay_doppler_channel(
        windowed.permute(1, 0, 2), l_min=l_min, l_max=l_max
    )
    zero = h_dd[:, choices.profile.num_symbols // 2, :]
    return float(zero.abs().square().flatten().median().item())


def _log_parabolic_offset(left: float, center: float, right: float) -> float:
    floor = 1e-30
    a, b, c = (
        math.log(max(left, floor)),
        math.log(max(center, floor)),
        math.log(max(right, floor)),
    )
    denominator = a - 2.0 * b + c
    if abs(denominator) <= 1e-18:
        return 0.0
    offset = 0.5 * (a - c) / denominator
    return max(-0.5, min(0.5, offset))


def _peak_index(power: torch.Tensor) -> tuple[int, int]:
    flat = int(torch.argmax(power).item())
    lag_index, angle_index = divmod(flat, power.shape[1])
    return lag_index, angle_index


def estimate_candidate(
    angular_delay: AngularDelayMap,
    *,
    choices: ProcessingChoices,
    point_spread: PointSpread | None,
    gnb_position_m: Sequence[float],
    speed_of_light_m_per_s: float,
    h_eff: torch.Tensor | None = None,
) -> CandidateEstimate:
    """Detect and interpolate the direct candidate inside the fixed sector."""
    power = angular_delay.power_w
    if power.numel() == 0:
        raise ReceiverError("angular-delay map is empty")
    lag_index, angle_index = _peak_index(power)
    peak = float(power[lag_index, angle_index].item())
    noise = float(angular_delay.noise_per_cell_w)
    if noise <= 0.0:
        raise ReceiverError("noise estimate must be positive")
    peak_to_noise_db = 10.0 * math.log10(max(peak, 1e-300) / noise)
    detected = peak_to_noise_db >= choices.detection_floor_db
    if not detected:
        return CandidateEstimate(
            detected=False,
            delay_s=float(angular_delay.delay_s[lag_index].item()),
            azimuth_deg=float(angular_delay.aoa_deg[angle_index].item()),
            range_m=float("nan"),
            position_m=(float("nan"),) * 3,
            delay_sigma_s=float("nan"),
            azimuth_sigma_deg=float("nan"),
            peak_power_w=peak,
            noise_per_cell_w=noise,
            peak_to_noise_db=peak_to_noise_db,
            detection_floor_db=choices.detection_floor_db,
            status="ACQUISITION_FAILED",
        )
    delay_offset = 0.0
    if 0 < lag_index < power.shape[0] - 1:
        delay_offset = _log_parabolic_offset(
            float(power[lag_index - 1, angle_index].item()),
            peak,
            float(power[lag_index + 1, angle_index].item()),
        )
    angle_offset = 0.0
    if 0 < angle_index < power.shape[1] - 1:
        angle_offset = _log_parabolic_offset(
            float(power[lag_index, angle_index - 1].item()),
            peak,
            float(power[lag_index, angle_index + 1].item()),
        )
    azimuth_deg = float(
        angular_delay.aoa_deg[angle_index].item()
    ) + angle_offset * angular_delay.aoa_step_deg
    if h_eff is None:
        delay_s = float(
            angular_delay.delay_s[lag_index].item()
        ) + delay_offset * angular_delay.delay_bin_s
    else:
        dense_profile, dense_spacing = _zero_padded_delay_profile(
            h_eff, choices, angle_deg=azimuth_deg
        )
        dense_index = int(torch.argmax(dense_profile).item())
        dense_offset = 0.0
        if 0 < dense_index < dense_profile.numel() - 1:
            dense_offset = _log_parabolic_offset(
                float(dense_profile[dense_index - 1].item()),
                float(dense_profile[dense_index].item()),
                float(dense_profile[dense_index + 1].item()),
            )
        delay_s = (dense_index + dense_offset) * dense_spacing
    snr = max(peak / noise, 1.0)
    if point_spread is None:
        delay_sigma = 0.0
        angle_sigma = 0.0
    else:
        delay_sigma = (
            point_spread.delay_half_width_s / math.sqrt(2.0 * snr)
        )
        angle_sigma = (
            point_spread.aoa_half_width_deg / math.sqrt(2.0 * snr)
        )
    radius = 0.5 * speed_of_light_m_per_s * delay_s
    angle_rad = math.radians(azimuth_deg)
    position = (
        float(gnb_position_m[0]) + radius * math.cos(angle_rad),
        float(gnb_position_m[1]) + radius * math.sin(angle_rad),
        float(gnb_position_m[2]),
    )
    return CandidateEstimate(
        detected=True,
        delay_s=delay_s,
        azimuth_deg=azimuth_deg,
        range_m=radius,
        position_m=position,
        delay_sigma_s=delay_sigma,
        azimuth_sigma_deg=angle_sigma,
        peak_power_w=peak,
        noise_per_cell_w=noise,
        peak_to_noise_db=peak_to_noise_db,
        detection_floor_db=choices.detection_floor_db,
        status="ACQUIRED",
    )


def _crossing_position(
    values: torch.Tensor,
    inside: int,
    outside: int,
    cutoff: float,
) -> float:
    floor = 1e-300
    a = math.log(max(float(values[inside].item()), floor))
    b = math.log(max(float(values[outside].item()), floor))
    denominator = a - b
    if abs(denominator) <= 1e-30:
        return float(inside)
    fraction = (a - math.log(max(cutoff, floor))) / denominator
    fraction = min(max(fraction, 0.0), 1.0)
    return float(inside) + fraction * float(outside - inside)


def _interpolated_width_bins(
    values: torch.Tensor,
    peak_index: int,
    *,
    threshold: float,
) -> float:
    peak = float(values[peak_index].item())
    if peak <= 0.0:
        raise ReceiverError("point-spread peak is non-positive")
    cutoff = peak * threshold
    left = peak_index
    while left > 0 and float(values[left - 1].item()) >= cutoff:
        left -= 1
    right = peak_index
    while right < values.numel() - 1 and float(values[right + 1].item()) >= cutoff:
        right += 1
    if left == 0 or right == values.numel() - 1:
        raise ReceiverError("point-spread threshold crossing is outside the grid")
    left_cross = _crossing_position(values, left, left - 1, cutoff)
    right_cross = _crossing_position(values, right, right + 1, cutoff)
    width = right_cross - left_cross
    if width <= 0.0:
        raise ReceiverError("interpolated point-spread width is not positive")
    return width


def _zero_padded_delay_profile(
    h_eff: torch.Tensor,
    choices: ProcessingChoices,
    *,
    angle_deg: float,
) -> tuple[torch.Tensor, float]:
    pad = 1 << 16
    window = _delay_window(choices).to(h_eff.dtype)
    vector = steering_vector(
        angle_deg,
        choices.receiver_array,
        dtype=torch.complex64,
    )
    combined = (
        vector.conj()[:, None] * (h_eff * window[None, None, :])
    ).sum(dim=0)
    spectrum = torch.fft.ifft(combined, n=pad, dim=-1)
    power = spectrum.abs().square().sum(dim=0)
    spacing_s = 1.0 / (
        pad * choices.profile.subcarrier_spacing_hz
    )
    return power, spacing_s


def measure_point_spread(
    angular_delay: AngularDelayMap,
    *,
    choices: ProcessingChoices,
    h_eff: torch.Tensor | None = None,
) -> PointSpread:
    """Measure the delay and angle main-lobe widths of the processor."""
    power = angular_delay.power_w
    lag_index, angle_index = _peak_index(power)
    threshold = 10.0 ** (choices.psf_threshold_db / 10.0)
    angle_profile = power[lag_index, :]
    angle_bins = _interpolated_width_bins(
        angle_profile, angle_index, threshold=threshold
    )
    aoa_width = angle_bins * angular_delay.aoa_step_deg
    peak_aoa_deg = float(angular_delay.aoa_deg[angle_index].item())
    if 0 < angle_index < angle_profile.numel() - 1:
        angle_offset = _log_parabolic_offset(
            float(angle_profile[angle_index - 1].item()),
            float(angle_profile[angle_index].item()),
            float(angle_profile[angle_index + 1].item()),
        )
        peak_aoa_deg += angle_offset * angular_delay.aoa_step_deg
    if h_eff is None:
        delay_profile = power[:, angle_index]
        delay_bins = _interpolated_width_bins(
            delay_profile, lag_index, threshold=threshold
        )
        delay_width = delay_bins * angular_delay.delay_bin_s
        peak_delay_s = float(angular_delay.delay_s[lag_index].item())
    else:
        delay_profile, spacing_s = _zero_padded_delay_profile(
            h_eff, choices, angle_deg=peak_aoa_deg
        )
        dense_index = int(torch.argmax(delay_profile).item())
        delay_bins = _interpolated_width_bins(
            delay_profile, dense_index, threshold=threshold
        )
        delay_width = delay_bins * spacing_s
        peak_delay_s = dense_index * spacing_s
        if 0 < dense_index < delay_profile.numel() - 1:
            offset = _log_parabolic_offset(
                float(delay_profile[dense_index - 1].item()),
                float(delay_profile[dense_index].item()),
                float(delay_profile[dense_index + 1].item()),
            )
            peak_delay_s += offset * spacing_s
    return PointSpread(
        delay_width_s=delay_width,
        aoa_width_deg=aoa_width,
        delay_half_width_s=0.5 * delay_width,
        aoa_half_width_deg=0.5 * aoa_width,
        peak_delay_s=peak_delay_s,
        peak_aoa_deg=peak_aoa_deg,
    )


def theoretical_window_widths(
    choices: ProcessingChoices,
) -> PointSpreadTheory:
    """Evaluate the declared window's continuous -3 dB main-lobe width."""
    threshold = 10.0 ** (choices.psf_threshold_db / 10.0)
    profile = choices.profile
    n = profile.fft_size
    if choices.delay_window == "rectangular":
        delay_window = torch.ones(n, device="cuda", dtype=torch.float64)
    else:
        delay_window = torch.hann_window(
            n, periodic=False, device="cuda", dtype=torch.float64
        )
    pad = 1 << 16
    impulse = torch.fft.fftshift(
        torch.fft.ifft(delay_window, n=pad)
    )
    impulse_power = impulse.abs().square()
    delay_peak = pad // 2
    delay_width_bins = _interpolated_width_bins(
        impulse_power, delay_peak, threshold=threshold
    )
    delay_width = delay_width_bins / (pad * profile.subcarrier_spacing_hz)

    num_rx = choices.receiver_array.num_elements
    if choices.aoa_window == "rectangular":
        aoa_window = torch.ones(
            num_rx, device="cuda", dtype=torch.float64
        )
    else:
        aoa_window = torch.hann_window(
            num_rx, periodic=False, device="cuda", dtype=torch.float64
        )
    aoa_window = aoa_window / torch.linalg.vector_norm(aoa_window)
    angle_step = 0.001
    angles = torch.arange(
        -90.0,
        90.0 + 0.5 * angle_step,
        angle_step,
        device="cuda",
        dtype=torch.float64,
    )
    positions = choices.receiver_array.positions_m(
        device="cuda", dtype=torch.float64
    )[:, 1]
    phase = (
        2.0
        * math.pi
        / choices.receiver_array.wavelength_m
        * positions[None, :]
        * torch.sin(torch.deg2rad(angles))[:, None]
    )
    array_factor = (
        aoa_window[None, :] * torch.exp(1j * phase)
    ).sum(dim=1)
    aoa_power = array_factor.abs().square()
    aoa_peak = int(torch.argmax(aoa_power).item())
    aoa_width = (
        _interpolated_width_bins(
            aoa_power, aoa_peak, threshold=threshold
        )
        * angle_step
    )
    return PointSpreadTheory(
        delay_width_s=delay_width,
        aoa_width_deg=aoa_width,
        delay_bin_s=profile.delay_bin_spacing_s,
        aoa_step_deg=angle_step,
        delay_evaluation_bins=pad,
        aoa_evaluation_bins=int(angles.numel()),
    )


def gate_energy(
    angular_delay: AngularDelayMap,
    *,
    delay_min_s: float,
    delay_max_s: float,
    aoa_min_deg: float,
    aoa_max_deg: float,
) -> GateEnergy:
    """Sum measured power and estimated noise over a fixed gate."""
    delay_mask = (
        (angular_delay.delay_s >= delay_min_s - 1e-15)
        & (angular_delay.delay_s <= delay_max_s + 1e-15)
    )
    angle_mask = (
        (angular_delay.aoa_deg >= aoa_min_deg - 1e-12)
        & (angular_delay.aoa_deg <= aoa_max_deg + 1e-12)
    )
    delay_bins = int(delay_mask.sum().item())
    angle_bins = int(angle_mask.sum().item())
    if delay_bins == 0 or angle_bins == 0:
        raise ReceiverError("gate contains no delay or angle cells")
    selected = angular_delay.power_w[delay_mask][:, angle_mask]
    energy = float(selected.sum().item())
    cells = delay_bins * angle_bins
    noise = angular_delay.noise_per_cell_w * cells
    return GateEnergy(
        energy_w=energy,
        noise_w=noise,
        net_energy_w=energy - noise,
        cells=cells,
        delay_bins=delay_bins,
        aoa_bins=angle_bins,
    )
