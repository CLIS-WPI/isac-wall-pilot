"""OFDM probe generation with a unitary FFT convention and explicit CP.

The frequency-domain QPSK grid has unit-modulus symbols.  The time-domain
symbol is ``ifft(X, norm="ortho")``; the CP is the last ``cp_length`` samples
of that symbol, prepended without any phase discontinuity.  The record is the
concatenation of the per-symbol CP+useful blocks in transmission order.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch

from .config import PilotConfig


class WaveformError(ValueError):
    """Raised when a waveform profile or request is invalid."""


@dataclass(frozen=True)
class OfdmProfile:
    name: str
    fft_size: int
    subcarrier_spacing_hz: float
    cp_length: int
    num_symbols: int
    occupied_bandwidth_hz: float
    sample_rate_hz: float

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "OfdmProfile":
        profile = cls(
            name=str(values["name"]),
            fft_size=int(values["fft_size"]),
            subcarrier_spacing_hz=float(values["subcarrier_spacing_hz"]),
            cp_length=int(values["cp_length"]),
            num_symbols=int(values["num_symbols"]),
            occupied_bandwidth_hz=float(values["occupied_bandwidth_hz"]),
            sample_rate_hz=float(values["sample_rate_hz"]),
        )
        if profile.fft_size * profile.subcarrier_spacing_hz != profile.sample_rate_hz:
            raise WaveformError(
                f"{profile.name}: fft_size * scs != sample_rate_hz"
            )
        if profile.occupied_bandwidth_hz != profile.sample_rate_hz:
            raise WaveformError(
                f"{profile.name}: occupied bandwidth is not the full grid"
            )
        if not 0 < profile.cp_length < profile.fft_size:
            raise WaveformError(f"{profile.name}: invalid CP length")
        return profile

    @property
    def symbol_duration_s(self) -> float:
        return (self.fft_size + self.cp_length) / self.sample_rate_hz

    @property
    def record_duration_s(self) -> float:
        return self.num_symbols * self.symbol_duration_s

    @property
    def delay_bin_spacing_s(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def cp_fraction_nominal(self) -> float:
        return self.cp_length / (self.fft_size + self.cp_length)


@dataclass(frozen=True)
class OfdmRecord:
    profile: OfdmProfile
    symbols: torch.Tensor
    useful: torch.Tensor
    record: torch.Tensor
    seed: int

    @property
    def sample_rate_hz(self) -> float:
        return self.profile.sample_rate_hz


@dataclass(frozen=True)
class PowerMeasurement:
    beam_power_w: float
    mean_power_per_element_w: float
    total_mean_power_w: float
    cp_energy_j: float
    record_energy_j: float
    cp_fraction_measured: float
    cp_fraction_nominal: float
    sample_rate_hz: float
    record_duration_s: float

    @property
    def illumination_energy_j(self) -> float:
        return self.record_energy_j


def profile_from_config(config: PilotConfig, name: str) -> OfdmProfile:
    return OfdmProfile.from_mapping(config.profile(name))


def _qpsk_table(device: str) -> torch.Tensor:
    values = torch.tensor(
        [1.0 + 1.0j, -1.0 + 1.0j, -1.0 - 1.0j, 1.0 - 1.0j],
        dtype=torch.complex64,
        device=device,
    )
    return values / sqrt(2.0)


def qpsk_symbols(
    profile: OfdmProfile,
    seed: int,
    *,
    device: str = "cuda",
) -> torch.Tensor:
    if not isinstance(seed, int):
        raise WaveformError("seed must be an explicit integer")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    indices = torch.randint(
        0,
        4,
        (profile.num_symbols, profile.fft_size),
        generator=generator,
        device=device,
        dtype=torch.int64,
    )
    symbols = _qpsk_table(device)[indices]
    if not bool(
        torch.allclose(
            symbols.abs(), torch.ones_like(symbols.real), atol=1e-6
        )
    ):
        raise WaveformError("QPSK symbols are not unit modulus")
    return symbols


def generate_ofdm_record(
    profile: OfdmProfile,
    seed: int,
    *,
    device: str = "cuda",
) -> OfdmRecord:
    symbols = qpsk_symbols(profile, seed, device=device)
    useful = torch.fft.ifft(symbols, dim=-1, norm="ortho")
    cp = useful[..., -profile.cp_length :]
    record = torch.cat([cp, useful], dim=-1)
    if record.shape != (
        profile.num_symbols,
        profile.cp_length + profile.fft_size,
    ):
        raise WaveformError(f"{profile.name}: unexpected record shape {record.shape}")
    return OfdmRecord(
        profile=profile,
        symbols=symbols,
        useful=useful,
        record=record,
        seed=seed,
    )


def demodulate_ofdm(
    record: torch.Tensor,
    profile: OfdmProfile,
) -> torch.Tensor:
    expected = (profile.num_symbols, profile.cp_length + profile.fft_size)
    if tuple(record.shape) != expected:
        raise WaveformError(
            f"record shape {tuple(record.shape)} does not match {expected}"
        )
    useful = record[:, profile.cp_length :]
    return torch.fft.fft(useful, dim=-1, norm="ortho")


def measure_transmit_power(
    record: OfdmRecord,
    beam_weights: torch.Tensor,
) -> PowerMeasurement:
    weights = torch.as_tensor(beam_weights)
    if weights.ndim != 1:
        raise WaveformError("beam weights must be one-dimensional")
    beam_power = float(weights.abs().square().sum().item())
    if beam_power <= 0.0:
        raise WaveformError("beam weights have zero power")
    waveform = record.record
    mean_per_element = float(waveform.abs().square().mean().item())
    total_mean = beam_power * mean_per_element
    cp_units = float(waveform[:, : record.profile.cp_length].abs().square().sum().item())
    record_units = float(waveform.abs().square().sum().item())
    sample_rate = record.sample_rate_hz
    cp_energy = beam_power * cp_units / sample_rate
    record_energy = beam_power * record_units / sample_rate
    measured_cp_fraction = cp_units / record_units
    return PowerMeasurement(
        beam_power_w=beam_power,
        mean_power_per_element_w=mean_per_element,
        total_mean_power_w=total_mean,
        cp_energy_j=cp_energy,
        record_energy_j=record_energy,
        cp_fraction_measured=measured_cp_fraction,
        cp_fraction_nominal=record.profile.cp_fraction_nominal,
        sample_rate_hz=sample_rate,
        record_duration_s=record.profile.record_duration_s,
    )


def dbm_to_watt(power_dbm: float) -> float:
    return 10.0 ** (float(power_dbm) / 10.0) * 1e-3


def waveform_reference(
    config: PilotConfig,
    seed: int,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    array = config.section("arrays", "tx")
    beam_power = dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))
    weights = torch.full(
        (int(array["num_elements"]),),
        sqrt(beam_power / int(array["num_elements"])),
        dtype=torch.complex64,
        device=device,
    )
    entries = []
    for profile_values in config.get("waveform", "profiles"):
        profile = OfdmProfile.from_mapping(profile_values)
        record = generate_ofdm_record(profile, seed, device=device)
        measurement = measure_transmit_power(record, weights)
        recovered = demodulate_ofdm(record.record, profile)
        roundtrip_error = float(
            (recovered - record.symbols).abs().max().item()
        )
        entries.append(
            {
                "name": profile.name,
                "fft_size": profile.fft_size,
                "cp_length": profile.cp_length,
                "num_symbols": profile.num_symbols,
                "sample_rate_hz": profile.sample_rate_hz,
                "occupied_bandwidth_hz": profile.occupied_bandwidth_hz,
                "delay_bin_spacing_ns": profile.delay_bin_spacing_s * 1e9,
                "symbol_duration_s": profile.symbol_duration_s,
                "record_duration_s": profile.record_duration_s,
                "measured_mean_power_w": measurement.total_mean_power_w,
                "beam_power_w": measurement.beam_power_w,
                "mean_power_per_element_w": measurement.mean_power_per_element_w,
                "cp_energy_j": measurement.cp_energy_j,
                "record_energy_j": measurement.record_energy_j,
                "cp_fraction_measured": measurement.cp_fraction_measured,
                "cp_fraction_nominal": measurement.cp_fraction_nominal,
                "fft_roundtrip_max_abs_error": roundtrip_error,
            }
        )
    return {"seed": seed, "entries": entries}
