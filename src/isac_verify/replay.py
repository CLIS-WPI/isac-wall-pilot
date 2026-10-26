"""Causal scalar replay relay, matched calibration, and proper-complex noise.

The relay is a single-antenna fixed-filter repeater.  Its calibration gain is
computed by the evaluator from actual target, downlink, and uplink path data;
the planner never receives the calibration.  The relay transmit-power check is
a feasibility test only: an infeasible calibration is never rescaled.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10, pi, sqrt
from typing import Sequence

import torch

from .beams import dbm_to_watt
from .config import PilotConfig
from .geometry import replay_added_delay_s, require_latency_floor
from .waveform import OfdmProfile


class ReplayError(ValueError):
    """Raised when a relay calibration or noise request is invalid."""


@dataclass(frozen=True)
class RelayCalibration:
    target_delay_s: float
    downlink_delay_s: float
    uplink_delay_s: float
    measured_added_delay_s: float
    analytical_added_delay_s: float
    delay_difference_s: float
    gain: complex
    gain_abs: float
    gain_phase_rad: float
    reference_beam_phi_deg: float
    target_reference_response: complex
    downlink_reference_response: complex
    uplink_reference_response: complex
    rx_steering_max_relative_difference: float
    relay_transmit_power_w: float
    max_transmit_power_w: float
    latency_floor_s: float
    latency_feasible: bool
    power_feasible: bool
    feasible: bool

    def as_dict(self) -> dict[str, float | bool | list[float]]:
        return {
            "target_delay_ns": self.target_delay_s * 1e9,
            "downlink_delay_ns": self.downlink_delay_s * 1e9,
            "uplink_delay_ns": self.uplink_delay_s * 1e9,
            "measured_added_delay_ns": self.measured_added_delay_s * 1e9,
            "analytical_added_delay_ns": self.analytical_added_delay_s * 1e9,
            "delay_difference_ns": self.delay_difference_s * 1e9,
            "gain": [self.gain.real, self.gain.imag],
            "gain_abs": self.gain_abs,
            "gain_phase_rad": self.gain_phase_rad,
            "reference_beam_phi_deg": self.reference_beam_phi_deg,
            "target_reference_response": [
                self.target_reference_response.real,
                self.target_reference_response.imag,
            ],
            "downlink_reference_response": [
                self.downlink_reference_response.real,
                self.downlink_reference_response.imag,
            ],
            "uplink_reference_response": [
                self.uplink_reference_response.real,
                self.uplink_reference_response.imag,
            ],
            "rx_steering_max_relative_difference": (
                self.rx_steering_max_relative_difference
            ),
            "relay_transmit_power_w": self.relay_transmit_power_w,
            "relay_transmit_power_dbm": (
                10.0 * log10(self.relay_transmit_power_w / 1e-3)
                if self.relay_transmit_power_w > 0.0
                else None
            ),
            "max_transmit_power_w": self.max_transmit_power_w,
            "latency_floor_ns": self.latency_floor_s * 1e9,
            "latency_feasible": self.latency_feasible,
            "power_feasible": self.power_feasible,
            "feasible": self.feasible,
        }


def noise_variance_w(
    config: PilotConfig,
    bandwidth_hz: float,
    *,
    noise_figure_db: float | None = None,
    temperature_k: float | None = None,
) -> float:
    if bandwidth_hz <= 0.0:
        raise ReplayError("bandwidth must be positive")
    if noise_figure_db is None:
        noise_figure_db = float(config.get("noise", "receiver_noise_figure_db"))
    if temperature_k is None:
        temperature_k = float(config.get("noise", "temperature_k"))
    boltzmann = float(config.get("constants", "boltzmann_j_per_k"))
    if temperature_k <= 0.0:
        raise ReplayError("temperature must be positive")
    if noise_figure_db < 0.0:
        raise ReplayError("noise figure must be non-negative")
    return (
        boltzmann
        * temperature_k
        * bandwidth_hz
        * 10.0 ** (float(noise_figure_db) / 10.0)
    )


def relay_noise_variance_w(
    config: PilotConfig, bandwidth_hz: float
) -> float:
    return noise_variance_w(
        config,
        bandwidth_hz,
        noise_figure_db=float(config.get("noise", "relay_noise_figure_db")),
    )


def proper_complex_noise(
    shape: Sequence[int],
    variance_w: float,
    seed: int,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    if not isinstance(seed, int):
        raise ReplayError("seed must be an explicit integer")
    if variance_w < 0.0:
        raise ReplayError("noise variance must be non-negative")
    if variance_w == 0.0:
        return torch.zeros(tuple(shape), device=device, dtype=dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    standard_deviation = sqrt(0.5 * variance_w)
    real = torch.randn(
        tuple(shape), generator=generator, device=device, dtype=torch.float32
    )
    imag = torch.randn(
        tuple(shape), generator=generator, device=device, dtype=torch.float32
    )
    return ((real + 1j * imag) * standard_deviation).to(dtype)


def mean_power_w(value: torch.Tensor) -> float:
    if value.numel() == 0:
        raise ReplayError("cannot measure power of an empty tensor")
    return float(value.abs().square().mean().item())


def calibrate_matched_relay(
    *,
    target_coefficient: torch.Tensor,
    target_delay_s: float,
    downlink_coefficient: torch.Tensor,
    downlink_delay_s: float,
    uplink_coefficient: torch.Tensor,
    uplink_delay_s: float,
    reference_weights: torch.Tensor,
    reference_rx_weights: torch.Tensor,
    config: PilotConfig,
    reference_beam_phi_deg: float,
) -> RelayCalibration:
    target = torch.as_tensor(target_coefficient)
    downlink = torch.as_tensor(downlink_coefficient).reshape(-1)
    uplink = torch.as_tensor(uplink_coefficient).reshape(-1)
    weights = torch.as_tensor(reference_weights).reshape(-1)
    rx_weights = torch.as_tensor(reference_rx_weights).reshape(-1)
    if target.ndim != 2:
        raise ReplayError(f"target must be receive-by-transmit, got {target.shape}")
    num_rx, num_tx = target.shape
    if downlink.numel() != num_tx:
        raise ReplayError("downlink dimension does not match the target")
    if uplink.numel() != num_rx:
        raise ReplayError("uplink dimension does not match the target")
    if weights.numel() != num_tx:
        raise ReplayError("reference beam dimension does not match the target")
    if rx_weights.numel() != num_rx:
        raise ReplayError("reference receive dimension does not match the target")
    if not target.is_cuda or not downlink.is_cuda or not uplink.is_cuda:
        raise ReplayError("relay calibration tensors must stay on CUDA")
    if not weights.is_cuda or not rx_weights.is_cuda:
        raise ReplayError("reference array weights must stay on CUDA")

    target_reference = torch.vdot(rx_weights, target @ weights)
    downlink_reference = downlink @ weights
    uplink_reference = torch.vdot(rx_weights, uplink)
    measured_added_delay = (
        float(target_delay_s) - float(downlink_delay_s) - float(uplink_delay_s)
    )
    analytical_added_delay = replay_added_delay_s(
        float(config.get("physical", "candidate_position_m")[0]),
        float(config.get("physical", "relay_position_m")[0]),
        config.speed_of_light,
    )
    latency_floor = (
        float(config.get("geometry_reference", "latency_floor_ns")) * 1e-9
    )
    require_latency_floor(measured_added_delay, latency_floor)
    denominator = uplink_reference * downlink_reference
    if bool((denominator == 0).item()):
        raise ReplayError("matched relay denominator is zero")
    gain = target_reference / denominator
    relay_power = float(gain.abs().square().item()) * float(
        downlink_reference.abs().square().item()
    )
    max_power = dbm_to_watt(config.get("relay", "max_average_transmit_power_dbm"))
    power_feasible = relay_power <= max_power * (1.0 + 1e-12)
    gain_complex = complex(gain.item())

    column_norms = torch.linalg.vector_norm(target, dim=0)
    if bool((column_norms == 0).any().item()):
        raise ReplayError("target has a zero receive-steering column")
    target_columns = target / column_norms
    column = int(torch.argmax(column_norms).item())
    target_steering = target_columns[:, column]
    uplink_steering = uplink / torch.linalg.vector_norm(uplink)
    overlap = torch.vdot(target_steering, uplink_steering)
    if bool((overlap == 0).item()):
        raise ReplayError("target and relay receive steering are orthogonal")
    aligned_uplink = uplink_steering * overlap.conj()
    rx_steering_error = float(
        (
            torch.linalg.vector_norm(target_steering - aligned_uplink)
            / torch.linalg.vector_norm(target_steering)
        ).item()
    )
    return RelayCalibration(
        target_delay_s=float(target_delay_s),
        downlink_delay_s=float(downlink_delay_s),
        uplink_delay_s=float(uplink_delay_s),
        measured_added_delay_s=measured_added_delay,
        analytical_added_delay_s=analytical_added_delay,
        delay_difference_s=measured_added_delay - analytical_added_delay,
        gain=gain_complex,
        gain_abs=abs(gain_complex),
        gain_phase_rad=float(torch.angle(gain).item()),
        reference_beam_phi_deg=float(reference_beam_phi_deg),
        target_reference_response=complex(target_reference.item()),
        downlink_reference_response=complex(downlink_reference.item()),
        uplink_reference_response=complex(uplink_reference.item()),
        rx_steering_max_relative_difference=rx_steering_error,
        relay_transmit_power_w=relay_power,
        max_transmit_power_w=max_power,
        latency_floor_s=latency_floor,
        latency_feasible=True,
        power_feasible=power_feasible,
        feasible=power_feasible,
    )


def _require_feasible(calibration: RelayCalibration) -> None:
    if not calibration.latency_feasible:
        raise ReplayError("relay calibration violates the latency floor")
    if not calibration.power_feasible:
        raise ReplayError(
            "relay calibration exceeds the average transmit-power cap; "
            "the matched attack is infeasible and must not be rescaled"
        )


def relay_cfr(
    downlink_cfr: torch.Tensor,
    uplink_cfr: torch.Tensor,
    calibration: RelayCalibration,
    frequencies_hz: torch.Tensor,
) -> torch.Tensor:
    _require_feasible(calibration)
    down = torch.as_tensor(downlink_cfr).reshape(-1)
    up = torch.as_tensor(uplink_cfr).reshape(-1)
    if down.numel() != up.numel():
        raise ReplayError("downlink and uplink CFR lengths do not agree")
    frequencies = torch.as_tensor(frequencies_hz).reshape(-1)
    if frequencies.numel() != down.numel():
        raise ReplayError("frequency grid does not match the CFR")
    added_phase = torch.exp(
        -2j * pi * frequencies * calibration.measured_added_delay_s
    )
    return calibration.gain * down * up * added_phase


def forwarded_relay_noise(
    uplink_cfr: torch.Tensor,
    calibration: RelayCalibration,
    *,
    variance_w: float,
    seed: int,
    shape: Sequence[int],
    frequencies_hz: torch.Tensor,
) -> torch.Tensor:
    _require_feasible(calibration)
    up = torch.as_tensor(uplink_cfr)
    if up.ndim == 1:
        up = up.reshape(1, -1)
    if up.ndim != 2:
        raise ReplayError("uplink CFR must be receive-by-frequency")
    frequencies = torch.as_tensor(frequencies_hz).reshape(-1)
    if up.shape[1] != frequencies.numel():
        raise ReplayError("frequency grid does not match the uplink CFR")
    noise_shape = tuple(shape)
    if (
        len(noise_shape) != 3
        or noise_shape[1] != up.shape[0]
        or noise_shape[2] != up.shape[1]
    ):
        raise ReplayError(
            "forwarded noise shape must be symbols-by-rx-by-frequency"
        )
    noise = proper_complex_noise(noise_shape, variance_w, seed)
    added_phase = torch.exp(
        -2j * pi * frequencies * calibration.measured_added_delay_s
    )
    return (
        calibration.gain
        * up[None, :, :]
        * added_phase[None, None, :]
        * noise
    )


def receiver_noise(
    profile: OfdmProfile,
    config: PilotConfig,
    seed: int,
    *,
    num_records: int = 1,
) -> torch.Tensor:
    if num_records < 1:
        raise ReplayError("num_records must be positive")
    variance = noise_variance_w(config, profile.sample_rate_hz)
    return proper_complex_noise(
        (num_records, profile.num_symbols, profile.fft_size),
        variance,
        seed,
    )


def frequency_shift_record(
    record: torch.Tensor,
    frequency_shift_hz: float,
    sample_rate_hz: float,
) -> torch.Tensor:
    if sample_rate_hz <= 0.0:
        raise ReplayError("sample rate must be positive")
    if record.ndim != 1:
        raise ReplayError("frequency-shift record must be one-dimensional")
    samples = torch.arange(record.numel(), device=record.device, dtype=torch.float64)
    phase = (
        2.0
        * pi
        * float(frequency_shift_hz)
        * samples
        / float(sample_rate_hz)
    )
    return record * torch.exp(1j * phase).to(record.dtype)
