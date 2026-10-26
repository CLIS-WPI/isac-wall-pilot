"""Sionna link adapters with absolute delays and baseband CFR conversion.

Experiment A uses an empty scene with no walls or background objects.  The
monostatic gNB TX -> gNB RX self link is never solved for background: relay
links are solved as gNB -> relay and relay -> gNB separately.  This implements
D6's ideal TX/RX isolation assumption without dropping any other enabled path.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Any, Sequence

import torch
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.rt import (
    PathSolver,
    PlanarArray,
    Receiver,
    Transmitter,
    load_scene,
)
from sionna.rt.rcs import ConstantRCSSensingTarget, RCSSolver

from .beams import ArrayGeometry, array_response, dbm_to_watt, direct_beam
from .config import PilotConfig
from .replay import RelayCalibration, calibrate_matched_relay
from .waveform import OfdmProfile, profile_from_config


class ChannelError(ValueError):
    """Raised when a required single-path or tensor-shape assumption fails."""


@dataclass(frozen=True)
class PathChannel:
    name: str
    a: torch.Tensor
    tau: torch.Tensor
    valid: torch.Tensor
    num_rx_antennas: int
    num_tx_antennas: int

    def cfr(self, profile: OfdmProfile) -> torch.Tensor:
        if self.a.ndim != 6 or self.tau.ndim != 3:
            raise ChannelError(
                f"{self.name}: unexpected CIR shapes {self.a.shape}, "
                f"{self.tau.shape}"
            )
        freqs = subcarrier_frequencies(
            profile.fft_size,
            profile.subcarrier_spacing_hz,
            device="cuda",
        )
        return cir_to_ofdm_channel(
            freqs,
            self.a.unsqueeze(0),
            self.tau.unsqueeze(0),
            normalize=False,
        )

    def single_path(self) -> tuple[torch.Tensor, float]:
        valid = self.valid.reshape(-1)
        indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if indices.numel() != 1:
            raise ChannelError(
                f"{self.name}: expected exactly one valid path, got "
                f"{indices.numel()}"
            )
        path_index = int(indices[0].item()) % self.valid.shape[-1]
        coefficient = self.a[0, :, 0, :, path_index, 0]
        delay = float(self.tau[0, 0, path_index].item())
        return coefficient, delay

    def single_path_cfr(self, profile: OfdmProfile) -> torch.Tensor:
        coefficient, delay = self.single_path()
        freqs = subcarrier_frequencies(
            profile.fft_size,
            profile.subcarrier_spacing_hz,
            device="cuda",
        )
        a = coefficient.reshape(
            1, coefficient.shape[0], 1, 1, coefficient.shape[1], 1, 1
        )
        tau = torch.tensor(
            [[[[delay]]]], device=self.tau.device, dtype=self.tau.dtype
        )
        return cir_to_ofdm_channel(
            freqs,
            a,
            tau,
            normalize=False,
        )


@dataclass(frozen=True)
class SionnaExperimentA:
    target: PathChannel
    downlink: PathChannel
    uplink: PathChannel
    calibration: RelayCalibration
    profile: OfdmProfile

    def target_cfr(self) -> torch.Tensor:
        return self.target.cfr(self.profile)

    def target_effective(self, weights: torch.Tensor) -> torch.Tensor:
        return effective_response(self.target_cfr(), weights)

    def relay_effective(self, weights: torch.Tensor) -> torch.Tensor:
        if not self.calibration.feasible:
            raise ChannelError(
                "relay calibration is infeasible at the configured power cap"
            )
        downlink_cfr = self.downlink.cfr(self.profile)
        uplink_cfr = self.uplink.cfr(self.profile)
        downlink = downlink_cfr[0, 0, 0, 0, :, 0, :]
        uplink = uplink_cfr[0, 0, :, 0, 0, 0, :]
        weights = weights.to(downlink.dtype)
        if weights.ndim == 1:
            beamed = weights @ downlink
            relay = uplink * (
                self.calibration.gain
                * beamed
                * self._added_phase()
            )[None, :]
        elif weights.ndim == 2:
            beamed = weights @ downlink
            relay = uplink[None, :, :] * (
                self.calibration.gain
                * beamed
                * self._added_phase()[None, :]
            )[:, None, :]
        else:
            raise ChannelError("weights must be one- or two-dimensional")
        return relay

    def _added_phase(self) -> torch.Tensor:
        freqs = subcarrier_frequencies(
            self.profile.fft_size,
            self.profile.subcarrier_spacing_hz,
            device="cuda",
        )
        return torch.exp(
            -2j
            * pi
            * freqs
            * self.calibration.measured_added_delay_s
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


def _scene(
    config: PilotConfig,
    tx_array: PlanarArray,
    rx_array: PlanarArray,
    tx_name: str,
    tx_position: Sequence[float],
    rx_name: str,
    rx_position: Sequence[float],
) -> Any:
    scene = load_scene()
    scene.frequency = config.frequency_hz
    scene.tx_array = tx_array
    scene.rx_array = rx_array
    scene.add(
        [
            Transmitter(tx_name, position=list(tx_position)),
            Receiver(rx_name, position=list(rx_position)),
        ]
    )
    return scene


def _paths_to_channel(
    name: str,
    paths: Any,
    num_rx_antennas: int,
    num_tx_antennas: int,
) -> PathChannel:
    a, tau = paths.cir(out_type="torch", normalize_delays=False)
    valid = torch.as_tensor(paths.valid)
    if not a.is_cuda or not tau.is_cuda or not valid.is_cuda:
        raise ChannelError(f"{name}: CIR/tau/valid tensors left CUDA")
    return PathChannel(
        name=name,
        a=a,
        tau=tau,
        valid=valid,
        num_rx_antennas=num_rx_antennas,
        num_tx_antennas=num_tx_antennas,
    )


def solve_target_channel(config: PilotConfig) -> PathChannel:
    tx_array = _planar_array(config.section("arrays", "gnb_tx"))
    rx_array = _planar_array(config.section("arrays", "gnb_rx"))
    scene = _scene(
        config,
        tx_array,
        rx_array,
        "gnb",
        config.get("physical", "gnb_position_m"),
        "gnb-rx",
        config.get("physical", "gnb_position_m"),
    )
    target = ConstantRCSSensingTarget(
        name="candidate",
        sigma=float(config.get("target", "rcs_m2")),
        length=0.2,
        width=0.2,
        height=0.2,
        position=list(config.get("physical", "candidate_position_m")),
    )
    scene.add(target)
    solver = config.section("solver", "rcs_solver")
    paths = RCSSolver(deterministic=bool(solver["deterministic"]))(
        scene,
        max_depth=1,
        synthetic_array=True,
        los=True,
        specular_reflection=False,
        refraction=bool(solver["refraction"]),
        seed=int(solver["seed"]),
    )
    return _paths_to_channel(
        "target",
        paths,
        int(config.get("arrays", "gnb_rx", "num_elements")),
        int(config.get("arrays", "gnb_tx", "num_elements")),
    )


def solve_relay_link(
    config: PilotConfig,
    *,
    direction: str,
) -> PathChannel:
    gnb = config.get("physical", "gnb_position_m")
    relay = config.get("physical", "relay_position_m")
    if direction == "downlink":
        scene = _scene(
            config,
            _planar_array(config.section("arrays", "gnb_tx")),
            _planar_array(config.section("arrays", "relay")),
            "gnb",
            gnb,
            "relay",
            relay,
        )
        num_rx_antennas = 1
        num_tx_antennas = int(config.get("arrays", "gnb_tx", "num_elements"))
    elif direction == "uplink":
        scene = _scene(
            config,
            _planar_array(config.section("arrays", "relay")),
            _planar_array(config.section("arrays", "gnb_rx")),
            "relay",
            relay,
            "gnb-rx",
            gnb,
        )
        num_rx_antennas = int(config.get("arrays", "gnb_rx", "num_elements"))
        num_tx_antennas = 1
    else:
        raise ChannelError(f"unknown relay direction {direction!r}")
    solver = config.section("solver", "path_solver")
    paths = PathSolver(deterministic=bool(solver["deterministic"]))(
        scene,
        max_depth=0,
        synthetic_array=True,
        los=True,
        specular_reflection=False,
        diffuse_reflection=bool(solver["diffuse_reflection"]),
        refraction=bool(solver["refraction"]),
        diffraction=bool(solver["diffraction"]),
        seed=int(solver["seed"]),
    )
    return _paths_to_channel(
        direction, paths, num_rx_antennas, num_tx_antennas
    )


def build_sionna_experiment_a(config: PilotConfig) -> SionnaExperimentA:
    profile = profile_from_config(config, "baseline")
    target = solve_target_channel(config)
    downlink = solve_relay_link(config, direction="downlink")
    uplink = solve_relay_link(config, direction="uplink")
    target_coefficient, target_delay = target.single_path()
    down_coefficient, down_delay = downlink.single_path()
    up_coefficient, up_delay = uplink.single_path()
    tx_array = ArrayGeometry.from_config(config)
    rx_values = config.section("arrays", "gnb_rx")
    rx_array = ArrayGeometry(
        num_elements=int(rx_values["num_elements"]),
        spacing_wavelengths=float(rx_values["spacing_wavelengths"]),
        wavelength_m=config.wavelength_m,
    )
    reference_phi = float(
        config.get("experiment_a", "reference_beam_phi_deg")
    )
    reference = direct_beam(
        reference_phi,
        dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
        tx_array,
        dtype=torch.complex64,
    )
    reference_rx = array_response(
        reference_phi, rx_array, dtype=torch.complex64
    )
    calibration = calibrate_matched_relay(
        target_coefficient=target_coefficient,
        target_delay_s=target_delay,
        downlink_coefficient=down_coefficient,
        downlink_delay_s=down_delay,
        uplink_coefficient=up_coefficient,
        uplink_delay_s=up_delay,
        reference_weights=reference.weights,
        reference_rx_weights=reference_rx,
        config=config,
        reference_beam_phi_deg=reference_phi,
    )
    return SionnaExperimentA(
        target=target,
        downlink=downlink,
        uplink=uplink,
        calibration=calibration,
        profile=profile,
    )


@dataclass(frozen=True)
class AnalyticalMatchedChannels:
    target_cfr: torch.Tensor
    relay_cfr: torch.Tensor
    target_delay_s: float
    downlink_delay_s: float
    uplink_delay_s: float
    added_delay_s: float
    gain: complex
    downlink_coefficient: float
    target_coefficient: float
    profile: OfdmProfile


def analytical_matched_channels(
    config: PilotConfig,
    *,
    dtype: torch.dtype = torch.complex128,
    profile_name: str = "baseline",
) -> AnalyticalMatchedChannels:
    profile = profile_from_config(config, profile_name)
    tx_array = ArrayGeometry.from_config(config)
    rx_values = config.section("arrays", "gnb_rx")
    rx_array = ArrayGeometry(
        num_elements=int(rx_values["num_elements"]),
        spacing_wavelengths=float(rx_values["spacing_wavelengths"]),
        wavelength_m=config.wavelength_m,
    )
    candidate_range = float(config.get("physical", "candidate_position_m")[0])
    relay_range = float(config.get("physical", "relay_position_m")[0])
    sigma = float(config.get("target", "rcs_m2"))
    wavelength = config.wavelength_m
    target_coefficient = (
        wavelength
        * sqrt(sigma)
        / ((4.0 * pi) ** 1.5 * candidate_range**2)
    )
    downlink_coefficient = wavelength / (4.0 * pi * relay_range)
    uplink_coefficient = downlink_coefficient
    target_delay = 2.0 * candidate_range / config.speed_of_light
    downlink_delay = relay_range / config.speed_of_light
    uplink_delay = relay_range / config.speed_of_light
    added_delay = target_delay - downlink_delay - uplink_delay
    gain = target_coefficient / (downlink_coefficient * uplink_coefficient)
    freqs = subcarrier_frequencies(
        profile.fft_size,
        profile.subcarrier_spacing_hz,
        device="cuda",
    ).to(torch.float64)
    absolute_freqs = config.frequency_hz + freqs
    tx_response = array_response(
        0.0, tx_array, device="cuda", dtype=torch.complex128
    )
    rx_response = array_response(
        0.0, rx_array, device="cuda", dtype=torch.complex128
    )
    target_phase = torch.exp(-2j * pi * absolute_freqs * target_delay)
    down_phase = torch.exp(-2j * pi * absolute_freqs * downlink_delay)
    up_phase = torch.exp(-2j * pi * absolute_freqs * uplink_delay)
    delay_phase = torch.exp(-2j * pi * absolute_freqs * added_delay)
    target_cfr = (
        target_coefficient
        * rx_response[:, None, None]
        * tx_response[None, :, None]
        * target_phase[None, None, :]
    )
    down_cfr = (
        downlink_coefficient
        * tx_response[:, None]
        * down_phase[None, :]
    )
    up_cfr = (
        uplink_coefficient
        * rx_response[:, None]
        * up_phase[None, :]
    )
    relay_cfr = (
        gain
        * up_cfr[:, None, :]
        * down_cfr[None, :, :]
        * delay_phase[None, None, :]
    )
    if target_cfr.shape != relay_cfr.shape:
        raise ChannelError(
            f"target {tuple(target_cfr.shape)} and relay "
            f"{tuple(relay_cfr.shape)} CFR shapes differ"
        )
    return AnalyticalMatchedChannels(
        target_cfr=target_cfr.to(dtype),
        relay_cfr=relay_cfr.to(dtype),
        target_delay_s=target_delay,
        downlink_delay_s=downlink_delay,
        uplink_delay_s=uplink_delay,
        added_delay_s=added_delay,
        gain=complex(gain),
        downlink_coefficient=downlink_coefficient,
        target_coefficient=target_coefficient,
        profile=profile,
    )


def effective_response(cfr: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if cfr.ndim == 3:
        antenna = cfr
    elif cfr.ndim == 7:
        if (
            cfr.shape[0] != 1
            or cfr.shape[1] != 1
            or cfr.shape[3] != 1
            or cfr.shape[5] != 1
        ):
            raise ChannelError(
                "expected one batch, receive, transmit, and time step, "
                f"got {tuple(cfr.shape)}"
            )
        antenna = cfr[0, 0, :, 0, :, 0, :]
    else:
        raise ChannelError(
            f"expected 3-D or 7-D CFR, got shape {tuple(cfr.shape)}"
        )
    if antenna.shape[1] != weights.shape[-1]:
        raise ChannelError(
            f"CFR TX antenna count {antenna.shape[1]} != "
            f"weights {weights.shape[-1]}"
        )
    weights = weights.to(antenna.dtype)
    if weights.ndim == 1:
        return torch.einsum("t,rtf->rf", weights, antenna)
    if weights.ndim == 2:
        return torch.einsum("bt,rtf->brf", weights, antenna)
    raise ChannelError("weights must be one- or two-dimensional")


def effective_cfr_for_path_channel(
    channel: PathChannel,
    profile: OfdmProfile,
    weights: torch.Tensor,
) -> torch.Tensor:
    return effective_response(channel.cfr(profile), weights)
