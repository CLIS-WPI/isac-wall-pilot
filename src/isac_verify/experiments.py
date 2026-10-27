"""Experiment A orchestration and Phase 3/4 validation.

Experiment A is a matched LoS negative control.  The analytical and Sionna
channels are compared after the same beam weights, with no per-link
normalization and with absolute delays.  The validation command also rechecks
the Phase 2 material, geometry, beam, and waveform references and the Phase 4
finite-wall scene and path invariants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import log10, pi, sqrt
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ks_2samp

from .beams import (
    ArrayGeometry,
    beam_gain,
    beampattern,
    dbm_to_watt,
    direct_beam,
    projected_wall_beam,
)
from .channels import (
    SionnaExperimentA,
    analytical_matched_channels,
    build_sionna_experiment_a,
    effective_response,
)
from .config import PilotConfig
from .geometry import image_method_reference
from .material import build_material_reference
from .replay import (
    forwarded_relay_noise,
    noise_variance_w,
    proper_complex_noise,
    relay_noise_variance_w,
)
from .waveform import (
    profile_from_config,
    qpsk_symbols,
    waveform_reference,
)


class ValidationError(RuntimeError):
    """Raised when a measured validation invariant is outside its tolerance."""


@dataclass(frozen=True)
class BeamAction:
    name: str
    kind: str
    angle_deg: float | None
    weights: torch.Tensor


@dataclass(frozen=True)
class AnalyticalComparison:
    c128_errors: torch.Tensor
    c64_errors: torch.Tensor
    c128_max_relative_error: float
    c64_max_relative_error: float
    c128_family_max_relative_error: dict[str, float]
    c64_family_max_relative_error: dict[str, float]
    c128_response_ratio_deviation: float
    c64_response_ratio_deviation: float
    num_beams: int


@dataclass(frozen=True)
class SionnaComparison:
    errors: torch.Tensor
    max_relative_error: float
    family_max_relative_error: dict[str, float]
    tau64_max_relative_error: float
    response_ratio_deviation: float
    target_power_w: torch.Tensor
    relay_power_w: torch.Tensor
    num_beams: int


@dataclass(frozen=True)
class NoiseComparison:
    variance_w: float
    relay_variance_w: float
    measured_floor_w: float
    floor_ratio: float
    paired_max_relative_error: float
    independent_target_mean_power_w: float
    independent_relay_mean_power_w: float
    independent_ks_statistic: float
    independent_ks_pvalue: float
    forwarded_relay_noise_mean_power_w: float


@dataclass(frozen=True)
class RatioNoiseComparison:
    num_records: int
    num_symbols: int
    feature_names: list[str]
    target_means: torch.Tensor
    target_stds: torch.Tensor
    relay_means: torch.Tensor
    relay_stds: torch.Tensor
    ks_statistics: torch.Tensor
    ks_pvalues: torch.Tensor
    max_ks_statistic: float
    min_ks_pvalue: float
    max_abs_mean_difference: float


def _random_beams(
    count: int,
    power_w: float,
    num_elements: int,
    seed: int,
) -> torch.Tensor:
    if count < 1:
        raise ValidationError("random beam count must be positive")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    real = torch.randn(
        (count, num_elements),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    imag = torch.randn(
        (count, num_elements),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    weights = real + 1j * imag
    norms = torch.linalg.vector_norm(weights, dim=1, keepdim=True)
    return (sqrt(power_w) * weights / norms).to(torch.complex64)


def beam_actions(config: PilotConfig) -> list[BeamAction]:
    array = ArrayGeometry.from_config(config)
    power = dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))
    reference_phi = float(config.get("experiment_a", "reference_beam_phi_deg"))
    actions = [
        BeamAction(
            name="direct",
            kind="direct",
            angle_deg=reference_phi,
            weights=direct_beam(reference_phi, power, array).weights,
        )
    ]
    for phi in config.get("experiment_a", "dither_deg"):
        actions.append(
            BeamAction(
                name=f"dither_{float(phi):+g}",
                kind="dither",
                angle_deg=float(phi),
                weights=direct_beam(float(phi), power, array).weights,
            )
        )
    random_weights = _random_beams(
        int(config.get("experiment_a", "num_random_beams")),
        power,
        array.num_elements,
        int(config.get("experiment_a", "random_beam_seed")),
    )
    for index, weights in enumerate(random_weights):
        actions.append(
            BeamAction(
                name=f"random_{index:03d}",
                kind="random",
                angle_deg=None,
                weights=weights,
            )
        )
    return actions


def _ideal_weights(actions: Sequence[BeamAction]) -> torch.Tensor:
    return torch.stack([action.weights for action in actions]).to(torch.complex128)


def _relative_error_per_beam(
    reference: torch.Tensor,
    measured: torch.Tensor,
    floor_fraction: float,
) -> torch.Tensor:
    if reference.shape != measured.shape:
        raise ValidationError(
            f"reference shape {tuple(reference.shape)} != measured "
            f"{tuple(measured.shape)}"
        )
    magnitude = reference.abs()
    scale = float(magnitude.max().item())
    floor = max(scale * float(floor_fraction), torch.finfo(magnitude.dtype).tiny)
    error = (reference - measured).abs() / magnitude.clamp_min(floor)
    return error.max(dim=2).values


def _response_ratio_deviation(
    reference: torch.Tensor, measured: torch.Tensor
) -> float:
    ratio = measured / reference
    return float((ratio - 1.0).abs().max().item())


def _family_max_relative_error(
    errors: torch.Tensor, actions: Sequence[BeamAction]
) -> dict[str, float]:
    per_beam = errors.max(dim=1).values
    families: dict[str, list[float]] = {}
    for index, action in enumerate(actions):
        families.setdefault(action.kind, []).append(
            float(per_beam[index].item())
        )
    return {name: max(values) for name, values in families.items()}


def analytical_comparison(config: PilotConfig) -> AnalyticalComparison:
    actions = beam_actions(config)
    weights = _ideal_weights(actions)
    floor_fraction = float(
        config.get("experiment_a", "relative_error_floor_fraction")
    )
    channels_128 = analytical_matched_channels(
        config, dtype=torch.complex128
    )
    channels_64 = analytical_matched_channels(config, dtype=torch.complex64)
    target_128 = effective_response(
        channels_128.target_cfr.to(torch.complex128), weights
    )
    relay_128 = effective_response(
        channels_128.relay_cfr.to(torch.complex128), weights
    )
    weights_64 = weights.to(torch.complex64)
    target_64 = effective_response(channels_64.target_cfr, weights_64)
    relay_64 = effective_response(channels_64.relay_cfr, weights_64)
    errors_128 = _relative_error_per_beam(target_128, relay_128, floor_fraction)
    errors_64 = _relative_error_per_beam(target_64, relay_64, floor_fraction)
    return AnalyticalComparison(
        c128_errors=errors_128,
        c64_errors=errors_64,
        c128_max_relative_error=float(errors_128.max().item()),
        c64_max_relative_error=float(errors_64.max().item()),
        c128_family_max_relative_error=_family_max_relative_error(
            errors_128, actions
        ),
        c64_family_max_relative_error=_family_max_relative_error(
            errors_64, actions
        ),
        c128_response_ratio_deviation=_response_ratio_deviation(
            target_128, relay_128
        ),
        c64_response_ratio_deviation=_response_ratio_deviation(
            target_64, relay_64
        ),
        num_beams=len(actions),
    )


def sionna_comparison(
    config: PilotConfig,
    experiment: SionnaExperimentA | None = None,
) -> tuple[SionnaComparison, SionnaExperimentA]:
    setup = experiment or build_sionna_experiment_a(config)
    actions = beam_actions(config)
    weights = torch.stack([action.weights for action in actions])
    target = setup.target_effective(weights)
    relay = setup.relay_effective(weights)
    floor_fraction = float(
        config.get("experiment_a", "relative_error_floor_fraction")
    )
    errors = _relative_error_per_beam(target, relay, floor_fraction)
    target_tau64 = setup.target_effective(weights, tau_float64=True)
    relay_tau64 = setup.relay_effective(weights, tau_float64=True)
    errors_tau64 = _relative_error_per_beam(
        target_tau64, relay_tau64, floor_fraction
    )
    return (
        SionnaComparison(
            errors=errors,
            max_relative_error=float(errors.max().item()),
            family_max_relative_error=_family_max_relative_error(
                errors, actions
            ),
            tau64_max_relative_error=float(errors_tau64.max().item()),
            response_ratio_deviation=_response_ratio_deviation(target, relay),
            target_power_w=target.abs().square().mean(dim=(1, 2)),
            relay_power_w=relay.abs().square().mean(dim=(1, 2)),
            num_beams=len(actions),
        ),
        setup,
    )


def _phase_delay_s(cfr: torch.Tensor, spacing_hz: float) -> float:
    if cfr.numel() < 2:
        raise ValidationError("need at least two subcarriers for delay")
    phase_step = torch.angle((cfr[1:] * cfr[:-1].conj()).mean())
    return float((-phase_step / (2.0 * pi * spacing_hz)).item())


def delay_checks(
    config: PilotConfig, experiment: SionnaExperimentA
) -> dict[str, Any]:
    profile = experiment.profile
    target_coefficient, target_delay = experiment.target.single_path()
    down_coefficient, down_delay = experiment.downlink.single_path()
    up_coefficient, up_delay = experiment.uplink.single_path()
    reference = direct_beam(
        float(config.get("experiment_a", "reference_beam_phi_deg")),
        dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
        ArrayGeometry.from_config(config),
    ).weights
    target_cfr = effective_response(experiment.target_cfr(), reference)
    down_cfr = effective_response(
        experiment.downlink.cfr(profile), reference
    )
    relay_cfr = experiment.relay_effective(reference)
    measured_target = _phase_delay_s(
        target_cfr[0], profile.subcarrier_spacing_hz
    )
    measured_down = _phase_delay_s(
        down_cfr[0], profile.subcarrier_spacing_hz
    )
    measured_relay = _phase_delay_s(
        relay_cfr[0], profile.subcarrier_spacing_hz
    )
    return {
        "target_path_delay_ns": target_delay * 1e9,
        "downlink_path_delay_ns": down_delay * 1e9,
        "uplink_path_delay_ns": up_delay * 1e9,
        "target_cfr_delay_ns": measured_target * 1e9,
        "downlink_cfr_delay_ns": measured_down * 1e9,
        "relay_cfr_delay_ns": measured_relay * 1e9,
        "target_cfr_delay_error_ns": (measured_target - target_delay) * 1e9,
        "downlink_cfr_delay_error_ns": (measured_down - down_delay) * 1e9,
        "relay_cfr_delay_error_ns": (
            measured_relay - experiment.calibration.target_delay_s
        )
        * 1e9,
        "target_cir_shape": list(experiment.target.a.shape),
        "downlink_cir_shape": list(experiment.downlink.a.shape),
        "uplink_cir_shape": list(experiment.uplink.a.shape),
        "target_coefficient_shape": list(target_coefficient.shape),
        "downlink_coefficient_shape": list(down_coefficient.shape),
        "uplink_coefficient_shape": list(up_coefficient.shape),
        "target_coefficient": [
            complex(target_coefficient[0, 0].item()).real,
            complex(target_coefficient[0, 0].item()).imag,
        ],
        "downlink_coefficient": [
            complex(down_coefficient[0, 0].item()).real,
            complex(down_coefficient[0, 0].item()).imag,
        ],
        "uplink_coefficient": [
            complex(up_coefficient[0, 0].item()).real,
            complex(up_coefficient[0, 0].item()).imag,
        ],
    }


def noise_comparison(
    config: PilotConfig, experiment: SionnaExperimentA
) -> NoiseComparison:
    profile = experiment.profile
    reference = direct_beam(
        float(config.get("experiment_a", "reference_beam_phi_deg")),
        dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
        ArrayGeometry.from_config(config),
    ).weights
    target = effective_response(experiment.target_cfr(), reference)
    relay = experiment.relay_effective(reference)
    symbols = qpsk_symbols(
        profile,
        int(config.get("experiment_a", "noise_symbol_seed")),
    )
    variance = noise_variance_w(config, profile.sample_rate_hz)
    noise_shape = (
        profile.num_symbols,
        target.shape[0],
        profile.fft_size,
    )
    paired_noise = proper_complex_noise(
        noise_shape,
        variance,
        int(config.get("experiment_a", "paired_noise_seed")),
    )
    signal = symbols[:, None, :]
    target_observation = signal * target[None, :, :] + paired_noise
    relay_observation = signal * relay[None, :, :] + paired_noise
    paired_error = float(
        (target_observation - relay_observation).abs().max().item()
        / target_observation.abs().max().item()
    )
    independent_target_noise = proper_complex_noise(
        noise_shape,
        variance,
        int(config.get("experiment_a", "independent_target_noise_seed")),
    )
    independent_relay_noise = proper_complex_noise(
        noise_shape,
        variance,
        int(config.get("experiment_a", "independent_relay_noise_seed")),
    )
    target_independent = signal * target[None, :, :] + independent_target_noise
    relay_independent = signal * relay[None, :, :] + independent_relay_noise
    target_powers = (
        target_independent.abs().square().reshape(-1).detach().cpu().numpy()
    )
    relay_powers = (
        relay_independent.abs().square().reshape(-1).detach().cpu().numpy()
    )
    ks = ks_2samp(target_powers, relay_powers)
    relay_variance = relay_noise_variance_w(config, profile.sample_rate_hz)
    frequencies = torch.fft.fftfreq(
        profile.fft_size,
        d=1.0 / profile.sample_rate_hz,
        device="cuda",
    )
    uplink_cfr = experiment.uplink.cfr(profile)[0, 0, :, 0, 0, 0, :]
    relay_noise_record = forwarded_relay_noise(
        uplink_cfr,
        experiment.calibration,
        variance_w=relay_variance,
        seed=int(config.get("experiment_a", "forwarded_relay_noise_seed")),
        shape=(profile.num_symbols, experiment.uplink.num_rx_antennas, profile.fft_size),
        frequencies_hz=frequencies,
    )
    measured_floor = float(
        torch.fft.fft(paired_noise, dim=-1, norm="ortho")
        .abs()
        .square()
        .mean()
        .item()
    )
    return NoiseComparison(
        variance_w=variance,
        relay_variance_w=relay_variance,
        measured_floor_w=measured_floor,
        floor_ratio=measured_floor / variance,
        paired_max_relative_error=paired_error,
        independent_target_mean_power_w=float(
            target_independent.abs().square().mean().item()
        ),
        independent_relay_mean_power_w=float(
            relay_independent.abs().square().mean().item()
        ),
        independent_ks_statistic=float(ks.statistic),
        independent_ks_pvalue=float(ks.pvalue),
        forwarded_relay_noise_mean_power_w=float(
            relay_noise_record.abs().square().mean().item()
        ),
    )


def ratio_noise_comparison(
    config: PilotConfig, experiment: SionnaExperimentA
) -> RatioNoiseComparison:
    actions = beam_actions(config)
    weights = torch.stack([action.weights for action in actions])
    target = experiment.target_effective(weights)
    relay = experiment.relay_effective(weights)
    profile = experiment.profile
    num_records = int(
        config.get("experiment_a", "ratio_noise_records")
    )
    num_symbols = int(
        config.get("experiment_a", "ratio_noise_symbols")
    )
    if num_records < 2 or num_symbols < 1:
        raise ValidationError("ratio-noise record and symbol counts must be positive")
    symbols = qpsk_symbols(
        profile,
        int(config.get("experiment_a", "ratio_symbol_seed")),
    )[:num_symbols]
    variance = noise_variance_w(config, profile.sample_rate_hz)
    shape = (num_symbols, len(actions), target.shape[1], profile.fft_size)
    ratio_records: dict[str, torch.Tensor] = {}
    for world, response, base_seed in (
        (
            "target",
            target,
            int(config.get("experiment_a", "ratio_target_noise_seed")),
        ),
        (
            "relay",
            relay,
            int(config.get("experiment_a", "ratio_relay_noise_seed")),
        ),
    ):
        records = []
        for record in range(num_records):
            noise = proper_complex_noise(
                shape,
                variance,
                base_seed + record,
            )
            measured = (
                symbols[:, None, None, :]
                * response[None, :, :, :]
                + noise
            )
            beam_power = measured.abs().square().mean(dim=(0, 2, 3))
            records.append((beam_power[1:] / beam_power[0]).detach())
        ratio_records[world] = torch.stack(records)
        del noise, measured, beam_power
    target_records = ratio_records["target"]
    relay_records = ratio_records["relay"]
    target_means = target_records.mean(dim=0)
    target_stds = target_records.std(dim=0, unbiased=True)
    relay_means = relay_records.mean(dim=0)
    relay_stds = relay_records.std(dim=0, unbiased=True)
    ks_statistics = []
    ks_pvalues = []
    for index in range(target_records.shape[1]):
        result = ks_2samp(
            target_records[:, index].detach().cpu().numpy(),
            relay_records[:, index].detach().cpu().numpy(),
        )
        ks_statistics.append(float(result.statistic))
        ks_pvalues.append(float(result.pvalue))
    statistics = torch.tensor(ks_statistics, dtype=torch.float64)
    pvalues = torch.tensor(ks_pvalues, dtype=torch.float64)
    feature_names = [action.name for action in actions[1:]]
    return RatioNoiseComparison(
        num_records=num_records,
        num_symbols=num_symbols,
        feature_names=feature_names,
        target_means=target_means.detach().cpu(),
        target_stds=target_stds.detach().cpu(),
        relay_means=relay_means.detach().cpu(),
        relay_stds=relay_stds.detach().cpu(),
        ks_statistics=statistics,
        ks_pvalues=pvalues,
        max_ks_statistic=float(statistics.max().item()),
        min_ks_pvalue=float(pvalues.min().item()),
        max_abs_mean_difference=float(
            (target_means - relay_means).abs().max().item()
        ),
    )


def phase2_checks(config: PilotConfig) -> dict[str, Any]:
    material = build_material_reference(config)
    geometry = image_method_reference(
        candidate_range_m=float(config.get("geometry_reference", "candidate_range_m")),
        relay_range_m=float(config.get("geometry_reference", "relay_range_m")),
        wall_offset_m=float(config.get("geometry_reference", "wall_offset_m")),
        speed_of_light_m_per_s=config.speed_of_light,
        latency_floor_s=float(
            config.get("geometry_reference", "latency_floor_ns")
        )
        * 1e-9,
    )
    array = ArrayGeometry.from_config(config)
    power = dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))
    wall_beam = projected_wall_beam(
        float(config.get("geometry_reference", "wall_departure_candidate_deg")),
        0.0,
        power,
        array,
        min_projected_norm=float(config.get("beam", "projected_norm_floor")),
        dtype=torch.complex128,
    )
    if not wall_beam.usable:
        raise ValidationError("Phase 2 projected wall beam is unusable")
    scan = np.arange(-90.0, 90.0001, 0.01)
    pattern = beampattern(scan, wall_beam, array, dtype=torch.complex128)
    peak_gain = float(pattern.max().item())
    direct_gain = beam_gain(0.0, wall_beam, array, dtype=torch.complex128)
    null_depth_db = 10.0 * log10(max(direct_gain, 1e-300) / peak_gain)
    waveform = waveform_reference(
        config,
        int(config.get("seeds", "streams", "probe_symbols", "calibration")[0]),
    )
    power_errors = [
        abs(entry["measured_mean_power_w"] - power) / power
        for entry in waveform["entries"]
    ]
    return {
        "status": "PASS"
        if null_depth_db <= float(config.get("validation", "null_depth_min_db"))
        and max(power_errors)
        <= float(config.get("waveform", "power_relative_tolerance"))
        else "FAIL",
        "material_model": material["material_model"],
        "material_thickness_m": material["material"]["thickness_m"],
        "geometry": {
            "candidate_direct_round_trip_delay_ns": geometry[
                "candidate_direct_round_trip_delay_s"
            ]
            * 1e9,
            "relay_added_delay_ns": geometry["relay_added_delay_s"] * 1e9,
            "wall_departure_candidate_deg": geometry[
                "wall_departure_candidate_deg"
            ],
        },
        "null_depth_db": null_depth_db,
        "waveform_max_relative_power_error": max(power_errors),
    }


def _check_runtime(require_h100: bool) -> dict[str, Any]:
    import mitsuba as mi

    if not torch.cuda.is_available():
        raise ValidationError("torch.cuda.is_available() is False")
    name = torch.cuda.get_device_name(0)
    if require_h100 and "H100" not in name:
        raise ValidationError(f"GPU name {name!r} does not contain H100")
    variant = mi.variant()
    if variant is None or not variant.startswith("cuda"):
        raise ValidationError(
            f"Mitsuba variant {variant!r} is not CUDA-backed"
        )
    return {
        "gpu_name": name,
        "cuda_available": True,
        "mitsuba_variant": variant,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _figure(
    output: Path,
    analytical: AnalyticalComparison,
    sionna: SionnaComparison,
    config: PilotConfig,
) -> None:
    index = np.arange(analytical.num_beams)
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.semilogy(
        index,
        analytical.c128_errors.amax(dim=1).detach().cpu().numpy(),
        label="analytical complex128",
        linewidth=1.2,
    )
    axis.semilogy(
        index,
        analytical.c64_errors.amax(dim=1).detach().cpu().numpy(),
        label="analytical complex64",
        linewidth=1.2,
    )
    axis.semilogy(
        index,
        sionna.errors.amax(dim=1).detach().cpu().numpy(),
        label="Sionna LoS",
        linewidth=1.2,
    )
    axis.axhline(
        float(config.get("experiment_a", "analytical_c128_tolerance")),
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="complex128 tolerance",
    )
    axis.axhline(
        float(config.get("experiment_a", "analytical_c64_tolerance")),
        color="tab:gray",
        linestyle=":",
        linewidth=0.9,
        label="complex64 tolerance",
    )
    axis.axhline(
        float(config.get("experiment_a", "sionna_relative_tolerance")),
        color="tab:red",
        linestyle="-.",
        linewidth=0.9,
        label="Sionna tolerance",
    )
    axis.set_xlabel("beam action index")
    axis.set_ylabel("maximum relative target/relay difference")
    axis.set_title("Experiment A matched LoS negative control")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output / "experiment_a.png", dpi=180)
    figure.savefig(output / "experiment_a.pdf")
    plt.close(figure)


def phase4_diagnostics(
    config: PilotConfig, output_path: str | Path = "results/phase4"
) -> dict[str, Any]:
    from .wall_paths import write_phase4_outputs

    data = write_phase4_outputs(config, Path(output_path))
    nominal = data["nominal"]
    background = data["background"]
    convergence = data["convergence"]
    blocked = data["blocked"]
    twin = data["twin"]
    d8 = data["d8"]

    core = nominal["core_paths"]
    expected_core = {
        "direct",
        "outgoing_wall_positive_wall",
        "outgoing_wall_negative_wall",
        "reciprocal_wall_positive_wall",
        "reciprocal_wall_negative_wall",
    }
    comparisons = nominal["outgoing_wall_comparisons"]
    image_checks = all(
        abs(item["delay_error_ns"]) <= 1e-3
        and abs(item["aod_error_deg"]) <= 1e-2
        and abs(item["aoa_error_deg"]) <= 1e-6
        and item["reflection_point_error_m"] <= 1e-3
        for item in comparisons
    )
    amplitude_checks = all(
        abs(item["amplitude_relative_error"]) <= 1e-3 for item in comparisons
    )
    reciprocity_checks = True
    for item in comparisons:
        side = item["wall"]
        outgoing = item["measured"]
        reciprocal = core[f"reciprocal_wall_{side}"][0]
        reciprocity_checks = reciprocity_checks and (
            abs(reciprocal["delay_ns"] - outgoing["delay_ns"]) <= 1e-6
            and abs(reciprocal["aod_deg"]) <= 1e-6
            and abs(
                reciprocal["aoa_deg"] - item["image_method"]["aod_deg"]
            )
            <= 1e-2
            and abs(
                reciprocal["amplitude"] - outgoing["amplitude"]
            )
            <= 1e-4 * outgoing["amplitude"]
        )

    clutter = background["monostatic_wall_clutter"]
    clutter_checks = len(clutter) == 2 and all(
        abs(item["delay_error_ns"]) <= 1e-3
        and abs(item["aod_error_deg"]) <= 1e-2
        and abs(item["aoa_error_deg"]) <= 1e-2
        and item["reflection_point_error_m"] <= 1e-3
        for item in clutter
    )
    relay_wall = [
        item
        for item in background["relay_links"]
        if item["family"].startswith("wall_clutter_")
    ]
    downlink_wall = [
        item for item in relay_wall if item["direction"] == "downlink"
    ]
    relay_checks = (
        len(downlink_wall) == 2
        and all(
            abs(item["delay_error_ns"]) <= 1e-3
            and abs(item["aod_error_deg"]) <= 1e-2
            and item["reflection_point_error_m"] <= 1e-3
            for item in relay_wall
        )
        and sorted(round(item["aod_deg"], 3) for item in downlink_wall)
        == [-45.0, 45.0]
    )
    summaries = convergence["summaries"]
    path_counts = [item["path_count"] for item in summaries]
    convergence_checks = (
        int(config.get("scene", "nominal_depth")) == 2
        and path_counts[0] == 1
        and path_counts[1] == 5
        and all(
            later >= earlier
            for earlier, later in zip(path_counts, path_counts[1:])
        )
        and abs(summaries[-1]["total_power_change_fraction"]) <= 1e-2
    )
    blocked_checks = (
        blocked["positive_outgoing_present"] is False
        and blocked["positive_reciprocal_present"] is False
        and blocked["negative_outgoing_present"] is True
        and set(blocked["removed_families"])
        == {
            "outgoing_wall_positive_wall",
            "reciprocal_wall_positive_wall",
        }
        and blocked["blocked_path_count"] < blocked["unblocked_path_count"]
        and blocked["new_paths"] == []
    )
    twin_checks = all(
        abs(row["rt_delay_shift_ns"] - row["image_delay_shift_ns"]) <= 1e-3
        and abs(row["rt_aod_shift_deg"] - row["image_aod_shift_deg"]) <= 1e-2
        and abs(row["image_twin_vs_rt_twin_delay_ns"]) <= 1e-3
        and abs(row["image_twin_vs_rt_twin_aod_deg"]) <= 1e-2
        and row["image_delay_shift_ns"] > 0.0
        for row in twin["rows"]
    )
    d8_checks = (
        d8["thicknesses_m"] == [0.05, 0.10, 0.20]
        and d8["angles_deg"] == [63.435, 45.0]
        and len(d8["entries"]) == 6
        and d8["baseline_thickness_m"]
        == float(config.get("wall", "thickness_m"))
    )
    checks = {
        "phase4_finite_wall_paths": set(core) == expected_core,
        "phase4_image_method": image_checks,
        "phase4_amplitude": amplitude_checks,
        "phase4_reciprocity": reciprocity_checks,
        "phase4_background_clutter": clutter_checks,
        "phase4_relay_wall": relay_checks,
        "phase4_depth_convergence": convergence_checks,
        "phase4_blocked_path": blocked_checks,
        "phase4_twin_shift": twin_checks,
        "phase4_d8_reference": d8_checks,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "status": status,
        "checks": checks,
        "outputs": {
            "paths_diagnostic": str(Path(output_path) / "paths_diagnostic.parquet"),
            "geometry_reference": str(Path(output_path) / "geometry_reference.json"),
        },
        "finite_wall_reflection": {
            "core_paths": core,
            "outgoing_wall_comparisons": comparisons,
        },
        "background": {
            "monostatic_wall_clutter": clutter,
            "relay_wall_paths": relay_wall,
            "all_relay_links": background["relay_links"],
        },
        "depth_convergence": summaries,
        "blocked": {
            "blocker": blocked["blocker_scene_metadata"],
            "unblocked_path_count": blocked["unblocked_path_count"],
            "blocked_path_count": blocked["blocked_path_count"],
            "removed_families": blocked["removed_families"],
            "new_paths": blocked["new_paths"],
        },
        "twin": {
            "epsilon_plus_m": twin["epsilon_plus_m"],
            "epsilon_minus_m": twin["epsilon_minus_m"],
            "rows": twin["rows"],
            "wall_geometry": twin["wall_geometry"],
        },
        "d8_wall_thickness": d8,
        "labels": {
            "d8": "follow-up wall-thickness stress; not the BRIEF baseline",
            "background": (
                "monostatic self-link is diagnostic only; D6 ideal isolation "
                "still excludes direct gNB leakage from the experiment"
            ),
        },
    }
    (Path(output_path) / "phase4_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _print_phase4_tables(phase4: dict[str, Any]) -> None:
    print("Phase 4 finite-wall target paths:")
    print(
        "wall              kind         delay_ns     AoD_deg    AoA_deg"
        "    |a|          interactions"
    )
    core = phase4["finite_wall_reflection"]["core_paths"]
    for family in (
        "direct",
        "outgoing_wall_positive_wall",
        "reciprocal_wall_positive_wall",
        "outgoing_wall_negative_wall",
        "reciprocal_wall_negative_wall",
    ):
        for record in core[family]:
            print(
                f"{family:17s} {record['path_signature']:12s} "
                f"{record['delay_ns']:10.6f} {record['aod_deg']:10.6f} "
                f"{record['aoa_deg']:10.6f} {record['amplitude']:.6e} "
                f"{'>'.join(record['interaction_names'])}"
            )
    print("Phase 4 finite-wall image-method comparisons:")
    print(
        "wall              delay_err_ns  AoD_err_deg  AoA_err_deg"
        "  point_err_m  amplitude_rel_err"
    )
    for item in phase4["finite_wall_reflection"]["outgoing_wall_comparisons"]:
        print(
            f"{item['wall']:17s} {item['delay_error_ns']:12.6e} "
            f"{item['aod_error_deg']:12.6e} {item['aoa_error_deg']:12.6e} "
            f"{item['reflection_point_error_m']:12.6e} "
            f"{item['amplitude_relative_error']:12.6e}"
        )
    print("Phase 4 monostatic wall clutter:")
    print("wall              delay_ns     AoD_deg    AoA_deg")
    for item in phase4["background"]["monostatic_wall_clutter"]:
        print(
            f"{item['family']:17s} {item['delay_ns']:10.6f} "
            f"{item['aod_deg']:10.6f} {item['aoa_deg']:10.6f}"
        )
    print("Phase 4 relay wall paths:")
    print("direction  wall              delay_ns     AoD_deg    AoA_deg")
    for item in phase4["background"]["relay_wall_paths"]:
        print(
            f"{item['direction']:9s} {item['family']:17s} "
            f"{item['delay_ns']:10.6f} {item['aod_deg']:10.6f} "
            f"{item['aoa_deg']:10.6f}"
        )
    print("Phase 4 depth convergence:")
    print("depth  paths  total_power_W  power_change_fraction  families")
    for item in phase4["depth_convergence"]:
        fraction = item.get("total_power_change_fraction")
        print(
            f"{item['max_depth']:5d} {item['path_count']:6d} "
            f"{item['total_path_power_w']:.6e} "
            f"{'n/a' if fraction is None else f'{fraction:.6e}':>21s} "
            f"{item['family_counts']}"
        )
    print("Phase 4 blocked contribution:")
    print(
        f"blocker={phase4['blocked']['blocker']['name']} "
        f"center={phase4['blocked']['blocker']['center_m']}"
    )
    print(
        f"paths {phase4['blocked']['unblocked_path_count']} -> "
        f"{phase4['blocked']['blocked_path_count']}; "
        f"removed={phase4['blocked']['removed_families']}; "
        f"new={len(phase4['blocked']['new_paths'])}"
    )
    print("Phase 4 twin sweep:")
    print(
        "wall              image_delay_shift_ns  rt_delay_shift_ns"
        "  image_AoD_shift_deg  rt_AoD_shift_deg"
    )
    for item in phase4["twin"]["rows"]:
        print(
            f"{item['wall']:17s} {item['image_delay_shift_ns']:20.9f} "
            f"{item['rt_delay_shift_ns']:17.9f} "
            f"{item['image_aod_shift_deg']:20.9f} "
            f"{item['rt_aod_shift_deg']:16.9f}"
        )
    print("Phase 4 D8 wall-thickness stress (follow-up, not baseline):")
    print("thickness_m  angle_deg  |r_TE|      |r_TM|")
    for item in phase4["d8_wall_thickness"]["entries"]:
        print(
            f"{item['thickness_m']:11.3f} {item['incidence_angle_deg']:9.3f} "
            f"{item['r_te']['magnitude']:.6f} {item['r_tm']['magnitude']:.6f}"
        )


def run_validation(
    config_path: str | Path,
    require_h100: bool,
    output_path: str | Path = "results/validate",
    phase4_output_path: str | Path = "results/phase4",
) -> int:
    config = PilotConfig.load(config_path)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    runtime = _check_runtime(require_h100)
    phase2 = phase2_checks(config)
    analytical = analytical_comparison(config)
    sionna, experiment = sionna_comparison(config)
    delays = delay_checks(config, experiment)
    noise = noise_comparison(config, experiment)
    ratio_noise = ratio_noise_comparison(config, experiment)
    phase4 = phase4_diagnostics(config, phase4_output_path)
    c128_tolerance = float(
        config.get("experiment_a", "analytical_c128_tolerance")
    )
    c64_tolerance = float(
        config.get("experiment_a", "analytical_c64_tolerance")
    )
    sionna_tolerance = float(
        config.get("experiment_a", "sionna_relative_tolerance")
    )
    noise_tolerance = float(
        config.get("experiment_a", "noise_floor_relative_tolerance")
    )
    checks = {
        "phase2": phase2["status"] == "PASS",
        "analytical_c128": analytical.c128_max_relative_error
        <= c128_tolerance,
        "analytical_c64": analytical.c64_max_relative_error <= c64_tolerance,
        "sionna": sionna.max_relative_error <= sionna_tolerance,
        "noise_floor": abs(noise.floor_ratio - 1.0) <= noise_tolerance,
        "paired_noise": noise.paired_max_relative_error <= 1e-5,
        "relay_power_cap": experiment.calibration.power_feasible,
        "relay_latency": experiment.calibration.latency_feasible,
        "rx_steering": (
            experiment.calibration.rx_steering_max_relative_difference
            <= c64_tolerance
        ),
        **phase4["checks"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "status": status,
        "checks": checks,
        "runtime": runtime,
        "d6": {
            "self_link": "excluded from background computations",
            "assumption": "ideal gNB TX/RX isolation; direct leakage cancelled",
            "relay_links": "gNB-to-relay and relay-to-gNB solved separately",
            "phase4_diagnostic_note": (
                "monostatic finite-wall self-link solved only to expose "
                "background clutter; the D6 ideal-isolation model still "
                "excludes direct gNB leakage from the experiment"
            ),
        },
        "phase2": phase2,
        "experiment_a": {
            "num_beams": analytical.num_beams,
            "reference_beam_phi_deg": float(
                config.get("experiment_a", "reference_beam_phi_deg")
            ),
            "dither_deg": config.get("experiment_a", "dither_deg"),
            "num_random_beams": int(
                config.get("experiment_a", "num_random_beams")
            ),
            "random_beam_seed": int(
                config.get("experiment_a", "random_beam_seed")
            ),
            "analytical_c128_max_relative_error": analytical.c128_max_relative_error,
            "analytical_c64_max_relative_error": analytical.c64_max_relative_error,
            "analytical_c128_family_max_relative_error": (
                analytical.c128_family_max_relative_error
            ),
            "analytical_c64_family_max_relative_error": (
                analytical.c64_family_max_relative_error
            ),
            "analytical_c128_tolerance": c128_tolerance,
            "analytical_c64_tolerance": c64_tolerance,
            "analytical_c128_response_ratio_deviation": analytical.c128_response_ratio_deviation,
            "analytical_c64_response_ratio_deviation": analytical.c64_response_ratio_deviation,
            "sionna_max_relative_error": sionna.max_relative_error,
            "sionna_family_max_relative_error": (
                sionna.family_max_relative_error
            ),
            "sionna_tau64_max_relative_error": sionna.tau64_max_relative_error,
            "sionna_tolerance": sionna_tolerance,
            "sionna_response_ratio_deviation": sionna.response_ratio_deviation,
            "rx_steering_max_relative_difference": (
                experiment.calibration.rx_steering_max_relative_difference
            ),
            "target_cir_shape": delays["target_cir_shape"],
            "downlink_cir_shape": delays["downlink_cir_shape"],
            "uplink_cir_shape": delays["uplink_cir_shape"],
            "sionna_target_power_min_w": float(
                sionna.target_power_w.min().item()
            ),
            "sionna_target_power_max_w": float(
                sionna.target_power_w.max().item()
            ),
            "sionna_relay_power_min_w": float(
                sionna.relay_power_w.min().item()
            ),
            "sionna_relay_power_max_w": float(
                sionna.relay_power_w.max().item()
            ),
        },
        "calibration": experiment.calibration.as_dict(),
        "delays": delays,
        "ratio_noise": {
            "num_records": ratio_noise.num_records,
            "num_symbols": ratio_noise.num_symbols,
            "feature_names": ratio_noise.feature_names,
            "target_means": ratio_noise.target_means,
            "target_stds": ratio_noise.target_stds,
            "relay_means": ratio_noise.relay_means,
            "relay_stds": ratio_noise.relay_stds,
            "ks_statistics": ratio_noise.ks_statistics,
            "ks_pvalues": ratio_noise.ks_pvalues,
            "max_ks_statistic": ratio_noise.max_ks_statistic,
            "min_ks_pvalue": ratio_noise.min_ks_pvalue,
            "max_abs_mean_difference": ratio_noise.max_abs_mean_difference,
            "definition": (
                "per non-direct beam action, ratio of mean measured power "
                "over symbols, receive elements, and subcarriers to the "
                "direct-beam measured power; target and relay use independent "
                "proper-complex noise records"
            ),
        },
        "phase4": phase4,
        "noise": {
            "variance_w": noise.variance_w,
            "relay_variance_w": noise.relay_variance_w,
            "measured_floor_w": noise.measured_floor_w,
            "floor_ratio": noise.floor_ratio,
            "paired_max_relative_error": noise.paired_max_relative_error,
            "independent_target_mean_power_w": noise.independent_target_mean_power_w,
            "independent_relay_mean_power_w": noise.independent_relay_mean_power_w,
            "independent_ks_statistic": noise.independent_ks_statistic,
            "independent_ks_pvalue": noise.independent_ks_pvalue,
            "forwarded_relay_noise_mean_power_w": noise.forwarded_relay_noise_mean_power_w,
        },
    }
    (output / "validation_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8"
    )
    _figure(output, analytical, sionna, config)
    print(f"validation status: {status}")
    print(
        "analytical c128 max relative error: "
        f"{analytical.c128_max_relative_error:.3e}"
    )
    print(
        "analytical c64 max relative error: "
        f"{analytical.c64_max_relative_error:.3e}"
    )
    print(f"Sionna max relative error: {sionna.max_relative_error:.3e}")
    print(
        "Sionna max relative error with float64 tau: "
        f"{sionna.tau64_max_relative_error:.3e}"
    )
    print(
        "family max relative error (c128/c64/Sionna): "
        f"direct={analytical.c128_family_max_relative_error['direct']:.3e}/"
        f"{analytical.c64_family_max_relative_error['direct']:.3e}/"
        f"{sionna.family_max_relative_error['direct']:.3e}; "
        f"dither={analytical.c128_family_max_relative_error['dither']:.3e}/"
        f"{analytical.c64_family_max_relative_error['dither']:.3e}/"
        f"{sionna.family_max_relative_error['dither']:.3e}; "
        f"random={analytical.c128_family_max_relative_error['random']:.3e}/"
        f"{analytical.c64_family_max_relative_error['random']:.3e}/"
        f"{sionna.family_max_relative_error['random']:.3e}"
    )
    print(
        "ratio features: "
        f"{ratio_noise.num_records} records x {ratio_noise.num_symbols} symbols, "
        f"max KS={ratio_noise.max_ks_statistic:.6f}, "
        f"min p={ratio_noise.min_ks_pvalue:.6f}, "
        f"max |mean target-relay|={ratio_noise.max_abs_mean_difference:.6e}"
    )
    print(
        "ratio mean/std ranges: target mean "
        f"[{float(ratio_noise.target_means.min()):.6e}, "
        f"{float(ratio_noise.target_means.max()):.6e}], "
        f"target std [{float(ratio_noise.target_stds.min()):.6e}, "
        f"{float(ratio_noise.target_stds.max()):.6e}], "
        f"relay mean [{float(ratio_noise.relay_means.min()):.6e}, "
        f"{float(ratio_noise.relay_means.max()):.6e}], "
        f"relay std [{float(ratio_noise.relay_stds.min()):.6e}, "
        f"{float(ratio_noise.relay_stds.max()):.6e}]"
    )
    print(f"target CIR shape: {delays['target_cir_shape']}")
    print(f"downlink CIR shape: {delays['downlink_cir_shape']}")
    print(f"uplink CIR shape: {delays['uplink_cir_shape']}")
    print(
        "RX steering max relative difference: "
        f"{experiment.calibration.rx_steering_max_relative_difference:.3e}"
    )
    print(
        "measured delta: "
        f"{experiment.calibration.measured_added_delay_s * 1e9:.9f} ns"
    )
    print(
        "analytical delta: "
        f"{experiment.calibration.analytical_added_delay_s * 1e9:.9f} ns"
    )
    print(
        "delay difference: "
        f"{experiment.calibration.delay_difference_s * 1e9:.9f} ns"
    )
    print(
        "relay gain: "
        f"{experiment.calibration.gain.real:.6f}"
        f"{experiment.calibration.gain.imag:+.6f}j"
    )
    print(
        "relay transmit power: "
        f"{experiment.calibration.relay_transmit_power_w:.6e} W "
        f"(cap {experiment.calibration.max_transmit_power_w:.6e} W, "
        f"feasible={experiment.calibration.power_feasible})"
    )
    print(
        "receiver noise variance: "
        f"{noise.variance_w:.6e} W, measured floor ratio "
        f"{noise.floor_ratio:.6f}"
    )
    print(f"phase 4 status: {phase4['status']}")
    _print_phase4_tables(phase4)
    print(f"wrote {output / 'validation_summary.json'}")
    print(f"wrote {Path(phase4_output_path) / 'phase4_summary.json'}")
    print(f"wrote {Path(phase4_output_path) / 'paths_diagnostic.parquet'}")
    print(f"wrote {Path(phase4_output_path) / 'geometry_reference.json'}")
    print(f"wrote {output / 'experiment_a.png'}")
    print(f"wrote {output / 'experiment_a.pdf'}")
    if status != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise ValidationError(f"validation failed: {', '.join(failed)}")
    return 0
