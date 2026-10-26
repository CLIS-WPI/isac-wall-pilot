from __future__ import annotations

from math import pi

import pytest
import torch

from isac_verify.beams import ArrayGeometry, dbm_to_watt, direct_beam
from isac_verify.channels import (
    SionnaExperimentA,
    build_sionna_experiment_a,
    effective_response,
)
from isac_verify.config import PilotConfig
from isac_verify.experiments import (
    analytical_comparison,
    beam_actions,
    delay_checks,
    noise_comparison,
    sionna_comparison,
)
from isac_verify.replay import (
    forwarded_relay_noise,
    noise_variance_w,
    proper_complex_noise,
    relay_noise_variance_w,
)


@pytest.fixture(scope="session")
def experiment_a(config: PilotConfig) -> SionnaExperimentA:
    return build_sionna_experiment_a(config)


def _reference_beam(config: PilotConfig) -> torch.Tensor:
    array = ArrayGeometry.from_config(config)
    power = dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))
    return direct_beam(
        float(config.get("experiment_a", "reference_beam_phi_deg")),
        power,
        array,
    ).weights


def test_radar_equation_single_target(
    config: PilotConfig, experiment_a: SionnaExperimentA
) -> None:
    profile = experiment_a.profile
    weights = _reference_beam(config)
    response = effective_response(experiment_a.target_cfr(), weights)
    assert response.shape == (16, profile.fft_size)
    measured_power = float(response.abs().square().mean().item())
    candidate_range = float(config.get("physical", "candidate_position_m")[0])
    sigma = float(config.get("target", "rcs_m2"))
    array = ArrayGeometry.from_config(config)
    expected_power = (
        dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))
        * array.num_elements
        * config.wavelength_m**2
        * sigma
        / ((4.0 * pi) ** 3 * candidate_range**4)
    )
    assert measured_power == pytest.approx(expected_power, rel=1e-3)
    assert profile.fft_size > 0
    assert profile.num_symbols > 0


def test_one_way_free_space_link_power(
    config: PilotConfig, experiment_a: SionnaExperimentA
) -> None:
    profile = experiment_a.profile
    weights = _reference_beam(config)
    response = effective_response(experiment_a.downlink.cfr(profile), weights)
    assert response.shape == (1, profile.fft_size)
    measured_power = float(response.abs().square().mean().item())
    relay_range = float(config.get("physical", "relay_position_m")[0])
    array = ArrayGeometry.from_config(config)
    expected_power = (
        dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))
        * array.num_elements
        * (config.wavelength_m / (4.0 * pi * relay_range)) ** 2
    )
    assert measured_power == pytest.approx(expected_power, rel=1e-3)


def test_absolute_delays_survive_cir_and_cfr(
    config: PilotConfig, experiment_a: SionnaExperimentA
) -> None:
    checks = delay_checks(config, experiment_a)
    assert checks["target_cir_shape"] == [1, 16, 1, 16, 1, 1]
    assert checks["downlink_cir_shape"] == [1, 1, 1, 16, 1, 1]
    assert checks["uplink_cir_shape"] == [1, 16, 1, 1, 1, 1]
    target_delay = float(checks["target_path_delay_ns"])
    down_delay = float(checks["downlink_path_delay_ns"])
    up_delay = float(checks["uplink_path_delay_ns"])
    assert target_delay > 1e-7
    assert down_delay > 0.0
    assert up_delay == pytest.approx(down_delay, rel=1e-6)
    assert target_delay == pytest.approx(
        2.0
        * float(config.get("physical", "candidate_position_m")[0])
        / config.speed_of_light
        * 1e9,
        rel=1e-5,
    )
    assert experiment_a.calibration.measured_added_delay_s * 1e9 == pytest.approx(
        target_delay - down_delay - up_delay,
        abs=1e-6,
    )
    assert abs(float(checks["target_cfr_delay_error_ns"])) < 1e-3
    assert abs(float(checks["downlink_cfr_delay_error_ns"])) < 1e-3
    assert abs(float(checks["relay_cfr_delay_error_ns"])) < 1e-3
    assert float(checks["relay_cfr_delay_ns"]) == pytest.approx(
        target_delay, abs=1e-3
    )


def test_experiment_a_indistinguishable_for_every_beam(
    config: PilotConfig, experiment_a: SionnaExperimentA
) -> None:
    analytical = analytical_comparison(config)
    sionna, _ = sionna_comparison(config, experiment_a)
    assert analytical.num_beams == len(beam_actions(config))
    assert analytical.num_beams == 70
    assert analytical.c128_errors.shape == (70, 16)
    assert analytical.c64_errors.shape == (70, 16)
    assert sionna.errors.shape == (70, 16)
    c128_tolerance = float(
        config.get("experiment_a", "analytical_c128_tolerance")
    )
    c64_tolerance = float(
        config.get("experiment_a", "analytical_c64_tolerance")
    )
    sionna_tolerance = float(
        config.get("experiment_a", "sionna_relative_tolerance")
    )
    assert analytical.c128_max_relative_error <= c128_tolerance
    assert analytical.c64_max_relative_error <= c64_tolerance
    assert sionna.max_relative_error <= sionna_tolerance
    assert bool((analytical.c128_errors <= c128_tolerance).all().item())
    assert bool((analytical.c64_errors <= c64_tolerance).all().item())
    assert bool((sionna.errors <= sionna_tolerance).all().item())
    assert analytical.c128_response_ratio_deviation <= 1e-10
    assert analytical.c64_response_ratio_deviation <= 1e-5
    assert sionna.response_ratio_deviation <= 1e-3
    assert (
        experiment_a.calibration.rx_steering_max_relative_difference
        <= c64_tolerance
    )


def test_noise_floor_and_forwarded_relay_noise(
    config: PilotConfig, experiment_a: SionnaExperimentA
) -> None:
    profile = experiment_a.profile
    variance = noise_variance_w(config, profile.sample_rate_hz)
    expected = (
        float(config.get("constants", "boltzmann_j_per_k"))
        * float(config.get("noise", "temperature_k"))
        * profile.sample_rate_hz
        * 10.0
        ** (float(config.get("noise", "receiver_noise_figure_db")) / 10.0)
    )
    assert variance == pytest.approx(expected, rel=1e-12)
    noise = proper_complex_noise(
        (profile.num_symbols, profile.fft_size),
        variance,
        int(config.get("experiment_a", "paired_noise_seed")),
    )
    floor = float(
        torch.fft.fft(noise, dim=-1, norm="ortho").abs().square().mean().item()
    )
    assert floor == pytest.approx(variance, rel=0.02)
    comparison = noise_comparison(config, experiment_a)
    assert comparison.paired_max_relative_error <= 1e-5
    relay_variance = relay_noise_variance_w(config, profile.sample_rate_hz)
    assert relay_variance == pytest.approx(variance, rel=1e-12)
    assert comparison.forwarded_relay_noise_mean_power_w > 0.0
    frequencies = torch.fft.fftfreq(
        profile.fft_size,
        d=1.0 / profile.sample_rate_hz,
        device="cuda",
    )
    uplink = effective_response(
        experiment_a.uplink.cfr(profile),
        torch.ones(1, device="cuda", dtype=torch.complex64),
    )
    forwarded = forwarded_relay_noise(
        uplink,
        experiment_a.calibration,
        variance_w=relay_variance,
        seed=int(config.get("experiment_a", "forwarded_relay_noise_seed")),
        shape=(profile.num_symbols, 16, profile.fft_size),
        frequencies_hz=frequencies,
    )
    expected_forwarded = (
        abs(experiment_a.calibration.gain) ** 2
        * float(uplink.abs().square().mean().item())
        * relay_variance
    )
    assert float(forwarded.abs().square().mean().item()) == pytest.approx(
        expected_forwarded, rel=0.02
    )
