"""Dense delay refinement must follow the coarse map's signal path (P8-4)."""

from __future__ import annotations

import math

import pytest
import torch

from isac_verify.beams import ArrayGeometry, array_response, dbm_to_watt, direct_beam
from isac_verify.config import PilotConfig
from isac_verify.phase7 import _observe
from isac_verify.phase8a import geometry_config, prepare_geometry
from isac_verify.receiver import (
    LinkPath,
    ProcessingChoices,
    _zero_padded_delay_profile,
    angular_delay_map,
    delay_doppler_map,
    estimate_candidate,
    make_observation,
    noise_calibration,
)
from isac_verify.wall_paths import path_records


def _direct_weights(config: PilotConfig) -> torch.Tensor:
    return direct_beam(
        float(config.get("experiment_a", "reference_beam_phi_deg")),
        dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
        ArrayGeometry.from_config(config),
    ).weights


def _plane_wave(
    choices: ProcessingChoices,
    weights: torch.Tensor,
    *,
    aoa_deg: float,
    delay_s: float,
    amplitude: float,
) -> LinkPath:
    """Path whose beamed response is ``amplitude`` times the RX response at ``aoa_deg``."""
    rx = array_response(aoa_deg, choices.receiver_array, dtype=torch.complex64)  # RT arrival convention (A12.1)
    tx = weights.conj() / weights.abs().square().sum()
    return LinkPath(
        label=f"synthetic:{aoa_deg}",
        family="synthetic",
        matrix=amplitude * rx[:, None] * tx[None, :],
        delay_s=delay_s,
        aod_deg=0.0,
        aoa_deg=aoa_deg,
        vertex_points_m=(),
    )


def _two_path_observation(config: PilotConfig, choices: ProcessingChoices, *, noiseless: bool):
    weights = _direct_weights(config)
    weak = _plane_wave(choices, weights, aoa_deg=0.0, delay_s=400e-9, amplitude=1e-6)
    strong = _plane_wave(choices, weights, aoa_deg=30.0, delay_s=440e-9, amplitude=3e-6)
    observation = make_observation(
        world="target",
        beam_name="test",
        beam_weights=weights,
        symbols=torch.ones(
            (choices.profile.num_symbols, choices.profile.fft_size),
            device="cuda",
            dtype=torch.complex64,
        ),
        target_paths=(weak, strong),
        profile=choices.profile,
        config=config,
        noise_seed=4242,
        relay_noise_seed=4243,
        noiseless=noiseless,
    )
    angular = angular_delay_map(
        delay_doppler_map(observation.h_eff, choices), choices, noise_per_cell_w=1e-30
    )
    return observation, angular


def test_dense_profile_matches_coarse_map_on_grid(config: PilotConfig) -> None:
    choices = ProcessingChoices.from_config(config)
    observation, angular = _two_path_observation(config, choices, noiseless=False)
    for angle_index in (int(torch.argmin(angular.aoa_deg.abs())), int(torch.argmin((angular.aoa_deg - 30.0).abs()))):
        angle = float(angular.aoa_deg[angle_index])
        dense, spacing = _zero_padded_delay_profile(observation.h_eff, choices, angle_deg=angle)
        oversample = int(round(choices.profile.delay_bin_spacing_s / spacing))
        coarse = angular.power_w[:, angle_index].to(torch.float64)
        on_grid = dense[::oversample][: coarse.numel()].to(torch.float64)
        peak = float(coarse.max())
        strong = coarse >= 1e-2 * peak
        relative = ((on_grid - coarse).abs() / coarse)[strong]
        # Per-cell relative error on cells within 20 dB of the peak; complex64
        # rounding dominates far below the peak, so those are bounded relative to the peak.
        assert float(relative.max()) <= 1e-5, float(relative.max())
        assert float((on_grid - coarse).abs().max()) <= 1e-5 * peak


def test_dense_profile_is_angularly_selective(config: PilotConfig) -> None:
    choices = ProcessingChoices.from_config(config)
    observation, angular = _two_path_observation(config, choices, noiseless=True)
    dense, spacing = _zero_padded_delay_profile(observation.h_eff, choices, angle_deg=0.0)
    peak_delay_s = int(torch.argmax(dense)) * spacing
    assert abs(peak_delay_s - 400e-9) <= 0.1e-9, peak_delay_s * 1e9


def test_narrow_corridor_candidate_is_the_direct_return(config: PilotConfig) -> None:
    """2h/D = 0.40 (cal_05-like): the reciprocal-wall pair must not capture the candidate."""
    choices = ProcessingChoices.from_config(config)
    config_g = geometry_config(
        config,
        {
            "candidate_distance_m": 65.0,
            "relay_distance_m": 30.0,
            "half_width_m": 13.0,
            "wall_thickness_m": 0.10,
        },
    )
    prepared = prepare_geometry(config_g)
    assert prepared["rejection"] is None
    true_delay_s = next(
        record["delay_ns"] * 1e-9
        for record in path_records(prepared["paths"]["target"], scene_label="test")
        if record["family"] == "direct"
    )
    noise = noise_calibration(choices=choices, config=config, records=4, seed=8801)
    weights = _direct_weights(config_g)
    links = prepared["links"]
    for world in ("target", "replay"):
        observation = _observe(
            config=config_g,
            world=world,
            beam_weights=weights,
            symbols=torch.ones(
                (choices.profile.num_symbols, choices.profile.fft_size),
                device="cuda",
                dtype=torch.complex64,
            ),
            profile=choices.profile,
            target_links=links["target"],
            background_links=links["background"],
            downlink_links=links["downlink"],
            uplink_links=links["uplink"],
            calibration=prepared["calibration"],
            probe_seed=0,
            noise_seed=0,
            relay_noise_seed=0,
            noiseless=True,
        )
        angular = angular_delay_map(
            delay_doppler_map(observation.h_eff, choices), choices, noise_per_cell_w=noise
        )
        candidate = estimate_candidate(
            angular,
            choices=choices,
            point_spread=None,
            gnb_position_m=config_g.get("physical", "gnb_position_m"),
            speed_of_light_m_per_s=config.speed_of_light,
            h_eff=observation.h_eff,
        )
        assert candidate.detected
        assert abs(candidate.delay_s - true_delay_s) <= 0.1e-9, (
            world,
            candidate.delay_s * 1e9,
            true_delay_s * 1e9,
        )


@pytest.mark.parametrize("decoy", ["below_300ns", "last_map_lag_beyond_500ns"])
def test_candidate_respects_declared_delay_search_window(
    config: PilotConfig, decoy: str
) -> None:
    """A stronger return outside delay_search_ns [300, 500] ns must not be acquired (P8-5)."""
    choices = ProcessingChoices.from_config(config)
    assert choices.delay_search_ns == (300.0, 500.0)
    if decoy == "below_300ns":
        decoy_delay_s, decoy_amplitude = 200e-9, 3e-6
    else:
        # The map's last lag (504.56 ns) is past 500 ns. The decoy wins that lag,
        # but its leakage one lag inside (-6.0 dB measured) stays below the direct return.
        spacing = choices.profile.delay_bin_spacing_s
        decoy_delay_s, decoy_amplitude = math.ceil(500e-9 / spacing) * spacing, 1.5e-6
        assert decoy_delay_s > 500e-9
    weights = _direct_weights(config)
    direct = _plane_wave(choices, weights, aoa_deg=0.0, delay_s=400e-9, amplitude=1e-6)
    decoy_path = _plane_wave(
        choices, weights, aoa_deg=0.0, delay_s=decoy_delay_s, amplitude=decoy_amplitude
    )
    observation = make_observation(
        world="target",
        beam_name="test",
        beam_weights=weights,
        symbols=torch.ones(
            (choices.profile.num_symbols, choices.profile.fft_size),
            device="cuda",
            dtype=torch.complex64,
        ),
        target_paths=(direct, decoy_path),
        profile=choices.profile,
        config=config,
        noise_seed=4242,
        relay_noise_seed=4243,
        noiseless=True,
    )
    angular = angular_delay_map(
        delay_doppler_map(observation.h_eff, choices), choices, noise_per_cell_w=1e-30
    )
    for h_eff in (None, observation.h_eff):
        candidate = estimate_candidate(
            angular,
            choices=choices,
            point_spread=None,
            gnb_position_m=config.get("physical", "gnb_position_m"),
            speed_of_light_m_per_s=config.speed_of_light,
            h_eff=h_eff,
        )
        tolerance_s = 0.1e-9 if h_eff is not None else 0.5 * angular.delay_bin_s
        assert abs(candidate.delay_s - 400e-9) <= tolerance_s, (
            decoy,
            h_eff is None,
            candidate.delay_s * 1e9,
        )
