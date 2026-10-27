"""Delay-axis and point-spread verification for the measured receiver."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .beams import ArrayGeometry, array_response, dbm_to_watt, direct_beam
from .config import PilotConfig
from .geometry import wall_delay_s
from .receiver import (
    LinkPath,
    ProcessingChoices,
    angular_delay_map,
    delay_doppler_map,
    estimate_candidate,
    make_observation,
    measure_point_spread,
)
from .verification import direct_gate_window
from .waveform import qpsk_symbols


@dataclass(frozen=True)
class DelayAxisRow:
    input_ns: float
    acquisition_ns: float
    psf_peak_ns: float
    gate_center_ns: float
    twin_prediction_ns: float
    max_module_error_ns: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_ns": self.input_ns,
            "acquisition_ns": self.acquisition_ns,
            "psf_peak_ns": self.psf_peak_ns,
            "gate_center_ns": self.gate_center_ns,
            "twin_prediction_ns": self.twin_prediction_ns,
            "max_module_error_ns": self.max_module_error_ns,
        }


def delay_axis_verification(
    config: PilotConfig,
    choices: ProcessingChoices,
    delays_ns: Sequence[float] = (
        300.0,
        400.276914,
        423.900055,
        441.726946,
        500.0,
    ),
) -> dict[str, Any]:
    """Run a noiseless single-path delay-axis check through each module."""
    array = ArrayGeometry.from_config(config)
    response = array_response(0.0, array, dtype=torch.complex64)
    matrix = torch.outer(response, response)
    beam = direct_beam(
        float(config.get("experiment_a", "reference_beam_phi_deg")),
        dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
        array,
    )
    symbols = qpsk_symbols(choices.profile, 1)
    rows: list[DelayAxisRow] = []
    for delay_ns in delays_ns:
        delay_s = float(delay_ns) * 1e-9
        path = LinkPath(
            label=f"axis:{delay_ns}",
            family="synthetic",
            matrix=matrix,
            delay_s=delay_s,
            aod_deg=0.0,
            aoa_deg=0.0,
            vertex_points_m=(),
        )
        observation = make_observation(
            world="background",
            beam_name="direct",
            beam_weights=beam.weights,
            symbols=symbols,
            target_paths=(path,),
            profile=choices.profile,
            config=config,
            noise_seed=1,
            relay_noise_seed=2,
            noiseless=True,
        )
        dd = delay_doppler_map(observation.h_eff, choices)
        angular = angular_delay_map(dd, choices, noise_per_cell_w=1e-30)
        candidate = estimate_candidate(
            angular,
            choices=choices,
            point_spread=None,
            gnb_position_m=config.get("physical", "gnb_position_m"),
            speed_of_light_m_per_s=config.speed_of_light,
            h_eff=observation.h_eff,
        )
        psf = measure_point_spread(
            angular, choices=choices, h_eff=observation.h_eff
        )
        gate = direct_gate_window(candidate, psf)
        range_m = 0.5 * config.speed_of_light * delay_s
        twin_prediction = wall_delay_s(
            range_m,
            float(config.get("physical", "walls", "positive", "offset_m")),
            config.speed_of_light,
        )
        errors = [
            abs(candidate.delay_s - delay_s),
            abs(psf.peak_delay_s - delay_s),
            abs(gate.center_delay_s - delay_s),
        ]
        rows.append(
            DelayAxisRow(
                input_ns=float(delay_ns),
                acquisition_ns=candidate.delay_s * 1e9,
                psf_peak_ns=psf.peak_delay_s * 1e9,
                gate_center_ns=gate.center_delay_s * 1e9,
                twin_prediction_ns=twin_prediction * 1e9,
                max_module_error_ns=max(errors) * 1e9,
            )
        )
    sample_rate = choices.profile.sample_rate_hz
    lag_spacing = choices.profile.delay_bin_spacing_s
    expected_spacing = 1.0 / (
        choices.profile.fft_size * choices.profile.subcarrier_spacing_hz
    )
    guard_l_min = int(
        math.ceil(choices.guard_delay_ns[0] * 1e-9 * sample_rate)
    )
    guard_l_max = int(
        math.floor(choices.guard_delay_ns[1] * 1e-9 * sample_rate)
    )
    guard_low_ns = guard_l_min * lag_spacing * 1e9
    guard_high_ns = guard_l_max * lag_spacing * 1e9
    return {
        "rows": [row.as_dict() for row in rows],
        "max_error_ns": max(row.max_module_error_ns for row in rows),
        "tolerance_ns": 0.1,
        "lag_spacing_ns": lag_spacing * 1e9,
        "expected_lag_spacing_ns": expected_spacing * 1e9,
        "lag_spacing_error_ns": abs(lag_spacing - expected_spacing) * 1e9,
        "guard_lag_range": [guard_l_min, guard_l_max],
        "guard_delay_ns": [guard_low_ns, guard_high_ns],
        "guard_configured_ns": list(choices.guard_delay_ns),
        "guard_max_error_ns": max(
            abs(guard_low_ns - choices.guard_delay_ns[0]),
            abs(guard_high_ns - choices.guard_delay_ns[1]),
        ),
        "source_lag_mapping": "delay = lag / (fft_size * subcarrier_spacing_hz)",
    }
