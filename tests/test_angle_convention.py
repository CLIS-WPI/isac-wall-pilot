"""ANALYSIS_PLAN A12.2: the receiver angle map uses the link-angle sign (A12.1, DEVIATIONS P8-15)."""

from __future__ import annotations

import copy
import functools
import math
import sys
from pathlib import Path

import pytest
import torch

import isac_verify.phase7 as phase7
from isac_verify.beams import ArrayGeometry, array_response, dbm_to_watt, direct_beam
from isac_verify.config import PilotConfig
from isac_verify.phase8a import prepare_geometry
from isac_verify.receiver import (
    LinkPath,
    ProcessingChoices,
    angular_delay_map,
    delay_doppler_map,
    gate_energy,
    make_observation,
    noise_calibration,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from wpc_passive import passive_gate  # noqa: E402  (the A11/A12.4 reciprocal-path gate)


def _direct_weights(config: PilotConfig) -> torch.Tensor:
    return direct_beam(
        float(config.get("experiment_a", "reference_beam_phi_deg")),
        dbm_to_watt(config.get("waveform", "nominal_total_power_dbm")),
        ArrayGeometry.from_config(config),
    ).weights


def _ones(choices: ProcessingChoices) -> torch.Tensor:
    return torch.ones((choices.profile.num_symbols, choices.profile.fft_size), device="cuda", dtype=torch.complex64)


@pytest.mark.parametrize("phi_deg", [30.0, -20.0])
def test_single_path_peaks_at_its_signed_link_angle(config: PilotConfig, phi_deg: float) -> None:
    choices = ProcessingChoices.from_config(config)
    weights = _direct_weights(config)
    rx = array_response(phi_deg, choices.receiver_array, dtype=torch.complex64)  # RT arrival convention
    tx = weights.conj() / weights.abs().square().sum()
    path = LinkPath(label=f"synthetic:{phi_deg}", family="synthetic", matrix=rx[:, None] * tx[None, :],
                    delay_s=400e-9, aod_deg=0.0, aoa_deg=phi_deg, vertex_points_m=())
    observation = make_observation(world="target", beam_name="test", beam_weights=weights, symbols=_ones(choices),
                                   target_paths=(path,), profile=choices.profile, config=config, noise_seed=1,
                                   relay_noise_seed=2, noiseless=True)
    angular = angular_delay_map(delay_doppler_map(observation.h_eff, choices), choices, noise_per_cell_w=0.0)
    peak = int(torch.argmax(angular.power_w).item())
    peak_deg = float(angular.aoa_deg[peak % angular.aoa_deg.numel()].item())
    assert abs(peak_deg - phi_deg) <= choices.aoa_grid_step_deg, (phi_deg, peak_deg)


def _asymmetric_off_axis_config(config: PilotConfig) -> PilotConfig:
    data = copy.deepcopy(config.data)
    data["physical"]["walls"]["positive"]["offset_m"] = 12.0
    data["physical"]["walls"]["negative"]["offset_m"] = -18.0
    gnb = data["physical"]["gnb_position_m"]
    bearing = math.radians(10.0)
    data["physical"]["candidate_position_m"] = [gnb[0] + 50.0 * math.cos(bearing), gnb[1] + 50.0 * math.sin(bearing),
                                                gnb[2]]
    data["physical"]["relay_position_m"] = [gnb[0] + 20.0, gnb[1], gnb[2]]
    # Range 50 m keeps the direct return inside delay_search_ns. The direct action looks at the candidate's bearing (with the 0-deg beam, +10 deg sits in its first sidelobe).
    data["experiment_a"]["reference_beam_phi_deg"] = 10.0
    return PilotConfig(path=config.path, data=data)


def test_asymmetric_corridor_off_axis_gates_hold_the_planned_wall(config: PilotConfig, monkeypatch) -> None:
    config_g = _asymmetric_off_axis_config(config)
    choices = ProcessingChoices.from_config(config_g)
    prepared = prepare_geometry(config_g)
    links = prepared["links"]
    monkeypatch.setattr(phase7, "_observe", functools.partial(phase7._observe, noiseless=True))
    noise_full = noise_calibration(choices=choices, config=config_g, records=4, seed=8801)
    direct_weights = _direct_weights(config_g)
    seeds = {"streams": {"direct": {"probe_symbols": [11], "receiver_noise": [12], "relay_noise": [13]}}}
    base = phase7._acquire_record(config=config_g, choices=choices, world="target", record_index=0,
                                  direct_weights=direct_weights, links=links, calibration=prepared["calibration"],
                                  noise_full=noise_full, timing={}, seeds=seeds)
    # Sign sanity check only (pre-fix: about -10 deg); the positive-wall pair 10 ns behind the direct return biases
    # the estimate by about 2 deg. The A12.2(b) criteria are the gate fractions below.
    assert abs(base["candidate"].azimuth_deg - 10.0) <= 3.0, base["candidate"].azimuth_deg
    offsets = base["nominal_offsets"]
    plan, _, _ = phase7._plan_for(config=config_g, choices=choices, base=base, world="target", method="exact_oracle",
                                  sigma_m=0.0, twin_offsets=offsets, direct_weights=direct_weights, timing={})
    side = plan.wall_side
    other = {"positive": "negative", "negative": "positive"}[side]
    reciprocal = passive_gate(config_g, plan, base["candidate"], offsets, 0.0)
    by_family = {p.family: p for p in links["target"]}

    def fraction(family: str, weights: torch.Tensor, gate) -> float:
        observation = phase7._observe(config=config_g, world="target", beam_weights=weights,
                                      symbols=_ones(choices), profile=choices.profile,
                                      target_links=(by_family[family],), background_links=(), downlink_links=(),
                                      uplink_links=(), calibration=None, probe_seed=0, noise_seed=0,
                                      relay_noise_seed=0)
        angular = phase7._map(observation, choices, noise_full, {})
        inside = gate_energy(angular, delay_min_s=gate.delay_min_s, delay_max_s=gate.delay_max_s,
                             aoa_min_deg=gate.aoa_min_deg, aoa_max_deg=gate.aoa_max_deg).energy_w
        return inside / float(angular.power_w.sum().item())

    assert {f"outgoing_wall_{s}_wall" for s in (side, other)} <= set(by_family), sorted(by_family)
    probe = {s: fraction(f"outgoing_wall_{s}_wall", plan.beam.weights, plan.wall_gate) for s in (side, other)}
    passive = {s: fraction(f"reciprocal_wall_{s}_wall", direct_weights, reciprocal) for s in (side, other)}
    print(f"A12.2(b) planned side={side} wall-gate fractions={probe} reciprocal-gate fractions={passive}")
    assert probe[side] >= 0.5 and passive[side] >= 0.5, (side, probe, passive)
    assert probe[other] <= 0.05 and passive[other] <= 0.05, (side, probe, passive)
