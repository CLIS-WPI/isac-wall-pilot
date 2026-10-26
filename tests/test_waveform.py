from __future__ import annotations

import json

import pytest
import torch

from isac_verify.config import PilotConfig
from isac_verify.waveform import (
    dbm_to_watt,
    demodulate_ofdm,
    generate_ofdm_record,
    measure_transmit_power,
    profile_from_config,
    qpsk_symbols,
    waveform_reference,
)


def _uniform_weights(config: PilotConfig) -> torch.Tensor:
    power = dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))
    count = int(config.get("arrays", "gnb_tx", "num_elements"))
    return torch.full(
        (count,), (power / count) ** 0.5, dtype=torch.complex64, device="cuda"
    )


def test_ofdm_profiles_power_cp_and_roundtrip(config: PilotConfig, output_dir) -> None:
    tolerance = float(config.get("waveform", "power_relative_tolerance"))
    power = dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))
    weights = _uniform_weights(config)
    seed = int(config.get("seeds", "streams", "probe_symbols", "calibration")[0])
    entries = []
    for name in config.get("waveform", "bandwidth_comparison_profiles"):
        profile = profile_from_config(config, name)
        record = generate_ofdm_record(profile, seed)
        assert record.record.is_cuda
        measurement = measure_transmit_power(record, weights)
        assert measurement.total_mean_power_w == pytest.approx(power, rel=tolerance)
        assert measurement.beam_power_w == pytest.approx(power, rel=1e-6)
        assert measurement.record_duration_s == pytest.approx(1.2e-3, abs=1e-12)
        assert measurement.cp_fraction_measured > 0.0
        assert measurement.cp_energy_j > 0.0
        assert measurement.record_energy_j > measurement.cp_energy_j
        recovered = demodulate_ofdm(record.record, profile)
        error = float((recovered - record.symbols).abs().max().item())
        assert error < 1e-5
        entries.append(
            {
                "name": name,
                "measured_total_mean_power_w": measurement.total_mean_power_w,
                "declared_power_w": power,
                "relative_power_error": (measurement.total_mean_power_w - power) / power,
                "cp_energy_j": measurement.cp_energy_j,
                "record_energy_j": measurement.record_energy_j,
                "cp_fraction_measured": measurement.cp_fraction_measured,
                "cp_fraction_nominal": measurement.cp_fraction_nominal,
                "fft_roundtrip_max_abs_error": error,
                "delay_bin_spacing_ns": profile.delay_bin_spacing_s * 1e9,
                "record_duration_s": measurement.record_duration_s,
            }
        )
    (output_dir / "waveform_power.json").write_text(
        json.dumps({"seed": seed, "entries": entries}, indent=2) + "\n"
    )


def test_fresh_seeded_qpsk(config: PilotConfig) -> None:
    profile = profile_from_config(config, "baseline")
    seed = int(config.get("seeds", "streams", "probe_symbols", "calibration")[1])
    first = qpsk_symbols(profile, seed)
    second = qpsk_symbols(profile, seed)
    third = qpsk_symbols(profile, seed + 1)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    assert torch.allclose(first.abs(), torch.ones_like(first.real), atol=1e-6)


def test_waveform_reference_written(config: PilotConfig, output_dir) -> None:
    seed = int(config.get("seeds", "streams", "probe_symbols", "calibration")[2])
    reference = waveform_reference(config, seed)
    assert len(reference["entries"]) == 3
    (output_dir / "waveform_reference.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )
