from __future__ import annotations

from types import SimpleNamespace

import pytest

from isac_verify.config import PilotConfig
from isac_verify.phase5 import (
    DiagonalModelInvalid,
    PathTerm,
    check_diagonal_validity,
    generate_continuous_record,
    multipath_waveform,
    normalized_correlation_at_delay,
    phase5_diagnostics,
    pre_arrival_energy_fraction,
    relay_pair_terms,
)
from isac_verify.waveform import profile_from_config


def test_relay_cascade_delay_is_sum_of_all_three_legs() -> None:
    downlink = [
        PathTerm("d0", "los", 2.0 + 1.0j, 100e-9, 5.0, (), ()),
        PathTerm("d1", "wall_clutter_positive_wall", 0.5 - 0.25j, 140e-9, 0.3, (), ()),
    ]
    uplink = [
        PathTerm("u0", "los", 1.5 - 0.5j, 120e-9, 2.5, (), ()),
        PathTerm("u1", "wall_clutter_positive_wall", 0.25 + 0.125j, 160e-9, 0.1, (), ()),
    ]
    calibration = SimpleNamespace(
        feasible=True,
        gain=3.0 - 2.0j,
        measured_added_delay_s=200e-9,
    )
    pairs = relay_pair_terms(downlink, uplink, calibration)
    assert len(pairs) == 4
    for pair, down, up in zip(
        pairs,
        [downlink[0], downlink[0], downlink[1], downlink[1]],
        [uplink[0], uplink[1], uplink[0], uplink[1]],
        strict=True,
    ):
        assert pair.delay_s == pytest.approx(
            down.delay_s + 200e-9 + up.delay_s, abs=1e-18
        )
        assert pair.coefficient == pytest.approx(
            calibration.gain * down.coefficient * up.coefficient, abs=1e-15
        )


def test_fractional_delay_has_no_pre_arrival_energy(config: PilotConfig) -> None:
    profile = profile_from_config(config, "baseline")
    record = generate_continuous_record(profile, 3101, oversampling_factor=4)
    delay_s = 123.456e-9
    term = PathTerm("fractional", "los", 1.0 + 0.0j, delay_s, 1.0, (), ())
    waveform = multipath_waveform(
        record.record.reshape(-1),
        [term],
        sample_rate_hz=record.sample_rate_hz,
        guard_samples=record.block_samples,
    )
    fraction = pre_arrival_energy_fraction(
        waveform,
        guard_samples=record.block_samples,
        delay_s=delay_s,
        sample_rate_hz=record.sample_rate_hz,
    )
    floor = float(config.get("phase5", "fractional_delay_pre_arrival_floor"))
    assert fraction <= floor
    assert fraction < 1e-3


@pytest.fixture(scope="module")
def phase5_summary(config: PilotConfig, tmp_path_factory) -> dict:
    output = tmp_path_factory.mktemp("phase5-tests")
    return phase5_diagnostics(config, output)


def test_diagonal_model_agrees_with_continuous_reference(
    phase5_summary: dict,
) -> None:
    tolerance = float(
        phase5_summary["agreements"]["direct"]["target"]["tolerance"]
    )
    for beam in ("direct", "wall"):
        for link in ("target", "relay"):
            row = phase5_summary["agreements"][beam][link]
            assert row["max_relative_error"] <= tolerance
            assert row["edge_max_relative_error"] <= tolerance
            assert row["interior_max_relative_error"] <= tolerance
            assert row["median_relative_error"] <= row["max_relative_error"]


def test_invalid_cp_or_doppler_regime_is_refused(config: PilotConfig) -> None:
    profile = profile_from_config(config, "baseline")
    valid = [
        PathTerm("los", "los", 1.0 + 0.0j, 0.0, 1.0, (), ())
    ]
    checks = check_diagonal_validity(valid, profile, config)
    assert checks["cp_valid"] is True
    assert checks["doppler_valid"] is True

    cp_duration = profile.cp_length / profile.sample_rate_hz
    too_late = [
        PathTerm("late", "los", 1.0 + 0.0j, cp_duration * 1.01, 1.0, (), ())
    ]
    with pytest.raises(DiagonalModelInvalid, match="diagonal OFDM model invalid"):
        check_diagonal_validity(too_late, profile, config)

    limit = float(config.get("phase5", "doppler_symbol_cycles_max"))
    too_fast = limit / profile.symbol_duration_s * 1.01
    with pytest.raises(DiagonalModelInvalid, match="diagonal OFDM model invalid"):
        check_diagonal_validity(
            valid, profile, config, frequency_shift_hz=too_fast
        )


def test_boundary_crossing_probe_matches_current_not_stale(
    config: PilotConfig,
) -> None:
    profile = profile_from_config(config, "baseline")
    record = generate_continuous_record(profile, 3101, oversampling_factor=4)
    stale = generate_continuous_record(profile, 3102, oversampling_factor=4)
    block = record.block_samples
    guard = 3 * block
    delay_s = 1.25 * block / record.sample_rate_hz
    term = PathTerm("delay", "los", 1.0 + 0.0j, delay_s, 1.0, (), ())
    reference = record.record.reshape(-1)[:block]
    current = multipath_waveform(
        record.record.reshape(-1),
        [term],
        sample_rate_hz=record.sample_rate_hz,
        guard_samples=guard,
    )
    stale_waveform = multipath_waveform(
        stale.record.reshape(-1),
        [term],
        sample_rate_hz=record.sample_rate_hz,
        guard_samples=guard,
    )
    current_correlation = normalized_correlation_at_delay(
        reference,
        current,
        guard_samples=guard,
        delay_s=delay_s,
        sample_rate_hz=record.sample_rate_hz,
        search_samples=0,
    )
    stale_correlation = normalized_correlation_at_delay(
        reference,
        stale_waveform,
        guard_samples=guard,
        delay_s=delay_s,
        sample_rate_hz=record.sample_rate_hz,
        search_samples=0,
    )
    assert current_correlation > 0.99
    assert stale_correlation < 0.5
    assert current_correlation > stale_correlation
