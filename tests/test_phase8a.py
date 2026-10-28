from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isac_verify.config import PilotConfig
from isac_verify.phase8a import (
    Phase8Error,
    draw_geometry,
    frozen_decision,
    geometry_config,
    geometry_seed_streams,
    noise_only_seed_fn,
    run_calibration,
    twin_draws,
)


def _flatten(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _flatten(item)
    elif isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield int(value)


def test_calibration_draws_follow_the_declared_ranges(config: PilotConfig) -> None:
    geometry = config.section("phase8", "geometry")
    for slot in range(12):
        for attempt in range(5):
            draw = draw_geometry(config, "calibration", slot, attempt)
            assert 50.0 <= draw["candidate_distance_m"] <= 70.0
            assert 20.0 <= draw["relay_distance_m"] <= 40.0
            assert 12.0 <= draw["half_width_m"] <= 18.0
            assert draw["wall_thickness_m"] == geometry["wall_thickness_m"] == 0.10
            assert draw == draw_geometry(config, "calibration", slot, attempt)
    first = draw_geometry(config, "calibration", 0, 0)
    second = draw_geometry(config, "calibration", 0, 1)
    assert first["candidate_distance_m"] != second["candidate_distance_m"]


def test_heldout_stream_is_refused(config: PilotConfig) -> None:
    with pytest.raises(Phase8Error, match="explicit human approval"):
        draw_geometry(config, "heldout", 0, 0)
    with pytest.raises(Phase8Error, match="explicit human approval"):
        geometry_seed_streams(config, "heldout", 0, 32)
    with pytest.raises(Phase8Error, match="explicit human approval"):
        twin_draws(config, "heldout", 0, 4)


def test_heldout_cli_refuses_without_approval() -> None:
    from isac_verify.__main__ import main

    with pytest.raises(SystemExit, match="--approved"):
        main(["heldout", "--output", "/tmp/never_written"])
    assert not Path("/tmp/never_written").exists()


def test_frozen_decision_reads_only_phase8_frozen(config: PilotConfig) -> None:
    frozen = config.section("phase8", "frozen")
    factor, fitted = frozen_decision(config)
    assert factor == frozen["gate_inflation_c"]
    for method, values in frozen["thresholds"].items():
        assert fitted["mapped"][method]["floor_k"]["value"] == values["noise_sigmas"]
        assert fitted["mapped"][method]["s_lo"]["value"] == values["supported_min"]
        assert fitted["mapped"][method]["s_hi"]["value"] == values["supported_max"]
    assert {m: v["value"] for m, v in fitted["common"].items()} == frozen["t_common"]
    data = json.loads(json.dumps(config.data))
    data["phase8"]["frozen"]["thresholds"] = None
    with pytest.raises(Phase8Error, match="incomplete"):
        frozen_decision(PilotConfig(path=config.path, data=data))


def test_geometry_config_is_collinear_and_symmetric(config: PilotConfig) -> None:
    draw = draw_geometry(config, "calibration", 3, 0)
    local = geometry_config(config, draw)
    candidate = local.get("physical", "candidate_position_m")
    relay = local.get("physical", "relay_position_m")
    assert candidate[1] == relay[1] == 0.0
    assert candidate[0] == draw["candidate_distance_m"]
    assert relay[0] == draw["relay_distance_m"]
    assert local.get("physical", "walls", "positive", "offset_m") == draw["half_width_m"]
    assert local.get("physical", "walls", "negative", "offset_m") == -draw["half_width_m"]
    assert config.get("physical", "candidate_position_m") == [60.0, 0.0, 2.0]


def test_declared_seeds_are_unique_across_geometries_and_actions(
    config: PilotConfig,
) -> None:
    seen: list[int] = []
    for slot in range(12):
        streams = geometry_seed_streams(config, "calibration", slot, 32)
        seen.extend(_flatten(streams))
        noise = noise_only_seed_fn(config, "calibration", slot, 5)
        for world in ("target", "replay"):
            for sigma_index in range(5):
                for record in range(32):
                    seen.append(noise(world, sigma_index, record))
    assert len(seen) == len(set(seen))
    assert max(seen) < 2**31


def test_direct_seeds_are_common_and_extra_actions_independent(
    config: PilotConfig,
) -> None:
    streams = geometry_seed_streams(config, "calibration", 0, 32)
    direct = {
        kind: streams["direct"][kind][5]
        for kind in ("probe_symbols", "receiver_noise", "relay_noise")
    }
    extras = set()
    for method, values in streams["wall_action"].items():
        extras.update(values[kind][5] for kind in direct)
    mapfree = streams["mapfree"]
    extras.update(mapfree["mapfree_two_beam"][kind][5] for kind in direct)
    for group in ("coarse", "refine"):
        for kind in direct:
            extras.update(mapfree["mapfree_coarse_to_fine"][group][kind][5])
    assert set(direct.values()).isdisjoint(extras)
    assert len(extras) == 9 + 3 + 12 + 6


def test_twin_draws_are_independent_per_wall_and_reproducible(
    config: PilotConfig,
) -> None:
    first = twin_draws(config, "calibration", 2, 4)
    again = twin_draws(config, "calibration", 2, 4)
    assert first == again
    positives = [draw["positive"] for draw in first]
    negatives = [draw["negative"] for draw in first]
    assert positives != negatives
    assert len(set(positives)) == 4


def test_one_calibration_geometry_end_to_end(
    config: PilotConfig, tmp_path: Path
) -> None:
    output = tmp_path / "phase8a"
    rc = run_calibration(
        "configs/pilot.yaml",
        output,
        require_h100=False,
        geometry_limit=1,
        record_limit=4,
    )
    assert rc == 0
    for name in (
        "calibration_trials.parquet",
        "thresholds.json",
        "coverage.csv",
        "geometries_calibration.csv",
        "rejected_draws.csv",
        "calibration_summary.json",
        "timing.json",
    ):
        assert (output / name).exists(), name
    trials = pd.read_parquet(output / "calibration_trials.parquet")
    assert len(trials) == 1 * 5 * 2 * 4 * 6
    assert set(trials["split"]) == {"calibration"}
    assert set(trials["geometry_id"]) == {"cal_00"}
    mapped = trials[trials["method"] == "imperfect_robust"]
    assert np.isfinite(mapped["floor_excess_c1"]).all()
    assert np.isfinite(mapped["floor_excess_c2"]).all()
    assert mapped["cover_delay_c1"].isin([0.0, 1.0]).all()
    assert (mapped["E_W_c1"] == mapped["E_W"]).all()
    assert (mapped["S_c1"] == mapped["S"]).all()
    assert (mapped["E_W_c2"] != mapped["E_W_c1"]).any()
    direct = trials[trials["method"] == "direct_only"]
    assert direct["floor_excess_c1"].isna().all()
    summary = json.loads((output / "calibration_summary.json").read_text())
    assert summary["held_out_generated"] is False
    assert summary["noise_only_requirement_met"] is False
    thresholds = json.loads((output / "thresholds.json").read_text())
    assert set(thresholds["floor_k"]) == {
        "exact_oracle",
        "imperfect_narrow",
        "imperfect_robust",
    }
    assert not (output / "heldout").exists()
