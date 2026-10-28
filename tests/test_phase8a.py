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


def test_stress_cli_refuses_without_approval() -> None:
    from isac_verify.__main__ import main

    with pytest.raises(SystemExit, match="--approved"):
        main(["stress", "--level", "bw246", "--output", "/tmp/never_written_stress"])
    assert not Path("/tmp/never_written_stress").exists()


def test_stress_config_changes_only_grid_and_level(config: PilotConfig) -> None:
    from isac_verify.stress import stress_config

    with pytest.raises(Phase8Error, match="unknown stress level"):
        stress_config(config, "baseline")
    stressed = stress_config(config, "bw246").data
    assert stressed["phase8"]["grid"]["sigma_map_m"] == [0.5]
    assert stressed["phase8"]["grid"]["twin_realizations_per_sigma"] == 2
    assert stressed["phase8"]["stress_runtime"]["declared_sigma_map_m"] == config.get("phase8", "grid", "sigma_map_m")
    assert stressed["phase6"]["processing"]["profile"] == "bw246"
    assert stressed["phase8"]["frozen"] == config.get("phase8", "frozen")
    for key in set(config.data) - {"phase6", "phase8", "rf"}:
        assert stressed[key] == config.data[key], key
    assert config.get("phase6", "processing", "profile") == "baseline"
    assert "stress_runtime" not in config.section("phase8")
    standoff = stress_config(config, "standoff_20m").data
    assert standoff["phase8"]["stress_runtime"]["relay_standoff_m"] == 20.0
    assert standoff["phase8"]["grid"]["methods"] == ["exact_oracle", "imperfect_robust", "mapfree_coarse_to_fine"]
    assert standoff["phase6"] == config.data["phase6"] and standoff["rf"] == config.data["rf"]
    assert stress_config(config, "standoff_03m").data["phase8"]["stress_runtime"]["relay_standoff_m"] == 3.0


def test_array_error_level_and_link_perturbation(config: PilotConfig) -> None:
    """A7.1 notes: D11 touches only gNB element sides of the true links; ones leave them unchanged."""
    import torch

    from isac_verify.receiver import LinkPath
    from isac_verify.stress import ARRAY_ERROR_LEVEL, array_error_factors, perturb_links, stress_config

    stressed = stress_config(config, ARRAY_ERROR_LEVEL).data
    spec = stressed["phase8"]["stress_runtime"]["array_error"]
    assert spec["amplitude_error_db_rms"] == 0.5 and spec["phase_error_deg_rms"] == 5.0
    assert spec["realizations"] == 4 and spec["seed"] == 20261028
    assert stressed["phase8"]["grid"]["methods"] == config.get("phase8", "stress", "grid", "methods")
    assert stressed["phase6"] == config.data["phase6"] and stressed["rf"] == config.data["rf"]
    e_tx, e_rx = array_error_factors(spec, 3, 1, 16)
    assert np.array_equal(e_tx, array_error_factors(spec, 3, 1, 16)[0])
    assert not np.allclose(e_tx, array_error_factors(spec, 3, 2, 16)[0]) and not np.allclose(e_tx, e_rx)
    big_tx, _ = array_error_factors(spec, 0, 0, 200_000)
    assert abs(np.std(20 * np.log10(np.abs(big_tx))) - 0.5) < 0.01
    assert abs(np.std(np.rad2deg(np.angle(big_tx))) - 5.0) < 0.05

    def link(rows: int, cols: int) -> LinkPath:
        m = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols).to(torch.complex64) + 1
        return LinkPath("x", "f", m, 1e-7, 0.0, 0.0, ())

    links = {"target": (link(16, 16),), "background": (link(16, 16),),
             "downlink": (link(1, 16),), "uplink": (link(16, 1),)}
    same = perturb_links(links, np.ones(16), np.ones(16))
    assert all(torch.equal(same[k][0].matrix, links[k][0].matrix) for k in links)
    out = perturb_links(links, e_tx, e_rx)
    tx, rx = torch.as_tensor(e_tx, dtype=torch.complex64), torch.as_tensor(e_rx, dtype=torch.complex64)
    assert torch.allclose(out["target"][0].matrix, rx[:, None] * links["target"][0].matrix * tx[None, :])
    assert torch.allclose(out["background"][0].matrix, rx[:, None] * links["background"][0].matrix * tx[None, :])
    assert torch.allclose(out["downlink"][0].matrix, links["downlink"][0].matrix * tx[None, :])
    assert torch.allclose(out["uplink"][0].matrix, rx[:, None] * links["uplink"][0].matrix)


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


TRUTH_AND_EVALUATOR_COLUMNS = (
    "world", "sigma_map_m", "geometry_id", "twin_index", "true_wall_delay_ns", "true_wall_aoa_deg",
    "true_positive_wall_m", "true_negative_wall_m", "true_candidate_azimuth_deg", "true_relay_azimuth_deg",
    "candidate_distance_m", "relay_distance_m", "half_width_m", "candidate_aoa_error_deg",
    "twin_map_error_positive_m", "twin_map_error_negative_m", "cover_delay_c1", "cover_aoa_c1",
    "receiver_noise_seed", "relay_noise_seed", "probe_seed",
)


def test_frozen_decision_ignores_truth_and_labels(config: PilotConfig) -> None:
    """A4.3: decisions depend only on measured columns and phase8.frozen, never on truth or labels."""
    from isac_verify.calibration import classify_trials

    rng = np.random.default_rng(0)
    n = 600
    methods = ["direct_only", "exact_oracle", "imperfect_narrow", "imperfect_robust",
               "mapfree_two_beam", "mapfree_coarse_to_fine"]
    frame = pd.DataFrame({
        "method": rng.choice(methods, n),
        "status": rng.choice(["UNRESOLVED", "ACQUISITION_FAILED"], n, p=[0.95, 0.05]),
        "direct_reliable": rng.choice([0.0, 1.0], n, p=[0.1, 0.9]),
        "ref_det_c1": rng.choice([0.0, 1.0], n, p=[0.1, 0.9]),
        "S_c1": rng.uniform(-0.05, 1.5, n),
        "N_W_c1": rng.uniform(0.5, 1.5, n),
        "wall_excess_snr": rng.uniform(0.0, 10.0, n),
        "wall_excess_snr_c1": rng.uniform(0.0, 40.0, n),
    })
    frame["E_W_c1"] = frame["N_W_c1"] * (1.0 + frame["wall_excess_snr_c1"] * rng.uniform(0.0, 0.2, n))
    for column in TRUTH_AND_EVALUATOR_COLUMNS:
        frame[column] = rng.normal(size=n)
    frame["world"] = rng.choice(["target", "replay"], n)
    factor, fitted = frozen_decision(config)
    decided = ["decision", "decision_reason", "supported", "supported_common"]
    reference = classify_trials(frame, fitted, factor)[decided]
    assert reference["supported"].any() and reference["supported_common"].any()
    perturbed = frame.copy()
    for column in TRUTH_AND_EVALUATOR_COLUMNS:
        perturbed[column] = rng.permutation(perturbed[column].to_numpy()) * 3.0 if column != "world" else \
            perturbed[column].map({"target": "replay", "replay": "target"})
    pd.testing.assert_frame_equal(classify_trials(perturbed, fitted, factor)[decided], reference)


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


def test_block45_levels_power_thickness_rcs(config: PilotConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """A8 notes: probe power is a runtime override, thickness reaches truth and twin, RCS scales only
    target paths with a reflection before the sensing interaction."""
    import torch

    from isac_verify import stress
    from isac_verify.receiver import LinkPath

    levels = stress.stress_levels(config)
    for name in ("probe_power_m20dbm", "probe_power_m10dbm", "probe_power_0dbm", "probe_power_p20dbm",
                 "wall_thickness_50mm", "wall_thickness_200mm", "rcs_p6db", "rcs_m6db"):
        assert name in levels, name
    power = stress.stress_config(config, "probe_power_m20dbm").data
    assert power["phase8"]["stress_runtime"]["probe_power_dbm"] == -20.0
    assert power["waveform"] == config.data["waveform"]  # acceptance and relay calibration stay nominal
    assert stress.with_probe_power(config, -20.0).get("waveform", "nominal_total_power_dbm") == -20.0
    assert config.get("waveform", "nominal_total_power_dbm") == 10.0
    thick = stress.stress_config(config, "wall_thickness_50mm").data
    assert thick["phase8"]["geometry"]["wall_thickness_m"] == 0.05 and thick["wall"]["thickness_m"] == 0.05
    assert "probe_power_dbm" not in thick["phase8"]["stress_runtime"]
    rcs = stress.stress_config(config, "rcs_m6db").data
    assert rcs["phase8"]["stress_runtime"]["rcs_reflected_illumination_db"] == -6.0
    assert rcs["wall"] == config.data["wall"] and rcs["phase6"] == config.data["phase6"]

    names = {0: ["SENSING"], 1: ["SPECULAR", "SENSING"], 2: ["SENSING", "SPECULAR"],
             3: ["SPECULAR", "SENSING", "SPECULAR"], 4: ["SENSING", "SPECULAR", "SPECULAR"]}
    monkeypatch.setattr(stress, "path_records", lambda path_set, scene_label: [
        {"path_index": i, "interaction_names": n} for i, n in names.items()])
    assert stress.reflected_illumination_indices(None) == {1, 3}
    ones = torch.ones(2, 2, dtype=torch.complex64)
    links = {"target": tuple(LinkPath(f"true_target:f:{i}", "f", ones, 1e-7, 0.0, 0.0, ()) for i in names),
             "downlink": (LinkPath("true_downlink:los:0", "los", ones, 1e-7, 0.0, 0.0, ()),)}
    out = stress.scale_reflected_illumination({"paths": {"target": None}, "links": links}, 6.0)
    gain = 10 ** (6 / 20)
    for i, path in enumerate(out["links"]["target"]):
        assert torch.allclose(path.matrix, ones * (gain if i in (1, 3) else 1.0)), i
    assert out["links"]["downlink"] is links["downlink"]
