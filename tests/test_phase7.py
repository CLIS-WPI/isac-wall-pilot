from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from isac_verify.config import PilotConfig
from isac_verify.metrics import integrity_checks, summarize_trials
from isac_verify.phase7 import METHODS, Phase7Error, _mapfree_look_plan, run_smoke
from isac_verify.report import build_report


def _reduced_config(config: PilotConfig, path: Path) -> Path:
    data = copy.deepcopy(config.data)
    data["statistics"]["smoke"]["sigma_map_m"] = [0.0]
    data["statistics"]["smoke"]["worlds"] = ["target", "replay"]
    data["statistics"]["smoke"]["noise_records_per_case"] = 1
    data["phase7"]["report"]["dpi"] = 72
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def smoke_run(config: PilotConfig, tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("phase7-smoke")
    config_path = _reduced_config(config, root / "reduced.yaml")
    output = root / "results"
    rc = run_smoke(
        config_path,
        output,
        require_h100=False,
    )
    assert rc == 0
    return {"config": config_path, "output": output}


def test_scientific_run_refuses_cpu_backend(config: PilotConfig, tmp_path: Path) -> None:
    data = copy.deepcopy(config.data)
    data["device"]["backend"] = "cpu"
    path = tmp_path / "cpu.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(Phase7Error, match="refuses non-CUDA"):
        run_smoke(path, tmp_path / "out", require_h100=False)


def test_report_rebuilds_from_saved_results(smoke_run: dict[str, Path]) -> None:
    output = smoke_run["output"]
    report_dir = output / "report"
    shutil.rmtree(report_dir)
    assert build_report(output, report_dir) == 0
    expected = [
        "report.md",
        "aggregate_metrics.csv",
        "01_geometry.png",
        "02_experiment_a.png",
        "03_beampatterns.png",
        "04_predicted_delays.png",
        "05_spectra.png",
        "06_score_distributions.png",
        "07_status_rates.png",
        "09_runtime_memory.png",
    ]
    for name in expected:
        assert (report_dir / name).exists(), name
    text = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "Smoke stage, no statistical conclusion" in text
    assert "no statistical conclusion" in text.lower()


def test_repeated_seeded_run_agrees(smoke_run: dict[str, Path], tmp_path: Path) -> None:
    repeat = tmp_path / "repeat"
    rc = run_smoke(
        smoke_run["config"],
        repeat,
        require_h100=False,
    )
    assert rc == 0
    first = pd.read_parquet(smoke_run["output"] / "trials.parquet")
    second = pd.read_parquet(repeat / "trials.parquet")
    keys = ["method", "sigma_map_m", "world", "record_index"]
    first = first.sort_values(keys).reset_index(drop=True)
    second = second.sort_values(keys).reset_index(drop=True)
    assert first[keys].equals(second[keys])
    assert first["S_available"].tolist() == second["S_available"].tolist()
    assert (
        first["wall_excess_snr_available"].tolist()
        == second["wall_excess_snr_available"].tolist()
    )
    numeric = [
        "E_D",
        "N_D",
        "E_W",
        "N_W",
        "S",
        "wall_excess_snr",
        "null_depth_db",
        "candidate_delay_ns",
        "energy_symbols",
        "transmit_energy_j",
    ]
    for column in numeric:
        left = first[column].to_numpy(dtype=float)
        right = second[column].to_numpy(dtype=float)
        assert np.allclose(left, right, rtol=1e-5, atol=1e-12, equal_nan=True), column


def test_score_columns_are_availability_masked(smoke_run: dict[str, Path]) -> None:
    trials = pd.read_parquet(smoke_run["output"] / "trials.parquet")
    assert set(METHODS) == set(trials["method"])
    direct = trials[trials["method"] == "direct_only"]
    assert direct["S_available"].eq(False).all()
    assert direct["S"].isna().all()
    assert direct["wall_excess_snr_available"].eq(False).all()
    assert direct["wall_excess_snr"].isna().all()
    mapped = trials[
        trials["method"].isin(["exact_oracle", "imperfect_narrow", "imperfect_robust"])
    ]
    assert mapped["S_available"].eq(True).all()
    assert np.isfinite(mapped["S"].to_numpy(dtype=float)).all()
    mapfree = trials[trials["method"].str.startswith("mapfree")]
    assert mapfree["S_available"].eq(False).all()
    assert mapfree["S"].isna().all()
    assert mapfree["wall_excess_snr_available"].eq(True).all()
    assert np.isfinite(mapfree["wall_excess_snr"].to_numpy(dtype=float)).all()
    checks = integrity_checks(
        trials,
        expected_rows=len(trials),
        methods=METHODS,
        sigmas=[0.0],
        worlds=["target", "replay"],
        records=1,
    )
    assert checks["S_availability_matches_finiteness"] is True
    assert checks["wall_excess_snr_availability_matches_finiteness"] is True
    summary = summarize_trials(trials)
    assert summary["S_available_count"].max() == 1
    assert summary["wall_excess_snr_available_count"].max() == 1


def test_mapfree_two_beam_spends_one_extra_action(config: PilotConfig) -> None:
    plan = _mapfree_look_plan(
        "mapfree_two_beam",
        config=config,
        beam_seed=int(config.get("seeds", "master")) + 7601,
    )
    assert len(plan) == 1
    assert plan[0][1] == int(config.get("beam", "map_free", "total_extra_symbols"))


def test_noise_normalization_is_unity_within_interval(
    smoke_run: dict[str, Path],
) -> None:
    noise = pd.read_csv(smoke_run["output"] / "noise_normalization.csv")
    assert len(noise) >= 10
    assert noise["records"].min() >= 128
    assert noise["ci95_low"].le(1.0).all()
    assert noise["ci95_high"].ge(1.0).all()
    assert noise["mean_e_over_n"].between(0.5, 2.0).all()
