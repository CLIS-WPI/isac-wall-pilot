"""Phase 8C stresses (ANALYSIS_PLAN A3.3 / A5): one factor at a time on the held-out grid.

A level is a config override written to ``<output>/stress_config.yaml`` and run
through the unchanged held-out harness (``phase8a.run_calibration``) with the
frozen thresholds. Nothing here runs unless the ``stress`` CLI is invoked; the
baseline config and code path are untouched.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .config import PilotConfig
from .phase8a import HELDOUT, Phase8Error, run_calibration


def _standoff_level(delta_r_m: float) -> str:
    return f"standoff_{delta_r_m:02.0f}m"


def stress_levels(config: PilotConfig) -> list[str]:
    # shortcut: blocks 1-2 (bandwidth, standoff) only; later A5.3 blocks add their levels here before they run.
    standoffs = config.get("phase8", "stress", "standoff", "delta_r_m")
    return [str(name) for name in config.get("phase8", "stress", "bandwidth_profiles")] + [
        _standoff_level(float(value)) for value in standoffs
    ]


def stress_config(config: PilotConfig, level: str) -> PilotConfig:
    """Baseline config with the A3.3 stress grid and one level applied."""
    if level not in stress_levels(config):
        raise Phase8Error(f"unknown stress level {level!r}; declared: {stress_levels(config)}")
    data = copy.deepcopy(config.data)
    grid = data["phase8"]["grid"]
    declared_sigmas = list(grid["sigma_map_m"])
    grid.update(copy.deepcopy(config.section("phase8", "stress", "grid")))
    data["phase8"]["stress_runtime"] = {"level": level, "declared_sigma_map_m": declared_sigmas}
    standoff = config.section("phase8", "stress", "standoff")
    for delta_r in standoff["delta_r_m"]:
        if level == _standoff_level(float(delta_r)):
            # A5.2: relay at R_T - Delta_R (phase8a._run_slots), baseline profile, A5.2 methods.
            grid["methods"] = list(standoff["methods"])
            data["phase8"]["stress_runtime"]["relay_standoff_m"] = float(delta_r)
            return PilotConfig(path=config.path, data=data)
    # Bandwidth level: waveform and receiver processing profile; the row label follows it.
    data["phase6"]["processing"]["profile"] = level
    data["rf"]["baseline_bandwidth_profile"] = level
    return PilotConfig(path=config.path, data=data)


def run_stress(
    config_path: str | Path,
    output_path: str | Path,
    level: str,
    *,
    require_h100: bool = True,
    workers: int = 1,
) -> int:
    """Run one stress level on the held-out geometries (caller must have approved held-out)."""
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "stress_config.yaml"
    path.write_text(
        yaml.safe_dump(stress_config(PilotConfig.load(config_path), level).data, sort_keys=False),
        encoding="utf-8",
    )
    return run_calibration(path, output, require_h100=require_h100, workers=workers, split=HELDOUT)
