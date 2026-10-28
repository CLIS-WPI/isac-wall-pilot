"""Phase 8C stresses (ANALYSIS_PLAN A3.3 / A5): one factor at a time on the held-out grid.

A level is a config override written to ``<output>/stress_config.yaml`` and run
through the unchanged held-out harness (``phase8a.run_calibration``) with the
frozen thresholds. Nothing here runs unless the ``stress`` CLI is invoked; the
baseline config and code path are untouched.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .config import PilotConfig
from .phase8a import HELDOUT, Phase8Error, run_calibration
from .receiver import LinkPath
from .wall_paths import path_records

ARRAY_ERROR_LEVEL = "array_error_d11"


def _standoff_level(delta_r_m: float) -> str:
    return f"standoff_{delta_r_m:02.0f}m"


def _power_level(dbm: float) -> str:
    return f"probe_power_{'m' if dbm < 0 else 'p' if dbm > 0 else ''}{abs(dbm):.0f}dbm"


def _thickness_level(thickness_m: float) -> str:
    return f"wall_thickness_{thickness_m * 1000:.0f}mm"


def _rcs_level(db: float) -> str:
    return f"rcs_{'p' if db > 0 else 'm'}{abs(db):.0f}db"


def _level_values(config: PilotConfig) -> dict[str, tuple[str, float]]:
    stress = config.section("phase8", "stress")
    out = {_standoff_level(float(v)): ("standoff", float(v)) for v in stress["standoff"]["delta_r_m"]}
    out.update({_power_level(float(v)): ("probe_power", float(v)) for v in stress["probe_power_dbm"]})
    out.update({_thickness_level(float(v)): ("thickness", float(v)) for v in stress["wall_thickness_m"]})
    out.update({_rcs_level(float(v)): ("rcs", float(v)) for v in stress["rcs_reflected_illumination_db"]})
    return out


def stress_levels(config: PilotConfig) -> list[str]:
    # A3.3 levels marked NOT RUN by Amendment 8 (relay gain, material, blocker) are deliberately absent.
    return (
        [str(name) for name in config.get("phase8", "stress", "bandwidth_profiles")]
        + [ARRAY_ERROR_LEVEL]
        + list(_level_values(config))
    )


def stress_config(config: PilotConfig, level: str) -> PilotConfig:
    """Baseline config with the A3.3 stress grid and one level applied."""
    if level not in stress_levels(config):
        raise Phase8Error(f"unknown stress level {level!r}; declared: {stress_levels(config)}")
    data = copy.deepcopy(config.data)
    grid = data["phase8"]["grid"]
    declared_sigmas = list(grid["sigma_map_m"])
    grid.update(copy.deepcopy(config.section("phase8", "stress", "grid")))
    data["phase8"]["stress_runtime"] = {"level": level, "declared_sigma_map_m": declared_sigmas}
    if level == ARRAY_ERROR_LEVEL:
        # A7.1: per-element gNB TX/RX error on the true links only (phase8a.run_geometry), A3.3 grid.
        data["phase8"]["stress_runtime"]["array_error"] = {
            **copy.deepcopy(config.section("phase8", "stress", "array_calibration")),
            "seed": int(config.get("phase8", "stress", "array_calibration_seed")),
        }
        return PilotConfig(path=config.path, data=data)
    kind, value = _level_values(config).get(level, (None, None))
    if kind == "standoff":
        # A5.2: relay at R_T - Delta_R (phase8a._run_slots), baseline profile, A5.2 methods.
        grid["methods"] = list(config.get("phase8", "stress", "standoff", "methods"))
        data["phase8"]["stress_runtime"]["relay_standoff_m"] = value
        return PilotConfig(path=config.path, data=data)
    if kind == "probe_power":
        # A8.1: probe actions only (phase8a._run_slots); acceptance and relay calibration stay nominal.
        data["phase8"]["stress_runtime"]["probe_power_dbm"] = value
        return PilotConfig(path=config.path, data=data)
    if kind == "thickness":
        # A8.2 / D8: truth and twin; draw_geometry -> geometry_config carries it into wall.thickness_m.
        data["phase8"]["geometry"]["wall_thickness_m"] = value
        data["wall"]["thickness_m"] = value
        return PilotConfig(path=config.path, data=data)
    if kind == "rcs":
        # A8.2: reflected-illumination true target paths scaled (phase8a._run_slots); twin unaware.
        data["phase8"]["stress_runtime"]["rcs_reflected_illumination_db"] = value
        return PilotConfig(path=config.path, data=data)
    # Bandwidth level: waveform and receiver processing profile; the row label follows it.
    data["phase6"]["processing"]["profile"] = level
    data["rf"]["baseline_bandwidth_profile"] = level
    return PilotConfig(path=config.path, data=data)


def array_error_factors(spec: dict, slot: int, realization: int, elements: int) -> tuple[np.ndarray, np.ndarray]:
    """A7.1 notes: complex per-element factors (TX, RX) for one realization of one geometry slot."""
    rng = np.random.default_rng([int(spec["seed"]), int(slot), int(realization)])
    amp_db, phase_deg = float(spec["amplitude_error_db_rms"]), float(spec["phase_error_deg_rms"])
    tx_a, tx_p = rng.normal(0.0, amp_db, elements), rng.normal(0.0, phase_deg, elements)
    rx_a, rx_p = rng.normal(0.0, amp_db, elements), rng.normal(0.0, phase_deg, elements)
    factor = lambda a, p: 10.0 ** (a / 20.0) * np.exp(1j * np.deg2rad(p))  # noqa: E731
    return factor(tx_a, tx_p), factor(rx_a, rx_p)


def perturb_links(
    links: dict[str, tuple[LinkPath, ...]], e_tx: np.ndarray, e_rx: np.ndarray
) -> dict[str, tuple[LinkPath, ...]]:
    """Apply gNB element factors to the true link matrices [rx, tx]; the single-element relay side is untouched."""
    def scaled(path: LinkPath, rx: bool, tx: bool) -> LinkPath:
        m = path.matrix
        if rx:
            m = torch.as_tensor(e_rx, dtype=m.dtype, device=m.device)[:, None] * m
        if tx:
            m = m * torch.as_tensor(e_tx, dtype=m.dtype, device=m.device)[None, :]
        return replace(path, matrix=m)

    sides = {"target": (True, True), "background": (True, True), "downlink": (False, True), "uplink": (True, False)}
    if set(links) != set(sides):
        raise Phase8Error(f"unexpected link groups {sorted(links)}")
    return {name: tuple(scaled(p, *sides[name]) for p in paths) for name, paths in links.items()}


def with_probe_power(config_g: PilotConfig, dbm: float) -> PilotConfig:
    """A8.1: the geometry config with every probe action at ``dbm``."""
    data = copy.deepcopy(config_g.data)
    data["waveform"]["nominal_total_power_dbm"] = float(dbm)
    return PilotConfig(path=config_g.path, data=data)


def reflected_illumination_indices(path_set: Any) -> set[int]:
    """Target path indices with a specular reflection before the sensing interaction (A8 notes)."""
    out = set()
    for record in path_records(path_set, scene_label="true_target"):
        names = [str(name) for name in record["interaction_names"]]
        if "SENSING" not in names:
            raise Phase8Error(f"target path {record['path_index']} has no sensing interaction: {names}")
        if "SPECULAR" in names[: names.index("SENSING")]:
            out.add(int(record["path_index"]))
    return out


def scale_reflected_illumination(prepared: dict[str, Any], db: float) -> dict[str, Any]:
    """A8.2: amplitude-scale reflected-illumination true target links by 10^(db/20); other links untouched."""
    indices = reflected_illumination_indices(prepared["paths"]["target"])
    factor = 10.0 ** (float(db) / 20.0)
    target = tuple(
        replace(path, matrix=path.matrix * factor) if int(path.label.rsplit(":", 1)[1]) in indices else path
        for path in prepared["links"]["target"]
    )
    return {**prepared, "links": {**prepared["links"], "target": target}}


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
