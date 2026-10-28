"""Phase 8 Part A: calibration geometries, trial grid, and frozen thresholds.

This module implements ``docs/ANALYSIS_PLAN.md``. It generates and runs only
the declared calibration geometries. The held-out stream is declared in the
configuration but this module refuses to draw from it: the held-out run needs
explicit human approval.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from . import calibration as cal
from .beams import ArrayGeometry, dbm_to_watt, direct_beam
from .config import PilotConfig
from .environment import _nvml_process_snapshot
from .geometry import GeometryError, LatencyFloorError, horizontal_range_and_azimuth
from .phase7 import (
    METHODS,
    _acquire_record,
    _calibrate,
    _jsonable,
    _now,
    _nvml_snapshot,
    _require_gpu,
    _true_links,
    _true_paths,
    evaluate_record,
)
from .receiver import ProcessingChoices, noise_calibration
from .wall_paths import find_outgoing_wall_path, path_records

SPLIT = "calibration"
HELDOUT = "heldout"
_heldout_approved = False
RECORD_BLOCKS = {
    "direct": 0,
    "exact_oracle": 1,
    "imperfect_narrow": 2,
    "imperfect_robust": 3,
    "mapfree_two_beam": 4,
    "mapfree_coarse": 5,
    "mapfree_refine": 6,
}


class Phase8Error(RuntimeError):
    """Raised when a Phase 8A invariant is violated."""


def approve_heldout() -> None:
    """Unlock the held-out stream for this process (CLI ``heldout --approved``)."""
    global _heldout_approved
    _heldout_approved = True


def _require_calibration(split: str) -> None:
    if split == HELDOUT and _heldout_approved:
        return
    if split != SPLIT:
        raise Phase8Error(
            f"split {split!r} is not available in Phase 8A: the held-out "
            "geometries need explicit human approval"
        )


def _stream(config: PilotConfig, name: str, split: str) -> list[int]:
    _require_calibration(split)
    return [int(value) for value in config.get("seeds", "streams", name, split)]


def draw_geometry(
    config: PilotConfig, split: str, slot: int, attempt: int
) -> dict[str, float]:
    """Draw one geometry attempt for a slot of the declared stream (rule 1)."""
    _require_calibration(split)
    seeds = _stream(config, "physical_geometry", split)
    geometry = config.section("phase8", "geometry")
    rng = np.random.default_rng([int(seeds[slot]), int(attempt)])
    candidate_low, candidate_high = geometry["candidate_distance_m_range"]
    relay_low, relay_high = geometry["relay_distance_m_range"]
    half_low, half_high = geometry["true_corridor_half_width_m_range"]
    return {
        "slot": int(slot),
        "attempt": int(attempt),
        "stream_seed": int(seeds[slot]),
        "candidate_distance_m": float(rng.uniform(candidate_low, candidate_high)),
        "relay_distance_m": float(rng.uniform(relay_low, relay_high)),
        "half_width_m": float(rng.uniform(half_low, half_high)),
        "wall_thickness_m": float(geometry["wall_thickness_m"]),
    }


def geometry_config(
    config: PilotConfig, geometry: dict[str, float]
) -> PilotConfig:
    data = copy.deepcopy(config.data)
    gnb = data["physical"]["gnb_position_m"]
    height = float(gnb[2])
    data["physical"]["candidate_position_m"] = [
        float(geometry["candidate_distance_m"]),
        0.0,
        height,
    ]
    data["physical"]["relay_position_m"] = [
        float(geometry["relay_distance_m"]),
        0.0,
        height,
    ]
    data["physical"]["walls"]["positive"]["offset_m"] = float(
        geometry["half_width_m"]
    )
    data["physical"]["walls"]["negative"]["offset_m"] = -float(
        geometry["half_width_m"]
    )
    data["wall"]["thickness_m"] = float(geometry["wall_thickness_m"])
    return PilotConfig(path=config.path, data=data)


def _record_seed(base: int, block: int, record: int, look: int = 0) -> int:
    return int(base) * 100000 + int(block) * 1000 + int(record) * 10 + int(look)


def geometry_seed_streams(
    config: PilotConfig, split: str, slot: int, records: int
) -> dict[str, Any]:
    """Explicit per-record seeds for one geometry slot (P7-4 layout)."""
    _require_calibration(split)
    if records > 100:
        raise Phase8Error("seed layout supports at most 100 records")
    kinds = {
        "probe_symbols": _stream(config, "probe_symbols", split)[slot],
        "receiver_noise": _stream(config, "receiver_noise", split)[slot],
        "relay_noise": _stream(config, "relay_noise", split)[slot],
    }

    def series(block: int) -> dict[str, list[int]]:
        return {
            kind: [_record_seed(base, block, record) for record in range(records)]
            for kind, base in kinds.items()
        }

    def matrix(block: int, looks: int) -> dict[str, list[list[int]]]:
        return {
            kind: [
                [_record_seed(base, block, record, look) for look in range(looks)]
                for record in range(records)
            ]
            for kind, base in kinds.items()
        }

    beam = _stream(config, "diagnostic_beam_selection", split)[slot]
    return {
        "direct": series(RECORD_BLOCKS["direct"]),
        "beam_selection": [
            int(beam) * 100000 + record for record in range(records)
        ],
        "wall_action": {
            method: series(RECORD_BLOCKS[method])
            for method in ("exact_oracle", "imperfect_narrow", "imperfect_robust")
        },
        "mapfree": {
            "mapfree_two_beam": series(RECORD_BLOCKS["mapfree_two_beam"]),
            "mapfree_coarse_to_fine": {
                "coarse": matrix(RECORD_BLOCKS["mapfree_coarse"], 4),
                "refine": matrix(RECORD_BLOCKS["mapfree_refine"], 2),
            },
        },
    }


def twin_draws(config: PilotConfig, split: str, slot: int, count: int) -> list[dict[str, float]]:
    seed = _stream(config, "twin_error", split)[slot]
    draws = []
    for index in range(count):
        rng = np.random.default_rng([int(seed), int(index)])
        draws.append(
            {"positive": float(rng.normal()), "negative": float(rng.normal())}
        )
    return draws


def noise_only_seed_fn(config: PilotConfig, split: str, slot: int, sigma_count: int):
    base = _stream(config, "noise_only_floor", split)[slot]
    worlds = [str(w) for w in config.get("phase8", "grid", "worlds")]

    def seed(world: str, sigma_index: int, record: int) -> int:
        block = worlds.index(world) * sigma_count + sigma_index
        return int(base) * 100000 + block * 1000 + int(record)

    return seed


def prepare_geometry(config_g: PilotConfig) -> dict[str, Any]:
    """Solve the true scene and calibrate the matched relay (rule 1)."""
    paths = _true_paths(config_g)
    links = _true_links(paths)
    rejection: tuple[str, str] | None = None
    calibration = None
    try:
        calibration = _calibrate(config_g, paths)
    except LatencyFloorError as exc:
        rejection = ("latency_floor", str(exc))
    except GeometryError as exc:
        rejection = ("causality", str(exc))
    if calibration is not None and not calibration.feasible:
        rejection = (
            "power_cap_10dbm",
            f"relay transmit power {calibration.relay_transmit_power_w:.6e} W "
            f"exceeds cap {calibration.max_transmit_power_w:.6e} W",
        )
    records = path_records(paths["target"], scene_label="true_target")
    true_wall = {}
    for side in ("positive", "negative"):
        record = find_outgoing_wall_path(records, f"{side}_wall")
        true_wall[side] = {
            "delay_ns": float(record["delay_ns"]),
            "aoa_deg": float(record["aoa_deg"]),
        }
    gnb = config_g.get("physical", "gnb_position_m")
    return {
        "paths": paths,
        "links": links,
        "calibration": calibration,
        "rejection": rejection,
        "true_wall": true_wall,
        "true_relay_deg": horizontal_range_and_azimuth(
            config_g.get("physical", "relay_position_m"), gnb
        )[1],
        "true_candidate_deg": horizontal_range_and_azimuth(
            config_g.get("physical", "candidate_position_m"), gnb
        )[1],
        "target_path_count": int(paths["target"].path_count),
    }


def accepted_geometry(
    config: PilotConfig, slot: int, split: str = SPLIT
) -> tuple[dict[str, float], PilotConfig, dict[str, Any], list[dict[str, Any]]]:
    """Draw until the matched relay is feasible; log every rejected draw."""
    _require_calibration(split)
    rejected: list[dict[str, Any]] = []
    attempt = 0
    while True:
        geometry = draw_geometry(config, split, slot, attempt)
        config_g = geometry_config(config, geometry)
        prepared = prepare_geometry(config_g)
        if prepared["rejection"] is None:
            return geometry, config_g, prepared, rejected
        reason, detail = prepared["rejection"]
        rejected.append({**geometry, "reason": reason, "detail": detail})
        attempt += 1
        if attempt > 1000:
            raise Phase8Error(f"slot {slot}: no feasible geometry in 1000 draws")


def _geometry_row(
    geometry: dict[str, float],
    prepared: dict[str, Any],
    geometry_id: str,
    rejected: int,
    split: str = SPLIT,
) -> dict[str, Any]:
    calibration = prepared["calibration"]
    return {
        "geometry_id": geometry_id,
        "split": split,
        "slot": geometry["slot"],
        "attempt_accepted": geometry["attempt"],
        "rejected_draws_before": int(rejected),
        "stream_seed": geometry["stream_seed"],
        "candidate_distance_m": geometry["candidate_distance_m"],
        "relay_distance_m": geometry["relay_distance_m"],
        "half_width_m": geometry["half_width_m"],
        "wall_thickness_m": geometry["wall_thickness_m"],
        "relay_added_delay_ns": calibration.measured_added_delay_s * 1e9,
        "relay_gain_abs": calibration.gain_abs,
        "relay_transmit_power_w": calibration.relay_transmit_power_w,
        "relay_transmit_power_dbm": 10.0
        * math.log10(calibration.relay_transmit_power_w / 1e-3),
        "true_target_delay_ns": calibration.target_delay_s * 1e9,
        "true_positive_wall_delay_ns": prepared["true_wall"]["positive"]["delay_ns"],
        "true_negative_wall_delay_ns": prepared["true_wall"]["negative"]["delay_ns"],
        "true_positive_wall_aoa_deg": prepared["true_wall"]["positive"]["aoa_deg"],
        "true_negative_wall_aoa_deg": prepared["true_wall"]["negative"]["aoa_deg"],
        "target_path_count": prepared["target_path_count"],
    }


def run_geometry(
    *,
    config: PilotConfig,
    config_g: PilotConfig,
    geometry: dict[str, float],
    prepared: dict[str, Any],
    geometry_id: str,
    choices: ProcessingChoices,
    noise_full: float,
    noise_by_symbols: dict[int, float],
    direct_weights: torch.Tensor,
    records: int,
    timing: dict[str, float],
    split: str = SPLIT,
) -> pd.DataFrame:
    """Run the full trial grid for one accepted geometry of ``split``."""
    slot = int(geometry["slot"])
    grid = config.section("phase8", "grid")
    sigmas = [float(value) for value in grid["sigma_map_m"]]
    worlds = [str(value) for value in grid["worlds"]]
    realizations = int(grid["twin_realizations_per_sigma"])
    inflations = [
        float(value)
        for value in config.get("phase8", "rules", "coverage", "inflation_candidates")
    ]
    methods = tuple(method for method in METHODS if method in grid["methods"])
    seeds = {"streams": geometry_seed_streams(config, split, slot, records)}
    twin_z = twin_draws(config, split, slot, realizations)
    # A stress runs a sigma subset; each sigma keeps its declared noise-only seed block (A3.3).
    declared = [
        float(value)
        for value in config.section("phase8").get("stress_runtime", {}).get("declared_sigma_map_m", sigmas)
    ]
    declared_seed = noise_only_seed_fn(config, split, slot, len(declared))

    def noise_seed(world: str, sigma_index: int, record: int) -> int:
        return declared_seed(world, declared.index(sigmas[sigma_index]), record)
    rows: list[dict[str, Any]] = []
    representative: dict[str, Any] = {}
    columns = {
        "candidate_distance_m": geometry["candidate_distance_m"],
        "relay_distance_m": geometry["relay_distance_m"],
        "half_width_m": geometry["half_width_m"],
    }
    array_error = config.section("phase8").get("stress_runtime", {}).get("array_error")
    if array_error is not None:
        # A7.1: record r sees realization r mod R of the gNB element errors; planner and twin stay ideal.
        from .stress import array_error_factors, perturb_links

        elements = int(config.get("arrays", "gnb_tx", "num_elements"))
        perturbed = [
            perturb_links(prepared["links"], *array_error_factors(array_error, slot, k, elements))
            for k in range(int(array_error["realizations"]))
        ]
    for world in worlds:
        for record_index in range(records):
            links = (
                prepared["links"]
                if array_error is None
                else perturbed[record_index % len(perturbed)]
            )
            base = _acquire_record(
                config=config_g,
                choices=choices,
                world=world,
                record_index=record_index,
                direct_weights=direct_weights,
                links=links,
                calibration=prepared["calibration"],
                noise_full=noise_full,
                timing=timing,
                seeds=seeds,
            )
            representative.clear()
            for row in evaluate_record(
                config=config_g,
                choices=choices,
                world=world,
                record_index=record_index,
                base=base,
                sigmas=sigmas,
                twin_z=twin_z,
                seeds=seeds,
                direct_weights=direct_weights,
                links=links,
                calibration=prepared["calibration"],
                noise_full=noise_full,
                noise_by_symbols=noise_by_symbols,
                timing=timing,
                representative=representative,
                true_candidate_deg=prepared["true_candidate_deg"],
                true_relay_deg=prepared["true_relay_deg"],
                methods=methods,
                geometry_id=geometry_id,
                split=split,
                inflations=inflations,
                true_wall=prepared["true_wall"],
                noise_only_seed=noise_seed,
            ):
                row.update(columns)
                if array_error is not None:
                    row["array_error_realization"] = record_index % len(perturbed)
                rows.append(row)
            del base
    representative.clear()
    return pd.DataFrame(rows)


def _json_dump(value: Any) -> str:
    return json.dumps(_jsonable(value), indent=2, sort_keys=False)


def _shard_paths(output: Path, slot: int) -> tuple[Path, Path]:
    shards = output / "shards"
    return (
        shards / f"geometry_{slot:02d}.parquet",
        shards / f"geometry_{slot:02d}.json",
    )


def run_calibration(
    config_path: str | Path,
    output_path: str | Path = "results/phase8a",
    *,
    require_h100: bool = True,
    geometry_limit: int | None = None,
    record_limit: int | None = None,
    workers: int = 1,
    worker_slots: list[int] | None = None,
    split: str = SPLIT,
) -> int:
    """Run the calibration geometries and compute the frozen thresholds.

    ``workers > 1`` runs the same per-geometry code in parallel processes on the
    same GPU, each owning a fixed subset of slots; ``worker_slots`` is that
    worker mode (no preflight, no analysis).
    """
    config = PilotConfig.load(config_path)
    _require_gpu(config)
    output = Path(output_path)
    (output / "shards").mkdir(parents=True, exist_ok=True)
    _require_calibration(split)
    total_slots = int(
        config.get(
            "phase8",
            "geometry",
            "heldout_geometries" if split == HELDOUT else "calibration_geometries",
        )
    )
    slots = total_slots if geometry_limit is None else min(total_slots, int(geometry_limit))
    records = (
        int(record_limit)
        if record_limit is not None
        else int(config.get("phase8", "grid", "records_per_world"))
    )
    if worker_slots is not None:
        return _run_worker(config, output, worker_slots, records, split)
    from .environment import run_preflight

    if run_preflight(require_h100=require_h100, output_path=output) != 0:
        raise Phase8Error("preflight failed")
    if split == HELDOUT:
        _record_source_commit(output)
    resolved = dict(config.data)
    resolved["phase8_runtime"] = {
        "split": split,
        "geometries": slots,
        "records_per_world": records,
        "held_out_generated": split == HELDOUT,
    }
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "policy": "explicit_per_stream_never_none",
        "split": split,
        "geometry_slots": slots,
        "records_per_world": records,
        "layout": "stream_seed*100000 + block*1000 + record*10 + look",
        "blocks": RECORD_BLOCKS,
        "slots": [
            {
                "slot": slot,
                "physical_geometry": _stream(config, "physical_geometry", split)[slot],
                "twin_error": _stream(config, "twin_error", split)[slot],
                "noise_only_floor": _stream(config, "noise_only_floor", split)[slot],
                "streams": geometry_seed_streams(config, split, slot, records),
            }
            for slot in range(slots)
        ],
    }
    (output / f"seed_manifest_{split}.json").write_text(
        _json_dump(manifest), encoding="utf-8"
    )
    timing: dict[str, Any] = {"per_geometry_s": {}}
    run_start = _now()
    timing["nvml_start"] = _nvml_snapshot()
    gpu_processes, stop_gpu_monitor = _start_gpu_process_monitor(run_start)
    torch.cuda.reset_peak_memory_stats()
    if workers > 1:
        timing.update(
            _run_parallel(config_path, output, slots, records, workers, record_limit, split)
        )
    else:
        timing["per_geometry_s"] = _run_slots(
            config, output, list(range(slots)), records, split
        )["per_geometry_s"]
    timing["workers"] = workers
    timing["end_to_end_s"] = _now() - run_start
    timing["nvml_end"] = _nvml_snapshot()
    stop_gpu_monitor()
    timing["gpu_processes_observed"] = _summarize_gpu_processes(gpu_processes)
    timing["pytorch_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    timing["pytorch_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    if split == HELDOUT:
        return analyze_heldout(
            output, config, timing=timing, expected_geometries=slots, records=records
        )
    return analyze_calibration(
        output, config, timing=timing, expected_geometries=slots, records=records
    )


def _run_parallel(
    config_path: str | Path,
    output: Path,
    slots: int,
    records: int,
    workers: int,
    record_limit: int | None,
    split: str = SPLIT,
) -> dict[str, Any]:
    """Run slot subsets in ``workers`` processes and merge their worker records."""
    assignments = [list(range(slots))[k::workers] for k in range(min(workers, slots))]
    (output / "workers").mkdir(exist_ok=True)
    processes = []
    for assigned in assignments:
        command = [
            sys.executable, "-m", "isac_verify",
            *(["heldout", "--approved"] if split == HELDOUT else ["calibrate"]),
            "--config", str(config_path), "--output", str(output),
            "--worker-slots", ",".join(str(slot) for slot in assigned),
        ]
        if record_limit is not None:
            command += ["--record-limit", str(record_limit)]
        log = open(output / "workers" / f"worker_{assigned[0]:02d}.log", "w", encoding="utf-8")
        processes.append((assigned, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
    failed = []
    for assigned, process, log in processes:
        if process.wait() != 0:
            failed.append(assigned)
        log.close()
    if failed:
        raise Phase8Error(f"calibration workers failed for slots {failed}")
    parts = [
        json.loads((output / "workers" / f"worker_{assigned[0]:02d}.json").read_text(encoding="utf-8"))
        for assigned in assignments
    ]
    # Every worker recomputes the noise calibration from the same seeds; it must agree exactly.
    for part in parts[1:]:
        if part["noise"] != parts[0]["noise"]:
            raise Phase8Error("noise calibration differs between workers")
    merged: dict[str, Any] = {"per_geometry_s": {}}
    for part in parts:
        merged["per_geometry_s"].update(part["per_geometry_s"])
    merged["worker_slots"] = assignments
    merged["noise_calibration_w"] = parts[0]["noise"]
    merged["worker_pytorch_peak_allocated_bytes"] = [p["pytorch_peak_allocated_bytes"] for p in parts]
    merged["worker_pytorch_peak_reserved_bytes"] = [p["pytorch_peak_reserved_bytes"] for p in parts]
    merged["worker_wall_s"] = [p["wall_s"] for p in parts]
    return merged


def _run_worker(
    config: PilotConfig, output: Path, assigned: list[int], records: int, split: str = SPLIT
) -> int:
    start = _now()
    torch.cuda.reset_peak_memory_stats()
    part = _run_slots(config, output, assigned, records, split)
    part["wall_s"] = _now() - start
    part["pytorch_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    part["pytorch_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    (output / "workers" / f"worker_{assigned[0]:02d}.json").write_text(
        _json_dump(part), encoding="utf-8"
    )
    return 0


def _run_slots(
    config: PilotConfig, output: Path, assigned: list[int], records: int, split: str = SPLIT
) -> dict[str, Any]:
    """Run the given slots of ``split``; shared by the sequential and worker paths."""
    per_geometry_s: dict[str, float] = {}
    choices = ProcessingChoices.from_config(config)
    noise_records = int(config.get("phase6", "processing", "noise_calibration_records"))
    noise_full = noise_calibration(
        choices=choices,
        config=config,
        records=noise_records,
        seed=int(config.get("phase6", "processing", "noise_calibration_seed")),
    )
    noise_by_symbols = {
        count: noise_calibration(
            choices=replace(choices, profile=replace(choices.profile, num_symbols=count)),
            config=config,
            records=noise_records,
            seed=7900 + count,
        )
        for count in (8, 16, 32, choices.profile.num_symbols)
    }
    runtime = config.section("phase8").get("stress_runtime", {})
    probe_power = runtime.get("probe_power_dbm")
    direct_weights = direct_beam(
        float(config.get("experiment_a", "reference_beam_phi_deg")),
        dbm_to_watt(config.get("waveform", "nominal_total_power_dbm") if probe_power is None else probe_power),
        ArrayGeometry.from_config(config),
    ).weights
    for slot in assigned:
        geometry_id = f"{'ho' if split == HELDOUT else 'cal'}_{slot:02d}"
        shard, meta_path = _shard_paths(output, slot)
        if shard.exists() and meta_path.exists():
            print(f"{geometry_id}: shard exists, skipping", flush=True)
            continue
        start = _now()
        geometry, config_g, prepared, rejected = accepted_geometry(config, slot, split)
        standoff = config.section("phase8").get("stress_runtime", {}).get("relay_standoff_m")
        if standoff is not None:
            # A5.2: relay at R_T - Delta_R, re-matched by the same rule; an infeasible pair is logged, not redrawn.
            geometry = {**geometry, "relay_distance_m": geometry["candidate_distance_m"] - float(standoff)}
            config_g = geometry_config(config, geometry)
            prepared = prepare_geometry(config_g)
            if prepared["rejection"] is not None:
                pd.DataFrame().to_parquet(shard, index=False)
                meta_path.write_text(_json_dump({
                    "geometry": {"geometry_id": geometry_id, "split": split, **geometry,
                                 "relay_standoff_m": float(standoff),
                                 "standoff_rejection": prepared["rejection"][0],
                                 "standoff_rejection_detail": prepared["rejection"][1]},
                    "rejected": rejected, "rows": 0, "elapsed_s": _now() - start,
                    "geometry_solve_s": _now() - start, "stage_s": {},
                }), encoding="utf-8")
                print(f"{geometry_id}: standoff {standoff} m infeasible ({prepared['rejection'][0]})", flush=True)
                continue
        if probe_power is not None or runtime.get("rcs_reflected_illumination_db") is not None:
            from .stress import scale_reflected_illumination, with_probe_power

            if probe_power is not None:  # A8.1: after acceptance and relay calibration at the nominal power
                config_g = with_probe_power(config_g, float(probe_power))
            if runtime.get("rcs_reflected_illumination_db") is not None:  # A8.2: truth only
                prepared = scale_reflected_illumination(prepared, float(runtime["rcs_reflected_illumination_db"]))
        solve_s = _now() - start
        geom_timing: dict[str, float] = {}
        frame = run_geometry(
            config=config,
            config_g=config_g,
            geometry=geometry,
            prepared=prepared,
            geometry_id=geometry_id,
            choices=choices,
            noise_full=noise_full,
            noise_by_symbols=noise_by_symbols,
            direct_weights=direct_weights,
            records=records,
            timing=geom_timing,
            split=split,
        )
        elapsed = _now() - start
        frame.to_parquet(shard, index=False)
        meta_path.write_text(
            _json_dump(
                {
                    "geometry": _geometry_row(
                        geometry, prepared, geometry_id, len(rejected), split
                    ),
                    "rejected": rejected,
                    "rows": int(len(frame)),
                    "elapsed_s": elapsed,
                    "geometry_solve_s": solve_s,
                    "stage_s": geom_timing,
                }
            ),
            encoding="utf-8",
        )
        per_geometry_s[geometry_id] = elapsed
        print(
            f"{geometry_id}: rows={len(frame)} rejected={len(rejected)} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        torch.cuda.empty_cache()
    return {
        "per_geometry_s": per_geometry_s,
        "noise": {"full": noise_full, **{str(k): v for k, v in noise_by_symbols.items()}},
    }


GPU_MONITOR_INTERVAL_S = 30.0


def _start_gpu_process_monitor(run_start: float) -> tuple[list[dict[str, Any]], Any]:
    """Sample NVML's process list on device 0 every GPU_MONITOR_INTERVAL_S until stopped."""
    samples: list[dict[str, Any]] = []
    stop = threading.Event()

    def sample() -> None:
        while True:
            try:
                snapshot = _nvml_process_snapshot()
            except Exception as exc:  # noqa: BLE001
                snapshot = {"error": f"{type(exc).__name__}: {exc}"}
            samples.append({"t_s": _now() - run_start, **snapshot})
            if stop.wait(GPU_MONITOR_INTERVAL_S):
                return

    thread = threading.Thread(target=sample, name="nvml-process-monitor", daemon=True)
    thread.start()

    def finish() -> None:
        stop.set()
        thread.join()

    return samples, finish


def _summarize_gpu_processes(samples: list[dict[str, Any]]) -> dict[str, Any]:
    processes: dict[str, dict[str, Any]] = {}
    for sample in samples:
        for kind in ("compute_processes", "graphics_processes"):
            for entry in sample.get(kind, []):
                key = f"{kind.split('_')[0]}:{entry['pid']}"
                row = processes.setdefault(
                    key,
                    {"pid": entry["pid"], "kind": kind.split("_")[0], "first_seen_s": sample["t_s"],
                     "samples_seen": 0, "max_used_gpu_memory_bytes": None},
                )
                row["last_seen_s"] = sample["t_s"]
                row["samples_seen"] += 1
                used = entry["used_gpu_memory_bytes"]
                if used is not None:
                    row["max_used_gpu_memory_bytes"] = max(row["max_used_gpu_memory_bytes"] or 0, used)
    return {
        "label": "NVML process list on device 0, host PIDs. Includes this run's own "
        "process, whose host PID is not visible inside the container.",
        "interval_s": GPU_MONITOR_INTERVAL_S,
        "samples": len(samples),
        "sample_errors": [sample for sample in samples if "error" in sample],
        "processes": sorted(processes.values(), key=lambda row: (row["kind"], row["pid"])),
    }


def analyze_calibration(
    output: str | Path,
    config: PilotConfig,
    *,
    timing: dict[str, Any] | None = None,
    expected_geometries: int | None = None,
    records: int | None = None,
) -> int:
    """Compute rules 3-6 from saved shards and write the Phase 8A artifacts."""
    output = Path(output)
    shard_files = sorted((output / "shards").glob("geometry_*.parquet"))
    if not shard_files:
        raise Phase8Error("no calibration shards found")
    frames = [pd.read_parquet(path) for path in shard_files]
    trials = pd.concat(frames, ignore_index=True)
    metas = [
        json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        for path in shard_files
    ]
    if set(trials["split"]) != {SPLIT}:
        raise Phase8Error("calibration artifacts contain a non-calibration split")
    rules = config.section("phase8", "rules")
    inflations = [float(v) for v in rules["coverage"]["inflation_candidates"]]
    coverage = cal.coverage_table(trials, inflations)
    coverage.to_csv(output / "coverage.csv", index=False)
    choice = cal.select_inflation(
        coverage,
        method=str(rules["coverage"]["selection_method"]),
        required=float(rules["coverage"]["required"]),
        candidates=inflations,
    )
    factor = float(choice["value"])
    fitted = cal.fit_thresholds(
        trials,
        factor,
        false_rate=float(rules["floor"]["false_exceedance_rate"]),
        replay_rate_max=float(rules["band"]["replay_supported_rate_max"]),
        s_hi_quantile=float(rules["band"]["s_hi_quantile"]),
        common_quantile=float(rules["common"]["replay_quantile"]),
    )
    classified = cal.classify_trials(trials, fitted, factor)
    classified.to_parquet(output / "calibration_trials.parquet", index=False)
    rates = cal.rate_table(classified)
    rates.to_csv(output / "rates.csv", index=False)
    geometries = pd.DataFrame([meta["geometry"] for meta in metas])
    geometries.to_csv(output / "geometries_calibration.csv", index=False)
    rejected_rows = [row for meta in metas for row in meta["rejected"]]
    rejected_columns = [
        "slot",
        "attempt",
        "stream_seed",
        "candidate_distance_m",
        "relay_distance_m",
        "half_width_m",
        "wall_thickness_m",
        "reason",
        "detail",
    ]
    pd.DataFrame(rejected_rows, columns=rejected_columns).to_csv(
        output / "rejected_draws.csv", index=False
    )
    floor_min = int(rules["floor"]["min_noise_only_records"])
    thresholds: dict[str, Any] = {
        "split": SPLIT,
        "plan": "docs/ANALYSIS_PLAN.md",
        "gate_inflation_c": {
            **choice,
            "rule": "rule 6: smallest c in the declared candidates whose pooled "
            "target-world gate coverage for the primary method is >= required",
        },
        "floor_k": {},
        "s_hi": {},
        "s_lo": {},
        "t_common": {},
    }
    for method, values in fitted["mapped"].items():
        thresholds["floor_k"][method] = {
            **values["floor_k"],
            "rule": "rule 3: order statistic x_(ceil(0.99 n)) of the noise-only "
            "wall-gate excess (E_W - N_W)/N_W through the actual pipeline",
            "min_records_required": floor_min,
            "min_records_met": bool(values["floor_k"]["n"] >= floor_min),
        }
        thresholds["s_hi"][method] = {
            **values["s_hi"],
            "rule": "rule 4: quantile of target-world S among floor-passing rows",
        }
        thresholds["s_lo"][method] = {
            **values["s_lo"],
            "rule": "rule 4: smallest S with replay-world SUPPORTED rate <= max_rate",
        }
    for method, values in fitted["common"].items():
        thresholds["t_common"][method] = {
            **values,
            "rule": "rule 5: replay-world quantile of wall_excess_snr pooled over sigma",
        }
    thresholds["decision_thresholds"] = cal.thresholds_for_decision(fitted)
    thresholds["config_block"] = {
        "gate_inflation_c": factor,
        "thresholds": thresholds["decision_thresholds"],
        "t_common": {m: v["value"] for m, v in fitted["common"].items()},
    }
    (output / "thresholds.json").write_text(_json_dump(thresholds), encoding="utf-8")
    expected = None
    if expected_geometries is not None and records is not None:
        expected = (
            expected_geometries
            * len(config.get("phase8", "grid", "sigma_map_m"))
            * len(config.get("phase8", "grid", "worlds"))
            * records
            * len(METHODS)
        )
    rejected_frame = pd.DataFrame(rejected_rows, columns=rejected_columns)
    summary: dict[str, Any] = {
        "split": SPLIT,
        "geometries": int(len(geometries)),
        "rows": int(len(trials)),
        "expected_rows": expected,
        "rows_match": (expected is None) or int(len(trials)) == expected,
        "held_out_generated": False,
        "gate_inflation": choice,
        "rejected_draws": int(len(rejected_frame)),
        "rejected_by_reason": rejected_frame["reason"].value_counts().to_dict()
        if len(rejected_frame)
        else {},
        "noise_only_records_per_method": {
            method: values["floor_k"]["n"] for method, values in fitted["mapped"].items()
        },
        "noise_only_requirement_met": all(
            values["floor_k"]["n"] >= floor_min for values in fitted["mapped"].values()
        ),
        "acquisition_failed_rows": int((trials["status"] == "ACQUISITION_FAILED").sum()),
        "decision_counts": classified["decision"].value_counts().to_dict(),
        "decision_reason_counts": classified["decision_reason"].value_counts().to_dict(),
        "wall_gate_truncated_rows": int(
            (pd.to_numeric(trials["wall_gate_truncated"], errors="coerce") == 1.0).sum()
        ),
        "path_availability_counts": trials.groupby(["method", "path_availability"])
        .size()
        .rename("n")
        .reset_index()
        .to_dict(orient="records"),
        "realized_pooled_rates": {},
        "status": "CALIBRATION_COMPLETE",
    }
    for method in cal.MAPPED_METHODS:
        subset = classified[classified["method"] == method]
        summary["realized_pooled_rates"][method] = {
            "replay_supported": float(
                subset[subset["world"] == "replay"]["supported"].mean()
            ),
            "target_supported": float(
                subset[subset["world"] == "target"]["supported"].mean()
            ),
        }
    for method in cal.WALL_PROBING_METHODS:
        subset = classified[classified["method"] == method]
        summary["realized_pooled_rates"].setdefault(method, {}).update(
            {
                "replay_supported_common": float(
                    subset[subset["world"] == "replay"]["supported_common"].mean()
                ),
                "target_supported_common": float(
                    subset[subset["world"] == "target"]["supported_common"].mean()
                ),
            }
        )
    (output / "calibration_summary.json").write_text(
        _json_dump(summary), encoding="utf-8"
    )
    timing = dict(timing or {})
    timing["per_geometry_s_from_shards"] = {
        meta["geometry"]["geometry_id"]: meta["elapsed_s"] for meta in metas
    }
    timing["stage_s_from_shards"] = {
        meta["geometry"]["geometry_id"]: meta["stage_s"] for meta in metas
    }
    timing["rows"] = int(len(trials))
    (output / "timing.json").write_text(_json_dump(timing), encoding="utf-8")
    print(
        f"phase 8a calibration complete: rows={len(trials)} "
        f"geometries={len(geometries)} c={factor:g} "
        f"coverage_met={choice['met']} rejected={len(rejected_frame)}",
        flush=True,
    )
    return 0


def _record_source_commit(output: Path) -> None:
    """Add the host-recorded source commit (scripts/drun.sh) to environment.json; refuse if unknown or dirty."""
    commit = os.environ.get("ISAC_GIT_COMMIT", "")
    dirty = os.environ.get("ISAC_GIT_DIRTY", "")
    if not commit or dirty != "0":
        raise Phase8Error(
            f"held-out run needs a clean, known source commit (commit={commit!r}, dirty={dirty!r})"
        )
    path = output / "environment.json"
    environment = json.loads(path.read_text(encoding="utf-8"))
    environment["source_commit"] = {"commit": commit, "dirty_paths": int(dirty)}
    path.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def frozen_decision(config: PilotConfig) -> tuple[float, dict[str, Any]]:
    """Gate inflation and thresholds from ``phase8.frozen`` only, in ``classify_trials`` form."""
    frozen = config.section("phase8", "frozen")
    if frozen.get("gate_inflation_c") is None or not frozen.get("thresholds") or not frozen.get("t_common"):
        raise Phase8Error("phase8.frozen is incomplete; the held-out run needs frozen thresholds")
    fitted = {
        "mapped": {
            method: {
                "floor_k": {"value": float(values["noise_sigmas"])},
                "s_lo": {"value": float(values["supported_min"])},
                "s_hi": {"value": float(values["supported_max"])},
            }
            for method, values in frozen["thresholds"].items()
        },
        "common": {method: {"value": float(value)} for method, value in frozen["t_common"].items()},
    }
    if set(fitted["mapped"]) != set(cal.MAPPED_METHODS) or set(fitted["common"]) != set(
        cal.WALL_PROBING_METHODS
    ):
        raise Phase8Error("phase8.frozen does not cover every method")
    return float(frozen["gate_inflation_c"]), fitted


def analyze_heldout(
    output: str | Path,
    config: PilotConfig,
    *,
    timing: dict[str, Any] | None = None,
    expected_geometries: int | None = None,
    records: int | None = None,
) -> int:
    """Apply the frozen decision (``phase8.frozen`` only) to the held-out shards. Nothing is fitted."""
    output = Path(output)
    shard_files = sorted((output / "shards").glob("geometry_*.parquet"))
    if not shard_files:
        raise Phase8Error("no held-out shards found")
    trials = pd.concat([pd.read_parquet(path) for path in shard_files], ignore_index=True)
    metas = [json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) for path in shard_files]
    if set(trials["split"]) != {HELDOUT}:
        raise Phase8Error("held-out artifacts contain a non-held-out split")
    factor, fitted = frozen_decision(config)
    classified = cal.classify_trials(trials, fitted, factor)
    classified.to_parquet(output / "heldout_trials.parquet", index=False)
    cal.rate_table(classified).to_csv(output / "rates.csv", index=False)
    inflations = [float(v) for v in config.get("phase8", "rules", "coverage", "inflation_candidates")]
    cal.coverage_table(trials, inflations).to_csv(output / "coverage.csv", index=False)
    geometries = pd.DataFrame([meta["geometry"] for meta in metas])
    geometries.to_csv(output / "geometries_heldout.csv", index=False)
    rejected = pd.DataFrame([row for meta in metas for row in meta["rejected"]])
    rejected.to_csv(output / "rejected_draws.csv", index=False)
    expected = None
    if expected_geometries is not None and records is not None:
        expected = (
            expected_geometries
            * len(config.get("phase8", "grid", "sigma_map_m"))
            * len(config.get("phase8", "grid", "worlds"))
            * records
            * len(config.get("phase8", "grid", "methods"))
        )
    summary = {
        "split": HELDOUT,
        "geometries": int(len(geometries)),
        "rows": int(len(trials)),
        "expected_rows": expected,
        "rows_match": (expected is None) or int(len(trials)) == expected,
        "decision_source": "configs phase8.frozen",
        "gate_inflation_c": factor,
        "rejected_draws": int(len(rejected)),
        "acquisition_failed_rows": int((trials["status"] == "ACQUISITION_FAILED").sum()),
        "decision_counts": classified["decision"].value_counts().to_dict(),
        "status": "HELDOUT_COMPLETE",
    }
    (output / "heldout_summary.json").write_text(_json_dump(summary), encoding="utf-8")
    timing = dict(timing or {})
    timing["per_geometry_s_from_shards"] = {m["geometry"]["geometry_id"]: m["elapsed_s"] for m in metas}
    timing["stage_s_from_shards"] = {m["geometry"]["geometry_id"]: m["stage_s"] for m in metas}
    timing["rows"] = int(len(trials))
    (output / "timing.json").write_text(_json_dump(timing), encoding="utf-8")
    print(
        f"phase 8 held-out complete: rows={len(trials)} geometries={len(geometries)} "
        f"c={factor:g} rejected={len(rejected)}",
        flush=True,
    )
    return 0
