"""Phase 7 report rebuilt entirely from saved smoke results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .beams import ArrayGeometry, BeamResult, beam_gain
from .config import PilotConfig
from .metrics import METHOD_LABELS, METHOD_ORDER, summarize_trials
from .receiver import (
    ProcessingChoices,
    angular_delay_map,
    delay_doppler_map,
)


class ReportError(RuntimeError):
    """Raised when saved results are incomplete."""


def _input_dir(path: str | Path) -> Path:
    root = Path(path)
    if (root / "trials.parquet").exists():
        return root
    if (root / "phase7" / "trials.parquet").exists():
        return root / "phase7"
    raise ReportError(f"{root} does not contain trials.parquet")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)


def _save(fig: Any, output: Path, name: str, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=dpi)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def _geometry_figure(
    output: Path,
    *,
    config: PilotConfig,
    geometry: dict[str, Any],
    paths: pd.DataFrame,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    walls = config.section("physical", "walls")
    for side, spec in walls.items():
        offset = float(spec["offset_m"])
        ax.plot(
            [float(spec["x_min_m"]), float(spec["x_max_m"])],
            [offset, offset],
            color="#444444",
            linewidth=2.0,
            label=f"true {side} wall",
        )
    twin = geometry.get("twin_wall_offsets_m", {})
    for sigma, entries in twin.items():
        for entry in entries[:1]:
            ax.plot(
                [float(walls["positive"]["x_min_m"]), float(walls["positive"]["x_max_m"])],
                [float(entry["positive"]), float(entry["positive"])],
                linestyle="--",
                linewidth=1.0,
                color="#888888" if float(sigma) == 0.0 else "#c44e52",
                label=f"twin wall sigma={sigma} m",
            )
    gnb = geometry["gnb_position_m"]
    relay = geometry["relay_position_m"]
    candidate = geometry["candidate_position_m"]
    ax.scatter([gnb[0]], [gnb[1]], marker="s", s=70, color="#1f77b4", label="gNB")
    ax.scatter([relay[0]], [relay[1]], marker="^", s=70, color="#ff7f0e", label="relay")
    ax.scatter(
        [candidate[0]], [candidate[1]], marker="o", s=70, color="#2ca02c", label="candidate"
    )
    if not paths.empty:
        plotted: set[tuple[Any, ...]] = set()
        for _, row in paths.iterrows():
            if isinstance(row.get("vertex_points_json"), str):
                vertices = json.loads(row["vertex_points_json"])
            else:
                vertices = row.get("vertex_points_m")
            if isinstance(vertices, np.ndarray):
                vertices = vertices.tolist()
            if vertices is None or len(vertices) == 0:
                continue
            vertices = [tuple(float(value) for value in point) for point in vertices]
            signature = (str(row["link"]), str(row["family"]), tuple(vertices))
            if signature in plotted:
                continue
            plotted.add(signature)
            xs = [float(gnb[0])] + [float(point[0]) for point in vertices] + [float(gnb[0])]
            ys = [float(gnb[1])] + [float(point[1]) for point in vertices] + [float(gnb[1])]
            if str(row["link"]) == "target" and "wall" in str(row["family"]):
                ax.plot(xs, ys, color="#9467bd", linewidth=0.9, alpha=0.8)
            elif str(row["link"]) == "downlink":
                ax.plot(xs, ys, color="#ff7f0e", linewidth=0.7, alpha=0.5)
            elif str(row["link"]) == "uplink":
                ax.plot(xs, ys, color="#d62728", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Smoke geometry: true walls, twin walls, and solved paths")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    _save(fig, output, "01_geometry", dpi)


def _experiment_a_figure(
    output: Path,
    *,
    trials: pd.DataFrame,
    dpi: int,
) -> None:
    mapped = trials[trials["method"].isin(["exact_oracle", "imperfect_narrow", "imperfect_robust"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for method in ["exact_oracle", "imperfect_narrow", "imperfect_robust"]:
        group = mapped[mapped["method"] == method]
        target = group[group["world"] == "target"]["E_W"].to_numpy(dtype=float)
        replay = group[group["world"] == "replay"]["E_W"].to_numpy(dtype=float)
        count = min(len(target), len(replay))
        if count == 0:
            continue
        ax.scatter(
            target[:count],
            replay[:count],
            s=16,
            alpha=0.65,
            label=METHOD_LABELS[method],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("target-world wall energy [W]")
    ax.set_ylabel("replay-world wall energy [W]")
    ax.set_title("Experiment A paired worlds: target versus replay")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    _save(fig, output, "02_experiment_a", dpi)


def _beampattern_figure(
    output: Path,
    *,
    config: PilotConfig,
    arrays: Any,
    dpi: int,
) -> None:
    weights = {}
    for key in ("direct_weights", "target_exact_oracle_weights", "target_imperfect_robust_weights"):
        if key in arrays:
            weights[key] = torch.tensor(arrays[key]).to("cuda", torch.complex64)
    if not weights:
        return
    array = ArrayGeometry.from_config(config)
    angles = np.arange(-60.0, 60.0 + 1e-9, 0.25)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, value in weights.items():
        beam = BeamResult(weights=value, status="ok", phi_deg=0.0)
        gains = np.asarray(
            [10.0 * math.log10(max(beam_gain(float(angle), beam, array), 1e-30)) for angle in angles]
        )
        gains -= gains.max()
        ax.plot(angles, gains, linewidth=1.3, label=key.replace("_", " "))
    ax.set_xlabel("azimuth [deg]")
    ax.set_ylabel("gain relative to peak [dB]")
    ax.set_title("Saved smoke beams: direct and wall probes")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    _save(fig, output, "03_beampatterns", dpi)


def _delay_figure(
    output: Path,
    *,
    config: PilotConfig,
    trials: pd.DataFrame,
    dpi: int,
) -> None:
    refs = {
        "direct": float(config.get("geometry_reference", "candidate_direct_round_trip_delay_ns")),
        "target wall": float(config.get("geometry_reference", "target_wall_direct_return_delay_ns")),
        "relay replay": float(config.get("geometry_reference", "relay_wall_plus_replay_delay_ns")),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, value in refs.items():
        ax.axvline(value, color="#222222", linestyle=":", linewidth=1.0)
        ax.text(value, 0.98, label, rotation=90, fontsize=7, va="top", ha="right")
    subset = trials[~trials["method"].isin(["direct_only", "mapfree_two_beam", "mapfree_coarse_to_fine"])]
    for method in ["exact_oracle", "imperfect_narrow", "imperfect_robust"]:
        group = subset[subset["method"] == method]
        if group.empty:
            continue
        values = group["wall_gate_delay_min_ns"].dropna().unique()
        ax.scatter(
            values,
            [list(METHOD_ORDER).index(method)] * len(values),
            s=16,
            alpha=0.7,
            label=METHOD_LABELS[method],
        )
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABELS[name] for name in METHOD_ORDER], fontsize=7)
    ax.set_xlabel("wall-gate lower delay [ns]")
    ax.set_title("Predicted wall gates against saved true-path delays")
    ax.grid(True, alpha=0.25)
    _save(fig, output, "04_predicted_delays", dpi)


def _spectra_figure(
    output: Path,
    *,
    config: PilotConfig,
    arrays: Any,
    dpi: int,
) -> None:
    choices = ProcessingChoices.from_config(config)
    entries = [
        ("target_direct_h_eff", "target direct"),
        ("target_exact_oracle_wall_h_eff", "target exact wall"),
        ("replay_exact_oracle_wall_h_eff", "replay exact wall"),
    ]
    entries = [(key, label) for key, label in entries if key in arrays]
    if not entries:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for key, label in entries:
        h_eff = torch.tensor(arrays[key]).to("cuda", torch.complex64)
        dd = delay_doppler_map(h_eff, choices)
        angular = angular_delay_map(dd, choices, noise_per_cell_w=0.0)
        flat = int(torch.argmax(angular.power_w).item())
        lag_index, angle_index = divmod(flat, angular.power_w.shape[1])
        delay = angular.power_w[:, angle_index].detach().cpu().numpy()
        angle = angular.power_w[lag_index, :].detach().cpu().numpy()
        delay_db = 10.0 * np.log10(np.maximum(delay, delay.max() * 1e-9))
        angle_db = 10.0 * np.log10(np.maximum(angle, angle.max() * 1e-9))
        axes[0].plot(angular.delay_s.cpu().numpy() * 1e9, delay_db, label=label)
        axes[1].plot(angular.aoa_deg.cpu().numpy(), angle_db, label=label)
    axes[0].set_xlim(380.0, 470.0)
    axes[0].set_xlabel("delay [ns]")
    axes[0].set_ylabel("normalized power [dB]")
    axes[1].set_xlabel("AoA [deg]")
    axes[1].set_ylabel("normalized power [dB]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Representative saved target and replay spectra")
    _save(fig, output, "05_spectra", dpi)


def _score_figure(output: Path, *, trials: pd.DataFrame, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    panels = (
        (
            axes[0],
            trials[trials["S_available"]],
            "S",
            "Twin-normalized S (available mapped rows)",
        ),
        (
            axes[1],
            trials[trials["wall_excess_snr_available"]],
            "wall_excess_snr",
            "Wall excess SNR (available extra-action rows)",
        ),
    )
    for ax, frame, column, title in panels:
        labels = []
        data = []
        for method in METHOD_ORDER:
            for sigma in sorted(frame["sigma_map_m"].unique()) if not frame.empty else []:
                values = frame[
                    (frame["method"] == method) & (frame["sigma_map_m"] == sigma)
                ][column].to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                labels.append(
                    f"{METHOD_LABELS[method]}\n$\\sigma$={sigma:g}, n={len(values)}"
                )
                data.append(values)
        if data:
            ax.boxplot(data, showfliers=False)
            ax.set_xticklabels(labels, rotation=75, fontsize=6)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("twin-normalized S")
    axes[1].set_ylabel("wall excess SNR")
    _save(fig, output, "06_score_distributions", dpi)


def _status_figure(output: Path, *, summary: pd.DataFrame, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    for method in METHOD_ORDER:
        group = summary[summary["method"] == method]
        for world, style in (("target", "-"), ("replay", "--")):
            subset = group[group["world"] == world]
            if subset.empty:
                continue
            axes[0].plot(subset["sigma_map_m"], subset["acquisition_rate"], style, marker="o", markersize=3, label=f"{METHOD_LABELS[method]} {world}")
            axes[1].plot(subset["sigma_map_m"], subset["not_observable_rate"], style, marker="o", markersize=3)
    axes[0].set_ylabel("acquisition rate")
    axes[1].set_ylabel("NOT_OBSERVABLE rate")
    for ax in axes:
        ax.set_xlabel("sigma_map [m]")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=5, ncol=2)
    fig.suptitle("Smoke status rates versus declared map error")
    _save(fig, output, "07_status_rates", dpi)


def _timing_figure(output: Path, *, timing: dict[str, Any], dpi: int) -> None:
    keys = [
        "cold_jit_s",
        "true_scene_solve_s",
        "twin_scene_solve_s",
        "planning_s",
        "propagation_s",
        "receiver_s",
        "serialization_s",
        "warm_steady_state_s",
        "end_to_end_s",
    ]
    labels = [key.replace("_s", "").replace("_", " ") for key in keys]
    values = [float(timing.get(key) or 0.0) for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    axes[0].barh(labels, values, color="#4c72b0")
    axes[0].set_xscale("symlog")
    axes[0].set_xlabel("seconds (symlog)")
    axes[0].set_title("Smoke timing by category")
    start = timing.get("nvml_start", {})
    end = timing.get("nvml_end", {})
    memory_labels = ["NVML used start", "NVML used end", "PyTorch peak allocated", "PyTorch peak reserved"]
    memory_values = [
        float(start.get("used_bytes", 0) or 0),
        float(end.get("used_bytes", 0) or 0),
        float(timing.get("pytorch_peak_allocated_bytes", 0) or 0),
        float(timing.get("pytorch_peak_reserved_bytes", 0) or 0),
    ]
    axes[1].barh(memory_labels, memory_values, color="#55a868")
    axes[1].set_xlabel("bytes")
    axes[1].set_title("Memory (NVML separate from allocator)")
    for ax in axes:
        ax.grid(True, axis="x", alpha=0.25)
    _save(fig, output, "09_runtime_memory", dpi)


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def build_report(input_path: str | Path, output_path: str | Path) -> int:
    source = _input_dir(input_path)
    output = Path(output_path)
    _prepare(output)
    trials = pd.read_parquet(source / "trials.parquet")
    paths = pd.read_parquet(source / "paths_diagnostic.parquet")
    config = PilotConfig.load(source / "resolved_config.yaml")
    timing = _read_json(source / "timing.json")
    decision = _read_json(source / "scientific_decision.json")
    geometry = _read_json(source / "geometry_reference.json")
    dpi = int(config.get("phase7", "report", "dpi"))
    records_per_cell = int(trials.groupby(["method", "sigma_map_m", "world"]).size().max())
    summary = summarize_trials(trials)
    summary.to_csv(output / "aggregate_metrics.csv", index=False)
    arrays: dict[str, Any] = {}
    npz_path = source / "representative_waveforms.npz"
    if npz_path.exists():
        with np.load(npz_path) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
    _geometry_figure(output, config=config, geometry=geometry, paths=paths, dpi=dpi)
    _experiment_a_figure(output, trials=trials, dpi=dpi)
    _beampattern_figure(output, config=config, arrays=arrays, dpi=dpi)
    _delay_figure(output, config=config, trials=trials, dpi=dpi)
    _spectra_figure(output, config=config, arrays=arrays, dpi=dpi)
    _score_figure(output, trials=trials, dpi=dpi)
    _status_figure(output, summary=summary, dpi=dpi)
    _timing_figure(output, timing=timing, dpi=dpi)
    lines = [
        "# PHASE 7 SMOKE REPORT",
        "",
        "**Smoke stage, no statistical conclusion.** Thresholds are null; no "
        "`SUPPORTED` decision is issued. This report verifies implementation, "
        "pairing, output completeness, and reproducibility only.",
        "",
        "## Scope",
        "",
        "- Geometry: nominal smoke corridor.",
        "- Map-error settings: " + ", ".join(str(value) for value in geometry.get("twin_wall_offsets_m", {}).keys()) + " m.",
        "- Worlds: target and replay on the same true map.",
        "- Records per method/sigma/world: " + str(records_per_cell),
        "- Twin realizations per sigma: "
        + str(int(config.get("phase7_runtime", "twin_realizations_per_sigma"))),
        "- Scientific decision: " + str(decision.get("scientific_decision")),
        "",
        "## Integrity",
        "",
        "```json",
        json.dumps(decision.get("implementation_validity", {}), indent=2),
        "```",
        "",
        "## Aggregate table",
        "",
        "| method | world | sigma [m] | n | acquisition | not observable | S n | S median | S 5-95% | wall-excess n | wall-excess median | null depth [dB] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        s_range = (
            f"{_format_number(row['S_q05'])} to {_format_number(row['S_q95'])}"
            if pd.notna(row.get("S_q05"))
            else "n/a"
        )
        lines.append(
            "| {method} | {world} | {sigma:.3g} | {records} | {acq:.1%} | {no:.1%} | "
            "{sn} | {smed} | {srange} | {exn} | {exmed} | {null} |".format(
                method=METHOD_LABELS.get(row["method"], row["method"]),
                world=row["world"],
                sigma=float(row["sigma_map_m"]),
                records=int(row["records"]),
                acq=float(row["acquisition_rate"]),
                no=float(row["not_observable_rate"]),
                sn=int(row["S_available_count"]),
                smed=_format_number(row.get("S_median")),
                srange=s_range,
                exn=int(row["wall_excess_snr_available_count"]),
                exmed=_format_number(row.get("wall_excess_snr_median")),
                null=_format_number(row.get("null_depth_db_median")),
            )
        )
    lines += [
        "",
        "`S` is populated only where `S_available=True` (mapped methods). "
        "`wall_excess_snr` is computed the same way for every method that spends "
        "an extra wall/scan action and is populated only where "
        "`wall_excess_snr_available=True`. All unavailable entries are NaN; the "
        "table reports the available count for each column.",
        "",
        "## Figures",
        "",
        "1. `01_geometry` — true/twin walls and solved target/relay paths.",
        "2. `02_experiment_a` — paired target versus replay wall energy.",
        "3. `03_beampatterns` — saved direct, exact, and robust beams.",
        "4. `04_predicted_delays` — predicted wall gates against true-path delays.",
        "5. `05_spectra` — representative target/replay delay and angle spectra.",
        "6. `06_score_distributions` — score distributions without a threshold.",
        "7. `07_status_rates` — acquisition and NOT_OBSERVABLE rates versus sigma.",
        "9. `09_runtime_memory` — timing categories and separate memory measurements.",
        "",
        "## Timing",
        "",
        "```json",
        json.dumps(timing, indent=2),
        "```",
        "",
        "## Claim limits",
        "",
        "- No confidence interval, p-value, or threshold-based scientific claim is made.",
        "- The smoke run shares one base acquisition per world across records; it does "
        "not estimate acquisition probability.",
        "- Results are implementation evidence for the next pilot stage, not an "
        "estimate of wall-assisted sensing performance.",
        "",
        "## Files",
        "",
        "- `trials.parquet`",
        "- `aggregate_metrics.csv`",
        "- `scientific_decision.json`",
        "- `seed_manifest.json`",
        "- `geometry_reference.json`",
        "- `timing.json`",
        "- `paths_diagnostic.parquet`",
        "- `representative_waveforms.npz`",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote report to {output}")
    return 0
