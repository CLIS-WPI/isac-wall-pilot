"""GPU environment checks and preflight manifest (BRIEF §3.4).

Everything here is measured at runtime. Any CUDA/OptiX failure raises
:class:`PreflightError`; there is no CPU/LLVM fallback.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

SPEED_OF_LIGHT = 299_792_458.0

PREFLIGHT_SCENE_XML = """<scene version="3.0.0">
  <bsdf type="itu-radio-material" id="wall-mat">
    <string name="type" value="concrete"/>
    <float name="thickness" value="0.1"/>
  </bsdf>
  <shape type="rectangle" id="wall-a">
    <transform name="to_world">
      <scale x="20" y="5" z="1"/>
      <rotate x="1" y="0" z="0" angle="90"/>
      <translate x="0" y="-2" z="2.5"/>
    </transform>
    <ref id="wall-mat" name="bsdf"/>
  </shape>
  <shape type="rectangle" id="wall-b">
    <transform name="to_world">
      <scale x="5" y="10" z="1"/>
      <rotate x="0" y="1" z="0" angle="90"/>
      <translate x="8" y="3" z="2.5"/>
    </transform>
    <ref id="wall-mat" name="bsdf"/>
  </shape>
</scene>"""

TX_POSITION = (-4.0, 2.0, 2.0)
RX_POSITION = (4.0, 2.0, 2.0)
TARGET_POSITION = (0.0, 5.0, 2.0)

PACKAGE_NAMES = (
    "torch",
    "sionna",
    "sionna-rt",
    "mitsuba",
    "drjit",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "matplotlib",
    "PyYAML",
    "pytest",
    "nvidia-ml-py",
    "isac-verify",
)


class PreflightError(RuntimeError):
    """Raised when a mandatory CUDA/OptiX preflight check fails."""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _python_info() -> dict[str, Any]:
    return {
        "version": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": sys.executable,
        "packages": _package_versions(),
    }


def _nvml_device_snapshot() -> dict[str, Any]:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        return {
            "name": name.decode() if isinstance(name, bytes) else name,
            "uuid": uuid.decode() if isinstance(uuid, bytes) else uuid,
            "total_memory_bytes": int(memory.total),
            "used_memory_bytes": int(memory.used),
            "free_memory_bytes": int(memory.free),
            "compute_capability": {"major": int(major), "minor": int(minor)},
            "driver_version": pynvml.nvmlSystemGetDriverVersion(),
            "nvml_version": pynvml.nvmlSystemGetNVMLVersion(),
            "label": "NVML device observation (whole device/context, not "
            "PyTorch allocator statistics)",
        }
    finally:
        pynvml.nvmlShutdown()


def _nvml_process_snapshot() -> dict[str, Any]:
    import pynvml

    def convert(processes: Any) -> list[dict[str, Any]]:
        result = []
        for process in processes:
            entry: dict[str, Any] = {"pid": int(process.pid)}
            used = getattr(process, "usedGpuMemory", None)
            entry["used_gpu_memory_bytes"] = None if used is None else int(used)
            result.append(entry)
        return result

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return {
            "compute_processes": convert(
                pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            ),
            "graphics_processes": convert(
                pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            ),
            "label": "NVML process observations (host PIDs; not PyTorch "
            "allocator statistics)",
        }
    finally:
        pynvml.nvmlShutdown()


def _torch_checks(require_h100: bool) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise PreflightError("torch.cuda.is_available() is False")
    if torch.cuda.device_count() < 1:
        raise PreflightError("No CUDA device is visible to PyTorch")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(0)
    torch_name = torch.cuda.get_device_name(0)

    if require_h100 and "H100" not in torch_name:
        raise PreflightError(
            f"--require-h100 set but torch reports GPU name {torch_name!r}"
        )

    torch.manual_seed(0)
    torch.cuda.synchronize(device)

    started = time.perf_counter()
    matrix = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    product = matrix @ matrix
    torch.cuda.synchronize(device)
    matmul_seconds = time.perf_counter() - started
    if not bool(torch.isfinite(product).all().item()):
        raise PreflightError("CUDA matrix product contains non-finite values")

    started = time.perf_counter()
    signal = torch.randn(16, 2048, 14, device=device, dtype=torch.complex64)
    spectrum = torch.fft.fft(signal, dim=1)
    torch.cuda.synchronize(device)
    fft_seconds = time.perf_counter() - started
    if not bool(torch.isfinite(torch.view_as_real(spectrum)).all().item()):
        raise PreflightError("CUDA FFT contains non-finite values")

    del matrix, product, signal, spectrum
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    batch_description = (
        "complex64[16, 2048, 14] OFDM-like batch, FFT along the 2048 axis, "
        "then elementwise power"
    )
    batch = torch.randn(16, 2048, 14, device=device, dtype=torch.complex64)
    batch_spectrum = torch.fft.fft(batch, dim=1)
    batch_power = batch_spectrum * batch_spectrum.conj()
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(torch.view_as_real(batch_power)).all().item()):
        raise PreflightError("representative CUDA batch contains non-finite values")
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    del batch, batch_spectrum, batch_power
    torch.cuda.empty_cache()

    uuid = getattr(properties, "uuid", None)
    if uuid is not None:
        uuid = str(uuid)

    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "selected_device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": {
            "name": torch_name,
            "uuid": uuid,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": {
                "major": int(properties.major),
                "minor": int(properties.minor),
            },
            "multi_processor_count": int(properties.multi_processor_count),
        },
        "cuda_matmul": {
            "shape": [4096, 4096],
            "dtype": "float32",
            "device": str(device),
            "seconds": matmul_seconds,
            "finite": True,
        },
        "cuda_fft": {
            "shape": [16, 2048, 14],
            "dtype": "complex64",
            "device": str(device),
            "seconds": fft_seconds,
            "finite": True,
        },
        "memory": {
            "pytorch_allocator": {
                "batch": batch_description,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "label": "PyTorch caching allocator statistics",
            },
            "nvml_device": _nvml_device_snapshot(),
            "nvml_processes": _nvml_process_snapshot(),
        },
    }


def _require_cuda_variant() -> dict[str, Any]:
    import drjit
    import mitsuba as mi
    import sionna.rt  # noqa: F401  (selects the installed Sionna RT backend)

    variant = mi.variant()
    if variant is None or not variant.startswith("cuda"):
        raise PreflightError(
            "Mitsuba variant is not CUDA-backed "
            f"(mi.variant()={variant!r}); refusing CPU/LLVM fallback"
        )
    return {
        "variant": variant,
        "cuda_backend": True,
        "mitsuba_version": mi.__version__,
        "drjit_version": getattr(drjit, "__version__", None),
    }


def _array_numpy(value: Any) -> Any:
    import numpy as np

    return np.asarray(value)


def _ray_tracing_checks() -> dict[str, Any]:
    import numpy as np
    from sionna.rt import (
        PathSolver,
        PlanarArray,
        RadioMaterial,
        Receiver,
        Transmitter,
        load_scene_from_string,
    )
    from sionna.rt.rcs import ConstantRCSSensingTarget, RCSSolver

    scene = load_scene_from_string(PREFLIGHT_SCENE_XML, merge_shapes=False)
    if set(scene.objects) != {"wall-a", "wall-b"}:
        raise PreflightError(
            f"custom two-wall scene objects are {sorted(scene.objects)!r}"
        )

    material = RadioMaterial(
        name="wall-material",
        thickness=0.1,
        relative_permittivity=5.0,
        conductivity=0.02,
        scattering_coefficient=0.0,
    )
    scene.add(material)
    for obj in scene.objects.values():
        obj.radio_material = material.name

    array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.tx_array = array
    scene.rx_array = array
    scene.add(
        [
            Transmitter("tx", position=list(TX_POSITION)),
            Receiver("rx", position=list(RX_POSITION)),
        ]
    )

    paths = PathSolver()(scene, max_depth=3, seed=42)
    valid = _array_numpy(paths.valid).reshape(-1)
    interactions = _array_numpy(paths.interactions)
    if interactions.ndim != 4 or interactions.shape[1:] != paths.valid.shape:
        raise PreflightError(
            f"unexpected Paths.interactions shape {interactions.shape!r}"
        )
    specular = np.any(interactions == 1, axis=0).reshape(-1)
    if not bool(np.any(valid)):
        raise PreflightError("PathSolver returned no valid path")
    if not bool(np.any(valid & specular)):
        raise PreflightError("PathSolver returned no valid specular path")

    cir, tau = paths.cir(out_type="torch", normalize_delays=False)
    if not (cir.is_cuda and tau.is_cuda):
        raise PreflightError(
            "Paths.cir(out_type='torch') did not return CUDA tensors "
            f"(cir device {cir.device}, tau device {tau.device})"
        )
    if not bool(torch_finite(cir)):
        raise PreflightError("PathSolver CIR contains non-finite values")
    cir_absmax = float(cir.abs().max().item())
    if cir_absmax <= 0.0:
        raise PreflightError("PathSolver CIR is identically zero")

    tx = np.asarray(TX_POSITION, dtype=np.float64)
    rx = np.asarray(RX_POSITION, dtype=np.float64)
    los_distance = float(np.linalg.norm(rx - tx))
    expected_los_delay = los_distance / SPEED_OF_LIGHT
    measured_los_delay = float(tau.min().item())

    paths_result = {
        "scene": "custom two-wall scene from XML (wall-a y=-2, wall-b x=8)",
        "max_depth": 3,
        "seed": 42,
        "num_valid": int(valid.sum()),
        "num_valid_specular_paths": int((valid & specular).sum()),
        "num_paths": int(valid.size),
        "tau_s": [float(v) for v in tau.detach().cpu().reshape(-1).tolist()],
        "interactions": interactions.tolist(),
        "cir_shape": list(cir.shape),
        "cir_device": str(cir.device),
        "cir_dtype": str(cir.dtype),
        "cir_abs_max": cir_absmax,
        "los_distance_m": los_distance,
        "los_expected_delay_s": expected_los_delay,
        "los_measured_min_delay_s": measured_los_delay,
        "los_relative_delay_error": abs(
            measured_los_delay - expected_los_delay
        )
        / expected_los_delay,
    }

    target = ConstantRCSSensingTarget(
        name="target",
        sigma=1.0,
        length=0.2,
        width=0.2,
        height=0.2,
        position=list(TARGET_POSITION),
    )
    scene.add(target)
    if "target" not in scene.sensing_targets:
        raise PreflightError("ConstantRCSSensingTarget was not added to the scene")

    rcs_paths = RCSSolver()(scene, max_depth=1, seed=42)
    rcs_valid = _array_numpy(rcs_paths.valid).reshape(-1)
    if not bool(np.any(rcs_valid)):
        raise PreflightError("RCSSolver returned no valid target path")

    rcs_cir, rcs_tau = rcs_paths.cir(out_type="torch", normalize_delays=False)
    if not (rcs_cir.is_cuda and rcs_tau.is_cuda):
        raise PreflightError(
            "RCSSolver CIR did not return CUDA tensors "
            f"(cir device {rcs_cir.device}, tau device {rcs_tau.device})"
        )
    if not bool(torch_finite(rcs_cir)):
        raise PreflightError("RCSSolver CIR contains non-finite values")
    rcs_absmax = float(rcs_cir.abs().max().item())
    if rcs_absmax <= 0.0:
        raise PreflightError("RCSSolver target return is identically zero")

    target_vec = np.asarray(TARGET_POSITION, dtype=np.float64)
    two_leg_distance = float(np.linalg.norm(target_vec - tx) + np.linalg.norm(rx - target_vec))
    expected_target_delay = two_leg_distance / SPEED_OF_LIGHT
    measured_target_delay = float(rcs_tau.min().item())

    return {
        "pathsolver": paths_result,
        "rcssolver": {
            "target": {
                "class": "ConstantRCSSensingTarget",
                "name": target.name,
                "sigma_m2": float(target.sigma[0]),
                "length_m": 0.2,
                "width_m": 0.2,
                "height_m": 0.2,
                "position": list(TARGET_POSITION),
            },
            "max_depth": 1,
            "seed": 42,
            "num_valid": int(rcs_valid.sum()),
            "tau_s": [
                float(v)
                for v in rcs_tau.detach().cpu().reshape(-1).tolist()
            ],
            "cir_shape": list(rcs_cir.shape),
            "cir_device": str(rcs_cir.device),
            "cir_dtype": str(rcs_cir.dtype),
            "cir_abs_max": rcs_absmax,
            "two_leg_distance_m": two_leg_distance,
            "expected_delay_s": expected_target_delay,
            "measured_min_delay_s": measured_target_delay,
            "relative_delay_error": abs(measured_target_delay - expected_target_delay)
            / expected_target_delay,
        },
    }


def torch_finite(tensor: Any) -> bool:
    import torch

    if torch.is_complex(tensor):
        return bool(torch.isfinite(torch.view_as_real(tensor)).all().item())
    return bool(torch.isfinite(tensor).all().item())


def _container_image(output: Path) -> dict[str, Any]:
    info_path = output / "image_info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as handle:
            info = json.load(handle)
        info["source"] = str(info_path)
        info["visible_in_container"] = False
        info["note"] = (
            "Collected on the host by scripts/record_image.sh; a running "
            "container cannot inspect its own image ID/digest."
        )
        return info
    return {
        "visible_in_container": False,
        "note": (
            "image_info.json not found; run scripts/record_image.sh on the "
            "host to record the image ID/digest."
        ),
    }


def run_preflight(require_h100: bool, output_path: str | Path) -> int:
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    environment: dict[str, Any] = {
        "status": "RUNNING",
        "timestamp_utc": _utc_now(),
        "require_h100": require_h100,
        "framework": "PyTorch",
    }
    exit_code = 0

    log("ISAC preflight starting")
    log(f"timestamp_utc: {environment['timestamp_utc']}")
    log(f"require_h100: {require_h100}")

    try:
        environment["python"] = _python_info()
        log(
            "python: "
            f"{environment['python']['python_version']} "
            f"({environment['python']['platform']})"
        )

        environment["container_image"] = _container_image(output)
        image_id = environment["container_image"].get("image_id")
        log(f"container image id (host-recorded): {image_id!r}")

        environment["mitsuba"] = _require_cuda_variant()
        log(
            "mitsuba variant: "
            f"{environment['mitsuba']['variant']} "
            f"(mitsuba {environment['mitsuba']['mitsuba_version']}, "
            f"drjit {environment['mitsuba']['drjit_version']})"
        )

        environment["gpu"] = _torch_checks(require_h100)
        log(
            "GPU: "
            f"{environment['gpu']['device']['name']} "
            f"uuid={environment['gpu']['device']['uuid']} "
            f"cc={environment['gpu']['device']['compute_capability']['major']}."
            f"{environment['gpu']['device']['compute_capability']['minor']}"
        )
        log(
            "torch: "
            f"{environment['gpu']['torch_version']} "
            f"cuda {environment['gpu']['cuda_version']} "
            f"device_count={environment['gpu']['device_count']} "
            f"selected={environment['gpu']['selected_device']}"
        )
        log(
            "cuda matmul: "
            f"{environment['gpu']['cuda_matmul']['seconds']:.6f} s"
        )
        log(
            "cuda fft: "
            f"{environment['gpu']['cuda_fft']['seconds']:.6f} s"
        )
        log(
            "pytorch peak: "
            f"allocated={environment['gpu']['memory']['pytorch_allocator']['peak_allocated_bytes']} "
            f"reserved={environment['gpu']['memory']['pytorch_allocator']['peak_reserved_bytes']}"
        )
        log(
            "nvml device memory: "
            f"total={environment['gpu']['memory']['nvml_device']['total_memory_bytes']} "
            f"used={environment['gpu']['memory']['nvml_device']['used_memory_bytes']} "
            f"free={environment['gpu']['memory']['nvml_device']['free_memory_bytes']}"
        )
        log(
            "nvml driver: "
            f"{environment['gpu']['memory']['nvml_device']['driver_version']}"
        )

        environment["ray_tracing"] = _ray_tracing_checks()
        paths_result = environment["ray_tracing"]["pathsolver"]
        log(
            "PathSolver: "
            f"valid={paths_result['num_valid']}/{paths_result['num_paths']} "
            f"specular_valid={paths_result['num_valid_specular_paths']} "
            f"cir={tuple(paths_result['cir_shape'])} "
            f"device={paths_result['cir_device']} "
            f"|a|max={paths_result['cir_abs_max']:.6e} "
            f"LoS delay rel.err={paths_result['los_relative_delay_error']:.3e}"
        )
        rcs_result = environment["ray_tracing"]["rcssolver"]
        log(
            "RCSSolver: "
            f"valid={rcs_result['num_valid']} "
            f"cir={tuple(rcs_result['cir_shape'])} "
            f"device={rcs_result['cir_device']} "
            f"|a|max={rcs_result['cir_abs_max']:.6e} "
            f"target delay rel.err={rcs_result['relative_delay_error']:.3e}"
        )

        environment["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        environment["status"] = "FAIL"
        environment["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        log(f"PREFLIGHT FAILED: {type(exc).__name__}: {exc}")
        log(environment["error"]["traceback"])
        exit_code = 1

    environment["finished_utc"] = _utc_now()
    try:
        environment["gpu_memory_after"] = _nvml_device_snapshot()
    except Exception as exc:  # noqa: BLE001
        environment["gpu_memory_after_error"] = f"{type(exc).__name__}: {exc}"

    environment_path = output / "environment.json"
    with environment_path.open("w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2, sort_keys=True)
        handle.write("\n")

    log(f"status: {environment['status']}")
    log(f"wrote {environment_path}")
    (output / "preflight.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return exit_code
