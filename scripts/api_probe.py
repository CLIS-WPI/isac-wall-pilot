"""Phase 1 Sionna API probe and analytical sanity checks.

Run inside the GPU container from the repository root:

    ISAC_DEV=1 scripts/drun.sh python scripts/api_probe.py

This script is read-only with respect to the package: it writes one JSON
evidence file under ``results/phase1/``. It hard-fails if the Mitsuba variant
is not CUDA-backed or if any analytical check fails.
"""

from __future__ import annotations

import inspect
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import mitsuba as mi
import sionna.rt as rt
from sionna.phy.channel import (
    cir_to_ofdm_channel,
    ofdm_to_delay_doppler_channel,
    ofdm_to_time_channel,
    subcarrier_frequencies,
    time_to_doppler_channel,
)
from sionna.phy.isac import angular_delay_doppler_spectrum, steering_vectors
from sionna.rt import (
    PathSolver,
    PlanarArray,
    RadioMaterial,
    Receiver,
    Transmitter,
    load_scene,
    load_scene_from_string,
)
from sionna.rt.constants import InteractionType
from sionna.rt.rcs import ConstantRCSSensingTarget, RCSSolver
from sionna.rt.utils.electromagnetics import (
    complex_relative_permittivity,
    fresnel_reflection_coefficients_simplified,
    itu_coefficients_single_layer_slab,
)

SPEED_OF_LIGHT = 299_792_458.0
FC = 3.5e9
FFT_SIZE = 2048
SCS = 60e3
SYMBOLS = 64
TX_POS = (-4.0, 2.0, 2.0)
RX_POS = (4.0, 2.0, 2.0)
TARGET_POS = (0.0, 5.0, 2.0)

SCENE_XML = """<scene version="3.0.0">
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


class ProbeFailure(RuntimeError):
    """Raised when a measured API fact or analytical check fails."""


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    try:
        return float(value[0])
    except (TypeError, IndexError):
        return float(value)


def _vec3(value: Any) -> list[float]:
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return [float(value.x[0]), float(value.y[0]), float(value.z[0])]
    array = np.asarray(value).reshape(-1)
    return [float(v) for v in array[:3]]


def _complex(value: Any) -> complex:
    if isinstance(value, torch.Tensor):
        item = value.detach().cpu().reshape(-1)[0].item()
        return complex(item)
    return complex(value)


def _cplx(value: Any) -> complex:
    """Convert a Dr.Jit Complex2f scalar to a Python complex."""
    return complex(float(value.real[0]), float(value.imag[0]))


def _cuda_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if not value.is_cuda:
            raise ProbeFailure(f"tensor is not on CUDA: {value.device}")
        return value
    tensor = value.torch()
    if not tensor.is_cuda:
        raise ProbeFailure(f"converted tensor is not on CUDA: {tensor.device}")
    return tensor


def _complex_tensor(real: Any, imag: Any) -> torch.Tensor:
    return torch.complex(_cuda_tensor(real), _cuda_tensor(imag))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeFailure(message)


def _max_abs(value: torch.Tensor) -> float:
    return float(value.abs().max().item())


def _assert_close(name: str, measured: float, expected: float, tol: float) -> dict[str, Any]:
    error = abs(measured - expected)
    _require(error <= tol, f"{name}: measured {measured!r}, expected {expected!r}, error {error!r} > {tol!r}")
    return {"measured": measured, "expected": expected, "abs_error": error, "tolerance": tol}


def _assert_complex_close(
    name: str, measured: complex, expected: complex, tol: float
) -> dict[str, Any]:
    error = abs(measured - expected)
    _require(error <= tol, f"{name}: measured {measured!r}, expected {expected!r}, error {error!r} > {tol!r}")
    return {
        "measured": [measured.real, measured.imag],
        "expected": [expected.real, expected.imag],
        "abs_error": error,
        "tolerance": tol,
    }


def _make_scene(
    *,
    ntx: int = 1,
    nrx: int = 1,
    tx_pos: tuple[float, float, float] = TX_POS,
    rx_pos: tuple[float, float, float] = RX_POS,
    walls: bool = True,
    wall_material: bool = True,
) -> tuple[Any, PlanarArray, PlanarArray]:
    if walls:
        scene = load_scene_from_string(SCENE_XML, merge_shapes=False)
    else:
        scene = load_scene()
    if wall_material and walls:
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
    tx_array = PlanarArray(
        num_rows=1,
        num_cols=ntx,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    rx_array = PlanarArray(
        num_rows=1,
        num_cols=nrx,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.tx_array = tx_array
    scene.rx_array = rx_array
    scene.add(
        [
            Transmitter("tx", position=list(tx_pos)),
            Receiver("rx", position=list(rx_pos)),
        ]
    )
    return scene, tx_array, rx_array


def _add_target(
    scene: Any,
    position: tuple[float, float, float] = TARGET_POS,
    sigma: float = 1.0,
) -> ConstantRCSSensingTarget:
    target = ConstantRCSSensingTarget(
        name="target",
        sigma=sigma,
        length=0.2,
        width=0.2,
        height=0.2,
        position=list(position),
    )
    scene.add(target)
    return target


def probe_paths_and_angles() -> dict[str, Any]:
    scene, _, _ = _make_scene()
    paths = PathSolver()(scene, max_depth=1, seed=42)
    tau = _cuda_tensor(paths.tau)
    valid = _cuda_tensor(paths.valid)
    a_real = _cuda_tensor(paths.a[0])
    a_imag = _cuda_tensor(paths.a[1])
    interactions = _cuda_tensor(paths.interactions)
    vertices = _cuda_tensor(paths.vertices)
    theta_t = _cuda_tensor(paths.theta_t)
    phi_t = _cuda_tensor(paths.phi_t)
    theta_r = _cuda_tensor(paths.theta_r)
    phi_r = _cuda_tensor(paths.phi_r)

    los_index = int(torch.nonzero(torch.all(interactions == 0, dim=0).reshape(-1))[0].item())
    specular_indices = [
        int(i)
        for i in torch.nonzero(
            (interactions[0].reshape(-1) == int(InteractionType.SPECULAR))
            & valid.reshape(-1)
        ).reshape(-1)
    ]
    _require(len(specular_indices) == 2, f"expected two specular paths, got {specular_indices}")

    tx = np.asarray(TX_POS, dtype=np.float64)
    rx = np.asarray(RX_POS, dtype=np.float64)
    los_distance = float(np.linalg.norm(rx - tx))
    los_tau = float(tau.reshape(-1)[los_index].item())
    los_expected_tau = los_distance / SPEED_OF_LIGHT

    _assert_close("LoS theta_t", float(theta_t.reshape(-1)[los_index]), np.pi / 2, 1e-5)
    _assert_close("LoS phi_t", float(phi_t.reshape(-1)[los_index]), 0.0, 1e-5)
    _assert_close("LoS theta_r", float(theta_r.reshape(-1)[los_index]), np.pi / 2, 1e-5)
    _assert_close("LoS phi_r (points RX -> TX)", float(phi_r.reshape(-1)[los_index]), np.pi, 1e-5)
    _assert_close("LoS delay", los_tau, los_expected_tau, 1e-12)

    reflection_geometry = []
    for index in specular_indices:
        vertex = [float(v) for v in vertices.reshape(-1, 3)[index].tolist()]
        leg1 = float(np.linalg.norm(np.asarray(vertex) - tx))
        leg2 = float(np.linalg.norm(rx - np.asarray(vertex)))
        total = leg1 + leg2
        measured_tau = float(tau.reshape(-1)[index].item())
        _assert_close(f"reflection delay vertex {index}", measured_tau, total / SPEED_OF_LIGHT, 2e-12)
        reflection_geometry.append(
            {
                "path_index": index,
                "vertex_m": vertex,
                "leg1_m": leg1,
                "leg2_m": leg2,
                "total_m": total,
                "tau_s": measured_tau,
            }
        )

    return {
        "signatures": {
            "PathSolver": str(inspect.signature(PathSolver)),
            "PathSolver.__call__": str(inspect.signature(PathSolver.__call__)),
        },
        "shapes": {
            "tau": list(tau.shape),
            "valid": list(valid.shape),
            "a_real": list(a_real.shape),
            "a_imag": list(a_imag.shape),
            "interactions": list(interactions.shape),
            "vertices": list(vertices.shape),
            "theta_t": list(theta_t.shape),
            "phi_t": list(phi_t.shape),
            "theta_r": list(theta_r.shape),
            "phi_r": list(phi_r.shape),
        },
        "dtypes": {
            "tau": str(tau.dtype),
            "valid": str(valid.dtype),
            "a_real": str(a_real.dtype),
            "interactions": str(interactions.dtype),
        },
        "devices": {
            "tau": str(tau.device),
            "a_real": str(a_real.device),
        },
        "interaction_constants": {
            "NONE": int(InteractionType.NONE),
            "SPECULAR": int(InteractionType.SPECULAR),
            "DIFFUSE": int(InteractionType.DIFFUSE),
            "REFRACTION": int(InteractionType.REFRACTION),
            "DIFFRACTION": int(InteractionType.DIFFRACTION),
            "SENSING": int(InteractionType.SENSING),
        },
        "los": {
            "path_index": los_index,
            "tau_s": los_tau,
            "expected_tau_s": los_expected_tau,
            "a_real": float(a_real.reshape(-1)[los_index].item()),
            "a_imag": float(a_imag.reshape(-1)[los_index].item()),
            "theta_t_rad": float(theta_t.reshape(-1)[los_index].item()),
            "phi_t_rad": float(phi_t.reshape(-1)[los_index].item()),
            "theta_r_rad": float(theta_r.reshape(-1)[los_index].item()),
            "phi_r_rad": float(phi_r.reshape(-1)[los_index].item()),
        },
        "reflections": reflection_geometry,
        "checks": {
            "los_delay": _assert_close("LoS delay", los_tau, los_expected_tau, 1e-12),
            "angles": {
                "theta_t": np.pi / 2,
                "phi_t": 0.0,
                "theta_r": np.pi / 2,
                "phi_r": np.pi,
                "tolerance_rad": 1e-5,
            },
        },
    }


def probe_cir() -> dict[str, Any]:
    scene, _, _ = _make_scene()
    paths = PathSolver()(scene, max_depth=1, seed=42)
    signature = str(inspect.signature(paths.cir))
    parameters = inspect.signature(paths.cir).parameters
    _require("normalize_delays" in parameters, "Paths.cir lost normalize_delays")
    _require("normalize" not in parameters, "Paths.cir unexpectedly has an amplitude normalize argument")

    a_abs, tau_abs = paths.cir(out_type="torch", normalize_delays=False)
    a_norm, tau_norm = paths.cir(out_type="torch")
    a_abs = _cuda_tensor(a_abs)
    tau_abs = _cuda_tensor(tau_abs)
    a_norm = _cuda_tensor(a_norm)
    tau_norm = _cuda_tensor(tau_norm)
    _require(_max_abs(a_abs) > 0.0, "absolute-delay CIR is zero")
    _require(_max_abs(a_norm) > 0.0, "normalized-delay CIR is zero")
    _assert_close("normalized minimum delay", float(tau_norm.min().item()), 0.0, 1e-12)
    _require(float(tau_abs.min().item()) > 0.0, "absolute delays start at zero")
    _require(float(tau_abs.min().item()) != float(tau_norm.min().item()), "delay normalization had no effect")

    return {
        "signature": signature,
        "parameters": list(parameters),
        "normalize_delays_default": parameters["normalize_delays"].default,
        "amplitude_normalize_present": "normalize" in parameters,
        "absolute": {
            "a_shape": list(a_abs.shape),
            "a_dtype": str(a_abs.dtype),
            "a_device": str(a_abs.device),
            "tau_shape": list(tau_abs.shape),
            "tau_dtype": str(tau_abs.dtype),
            "tau_device": str(tau_abs.device),
            "tau_min_s": float(tau_abs.min().item()),
            "tau_max_s": float(tau_abs.max().item()),
            "a_abs_max": _max_abs(a_abs),
        },
        "default_normalized": {
            "tau_min_s": float(tau_norm.min().item()),
            "tau_max_s": float(tau_norm.max().item()),
            "tau_values_s": [float(v) for v in tau_norm.reshape(-1).tolist()],
        },
        "doc_note": "Paths.cir returns baseband a_i e^{-j2pi f_c tau_i}; only cfr/taps expose amplitude normalize.",
    }


def check_los_amplitude_and_carrier_phase() -> dict[str, Any]:
    scene, _, _ = _make_scene()
    paths = PathSolver()(scene, max_depth=0, seed=42)
    a_cir, tau = paths.cir(out_type="torch", normalize_delays=False)
    a_cir = _cuda_tensor(a_cir)
    tau = _cuda_tensor(tau)
    a_pass = _complex_tensor(paths.a[0], paths.a[1])
    wavelength = float(scene.wavelength[0])
    distance = float(np.linalg.norm(np.asarray(RX_POS) - np.asarray(TX_POS)))
    expected_abs = wavelength / (4.0 * np.pi * distance)
    measured_abs = abs(_complex(a_pass.reshape(-1)[0]))
    amplitude = _assert_close("LoS passband |a| = lambda/(4*pi*d)", measured_abs, expected_abs, 1e-7)

    delay = _assert_close("LoS tau = d/c", float(tau.reshape(-1)[0]), distance / SPEED_OF_LIGHT, 1e-12)
    expected_cir = _complex(a_pass.reshape(-1)[0]) * np.exp(
        -2j * np.pi * FC * float(tau.reshape(-1)[0])
    )
    measured_cir = _complex(a_cir.reshape(-1)[0])
    cir_phase = _assert_complex_close(
        "Paths.cir includes e^{-j2pi f_c tau}",
        measured_cir,
        expected_cir,
        1e-6,
    )
    return {
        "wavelength_m": wavelength,
        "distance_m": distance,
        "passband_a": [_complex(a_pass.reshape(-1)[0]).real, _complex(a_pass.reshape(-1)[0]).imag],
        "expected_passband_abs": expected_abs,
        "measured_passband_abs": measured_abs,
        "tau_s": float(tau.reshape(-1)[0]),
        "expected_tau_s": distance / SPEED_OF_LIGHT,
        "cir": [measured_cir.real, measured_cir.imag],
        "expected_cir": [expected_cir.real, expected_cir.imag],
        "checks": {"amplitude": amplitude, "delay": delay, "carrier_phase": cir_phase},
    }


def check_cfr_and_normalization() -> dict[str, Any]:
    scene, _, _ = _make_scene()
    paths = PathSolver()(scene, max_depth=0, seed=42)
    a_cir, tau = paths.cir(out_type="torch", normalize_delays=False)
    a_cir = _cuda_tensor(a_cir)
    tau = _cuda_tensor(tau)
    freqs = subcarrier_frequencies(FFT_SIZE, SCS, device="cuda")
    _require(freqs.is_cuda, "subcarrier_frequencies did not return a CUDA tensor")

    h_cir = cir_to_ofdm_channel(freqs, a_cir.unsqueeze(0), tau.unsqueeze(0), normalize=False)
    h_cfr = paths.cfr(
        freqs,
        normalize_delays=False,
        normalize=False,
        out_type="torch",
    )
    h_cfr = _cuda_tensor(h_cfr)
    _require(h_cir.is_cuda and h_cfr.is_cuda, "CFR tensors are not on CUDA")
    max_difference = float((h_cir.reshape(-1) - h_cfr.reshape(-1)).abs().max().item())
    shape_check = {
        "cir_to_ofdm_channel": list(h_cir.shape),
        "Paths.cfr": list(h_cfr.shape),
        "note": "Paths.cfr omits the singleton num_paths axis.",
    }
    _require(max_difference < 1e-6, f"cir_to_ofdm_channel and Paths.cfr differ by {max_difference}")

    a0 = a_cir.reshape(-1)[0]
    tau0 = float(tau.reshape(-1)[0])
    ratio = h_cir.reshape(-1) / a0
    expected = torch.exp(-2j * np.pi * freqs * tau0)
    residual = torch.angle(ratio * torch.exp(2j * np.pi * freqs * tau0))
    phase_error = _max_abs(residual)
    _require(phase_error < 1e-4, f"subcarrier phase residual {phase_error} exceeds 1e-4")

    h_abs = cir_to_ofdm_channel(
        freqs + FC,
        a_cir.unsqueeze(0),
        tau.unsqueeze(0),
        normalize=False,
    )
    absolute_difference = float((h_abs.reshape(-1) - h_cir.reshape(-1)).abs().max().item())
    _require(absolute_difference > 1e-4, "passing absolute frequencies did not add a carrier phase")

    h_norm = cir_to_ofdm_channel(
        freqs,
        a_cir.unsqueeze(0),
        tau.unsqueeze(0),
        normalize=True,
    )
    raw_power = float(h_cir.abs().square().mean().item())
    normalized_power = float(h_norm.abs().square().mean().item())
    _assert_close("normalize=True mean power", normalized_power, 1.0, 1e-5)

    return {
        "signatures": {
            "Paths.cfr": str(inspect.signature(paths.cfr)),
            "subcarrier_frequencies": str(inspect.signature(subcarrier_frequencies)),
            "cir_to_ofdm_channel": str(inspect.signature(cir_to_ofdm_channel)),
        },
        "frequencies": {
            "shape": list(freqs.shape),
            "dtype": str(freqs.dtype),
            "device": str(freqs.device),
            "first_three_hz": [float(v) for v in freqs[:3].tolist()],
            "center_pair_hz": [float(v) for v in freqs[FFT_SIZE // 2 - 1 : FFT_SIZE // 2 + 1].tolist()],
            "last_three_hz": [float(v) for v in freqs[-3:].tolist()],
        },
        "shapes": shape_check,
        "max_abs_difference": max_difference,
        "phase_residual_rad_max": phase_error,
        "absolute_frequency_difference": absolute_difference,
        "normalization": {
            "raw_mean_power": raw_power,
            "normalized_mean_power": normalized_power,
        },
        "checks": {
            "cir_to_ofdm_channel_equals_Paths_cfr": max_difference,
            "subcarrier_phase_uses_baseband_frequency": phase_error,
            "absolute_frequency_is_double_counting": absolute_difference,
            "normalize_true_unit_power": normalized_power,
        },
    }


def check_reflection_and_material() -> dict[str, Any]:
    scene, _, _ = _make_scene()
    scene.frequency = FC
    paths = PathSolver()(scene, max_depth=1, seed=42)
    a_rt = _complex_tensor(paths.a[0], paths.a[1]).reshape(-1)
    tau = _cuda_tensor(paths.tau).reshape(-1)
    valid = _cuda_tensor(paths.valid).reshape(-1)
    interactions = _cuda_tensor(paths.interactions)
    vertices = _cuda_tensor(paths.vertices).reshape(-1, 3)
    wavelength = float(scene.wavelength[0])
    tx = np.asarray(TX_POS, dtype=np.float64)
    rx = np.asarray(RX_POS, dtype=np.float64)

    measured = []
    for index in torch.nonzero(
        (interactions[0].reshape(-1) == int(InteractionType.SPECULAR)) & valid
    ).reshape(-1).tolist():
        vertex = np.asarray(vertices[index].tolist(), dtype=np.float64)
        leg1 = float(np.linalg.norm(vertex - tx))
        leg2 = float(np.linalg.norm(rx - vertex))
        length = leg1 + leg2
        measured_a = _complex(a_rt[index])
        ratio = measured_a / (wavelength / (4.0 * np.pi * length))
        normal = np.array([0.0, 1.0, 0.0]) if abs(vertex[1] + 2.0) < 1e-3 else np.array([1.0, 0.0, 0.0])
        cos_theta = abs(float(np.dot(tx - vertex, normal))) / leg1
        eta = complex_relative_permittivity(
            mi.Float(5.0), mi.Float(0.02), mi.Float(2.0 * np.pi * FC)
        )
        r_te, r_tm, _, _ = itu_coefficients_single_layer_slab(
            mi.Float(cos_theta), eta, mi.Float(0.1), mi.Float(wavelength)
        )
        expected_ratio = _cplx(r_te)
        check = _assert_complex_close(
            f"reflection coefficient path {int(index)}",
            ratio,
            expected_ratio,
            1e-4,
        )
        measured.append(
            {
                "path_index": int(index),
                "vertex_m": [float(v) for v in vertex.tolist()],
                "leg1_m": leg1,
                "leg2_m": leg2,
                "length_m": length,
                "cos_theta": cos_theta,
                "tau_s": float(tau[index].item()),
                "a": [measured_a.real, measured_a.imag],
                "free_space_amplitude": wavelength / (4.0 * np.pi * length),
                "ratio": [ratio.real, ratio.imag],
                "slab_r_te": [expected_ratio.real, expected_ratio.imag],
                "slab_r_tm": [_cplx(r_tm).real, _cplx(r_tm).imag],
                "check": check,
            }
        )

    material = scene.radio_materials["wall-material"]
    material_facts = {
        "class": type(material).__name__,
        "thickness_m": _float(material.thickness),
        "relative_permittivity": _float(material.relative_permittivity),
        "conductivity_s_per_m": _float(material.conductivity),
        "scattering_coefficient": _float(material.scattering_coefficient),
        "xpd_coefficient": _float(material.xpd_coefficient),
        "frequency_update_callback": repr(material.frequency_update_callback),
    }
    eta = complex_relative_permittivity(
        mi.Float(5.0), mi.Float(0.02), mi.Float(2.0 * np.pi * FC)
    )
    r_te_simple, r_tm_simple = fresnel_reflection_coefficients_simplified(
        mi.Float(4.0 / np.sqrt(32.0)), eta
    )
    fresnel_simple = {
        "cos_theta": 4.0 / np.sqrt(32.0),
        "r_te": [_cplx(r_te_simple).real, _cplx(r_te_simple).imag],
        "r_tm": [_cplx(r_tm_simple).real, _cplx(r_tm_simple).imag],
        "note": "single vacuum-material interface; RT uses the finite-thickness slab function",
    }

    scene, _, _ = _make_scene()
    scene.frequency = 3.5e9
    p35 = PathSolver()(scene, max_depth=0, seed=42)
    a35 = _complex(_complex_tensor(p35.a[0], p35.a[1]).reshape(-1)[0])
    tau35 = float(_cuda_tensor(p35.tau).reshape(-1)[0])
    wl35 = float(scene.wavelength[0])
    scene.frequency = 7e9
    p7 = PathSolver()(scene, max_depth=0, seed=42)
    a7 = _complex(_complex_tensor(p7.a[0], p7.a[1]).reshape(-1)[0])
    tau7 = float(_cuda_tensor(p7.tau).reshape(-1)[0])
    wl7 = float(scene.wavelength[0])
    _assert_close("wavelength ratio after frequency change", wl35 / wl7, 2.0, 1e-6)
    _assert_close("LoS amplitude ratio after frequency change", abs(a35) / abs(a7), 2.0, 1e-6)
    _assert_close("LoS delay invariance after frequency change", tau7, tau35, 1e-12)

    return {
        "reflections": measured,
        "material": material_facts,
        "fresnel_simplified": fresnel_simple,
        "frequency": {
            "lambda_3p5_ghz_m": wl35,
            "lambda_7_ghz_m": wl7,
            "a_3p5_ghz": [a35.real, a35.imag],
            "a_7_ghz": [a7.real, a7.imag],
            "a_ratio_abs": abs(a35) / abs(a7),
            "tau_3p5_ghz_s": tau35,
            "tau_7_ghz_s": tau7,
            "checks": {
                "lambda_ratio": wl35 / wl7,
                "amplitude_ratio": abs(a35) / abs(a7),
                "delay_difference_s": abs(tau7 - tau35),
            },
        },
    }


def probe_rcs() -> dict[str, Any]:
    scene, _, _ = _make_scene()
    target = _add_target(scene)
    signatures = {
        "RCSSolver": str(inspect.signature(RCSSolver)),
        "RCSSolver.__call__": str(inspect.signature(RCSSolver.__call__)),
        "ConstantRCSSensingTarget": str(inspect.signature(ConstantRCSSensingTarget)),
    }
    depth_counts = {}
    for depth in (1, 2, 3):
        paths = RCSSolver()(scene, max_depth=depth, seed=42)
        valid = _cuda_tensor(paths.valid).reshape(-1)
        interactions = _cuda_tensor(paths.interactions)
        tau = _cuda_tensor(paths.tau).reshape(-1)
        sensing = torch.any(
            interactions == int(InteractionType.SENSING), dim=0
        ).reshape(-1)
        depth_counts[depth] = {
            "valid": int(valid.sum().item()),
            "total_slots": int(valid.numel()),
            "num_interactions_rows": int(interactions.shape[0]),
            "sensing_valid": int((valid & sensing).sum().item()),
            "tau_s": [float(v) for v in tau[valid].tolist()],
            "interactions": interactions.to(torch.int64).reshape(interactions.shape[0], -1)[:, valid].T.tolist(),
        }
    _require(depth_counts[1]["valid"] == 1, f"RCSSolver max_depth=1 valid {depth_counts[1]['valid']} != 1")
    _require(depth_counts[2]["valid"] == 5, f"RCSSolver max_depth=2 valid {depth_counts[2]['valid']} != 5")
    _require(depth_counts[3]["valid"] == 11, f"RCSSolver max_depth=3 valid {depth_counts[3]['valid']} != 11")

    tx = np.asarray(TX_POS, dtype=np.float64)
    rx = np.asarray(RX_POS, dtype=np.float64)
    target_vec = np.asarray(TARGET_POS, dtype=np.float64)
    two_leg = float(np.linalg.norm(target_vec - tx) + np.linalg.norm(rx - target_vec))
    los_tau = depth_counts[1]["tau_s"][0]
    _assert_close("RCSSolver two-leg delay", los_tau, two_leg / SPEED_OF_LIGHT, 1e-12)

    occlusion_scene, _, _ = _make_scene(tx_pos=(-4.0, 0.0, 2.0), rx_pos=(4.0, 0.0, 2.0))
    before = PathSolver()(occlusion_scene, max_depth=0, seed=42)
    before_valid = int(_cuda_tensor(before.valid).sum().item())
    _require(before_valid == 1, f"occlusion control expected one LoS path, got {before_valid}")
    target_block = _add_target(occlusion_scene, position=(0.0, 0.0, 2.0))
    after = PathSolver()(occlusion_scene, max_depth=0, seed=42)
    after_valid = int(_cuda_tensor(after.valid).sum().item())
    _require(after_valid == 0, f"target did not block PathSolver LoS (valid={after_valid})")

    scene_sigma, _, _ = _make_scene()
    _add_target(scene_sigma, sigma=1.0)
    p1 = RCSSolver()(scene_sigma, max_depth=1, seed=42)
    a1 = _complex(_complex_tensor(p1.a[0], p1.a[1]).reshape(-1)[0])
    scene_sigma4, _, _ = _make_scene()
    _add_target(scene_sigma4, sigma=4.0)
    p4 = RCSSolver()(scene_sigma4, max_depth=1, seed=42)
    a4 = _complex(_complex_tensor(p4.a[0], p4.a[1]).reshape(-1)[0])
    ratio = abs(a4) / abs(a1)
    _assert_close("RCS amplitude sqrt(sigma) scaling", ratio, 2.0, 1e-5)

    empty_scene, _, _ = _make_scene(walls=False, wall_material=False)
    try:
        RCSSolver()(empty_scene, max_depth=1, seed=42)
    except ValueError as exc:
        empty_error = f"{type(exc).__name__}: {exc}"
    else:
        raise ProbeFailure("RCSSolver without a sensing target did not raise ValueError")

    return {
        "signatures": signatures,
        "target": {
            "class": type(target).__name__,
            "name": target.name,
            "sigma_m2": _float(target.sigma),
            "position_m": _vec3(target.position),
            "radio_material": type(target.radio_material).__name__,
        },
        "depth_counts": depth_counts,
        "target_two_leg_distance_m": two_leg,
        "target_delay_s": los_tau,
        "occlusion": {
            "path_solver_before_target_valid": before_valid,
            "path_solver_after_target_valid": after_valid,
            "target_radio_material": type(target_block.radio_material).__name__,
        },
        "sigma_scaling": {
            "a_sigma_1": [a1.real, a1.imag],
            "a_sigma_4": [a4.real, a4.imag],
            "abs_ratio": ratio,
        },
        "empty_solver_error": empty_error,
        "checks": {
            "depth_1_paths": depth_counts[1]["valid"],
            "depth_2_paths": depth_counts[2]["valid"],
            "depth_3_paths": depth_counts[3]["valid"],
            "sigma_ratio": ratio,
        },
    }


def probe_synthetic_array() -> dict[str, Any]:
    scene, tx_array, rx_array = _make_scene(
        ntx=16, nrx=16, tx_pos=(-4.0, 2.0, 2.0), rx_pos=(4.0, 4.0, 2.0), walls=False, wall_material=False
    )
    p_syn = PathSolver()(scene, max_depth=0, synthetic_array=True, seed=42)
    p_elem = PathSolver()(scene, max_depth=0, synthetic_array=False, seed=42)
    tau_syn = _cuda_tensor(p_syn.tau)
    a_syn = _complex_tensor(p_syn.a[0], p_syn.a[1])
    valid_syn = _cuda_tensor(p_syn.valid)
    tau_elem = _cuda_tensor(p_elem.tau)
    a_elem = _complex_tensor(p_elem.a[0], p_elem.a[1])
    valid_elem = _cuda_tensor(p_elem.valid)
    _require(int(valid_syn.sum().item()) == 1, f"synthetic valid count {int(valid_syn.sum().item())} != 1")
    _require(int(valid_elem.sum().item()) == 256, f"per-element valid count {int(valid_elem.sum().item())} != 256")

    wavelength = float(scene.wavelength[0])
    theta_t = float(_cuda_tensor(p_syn.theta_t).reshape(-1)[0])
    phi_t = float(_cuda_tensor(p_syn.phi_t).reshape(-1)[0])
    theta_r = float(_cuda_tensor(p_syn.theta_r).reshape(-1)[0])
    phi_r = float(_cuda_tensor(p_syn.phi_r).reshape(-1)[0])
    pos_tx = torch.as_tensor(np.asarray(tx_array.positions(wavelength)).T, dtype=torch.float32, device="cuda")
    pos_rx = torch.as_tensor(np.asarray(rx_array.positions(wavelength)).T, dtype=torch.float32, device="cuda")
    sv_tx = steering_vectors(pos_tx, theta_t, phi_t, wavelength, mode="paired")
    sv_rx = steering_vectors(pos_rx, theta_r, phi_r, wavelength, mode="paired")
    rt_tx = a_syn[0, 0, 0, :, 0]
    rt_rx = a_syn[0, :, 0, 0, 0]
    tx_error = _max_abs(rt_tx / rt_tx[0] - sv_tx / sv_tx[0])
    rx_error = _max_abs(rt_rx / rt_rx[0] - sv_rx / sv_rx[0])
    _require(tx_error < 2e-5, f"TX synthetic-array phase error {tx_error} exceeds 2e-5")
    _require(rx_error < 2e-5, f"RX synthetic-array phase error {rx_error} exceeds 2e-5")

    far_scene, _, _ = _make_scene(
        ntx=16, nrx=16, tx_pos=(-4.0, 2.0, 2.0), rx_pos=(96.0, 2.0, 2.0), walls=False, wall_material=False
    )
    far_syn = PathSolver()(far_scene, max_depth=0, synthetic_array=True, seed=42)
    far_elem = PathSolver()(far_scene, max_depth=0, synthetic_array=False, seed=42)
    far_a_syn = _complex_tensor(far_syn.a[0], far_syn.a[1])
    far_a_elem = _complex_tensor(far_elem.a[0], far_elem.a[1])
    far_tau_syn = _cuda_tensor(far_syn.tau)
    far_tau_elem = _cuda_tensor(far_elem.tau)
    far_phase_error = _max_abs(
        far_a_syn[0, 0, 0, :, 0] / far_a_syn[0, 0, 0, 0, 0]
        - far_a_elem[0, 0, 0, :, 0] / far_a_elem[0, 0, 0, 0, 0]
    )
    _require(far_phase_error < 1e-3, f"far-field synthetic/per-element phase error {far_phase_error} exceeds 1e-3")
    _assert_close(
        "far-field center delay",
        float(far_tau_syn.reshape(-1)[0]),
        float(far_tau_elem[0, 0, 0, 0, 0].item()),
        1e-12,
    )

    rcs_scene, _, _ = _make_scene(ntx=4, nrx=4)
    _add_target(rcs_scene)
    r_syn = RCSSolver()(rcs_scene, max_depth=1, synthetic_array=True, seed=42)
    r_elem = RCSSolver()(rcs_scene, max_depth=1, synthetic_array=False, seed=42)
    r_syn_tau = _cuda_tensor(r_syn.tau)
    r_elem_tau = _cuda_tensor(r_elem.tau)
    r_syn_valid = int(_cuda_tensor(r_syn.valid).sum().item())
    r_elem_valid = int(_cuda_tensor(r_elem.valid).sum().item())
    _require(r_syn_valid == 1, f"RCS synthetic valid {r_syn_valid} != 1")
    _require(r_elem_valid == 16, f"RCS per-element valid {r_elem_valid} != 16")

    return {
        "path_solver": {
            "synthetic_tau_shape": list(tau_syn.shape),
            "synthetic_a_shape": list(a_syn.shape),
            "synthetic_valid": int(valid_syn.sum().item()),
            "per_element_tau_shape": list(tau_elem.shape),
            "per_element_a_shape": list(a_elem.shape),
            "per_element_valid": int(valid_elem.sum().item()),
            "synthetic_center_tau_s": float(tau_syn.reshape(-1)[0].item()),
            "per_element_first_taus_s": [float(v) for v in tau_elem.reshape(-1)[:4].tolist()],
            "tx_phase_error": tx_error,
            "rx_phase_error": rx_error,
        },
        "far_field": {
            "distance_m": 100.0,
            "synthetic_tau_s": float(far_tau_syn.reshape(-1)[0].item()),
            "per_element_center_tau_s": float(far_tau_elem[0, 0, 0, 0, 0].item()),
            "phase_error": far_phase_error,
        },
        "rcs": {
            "synthetic_tau_shape": list(r_syn_tau.shape),
            "synthetic_valid": r_syn_valid,
            "per_element_tau_shape": list(r_elem_tau.shape),
            "per_element_valid": r_elem_valid,
        },
    }


def probe_conversions() -> dict[str, Any]:
    device = "cuda"
    freqs = subcarrier_frequencies(FFT_SIZE, SCS, device=device)
    l0 = 10
    tau0 = l0 / (FFT_SIZE * SCS)
    hf = (
        torch.exp(-2j * np.pi * freqs * tau0)
        .reshape(1, 1, 1, 1, 1, 1, FFT_SIZE)
        .expand(1, 1, 1, 1, 1, SYMBOLS, FFT_SIZE)
    )
    ht = ofdm_to_time_channel(hf, l_min=0, l_max=63)
    peak = int(torch.argmax(ht.abs().reshape(-1)).item())
    _require(peak % ht.shape[-1] == l0, f"ofdm_to_time_channel peak {peak % ht.shape[-1]} != {l0}")

    ht_neg = ofdm_to_time_channel(hf, l_min=-10, l_max=63)
    peak_neg = int(torch.argmax(ht_neg.abs().reshape(-1)).item()) % ht_neg.shape[-1]
    _require(peak_neg == l0 + 10, f"negative l_min peak {peak_neg} != {l0 + 10}")

    q0 = 2
    htd = torch.zeros(1, 1, 1, 1, 1, SYMBOLS, 8, device=device, dtype=torch.complex64)
    htd[..., 3] = torch.exp(
        2j * np.pi * q0 * torch.arange(SYMBOLS, device=device) / SYMBOLS
    ).reshape(1, 1, 1, 1, 1, SYMBOLS)
    hdd = time_to_doppler_channel(htd)
    q_peak = int(torch.argmax(hdd.abs().sum(dim=-1).reshape(-1)).item())
    _require(q_peak == q0 + SYMBOLS // 2, f"time_to_doppler peak {q_peak} != {q0 + SYMBOLS // 2}")

    hf_dd = (
        torch.exp(2j * np.pi * q0 * torch.arange(SYMBOLS, device=device) / SYMBOLS)
        .reshape(1, 1, 1, 1, 1, SYMBOLS, 1)
        * torch.exp(-2j * np.pi * freqs * tau0).reshape(1, 1, 1, 1, 1, 1, FFT_SIZE)
    )
    hdd2 = ofdm_to_delay_doppler_channel(hf_dd, l_min=0, l_max=63)
    flat_peak = int(torch.argmax(hdd2.abs().reshape(-1)).item())
    q_index, l_index = divmod(flat_peak, hdd2.shape[-1])
    _require(q_index == q0 + SYMBOLS // 2, f"delay-Doppler q index {q_index} != {q0 + SYMBOLS // 2}")
    _require(l_index == l0, f"delay-Doppler lag index {l_index} != {l0}")

    return {
        "ofdm_to_time_channel": {
            "signature": str(inspect.signature(ofdm_to_time_channel)),
            "input_shape": list(hf.shape),
            "output_shape": list(ht.shape),
            "l0": l0,
            "peak_index": peak % ht.shape[-1],
            "peak_value": float(ht.abs().reshape(-1)[peak].item()),
            "negative_l_min_output_shape": list(ht_neg.shape),
            "negative_l_min_peak_index": peak_neg,
        },
        "time_to_doppler_channel": {
            "signature": str(inspect.signature(time_to_doppler_channel)),
            "input_shape": list(htd.shape),
            "output_shape": list(hdd.shape),
            "q0": q0,
            "q_peak_index": q_peak,
            "center_index": SYMBOLS // 2,
        },
        "ofdm_to_delay_doppler_channel": {
            "signature": str(inspect.signature(ofdm_to_delay_doppler_channel)),
            "input_shape": list(hf_dd.shape),
            "output_shape": list(hdd2.shape),
            "q_index": q_index,
            "l_index": l_index,
        },
        "checks": {
            "ofdm_to_time_channel_lag": peak % ht.shape[-1],
            "negative_l_min_lag": peak_neg,
            "time_to_doppler_bin": q_peak,
            "delay_doppler_bin": [q_index, l_index],
        },
    }


def probe_isac() -> dict[str, Any]:
    wavelength = SPEED_OF_LIGHT / FC
    M = 4
    pos = torch.tensor(
        [[0.0, (m - (M - 1) / 2) * 0.5 * wavelength, 0.0] for m in range(M)],
        dtype=torch.float32,
        device="cuda",
    )
    theta_r, phi_r = np.pi / 2, 0.6
    theta_t, phi_t = np.pi / 2, -0.4
    ar = steering_vectors(pos, theta_r, phi_r, wavelength, mode="paired")
    at = steering_vectors(pos, theta_t, phi_t, wavelength, mode="paired")
    _require(tuple(ar.shape) == (M,), f"paired scalar steering shape {tuple(ar.shape)} != ({M},)")
    cart = steering_vectors(pos, theta_r, np.array([-0.8, 0.6]), wavelength, mode="cartesian")
    _require(tuple(cart.shape) == (1, 2, M), f"cartesian steering shape {tuple(cart.shape)} != (1,2,{M})")
    _assert_close("steering vector unit norm", float(ar.abs().square().sum().item()), 1.0, 1e-6)

    H = ar[:, None] * at[None, :]
    hdd = torch.zeros(1, 1, M, 1, M, 1, 1, dtype=torch.complex64, device="cuda")
    hdd[0, 0, :, 0, :, 0, 0] = H
    phis = np.array([-0.8, -0.4, 0.0, 0.6, 0.8])
    scan_r = steering_vectors(pos, np.pi / 2, phis, wavelength, mode="cartesian").reshape(-1, M)
    scan_t = steering_vectors(pos, np.pi / 2, phis, wavelength, mode="cartesian").reshape(-1, M)
    spectrum = angular_delay_doppler_spectrum(hdd, scan_r, scan_t, mode="cartesian")
    peak = spectrum[0, 0, 3, 0, 1, 0, 0]
    maximum = float(spectrum.max().item())
    _assert_close("ADS matched peak", float(peak.item()), 1.0, 1e-5)
    _require(peak.item() >= maximum - 1e-6, f"ADS peak {peak.item()} is not the global maximum {maximum}")

    return {
        "steering_vectors": {
            "signature": str(inspect.signature(steering_vectors)),
            "formula": "a_m = exp(+j 2*pi/lambda * d_m . r_hat(theta, phi)) / sqrt(M)",
            "positions_m": pos.cpu().tolist(),
            "paired_scalar_shape": list(ar.shape),
            "cartesian_shape": list(cart.shape),
            "unit_norm": float(ar.abs().square().sum().item()),
        },
        "angular_delay_doppler_spectrum": {
            "signature": str(inspect.signature(angular_delay_doppler_spectrum)),
            "input_hdd_shape": list(hdd.shape),
            "output_shape": list(spectrum.shape),
            "rx_scan_phis_rad": phis.tolist(),
            "tx_scan_phis_rad": phis.tolist(),
            "matched_index": [3, 1],
            "matched_value": float(peak.item()),
            "global_max": maximum,
            "full_abs": spectrum.abs().reshape(5, 5).cpu().tolist(),
        },
        "checks": {"matched_peak": float(peak.item()), "unit_norm": float(ar.abs().square().sum().item())},
    }


def probe_interop() -> dict[str, Any]:
    scene, _, _ = _make_scene()
    paths = PathSolver()(scene, max_depth=0, seed=42)
    tensors = {}
    for name, value in (
        ("a_real", paths.a[0]),
        ("a_imag", paths.a[1]),
        ("tau", paths.tau),
        ("valid", paths.valid),
        ("interactions", paths.interactions),
    ):
        tensor = _cuda_tensor(value)
        tensors[name] = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "device": str(tensor.device),
            "requires_grad": bool(tensor.requires_grad),
            "data_ptr_nonzero": bool(tensor.data_ptr() != 0),
        }
        _require(tensor.is_cuda, f"{name} is not CUDA")
        _require(not tensor.requires_grad, f"{name} unexpectedly requires grad")
    a_cir, tau_cir = paths.cir(out_type="torch", normalize_delays=False)
    a_cir = _cuda_tensor(a_cir)
    tau_cir = _cuda_tensor(tau_cir)
    _require(a_cir.device == tau_cir.device, "CIR and tau devices differ")
    return {
        "method": "mi.TensorXf.torch(); torch.as_tensor(mi.TensorXf) also works",
        "tensors": tensors,
        "cir": {
            "a_shape": list(a_cir.shape),
            "a_dtype": str(a_cir.dtype),
            "a_device": str(a_cir.device),
            "a_requires_grad": bool(a_cir.requires_grad),
            "tau_shape": list(tau_cir.shape),
            "tau_dtype": str(tau_cir.dtype),
            "tau_device": str(tau_cir.device),
        },
    }


def probe_determinism() -> dict[str, Any]:
    scene, _, _ = _make_scene()
    results = {}
    for deterministic in (False, True):
        p1 = PathSolver(deterministic=deterministic)(scene, max_depth=1, seed=42)
        p2 = PathSolver(deterministic=deterministic)(scene, max_depth=1, seed=42)
        tau1 = _cuda_tensor(p1.tau)
        tau2 = _cuda_tensor(p2.tau)
        a1 = _complex_tensor(p1.a[0], p1.a[1])
        a2 = _complex_tensor(p2.a[0], p2.a[1])
        exact_tau = bool(torch.equal(tau1, tau2))
        exact_a = bool(torch.equal(a1, a2))
        _require(exact_tau, f"PathSolver deterministic={deterministic} tau differs")
        _require(exact_a, f"PathSolver deterministic={deterministic} a differs")
        results[f"deterministic_{deterministic}"] = {
            "tau_exact_equal": exact_tau,
            "a_exact_equal": exact_a,
            "tau_max_abs_difference": float((tau1 - tau2).abs().max().item()),
            "a_max_abs_difference": float((a1 - a2).abs().max().item()),
        }

    rcs_results = {}
    for deterministic in (False, True):
        rcs_scene, _, _ = _make_scene()
        _add_target(rcs_scene)
        r1 = RCSSolver(deterministic=deterministic)(rcs_scene, max_depth=1, seed=42)
        r2 = RCSSolver(deterministic=deterministic)(rcs_scene, max_depth=1, seed=42)
        tau1 = _cuda_tensor(r1.tau)
        tau2 = _cuda_tensor(r2.tau)
        a1 = _complex_tensor(r1.a[0], r1.a[1])
        a2 = _complex_tensor(r2.a[0], r2.a[1])
        exact_tau = bool(torch.equal(tau1, tau2))
        exact_a = bool(torch.equal(a1, a2))
        _require(exact_tau, f"RCSSolver deterministic={deterministic} tau differs")
        _require(exact_a, f"RCSSolver deterministic={deterministic} a differs")
        rcs_results[f"deterministic_{deterministic}"] = {
            "tau_exact_equal": exact_tau,
            "a_exact_equal": exact_a,
        }

    return {
        "path_solver": results,
        "rcs_solver": rcs_results,
        "caveat": "Sionna states that a fixed seed alone does not guarantee deterministic results; deterministic=True is required.",
    }


def probe_frequency_callback() -> dict[str, Any]:
    scene = load_scene()

    def callback(f: Any) -> tuple[Any, Any]:
        return f / 1e9, 0.01

    material = RadioMaterial(name="callback-material", frequency_update_callback=callback)
    scene.add(material)
    scene.frequency = 3.5e9
    eps35 = float(material.relative_permittivity[0])
    sigma35 = float(material.conductivity[0])
    scene.frequency = 7e9
    eps7 = float(material.relative_permittivity[0])
    sigma7 = float(material.conductivity[0])
    _assert_close("callback epsilon at 3.5 GHz", eps35, 3.5, 1e-6)
    _assert_close("callback epsilon at 7 GHz", eps7, 7.0, 1e-6)
    _assert_close("callback conductivity", sigma7, 0.01, 1e-8)
    return {
        "scene_frequency_signature": str(inspect.signature(type(scene).frequency.fset)),
        "callback_signature": "Callable[[mi.Float], Tuple[mi.Float, mi.Float]]",
        "epsilon_3p5_ghz": eps35,
        "epsilon_7_ghz": eps7,
        "conductivity_3p5_ghz": sigma35,
        "conductivity_7_ghz": sigma7,
        "note": "Scene.frequency calls frequency_update() on every radio material; constant materials have callback None.",
    }


def probe_array() -> dict[str, Any]:
    array = PlanarArray(
        num_rows=1,
        num_cols=16,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    wavelength = SPEED_OF_LIGHT / FC
    normalized = np.asarray(array.normalized_positions)
    positions = np.asarray(array.positions(wavelength))
    _require(normalized.shape == (3, 16), f"normalized_positions shape {normalized.shape} != (3,16)")
    _require(positions.shape == (3, 16), f"positions shape {positions.shape} != (3,16)")
    y = normalized[1]
    _require(np.all(np.diff(y) > 0), "ULA element y positions are not increasing")
    _assert_close("ULA first normalized y", float(y[0]), -3.75, 1e-6)
    _assert_close("ULA last normalized y", float(y[-1]), 3.75, 1e-6)
    _assert_close("ULA physical span m", float(positions[1, -1] - positions[1, 0]), 7.5 * wavelength, 1e-6)
    return {
        "signature": str(inspect.signature(PlanarArray)),
        "num_ant": int(array.num_ant),
        "array_size": int(array.array_size),
        "pattern": "iso",
        "polarization": "V",
        "normalized_positions_shape": list(normalized.shape),
        "normalized_y_wavelengths": [float(v) for v in y.tolist()],
        "positions_shape": list(positions.shape),
        "positions_m": positions.tolist(),
        "broadside": "+x (array lies in the y-z plane at x=0)",
        "element_order": "index increases with y; row 0, then columns 0..15",
        "phase_sign": "RT synthetic array applies exp(+j 2*pi/lambda * p . k) for both TX departure and RX arrival direction vectors.",
    }


def main() -> int:
    output_dir = Path("results/phase1")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "api_probe.json"
    evidence: dict[str, Any] = {
        "status": "RUNNING",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variant": mi.variant(),
        "torch_cuda": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "sections": {},
    }
    try:
        _require(mi.variant() is not None and mi.variant().startswith("cuda"), f"non-CUDA Mitsuba variant {mi.variant()!r}")
        _require(torch.cuda.is_available(), "torch.cuda.is_available() is False")
        evidence["device"] = torch.cuda.get_device_name(0)
        evidence["sections"]["paths_and_angles"] = probe_paths_and_angles()
        evidence["sections"]["cir"] = probe_cir()
        evidence["sections"]["los_amplitude_carrier_phase"] = check_los_amplitude_and_carrier_phase()
        evidence["sections"]["cfr_and_normalization"] = check_cfr_and_normalization()
        evidence["sections"]["reflection_and_material"] = check_reflection_and_material()
        evidence["sections"]["rcs"] = probe_rcs()
        evidence["sections"]["synthetic_array"] = probe_synthetic_array()
        evidence["sections"]["conversions"] = probe_conversions()
        evidence["sections"]["isac"] = probe_isac()
        evidence["sections"]["array"] = probe_array()
        evidence["sections"]["frequency_callback"] = probe_frequency_callback()
        evidence["sections"]["interop"] = probe_interop()
        evidence["sections"]["determinism"] = probe_determinism()
        evidence["status"] = "PASS"
        exit_code = 0
    except Exception as exc:  # noqa: BLE001
        evidence["status"] = "FAIL"
        evidence["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        exit_code = 1
        print(f"API PROBE FAILED: {type(exc).__name__}: {exc}", flush=True)
        print(evidence["error"]["traceback"], flush=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"status: {evidence['status']}")
    print(f"wrote {output_path}")
    if evidence["status"] == "PASS":
        for name, section in evidence["sections"].items():
            print(f"PASS {name}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
