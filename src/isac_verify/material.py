"""Sionna slab-model wall material reference at the pilot frequency."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mitsuba as mi

from sionna.rt import RadioMaterial
from sionna.rt.utils.electromagnetics import (
    complex_relative_permittivity,
    itu_coefficients_single_layer_slab,
)

from .config import PilotConfig


class MaterialError(ValueError):
    """Raised when the configured wall material is inconsistent with the API."""


def default_radio_material_thickness_m() -> float:
    material = RadioMaterial(name="phase2-default-thickness-inspection")
    return float(material.thickness[0])


def _complex_pair(value: Any) -> list[float]:
    return [float(value.real[0]), float(value.imag[0])]


def _coefficient_entry(
    value: Any,
) -> dict[str, Any]:
    real, imag = _complex_pair(value)
    magnitude = math.hypot(real, imag)
    phase = math.atan2(imag, real)
    return {
        "real": real,
        "imag": imag,
        "magnitude": magnitude,
        "phase_rad": phase,
        "magnitude_db": 20.0 * math.log10(magnitude) if magnitude > 0.0 else None,
    }


def build_material_reference(config: PilotConfig) -> dict[str, Any]:
    wall = config.section("wall")
    frequency_hz = float(wall["reference_frequency_hz"])
    if frequency_hz != config.frequency_hz:
        raise MaterialError(
            "wall reference frequency does not match the configured carrier"
        )
    thickness_m = float(wall["thickness_m"])
    default_thickness = default_radio_material_thickness_m()
    if not math.isclose(thickness_m, default_thickness, rel_tol=0.0, abs_tol=1e-9):
        raise MaterialError(
            f"configured thickness {thickness_m} is not the Sionna default "
            f"{default_thickness}"
        )
    if wall["material_model"] != "sionna_slab":
        raise MaterialError(f"unsupported material model {wall['material_model']!r}")
    wavelength_m = config.wavelength_m
    omega = 2.0 * math.pi * frequency_hz
    relative_permittivity = float(wall["relative_permittivity"])
    conductivity = float(wall["conductivity_s_per_m"])
    eta = complex_relative_permittivity(
        mi.Float(relative_permittivity),
        mi.Float(conductivity),
        mi.Float(omega),
    )
    entries = []
    for angle_deg in wall["reference_incidence_angles_deg"]:
        angle = float(angle_deg)
        cos_theta = math.cos(math.radians(angle))
        r_te, r_tm, t_te, t_tm = itu_coefficients_single_layer_slab(
            mi.Float(cos_theta),
            eta,
            mi.Float(thickness_m),
            mi.Float(wavelength_m),
        )
        entries.append(
            {
                "incidence_angle_deg": angle,
                "cos_theta": cos_theta,
                "sin_theta": math.sin(math.radians(angle)),
                "r_te": _coefficient_entry(r_te),
                "r_tm": _coefficient_entry(r_tm),
                "t_te": _coefficient_entry(t_te),
                "t_tm": _coefficient_entry(t_tm),
            }
        )
    return {
        "frequency_hz": frequency_hz,
        "wavelength_m": wavelength_m,
        "material_model": "Sionna finite-thickness single-layer slab",
        "angle_convention": {
            "name": "incidence_from_wall_normal",
            "description": (
                "theta_i is measured from the inward wall normal; "
                "theta_i=0 is normal incidence"
            ),
            "incidence_angles_deg": [
                float(v) for v in wall["reference_incidence_angles_deg"]
            ],
        },
        "material": {
            "class": "RadioMaterial",
            "relative_permittivity": relative_permittivity,
            "conductivity_s_per_m": conductivity,
            "scattering_coefficient": float(wall["scattering_coefficient"]),
            "xpd_coefficient": float(wall["xpd_coefficient"]),
            "thickness_m": thickness_m,
            "api_default_thickness_m": default_thickness,
        },
        "complex_relative_permittivity": _complex_pair(eta),
        "coefficients": entries,
        "phase_convention": (
            "Sionna slab coefficient with exp(-j omega t) time dependence; "
            "the complex value is recorded exactly as returned by Sionna"
        ),
        "controlled_parameter": bool(wall["controlled_parameter"]),
        "swept": bool(wall["swept"]),
    }




def slab_coefficients(
    config: PilotConfig, angle_deg: float, thickness_m: float | None = None
) -> dict[str, Any]:
    wall = config.section("wall")
    frequency_hz = float(wall["reference_frequency_hz"])
    if frequency_hz != config.frequency_hz:
        raise MaterialError(
            "wall reference frequency does not match the configured carrier"
        )
    if wall["material_model"] != "sionna_slab":
        raise MaterialError(f"unsupported material model {wall['material_model']!r}")
    thickness = (
        float(wall["thickness_m"]) if thickness_m is None else float(thickness_m)
    )
    if thickness <= 0.0:
        raise MaterialError("wall thickness must be positive")
    omega = 2.0 * math.pi * frequency_hz
    eta = complex_relative_permittivity(
        mi.Float(float(wall["relative_permittivity"])),
        mi.Float(float(wall["conductivity_s_per_m"])),
        mi.Float(omega),
    )
    cos_theta = math.cos(math.radians(float(angle_deg)))
    r_te, r_tm, t_te, t_tm = itu_coefficients_single_layer_slab(
        mi.Float(cos_theta),
        eta,
        mi.Float(thickness),
        mi.Float(config.wavelength_m),
    )
    return {
        "incidence_angle_deg": float(angle_deg),
        "incidence_angle_from_normal_deg": float(angle_deg),
        "cos_theta": cos_theta,
        "sin_theta": math.sin(math.radians(float(angle_deg))),
        "thickness_m": thickness,
        "r_te": _coefficient_entry(r_te),
        "r_tm": _coefficient_entry(r_tm),
        "t_te": _coefficient_entry(t_te),
        "t_tm": _coefficient_entry(t_tm),
    }


def build_wall_thickness_stress(config: PilotConfig) -> dict[str, Any]:
    thicknesses = [
        float(value) for value in config.get("stress", "wall_thickness_m")
    ]
    if not thicknesses or any(value <= 0.0 for value in thicknesses):
        raise MaterialError("stress wall thicknesses must be positive")
    angles = [63.435, 45.0]
    entries = []
    for thickness in thicknesses:
        for angle in angles:
            entry = slab_coefficients(config, angle, thickness)
            entries.append(entry)
    return {
        "label": "follow-up wall-thickness stress; not the BRIEF baseline",
        "baseline_thickness_m": float(config.get("wall", "thickness_m")),
        "angles_deg": angles,
        "thicknesses_m": thicknesses,
        "entries": entries,
        "material": {
            "relative_permittivity": float(
                config.get("wall", "relative_permittivity")
            ),
            "conductivity_s_per_m": float(
                config.get("wall", "conductivity_s_per_m")
            ),
            "frequency_hz": config.frequency_hz,
            "wavelength_m": config.wavelength_m,
        },
    }


def write_material_reference(
    config: PilotConfig, output_path: str | Path | None = None
) -> dict[str, Any]:
    reference = build_material_reference(config)
    target = Path(
        output_path
        if output_path is not None
        else config.get("output", "material_reference")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(reference, handle, indent=2)
        handle.write("\n")
    return reference
