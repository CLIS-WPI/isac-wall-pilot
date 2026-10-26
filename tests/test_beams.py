from __future__ import annotations

import json
from math import log10

import numpy as np
import pytest
import torch

from isac_verify.beams import (
    ArrayGeometry,
    beam_gain,
    beam_power_w,
    beampattern,
    dbm_to_watt,
    direct_beam,
    map_free_beams,
    map_free_codebook_deg,
    projected_wall_beam,
    robust_wall_beam,
)
from isac_verify.config import PilotConfig


def _power(config: PilotConfig) -> float:
    return dbm_to_watt(config.get("waveform", "nominal_total_power_dbm"))


def test_analytical_direct_peak(config: PilotConfig) -> None:
    array = ArrayGeometry.from_config(config)
    power = _power(config)
    requested = config.get("validation", "rt_adapter", "requested_phi_deg")
    for phi in requested:
        beam = direct_beam(float(phi), power, array, dtype=torch.complex128)
        scan = np.arange(-90.0, 90.0001, 0.05)
        pattern = beampattern(scan, beam, array, dtype=torch.complex128)
        measured = float(scan[int(torch.argmax(pattern))])
        assert abs(measured - float(phi)) <= 0.05


def test_projected_null_depth(config: PilotConfig, output_dir) -> None:
    array = ArrayGeometry.from_config(config)
    power = _power(config)
    phi_wall = float(config.get("geometry_reference", "wall_departure_candidate_deg"))
    beam = projected_wall_beam(
        phi_wall,
        0.0,
        power,
        array,
        min_projected_norm=float(config.get("beam", "projected_norm_floor")),
        dtype=torch.complex128,
    )
    assert beam.usable
    scan = np.arange(-90.0, 90.0001, 0.01)
    pattern = beampattern(scan, beam, array, dtype=torch.complex128)
    peak = float(pattern.max())
    direct_gain = beam_gain(0.0, beam, array, dtype=torch.complex128)
    null_db = 10.0 * log10(max(direct_gain, 1e-300) / peak)
    assert null_db <= float(config.get("validation", "null_depth_min_db"))
    payload = {
        "phi_wall_deg": phi_wall,
        "phi_direct_deg": 0.0,
        "peak_gain": peak,
        "direct_gain": direct_gain,
        "null_depth_db": null_db,
        "projected_norm": beam.projected_norm,
    }
    (output_dir / "null_depth.json").write_text(json.dumps(payload, indent=2) + "\n")


def test_all_beam_types_preserve_power(config: PilotConfig) -> None:
    array = ArrayGeometry.from_config(config)
    power = _power(config)
    tolerance = float(config.get("validation", "beam_power_relative_tolerance"))
    direct = direct_beam(0.0, power, array)
    assert beam_power_w(direct) == pytest.approx(power, rel=tolerance)
    projected = projected_wall_beam(
        26.565,
        0.0,
        power,
        array,
        min_projected_norm=float(config.get("beam", "projected_norm_floor")),
    )
    assert projected.usable
    assert beam_power_w(projected) == pytest.approx(power, rel=tolerance)
    robust = robust_wall_beam(
        candidate_range_m=60.0,
        wall_offset_m=15.0,
        sigma_map_m=float(config.get("beam", "uncertainty", "sigma_map_m")),
        sigma_candidate_range_m=float(
            config.get("beam", "uncertainty", "sigma_candidate_range_m")
        ),
        phi_direct_deg=0.0,
        tx_power_w=power,
        array=array,
        num_samples=int(config.get("beam", "num_robust_samples")),
        reference_index=int(config.get("steering", "reference_element_index")),
        min_projected_norm=float(config.get("beam", "projected_norm_floor")),
    )
    assert robust.usable
    assert robust.sample_angles_deg is not None
    assert robust.sample_angles_deg.numel() == 257
    assert beam_power_w(robust) == pytest.approx(power, rel=tolerance)
    codebook = map_free_codebook_deg(config)
    beams = map_free_beams(
        0.0,
        power,
        array,
        codebook,
        min_projected_norm=float(config.get("beam", "projected_norm_floor")),
    )
    assert len(beams) == len(codebook)
    for beam in beams:
        assert beam.usable
        assert beam_power_w(beam) == pytest.approx(power, rel=tolerance)


def test_rt_adapter_peak_table(config: PilotConfig, output_dir) -> None:
    import mitsuba as mi
    import sionna.rt as rt
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    assert mi.variant().startswith("cuda")
    tx_values = config.section("arrays", "gnb_tx")
    rx_values = config.section("arrays", "phase2_validation_rx")
    validation = config.section("validation", "rt_adapter")
    radius = float(validation["ring_radius_m"])
    spacing = float(validation["receiver_spacing_deg"])
    arc = validation["visible_arc_deg"]
    angles = np.arange(float(arc[0]), float(arc[1]) + 0.5 * spacing, spacing)
    tx_array = PlanarArray(
        num_rows=1,
        num_cols=int(tx_values["num_elements"]),
        vertical_spacing=float(tx_values["spacing_wavelengths"]),
        horizontal_spacing=float(tx_values["spacing_wavelengths"]),
        pattern=str(tx_values["pattern"]),
        polarization=str(tx_values["polarization"]),
    )
    rx_array = PlanarArray(
        num_rows=int(rx_values["num_rows"]),
        num_cols=int(rx_values["num_cols"]),
        vertical_spacing=float(rx_values["vertical_spacing"]),
        horizontal_spacing=float(rx_values["horizontal_spacing"]),
        pattern=str(rx_values["pattern"]),
        polarization=str(rx_values["polarization"]),
    )
    scene = load_scene()
    scene.frequency = config.frequency_hz
    scene.tx_array = tx_array
    scene.rx_array = rx_array
    gnb = config.get("physical", "gnb_position_m")
    devices = [Transmitter("gnb", position=list(gnb))]
    for index, angle in enumerate(angles):
        azimuth = np.deg2rad(angle)
        devices.append(
            Receiver(
                f"ring-{index}",
                position=[
                    float(radius * np.cos(azimuth)),
                    float(radius * np.sin(azimuth)),
                    float(gnb[2]),
                ],
            )
        )
    scene.add(devices)
    solver = config.section("solver")
    paths = PathSolver(deterministic=bool(solver["path_solver"]["deterministic"]))(
        scene,
        max_depth=0,
        synthetic_array=bool(validation["synthetic_array"]),
        los=True,
        specular_reflection=False,
        diffuse_reflection=bool(solver["path_solver"]["diffuse_reflection"]),
        refraction=bool(solver["path_solver"]["refraction"]),
        diffraction=bool(solver["path_solver"]["diffraction"]),
        seed=int(solver["path_solver"]["seed"]),
    )
    valid = torch.as_tensor(paths.valid)
    assert bool(valid.all().item())
    real = torch.as_tensor(paths.a[0])
    imag = torch.as_tensor(paths.a[1])
    channel = (real + 1j * imag).reshape(len(angles), int(tx_values["num_elements"]))
    array = ArrayGeometry.from_config(config)
    power = _power(config)
    rows = []
    for requested in validation["requested_phi_deg"]:
        beam = direct_beam(float(requested), power, array, device="cuda")
        gains = (channel @ beam.weights).abs().square()
        index = int(torch.argmax(gains))
        measured = float(angles[index])
        error = abs(measured - float(requested))
        assert error <= float(validation["peak_tolerance_deg"])
        rows.append(
            {
                "requested_phi_deg": float(requested),
                "measured_phi_deg": measured,
                "absolute_error_deg": error,
                "peak_gain": float(gains[index].item()),
            }
        )
    payload = {
        "ring_radius_m": radius,
        "receiver_spacing_deg": spacing,
        "visible_arc_deg": [float(arc[0]), float(arc[1])],
        "note": (
            "The y-axis ULA is front-back ambiguous; the validation ring is the "
            "visible forward arc so that the requested azimuth is the unique peak."
        ),
        "rows": rows,
    }
    (output_dir / "rt_adapter.json").write_text(json.dumps(payload, indent=2) + "\n")
