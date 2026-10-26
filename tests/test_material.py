from __future__ import annotations

import json

import pytest

from isac_verify.config import PilotConfig
from isac_verify.material import build_material_reference, write_material_reference


def test_material_reference_table(config: PilotConfig, output_dir) -> None:
    reference = write_material_reference(
        config, output_dir / "material_reference.json"
    )
    assert reference["material_model"] == "Sionna finite-thickness single-layer slab"
    assert reference["angle_convention"]["name"] == "incidence_from_wall_normal"
    assert reference["material"]["thickness_m"] == pytest.approx(
        reference["material"]["api_default_thickness_m"], abs=1e-12
    )
    entries = reference["coefficients"]
    assert [entry["incidence_angle_deg"] for entry in entries] == [0.0, 26.565, 45.0, 63.435]
    by_angle = {entry["incidence_angle_deg"]: entry for entry in entries}
    normal = by_angle[0.0]
    assert normal["r_te"]["real"] == pytest.approx(-0.3660106659, abs=1e-6)
    assert normal["r_te"]["imag"] == pytest.approx(-0.2306077182, abs=1e-6)
    assert normal["r_tm"]["real"] == pytest.approx(0.3660109639, abs=1e-6)
    assert normal["r_tm"]["imag"] == pytest.approx(0.2306070179, abs=1e-6)
    forty_five = by_angle[45.0]
    assert forty_five["r_te"]["real"] == pytest.approx(-0.2030535042, abs=1e-6)
    assert forty_five["r_te"]["imag"] == pytest.approx(0.1115389690, abs=1e-6)
    saved = json.loads((output_dir / "material_reference.json").read_text())
    assert saved["frequency_hz"] == config.frequency_hz
    assert len(saved["coefficients"]) == 4
