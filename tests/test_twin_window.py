"""Self-check for the twin-derived consistency-window lower edge (wpe_defensive.twin_window_lo_s).

The edge is the earliest second-order forced LoS ghost over R_A in [5, R_T-standoff], minus one point-spread
width. The ghost excess (hypot(R_A, o) - R_A)/c grows with the image offset o and shrinks with R_A, so the edge
is set by the largest offset (double bounce) at R_A = 5 m. No GPU needed.
"""

import math
import sys
from pathlib import Path

from isac_verify.config import PilotConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import wpe_defensive as W  # noqa: E402


def test_twin_window_lo_closed_form_and_monotonicity():
    config = PilotConfig.load("configs/pilot.yaml")
    c = float(config.speed_of_light)
    refl_min_s, R_T, dsig, d_pos, d_neg, psf, smap = 395e-9, 50.0, 0.3e-9, 6.0, -6.0, 2.0e-9, 0.5

    lo = W.twin_window_lo_s(config, refl_min_s, R_T, dsig, d_pos, d_neg, psf, smap)
    # Closed form: double-bounce offset 2(|d_pos|+|d_neg|), walls inflated by 3*sigma, at R_A = 5 m sets the edge.
    infl = 3.0 * smap
    o = 2.0 * ((abs(d_pos) + infl) + (abs(d_neg) + infl))
    expected = refl_min_s - (math.hypot(5.0, o) - 5.0) / c - psf - 3.0 * dsig
    assert abs(lo - expected) < 1e-12, (lo, expected)

    # Below the reflector-gate lower edge, and wider walls / more PSF / more candidate sigma all push it earlier.
    assert lo < refl_min_s - psf
    assert W.twin_window_lo_s(config, refl_min_s, R_T, dsig, 12.0, -12.0, psf, smap) < lo
    assert W.twin_window_lo_s(config, refl_min_s, R_T, dsig, d_pos, d_neg, 4.0e-9, smap) < lo
    assert W.twin_window_lo_s(config, refl_min_s, R_T, 0.6e-9, d_pos, d_neg, psf, smap) < lo
