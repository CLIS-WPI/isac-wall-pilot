"""Steering vectors and beam construction for the horizontal y-axis ULA.

D5 convention: ``v(phi) = conj(a_tx(phi))`` where
``a_tx(phi)[m] = exp(+j 2 pi / lambda * p_m . k(phi)) / sqrt(M)``.
Physical directional gain is therefore ``|v(phi)^H w|^2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2, pi, sqrt
from typing import Sequence

import numpy as np
import torch
from scipy.stats import norm, qmc

from .config import PilotConfig


class BeamError(ValueError):
    """Raised when a beam cannot be constructed or has no usable power."""


@dataclass(frozen=True)
class ArrayGeometry:
    num_elements: int
    spacing_wavelengths: float
    wavelength_m: float

    @classmethod
    def from_config(cls, config: PilotConfig) -> "ArrayGeometry":
        tx = config.section("arrays", "gnb_tx")
        return cls(
            num_elements=int(tx["num_elements"]),
            spacing_wavelengths=float(tx["spacing_wavelengths"]),
            wavelength_m=config.wavelength_m,
        )

    def positions_m(
        self,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        offsets = torch.arange(self.num_elements, device=device, dtype=dtype)
        offsets = offsets - 0.5 * (self.num_elements - 1)
        y = offsets * self.spacing_wavelengths * self.wavelength_m
        return torch.stack(
            [torch.zeros_like(y), y, torch.zeros_like(y)], dim=-1
        )


@dataclass(frozen=True)
class BeamResult:
    weights: torch.Tensor | None
    status: str
    phi_deg: float
    projected_norm: float | None = None
    num_samples: int = 1
    sample_angles_deg: torch.Tensor | None = None

    @property
    def usable(self) -> bool:
        return self.weights is not None and self.status == "ok"


def dbm_to_watt(power_dbm: float) -> float:
    return 10.0 ** (float(power_dbm) / 10.0) * 1e-3


def array_response(
    phi_deg: float,
    array: ArrayGeometry,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    positions = array.positions_m(device=device, dtype=torch.float64)
    phi = torch.tensor(
        np.deg2rad(float(phi_deg)), device=device, dtype=torch.float64
    )
    direction = torch.stack(
        [torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)]
    )
    phase = (2.0 * pi / array.wavelength_m) * (positions @ direction)
    response = torch.exp(1j * phase) / sqrt(array.num_elements)
    return response.to(dtype)


def steering_vector(
    phi_deg: float,
    array: ArrayGeometry,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    return array_response(phi_deg, array, device=device, dtype=dtype).conj()


def beam_power_w(beam: BeamResult) -> float:
    if beam.weights is None:
        raise BeamError(f"beam status {beam.status!r} has no weights")
    return float(beam.weights.abs().square().sum().item())


def beam_gain(
    phi_deg: float,
    beam: BeamResult,
    array: ArrayGeometry,
    *,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> float:
    if beam.weights is None:
        raise BeamError(f"beam status {beam.status!r} has no weights")
    target_dtype = dtype or beam.weights.dtype
    vector = steering_vector(
        phi_deg, array, device=device, dtype=target_dtype
    )
    return float(torch.vdot(vector, beam.weights).abs().square().item())


def direct_beam(
    phi_deg: float,
    tx_power_w: float,
    array: ArrayGeometry,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.complex64,
) -> BeamResult:
    if tx_power_w <= 0.0:
        raise BeamError("transmit power must be positive")
    vector = steering_vector(phi_deg, array, device=device, dtype=dtype)
    weights = sqrt(tx_power_w) * vector
    return BeamResult(weights=weights, status="ok", phi_deg=float(phi_deg))


def projection_operator(
    phi_deg: float,
    array: ArrayGeometry,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    vector = steering_vector(phi_deg, array, device=device, dtype=dtype)
    return torch.eye(
        array.num_elements, device=device, dtype=dtype
    ) - torch.outer(vector, vector.conj())


def projected_wall_beam(
    phi_wall_deg: float,
    phi_direct_deg: float,
    tx_power_w: float,
    array: ArrayGeometry,
    *,
    min_projected_norm: float,
    device: str = "cuda",
    dtype: torch.dtype = torch.complex64,
) -> BeamResult:
    if tx_power_w <= 0.0:
        raise BeamError("transmit power must be positive")
    if min_projected_norm < 0.0:
        raise BeamError("projected-norm floor must be non-negative")
    v0 = steering_vector(
        phi_direct_deg, array, device=device, dtype=dtype
    )
    vw = steering_vector(phi_wall_deg, array, device=device, dtype=dtype)
    projection = vw - v0 * torch.vdot(v0, vw)
    projected_norm = float(torch.linalg.vector_norm(projection).item())
    if projected_norm < min_projected_norm:
        return BeamResult(
            weights=None,
            status="unusable_projected_norm",
            phi_deg=float(phi_wall_deg),
            projected_norm=projected_norm,
        )
    weights = sqrt(tx_power_w) * projection / projected_norm
    return BeamResult(
        weights=weights,
        status="ok",
        phi_deg=float(phi_wall_deg),
        projected_norm=projected_norm,
    )


def sobol_uncertainty_samples(
    num_samples: int,
) -> tuple[np.ndarray, str]:
    if num_samples < 2:
        raise BeamError("at least two uncertainty samples are required")
    n_sobol = num_samples - 1
    exponent = int(round(log2(n_sobol)))
    if 2**exponent != n_sobol:
        raise BeamError(
            "num_samples - 1 must be a power of two for the fixed Sobol construction"
        )
    engine = qmc.Sobol(d=2, scramble=False)
    points = engine.random(n_sobol)
    clipped = np.clip(
        points,
        1.0 / (2.0 * n_sobol),
        1.0 - 1.0 / (2.0 * n_sobol),
    )
    centered = np.vstack([np.array([[0.5, 0.5]]), clipped])
    description = (
        f"1 centered sample + {n_sobol} unscrambled Sobol(d=2) points; "
        "uniform quantiles mapped through the standard-normal ppf"
    )
    return centered, description


def uncertainty_wall_angles_deg(
    *,
    candidate_range_m: float,
    wall_offset_m: float,
    sigma_map_m: float,
    sigma_candidate_range_m: float,
    num_samples: int,
) -> tuple[torch.Tensor, str]:
    if candidate_range_m <= 0.0:
        raise BeamError("candidate range must be positive")
    if sigma_map_m < 0.0 or sigma_candidate_range_m < 0.0:
        raise BeamError("uncertainty standard deviations must be non-negative")
    quantiles, description = sobol_uncertainty_samples(num_samples)
    wall_errors = norm.ppf(quantiles[:, 0]) * sigma_map_m
    range_errors = norm.ppf(quantiles[:, 1]) * sigma_candidate_range_m
    ranges = candidate_range_m + range_errors
    if np.any(ranges <= 0.0):
        raise BeamError("candidate-range uncertainty produced a non-positive range")
    angles = np.degrees(
        np.arctan2(2.0 * (wall_offset_m + wall_errors), ranges)
    )
    return torch.tensor(angles, dtype=torch.float64), description


def robust_wall_beam(
    *,
    candidate_range_m: float,
    wall_offset_m: float,
    sigma_map_m: float,
    sigma_candidate_range_m: float,
    phi_direct_deg: float,
    tx_power_w: float,
    array: ArrayGeometry,
    num_samples: int,
    reference_index: int,
    min_projected_norm: float,
    device: str = "cuda",
    dtype: torch.dtype = torch.complex64,
) -> BeamResult:
    if num_samples != 257:
        raise BeamError("the declared robust beam requires exactly 257 samples")
    if not 0 <= reference_index < array.num_elements:
        raise BeamError("reference element index is outside the array")
    angles, description = uncertainty_wall_angles_deg(
        candidate_range_m=candidate_range_m,
        wall_offset_m=wall_offset_m,
        sigma_map_m=sigma_map_m,
        sigma_candidate_range_m=sigma_candidate_range_m,
        num_samples=num_samples,
    )
    vectors = torch.stack(
        [
            steering_vector(
                float(angle), array, device=device, dtype=dtype
            )
            for angle in angles
        ]
    )
    reference = vectors[:, reference_index]
    aligned = vectors * torch.exp(-1j * torch.angle(reference))[:, None]
    average = aligned.mean(dim=0)
    v0 = steering_vector(
        phi_direct_deg, array, device=device, dtype=dtype
    )
    projection = average - v0 * torch.vdot(v0, average)
    projected_norm = float(torch.linalg.vector_norm(projection).item())
    if projected_norm < min_projected_norm:
        return BeamResult(
            weights=None,
            status="unusable_projected_norm",
            phi_deg=float("nan"),
            projected_norm=projected_norm,
            num_samples=num_samples,
            sample_angles_deg=angles,
        )
    weights = sqrt(tx_power_w) * projection / projected_norm
    return BeamResult(
        weights=weights,
        status="ok",
        phi_deg=float("nan"),
        projected_norm=projected_norm,
        num_samples=num_samples,
        sample_angles_deg=angles,
    )


def map_free_codebook_deg(config: PilotConfig) -> list[float]:
    angles = config.get("beam", "map_free", "angles_deg")
    return [float(angle) for angle in angles]


def map_free_beams(
    phi_direct_deg: float,
    tx_power_w: float,
    array: ArrayGeometry,
    angles_deg: Sequence[float],
    *,
    min_projected_norm: float,
    device: str = "cuda",
    dtype: torch.dtype = torch.complex64,
) -> list[BeamResult]:
    return [
        projected_wall_beam(
            float(angle),
            phi_direct_deg,
            tx_power_w,
            array,
            min_projected_norm=min_projected_norm,
            device=device,
            dtype=dtype,
        )
        for angle in angles_deg
    ]


def beampattern(
    phis_deg: Sequence[float],
    beam: BeamResult,
    array: ArrayGeometry,
    *,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if beam.weights is None:
        raise BeamError(f"beam status {beam.status!r} has no weights")
    target_dtype = dtype or beam.weights.dtype
    positions = array.positions_m(device=device, dtype=torch.float64)
    phi = torch.deg2rad(
        torch.as_tensor(list(phis_deg), device=device, dtype=torch.float64)
    )
    directions = torch.stack(
        [torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)], dim=-1
    )
    phase = (2.0 * pi / array.wavelength_m) * (directions @ positions.T)
    steering = torch.exp(1j * phase).to(target_dtype).conj() / sqrt(
        array.num_elements
    )
    weights = beam.weights.to(target_dtype)
    gains = (steering.conj() @ weights).abs().square()
    return gains.to(torch.float64)
