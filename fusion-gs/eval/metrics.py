"""Evaluation metrics for predicted equilibria.

All functions take arrays on a ``gs.solver.Grid`` with shape (nr, nz), ``indexing="ij"``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from gs.solver import Grid


def relative_l2(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> float:
    """||pred - true|| / ||true|| over ``mask`` (all points if None)."""
    raise NotImplementedError


def gs_residual(psi: np.ndarray, grid: Grid, rhs: np.ndarray, mask: np.ndarray) -> float:
    """Normalized residual ||Delta* psi - rhs|| / ||rhs|| over interior plasma nodes.

    Only nodes whose four neighbours are also inside ``mask`` are used, so the
    irregular boundary does not dominate the value.
    """
    raise NotImplementedError


def find_magnetic_axis(psi: np.ndarray, grid: Grid, mask: np.ndarray) -> tuple[float, float, float]:
    """(R_axis, Z_axis, psi_axis) of the extremum of psi inside ``mask``, refined to
    sub-grid accuracy by a local quadratic fit. The extremum is the maximum if psi is
    mostly positive inside the mask and the minimum otherwise."""
    raise NotImplementedError


def flux_contour(psi: np.ndarray, grid: Grid, level: float, R_axis: float, Z_axis: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed contour psi = level surrounding the magnetic axis, as (R, Z) arrays."""
    raise NotImplementedError


def lcfs_shape_error(R_pred, Z_pred, R_true, Z_true) -> float:
    """Symmetric mean distance (metres) between two closed boundary curves."""
    raise NotImplementedError


def q_profile(
    psi: np.ndarray,
    grid: Grid,
    psi_axis: float,
    psi_boundary: float,
    R_axis: float,
    Z_axis: float,
    F: Callable[[np.ndarray], np.ndarray],
    psin_levels: np.ndarray,
) -> np.ndarray:
    """Safety factor q(psi_n) = F / (2 pi) * closed-integral dl / (R^2 B_p) on each level.

    ``F`` maps psi_n to F = R B_phi. B_p = |grad psi| / R is interpolated from a
    bicubic spline of psi.
    """
    raise NotImplementedError


def q95(psi, grid, psi_axis, psi_boundary, R_axis, Z_axis, F) -> float:
    raise NotImplementedError


def time_call(fn: Callable[[], object], repeats: int = 5) -> float:
    """Median wall-clock seconds of ``fn()`` over ``repeats`` calls (after one warm-up)."""
    raise NotImplementedError
