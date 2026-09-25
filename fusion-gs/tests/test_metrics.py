"""Analytic tests for eval.metrics.

Fields are constructed here. Only ``gs.solver.Grid`` is imported from ``gs``;
``gs.analytic`` and the solver routines are intentionally unused.
"""
from __future__ import annotations

import numpy as np
import pytest

from eval.metrics import (
    find_magnetic_axis,
    flux_contour,
    gs_residual,
    lcfs_shape_error,
    q95,
    q_profile,
    relative_l2,
    time_call,
)
from gs.solver import Grid


def _grid(rmin, rmax, zmin, zmax, nr, nz) -> Grid:
    return Grid(np.linspace(rmin, rmax, nr), np.linspace(zmin, zmax, nz))


def _full_mask(shape) -> np.ndarray:
    return np.ones(shape, dtype=bool)


# ---------------------------------------------------------------------------
# relative L2
# ---------------------------------------------------------------------------


def test_relative_l2_basic():
    true = np.array([3.0, 4.0])
    assert relative_l2(true, true) == pytest.approx(0.0)
    # ||(0, 3)|| / ||(3, 4)|| = 3/5
    assert relative_l2(np.array([3.0, 7.0]), true) == pytest.approx(0.6)
    assert relative_l2(np.zeros(2), true) == pytest.approx(1.0)


def test_relative_l2_mask():
    pred = np.array([[1.0, 10.0], [3.0, 4.0]])
    true = np.array([[1.0, 0.0], [3.0, 8.0]])
    mask = np.array([[True, False], [True, True]])
    # masked diff = [0, 0, -4], masked true = [1, 3, 8]
    num = np.sqrt(16.0)
    den = np.sqrt(1.0 + 9.0 + 64.0)
    assert relative_l2(pred, true, mask) == pytest.approx(num / den)
    # The rejected point would have dominated the unmasked norm.
    assert relative_l2(pred, true) != pytest.approx(relative_l2(pred, true, mask))


def test_relative_l2_zero_norm():
    zeros = np.zeros((2, 3))
    assert relative_l2(zeros, zeros) == 0.0
    assert relative_l2(np.ones((2, 3)), zeros) == float("inf")
    assert relative_l2(np.ones(4), np.zeros(4), np.zeros(4, dtype=bool)) == 0.0


# ---------------------------------------------------------------------------
# Grad-Shafranov residual
# ---------------------------------------------------------------------------


def test_gs_residual_quadratic_exact():
    """psi = R^2 has Delta* = 0; psi = Z^2 has Delta* = 2. Both are exact for the stencil."""
    grid = _grid(1.0, 3.0, -1.0, 1.0, 25, 21)
    rr, zz = grid.RR, grid.ZZ
    mask = _full_mask(rr.shape)

    psi_r = rr**2
    # rhs = 0, so the reported value is the absolute residual (roundoff).
    assert gs_residual(psi_r, grid, np.zeros_like(psi_r), mask) < 1e-10

    psi_z = zz**2
    assert gs_residual(psi_z, grid, np.full_like(psi_z, 2.0), mask) < 1e-10

    psi = 1.5 * rr**2 + 0.25 * zz**2 + 0.3 * rr**2 * zz + 2.0 * zz + 4.0
    # Delta*(R^2) = 0, Delta*(Z^2) = 2, Delta*(R^2 Z) = 0, Delta*(Z) = 0.
    rhs = np.full_like(psi, 0.5)
    assert gs_residual(psi, grid, rhs, mask) < 1e-10


def test_gs_residual_mask_ignores_exterior():
    grid = _grid(1.2, 2.4, -0.6, 0.6, 31, 31)
    rr, zz = grid.RR, grid.ZZ
    psi = rr**2 + zz**2
    rhs = np.full_like(psi, 2.0)
    radius = np.hypot(rr - 1.8, zz)
    mask = radius < 0.45
    # Corrupt psi and rhs outside the mask. Interior nodes only touch in-mask neighbours.
    psi_bad = psi.copy()
    rhs_bad = rhs.copy()
    psi_bad[~mask] = 1e6
    rhs_bad[~mask] = -1e6
    assert np.any(~mask)
    assert gs_residual(psi_bad, grid, rhs_bad, mask) < 1e-10


def test_gs_residual_requires_interior_nodes():
    grid = _grid(1.0, 2.0, -1.0, 1.0, 6, 6)
    psi = np.zeros((6, 6))
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, :] = True
    with pytest.raises(ValueError):
        gs_residual(psi, grid, psi, mask)


def test_gs_residual_r4_over_8_is_exact():
    """Continuous Delta*(R^4/8) = R^2.

    The conservative midpoint stencil integrates this field exactly: the half-point
    flux of R^4/8 equals R_{i+1/2}^2 / 2, and the centered difference of that
    quadratic is exact. The residual is roundoff on every resolution, which is
    stronger than the generic O(h^2) truncation of the scheme.
    """
    for n in (17, 33, 65):
        grid = _grid(1.0, 3.0, -1.0, 1.0, n, n)
        psi = grid.RR**4 / 8.0
        rhs = grid.RR**2
        rel = gs_residual(psi, grid, rhs, _full_mask(psi.shape))
        assert rel < 1e-10, f"n={n} residual {rel}"


def test_gs_residual_second_order():
    """psi = Z^4 has continuous Delta* = 12 Z^2 and a classic O(h^2) second difference.

    (R^4/8 is in the stencil's exactness space, so it cannot exhibit a nonzero
    convergence rate. Z^4 is the clean polynomial witness that the scheme is
    second order.)
    """
    errors = []
    for n in (17, 33, 65):
        grid = _grid(1.0, 3.0, -1.0, 1.0, n, n)
        psi = grid.ZZ**4
        rhs = 12.0 * grid.ZZ**2
        errors.append(gs_residual(psi, grid, rhs, _full_mask(psi.shape)))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert errors[0] > 1e-4
    for p, e0, e1 in zip(orders, errors[:-1], errors[1:]):
        assert 1.5 < p < 2.5, f"order {p} from {e0} -> {e1}"


# ---------------------------------------------------------------------------
# Magnetic axis
# ---------------------------------------------------------------------------


def test_find_magnetic_axis_negative_paraboloid():
    """psi = -((R-1.63)^2 + 2 (Z-0.071)^2) is an exact quadratic maximum.

    The mean is negative, and the only interior critical point is that maximum.
    """
    grid = _grid(1.2, 2.1, -0.4, 0.5, 11, 9)
    psi = -((grid.RR - 1.63) ** 2 + 2.0 * (grid.ZZ - 0.071) ** 2)
    r_axis, z_axis, psi_axis = find_magnetic_axis(psi, grid, _full_mask(psi.shape))
    assert r_axis == pytest.approx(1.63, abs=1e-6)
    assert z_axis == pytest.approx(0.071, abs=1e-6)
    assert psi_axis == pytest.approx(0.0, abs=1e-6)


def test_find_magnetic_axis_cross_term():
    # Vertex of -(u + 0.2 v)^2 - 3 v^2 is still (u, v) = (0, 0).
    grid = _grid(1.2, 2.1, -0.4, 0.5, 13, 12)
    u = grid.RR - 1.63
    v = grid.ZZ - 0.071
    psi = -((u + 0.2 * v) ** 2 + 3.0 * v**2)
    r_axis, z_axis, psi_axis = find_magnetic_axis(psi, grid, _full_mask(psi.shape))
    assert r_axis == pytest.approx(1.63, abs=1e-6)
    assert z_axis == pytest.approx(0.071, abs=1e-6)
    assert psi_axis == pytest.approx(0.0, abs=1e-6)


def test_find_magnetic_axis_positive_maximum_and_negative_minimum():
    grid = _grid(1.2, 2.1, -0.4, 0.5, 11, 9)
    quad = (grid.RR - 1.63) ** 2 + 2.0 * (grid.ZZ - 0.071) ** 2
    mask = _full_mask(quad.shape)

    # Mostly positive: the axis is the maximum, value 1.
    r_axis, z_axis, psi_axis = find_magnetic_axis(1.0 - quad, grid, mask)
    assert r_axis == pytest.approx(1.63, abs=1e-6)
    assert z_axis == pytest.approx(0.071, abs=1e-6)
    assert psi_axis == pytest.approx(1.0, abs=1e-6)

    # Mostly negative well: the axis is the minimum, value -2.
    # quad peaks near 0.7 on this box, so quad - 2 stays negative.
    r_axis, z_axis, psi_axis = find_magnetic_axis(quad - 2.0, grid, mask)
    assert r_axis == pytest.approx(1.63, abs=1e-6)
    assert z_axis == pytest.approx(0.071, abs=1e-6)
    assert psi_axis == pytest.approx(-2.0, abs=1e-6)


def test_find_magnetic_axis_linear_falls_back_to_node():
    grid = _grid(1.0, 2.0, -0.5, 0.5, 8, 7)
    psi = grid.RR.copy()
    r_axis, z_axis, psi_axis = find_magnetic_axis(psi, grid, _full_mask(psi.shape))
    assert r_axis == pytest.approx(float(grid.R[-1]))
    assert psi_axis == pytest.approx(float(grid.R[-1]))
    assert z_axis == pytest.approx(float(grid.Z[0]))


# ---------------------------------------------------------------------------
# Contours, shape error, safety factor
# ---------------------------------------------------------------------------

# Large-aspect-ratio circular equilibrium: psi = c * ((R-R0)^2 + Z^2).
_R0 = 3.0
_C = 1.0
_F0 = 4.5
_RB = 0.55


def _circular(n: int) -> tuple[Grid, np.ndarray]:
    grid = _grid(2.2, 3.8, -0.8, 0.8, n, n)
    psi = _C * ((grid.RR - _R0) ** 2 + grid.ZZ**2)
    return grid, psi


def _q_exact(r: float) -> float:
    """q = F / (2 |c| sqrt(R0^2 - r^2)) for psi = c r^2 and constant F.

    Follows from Bp = 2 |c| r / R, dl = r dθ, and
    ∫_0^{2π} dθ / (R0 + r cos θ) = 2π / sqrt(R0^2 - r^2).
    """
    return _F0 / (2.0 * abs(_C) * np.sqrt(_R0**2 - r**2))


def _q_quadrature(r: float, n: int = 200_000) -> float:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radius = _R0 + r * np.cos(theta)
    dtheta = theta[1] - theta[0]
    return _F0 / (2.0 * np.pi) * np.sum(dtheta / (2.0 * abs(_C) * radius))


def _F(psin: np.ndarray) -> np.ndarray:
    return np.full(np.shape(psin), _F0, dtype=float)


def test_q_closed_form_matches_quadrature():
    for psin in (0.2, 0.5, 0.8, 0.95):
        r = _RB * np.sqrt(psin)
        assert _q_exact(r) == pytest.approx(_q_quadrature(r), rel=1e-12)


def test_flux_contour_circle_and_shape():
    grid, psi = _circular(65)
    r_target = 0.3
    level = _C * r_target**2
    rc, zc = flux_contour(psi, grid, level, _R0, 0.0)
    assert rc[0] == pytest.approx(rc[-1])
    assert zc[0] == pytest.approx(zc[-1])
    radius = np.hypot(rc - _R0, zc)
    assert np.max(np.abs(radius - r_target)) < 1e-3

    theta = np.linspace(0.0, 2.0 * np.pi, 181)
    r_true = _R0 + r_target * np.cos(theta)
    z_true = r_target * np.sin(theta)
    assert lcfs_shape_error(rc, zc, r_true, z_true) < 1e-3


def test_flux_contour_smallest_enclosing():
    grid = _grid(2.2, 3.8, -0.8, 0.8, 81, 81)
    r2 = (grid.RR - _R0) ** 2 + grid.ZZ**2
    # psi = 0 on r = 0.2 and r = 0.4; both enclose the axis. Smallest area is r = 0.2.
    psi = (r2 - 0.2**2) * (r2 - 0.4**2)
    rc, zc = flux_contour(psi, grid, 0.0, _R0, 0.0)
    radius = np.hypot(rc - _R0, zc)
    assert np.max(np.abs(radius - 0.2)) < 2e-3
    assert np.min(np.abs(radius - 0.4)) > 0.05


def test_flux_contour_missing_raises():
    grid, psi = _circular(21)
    with pytest.raises(ValueError):
        flux_contour(psi, grid, -1.0, _R0, 0.0)


def test_lcfs_shape_error_circles():
    # Dense polylines so the chordal error is far below the 1 cm radial offset.
    theta = np.linspace(0.0, 2.0 * np.pi, 2001)
    r_in, z_in = 3.0 + 0.5 * np.cos(theta), 0.5 * np.sin(theta)
    assert lcfs_shape_error(r_in, z_in, r_in, z_in) < 1e-12

    theta_b = np.linspace(0.17, 2.0 * np.pi + 0.17, 1500)
    r_out = 3.0 + 0.51 * np.cos(theta_b)
    z_out = 0.51 * np.sin(theta_b)
    err = lcfs_shape_error(r_in, z_in, r_out, z_out)
    assert err == pytest.approx(0.01, abs=1e-4)


def test_q_profile_circular():
    levels = np.array([0.2, 0.5, 0.8, 0.95])
    exact = np.array([_q_exact(_RB * np.sqrt(p)) for p in levels])
    tolerances = {65: 1e-2, 129: 1e-3}
    for n, tol in tolerances.items():
        grid, psi = _circular(n)
        q = q_profile(psi, grid, 0.0, _C * _RB**2, _R0, 0.0, _F, levels)
        assert q.shape == levels.shape
        rel = np.max(np.abs(q - exact) / np.abs(exact))
        assert rel < tol, f"n={n} max relative q error {rel}"


def test_q95_matches_profile():
    grid, psi = _circular(65)
    psi_b = _C * _RB**2
    q_direct = q95(psi, grid, 0.0, psi_b, _R0, 0.0, _F)
    q_from_profile = q_profile(
        psi, grid, 0.0, psi_b, _R0, 0.0, _F, np.array([0.95])
    )[0]
    assert q_direct == pytest.approx(q_from_profile, rel=0.0, abs=0.0)
    assert q_direct == pytest.approx(_q_exact(_RB * np.sqrt(0.95)), rel=1e-2)


def test_q_profile_rejects_endpoints():
    grid, psi = _circular(17)
    with pytest.raises(ValueError):
        q_profile(psi, grid, 0.0, _C * _RB**2, _R0, 0.0, _F, np.array([0.0, 0.5]))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def test_time_call_positive():
    def _work():
        acc = 0
        for i in range(2000):
            acc += i
        return acc

    elapsed = time_call(_work, repeats=3)
    assert isinstance(elapsed, float)
    assert elapsed > 0.0
