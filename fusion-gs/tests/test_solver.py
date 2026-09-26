"""Finite-difference Grad-Shafranov solver."""
from __future__ import annotations

import numpy as np

from gs.analytic import solovev
from gs.solver import (
    MU0,
    Grid,
    ProfileParams,
    ShapeParams,
    delta_star_fd,
    solve_fixed_boundary,
    solve_linear,
)


def _l2(a):
    return float(np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2)))


def test_delta_star_fd_exact_on_quadratic():
    grid = Grid.uniform(1.0, 2.0, -0.5, 0.5, 21, 31)
    RR, ZZ = grid.RR, grid.ZZ
    # No odd power of R: the conservative midpoint stencil is exact on this quadratic.
    psi = 0.2 + 0.5 * ZZ - 1.3 * RR**2 + 0.7 * ZZ**2
    op = delta_star_fd(psi, grid)
    assert np.isnan(op[0, :]).all() and np.isnan(op[-1, :]).all()
    assert np.isnan(op[:, 0]).all() and np.isnan(op[:, -1]).all()
    # Δ*(a + b Z + c R^2 + d Z^2) = 2c - (1/R)(2c R) + 2d = 2d
    np.testing.assert_allclose(op[1:-1, 1:-1], 1.4, atol=1e-10)


def test_rectangle_convergence():
    def exact(R, Z):
        return np.sin(np.pi * (R - 1.0)) * np.cos(np.pi * Z) + 0.15 * R

    def rhs_of(R, Z):
        s = np.sin(np.pi * (R - 1.0))
        c = np.cos(np.pi * (R - 1.0))
        cz = np.cos(np.pi * Z)
        return -2.0 * np.pi**2 * s * cz - (np.pi * c * cz) / R - 0.15 / R

    def error(nr, nz):
        grid = Grid.uniform(1.0, 2.0, -0.5, 0.5, nr, nz)
        rhs = rhs_of(grid.RR, grid.ZZ)
        psi = solve_linear(grid, rhs, level_set=None, boundary_value=exact)
        # The discrete operator on the numerical solution reproduces rhs.
        op = delta_star_fd(psi, grid)
        np.testing.assert_allclose(op[1:-1, 1:-1], rhs[1:-1, 1:-1], atol=1e-8, rtol=1e-8)
        err = psi[1:-1, 1:-1] - exact(grid.RR, grid.ZZ)[1:-1, 1:-1]
        return _l2(err)

    e_coarse = error(33, 33)
    e_fine = error(65, 65)
    ratio = e_coarse / e_fine
    assert 3.5 < ratio < 4.5, f"expected ~4, got {ratio} (errors {e_coarse}, {e_fine})"


def _rel_l2(psi, exact, mask):
    diff = psi[mask] - exact[mask]
    return float(np.linalg.norm(diff) / np.linalg.norm(exact[mask]))


def test_solovev_shaped_boundary_convergence():
    eq = solovev(0.32, 1.7, 0.33, -0.155, R0=1.0)

    def run(n):
        grid = Grid.uniform(0.45, 1.55, -0.85, 0.85, n, n)
        rhs = eq.gs_rhs(grid.RR, grid.ZZ)
        psi = solve_linear(grid, rhs, level_set=eq.level_set, boundary_value=0.0)
        mask = np.asarray(eq.level_set(grid.RR, grid.ZZ), dtype=float) < 0.0
        exact = np.asarray(eq.psi(grid.RR, grid.ZZ), dtype=float)
        # Interior residual of the conservative operator, away from the cut cells.
        inner = mask.copy()
        inner[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
        op = delta_star_fd(psi, grid)
        rel_res = np.linalg.norm((op - rhs)[inner]) / np.linalg.norm(rhs[inner])
        assert rel_res < 1e-8
        return _rel_l2(psi, exact, mask)

    e65 = run(65)
    e129 = run(129)
    ratio = e65 / e129
    assert e129 < 1e-3, f"fine-grid relative L2 {e129}"
    assert ratio > 3.0, f"expected error drop > 3, got {ratio} ({e65} -> {e129})"


def test_nonlinear_fixed_boundary():
    shape = ShapeParams(R0=1.7, a=0.6, kappa=1.7, delta=0.3)
    profile = ProfileParams(Ip=1.0e6, beta0=0.5, alpha=1.0, gamma=2.0, B0=2.0)
    grid = Grid.uniform(0.90, 2.55, -1.25, 1.25, 65, 65)
    eq = solve_fixed_boundary(shape, profile, grid)

    assert eq.converged
    assert eq.n_iter < 200
    assert eq.psi_axis > 0.0
    assert eq.psi_boundary == 0.0
    assert abs(eq.Z_axis) < grid.dZ
    assert shape.level_set()(eq.R_axis, eq.Z_axis) < 0.0
    assert eq.R_axis > shape.R0 - shape.a
    assert eq.R_axis < shape.R0 + shape.a

    total = float(np.sum(eq.J) * grid.dR * grid.dZ)
    assert abs(total - profile.Ip) / profile.Ip < 1e-6

    op = delta_star_fd(eq.psi, grid)
    rhs = -MU0 * grid.RR * eq.J
    mask = eq.mask
    inner = np.zeros_like(mask)
    inner[2:-2, 2:-2] = (
        mask[2:-2, 2:-2]
        & mask[1:-3, 2:-2]
        & mask[3:-1, 2:-2]
        & mask[2:-2, 1:-3]
        & mask[2:-2, 3:-1]
    )
    rel = np.linalg.norm((op - rhs)[inner]) / np.linalg.norm(rhs[inner])
    assert rel < 1e-2, f"interior GS residual {rel}"

    assert eq.pressure(0.0) > 0.0
    assert abs(eq.pressure(1.0)) < 1e-8 * eq.pressure(0.0)
    assert abs(eq.F(1.0) - shape.R0 * profile.B0) < 1e-8 * shape.R0 * profile.B0
    # Spot-check the psi-integral identity at the axis for this (alpha, gamma).
    # int_1^0 (1-s)^2 ds = -1/3, p(0) = (psi_b - psi_axis) * p'(scale) * that.
    expected_p0 = (eq.psi_boundary - eq.psi_axis) * (eq.lam * profile.beta0 / shape.R0) * (-1.0 / 3.0)
    assert abs(eq.pressure(0.0) - expected_p0) / expected_p0 < 1e-4


def test_shape_level_set_matches_solovev_A0():
    shape = ShapeParams(R0=1.7, a=0.6, kappa=1.7, delta=0.3)
    level = shape.level_set()
    ref = solovev(epsilon=shape.a / shape.R0, kappa=shape.kappa, delta=shape.delta, A=0.0, R0=shape.R0)
    R = np.array([1.7, 1.2, 2.2, 1.7])
    Z = np.array([0.0, 0.2, -0.3, 2.5])
    np.testing.assert_allclose(level(R, Z), ref.level_set(R, Z), atol=1e-12)
    assert level(shape.R0, 0.0) < 0.0
