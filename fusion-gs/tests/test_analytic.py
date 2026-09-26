"""Cerfon-Freidberg Solov'ev equilibria."""
from __future__ import annotations

import numpy as np
import pytest

from gs.analytic import solovev


def _fd_delta_star(eq, R, Z, h=1e-3):
    """Fourth-order finite-difference Delta* of eq.psi."""
    p = eq.psi

    def d2(f0, fp, fm, fpp, fmm):
        return (-fmm + 16.0 * fm - 30.0 * f0 + 16.0 * fp - fpp) / (12.0 * h**2)

    def d1(fp, fm, fpp, fmm):
        return (fmm - 8.0 * fm + 8.0 * fp - fpp) / (12.0 * h)

    d2R = d2(p(R, Z), p(R + h, Z), p(R - h, Z), p(R + 2 * h, Z), p(R - 2 * h, Z))
    d2Z = d2(p(R, Z), p(R, Z + h), p(R, Z - h), p(R, Z + 2 * h), p(R, Z - 2 * h))
    dR = d1(p(R + h, Z), p(R - h, Z), p(R + 2 * h, Z), p(R - 2 * h, Z))
    return d2R - dR / R + d2Z


def _curvature_numbers(eq):
    alpha = float(np.arcsin(eq.delta))
    n1 = -((1.0 + alpha) ** 2) / (eq.epsilon * eq.kappa**2)
    n2 = ((1.0 - alpha) ** 2) / (eq.epsilon * eq.kappa**2)
    n3 = -eq.kappa / (eq.epsilon * np.cos(alpha) ** 2)
    return n1, n2, n3


def _derivs(eq, R, Z, h=1e-4):
    p = eq.psi
    px = (p(R + h, Z) - p(R - h, Z)) / (2 * h)
    py = (p(R, Z + h) - p(R, Z - h)) / (2 * h)
    pxx = (p(R + h, Z) - 2 * p(R, Z) + p(R - h, Z)) / h**2
    pyy = (p(R, Z + h) - 2 * p(R, Z) + p(R, Z - h)) / h**2
    return px, py, pxx, pyy


@pytest.mark.parametrize(
    "eps,kappa,delta,A,R0",
    [
        (0.32, 1.7, 0.33, -0.155, 1.0),
        (0.78, 2.0, 0.35, 0.0, 1.0),
        (0.32, 1.7, 0.33, -0.155, 2.5),
    ],
)
def test_pde_residual(eps, kappa, delta, A, R0):
    eq = solovev(eps, kappa, delta, A, R0=R0)
    pts = [
        (R0, 0.0),
        (R0 * (1.0 + 0.2 * eps), 0.1 * R0),
        (R0 * (1.0 - 0.15 * eps), -0.25 * eps * kappa * R0),
        (R0 * (1.0 - 0.5 * delta * eps), 0.4 * eps * kappa * R0),
    ]
    for R, Z in pts:
        residual = _fd_delta_star(eq, R, Z) - float(eq.gs_rhs(R, Z))
        assert abs(residual) < 1e-8


@pytest.mark.parametrize(
    "eps,kappa,delta,A",
    [
        (0.32, 1.7, 0.33, -0.155),
        (0.78, 2.0, 0.35, 0.0),
    ],
)
def test_boundary_conditions(eps, kappa, delta, A):
    eq = solovev(eps, kappa, delta, A, R0=1.0)
    n1, n2, n3 = _curvature_numbers(eq)
    outer = (1.0 + eps, 0.0)
    inner = (1.0 - eps, 0.0)
    top = (1.0 - delta * eps, kappa * eps)
    for R, Z in (outer, inner, top):
        assert abs(eq.psi(R, Z)) < 1e-10
    px, _py, _pxx, pyy = _derivs(eq, *outer)
    assert abs(pyy + n1 * px) < 1e-6
    px, _py, _pxx, pyy = _derivs(eq, *inner)
    assert abs(pyy + n2 * px) < 1e-6
    px, py, pxx, _pyy = _derivs(eq, *top)
    assert abs(px) < 1e-8
    assert abs(pxx + n3 * py) < 1e-5


def test_psi_negative_inside_and_small_on_miller_boundary():
    eq = solovev(0.32, 1.7, 0.33, -0.155, R0=1.0)
    assert eq.psi(eq.R0, 0.0) < 0.0
    Rb, Zb = eq.boundary(360)
    pb = eq.psi(Rb, Zb)
    # The psi=0 contour tracks the Miller target; mismatch is far below the axis value.
    assert np.max(np.abs(pb)) / abs(eq.psi(eq.R0, 0.0)) < 1e-3
    nstx = solovev(0.78, 2.0, 0.35, 0.0, R0=1.0)
    Rb, Zb = nstx.boundary(360)
    assert np.max(np.abs(nstx.psi(Rb, Zb))) / abs(nstx.psi(1.0, 0.0)) < 2e-3


def test_grad_psi_matches_finite_difference():
    eq = solovev(0.32, 1.7, 0.33, -0.155, R0=1.8)
    R, Z = 1.9, 0.15
    h = 1e-6
    dR = (eq.psi(R + h, Z) - eq.psi(R - h, Z)) / (2 * h)
    dZ = (eq.psi(R, Z + h) - eq.psi(R, Z - h)) / (2 * h)
    gR, gZ = eq.grad_psi(R, Z)
    assert abs(gR - dR) < 1e-8
    assert abs(gZ - dZ) < 1e-8


def test_level_set_box_excludes_far_field_negatives():
    eq = solovev(0.32, 1.7, 0.33, -0.155, R0=1.0)
    assert eq.level_set(eq.R0, 0.0) < 0.0
    assert abs(eq.level_set(eq.R0, 0.0) - eq.psi(eq.R0, 0.0)) < 1e-14
    # Inside the padded box, level_set equals psi (including a boundary sample).
    Rb, Zb = eq.boundary(64)
    assert np.allclose(eq.level_set(Rb, Zb), eq.psi(Rb, Zb), atol=1e-12)
    # Far above the plasma the raw polynomial is negative, but the level set is not.
    far_R, far_Z = 0.5, 3.0
    assert eq.psi(far_R, far_Z) < 0.0
    assert eq.level_set(far_R, far_Z) > 0.0
    # Just outside the outer midplane point the level set (and psi) is positive,
    # and a ray from the axis crosses zero once.
    assert eq.level_set(1.0 + eq.epsilon + 0.02, 0.0) > 0.0


@pytest.mark.parametrize(
    "eps,kappa,delta",
    [
        (0.25, 1.2, 0.0),
        (0.25, 1.2, 0.5),
        (0.25, 2.0, 0.0),
        (0.25, 2.0, 0.5),
        (0.40, 1.2, 0.0),
        (0.40, 1.2, 0.5),
        (0.40, 2.0, 0.0),
        (0.40, 2.0, 0.5),
    ],
)
def test_shape_corners_closed_negative_core(eps, kappa, delta):
    """A=0 Solov'ev shapes used by ShapeParams stay a single closed psi<0 region."""
    eq = solovev(eps, kappa, delta, A=0.0, R0=1.0)
    assert eq.psi(1.0, 0.0) < 0.0
    Rb, Zb = eq.boundary(180)
    pb = eq.psi(Rb, Zb)
    assert np.max(np.abs(pb)) / abs(eq.psi(1.0, 0.0)) < 5e-3
    # One sign change along several rays from the magnetic centre to the box edge.
    pad = 0.1 * eps
    xmin, xmax = 1.0 - eps - pad, 1.0 + eps + pad
    ymin, ymax = -(kappa * eps + pad), kappa * eps + pad
    for theta in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
        dx, dy = np.cos(theta), np.sin(theta)
        cands = []
        if abs(dx) > 1e-14:
            cands.extend(t for t in ((xmin - 1.0) / dx, (xmax - 1.0) / dx) if t > 0)
        if abs(dy) > 1e-14:
            cands.extend(t for t in (ymin / dy, ymax / dy) if t > 0)
        tmax = min(cands)
        t = np.linspace(0.0, tmax, 400)
        vals = eq.level_set(1.0 + dx * t, dy * t)
        signs = np.sign(vals)
        signs[signs == 0.0] = 1.0
        assert np.sum(signs[1:] * signs[:-1] < 0) == 1
