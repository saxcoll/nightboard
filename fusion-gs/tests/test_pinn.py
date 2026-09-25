"""PINN Grad-Shafranov tests.

Fast tests check the autograd operator, the ray boundary finder, and that one
optimizer step produces a finite loss. The slow test trains the Solov'ev case.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gs.analytic import solovev
from gs.solver import Grid, ShapeParams
from models.pinn import delta_star_autograd, predict_grid, ray_boundary_points, train_pinn


class _Poly(torch.nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, R, Z, cond=None):
        if self.kind == "r4":
            return R**4 / 8.0
        if self.kind == "r2z2":
            return (R**2) * (Z**2)
        raise AssertionError(self.kind)


def test_delta_star_autograd_polynomials():
    R = torch.linspace(0.7, 1.8, 12, dtype=torch.float64)
    Z = torch.linspace(-0.5, 0.6, 9, dtype=torch.float64)
    RR, ZZ = torch.meshgrid(R, Z, indexing="ij")
    Rf = RR.reshape(-1)
    Zf = ZZ.reshape(-1)

    ds = delta_star_autograd(_Poly("r4"), Rf, Zf)
    # Delta*(R^4/8) = R^2
    np.testing.assert_allclose(ds.detach().numpy(), (Rf**2).numpy(), rtol=1e-10, atol=1e-10)

    ds2 = delta_star_autograd(_Poly("r2z2"), Rf, Zf)
    # Delta*(R^2 Z^2) = 2 R^2
    np.testing.assert_allclose(ds2.detach().numpy(), (2.0 * Rf**2).numpy(), rtol=1e-10, atol=1e-10)


def test_ray_boundary_on_level_set():
    eq = solovev(0.32, 1.7, 0.33, -0.155, R0=1.0)
    Rb, Zb = ray_boundary_points(eq.level_set, (1.0, 0.0), 64)
    vals = np.asarray(eq.level_set(Rb, Zb), dtype=float)
    assert Rb.shape == (64,) and Zb.shape == (64,)
    assert np.isfinite(Rb).all() and np.isfinite(Zb).all()
    assert np.max(np.abs(vals)) < 1e-8

    shape = ShapeParams(1.7, 0.6, 1.7, 0.3)
    Rb2, Zb2 = ray_boundary_points(shape.level_set(), (1.7, 0.0), 48)
    vals2 = np.asarray(shape.level_set()(Rb2, Zb2), dtype=float)
    assert np.max(np.abs(vals2)) < 1e-8
    # Rays from the geometric centre hit both the high-field and low-field sides.
    assert Rb2.min() < 1.7 < Rb2.max()
    assert Zb2.min() < 0.0 < Zb2.max()


def test_one_training_step_finite():
    model, metrics = train_pinn("solovev", steps=1, seed=0, verbose=False)
    assert np.isfinite(metrics["final_loss"])
    assert metrics["final_loss"] > 0.0
    assert metrics["steps"] >= 1
    grid = Grid.uniform(0.6, 1.4, -0.4, 0.4, 8, 9)
    psi = predict_grid(model, grid)
    assert psi.shape == (8, 9)
    assert np.isfinite(psi).all()


@pytest.mark.slow
def test_solovev_rel_l2_below_threshold():
    model, metrics = train_pinn("solovev", steps=1200, seed=0, verbose=False)
    assert np.isfinite(metrics["rel_l2"])
    assert metrics["rel_l2"] < 1e-2, metrics
    grid = Grid.uniform(0.45, 1.55, -0.85, 0.85, 32, 32)
    pred = predict_grid(model, grid)
    assert pred.shape == (32, 32)
