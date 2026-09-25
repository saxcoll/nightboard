"""Fast checks that equilibrium, comparison, and sweep figures are written."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from eval.visualize import compare_figure, equilibrium_figure, shape_sweep_animation
from gs.solver import Grid, ProfileParams, ShapeParams, solve_fixed_boundary

_SHAPE = ShapeParams(1.7, 0.6, 1.7, 0.3)
_PROFILE = ProfileParams(1.0e6, 0.5, 1.0, 2.0, 2.0)


def _small_grid(n: int = 37) -> Grid:
    return Grid.uniform(0.95, 2.55, -1.25, 1.25, n, n)


def test_equilibrium_figure_writes_png(tmp_path):
    eq = solve_fixed_boundary(_SHAPE, _PROFILE, _small_grid())
    out = tmp_path / "equilibrium.png"
    equilibrium_figure(eq, out)
    raw = out.read_bytes()
    assert raw.startswith(b"\x89PNG")
    assert len(raw) > 5_000


def test_compare_figure_writes_png(tmp_path):
    eq = solve_fixed_boundary(_SHAPE, _PROFILE, _small_grid())
    # A scaled flux keeps a magnetic axis and a non-zero, finite relative L2.
    pred = np.asarray(eq.psi, dtype=float) * 0.97
    out = tmp_path / "compare.png"
    path, rel, axis_err = compare_figure(eq, pred, "scaled", out)
    raw = path.read_bytes()
    assert raw.startswith(b"\x89PNG")
    assert len(raw) > 5_000
    assert np.isfinite(rel) and rel > 0.0
    assert np.isfinite(axis_err)


def test_shape_sweep_writes_gif(tmp_path):
    out = tmp_path / "kappa.gif"
    grid = Grid.uniform(0.90, 2.60, -1.45, 1.45, 29, 29)
    shape_sweep_animation(
        out,
        param="kappa",
        values=np.array([1.3, 1.6, 1.9]),
        shape=_SHAPE,
        profile=_PROFILE,
        grid=grid,
        fps=8,
        dpi=60,
    )
    raw = out.read_bytes()
    assert raw[:6] in (b"GIF87a", b"GIF89a")
    assert len(raw) > 1_000
