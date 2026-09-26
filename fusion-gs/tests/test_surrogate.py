"""Fast tests for the params -> psi surrogate.

The Grad-Shafranov stencil is checked against ``gs.solver.delta_star_fd``.
The training test builds a 33x33 dataset with the real fixed-boundary solver.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

from gs.solver import (
    Grid,
    ProfileParams,
    ShapeParams,
    delta_star_fd,
    solve_fixed_boundary,
)
from models.surrogate import (
    FNOSurrogate,
    MLPCNNSurrogate,
    delta_star_fd_torch,
    load_surrogate,
    predict,
    train_surrogate,
)

PARAM_NAMES = ["R0", "a", "kappa", "delta", "Ip", "beta0", "alpha", "gamma", "B0"]


def _smooth_psi(grid: Grid, rng: np.random.Generator) -> np.ndarray:
    rr, zz = grid.RR, grid.ZZ
    r0, r1 = float(grid.R[0]), float(grid.R[-1])
    z0, z1 = float(grid.Z[0]), float(grid.Z[-1])
    psi = 0.15 * rr**2 + 0.04 * zz**2 + 0.01 * rr * zz
    for k in range(1, 4):
        for m in range(1, 4):
            amp = float(rng.normal())
            psi = psi + amp * np.sin(k * np.pi * (rr - r0) / (r1 - r0)) * np.cos(
                m * np.pi * (zz - z0) / (z1 - z0)
            )
    return psi


def test_delta_star_matches_solver():
    rng = np.random.default_rng(0)
    grid = Grid(np.linspace(0.9, 2.5, 41), np.linspace(-1.1, 1.3, 37))
    psi = _smooth_psi(grid, rng)
    ref = delta_star_fd(psi, grid)
    got = delta_star_fd_torch(
        torch.tensor(psi, dtype=torch.float64),
        torch.tensor(grid.R, dtype=torch.float64),
        grid.dR,
        grid.dZ,
    )
    finite = np.isfinite(ref)
    assert finite[1:-1, 1:-1].all()
    assert not finite[0].any() and not finite[-1].any()
    assert not finite[:, 0].any() and not finite[:, -1].any()
    np.testing.assert_allclose(got.numpy()[finite], ref[finite], rtol=1e-10, atol=1e-8)

    batched = got.unsqueeze(0).repeat(2, 1, 1)
    # A second copy through the batched kernel matches the single-field result.
    pair = delta_star_fd_torch(
        torch.tensor(np.stack([psi, psi]), dtype=torch.float64),
        torch.tensor(grid.R, dtype=torch.float64),
        grid.dR,
        grid.dZ,
    )
    assert torch.allclose(pair, batched)


def test_mlpcnn_forward_shape():
    net = MLPCNNSurrogate(n_params=9, nr=65, nz=65)
    y = net(torch.randn(4, 9))
    assert y.shape == (4, 65, 65)
    assert torch.isfinite(y).all()


def test_fno_forward_shape():
    net = FNOSurrogate(n_params=9, nr=65, nz=65)
    params = torch.randn(3, 9)
    R = torch.linspace(0.8, 2.76, 65)
    Z = torch.linspace(-1.9, 1.9, 65)
    mask = torch.zeros(3, 65, 65, dtype=torch.bool)
    mask[:, 12:52, 16:48] = True
    y = net(params, R, Z, mask)
    assert y.shape == (3, 65, 65)
    assert torch.isfinite(y).all()


def _write_tiny(path, n: int = 16, nr: int = 33, nz: int = 33, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    grid = Grid.uniform(0.80, 2.76, -1.90, 1.90, nr, nz)
    # Moderate shapes so every solve converges in few Picard iterations.
    R0 = rng.uniform(1.6, 1.8, size=n * 2)
    eps = rng.uniform(0.28, 0.36, size=n * 2)
    kappa = rng.uniform(1.3, 1.7, size=n * 2)
    delta = rng.uniform(0.0, 0.3, size=n * 2)
    Ip = rng.uniform(0.8e6, 1.4e6, size=n * 2)
    beta0 = rng.uniform(0.3, 0.6, size=n * 2)
    alpha = rng.uniform(1.0, 1.8, size=n * 2)
    gamma = rng.uniform(1.2, 2.2, size=n * 2)
    B0 = rng.uniform(1.8, 2.4, size=n * 2)
    kept = []
    for i in range(R0.size):
        if len(kept) >= n:
            break
        shape = ShapeParams(float(R0[i]), float(eps[i] * R0[i]), float(kappa[i]), float(delta[i]))
        profile = ProfileParams(
            float(Ip[i]), float(beta0[i]), float(alpha[i]), float(gamma[i]), float(B0[i])
        )
        eq = solve_fixed_boundary(shape, profile, grid)
        if not eq.converged or not np.isfinite(eq.psi).all() or eq.psi_axis <= 0:
            continue
        params = np.array(
            [shape.R0, shape.a, shape.kappa, shape.delta, profile.Ip, profile.beta0, profile.alpha, profile.gamma, profile.B0],
            dtype=np.float32,
        )
        kept.append(
            (
                eq.psi.astype(np.float32),
                eq.J.astype(np.float32),
                eq.mask.astype(bool),
                params,
                np.float32(eq.psi_axis),
                np.float32(eq.R_axis),
                np.float32(eq.Z_axis),
            )
        )
    if len(kept) < n:
        raise RuntimeError(f"only {len(kept)} converged solves, need {n}")
    order = rng.permutation(n)
    n_val, n_test = 2, 2
    n_train = n - n_val - n_test
    split = np.empty(n, dtype=np.int8)
    split[order[:n_train]] = 0
    split[order[n_train : n_train + n_val]] = 1
    split[order[n_train + n_val :]] = 2
    np.savez_compressed(
        path,
        R=np.asarray(grid.R, dtype=np.float64),
        Z=np.asarray(grid.Z, dtype=np.float64),
        psi=np.stack([k[0] for k in kept]),
        J=np.stack([k[1] for k in kept]),
        mask=np.stack([k[2] for k in kept]),
        params=np.stack([k[3] for k in kept]),
        param_names=np.asarray(PARAM_NAMES, dtype="U32"),
        psi_axis=np.array([k[4] for k in kept], dtype=np.float32),
        R_axis=np.array([k[5] for k in kept], dtype=np.float32),
        Z_axis=np.array([k[6] for k in kept], dtype=np.float32),
        converged=np.ones(n, dtype=bool),
        split=split,
    )


@pytest.fixture(scope="module")
def tiny_npz(tmp_path_factory):
    path = tmp_path_factory.mktemp("surr") / "tiny.npz"
    _write_tiny(path)
    return path


def test_one_epoch_loss_decreases(tiny_npz, tmp_path):
    _model, metrics = train_surrogate(
        tiny_npz,
        model="mlpcnn",
        epochs=1,
        physics_weight=0.0,
        seed=0,
        out_dir=tmp_path / "run",
        batch_size=8,
        lr=1e-3,
        patience=5,
        evaluate=False,
        num_threads=1,
    )
    assert metrics["epochs_ran"] == 1
    assert metrics["train_mse_after"] < metrics["train_mse_before"]
    assert np.isfinite(metrics["train_mse_after"])


def test_save_load_roundtrip(tiny_npz, tmp_path):
    out = tmp_path / "run"
    model, _metrics = train_surrogate(
        tiny_npz,
        model="mlpcnn",
        epochs=1,
        physics_weight=0.0,
        seed=1,
        out_dir=out,
        batch_size=8,
        lr=1e-3,
        patience=5,
        evaluate=False,
        num_threads=1,
    )
    loaded = load_surrogate(out / "checkpoint.pt")
    with np.load(tiny_npz, allow_pickle=False) as archive:
        params = np.array(archive["params"][:5])
    a = predict(model, params)
    b = predict(loaded, params)
    assert a.shape == (5, 33, 33)
    assert a.dtype == np.float32
    assert b.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all()
