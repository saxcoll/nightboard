"""Dataset format and a small end-to-end generation check."""
from __future__ import annotations

import numpy as np

from data.freegs_gen import FREEGS_PARAM_NAMES, generate_freegs
from data.generate import (
    FIXED_RMAX,
    FIXED_RMIN,
    FIXED_ZMAX,
    FIXED_ZMIN,
    PARAM_NAMES,
    assign_splits,
    generate_fixed,
    load_dataset,
    sample_fixed_params,
)


def test_sample_ranges_and_splits():
    params = sample_fixed_params(64, seed=0, ood=False)
    R0, a, kappa, delta, Ip, beta0, alpha, gamma, B0 = params.T
    assert np.all((R0 >= 1.5) & (R0 <= 1.9))
    eps = a / R0
    assert np.all((eps >= 0.25 - 1e-12) & (eps <= 0.40 + 1e-12))
    assert np.all((kappa >= 1.2) & (kappa <= 2.0))
    assert np.all((delta >= 0.0) & (delta <= 0.5))
    assert np.all((Ip >= 0.5e6) & (Ip <= 2.0e6))
    assert np.all((beta0 >= 0.2) & (beta0 <= 0.8))
    assert np.all((alpha >= 0.8) & (alpha <= 2.5))
    assert np.all((gamma >= 1.0) & (gamma <= 3.0))
    assert np.all((B0 >= 1.5) & (B0 <= 3.0))

    ood = sample_fixed_params(32, seed=0, ood=True)
    assert np.all(ood[:, 2] > 2.0) and np.all(ood[:, 2] <= 2.3)
    # Independent of the in-distribution stream.
    assert not np.allclose(ood[:32, 0], params[:32, 0])

    splits = assign_splits(8, seed=0, ood=False)
    assert splits.dtype == np.int8
    assert dict(zip(*np.unique(splits, return_counts=True))) == {0: 6, 1: 1, 2: 1}
    assert np.all(assign_splits(5, seed=1, ood=True) == 2)


def test_fixed_dataset_contract(tmp_path):
    out = tmp_path / "fixed_tiny.npz"
    info = generate_fixed(n=8, nr=33, nz=33, seed=0, out=out, workers=2)
    assert info["n_kept"] == 8
    assert info["n_failed"] == 0
    data = load_dataset(str(out))

    expected = [
        "R",
        "Z",
        "psi",
        "J",
        "mask",
        "params",
        "param_names",
        "psi_axis",
        "R_axis",
        "Z_axis",
        "converged",
        "split",
    ]
    assert list(data) == expected
    n = data["psi"].shape[0]
    assert n == 8
    assert data["R"].shape == (33,) and data["Z"].shape == (33,)
    assert data["R"].dtype == np.float64 and data["Z"].dtype == np.float64
    assert data["R"][0] == FIXED_RMIN and data["R"][-1] == FIXED_RMAX
    assert data["Z"][0] == FIXED_ZMIN and data["Z"][-1] == FIXED_ZMAX
    assert np.all(np.diff(data["R"]) > 0) and np.all(np.diff(data["Z"]) > 0)
    assert data["psi"].dtype == np.float32 and data["psi"].shape == (n, 33, 33)
    assert data["J"].dtype == np.float32 and data["J"].shape == (n, 33, 33)
    assert data["mask"].dtype == np.bool_ and data["mask"].shape == (n, 33, 33)
    assert data["params"].dtype == np.float32 and data["params"].shape == (n, 9)
    assert [str(x) for x in data["param_names"]] == PARAM_NAMES
    assert data["psi_axis"].dtype == np.float32 and data["psi_axis"].shape == (n,)
    assert data["R_axis"].shape == (n,) and data["Z_axis"].shape == (n,)
    assert data["converged"].dtype == np.bool_ and np.all(data["converged"])
    assert data["split"].dtype == np.int8
    assert np.array_equal(data["split"], assign_splits(n, seed=0, ood=False))

    assert np.all(data["psi_axis"] > 0.0)
    dR = float(data["R"][1] - data["R"][0])
    dZ = float(data["Z"][1] - data["Z"][0])
    names = PARAM_NAMES
    for i in range(n):
        mask = data["mask"][i]
        psi = data["psi"][i]
        assert mask.any()
        assert float(np.min(psi[mask])) >= -1e-4 * float(data["psi_axis"][i])
        Ip = float(data["params"][i, names.index("Ip")])
        current = float(np.sum(data["J"][i], dtype=np.float64) * dR * dZ)
        assert abs(current - Ip) / Ip < 1e-5
        assert abs(float(data["Z_axis"][i])) < 5.0 * dZ

    p = data["params"]
    R0, a, kappa, delta = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    Ip, beta0, alpha, gamma, B0 = p[:, 4], p[:, 5], p[:, 6], p[:, 7], p[:, 8]
    assert np.all((R0 >= 1.5) & (R0 <= 1.9))
    assert np.all((a / R0 >= 0.25 - 1e-5) & (a / R0 <= 0.40 + 1e-5))
    assert np.all((kappa >= 1.2) & (kappa <= 2.0))
    assert np.all((delta >= 0.0) & (delta <= 0.5))
    assert np.all((Ip >= 0.5e6) & (Ip <= 2.0e6))
    assert np.all((beta0 >= 0.2) & (beta0 <= 0.8))
    assert np.all((alpha >= 0.8) & (alpha <= 2.5))
    assert np.all((gamma >= 1.0) & (gamma <= 3.0))
    assert np.all((B0 >= 1.5) & (B0 <= 3.0))


def test_ood_dataset_is_all_test(tmp_path):
    out = tmp_path / "ood_tiny.npz"
    info = generate_fixed(n=4, nr=33, nz=33, seed=1, out=out, ood=True, workers=2)
    assert info["n_failed"] == 0 and info["n_kept"] == 4
    data = load_dataset(str(out))
    assert data["R"][0] == FIXED_RMIN and data["Z"][-1] == FIXED_ZMAX
    assert np.all(data["split"] == 2)
    kappa = data["params"][:, PARAM_NAMES.index("kappa")]
    assert np.all(kappa > 2.0) and np.all(kappa <= 2.3)
    assert np.all(data["psi_axis"] > 0.0)
    assert np.all(data["converged"])


def test_freegs_smoke(tmp_path):
    """One FreeGS sample on a 33x33 grid (about a second, under the slow mark)."""
    out = tmp_path / "freegs_tiny.npz"
    info = generate_freegs(n=1, out=out, seed=0, nx=33, ny=33, workers=1)
    assert info["n_kept"] == 1, info
    data = load_dataset(str(out))
    for key in (
        "R",
        "Z",
        "psi",
        "J",
        "mask",
        "params",
        "param_names",
        "psi_axis",
        "R_axis",
        "Z_axis",
        "converged",
        "split",
        "coil_names",
        "coil_currents",
        "q95",
    ):
        assert key in data
    assert data["psi"].shape == (1, 33, 33)
    assert data["psi"].dtype == np.float32
    assert data["mask"].dtype == np.bool_
    assert [str(x) for x in data["param_names"]] == FREEGS_PARAM_NAMES
    assert [str(x) for x in data["coil_names"]] == ["P1L", "P1U", "P2L", "P2U"]
    assert data["coil_currents"].shape == (1, 4)
    assert data["q95"].shape == (1,)
    assert np.isfinite(data["q95"][0]) and float(data["q95"][0]) > 0.0
    assert float(data["psi_axis"][0]) > 0.0
    mask = data["mask"][0]
    psi = data["psi"][0]
    assert mask.any()
    assert float(np.min(psi[mask])) >= -0.05 * float(data["psi_axis"][0])
    dR = float(data["R"][1] - data["R"][0])
    dZ = float(data["Z"][1] - data["Z"][0])
    Ip = float(data["params"][0, FREEGS_PARAM_NAMES.index("Ip")])
    current = float(np.sum(data["J"][0], dtype=np.float64) * dR * dZ)
    assert abs(current - Ip) / Ip < 0.05
    assert np.all(data["converged"])
    assert set(data["split"].tolist()).issubset({0, 1, 2})
