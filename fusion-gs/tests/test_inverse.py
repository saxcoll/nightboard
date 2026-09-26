"""Fast checks for Green's functions, diagnostics, and InverseNet."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gs.solver import Grid, ProfileParams, ShapeParams, delta_star_fd, solve_fixed_boundary
from models.diagnostics import (
    SensorSet,
    add_noise,
    channel_rms,
    greens_br,
    greens_bz,
    greens_psi,
)
from models.inverse import InverseNet, delta_star_torch, load_inverse, make_F, reconstruct, train_inverse


def test_greens_matches_freegs():
    import freegs.machine as fm

    rng = np.random.default_rng(1)
    Rc = rng.uniform(0.6, 2.4, size=(6, 1))
    Zc = rng.uniform(-1.4, 1.4, size=(6, 1))
    R = rng.uniform(0.5, 2.6, size=(1, 5))
    Z = rng.uniform(-1.6, 1.6, size=(1, 5))
    # Keep filaments off the field points so k^2 stays away from the clip, and
    # also include one near-coincident pair where both codes clip k^2.
    psi = greens_psi(Rc, Zc, R, Z)
    psi_ref = fm.Greens(Rc, Zc, R, Z)
    br = greens_br(Rc, Zc, R, Z)
    bz = greens_bz(Rc, Zc, R, Z)
    br_ref = fm.GreensBr(Rc, Zc, R, Z)
    bz_ref = fm.GreensBz(Rc, Zc, R, Z)
    np.testing.assert_allclose(psi, psi_ref, rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(br, br_ref, rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(bz, bz_ref, rtol=1e-6, atol=0.0)

    near = greens_psi(1.2, 0.0, 1.2 + 1e-8, 1e-8)
    near_ref = fm.Greens(1.2, 0.0, 1.2 + 1e-8, 1e-8)
    np.testing.assert_allclose(near, near_ref, rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(
        greens_br(1.4, -0.2, 1.9, 0.4),
        fm.GreensBr(1.4, -0.2, 1.9, 0.4),
        rtol=1e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        greens_bz(1.4, -0.2, 1.9, 0.4),
        fm.GreensBz(1.4, -0.2, 1.9, 0.4),
        rtol=1e-6,
        atol=0.0,
    )


def _center_grid():
    return Grid.uniform(0.80, 2.76, -1.90, 1.90, 33, 33)


def test_filament_response_matches_greens():
    grid = _center_grid()
    sensors = SensorSet.for_grid(grid, n_probes=10, n_flux=6, include_normal=True)
    i, j = 16, 16
    dA = grid.dR * grid.dZ
    J = np.zeros((grid.nr, grid.nz))
    J[i, j] = 1.0 / dA
    signals = sensors.measure(J, grid)
    assert signals.shape == (1, sensors.n_sensors)
    Rc, Zc = float(grid.R[i]), float(grid.Z[j])
    br = greens_br(Rc, Zc, sensors.probe_R, sensors.probe_Z)
    bz = greens_bz(Rc, Zc, sensors.probe_R, sensors.probe_Z)
    bt = br * sensors.tangent[:, 0] + bz * sensors.tangent[:, 1]
    bn = br * sensors.normal[:, 0] + bz * sensors.normal[:, 1]
    psi = greens_psi(Rc, Zc, sensors.flux_R, sensors.flux_Z)
    n_b = sensors.n_probes
    np.testing.assert_allclose(signals[0, :n_b], bt, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(signals[0, n_b : 2 * n_b], bn, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(signals[0, 2 * n_b : 2 * n_b + sensors.n_flux], psi, rtol=1e-8, atol=0.0)
    # Outward normals point away from the vessel centroid.
    centroid = sensors.positions[:n_b].mean(axis=0)
    align = np.mean(
        sensors.normal[:, 0] * (sensors.probe_R - centroid[0])
        + sensors.normal[:, 1] * (sensors.probe_Z - centroid[1])
    )
    assert align > 0.0


def test_rogowski_equals_total_current():
    grid = _center_grid()
    sensors = SensorSet.for_grid(grid, n_probes=8, n_flux=4)
    J = np.zeros((grid.nr, grid.nz))
    J[12:20, 12:22] = 2.5e5
    dA = grid.dR * grid.dZ
    signals = sensors.measure(J, grid)
    total = float(np.sum(J) * dA)
    np.testing.assert_allclose(signals[0, -1], total, rtol=1e-12, atol=0.0)
    # A second distribution that fills every interior node still matches the
    # cell-area sum, because the whole plasma sits inside the vessel.
    inside = sensors.inside_mask(grid)
    J2 = np.zeros_like(J)
    J2[inside] = 1.0e4
    signals2 = sensors.measure(J2, grid)
    np.testing.assert_allclose(signals2[0, -1], float(np.sum(J2) * dA), rtol=1e-12, atol=0.0)


def test_coil_response_matches_freegs_control():
    import freegs.machine as fm

    grid = Grid.uniform(0.1, 2.0, -1.0, 1.0, 17, 17)
    sensors = SensorSet.for_grid(grid, n_probes=8, n_flux=4, geometry="freegs")
    names = ("P1L", "P1U", "P2L", "P2U")
    matrix = sensors.coil_response_matrix(names)
    assert matrix.shape == (sensors.n_sensors, 4)
    coils = {label: coil for label, coil in fm.TestTokamak().coils}
    # P2L is a point coil: the flux-loop rows are Green's function times turns.
    flux_slice = slice(2 * sensors.n_probes, 2 * sensors.n_probes + sensors.n_flux)
    point = coils["P2L"]
    np.testing.assert_allclose(
        matrix[flux_slice, names.index("P2L")],
        np.asarray(point.controlPsi(sensors.flux_R, sensors.flux_Z), dtype=float),
        rtol=1e-8,
        atol=0.0,
    )
    br = greens_br(point.R, point.Z, sensors.probe_R, sensors.probe_Z)
    bz = greens_bz(point.R, point.Z, sensors.probe_R, sensors.probe_Z)
    bt = br * sensors.tangent[:, 0] + bz * sensors.tangent[:, 1]
    np.testing.assert_allclose(matrix[: sensors.n_probes, names.index("P2L")], bt, rtol=1e-8, atol=0.0)
    # Shaped PF coil: the matrix row matches controlPsi, not a single filament.
    shaped = coils["P1U"]
    np.testing.assert_allclose(
        matrix[flux_slice, names.index("P1U")],
        np.asarray(shaped.controlPsi(sensors.flux_R, sensors.flux_Z), dtype=float),
        rtol=1e-8,
        atol=0.0,
    )
    # TestTokamak PF coils lie outside the wall, so the Rogowski does not count them.
    assert np.all(matrix[-1, :] == 0.0)


def test_delta_star_torch_matches_solver():
    grid = Grid.uniform(0.8, 2.2, -1.0, 1.0, 21, 19)
    rng = np.random.default_rng(0)
    psi = rng.normal(size=(3, grid.nr, grid.nz))
    ref = np.stack([delta_star_fd(psi[i], grid)[1:-1, 1:-1] for i in range(3)])
    got = delta_star_torch(
        torch.as_tensor(psi, dtype=torch.float64),
        torch.as_tensor(np.asarray(grid.R, dtype=np.float64)),
        float(grid.dR),
        float(grid.dZ),
    ).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


def test_F_matches_equilibrium_profile():
    grid = Grid.uniform(0.80, 2.76, -1.90, 1.90, 33, 33)
    eq = solve_fixed_boundary(
        ShapeParams(1.7, 0.55, 1.6, 0.3),
        ProfileParams(1.0e6, 0.45, 1.4, 1.8, 2.2),
        grid,
    )
    names = ["R0", "a", "kappa", "delta", "Ip", "beta0", "alpha", "gamma", "B0"]
    row = np.array([1.7, 0.55, 1.6, 0.3, 1.0e6, 0.45, 1.4, 1.8, 2.2])
    F = make_F(row, names, grid, eq.psi, eq.mask, eq.psi_axis)
    assert F is not None
    for psin in (0.05, 0.5, 0.95, 1.0):
        got = float(np.asarray(F(psin)))
        ref = float(eq.F(psin))
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_inverse_net_forward_shape():
    net = InverseNet(n_sensors=11, nr=17, nz=19, n_aux=7, n_coils=2, latent=32, hidden=40, channels=8, base=8)
    signals = torch.randn(5, 11)
    coils = torch.randn(5, 2)
    psi, aux = net(signals, coils)
    assert psi.shape == (5, 17, 19)
    assert aux.shape == (5, 7)
    assert torch.isfinite(psi).all()
    assert torch.isfinite(aux).all()
    # Zero-initialized heads predict a vanishing flux and the aux mean (0 here).
    assert torch.max(torch.abs(psi)) == 0.0
    assert torch.max(torch.abs(aux)) == 0.0


def _write_tiny(path, n=16, nr=33, nz=33, seed=0):
    rng = np.random.default_rng(seed)
    grid = Grid.uniform(0.80, 2.76, -1.90, 1.90, nr, nz)
    rows = []
    attempts = 0
    while len(rows) < n and attempts < n * 6:
        attempts += 1
        R0 = float(rng.uniform(1.55, 1.85))
        eps = float(rng.uniform(0.28, 0.36))
        kappa = float(rng.uniform(1.3, 1.8))
        delta = float(rng.uniform(0.05, 0.4))
        Ip = float(rng.uniform(0.7e6, 1.6e6))
        beta0 = float(rng.uniform(0.3, 0.7))
        alpha = float(rng.uniform(1.0, 2.0))
        gamma = float(rng.uniform(1.2, 2.5))
        B0 = float(rng.uniform(1.7, 2.6))
        try:
            eq = solve_fixed_boundary(
                ShapeParams(R0, eps * R0, kappa, delta),
                ProfileParams(Ip, beta0, alpha, gamma, B0),
                grid,
            )
        except Exception:
            continue
        if not eq.converged or not np.isfinite(eq.psi_axis) or eq.psi_axis <= 0.0:
            continue
        rows.append(
            (
                np.asarray(eq.psi, dtype=np.float32),
                np.asarray(eq.J, dtype=np.float32),
                np.asarray(eq.mask, dtype=bool),
                np.asarray([R0, eps * R0, kappa, delta, Ip, beta0, alpha, gamma, B0], dtype=np.float32),
                np.float32(eq.psi_axis),
                np.float32(eq.R_axis),
                np.float32(eq.Z_axis),
            )
        )
    if len(rows) < n:
        raise RuntimeError(f"only {len(rows)} tiny equilibria converged")
    split = np.zeros(n, dtype=np.int8)
    split[-4:-2] = 1
    split[-2:] = 2
    np.savez_compressed(
        path,
        R=np.asarray(grid.R, dtype=np.float64),
        Z=np.asarray(grid.Z, dtype=np.float64),
        psi=np.stack([row[0] for row in rows]),
        J=np.stack([row[1] for row in rows]),
        mask=np.stack([row[2] for row in rows]),
        params=np.stack([row[3] for row in rows]),
        param_names=np.asarray(
            ["R0", "a", "kappa", "delta", "Ip", "beta0", "alpha", "gamma", "B0"], dtype="U32"
        ),
        psi_axis=np.array([row[4] for row in rows], dtype=np.float32),
        R_axis=np.array([row[5] for row in rows], dtype=np.float32),
        Z_axis=np.array([row[6] for row in rows], dtype=np.float32),
        converged=np.ones(n, dtype=bool),
        split=split,
    )


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny") / "dev_tiny.npz"
    _write_tiny(path)
    return path


def test_one_epoch_loss_decreases(tiny_dataset, tmp_path):
    _model, metrics = train_inverse(
        tiny_dataset,
        noise=0.0,
        epochs=1,
        seed=0,
        out_dir=tmp_path / "run",
        batch_size=16,
        evaluate=False,
        verbose=False,
        n_probes=8,
        n_flux=4,
        latent=32,
        hidden=48,
        physics_weight=0.05,
    )
    losses = np.asarray(metrics["train_loss"], dtype=float)
    assert losses.shape == (1,)
    assert np.isfinite(losses).all()
    assert np.isfinite(metrics["loss_initial"])
    assert losses[0] < metrics["loss_initial"]


def test_save_load_roundtrip(tiny_dataset, tmp_path):
    model, _metrics = train_inverse(
        tiny_dataset,
        noise=0.0,
        epochs=1,
        seed=1,
        out_dir=tmp_path / "run",
        batch_size=16,
        evaluate=False,
        verbose=False,
        n_probes=8,
        n_flux=4,
        latent=32,
        hidden=48,
    )
    loaded = load_inverse(tmp_path / "run" / "inverse.pt")
    rng = np.random.default_rng(2)
    signals = rng.normal(size=(3, model.n_sensors)).astype(np.float32)
    # Put signals on the scale of the stored normalization so the output is finite.
    signals = signals * loaded.sensor_std.numpy() + loaded.sensor_mean.numpy()
    a = reconstruct(model, signals)
    b = reconstruct(loaded, signals)
    assert a.shape == (3, model.nr, model.nz)
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8)
    assert np.isfinite(b).all()


def test_around_plasmas_encloses_mask_and_excludes_coils():
    grid = Grid.uniform(0.1, 2.0, -1.0, 1.0, 33, 33)
    mask = np.zeros((grid.nr, grid.nz), dtype=bool)
    mask[8:22, 10:24] = True
    sensors = SensorSet.around_plasmas(grid, mask, n_probes=8, n_flux=4, margin_cells=1)
    inside = sensors.inside_mask(grid)
    assert np.all(inside[mask])
    ii = np.argmin(np.abs(grid.R[:, None] - sensors.probe_R[None, :]), axis=0)
    jj = np.argmin(np.abs(grid.Z[:, None] - sensors.probe_Z[None, :]), axis=0)
    assert not np.any(mask[ii, jj])
    path_R = sensors.vessel_R
    path_Z = sensors.vessel_Z
    from matplotlib.path import Path

    path = Path(np.column_stack([np.append(path_R, path_R[0]), np.append(path_Z, path_Z[0])]))
    # PF coils of TestTokamak are far from this blob.
    assert not path.contains_point((1.75, 0.6))
    assert not path.contains_point((1.75, -0.6))


def test_noise_scales_with_rms():
    rng = np.random.default_rng(4)
    clean = rng.normal(size=(200, 3)) * np.array([1.0, 10.0, 1.0e6])
    rms = channel_rms(clean)
    noisy = add_noise(clean, 0.01 * rms, np.random.default_rng(5))
    realized = np.std(noisy - clean, axis=0)
    np.testing.assert_allclose(realized, 0.01 * rms, rtol=0.25)


@pytest.mark.skipif(
    not __import__("pathlib").Path("/workspace/fusion-gs/data/fixed_65.npz").is_file(),
    reason="fixed_65.npz is not built",
)
def test_vessel_encloses_fixed_plasmas():
    from pathlib import Path

    sensors = None
    grid = None
    for name in ("fixed_65.npz", "fixed_65_ood.npz"):
        path = Path("/workspace/fusion-gs/data") / name
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            R = np.array(archive["R"])
            Z = np.array(archive["Z"])
            union = np.array(archive["mask"]).any(axis=0)
        grid = Grid(R, Z)
        if sensors is None:
            sensors = SensorSet.for_grid(grid, n_probes=40, n_flux=20)
        inside = sensors.inside_mask(grid)
        assert int((union & ~inside).sum()) == 0, name
