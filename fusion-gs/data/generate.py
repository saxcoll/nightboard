"""Generate equilibrium datasets for the surrogate and inverse models.

Dataset file format (``.npz``), shared by every model. Strings are fixed-width
unicode arrays so ``np.load(..., allow_pickle=False)`` works.

    R, Z           (nr,), (nz,)       shared uniform grid in metres (float64)
    psi            (N, nr, nz) f32    poloidal flux, Wb/rad (0 on the plasma boundary)
    J              (N, nr, nz) f32    toroidal current density, A/m^2
    mask           (N, nr, nz) bool   True inside the plasma
    params         (N, P) f32         equilibrium parameters, columns named by param_names
    param_names    (P,) <U32
    psi_axis, R_axis, Z_axis  (N,) f32
    converged      (N,) bool          always True; non-converged solves are dropped
    split          (N,) i1            0 = train, 1 = val, 2 = test

Fixed-boundary files use

    param_names = ["R0", "a", "kappa", "delta", "Ip", "beta0", "alpha", "gamma", "B0"]

with ``a = eps * R0``. In-distribution ranges: R0 [1.5, 1.9] m, eps [0.25, 0.40],
kappa [1.2, 2.0], delta [0.0, 0.5], Ip [0.5e6, 2.0e6] A, beta0 [0.2, 0.8],
alpha [0.8, 2.5], gamma [1.0, 3.0], B0 [1.5, 3.0] T. Sampled with scrambled
Sobol. The out-of-distribution file uses the same ranges except kappa in
(2.0, 2.3]. That upper limit was checked: the Cerfon-Freidberg level set (A=0)
stays a single closed negative region and ``solve_fixed_boundary`` converges
at the corners and on Sobol probes. The OOD Sobol stream uses ``seed + 100003``
so it is not a paired copy of the in-distribution file. OOD ``split`` is 2.

One shared grid contains every in-distribution and OOD plasma. Miller extremes
over the full parameter domain (including kappa = 2.3) are R in [0.90, 2.66] m
and |Z| <= 1.748 m. The stored box is

    R in [0.80, 2.76],  Z in [-1.90, 1.90]

which leaves about 2.5 cells of margin at 65x65 (dR = 0.030625 m, dZ = 0.059375 m)
outside the Miller boundary, and at least 4 cells outside the discrete core mask.
Both ``fixed`` and ``fixed --ood`` use this box. Arrays are (nr, nz) with
``indexing="ij"``.

The random 80/10/10 split is a permutation from ``numpy.random.Generator(seed)``
applied after dropping failures. Counts use ``round(0.1 * N)`` for val and test.

Free-boundary files add ``coil_names`` (n_coils,) <U32, ``coil_currents``
(N, n_coils) f32 and ``q95`` (N,) f32. See ``data.freegs_gen`` for the FreeGS
parameter list and the psi sign convention (psi = 0 on the LCFS, psi_axis > 0).

CLI::

    python -m data.generate fixed --n 5000 --nr 65 --nz 65 --seed 0 --out data/fixed_65.npz
    python -m data.generate fixed --ood --n 500 --nr 65 --nz 65 --seed 0 --out data/fixed_65_ood.npz
    python -m data.generate freegs --n 600 --out data/freegs_65.npz
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.stats import qmc

from gs.solver import Grid, ProfileParams, ShapeParams, solve_fixed_boundary

PARAM_NAMES = ["R0", "a", "kappa", "delta", "Ip", "beta0", "alpha", "gamma", "B0"]

# Shared fixed-boundary box (metres). See the module docstring.
FIXED_RMIN = 0.80
FIXED_RMAX = 2.76
FIXED_ZMIN = -1.90
FIXED_ZMAX = 1.90

# kappa OOD interval (2.0, 2.3], verified closed and convergent.
OOD_KAPPA_MIN = 2.0
OOD_KAPPA_MAX = 2.3
OOD_SEED_OFFSET = 100003

_ID_BOUNDS = {
    "R0": (1.5, 1.9),
    "eps": (0.25, 0.40),
    "kappa": (1.2, 2.0),
    "delta": (0.0, 0.5),
    "Ip": (0.5e6, 2.0e6),
    "beta0": (0.2, 0.8),
    "alpha": (0.8, 2.5),
    "gamma": (1.0, 3.0),
    "B0": (1.5, 3.0),
}


def fixed_grid(nr: int, nz: int) -> Grid:
    """Uniform grid shared by every fixed-boundary sample, including OOD."""
    return Grid.uniform(FIXED_RMIN, FIXED_RMAX, FIXED_ZMIN, FIXED_ZMAX, int(nr), int(nz))


def sample_fixed_params(n: int, seed: int, ood: bool = False) -> np.ndarray:
    """Scrambled Sobol sample, shape (n, 9), columns in ``PARAM_NAMES`` order.

    ``a`` is stored as ``eps * R0``. OOD draws kappa uniformly in (2.0, 2.3]
    from an independent Sobol stream (``seed + OOD_SEED_OFFSET``).
    """
    if n < 1:
        raise ValueError("n must be positive")
    sobol_seed = int(seed) + (OOD_SEED_OFFSET if ood else 0)
    sampler = qmc.Sobol(d=9, scramble=True, seed=sobol_seed)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The balance properties of Sobol",
            category=UserWarning,
        )
        u = sampler.random(int(n))
    R0 = _ID_BOUNDS["R0"][0] + (_ID_BOUNDS["R0"][1] - _ID_BOUNDS["R0"][0]) * u[:, 0]
    eps = _ID_BOUNDS["eps"][0] + (_ID_BOUNDS["eps"][1] - _ID_BOUNDS["eps"][0]) * u[:, 1]
    if ood:
        # u = 0 -> 2.3, u = 1 -> just above 2.0, so kappa is in (2.0, 2.3].
        span = OOD_KAPPA_MAX - OOD_KAPPA_MIN
        kappa = OOD_KAPPA_MAX - span * u[:, 2] * (1.0 - 1e-12)
    else:
        kappa = _ID_BOUNDS["kappa"][0] + (
            _ID_BOUNDS["kappa"][1] - _ID_BOUNDS["kappa"][0]
        ) * u[:, 2]
    delta = _ID_BOUNDS["delta"][0] + (_ID_BOUNDS["delta"][1] - _ID_BOUNDS["delta"][0]) * u[:, 3]
    Ip = _ID_BOUNDS["Ip"][0] + (_ID_BOUNDS["Ip"][1] - _ID_BOUNDS["Ip"][0]) * u[:, 4]
    beta0 = _ID_BOUNDS["beta0"][0] + (_ID_BOUNDS["beta0"][1] - _ID_BOUNDS["beta0"][0]) * u[:, 5]
    alpha = _ID_BOUNDS["alpha"][0] + (_ID_BOUNDS["alpha"][1] - _ID_BOUNDS["alpha"][0]) * u[:, 6]
    gamma = _ID_BOUNDS["gamma"][0] + (_ID_BOUNDS["gamma"][1] - _ID_BOUNDS["gamma"][0]) * u[:, 7]
    B0 = _ID_BOUNDS["B0"][0] + (_ID_BOUNDS["B0"][1] - _ID_BOUNDS["B0"][0]) * u[:, 8]
    return np.column_stack([R0, eps * R0, kappa, delta, Ip, beta0, alpha, gamma, B0])


def assign_splits(n: int, seed: int, ood: bool = False) -> np.ndarray:
    """Split labels, dtype int8. OOD samples are all test (2).

    Otherwise a seeded permutation with counts
    ``n_val = round(0.1*n)``, ``n_test = round(0.1*n)``, ``n_train = n - n_val - n_test``.
    """
    split = np.empty(int(n), dtype=np.int8)
    if n <= 0:
        return split
    if ood:
        split[:] = 2
        return split
    n_val = int(round(0.1 * n))
    n_test = int(round(0.1 * n))
    if n_val + n_test > n:
        n_val = n // 10
        n_test = n // 10
    n_train = int(n) - n_val - n_test
    order = np.random.default_rng(int(seed)).permutation(int(n))
    split[order[:n_train]] = 0
    split[order[n_train : n_train + n_val]] = 1
    split[order[n_train + n_val :]] = 2
    return split


def _solve_fixed_task(task):
    """Worker: one fixed-boundary solve. Returns (index, payload or None, reason)."""
    index, params, nr, nz = task
    R0, a, kappa, delta, Ip, beta0, alpha, gamma, B0 = (float(v) for v in params)
    grid = fixed_grid(nr, nz)
    try:
        eq = solve_fixed_boundary(
            ShapeParams(R0, a, kappa, delta),
            ProfileParams(Ip, beta0, alpha, gamma, B0),
            grid,
        )
    except Exception as exc:
        return index, None, type(exc).__name__
    if not eq.converged:
        return index, None, "not_converged"
    if not (
        np.isfinite(eq.psi).all()
        and np.isfinite(eq.J).all()
        and np.isfinite(eq.psi_axis)
        and eq.psi_axis > 0.0
    ):
        return index, None, "nonfinite"
    payload = (
        np.asarray(eq.psi, dtype=np.float32),
        np.asarray(eq.J, dtype=np.float32),
        np.asarray(eq.mask, dtype=bool),
        np.asarray(params, dtype=np.float32),
        np.float32(eq.psi_axis),
        np.float32(eq.R_axis),
        np.float32(eq.Z_axis),
    )
    return index, payload, "ok"


def _consume(tasks, workers: int, label: str, every: int):
    t0 = time.perf_counter()
    results = []
    if workers <= 1:
        iterator = (_solve_fixed_task(task) for task in tasks)
        total = len(tasks)
        for done, item in enumerate(iterator, start=1):
            results.append(item)
            if done % every == 0 or done == total:
                kept_n = sum(r[1] is not None for r in results)
                print(
                    f"{label}: {done}/{total} kept={kept_n} "
                    f"elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            total = len(tasks)
            for done, item in enumerate(
                pool.imap_unordered(_solve_fixed_task, tasks, chunksize=8), start=1
            ):
                results.append(item)
                if done % every == 0 or done == total:
                    kept_n = sum(r[1] is not None for r in results)
                    print(
                        f"{label}: {done}/{total} kept={kept_n} "
                        f"elapsed={time.perf_counter() - t0:.1f}s",
                        flush=True,
                    )
    elapsed = time.perf_counter() - t0
    results.sort(key=lambda item: item[0])
    return results, elapsed


def generate_fixed(
    n: int,
    nr: int,
    nz: int,
    seed: int,
    out,
    ood: bool = False,
    workers: int = 2,
) -> dict:
    """Generate a fixed-boundary dataset and write it to ``out``.

    Non-converged and failed solves are dropped. Training files therefore
    contain only converged equilibria; ``converged`` is True for every row.
    """
    workers = max(1, min(int(workers), 2))
    params = sample_fixed_params(int(n), int(seed), ood=bool(ood))
    grid = fixed_grid(nr, nz)
    tasks = [(i, params[i], int(nr), int(nz)) for i in range(int(n))]
    label = "fixed-ood" if ood else "fixed"
    print(
        f"{label}: attempts={n} grid=[{FIXED_RMIN}, {FIXED_RMAX}] x "
        f"[{FIXED_ZMIN}, {FIXED_ZMAX}] nr={nr} nz={nz} dR={grid.dR:.6f} dZ={grid.dZ:.6f} "
        f"workers={workers} seed={seed}",
        flush=True,
    )
    results, elapsed = _consume(tasks, workers, label, every=max(1, min(100, int(n))))
    reasons: dict[str, int] = {}
    kept = []
    for _index, payload, reason in results:
        if payload is None:
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            kept.append(payload)
    n_kept = len(kept)
    if n_kept:
        psi = np.stack([p[0] for p in kept])
        J = np.stack([p[1] for p in kept])
        mask = np.stack([p[2] for p in kept])
        param_arr = np.stack([p[3] for p in kept])
        psi_axis = np.array([p[4] for p in kept], dtype=np.float32)
        R_axis = np.array([p[5] for p in kept], dtype=np.float32)
        Z_axis = np.array([p[6] for p in kept], dtype=np.float32)
    else:
        psi = np.zeros((0, int(nr), int(nz)), dtype=np.float32)
        J = np.zeros_like(psi)
        mask = np.zeros((0, int(nr), int(nz)), dtype=bool)
        param_arr = np.zeros((0, len(PARAM_NAMES)), dtype=np.float32)
        psi_axis = np.zeros((0,), dtype=np.float32)
        R_axis = np.zeros((0,), dtype=np.float32)
        Z_axis = np.zeros((0,), dtype=np.float32)
    payload = {
        "R": np.asarray(grid.R, dtype=np.float64),
        "Z": np.asarray(grid.Z, dtype=np.float64),
        "psi": np.asarray(psi, dtype=np.float32),
        "J": np.asarray(J, dtype=np.float32),
        "mask": np.asarray(mask, dtype=bool),
        "params": np.asarray(param_arr, dtype=np.float32),
        "param_names": np.asarray(PARAM_NAMES, dtype="U32"),
        "psi_axis": psi_axis,
        "R_axis": R_axis,
        "Z_axis": Z_axis,
        "converged": np.ones(n_kept, dtype=bool),
        "split": assign_splits(n_kept, int(seed), ood=bool(ood)),
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(
        f"{label}: kept={n_kept}/{n} dropped={int(n) - n_kept} reasons={reasons} "
        f"elapsed={elapsed:.1f}s -> {out_path}",
        flush=True,
    )
    return {
        "path": str(out_path),
        "n_attempted": int(n),
        "n_kept": n_kept,
        "n_failed": int(n) - n_kept,
        "reasons": reasons,
        "seconds": elapsed,
        "ood": bool(ood),
        "grid": (FIXED_RMIN, FIXED_RMAX, FIXED_ZMIN, FIXED_ZMAX, int(nr), int(nz)),
    }


def load_dataset(path: str) -> dict:
    """Load a dataset file into a dict of numpy arrays.

    Uses ``allow_pickle=False``. Arrays are copied out of the zip so they
    remain valid after the file is closed.
    """
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def _print_summary(path: str) -> None:
    data = load_dataset(path)
    n = int(data["psi"].shape[0])
    size = Path(path).stat().st_size
    print(f"summary {path}: N={n} bytes={size} keys={list(data)}", flush=True)
    if n == 0:
        return
    psi_a = data["psi_axis"]
    print(
        f"  psi_axis [{float(psi_a.min()):.6g}, {float(psi_a.max()):.6g}] "
        f"R_axis [{float(data['R_axis'].min()):.4f}, {float(data['R_axis'].max()):.4f}] "
        f"Z_axis [{float(data['Z_axis'].min()):.4f}, {float(data['Z_axis'].max()):.4f}]",
        flush=True,
    )
    names = [str(x) for x in data["param_names"]]
    if "R0" in names:
        r0 = data["params"][:, names.index("R0")]
        shift = data["R_axis"] - r0
        print(
            f"  Shafranov shift R_axis-R0 [{float(shift.min()):.4f}, {float(shift.max()):.4f}] m",
            flush=True,
        )
    if "R_geom" in names:
        r_geom = data["params"][:, names.index("R_geom")]
        shift = data["R_axis"] - r_geom
        print(
            f"  Shafranov shift R_axis-R_geom [{float(shift.min()):.4f}, {float(shift.max()):.4f}] m",
            flush=True,
        )
    splits, counts = np.unique(data["split"], return_counts=True)
    print(f"  split {dict(zip(splits.tolist(), counts.tolist()))}", flush=True)
    print(
        f"  grid R[{float(data['R'][0]):.4f}, {float(data['R'][-1]):.4f}] "
        f"Z[{float(data['Z'][0]):.4f}, {float(data['Z'][-1]):.4f}] "
        f"shape {data['R'].shape[0]}x{data['Z'].shape[0]}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-Shafranov equilibrium datasets")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fixed = sub.add_parser("fixed", help="fixed-boundary dataset from gs.solver")
    p_fixed.add_argument("--n", type=int, required=True)
    p_fixed.add_argument("--nr", type=int, default=65)
    p_fixed.add_argument("--nz", type=int, default=65)
    p_fixed.add_argument("--seed", type=int, default=0)
    p_fixed.add_argument("--out", type=str, required=True)
    p_fixed.add_argument("--ood", action="store_true", help="kappa in (2.0, 2.3], all splits = test")
    p_fixed.add_argument("--workers", type=int, default=2)

    p_free = sub.add_parser("freegs", help="free-boundary dataset from FreeGS TestTokamak")
    p_free.add_argument("--n", type=int, required=True)
    p_free.add_argument("--out", type=str, required=True)
    p_free.add_argument("--seed", type=int, default=0)
    p_free.add_argument("--nx", type=int, default=65)
    p_free.add_argument("--ny", type=int, default=65)
    p_free.add_argument("--workers", type=int, default=2)

    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    if args.cmd == "fixed":
        generate_fixed(args.n, args.nr, args.nz, args.seed, args.out, ood=args.ood, workers=args.workers)
    else:
        from data.freegs_gen import generate_freegs

        generate_freegs(args.n, args.out, seed=args.seed, nx=args.nx, ny=args.ny, workers=args.workers)
    _print_summary(args.out)
    print(f"total {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
