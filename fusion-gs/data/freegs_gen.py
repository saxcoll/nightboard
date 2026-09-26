"""Free-boundary equilibria from FreeGS (TestTokamak) on one shared grid.

Sign convention
---------------
FreeGS solves the same Grad-Shafranov equation as ``gs.solver``:

    Delta* psi = -mu0 * R * J_phi

with ``J_phi`` from ``ConstrainBetapIp`` (positive for ``Ip > 0``). The magnetic
axis is then a maximum of psi. Stored fields are shifted onto the fixed-boundary
convention (psi = 0 on the LCFS, psi_axis > 0 inside):

    s = +1 if psi_axis >= psi_bndry else -1
    psi_store = s * (psi_freegs - psi_bndry)
    J_store   = s * J_freegs

``s = +1`` on the TestTokamak scans used here. ``s = -1`` flips a reversed
solution so that psi and J stay consistent with ``Delta* psi = -mu0 R J`` and
with ``Ip > 0``. ``mask`` is FreeGS ``critical.core_mask`` (inside the LCFS,
connected to the axis), stored as bool.

Grid (all samples): ``R in [0.1, 2.0]``, ``Z in [-1.0, 1.0]``, ``nx = ny``
with ``nx = 2**k + 1`` (default 65). Indexing is ``"ij"``, matching ``gs.solver``.

Parameter vector (``FREEGS_PARAM_NAMES``)
-----------------------------------------
Control inputs, then measured geometry. For a single-null plasma
``xpt_upper_R`` and ``xpt_upper_Z`` are NaN and ``n_xpoints`` is 1.
``iso_R1, iso_Z1`` is the lower X-point; ``iso_R2, iso_Z2`` is the second
isoflux point (same psi). Measured ``R_geom, a_minor, kappa, delta`` follow
the FreeGS definitions (geometric centre, minor radius, elongation, average
triangularity). Coil set is TestTokamak ``P1L, P1U, P2L, P2U``.

Attempted ranges (Sobol): betap [0.05, 0.6], Ip [1e5, 4e5] A, fvac [1.0, 3.0],
alpha_m [0.8, 2.5], alpha_n [1.0, 3.0], shape points jittered by ±0.08 m
(upper X-point Z by ±0.05 m) about the FreeGS double-null example. About 30%
of attempts are lower single-null (one X-point plus one isoflux pair).

A sample is kept only if the solve finishes (maxits, or a per-solve alarm),
plasma current matches Ip to 2%, the axis lies inside a contained core mask,
q95 is finite and in (0.5, 40), and the measured shape is physical.
"""
from __future__ import annotations

import contextlib
import io
import multiprocessing as mp
import os
import signal
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import qmc

# Limit BLAS threads before numeric libraries spin pools inside workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

FREEGS_PARAM_NAMES = [
    "betap",
    "Ip",
    "fvac",
    "alpha_m",
    "alpha_n",
    "xpt_lower_R",
    "xpt_lower_Z",
    "xpt_upper_R",
    "xpt_upper_Z",
    "iso_R1",
    "iso_Z1",
    "iso_R2",
    "iso_Z2",
    "n_xpoints",
    "R_geom",
    "a_minor",
    "kappa",
    "delta",
]

# TestTokamak free-boundary box used for every sample.
FREEGS_RMIN = 0.1
FREEGS_RMAX = 2.0
FREEGS_ZMIN = -1.0
FREEGS_ZMAX = 1.0
FREEGS_COIL_NAMES = ["P1L", "P1U", "P2L", "P2U"]

# Nominal double-null example from the FreeGS docs, before jitter.
_XPT_L = (1.1, -0.6)
_XPT_U = (1.1, 0.8)
_ISO2 = (1.1, 0.6)
_JITTER = 0.08
_JITTER_UPPER_Z = 0.05
_SINGLE_NULL_FRACTION = 0.30

_SOLVE_MAXITS = 40
_SOLVE_TIMEOUT_S = 45


class _SolveTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _SolveTimeout("freegs solve exceeded time limit")


def _is_power_of_two_plus_one(n: int) -> bool:
    if n < 3:
        return False
    m = n - 1
    return m > 0 and (m & (m - 1)) == 0


def sample_freegs_params(n: int, seed: int) -> np.ndarray:
    """Sobol sample of solver inputs, shape (n, 12).

    Columns: betap, Ip, fvac, alpha_m, alpha_n, xpt_lower_R, xpt_lower_Z,
    xpt_upper_R, xpt_upper_Z, iso_R2, iso_Z2, topology (u in [0, 1), single-null
    when u < 0.30). Upper X-point coordinates are still filled for single-null
    rows; the solver ignores them.
    """
    if n < 1:
        raise ValueError("n must be positive")
    sampler = qmc.Sobol(d=12, scramble=True, seed=int(seed))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The balance properties of Sobol",
            category=UserWarning,
        )
        u = sampler.random(int(n))
    jit = _JITTER
    return np.column_stack(
        [
            0.05 + 0.55 * u[:, 0],
            1.0e5 + 3.0e5 * u[:, 1],
            1.0 + 2.0 * u[:, 2],
            0.8 + 1.7 * u[:, 3],
            1.0 + 2.0 * u[:, 4],
            _XPT_L[0] + jit * (2.0 * u[:, 5] - 1.0),
            _XPT_L[1] + jit * (2.0 * u[:, 6] - 1.0),
            _XPT_U[0] + jit * (2.0 * u[:, 7] - 1.0),
            _XPT_U[1] + _JITTER_UPPER_Z * (2.0 * u[:, 8] - 1.0),
            _ISO2[0] + jit * (2.0 * u[:, 9] - 1.0),
            _ISO2[1] + jit * (2.0 * u[:, 10] - 1.0),
            u[:, 11],
        ]
    )


def _geometry(eq) -> tuple[float, float, float, float] | None:
    """(R_geom, a, kappa, delta) from one separatrix trace, or None."""
    separatrix = np.asarray(eq.separatrix(npoints=180), dtype=float)
    if separatrix.ndim != 2 or separatrix.shape[0] < 16 or separatrix.shape[1] < 2:
        return None
    r = separatrix[:, 0]
    z = separatrix[:, 1]
    if not (np.isfinite(r).all() and np.isfinite(z).all()):
        return None
    r_out = float(np.max(r))
    r_in = float(np.min(r))
    a = 0.5 * (r_out - r_in)
    if a <= 0.0:
        return None
    r_geom = 0.5 * (r_out + r_in)
    z_top = float(np.max(z))
    z_bot = float(np.min(z))
    kappa = 0.5 * (z_top - z_bot) / a
    r_top = float(r[int(np.argmax(z))])
    r_bot = float(r[int(np.argmin(z))])
    delta = 0.5 * ((r_geom - r_top) + (r_geom - r_bot)) / a
    return r_geom, a, kappa, delta


def _quality(eq, Ip: float, psi: np.ndarray, J: np.ndarray, mask: np.ndarray, q95: float, geom) -> str:
    if geom is None:
        return "geometry"
    r_geom, a, kappa, delta = geom
    nr, nz = mask.shape
    n_core = int(mask.sum())
    if n_core < max(20, int(0.02 * nr * nz)):
        return "mask_small"
    if n_core > int(0.70 * nr * nz):
        return "mask_large"
    if mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any():
        return "mask_touches_edge"
    Rax, Zax, _ = eq.magneticAxis()
    Rax = float(Rax)
    Zax = float(Zax)
    if not (eq.Rmin < Rax < eq.Rmax and eq.Zmin < Zax < eq.Zmax):
        return "axis_outside"
    i = int(np.argmin(np.abs(eq.R_1D - Rax)))
    j = int(np.argmin(np.abs(eq.Z_1D - Zax)))
    if not mask[i, j]:
        return "axis_not_in_mask"
    Ip_m = float(eq.plasmaCurrent())
    if not np.isfinite(Ip_m) or abs(Ip_m - Ip) / abs(Ip) > 0.02:
        return "ip_mismatch"
    I_sum = float(np.sum(J, dtype=np.float64) * eq.dR * eq.dZ)
    if not np.isfinite(I_sum) or I_sum <= 0.0 or abs(I_sum - Ip) / abs(Ip) > 0.05:
        return "j_integral"
    if not np.isfinite(q95) or not (0.5 < q95 < 40.0):
        return "q95"
    if not (0.08 < a < 0.80 and 0.9 < kappa < 2.8 and abs(delta) < 0.95):
        return "shape"
    if not (0.85 < r_geom < 1.75):
        return "r_geom"
    if not np.isfinite(psi).all() or not np.isfinite(J).all():
        return "nonfinite"
    if float(np.max(psi)) <= 0.0:
        return "psi_sign"
    return "ok"


def solve_freegs_sample(params, nx: int = 65, ny: int = 65):
    """Solve one FreeGS equilibrium.

    Returns ``(payload, reason)``. ``payload`` is a dict of arrays for one
    sample, or None when ``reason != "ok"``.
    """
    import freegs
    from freegs import critical

    betap, Ip, fvac, alpha_m, alpha_n = (float(v) for v in params[:5])
    xlr, xlz, xur, xuz, iso_r, iso_z, topo = (float(v) for v in params[5:12])
    single = topo < _SINGLE_NULL_FRACTION
    if abs(iso_r - xlr) + abs(iso_z - xlz) < 1e-3:
        iso_r += 0.05

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(_SOLVE_TIMEOUT_S)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            tokamak = freegs.machine.TestTokamak()
            eq = freegs.Equilibrium(
                tokamak=tokamak,
                Rmin=FREEGS_RMIN,
                Rmax=FREEGS_RMAX,
                Zmin=FREEGS_ZMIN,
                Zmax=FREEGS_ZMAX,
                nx=int(nx),
                ny=int(ny),
            )
            profiles = freegs.jtor.ConstrainBetapIp(
                eq,
                betap,
                Ip,
                fvac,
                alpha_m=alpha_m,
                alpha_n=alpha_n,
            )
            xpoints = [(xlr, xlz)]
            if not single:
                xpoints.append((xur, xuz))
            constrain = freegs.control.constrain(
                xpoints=xpoints,
                isoflux=[(xlr, xlz, iso_r, iso_z)],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                freegs.solve(
                    eq,
                    profiles,
                    constrain,
                    maxits=_SOLVE_MAXITS,
                    rtol=1e-3,
                    atol=1e-10,
                    show=False,
                )
            psi_raw = np.asarray(eq.psi(), dtype=np.float64)
            psi_b = float(eq.psi_bndry)
            psi_a = float(eq.psi_axis)
            if not (np.isfinite(psi_b) and np.isfinite(psi_a)):
                return None, "psi_bounds"
            # Same GS sign as gs.solver. Flip only if this solve came out reversed.
            sign = 1.0 if (psi_a - psi_b) >= 0.0 else -1.0
            psi = sign * (psi_raw - psi_b)
            if eq.Jtor is None:
                J_raw = np.asarray(profiles.Jtor(eq.R, eq.Z, psi_raw), dtype=np.float64)
            else:
                J_raw = np.asarray(eq.Jtor, dtype=np.float64)
            J = sign * J_raw
            if eq.mask is None:
                opt, xpt = critical.find_critical(eq.R, eq.Z, psi_raw)
                if not opt:
                    return None, "no_opoint"
                mask_f = critical.core_mask(eq.R, eq.Z, psi_raw, opt, xpt, psi_b)
            else:
                mask_f = np.asarray(eq.mask, dtype=np.float64)
            mask = mask_f > 0.5
            q_pair = np.asarray(eq.q(np.array([0.95, 0.96])), dtype=float).reshape(-1)
            q95 = float(q_pair[0])
            geom = _geometry(eq)
            reason = _quality(eq, Ip, psi, J, mask, q95, geom)
            if reason != "ok":
                return None, reason
            r_geom, a, kappa, delta = geom
            Rax, Zax, _ = eq.magneticAxis()
            n_x = 1.0 if single else 2.0
            upper_r = np.nan if single else xur
            upper_z = np.nan if single else xuz
            param_row = np.array(
                [
                    betap,
                    Ip,
                    fvac,
                    alpha_m,
                    alpha_n,
                    xlr,
                    xlz,
                    upper_r,
                    upper_z,
                    xlr,
                    xlz,
                    iso_r,
                    iso_z,
                    n_x,
                    r_geom,
                    a,
                    kappa,
                    delta,
                ],
                dtype=np.float32,
            )
            currents = np.array([float(coil.current) for _, coil in tokamak.coils], dtype=np.float32)
            if currents.shape != (len(FREEGS_COIL_NAMES),):
                return None, "coils"
            payload = {
                "psi": np.asarray(psi, dtype=np.float32),
                "J": np.asarray(J, dtype=np.float32),
                "mask": np.asarray(mask, dtype=bool),
                "params": param_row,
                "psi_axis": np.float32(sign * (psi_a - psi_b)),
                "R_axis": np.float32(Rax),
                "Z_axis": np.float32(Zax),
                "q95": np.float32(q95),
                "coil_currents": currents,
                "sign": np.float32(sign),
            }
            return payload, "ok"
    except _SolveTimeout:
        return None, "timeout"
    except Exception as exc:
        return None, type(exc).__name__
    finally:
        signal.alarm(0)


def _solve_freegs_task(task):
    index, params, nx, ny = task
    payload, reason = solve_freegs_sample(params, nx=nx, ny=ny)
    return index, payload, reason


def _assign_splits(n: int, seed: int) -> np.ndarray:
    """Random 80/10/10 split. Imported logic duplicated to avoid an import cycle."""
    from data.generate import assign_splits

    return assign_splits(n, seed, ood=False)


def generate_freegs(
    n: int,
    out,
    seed: int = 0,
    nx: int = 65,
    ny: int = 65,
    workers: int = 2,
) -> dict:
    """Attempt ``n`` FreeGS solves and write the kept equilibria to ``out``."""
    if not _is_power_of_two_plus_one(int(nx)) or not _is_power_of_two_plus_one(int(ny)):
        raise ValueError("FreeGS nx and ny must be 2**k + 1 (e.g. 33 or 65)")
    workers = max(1, min(int(workers), 2))
    params = sample_freegs_params(int(n), int(seed))
    tasks = [(i, params[i], int(nx), int(ny)) for i in range(int(n))]
    R = np.linspace(FREEGS_RMIN, FREEGS_RMAX, int(nx))
    Z = np.linspace(FREEGS_ZMIN, FREEGS_ZMAX, int(ny))
    print(
        f"freegs: attempts={n} grid=[{FREEGS_RMIN}, {FREEGS_RMAX}] x "
        f"[{FREEGS_ZMIN}, {FREEGS_ZMAX}] nx={nx} ny={ny} workers={workers} seed={seed}",
        flush=True,
    )
    t0 = time.perf_counter()
    results: list[tuple[int, dict, str]] = []
    if workers == 1:
        for task in tasks:
            results.append(_solve_freegs_task(task))
            done = len(results)
            if done % 10 == 0 or done == len(tasks):
                kept_n = sum(r[1] is not None for r in results)
                print(
                    f"freegs: {done}/{len(tasks)} kept={kept_n} "
                    f"elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            for done, item in enumerate(
                pool.imap_unordered(_solve_freegs_task, tasks, chunksize=1), start=1
            ):
                results.append(item)
                if done % 10 == 0 or done == len(tasks):
                    kept_n = sum(r[1] is not None for r in results)
                    print(
                        f"freegs: {done}/{len(tasks)} kept={kept_n} "
                        f"elapsed={time.perf_counter() - t0:.1f}s",
                        flush=True,
                    )
    elapsed = time.perf_counter() - t0
    results.sort(key=lambda item: item[0])
    kept = [payload for _, payload, reason in results if payload is not None]
    reasons: dict[str, int] = {}
    for _, payload, reason in results:
        if payload is None:
            reasons[reason] = reasons.get(reason, 0) + 1
    n_kept = len(kept)
    if n_kept:
        psi = np.stack([p["psi"] for p in kept]).astype(np.float32, copy=False)
        J = np.stack([p["J"] for p in kept]).astype(np.float32, copy=False)
        mask = np.stack([p["mask"] for p in kept])
        param_arr = np.stack([p["params"] for p in kept]).astype(np.float32, copy=False)
        psi_axis = np.array([p["psi_axis"] for p in kept], dtype=np.float32)
        R_axis = np.array([p["R_axis"] for p in kept], dtype=np.float32)
        Z_axis = np.array([p["Z_axis"] for p in kept], dtype=np.float32)
        q95 = np.array([p["q95"] for p in kept], dtype=np.float32)
        coil_currents = np.stack([p["coil_currents"] for p in kept]).astype(np.float32, copy=False)
        signs = np.array([p["sign"] for p in kept], dtype=np.float32)
    else:
        psi = np.zeros((0, int(nx), int(ny)), dtype=np.float32)
        J = np.zeros_like(psi)
        mask = np.zeros((0, int(nx), int(ny)), dtype=bool)
        param_arr = np.zeros((0, len(FREEGS_PARAM_NAMES)), dtype=np.float32)
        psi_axis = np.zeros((0,), dtype=np.float32)
        R_axis = np.zeros((0,), dtype=np.float32)
        Z_axis = np.zeros((0,), dtype=np.float32)
        q95 = np.zeros((0,), dtype=np.float32)
        coil_currents = np.zeros((0, len(FREEGS_COIL_NAMES)), dtype=np.float32)
        signs = np.zeros((0,), dtype=np.float32)
    payload = {
        "R": np.asarray(R, dtype=np.float64),
        "Z": np.asarray(Z, dtype=np.float64),
        "psi": psi,
        "J": J,
        "mask": mask,
        "params": param_arr,
        "param_names": np.asarray(FREEGS_PARAM_NAMES, dtype="U32"),
        "psi_axis": psi_axis,
        "R_axis": R_axis,
        "Z_axis": Z_axis,
        "converged": np.ones(n_kept, dtype=bool),
        "split": _assign_splits(n_kept, int(seed)),
        "coil_names": np.asarray(FREEGS_COIL_NAMES, dtype="U32"),
        "coil_currents": coil_currents,
        "q95": q95,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    n_flip = int(np.sum(signs < 0)) if n_kept else 0
    print(
        f"freegs: kept={n_kept}/{n} failed={n - n_kept} sign_flips={n_flip} "
        f"reasons={reasons} elapsed={elapsed:.1f}s -> {out_path}",
        flush=True,
    )
    return {
        "path": str(out_path),
        "n_attempted": int(n),
        "n_kept": n_kept,
        "n_failed": int(n) - n_kept,
        "reasons": reasons,
        "sign_flips": n_flip,
        "seconds": elapsed,
        "grid": (FREEGS_RMIN, FREEGS_RMAX, FREEGS_ZMIN, FREEGS_ZMAX, int(nx), int(ny)),
    }
