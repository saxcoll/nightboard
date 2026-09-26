"""Finite-difference Grad-Shafranov solvers on a uniform (R, Z) grid.

Conventions (shared by every module in this project):

* Arrays on the grid have shape (nr, nz) with ``indexing="ij"``: ``psi[i, j]`` is at
  ``(grid.R[i], grid.Z[j])``.
* SI units: metres, amperes, tesla, psi in Wb/rad, J in A/m^2.
* GS equation: Delta* psi = R d/dR((1/R) dpsi/dR) + d2psi/dZ2 = -mu0 R J_phi.
* Nonlinear current profile (same form as FreeGS ``ConstrainBetapIp``):

      J_phi(R, psi_n) = lam * (beta0 R/R0 + (1 - beta0) R0/R) * (1 - psi_n^alpha)^gamma

  inside the plasma and 0 outside, with ``lam`` rescaled every iteration so that the total
  plasma current equals Ip. Hence p'(psi) = lam beta0 / R0 g(psi_n) and
  FF'(psi) = mu0 lam (1 - beta0) R0 g(psi_n) where g = (1 - psi_n^alpha)^gamma.
* psi = 0 on the plasma boundary; for Ip > 0 psi has a maximum psi_axis > 0 on the
  magnetic axis. psi_n = (psi - psi_axis) / (psi_boundary - psi_axis) is 0 on axis and
  1 on the boundary.
* The shaped plasma boundary is the psi_bar = 0 surface of a Cerfon-Freidberg Solov'ev
  equilibrium with the same (R0, a/R0, kappa, delta), see ``ShapeParams.level_set``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from gs.analytic import solovev

MU0 = 4e-7 * np.pi

LevelSet = Callable[[np.ndarray, np.ndarray], np.ndarray]

# Treat a cut closer than this fraction of the local grid spacing as a boundary node.
_TINY_CUT = 1e-3


@dataclass(frozen=True)
class Grid:
    R: np.ndarray
    Z: np.ndarray

    @classmethod
    def uniform(cls, Rmin: float, Rmax: float, Zmin: float, Zmax: float, nr: int, nz: int) -> "Grid":
        return cls(np.linspace(Rmin, Rmax, nr), np.linspace(Zmin, Zmax, nz))

    @property
    def nr(self) -> int:
        return self.R.size

    @property
    def nz(self) -> int:
        return self.Z.size

    @property
    def dR(self) -> float:
        return float(self.R[1] - self.R[0])

    @property
    def dZ(self) -> float:
        return float(self.Z[1] - self.Z[0])

    @property
    def RR(self) -> np.ndarray:
        return np.meshgrid(self.R, self.Z, indexing="ij")[0]

    @property
    def ZZ(self) -> np.ndarray:
        return np.meshgrid(self.R, self.Z, indexing="ij")[1]


@dataclass(frozen=True)
class ShapeParams:
    R0: float
    a: float
    kappa: float
    delta: float

    def level_set(self) -> LevelSet:
        """Level-set function (negative inside) for the shaped fixed boundary.

        The contour is the psi_bar = 0 surface of a Cerfon-Freidberg Solov'ev
        equilibrium with ``epsilon = a/R0`` and fixed ``A = 0``. A = 0 yields one
        closed psi < 0 region for epsilon in [0.25, 0.40], kappa in [1.2, 2.0]
        and delta in [0.0, 0.5].
        """
        eq = solovev(
            epsilon=self.a / self.R0,
            kappa=self.kappa,
            delta=self.delta,
            A=0.0,
            R0=self.R0,
        )
        return eq.level_set


@dataclass(frozen=True)
class ProfileParams:
    Ip: float
    beta0: float
    alpha: float
    gamma: float
    B0: float


@dataclass
class Equilibrium:
    grid: Grid
    shape: ShapeParams
    profile: ProfileParams
    psi: np.ndarray
    J: np.ndarray
    mask: np.ndarray
    psi_axis: float
    psi_boundary: float
    R_axis: float
    Z_axis: float
    lam: float
    n_iter: int
    converged: bool
    history: list[float] = field(default_factory=list)

    def psin(self) -> np.ndarray:
        return (self.psi - self.psi_axis) / (self.psi_boundary - self.psi_axis)

    def _g(self, psin) -> np.ndarray:
        s = np.clip(np.asarray(psin, dtype=float), 0.0, 1.0)
        return (1.0 - s ** self.profile.alpha) ** self.profile.gamma

    def pprime(self, psin) -> np.ndarray:
        return self.lam * self.profile.beta0 / self.shape.R0 * self._g(psin)

    def ffprime(self, psin) -> np.ndarray:
        return MU0 * self.lam * (1.0 - self.profile.beta0) * self.shape.R0 * self._g(psin)

    def _integral_from_1(self, psin, which: str) -> np.ndarray:
        """int_1^{psin} f(s) ds with f = p'(psi_n) or FF'(psi_n), trapezoid on a fine grid."""
        s = np.linspace(0.0, 1.0, 4001)
        f = self.pprime(s) if which == "p" else self.ffprime(s)
        ds = s[1] - s[0]
        cum = np.empty_like(s)
        cum[0] = 0.0
        cum[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * ds)
        integ = np.interp(np.asarray(psin, dtype=float), s, cum) - cum[-1]
        return integ

    def F(self, psin) -> np.ndarray:
        """Toroidal field function F = R B_phi, with F(1) = R0 B0."""
        integ = self._integral_from_1(psin, "f")
        f2 = (self.shape.R0 * self.profile.B0) ** 2 + 2.0 * (
            self.psi_boundary - self.psi_axis
        ) * integ
        out = np.sqrt(np.maximum(f2, 0.0))
        if np.ndim(psin) == 0:
            return float(out)
        return out

    def pressure(self, psin) -> np.ndarray:
        integ = self._integral_from_1(psin, "p")
        out = (self.psi_boundary - self.psi_axis) * integ
        if np.ndim(psin) == 0:
            return float(out)
        return out


def delta_star_fd(psi: np.ndarray, grid: Grid) -> np.ndarray:
    """Second-order finite-difference Delta* psi. Edge rows/columns are NaN.

    Radial part is the conservative midpoint flux form

        R_i / dR * ( F_{i+1/2} - F_{i-1/2} ),
        F_{i+1/2} = (psi_{i+1} - psi_i) / (dR * R_{i+1/2}),

    with R_{i+1/2} = (R_i + R_{i+1}) / 2, plus the standard second difference in Z.
    """
    psi = np.asarray(psi, dtype=float)
    nr, nz = psi.shape
    out = np.full((nr, nz), np.nan, dtype=float)
    if nr < 3 or nz < 3:
        return out
    dR = grid.dR
    dZ = grid.dZ
    R = grid.R
    r_half = 0.5 * (R[:-1] + R[1:])
    flux = (psi[1:, :] - psi[:-1, :]) / (dR * r_half[:, None])
    dflux = (flux[1:, :] - flux[:-1, :]) / dR
    radial = R[1:-1, None] * dflux
    d2z = (psi[:, 2:] - 2.0 * psi[:, 1:-1] + psi[:, :-2]) / dZ**2
    out[1:-1, 1:-1] = radial[:, 1:-1] + d2z[1:-1, :]
    return out


def _flatten(i: int, j: int, nz: int) -> int:
    return i * nz + j


def _eval_level(level_set: LevelSet, R, Z) -> float:
    val = np.asarray(level_set(R, Z), dtype=float)
    return float(val.reshape(-1)[0])


def _cut_fraction(level_set: LevelSet, r0: float, z0: float, r1: float, z1: float, f0: float, f1: float) -> float:
    """Fraction t in (0, 1] from (r0, z0) to the level_set root toward (r1, z1)."""
    if f1 == 0.0:
        return 1.0

    def func(t: float) -> float:
        return _eval_level(level_set, r0 + t * (r1 - r0), z0 + t * (z1 - z0))

    try:
        if func(0.0) * f1 < 0.0:
            return float(brentq(func, 0.0, 1.0, xtol=1e-12))
    except ValueError:
        pass
    denom = f0 - f1
    if denom == 0.0:
        return 0.5
    return float(np.clip(f0 / denom, 1e-6, 1.0))


def _g_field(grid: Grid, boundary_value) -> np.ndarray:
    if callable(boundary_value):
        g = np.asarray(boundary_value(grid.RR, grid.ZZ), dtype=float)
        if g.shape != (grid.nr, grid.nz):
            g = np.broadcast_to(g, (grid.nr, grid.nz)).copy()
        return g
    return np.full((grid.nr, grid.nz), float(boundary_value), dtype=float)


class _LinearOp:
    """Factored Delta* operator with Dirichlet data baked into a fixed RHS offset."""

    def __init__(self, lu, b_fixed: np.ndarray, unknown: np.ndarray):
        self.lu = lu
        self.b_fixed = b_fixed
        self.unknown = unknown

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        b = self.b_fixed.copy()
        b[self.unknown.ravel()] += np.asarray(rhs, dtype=float).ravel()[self.unknown.ravel()]
        return self.lu.solve(b).reshape(self.unknown.shape)


def _build_operator(grid: Grid, level_set: LevelSet | None, boundary_value) -> _LinearOp:
    nr, nz = grid.nr, grid.nz
    n = nr * nz
    dR = grid.dR
    dZ = grid.dZ
    R = grid.R
    Z = grid.Z
    g = _g_field(grid, boundary_value)

    edge = np.zeros((nr, nz), dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    if level_set is None:
        phi = np.full((nr, nz), -1.0)
        inside = ~edge
    else:
        phi = np.asarray(level_set(grid.RR, grid.ZZ), dtype=float)
        inside = (phi < 0.0) & ~edge

    # h' along each grid direction for nodes that see an exterior neighbour.
    # None means the neighbour is inside the level set (full grid step).
    cut_h = {
        "ip": np.full((nr, nz), np.nan),
        "im": np.full((nr, nz), np.nan),
        "jp": np.full((nr, nz), np.nan),
        "jm": np.full((nr, nz), np.nan),
    }

    def _maybe_cut(i, j, di, dj, step, key):
        ni, nj = i + di, j + dj
        if phi[ni, nj] < 0.0:
            return
        t = _cut_fraction(
            level_set,
            float(R[i]),
            float(Z[j]),
            float(R[ni]),
            float(Z[nj]),
            float(phi[i, j]),
            float(phi[ni, nj]),
        )
        cut_h[key][i, j] = t * step

    if level_set is not None:
        ii, jj = np.where(inside)
        for i, j in zip(ii.tolist(), jj.tolist()):
            _maybe_cut(i, j, 1, 0, dR, "ip")
            _maybe_cut(i, j, -1, 0, dR, "im")
            _maybe_cut(i, j, 0, 1, dZ, "jp")
            _maybe_cut(i, j, 0, -1, dZ, "jm")
        tiny = np.zeros((nr, nz), dtype=bool)
        for key, step in (("ip", dR), ("im", dR), ("jp", dZ), ("jm", dZ)):
            h = cut_h[key]
            tiny |= inside & np.isfinite(h) & (h < _TINY_CUT * step)
        inside = inside & ~tiny

    unknown = inside
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diag = np.zeros(n, dtype=float)
    b_fixed = np.zeros(n, dtype=float)

    def add(r, c, v):
        rows.append(r)
        cols.append(c)
        data.append(v)

    # Dirichlet rows (box edge, outside the level set, and tiny-cut nodes).
    dirichlet = ~unknown
    for i, j in zip(*np.where(dirichlet)):
        k = _flatten(i, j, nz)
        diag[k] = 1.0
        b_fixed[k] = g[i, j]

    def g_cut(i, j, di, dj, h_frac_dist, step):
        # Location of the root, a fraction h/step of the way to the neighbour.
        frac = h_frac_dist / step
        rc = float(R[i] + frac * (R[i + di] - R[i]))
        zc = float(Z[j] + frac * (Z[j + dj] - Z[j]))
        if callable(boundary_value):
            return _eval_level(boundary_value, rc, zc)
        return float(boundary_value)

    ii, jj = np.where(unknown)
    for i, j in zip(ii.tolist(), jj.tolist()):
        k = _flatten(i, j, nz)
        Ri = float(R[i])

        # --- radial neighbours ---
        h_m = cut_h["im"][i, j]
        h_p = cut_h["ip"][i, j]
        left_cut = np.isfinite(h_m)
        right_cut = np.isfinite(h_p)
        if not left_cut and not right_cut:
            r_plus = 0.5 * (R[i] + R[i + 1])
            r_minus = 0.5 * (R[i] + R[i - 1])
            a_r = Ri / (dR**2 * r_plus)
            a_l = Ri / (dR**2 * r_minus)
            a_c = -(a_r + a_l)
            add(k, _flatten(i + 1, j, nz), a_r)
            add(k, _flatten(i - 1, j, nz), a_l)
            diag[k] += a_c
        else:
            hm = float(h_m) if left_cut else dR
            hp = float(h_p) if right_cut else dR
            # Non-uniform 3-point formula, exact for quadratics:
            # d²/dR² - (1/R) d/dR.
            a_r = (2.0 - hm / Ri) / (hp * (hm + hp))
            a_l = (2.0 + hp / Ri) / (hm * (hm + hp))
            a_c = (-2.0 - (hp - hm) / Ri) / (hm * hp)
            diag[k] += a_c
            if right_cut:
                b_fixed[k] -= a_r * g_cut(i, j, 1, 0, hp, dR)
            else:
                add(k, _flatten(i + 1, j, nz), a_r)
            if left_cut:
                b_fixed[k] -= a_l * g_cut(i, j, -1, 0, hm, dR)
            else:
                add(k, _flatten(i - 1, j, nz), a_l)

        # --- vertical neighbours (pure second derivative) ---
        h_d = cut_h["jm"][i, j]
        h_u = cut_h["jp"][i, j]
        down_cut = np.isfinite(h_d)
        up_cut = np.isfinite(h_u)
        if not down_cut and not up_cut:
            az = 1.0 / dZ**2
            add(k, _flatten(i, j + 1, nz), az)
            add(k, _flatten(i, j - 1, nz), az)
            diag[k] += -2.0 * az
        else:
            hd = float(h_d) if down_cut else dZ
            hu = float(h_u) if up_cut else dZ
            a_u = 2.0 / ((hd + hu) * hu)
            a_d = 2.0 / ((hd + hu) * hd)
            a_c = -2.0 / (hd * hu)
            diag[k] += a_c
            if up_cut:
                b_fixed[k] -= a_u * g_cut(i, j, 0, 1, hu, dZ)
            else:
                add(k, _flatten(i, j + 1, nz), a_u)
            if down_cut:
                b_fixed[k] -= a_d * g_cut(i, j, 0, -1, hd, dZ)
            else:
                add(k, _flatten(i, j - 1, nz), a_d)

    idx = np.arange(n)
    add_rows = rows + idx.tolist()
    add_cols = cols + idx.tolist()
    add_data = data + diag.tolist()
    matrix = coo_matrix((add_data, (add_rows, add_cols)), shape=(n, n)).tocsc()
    lu = splu(matrix)
    return _LinearOp(lu, b_fixed, unknown)


def solve_linear(
    grid: Grid,
    rhs: np.ndarray,
    level_set: LevelSet | None = None,
    boundary_value: float | Callable[[np.ndarray, np.ndarray], np.ndarray] = 0.0,
) -> np.ndarray:
    """Solve Delta* psi = rhs with Dirichlet data.

    If ``level_set`` is None the domain is the whole rectangle and ``boundary_value`` is
    imposed on its edges. Otherwise the domain is {level_set < 0}; boundary cuts between
    grid nodes are located by root finding on ``level_set`` and treated with the
    Shortley-Weller stencil, so the scheme stays second order on curved boundaries.
    Nodes outside the domain are set to ``boundary_value`` evaluated there (0 by default).
    """
    op = _build_operator(grid, level_set, boundary_value)
    return op.solve(np.asarray(rhs, dtype=float))


def _magnetic_axis(psi: np.ndarray, mask: np.ndarray, grid: Grid) -> tuple[float, float, float]:
    """Maximum of psi inside mask, refined by a local quadratic fit when possible."""
    work = np.where(mask, psi, -np.inf)
    i, j = np.unravel_index(int(np.argmax(work)), psi.shape)
    r_node = float(grid.R[i])
    z_node = float(grid.Z[j])
    psi_node = float(psi[i, j])
    if i == 0 or j == 0 or i == grid.nr - 1 or j == grid.nz - 1:
        return r_node, z_node, psi_node
    patch = mask[i - 1 : i + 2, j - 1 : j + 2]
    if patch.shape == (3, 3) and np.all(patch):
        rr = grid.R[i - 1 : i + 2] - grid.R[i]
        zz = grid.Z[j - 1 : j + 2] - grid.Z[j]
        RR, ZZ = np.meshgrid(rr, zz, indexing="ij")
        r = RR.ravel()
        z = ZZ.ravel()
        p = psi[i - 1 : i + 2, j - 1 : j + 2].ravel()
        design = np.column_stack([np.ones_like(r), r, z, r**2, z**2, r * z])
        coef, *_ = np.linalg.lstsq(design, p, rcond=None)
        a, b, c, d, e, f = (float(v) for v in coef)
        hess = np.array([[2.0 * d, f], [f, 2.0 * e]], dtype=float)
        try:
            dr, dz = np.linalg.solve(hess, np.array([-b, -c], dtype=float))
        except np.linalg.LinAlgError:
            return r_node, z_node, psi_node
        dr = float(dr)
        dz = float(dz)
        if abs(dr) <= grid.dR and abs(dz) <= grid.dZ and d < 0.0 and e < 0.0:
            psi_axis = a + b * dr + c * dz + d * dr**2 + e * dz**2 + f * dr * dz
            return r_node + dr, z_node + dz, float(psi_axis)
    # Separable parabola along the cross, if those neighbours lie in the mask.
    r_axis, z_axis, psi_axis = r_node, z_node, psi_node
    if mask[i - 1, j] and mask[i + 1, j]:
        pl, pc, pr = float(psi[i - 1, j]), float(psi[i, j]), float(psi[i + 1, j])
        denom = pl - 2.0 * pc + pr
        if denom < 0.0:
            dr = 0.5 * (pl - pr) / denom * grid.dR
            dr = float(np.clip(dr, -grid.dR, grid.dR))
            r_axis = r_node + dr
            psi_axis = pc - (pl - pr) ** 2 / (8.0 * denom)
    if mask[i, j - 1] and mask[i, j + 1]:
        pd, pc, pu = float(psi[i, j - 1]), float(psi[i, j]), float(psi[i, j + 1])
        denom = pd - 2.0 * pc + pu
        if denom < 0.0:
            dz = 0.5 * (pd - pu) / denom * grid.dZ
            dz = float(np.clip(dz, -grid.dZ, grid.dZ))
            z_axis = z_node + dz
            corr = -(pd - pu) ** 2 / (8.0 * denom)  # positive
            # If the R-fit already replaced psi_axis, add only the Z correction
            # relative to the nodal value; otherwise use the Z vertex.
            if mask[i - 1, j] and mask[i + 1, j]:
                psi_axis = psi_axis + corr
            else:
                psi_axis = pc + corr
    return r_axis, z_axis, float(psi_axis)


def _profile_current(psi, mask, grid, shape: ShapeParams, profile: ProfileParams, psi_axis: float):
    psi_b = 0.0
    pn = (psi - psi_axis) / (psi_b - psi_axis)
    pn = np.clip(pn, 0.0, 1.0)
    g = (1.0 - pn ** profile.alpha) ** profile.gamma
    R = grid.RR
    bare = np.zeros_like(psi)
    bare[mask] = (
        profile.beta0 * R[mask] / shape.R0 + (1.0 - profile.beta0) * shape.R0 / R[mask]
    ) * g[mask]
    area_int = float(np.sum(bare) * grid.dR * grid.dZ)
    if area_int == 0.0:
        raise RuntimeError("current profile integrates to zero")
    lam = profile.Ip / area_int
    return bare * lam, lam


def solve_fixed_boundary(
    shape: ShapeParams,
    profile: ProfileParams,
    grid: Grid,
    tol: float = 1e-8,
    max_iter: int = 200,
    relax: float = 0.5,
) -> Equilibrium:
    """Picard iteration for the nonlinear fixed-boundary equilibrium."""
    level = shape.level_set()
    rr, zz = grid.RR, grid.ZZ
    mask = np.asarray(level(rr, zz), dtype=float) < 0.0
    area = float(mask.sum() * grid.dR * grid.dZ)
    if area <= 0.0:
        raise RuntimeError("level set does not intersect the grid")
    J = np.zeros((grid.nr, grid.nz), dtype=float)
    J[mask] = profile.Ip / area
    rhs0 = -MU0 * rr * J
    op = _build_operator(grid, level, 0.0)
    psi = op.solve(rhs0)

    history: list[float] = []
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        _r_axis, _z_axis, psi_axis = _magnetic_axis(psi, mask, grid)
        J, _lam = _profile_current(psi, mask, grid, shape, profile, psi_axis)
        rhs = -MU0 * grid.RR * J
        psi_new = op.solve(rhs)
        scale = float(np.max(np.abs(psi_new)))
        rel = float(np.max(np.abs(psi_new - psi)) / max(scale, 1e-30))
        history.append(rel)
        psi = relax * psi_new + (1.0 - relax) * psi
        if rel < tol:
            converged = True
            break

    R_axis, Z_axis, psi_axis = _magnetic_axis(psi, mask, grid)
    J, lam = _profile_current(psi, mask, grid, shape, profile, psi_axis)
    return Equilibrium(
        grid=grid,
        shape=shape,
        profile=profile,
        psi=psi,
        J=J,
        mask=mask,
        psi_axis=float(psi_axis),
        psi_boundary=0.0,
        R_axis=float(R_axis),
        Z_axis=float(Z_axis),
        lam=float(lam),
        n_iter=int(n_iter),
        converged=bool(converged),
        history=history,
    )
