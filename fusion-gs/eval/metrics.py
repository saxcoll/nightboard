"""Evaluation metrics for predicted equilibria.

All functions take arrays on a ``gs.solver.Grid`` with shape (nr, nz), ``indexing="ij"``.

``Delta*`` is the Grad-Shafranov operator

    Delta* psi = R d/dR((1/R) dpsi/dR) + d²psi/dZ²

discretized by a private second-order conservative finite-difference stencil that
uses arithmetic midpoints ``R_{i+1/2} = (R_i + R_{i+1}) / 2``. That stencil is
exact (to roundoff) for polynomials that the centered second difference and the
midpoint radial flux reproduce exactly, including ``psi = a + b Z + c Z² +
(d + e Z) R²`` and ``psi = R⁴/8`` (whose continuous image is ``R²``). On a
general smooth field the truncation error is O(h²).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from matplotlib.path import Path
from scipy.interpolate import RectBivariateSpline

from gs.solver import Grid

# Bicubic upsampling used only to locate flux contours for q. Derivatives still
# come from the spline built on the original grid.
_Q_UPSAMPLE = 4
# Arc-length samples per curve for the symmetric shape error.
_SHAPE_SAMPLES = 400


def relative_l2(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> float:
    """||pred - true|| / ||true|| over ``mask`` (all points if None)."""
    pred_a = np.asarray(pred, dtype=float)
    true_a = np.asarray(true, dtype=float)
    if mask is None:
        diff = pred_a.ravel() - true_a.ravel()
        ref = true_a.ravel()
    else:
        m = np.asarray(mask, dtype=bool)
        diff = pred_a[m] - true_a[m]
        ref = true_a[m]
    num = float(np.linalg.norm(diff))
    den = float(np.linalg.norm(ref))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def _delta_star(psi: np.ndarray, grid: Grid) -> np.ndarray:
    """Second-order conservative Delta* psi. Edge rows and columns are NaN.

    Radial piece, at nodes i = 1 .. nr-2::

        flux_{i+1/2} = (psi_{i+1} - psi_i) / (R_{i+1/2} dR)
        radial_i = R_i * (flux_{i+1/2} - flux_{i-1/2}) / dR

    which is ``R ∂/∂R((1/R) ∂psi/∂R)``. The vertical piece is the standard
    three-point second difference.
    """
    psi = np.asarray(psi, dtype=float)
    if psi.shape != (grid.nr, grid.nz):
        raise ValueError(f"psi shape {psi.shape} != {(grid.nr, grid.nz)}")
    dR = float(grid.dR)
    dZ = float(grid.dZ)
    if dR == 0.0 or dZ == 0.0:
        raise ValueError("grid spacing must be non-zero")
    R = np.asarray(grid.R, dtype=float)
    out = np.full(psi.shape, np.nan, dtype=float)
    if grid.nr < 3 or grid.nz < 3:
        return out
    r_half = 0.5 * (R[:-1] + R[1:])
    flux = (psi[1:, :] - psi[:-1, :]) / (r_half[:, None] * dR)
    radial = R[1:-1, None] * (flux[1:, :] - flux[:-1, :]) / dR
    d2z = (psi[:, 2:] - 2.0 * psi[:, 1:-1] + psi[:, :-2]) / (dZ * dZ)
    out[1:-1, 1:-1] = radial[:, 1:-1] + d2z[1:-1, :]
    return out


def _interior_nodes(mask: np.ndarray) -> np.ndarray:
    """Nodes where ``mask`` is true, all four orthogonal neighbours are true, and
    the node is not on the array edge."""
    mask = np.asarray(mask, dtype=bool)
    interior = np.zeros(mask.shape, dtype=bool)
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return interior
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return interior


def gs_residual(psi: np.ndarray, grid: Grid, rhs: np.ndarray, mask: np.ndarray) -> float:
    """Normalized residual ||Delta* psi - rhs|| / ||rhs|| over interior plasma nodes.

    Only nodes whose four neighbours are also inside ``mask`` are used, so the
    irregular boundary does not dominate the value. If ``||rhs||`` on those nodes
    is zero, the absolute residual ``||Delta* psi - rhs||`` is returned instead.
    """
    rhs_a = np.asarray(rhs, dtype=float)
    if rhs_a.shape != (grid.nr, grid.nz):
        raise ValueError(f"rhs shape {rhs_a.shape} != {(grid.nr, grid.nz)}")
    mask_a = np.asarray(mask, dtype=bool)
    if mask_a.shape != psi.shape:
        raise ValueError(f"mask shape {mask_a.shape} != psi shape {psi.shape}")
    interior = _interior_nodes(mask_a)
    if not np.any(interior):
        raise ValueError("no interior nodes with four in-mask neighbours")
    delta = _delta_star(psi, grid)
    diff = delta[interior] - rhs_a[interior]
    if not np.all(np.isfinite(diff)):
        raise ValueError("Delta* psi is not finite on the selected nodes")
    num = float(np.linalg.norm(diff))
    den = float(np.linalg.norm(rhs_a[interior]))
    # A homogeneous right-hand side has no relative scale. Return the absolute
    # residual so roundoff in an exact null field stays a small number rather
    # than becoming infinite.
    if den == 0.0:
        return num
    return num / den


def _local_extremum_indices(psi: np.ndarray, mask: np.ndarray, want_max: bool) -> list[tuple[int, int]]:
    """Indices of strict-or-tied local extrema whose full 3×3 neighbourhood lies in ``mask``."""
    if psi.shape[0] < 3 or psi.shape[1] < 3:
        return []
    m = mask
    core = (
        m[1:-1, 1:-1]
        & m[:-2, 1:-1]
        & m[2:, 1:-1]
        & m[1:-1, :-2]
        & m[1:-1, 2:]
        & m[:-2, :-2]
        & m[:-2, 2:]
        & m[2:, :-2]
        & m[2:, 2:]
    )
    center = psi[1:-1, 1:-1]
    neighbours = (
        psi[:-2, 1:-1],
        psi[2:, 1:-1],
        psi[1:-1, :-2],
        psi[1:-1, 2:],
        psi[:-2, :-2],
        psi[:-2, 2:],
        psi[2:, :-2],
        psi[2:, 2:],
    )
    is_ext = core.copy()
    for nb in neighbours:
        if want_max:
            is_ext &= center >= nb
        else:
            is_ext &= center <= nb
    ii, jj = np.nonzero(is_ext)
    return list(zip((ii + 1).tolist(), (jj + 1).tolist()))


def _refine_quadratic(
    psi: np.ndarray, grid: Grid, i: int, j: int
) -> tuple[float, float, float] | None:
    """Sub-grid extremum from a 3×3 least-squares quadratic, or None if rejected.

    The fit is rejected when the Hessian is not a definite extremum (saddle or
    rank-deficient), the normal equations are ill-conditioned, or the critical
    point lies outside the 3×3 cell. Coordinates are local to the centre node
    so the monomial basis stays well scaled.
    """
    if i < 1 or j < 1 or i > grid.nr - 2 or j > grid.nz - 2:
        return None
    x0 = float(grid.R[i])
    y0 = float(grid.Z[j])
    xs = np.asarray(grid.R[i - 1 : i + 2], dtype=float) - x0
    ys = np.asarray(grid.Z[j - 1 : j + 2], dtype=float) - y0
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    samples = np.asarray(psi[i - 1 : i + 2, j - 1 : j + 2], dtype=float).ravel()
    xr = xx.ravel()
    yr = yy.ravel()
    design = np.column_stack(
        [xr * xr, yr * yr, xr * yr, xr, yr, np.ones(xr.size)]
    )
    coef, _, rank, _ = np.linalg.lstsq(design, samples, rcond=None)
    if rank < 6:
        return None
    a, b, c, d, e, f = (float(v) for v in coef)
    hessian = np.array([[2.0 * a, c], [c, 2.0 * b]], dtype=float)
    # det > 0 and a != 0 ⇒ definite (max if a < 0, min if a > 0). det <= 0 is a
    # saddle or a degenerate direction; treat that as an ill-conditioned axis fit.
    det = float(hessian[0, 0] * hessian[1, 1] - hessian[0, 1] * hessian[1, 0])
    scale = max(abs(a), abs(b), abs(c), 1e-30)
    if det <= 0.0 or abs(det) < 1e-12 * scale * scale:
        return None
    try:
        cond = float(np.linalg.cond(hessian))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(cond) or cond > 1e10:
        return None
    try:
        sol = np.linalg.solve(hessian, np.array([-d, -e], dtype=float))
    except np.linalg.LinAlgError:
        return None
    x, y = float(sol[0]), float(sol[1])
    r_ax = x0 + x
    z_ax = y0 + y
    r_lo = float(grid.R[i - 1])
    r_hi = float(grid.R[i + 1])
    z_lo = float(grid.Z[j - 1])
    z_hi = float(grid.Z[j + 1])
    tol_r = 1e-9 * max(1.0, abs(r_lo), abs(r_hi))
    tol_z = 1e-9 * max(1.0, abs(z_lo), abs(z_hi))
    if not (r_lo - tol_r <= r_ax <= r_hi + tol_r and z_lo - tol_z <= z_ax <= z_hi + tol_z):
        return None
    psi_ax = a * x * x + b * y * y + c * x * y + d * x + e * y + f
    return float(r_ax), float(z_ax), float(psi_ax)


def _global_extremum(psi: np.ndarray, mask: np.ndarray, want_max: bool) -> tuple[int, int]:
    filled = np.where(mask, psi, -np.inf if want_max else np.inf)
    flat = int(np.argmax(filled) if want_max else np.argmin(filled))
    i, j = np.unravel_index(flat, psi.shape)
    return int(i), int(j)


def find_magnetic_axis(psi: np.ndarray, grid: Grid, mask: np.ndarray) -> tuple[float, float, float]:
    """(R_axis, Z_axis, psi_axis) of the extremum of psi inside ``mask``, refined to
    sub-grid accuracy by a local quadratic fit. The extremum is the maximum if psi is
    mostly positive inside the mask and the minimum otherwise.

    The search uses interior local extrema (full 3×3 neighbourhood inside ``mask``),
    so a boundary node of the mask is not mistaken for the axis. If the mask contains
    no local extremum of the preferred type, the other type is used — a negative
    field whose only critical point is a maximum (for example ``psi = -r²``) still
    returns that maximum. If the quadratic fit is ill-conditioned, is not a strict
    extremum, or leaves the 3×3 cell, the discrete node is returned instead.
    """
    psi_a = np.asarray(psi, dtype=float)
    if psi_a.shape != (grid.nr, grid.nz):
        raise ValueError(f"psi shape {psi_a.shape} != {(grid.nr, grid.nz)}")
    mask_a = np.asarray(mask, dtype=bool)
    if mask_a.shape != psi_a.shape:
        raise ValueError(f"mask shape {mask_a.shape} != psi shape {psi_a.shape}")
    if not np.any(mask_a):
        raise ValueError("mask is empty")
    want_max = float(np.mean(psi_a[mask_a])) > 0.0

    def _from_local(prefer_max: bool) -> tuple[tuple[float, float, float] | None, tuple[int, int] | None]:
        cands = _local_extremum_indices(psi_a, mask_a, prefer_max)
        if not cands:
            return None, None
        vals = np.array([psi_a[i, j] for i, j in cands], dtype=float)
        order = np.argsort(vals, kind="stable")
        if prefer_max:
            order = order[::-1]
        best: tuple[int, int] | None = None
        for k in order:
            i, j = cands[int(k)]
            if best is None:
                best = (i, j)
            refined = _refine_quadratic(psi_a, grid, i, j)
            if refined is not None:
                return refined, best
        return None, best

    refined, best = _from_local(want_max)
    if refined is not None:
        return refined
    if best is None:
        refined, best = _from_local(not want_max)
        if refined is not None:
            return refined
    if best is None:
        best = _global_extremum(psi_a, mask_a, want_max)
    i, j = best
    return float(grid.R[i]), float(grid.Z[j]), float(psi_a[i, j])


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _is_closed(points: np.ndarray, tol: float) -> bool:
    if points.shape[0] < 4:
        return False
    return float(np.hypot(points[0, 0] - points[-1, 0], points[0, 1] - points[-1, 1])) <= tol


def flux_contour(
    psi: np.ndarray, grid: Grid, level: float, R_axis: float, Z_axis: float
) -> tuple[np.ndarray, np.ndarray]:
    """Closed contour psi = level surrounding the magnetic axis, as (R, Z) arrays.

    Among closed contour lines that enclose ``(R_axis, Z_axis)``, the one with the
    smallest area is returned. The curve is closed: the first point equals the last.
    Orientation is whatever contourpy produces and is not guaranteed to be
    counter-clockwise. Raises ``ValueError`` if no such contour exists.
    """
    import contourpy

    psi_a = np.asarray(psi, dtype=float)
    if psi_a.shape != (grid.nr, grid.nz):
        raise ValueError(f"psi shape {psi_a.shape} != {(grid.nr, grid.nz)}")
    # contourpy wants z[ny, nx] with x along columns. psi[i, j] is (R[i], Z[j]),
    # so psi.T[j, i] lines up with x = R, y = Z.
    generator = contourpy.contour_generator(
        x=np.asarray(grid.R, dtype=float),
        y=np.asarray(grid.Z, dtype=float),
        z=np.ascontiguousarray(psi_a.T),
    )
    lines = generator.lines(float(level))
    spacing = max(abs(float(grid.dR)), abs(float(grid.dZ)), 1e-15)
    close_tol = 1e-8 * spacing
    best_area = np.inf
    best: np.ndarray | None = None
    axis = (float(R_axis), float(Z_axis))
    for line in lines:
        pts = np.asarray(line, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or not _is_closed(pts, close_tol):
            continue
        if not np.array_equal(pts[0], pts[-1]):
            pts = np.vstack([pts[:-1], pts[0]])
        if not Path(pts).contains_point(axis):
            continue
        area = _polygon_area(pts)
        if area <= 0.0 or not np.isfinite(area):
            continue
        if area < best_area:
            best_area = area
            best = pts
    if best is None:
        raise ValueError(
            f"no closed contour at level {level} encloses the magnetic axis ({R_axis}, {Z_axis})"
        )
    return best[:, 0].copy(), best[:, 1].copy()


def _open_ring(R: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop a duplicated closing vertex and any consecutive duplicates."""
    R = np.asarray(R, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    if R.size != Z.size or R.size < 2:
        raise ValueError("curve R and Z must be 1-d arrays of equal length >= 2")
    gap = float(np.hypot(R[0] - R[-1], Z[0] - Z[-1]))
    scale = 1.0 + float(np.hypot(R[0], Z[0]))
    if gap <= 1e-12 * scale:
        R = R[:-1]
        Z = Z[:-1]
    if R.size < 2:
        raise ValueError("curve is degenerate")
    keep = [0]
    for i in range(1, R.size):
        if np.hypot(R[i] - R[keep[-1]], Z[i] - Z[keep[-1]]) > 0.0:
            keep.append(i)
    R = R[np.array(keep)]
    Z = Z[np.array(keep)]
    if R.size < 2:
        raise ValueError("curve is degenerate")
    return R, Z


def _resample_closed(R: np.ndarray, Z: np.ndarray, n: int = _SHAPE_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """Uniform arc-length samples of a closed curve, without repeating the start."""
    R, Z = _open_ring(R, Z)
    dR = np.diff(R, append=R[0])
    dZ = np.diff(Z, append=Z[0])
    seg = np.hypot(dR, dZ)
    # Drop zero-length closing remnants so the parameter is strictly increasing.
    positive = seg > 0.0
    if not np.all(positive):
        # seg[i] is the step leaving vertex i. A zero step means vertex i+1 (mod)
        # duplicates vertex i; drop the destination, except for the closing step.
        idx = [i for i, good in enumerate(positive[:-1]) if good or i == 0]
        if not idx:
            idx = [0]
        R = R[np.array(idx)]
        Z = Z[np.array(idx)]
        dR = np.diff(R, append=R[0])
        dZ = np.diff(Z, append=Z[0])
        seg = np.hypot(dR, dZ)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total == 0.0:
        return np.full(n, R[0]), np.full(n, Z[0])
    query = np.linspace(0.0, total, n, endpoint=False)
    return np.interp(query, s, np.append(R, R[0])), np.interp(query, s, np.append(Z, Z[0]))


def _mean_distance_to_polyline(R_from, Z_from, R_to, Z_to) -> float:
    """Mean Euclidean distance from sample points to a closed polyline."""
    if np.hypot(R_to[0] - R_to[-1], Z_to[0] - Z_to[-1]) > 0.0:
        R_to = np.append(R_to, R_to[0])
        Z_to = np.append(Z_to, Z_to[0])
    x1 = R_to[:-1]
    y1 = Z_to[:-1]
    dx = np.diff(R_to)
    dy = np.diff(Z_to)
    len2 = dx * dx + dy * dy
    px = R_from[:, None] - x1[None, :]
    py = Z_from[:, None] - y1[None, :]
    denom = np.maximum(len2, 1e-30)
    t = np.clip((px * dx + py * dy) / denom, 0.0, 1.0)
    # Zero-length segments: t is meaningless; the clip still lands on the vertex.
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    dist = np.hypot(R_from[:, None] - nearest_x, Z_from[:, None] - nearest_y)
    return float(dist.min(axis=1).mean())


def lcfs_shape_error(R_pred, Z_pred, R_true, Z_true) -> float:
    """Symmetric mean distance (metres) between two closed boundary curves.

    Each curve is resampled to 400 points uniformly in arc length. The score is
    the average of the two directed mean nearest-point distances (point to
    polyline). Units are metres.
    """
    a_r, a_z = _resample_closed(R_pred, Z_pred)
    b_r, b_z = _resample_closed(R_true, Z_true)
    forward = _mean_distance_to_polyline(a_r, a_z, b_r, b_z)
    backward = _mean_distance_to_polyline(b_r, b_z, a_r, a_z)
    return 0.5 * (forward + backward)


def _eval_F(F: Callable[[np.ndarray], np.ndarray], psin_levels: np.ndarray) -> np.ndarray:
    """Evaluate ``F`` on every psi_n level.

    ``F`` is called once with the 1-d array of levels. A scalar return is
    broadcast (constant ``F``). An array return must have one value per level.
    """
    raw = np.asarray(F(np.asarray(psin_levels, dtype=float)), dtype=float)
    if raw.ndim == 0 or raw.size == 1:
        return np.full(psin_levels.shape, float(np.reshape(raw, ())), dtype=float)
    if raw.shape == psin_levels.shape:
        return raw
    if raw.size == psin_levels.size:
        return np.reshape(raw, psin_levels.shape)
    raise ValueError(
        f"F(psin) returned shape {raw.shape}, expected {psin_levels.shape} or a scalar"
    )


def _contour_integral(R: np.ndarray, Z: np.ndarray, Bp: np.ndarray) -> float:
    """∮ dl / (R² B_p) using segment midpoints. ``dl`` and ``B_p`` are non-negative."""
    if R.size < 2:
        raise ValueError("contour is too short to integrate")
    if np.hypot(R[0] - R[-1], Z[0] - Z[-1]) > 0.0:
        R = np.append(R, R[0])
        Z = np.append(Z, Z[0])
        Bp = np.append(Bp, Bp[0])
    dR = np.diff(R)
    dZ = np.diff(Z)
    dl = np.hypot(dR, dZ)
    r_mid = 0.5 * (R[:-1] + R[1:])
    bp_mid = 0.5 * (Bp[:-1] + Bp[1:])
    if np.any(bp_mid <= 0.0) or np.any(r_mid == 0.0):
        raise ValueError("B_p vanished on a flux contour (axis or X-point in the contour)")
    return float(np.sum(dl / (r_mid * r_mid * bp_mid)))


def q_profile(
    psi: np.ndarray,
    grid: Grid,
    psi_axis: float,
    psi_boundary: float,
    R_axis: float,
    Z_axis: float,
    F: Callable[[np.ndarray], np.ndarray],
    psin_levels: np.ndarray,
) -> np.ndarray:
    """Safety factor q(psi_n) = F / (2 pi) * closed-integral dl / (R^2 B_p) on each level.

    ``F`` maps psi_n to F = R B_phi. B_p = |grad psi| / R is interpolated from a
    bicubic spline of psi.

    ``psin_levels`` of exactly 0 (axis) or 1 (boundary) are not supported; use
    levels in the open interval (0, 1), preferably (0.02, 0.98). Each level is
    contoured on a 4× bicubic upsample of ``psi`` so the polygon tracks the
    surface more closely than the original mesh. The line integral uses segment
    midpoints. Because both ``dl`` and ``B_p`` are taken non-negative, the result
    does not depend on contour orientation; the sign of ``q`` is the sign of ``F``.
    """
    psi_a = np.asarray(psi, dtype=float)
    if psi_a.shape != (grid.nr, grid.nz):
        raise ValueError(f"psi shape {psi_a.shape} != {(grid.nr, grid.nz)}")
    if grid.nr < 4 or grid.nz < 4:
        raise ValueError("q_profile needs at least 4 points in each direction for a bicubic spline")
    psin = np.asarray(psin_levels, dtype=float)
    if psin.size == 0:
        return np.empty(psin.shape, dtype=float)
    if np.any((psin <= 0.0) | (psin >= 1.0)):
        raise ValueError("psin_levels must lie in (0, 1); 0 and 1 are not supported")
    f_vals = _eval_F(F, psin)
    spline = RectBivariateSpline(
        np.asarray(grid.R, dtype=float),
        np.asarray(grid.Z, dtype=float),
        np.ascontiguousarray(psi_a),
        kx=3,
        ky=3,
    )
    n_r = (grid.nr - 1) * _Q_UPSAMPLE + 1
    n_z = (grid.nz - 1) * _Q_UPSAMPLE + 1
    r_fine = np.linspace(float(grid.R[0]), float(grid.R[-1]), n_r)
    z_fine = np.linspace(float(grid.Z[0]), float(grid.Z[-1]), n_z)
    psi_fine = np.asarray(spline(r_fine, z_fine), dtype=float)
    grid_fine = Grid(r_fine, z_fine)
    psi_axis_f = float(psi_axis)
    psi_boundary_f = float(psi_boundary)
    q = np.empty(psin.shape, dtype=float)
    for index, psin_i in enumerate(psin.ravel()):
        level = psi_axis_f + float(psin_i) * (psi_boundary_f - psi_axis_f)
        r_c, z_c = flux_contour(psi_fine, grid_fine, level, float(R_axis), float(Z_axis))
        dpsi_dR = np.asarray(spline.ev(r_c, z_c, dx=1, dy=0), dtype=float)
        dpsi_dZ = np.asarray(spline.ev(r_c, z_c, dx=0, dy=1), dtype=float)
        bp = np.hypot(dpsi_dR, dpsi_dZ) / r_c
        integral = _contour_integral(r_c, z_c, bp)
        q.ravel()[index] = float(f_vals.ravel()[index]) / (2.0 * np.pi) * integral
    return q


def q95(psi, grid, psi_axis, psi_boundary, R_axis, Z_axis, F) -> float:
    """Safety factor at psi_n = 0.95.

    Same conventions as ``q_profile``. ``psin = 0.95`` is inside the supported
    open interval; the magnetic axis and the boundary itself are not.
    """
    q = q_profile(
        psi,
        grid,
        psi_axis,
        psi_boundary,
        R_axis,
        Z_axis,
        F,
        np.array([0.95], dtype=float),
    )
    return float(q[0])


def time_call(fn: Callable[[], object], repeats: int = 5) -> float:
    """Median wall-clock seconds of ``fn()`` over ``repeats`` calls (after one warm-up)."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    fn()
    samples = np.empty(repeats, dtype=float)
    for i in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples[i] = time.perf_counter() - t0
    return float(np.median(samples))
