"""Figures for fixed-boundary Grad–Shafranov equilibria.

From the project directory::

    python -m eval.visualize --out outputs/visualize

The default case is R0=1.7 m, a=0.6 m, kappa=1.7, delta=0.3, Ip=1 MA,
beta0=0.5, alpha=1, gamma=2, B0=2 T. Override any of those with the matching
flag, and the mesh size with ``--grid-n`` (default 129).

Outputs (under ``--out``):

* ``equilibrium.png`` — flux, current, profiles, and the scalar summary
* ``surrogate_vs_solver.png`` — best surrogate on the 65×65 training grid
* ``pinn_vs_solver.png`` — nonlinear PINN on its evaluation box
* ``kappa_sweep.gif``, ``delta_sweep.gif`` — shape scans, solver flux surfaces
  with the surrogate dashed on top when that checkpoint is present

Checkpoints that are not on disk are skipped.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from eval.metrics import find_magnetic_axis, flux_contour, q_profile, relative_l2
from gs.solver import MU0, Grid, ProfileParams, ShapeParams, solve_fixed_boundary

# Default showcase equilibrium. Same shape and profiles as models.pinn.NONLINEAR_*.
DEFAULT_SHAPE = dict(R0=1.7, a=0.6, kappa=1.7, delta=0.3)
DEFAULT_PROFILE = dict(Ip=1.0e6, beta0=0.5, alpha=1.0, gamma=2.0, B0=2.0)

# Surrogate training box (data.generate.fixed_grid). Do not change this when
# comparing a checkpoint: the network was trained on exactly this mesh.
SURROGATE_GRID = (0.80, 2.76, -1.90, 1.90, 65, 65)

_RC = {
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": None,
    "axes.facecolor": "white",
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
    "xtick.color": "#1a1a1a",
    "ytick.color": "#1a1a1a",
    "axes.edgecolor": "#2b2b2b",
    "mathtext.default": "regular",
}

_FLUX_LEVELS = np.linspace(0.1, 0.9, 9)
_SOLVER_COLOR = "#161616"
_MODEL_COLOR = "#d62728"
_PSI_CMAP = "viridis"
_J_CMAP = "magma"
_ERR_CMAP = "inferno"


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_file(relative: str) -> Path | None:
    rel = Path(relative)
    if rel.is_file():
        return rel
    candidate = _project_root() / rel
    if candidate.is_file():
        return candidate
    return None


def _is_axes(obj) -> bool:
    from matplotlib.axes import Axes

    return isinstance(obj, Axes)


def _ceil_sig(x: float, sig: int = 2) -> float:
    """Smallest number with ``sig`` significant figures that is >= x."""
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        return 1.0
    exp = math.floor(math.log10(x))
    mant = x / 10.0**exp
    factor = 10 ** (sig - 1)
    mant_c = math.ceil(mant * factor - 1e-12) / factor
    return float(mant_c * 10.0**exp)


def _nice_ceil(x: float) -> float:
    """Round a positive magnitude up to a 1–2–2.5–5 sequence."""
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        return 1.0
    exp = math.floor(math.log10(x))
    frac = x / 10.0**exp
    for step in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if frac <= step + 1e-12:
            return float(step * 10.0**exp)
    return float(10.0 * 10.0**exp)


def _view_from_mask(mask: np.ndarray, R: np.ndarray, Z: np.ndarray, pad_frac: float = 0.10):
    ii, jj = np.nonzero(mask)
    if ii.size == 0:
        raise ValueError("mask is empty")
    r0, r1 = float(R[ii.min()]), float(R[ii.max()])
    z0, z1 = float(Z[jj.min()]), float(Z[jj.max()])
    pad = max(r1 - r0, z1 - z0, 1e-3) * float(pad_frac)
    return (r0 - pad, r1 + pad, z0 - pad, z1 + pad)


def _union_view(views, pad_frac: float = 0.08):
    r0 = min(v[0] for v in views)
    r1 = max(v[1] for v in views)
    z0 = min(v[2] for v in views)
    z1 = max(v[3] for v in views)
    # views already include a pad; add a little more so labels and the LCFS
    # are not flush with the spines, and keep that pad identical every frame.
    pad = max(r1 - r0, z1 - z0, 1e-3) * float(pad_frac) * 0.35
    return (r0 - pad, r1 + pad, z0 - pad, z1 + pad)


def _lcfs_polyline(eq) -> tuple[np.ndarray, np.ndarray] | None:
    phi = np.asarray(eq.shape.level_set()(eq.grid.RR, eq.grid.ZZ), dtype=float)
    try:
        r, z = flux_contour(phi, eq.grid, 0.0, float(eq.R_axis), float(eq.Z_axis))
    except ValueError:
        return None
    return np.asarray(r, dtype=float), np.asarray(z, dtype=float)


def _clip_to_lcfs(artist, ax, lcfs):
    if lcfs is None:
        return
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    r, z = lcfs
    patch = PathPatch(Path(np.column_stack([r, z])), transform=ax.transData)
    artist.set_clip_path(patch)


def _style_map_axes(ax, view):
    r0, r1, z0, z1 = view
    ax.set_xlim(r0, r1)
    ax.set_ylim(z0, z1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.tick_params(direction="in", top=True, right=True, length=3.5, width=0.7)
    ax.margins(0.0)


def _point_at_angle(segments, r0: float, z0: float, angle: float, max_dang: float = 0.45):
    best = None
    best_dang = float(max_dang)
    origin = np.array([r0, z0], dtype=float)
    for seg in segments:
        seg = np.asarray(seg, dtype=float)
        if seg.ndim != 2 or seg.shape[0] == 0:
            continue
        rel = seg - origin
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        dang = np.abs(np.arctan2(np.sin(ang - angle), np.cos(ang - angle)))
        k = int(np.argmin(dang))
        if float(dang[k]) < best_dang:
            best_dang = float(dang[k])
            best = seg[k]
    if best is None:
        return None
    return float(best[0]), float(best[1])


# Poloidal angles (radians, 0 = outboard) for ψ_n = 0.1 … 0.9. Inner surfaces
# are small, so consecutive labels sit on opposite sides of the axis.
_LABEL_ANGLES = np.array([1.05, -1.10, 0.58, -0.72, 0.26, -0.32, 1.38, -1.40, 0.05])


def _label_flux_surfaces(ax, contour_set, r_axis: float, z_axis: float):
    """Place one ψ_n label on each surface, kept off the magnetic-axis marker."""
    import matplotlib.patheffects as pe

    levels = [float(v) for v in np.asarray(contour_set.levels, dtype=float)]
    n = len(levels)
    if n == 0:
        return
    if n == _LABEL_ANGLES.size:
        angles = _LABEL_ANGLES
    else:
        angles = np.linspace(-1.2, 1.2, n)
    stroke = [pe.withStroke(linewidth=2.6, foreground="white")]
    placed: list[tuple[float, float]] = [(r_axis, z_axis)]
    for level, segs, angle in zip(levels, contour_set.allsegs, angles):
        if not segs:
            continue
        point = None
        for extra in (0.0, 0.35, -0.35, 0.7, -0.7):
            candidate = _point_at_angle(segs, r_axis, z_axis, float(angle + extra))
            if candidate is None:
                continue
            if all(math.hypot(candidate[0] - px, candidate[1] - py) >= 0.07 for px, py in placed):
                point = candidate
                break
        if point is None:
            continue
        placed.append(point)
        text = ax.text(
            point[0],
            point[1],
            f"{level:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            color="#141414",
            zorder=8,
            clip_on=True,
        )
        text.set_path_effects(stroke)


def _draw_lcfs(ax, lcfs, *, color="#111111", lw=2.05, zorder=5):
    if lcfs is None:
        return
    r, z = lcfs
    ax.plot(r, z, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=zorder)


def _draw_axis_marker(ax, r: float, z: float, *, color="#111111"):
    ax.plot(
        r,
        z,
        marker="o",
        ms=7.5,
        mfc="white",
        mec=color,
        mew=1.15,
        linestyle="none",
        zorder=7,
    )
    ax.plot(r, z, marker="+", ms=11, color=color, mew=1.05, linestyle="none", zorder=8)


def _draw_flux_field(ax, eq, view, lcfs, *, label_surfaces: bool, title: str | None = None):
    """Filled ψ_n, labelled surfaces, LCFS, and the magnetic axis."""
    grid = eq.grid
    pn = np.asarray(eq.psin(), dtype=float)
    fill = ax.contourf(
        grid.R,
        grid.Z,
        pn.T,
        levels=np.linspace(0.0, 1.0, 33),
        cmap=_PSI_CMAP,
        vmin=0.0,
        vmax=1.0,
        zorder=1,
    )
    _clip_to_lcfs(fill, ax, lcfs)
    masked = np.ma.masked_where(~np.asarray(eq.mask, dtype=bool), pn)
    ax.contour(
        grid.R,
        grid.Z,
        masked.T,
        levels=_FLUX_LEVELS,
        colors="white",
        linewidths=1.9,
        zorder=3,
    )
    lines = ax.contour(
        grid.R,
        grid.Z,
        masked.T,
        levels=_FLUX_LEVELS,
        colors=_SOLVER_COLOR,
        linewidths=0.85,
        zorder=4,
    )
    if label_surfaces:
        _label_flux_surfaces(ax, lines, float(eq.R_axis), float(eq.Z_axis))
    _draw_lcfs(ax, lcfs)
    _draw_axis_marker(ax, float(eq.R_axis), float(eq.Z_axis))
    _style_map_axes(ax, view)
    if title:
        ax.set_title(title)
    return fill


def _draw_current(ax, eq, view, lcfs):
    grid = eq.grid
    j_ma = np.asarray(eq.J, dtype=float) / 1.0e6
    finite = j_ma[np.asarray(eq.mask, dtype=bool)]
    peak = float(np.max(finite)) if finite.size else 1.0
    vmax = _nice_ceil(peak)
    mesh = ax.pcolormesh(
        grid.R,
        grid.Z,
        j_ma.T,
        cmap=_J_CMAP,
        shading="auto",
        vmin=0.0,
        vmax=vmax,
        zorder=1,
    )
    _clip_to_lcfs(mesh, ax, lcfs)
    # Light rim so the dark magma edge stays visible on a white page.
    _draw_lcfs(ax, lcfs, color="white", lw=2.4, zorder=5)
    _draw_lcfs(ax, lcfs, color="#111111", lw=0.9, zorder=6)
    _draw_axis_marker(ax, float(eq.R_axis), float(eq.Z_axis))
    _style_map_axes(ax, view)
    ax.set_title(r"(b)  toroidal current")
    return mesh


def _profile_axes_style(ax):
    ax.tick_params(direction="in", top=True, right=True, length=3.2, width=0.7)
    ax.grid(True, which="major", color="#d5d5d5", lw=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(r"$\psi_n$")


def _draw_profiles(ax_p, ax_f, ax_q, summary):
    s = summary["s"]
    pressure = summary["pressure_kpa"]
    ax_p.fill_between(s, 0.0, pressure, color="#c45c26", alpha=0.22, linewidth=0, zorder=1)
    ax_p.plot(s, pressure, color="#9a3412", lw=1.7, zorder=2)
    ax_p.set_ylabel("pressure [kPa]")
    ax_p.set_title("(c)  profiles")
    pmax = float(np.max(pressure)) if pressure.size else 1.0
    ax_p.set_ylim(0.0, max(pmax * 1.12, 1e-3))
    _profile_axes_style(ax_p)

    F = summary["F"]
    ax_f.plot(s, F, color="#1d4e89", lw=1.7, zorder=2)
    ax_f.set_ylabel(r"$F = RB_\phi$  [T m]")
    lo, hi = float(np.min(F)), float(np.max(F))
    pad = max((hi - lo) * 0.25, 0.01)
    ax_f.set_ylim(lo - pad, hi + pad)
    _profile_axes_style(ax_f)

    q = summary["q"]
    q_s = summary["q_s"]
    if q is None:
        ax_q.text(0.5, 0.5, "q profile unavailable", ha="center", va="center", transform=ax_q.transAxes)
        ax_q.set_ylabel(r"$q$")
        _profile_axes_style(ax_q)
        return
    ax_q.plot(q_s, q, color="#0f6b4c", lw=1.7, zorder=2)
    q95 = float(summary["q95"])
    ax_q.axvline(0.95, color="#8a8a8a", lw=0.7, ls="--", zorder=1)
    ax_q.plot([0.95], [q95], marker="o", ms=5.0, color="#9b2226", linestyle="none", zorder=3)
    y_off = 8 if q95 < float(np.max(q)) * 0.92 else -14
    ax_q.annotate(
        f"$q_{{95}} = {q95:.2f}$",
        xy=(0.95, q95),
        xytext=(-46, y_off),
        textcoords="offset points",
        fontsize=8,
        color="#9b2226",
        zorder=4,
    )
    ax_q.set_ylabel(r"$q$")
    qmax = float(np.max(q))
    ax_q.set_ylim(0.0, max(qmax * 1.18, 0.1))
    _profile_axes_style(ax_q)


def _fmt_num(x: float, digits: int) -> str:
    return f"{float(x):.{digits}f}"


def _fmt_coord(x: float) -> str:
    """Four decimals, without a signed zero from roundoff about the midplane."""
    value = float(x)
    if abs(value) < 5e-4:
        return "0.0000"
    return f"{value:.4f}"


def _card_rows(eq, summary) -> list[tuple]:
    shape = eq.shape
    profile = eq.profile
    shift = float(summary["shift"])
    q95 = summary["q95"]
    beta = summary["beta_p"]
    q95_txt = f"{q95:.3f}" if np.isfinite(q95) else "n/a"
    beta_txt = f"{beta:.3f}" if np.isfinite(beta) else "n/a"
    status = "converged" if eq.converged else "not converged"
    ip_ma = profile.Ip / 1.0e6
    ip_txt = f"{ip_ma:.3f} MA" if abs(ip_ma) >= 0.01 else f"{profile.Ip:.3e} A"
    return [
        ("section", "Geometry"),
        ("item", r"$R_0$", f"{_fmt_num(shape.R0, 3)} m"),
        ("item", r"$a$", f"{_fmt_num(shape.a, 3)} m"),
        ("item", r"$\kappa$", _fmt_num(shape.kappa, 3)),
        ("item", r"$\delta$", _fmt_num(shape.delta, 3)),
        ("section", "Profiles"),
        ("item", r"$I_p$", ip_txt),
        ("item", r"$B_0$", f"{_fmt_num(profile.B0, 2)} T"),
        ("item", r"$\beta_0,\;\alpha,\;\gamma$", f"{profile.beta0:.2f},  {profile.alpha:.2f},  {profile.gamma:.2f}"),
        ("section", "Solution"),
        ("item", r"$\psi_{\mathrm{axis}}$", f"{eq.psi_axis:.4f} Wb/rad"),
        ("item", "magnetic axis", f"({_fmt_coord(eq.R_axis)}, {_fmt_coord(eq.Z_axis)}) m"),
        ("item", r"$R_{\mathrm{axis}}-R_0$", f"{shift:+.4f} m"),
        ("item", r"$q_{95}$", q95_txt),
        ("item", r"$\beta_p$", beta_txt),
        ("item", "iterations", f"{eq.n_iter}, {status}"),
    ]


def _draw_card(ax, eq, summary):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_facecolor("#f5f6f8")
    for spine in ax.spines.values():
        spine.set_color("#d0d4da")
        spine.set_linewidth(0.8)
    ax.set_title("(d)  equilibrium")
    rows = _card_rows(eq, summary)
    n_section = sum(1 for kind, *_ in rows if kind == "section")
    n_item = sum(1 for kind, *_ in rows if kind == "item")
    # Section rules take a little less than a data row. Keep a footer line.
    footer = 0.07
    usable = 0.90 - footer
    weight = n_section * 0.72 + n_item * 1.0
    step = usable / weight
    y = 0.94
    for kind, *payload in rows:
        if kind == "section":
            y -= step * 0.15
            ax.plot([0.06, 0.94], [y + 0.012, y + 0.012], color="#d5d8de", lw=0.6, transform=ax.transAxes, clip_on=False)
            ax.text(
                0.06,
                y,
                payload[0].upper(),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#5c6770",
            )
            y -= step * 0.72
        else:
            label, value = payload
            ax.text(0.06, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=10.5, color="#222222")
            ax.text(0.94, y, value, transform=ax.transAxes, ha="right", va="center", fontsize=10.5, color="#111111")
            y -= step
    ax.text(
        0.06,
        0.035,
        r"$\psi_n = 0$ on axis,  $1$ on the LCFS",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#5c6770",
    )


def _poloidal_beta(eq, lcfs) -> float:
    """β_p = 2 <p> ℓ^2 / (μ0 Ip^2), with ℓ the LCFS length and <p> the area average.

    The poloidal-field scale in the denominator is μ0 Ip / ℓ.
    """
    if lcfs is None or eq.profile.Ip == 0.0:
        return float("nan")
    r, z = lcfs
    ell = float(np.sum(np.hypot(np.diff(r), np.diff(z))))
    if ell <= 0.0 or not np.isfinite(ell):
        return float("nan")
    pn = np.clip(eq.psin(), 0.0, 1.0)
    pressure = np.asarray(eq.pressure(pn), dtype=float)
    mask = np.asarray(eq.mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    p_avg = float(np.mean(pressure[mask]))
    return float(2.0 * p_avg * ell**2 / (MU0 * float(eq.profile.Ip) ** 2))


def _summarize(eq) -> dict:
    s = np.linspace(0.0, 1.0, 401)
    pressure_kpa = np.asarray(eq.pressure(s), dtype=float) / 1.0e3
    F = np.asarray(eq.F(s), dtype=float)
    q_s = np.linspace(0.05, 0.95, 19)
    try:
        q = np.asarray(
            q_profile(
                eq.psi,
                eq.grid,
                float(eq.psi_axis),
                float(eq.psi_boundary),
                float(eq.R_axis),
                float(eq.Z_axis),
                eq.F,
                q_s,
            ),
            dtype=float,
        )
        q95 = float(q[-1])
    except (ValueError, RuntimeError):
        q = None
        q95 = float("nan")
    lcfs = _lcfs_polyline(eq)
    try:
        beta = _poloidal_beta(eq, lcfs)
    except (ValueError, RuntimeError):
        beta = float("nan")
    return {
        "s": s,
        "pressure_kpa": pressure_kpa,
        "F": F,
        "q_s": q_s,
        "q": q,
        "q95": q95,
        "beta_p": beta,
        "lcfs": lcfs,
        "shift": float(eq.R_axis - eq.shape.R0),
    }


def _rect(fig_w, fig_h, x, y, w, h):
    return [x / fig_w, y / fig_h, w / fig_w, h / fig_h]


def _compose_equilibrium(fig, eq, title: str | None):
    summary = _summarize(eq)
    view = _view_from_mask(eq.mask, eq.grid.R, eq.grid.Z, pad_frac=0.09)
    data_w = view[1] - view[0]
    data_h = view[3] - view[2]
    map_h = 5.05
    map_w = map_h * data_w / max(data_h, 1e-6)
    cbar_w = 0.15
    gap_cbar = 0.10
    gap_maps = 1.18
    info_gap = 0.78
    info_w = 3.45
    left = 0.92
    right = 0.28
    top = 0.92
    bottom = 0.55
    prof_h = 2.20
    gap_rows = 0.72

    x1 = left
    x2 = x1 + map_w + gap_cbar + cbar_w + gap_maps
    x_info = x2 + map_w + gap_cbar + cbar_w + info_gap
    fig_w = x_info + info_w + right
    fig_h = bottom + prof_h + gap_rows + map_h + top
    fig.set_size_inches(fig_w, fig_h)
    fig.clf()

    y_map = bottom + prof_h + gap_rows
    ax_psi = fig.add_axes(_rect(fig_w, fig_h, x1, y_map, map_w, map_h))
    cax_psi = fig.add_axes(_rect(fig_w, fig_h, x1 + map_w + gap_cbar, y_map, cbar_w, map_h))
    ax_j = fig.add_axes(_rect(fig_w, fig_h, x2, y_map, map_w, map_h))
    cax_j = fig.add_axes(_rect(fig_w, fig_h, x2 + map_w + gap_cbar, y_map, cbar_w, map_h))
    ax_info = fig.add_axes(_rect(fig_w, fig_h, x_info, y_map, info_w, map_h))

    row_right = x_info + info_w
    gap_p = 0.55
    prof_w = (row_right - left - 2.0 * gap_p) / 3.0
    ax_p = fig.add_axes(_rect(fig_w, fig_h, left, bottom, prof_w, prof_h))
    ax_f = fig.add_axes(_rect(fig_w, fig_h, left + prof_w + gap_p, bottom, prof_w, prof_h))
    ax_q = fig.add_axes(_rect(fig_w, fig_h, left + 2.0 * (prof_w + gap_p), bottom, prof_w, prof_h))

    lcfs = summary["lcfs"]
    fill = _draw_flux_field(ax_psi, eq, view, lcfs, label_surfaces=True, title="(a)  normalized flux")
    mesh = _draw_current(ax_j, eq, view, lcfs)
    _draw_profiles(ax_p, ax_f, ax_q, summary)
    _draw_card(ax_info, eq, summary)

    cb_psi = fig.colorbar(fill, cax=cax_psi)
    # Horizontal title, so the label does not sit in the gap next to panel (b).
    cb_psi.ax.set_title(r"$\psi_n$", fontsize=11, pad=6)
    cb_psi.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cb_psi.outline.set_linewidth(0.6)
    cb_psi.ax.tick_params(length=3, labelsize=8)

    cb_j = fig.colorbar(mesh, cax=cax_j)
    cb_j.set_label(r"$J_\phi$ [MA/m$^{2}$]", labelpad=6)
    cb_j.outline.set_linewidth(0.6)
    cb_j.ax.tick_params(length=3, labelsize=8)

    fig.suptitle(title or "Fixed-boundary Grad–Shafranov equilibrium", fontsize=14, y=0.985)
    fig.canvas.draw()
    # Equal aspect can shrink an axes inside its allocated box. Keep the
    # colorbars and the summary card locked to the flux axes.
    psi_pos = ax_psi.get_position()
    for ax, cax in ((ax_psi, cax_psi), (ax_j, cax_j)):
        pos = ax.get_position()
        cpos = cax.get_position()
        cax.set_position([cpos.x0, pos.y0, cpos.width, pos.height])
    info_pos = ax_info.get_position()
    ax_info.set_position([info_pos.x0, psi_pos.y0, info_pos.width, psi_pos.height])
    return summary


def plot_equilibrium(eq, ax_or_fig=None, title=None):
    """Draw one equilibrium.

    ``ax_or_fig`` None builds a new figure. A Figure is cleared and used for the
    full four-part layout (flux, current, profiles, summary). A single Axes
    receives only the poloidal flux panel.
    """
    plt = _pyplot()
    with plt.rc_context(_RC):
        if _is_axes(ax_or_fig):
            view = _view_from_mask(eq.mask, eq.grid.R, eq.grid.Z, pad_frac=0.09)
            _draw_flux_field(
                ax_or_fig,
                eq,
                view,
                _lcfs_polyline(eq),
                label_surfaces=True,
                title=title,
            )
            return ax_or_fig.figure
        from matplotlib.figure import Figure

        if isinstance(ax_or_fig, Figure):
            fig = ax_or_fig
        elif ax_or_fig is None:
            fig = plt.figure()
        else:
            raise TypeError("ax_or_fig must be None, a Figure, or an Axes")
        _compose_equilibrium(fig, eq, title)
        return fig


def equilibrium_figure(eq, out_path, title=None):
    """Save the multi-panel equilibrium figure and return the output path."""
    plt = _pyplot()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_equilibrium(eq, title=title)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def _style_error_colorbar(cb, vmax: float) -> None:
    """Tick labels that stay distinct when the error is a few percent or less."""
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter

    cb.locator = MaxNLocator(nbins=5)
    if vmax < 1e-3:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((-4, -3))
        cb.formatter = fmt
    elif vmax < 0.05:
        cb.formatter = FormatStrFormatter("%.3f")
    else:
        cb.formatter = FormatStrFormatter("%.2f")
    cb.update_ticks()
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(length=3, labelsize=8)


def _fmt_axis_error(metres: float) -> str:
    if not np.isfinite(metres):
        return "n/a"
    if abs(metres) < 0.1:
        return f"{metres * 1e3:.2f} mm"
    return f"{metres:.4f} m"


def _overlay_contours(ax, grid, field, levels, *, color, lw, ls, zorder):
    masked = np.ma.array(np.asarray(field, dtype=float), mask=~np.isfinite(field))
    return ax.contour(
        grid.R,
        grid.Z,
        masked.T,
        levels=levels,
        colors=color,
        linewidths=lw,
        linestyles=ls,
        zorder=zorder,
    )


def compare_figure(eq, psi_pred, label, out_path, note: str | None = None):
    """Solver (solid) versus a prediction (dashed), plus the normalized error.

    ``psi_pred`` is on ``eq.grid``. The error map is
    ``|psi_pred - psi| / psi_axis`` inside the plasma. The figure title carries
    the relative L2 and the magnetic-axis error.
    """
    plt = _pyplot()
    from matplotlib.lines import Line2D

    pred = np.asarray(psi_pred, dtype=float)
    if pred.shape != eq.psi.shape:
        raise ValueError(f"psi_pred shape {pred.shape} != equilibrium psi shape {eq.psi.shape}")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rel = float(relative_l2(pred, eq.psi, eq.mask))
    try:
        r_s, z_s, _ = find_magnetic_axis(eq.psi, eq.grid, eq.mask)
        r_p, z_p, _ = find_magnetic_axis(pred, eq.grid, eq.mask)
        axis_err = float(np.hypot(r_p - r_s, z_p - z_s))
    except ValueError:
        r_s = z_s = r_p = z_p = float("nan")
        axis_err = float("nan")

    lcfs = _lcfs_polyline(eq)
    view = _view_from_mask(eq.mask, eq.grid.R, eq.grid.Z, pad_frac=0.10)
    mask = np.asarray(eq.mask, dtype=bool)
    # Absolute flux levels of the solver, expressed as ψ_n so both contours
    # are the same ψ surfaces.
    solver_n = np.ma.masked_where(~mask, eq.psin())
    pred_n = np.ma.masked_where(~mask, 1.0 - pred / float(eq.psi_axis))
    err = np.abs(pred - np.asarray(eq.psi, dtype=float)) / abs(float(eq.psi_axis))
    err_in = err[mask]
    peak = float(np.max(err_in)) if err_in.size else 1e-6
    vmax = _ceil_sig(max(peak, 1e-8), 2)

    data_w = view[1] - view[0]
    data_h = view[3] - view[2]
    map_h = 5.70
    map_w = map_h * data_w / max(data_h, 1e-6)
    cbar_w = 0.16
    gap_cbar = 0.10
    gap_maps = 1.20
    left = 0.82
    right = 1.05
    top = 1.05
    bottom = 0.62
    x1 = left
    x2 = x1 + map_w + gap_cbar + cbar_w + gap_maps
    fig_w = x2 + map_w + gap_cbar + cbar_w + right
    fig_h = bottom + map_h + top

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(fig_w, fig_h))
        y0 = bottom
        ax_c = fig.add_axes(_rect(fig_w, fig_h, x1, y0, map_w, map_h))
        cax_c = fig.add_axes(_rect(fig_w, fig_h, x1 + map_w + gap_cbar, y0, cbar_w, map_h))
        ax_e = fig.add_axes(_rect(fig_w, fig_h, x2, y0, map_w, map_h))
        cax_e = fig.add_axes(_rect(fig_w, fig_h, x2 + map_w + gap_cbar, y0, cbar_w, map_h))

        pn = np.asarray(eq.psin(), dtype=float)
        fill = ax_c.contourf(
            eq.grid.R,
            eq.grid.Z,
            pn.T,
            levels=np.linspace(0.0, 1.0, 33),
            cmap=_PSI_CMAP,
            vmin=0.0,
            vmax=1.0,
            zorder=1,
        )
        _clip_to_lcfs(fill, ax_c, lcfs)
        ax_c.contour(
            eq.grid.R,
            eq.grid.Z,
            solver_n.T,
            levels=_FLUX_LEVELS,
            colors="white",
            linewidths=2.15,
            zorder=3,
        )
        ax_c.contour(
            eq.grid.R,
            eq.grid.Z,
            solver_n.T,
            levels=_FLUX_LEVELS,
            colors=_SOLVER_COLOR,
            linewidths=1.05,
            zorder=4,
        )
        ax_c.contour(
            eq.grid.R,
            eq.grid.Z,
            pred_n.T,
            levels=_FLUX_LEVELS,
            colors=_MODEL_COLOR,
            linewidths=1.35,
            linestyles="dashed",
            zorder=5,
        )
        _draw_lcfs(ax_c, lcfs, lw=2.05, zorder=6)
        _draw_axis_marker(ax_c, float(eq.R_axis), float(eq.Z_axis))
        if np.isfinite(axis_err) and axis_err > 0.015 * max(data_w, data_h):
            ax_c.plot(r_p, z_p, marker="x", ms=8, color=_MODEL_COLOR, mew=1.3, linestyle="none", zorder=8)
        _style_map_axes(ax_c, view)
        ax_c.set_title("(a)  flux surfaces")
        ax_c.legend(
            handles=[
                Line2D([0], [0], color=_SOLVER_COLOR, lw=1.4, label="solver"),
                Line2D([0], [0], color=_MODEL_COLOR, lw=1.4, ls="--", label=str(label)),
            ],
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            edgecolor="#d0d0d0",
            fontsize=8,
        )

        err_show = np.ma.masked_where(~mask, err)
        mesh = ax_e.contourf(
            eq.grid.R,
            eq.grid.Z,
            err_show.T,
            levels=np.linspace(0.0, vmax, 21),
            cmap=_ERR_CMAP,
            vmin=0.0,
            vmax=vmax,
            zorder=1,
        )
        _clip_to_lcfs(mesh, ax_e, lcfs)
        ax_e.contour(
            eq.grid.R,
            eq.grid.Z,
            solver_n.T,
            levels=[0.25, 0.5, 0.75],
            colors="white",
            linewidths=0.7,
            alpha=0.9,
            zorder=3,
        )
        _draw_lcfs(ax_e, lcfs, color="white", lw=2.3, zorder=4)
        _draw_lcfs(ax_e, lcfs, color="#111111", lw=0.85, zorder=5)
        _draw_axis_marker(ax_e, float(eq.R_axis), float(eq.Z_axis))
        _style_map_axes(ax_e, view)
        ax_e.set_title("(b)  absolute error inside the plasma")

        cb_c = fig.colorbar(fill, cax=cax_c)
        cb_c.set_label(r"$\psi_n$", labelpad=4)
        cb_c.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cb_c.outline.set_linewidth(0.6)
        cb_c.ax.tick_params(length=3, labelsize=8)
        cb_e = fig.colorbar(mesh, cax=cax_e)
        cb_e.set_label(r"$|\Delta\psi|\,/\,\psi_{\mathrm{axis}}$", labelpad=8)
        _style_error_colorbar(cb_e, vmax)

        fig.suptitle(
            f"{label} versus solver\nrelative L2 = {rel:.3e}      axis error = {_fmt_axis_error(axis_err)}",
            fontsize=13,
            y=0.985,
        )
        if note:
            fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=8.5, color="#444444")
        fig.canvas.draw()
        for ax, cax in ((ax_c, cax_c), (ax_e, cax_e)):
            pos = ax.get_position()
            cpos = cax.get_position()
            cax.set_position([cpos.x0, pos.y0, cpos.width, pos.height])
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
    return path, rel, axis_err


def _default_sweep_values(param: str) -> np.ndarray:
    if param == "kappa":
        return np.linspace(1.2, 2.0, 24)
    if param == "delta":
        return np.linspace(0.0, 0.5, 24)
    raise ValueError("param must be 'kappa' or 'delta'")


def _shape_at(shape: ShapeParams, param: str, value: float) -> ShapeParams:
    from dataclasses import replace

    if param not in ("kappa", "delta"):
        raise ValueError("param must be 'kappa' or 'delta'")
    return replace(shape, **{param: float(value)})


def covering_grid(shape: ShapeParams, n: int, pad: float = 0.20) -> Grid:
    """Uniform ``n×n`` grid that contains the plasma, with the LCFS off the box edge."""
    level = shape.level_set()
    R = np.linspace(shape.R0 - 2.2 * shape.a, shape.R0 + 1.8 * shape.a, 280)
    Z = np.linspace(-1.45 * shape.kappa * shape.a, 1.45 * shape.kappa * shape.a, 280)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    inside = np.asarray(level(RR, ZZ), dtype=float) < 0.0
    if not np.any(inside):
        raise RuntimeError("shape level set does not close inside the sampling box")
    r_in = R[np.any(inside, axis=1)]
    z_in = Z[np.any(inside, axis=0)]
    span_r = float(r_in[-1] - r_in[0])
    span_z = float(z_in[-1] - z_in[0])
    dr = pad * span_r + 0.04
    dz = pad * span_z + 0.04
    return Grid.uniform(
        float(r_in[0] - dr),
        float(r_in[-1] + dr),
        float(z_in[0] - dz),
        float(z_in[-1] + dz),
        int(n),
        int(n),
    )


def _sweep_grid(shape: ShapeParams, param: str, values: np.ndarray, n: int) -> Grid:
    """One grid large enough for every plasma in the scan."""
    boxes = []
    for value in values:
        sample = _shape_at(shape, param, float(value))
        # A coarse probe is enough to size the box; the solve uses ``n``.
        probe = covering_grid(sample, n=max(int(n), 33), pad=0.16)
        boxes.append((float(probe.R[0]), float(probe.R[-1]), float(probe.Z[0]), float(probe.Z[-1])))
    return Grid.uniform(
        min(b[0] for b in boxes),
        max(b[1] for b in boxes),
        min(b[2] for b in boxes),
        max(b[3] for b in boxes),
        int(n),
        int(n),
    )


def _param_vector(shape: ShapeParams, profile: ProfileParams) -> np.ndarray:
    return np.array(
        [
            shape.R0,
            shape.a,
            shape.kappa,
            shape.delta,
            profile.Ip,
            profile.beta0,
            profile.alpha,
            profile.gamma,
            profile.B0,
        ],
        dtype=np.float64,
    )


def _case_subtitle(shape: ShapeParams, profile: ProfileParams, param: str) -> str:
    bits = [rf"$R_0 = {shape.R0:.2f}\,\mathrm{{m}}$", rf"$a = {shape.a:.2f}\,\mathrm{{m}}$"]
    if param != "kappa":
        bits.append(rf"$\kappa = {shape.kappa:.2f}$")
    if param != "delta":
        bits.append(rf"$\delta = {shape.delta:.2f}$")
    bits.append(rf"$I_p = {profile.Ip / 1e6:.2f}\,\mathrm{{MA}}$")
    return "    ".join(bits)


def _param_math(param: str, value: float) -> str:
    symbol = {"kappa": r"\kappa", "delta": r"\delta"}[param]
    return rf"${symbol} = {value:.3f}$"


def shape_sweep_animation(
    out_path,
    param: str = "kappa",
    values=None,
    *,
    shape: ShapeParams | None = None,
    profile: ProfileParams | None = None,
    grid: Grid | None = None,
    surrogate=None,
    fps: int = 10,
    dpi: int = 120,
    grid_n: int = 97,
):
    """GIF of flux surfaces as elongation or triangularity is scanned.

    Solver surfaces are solid. When ``surrogate`` is a loaded model, its flux
    surfaces are dashed on the same axes (predicted on the model's own grid).
    Axis limits are fixed for the whole clip. ``values`` defaults to 24 frames,
    κ from 1.2 to 2.0 or δ from 0 to 0.5.
    """
    plt = _pyplot()
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    if param not in ("kappa", "delta"):
        raise ValueError("param must be 'kappa' or 'delta'")
    if values is None:
        values = _default_sweep_values(param)
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size < 1:
        raise ValueError("values must contain at least one frame")
    if not PillowWriter.isAvailable():
        raise RuntimeError("Pillow is required for GIF output (matplotlib PillowWriter)")

    shape = shape or ShapeParams(**DEFAULT_SHAPE)
    profile = profile or ProfileParams(**DEFAULT_PROFILE)
    if grid is None:
        grid = _sweep_grid(shape, param, values, int(grid_n))

    predict = None
    if surrogate is not None:
        import torch

        from models.surrogate import predict as _predict

        torch.set_num_threads(2)
        predict = _predict

    frames = []
    for value in values:
        shape_i = _shape_at(shape, param, float(value))
        eq = solve_fixed_boundary(shape_i, profile, grid)
        pred = None
        pred_grid = None
        if predict is not None:
            pred = np.asarray(predict(surrogate, _param_vector(shape_i, profile)), dtype=float)
            if getattr(surrogate, "grid_R", None) is not None:
                pred_grid = Grid(
                    np.asarray(surrogate.grid_R, dtype=float),
                    np.asarray(surrogate.grid_Z, dtype=float),
                )
            else:
                pred_grid = Grid(np.asarray(surrogate.R.detach().cpu().numpy(), dtype=float), np.asarray(surrogate.Z.detach().cpu().numpy(), dtype=float))
            if pred.shape != (pred_grid.nr, pred_grid.nz):
                raise RuntimeError(
                    f"surrogate output {pred.shape} does not match its grid {(pred_grid.nr, pred_grid.nz)}"
                )
        frames.append((float(value), eq, pred, pred_grid))

    views = [_view_from_mask(eq.mask, eq.grid.R, eq.grid.Z, pad_frac=0.06) for _, eq, _, _ in frames]
    view = _union_view(views, pad_frac=0.10)
    data_w = view[1] - view[0]
    data_h = view[3] - view[2]
    axes_h = 6.15
    axes_w = axes_h * data_w / max(data_h, 1e-6)
    left_in, bottom_in, right_in, top_in = 0.78, 0.62, 1.15, 0.95
    fig_w = left_in + axes_w + right_in
    fig_h = bottom_in + axes_h + top_in

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subtitle = _case_subtitle(shape, profile, param)

    with plt.rc_context(_RC):
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes(_rect(fig_w, fig_h, left_in, bottom_in, axes_w, axes_h))
        cax = fig.add_axes(
            _rect(fig_w, fig_h, left_in + axes_w + 0.12, bottom_in + 0.15, 0.16, axes_h - 0.30)
        )
        sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=_PSI_CMAP)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(r"$\psi_n$")
        cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cb.outline.set_linewidth(0.6)
        cb.ax.tick_params(length=3, labelsize=8)
        title_artist = fig.text(0.46, 0.975, "", ha="center", va="top", fontsize=15)
        fig.text(0.46, 0.928, subtitle, ha="center", va="top", fontsize=9, color="#333333")
        legend_handles = [
            Line2D([0], [0], color=_SOLVER_COLOR, lw=1.5, label="solver"),
        ]
        if surrogate is not None:
            legend_handles.append(Line2D([0], [0], color=_MODEL_COLOR, lw=1.5, ls="--", label="surrogate"))

        def update(index):
            value, eq, pred, pred_grid = frames[int(index)]
            ax.clear()
            lcfs = _lcfs_polyline(eq)
            _draw_flux_field(ax, eq, view, lcfs, label_surfaces=False)
            if pred is not None and pred_grid is not None:
                levels = float(eq.psi_axis) * (1.0 - _FLUX_LEVELS)
                # Same absolute ψ as the solid solver surfaces.
                phi = np.asarray(eq.shape.level_set()(pred_grid.RR, pred_grid.ZZ), dtype=float)
                shown = np.ma.masked_where(phi >= 0.0, pred)
                ax.contour(
                    pred_grid.R,
                    pred_grid.Z,
                    shown.T,
                    levels=np.sort(levels),
                    colors=_MODEL_COLOR,
                    linewidths=1.25,
                    linestyles="dashed",
                    zorder=5,
                )
            _style_map_axes(ax, view)
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                frameon=True,
                framealpha=0.94,
                edgecolor="#d0d0d0",
                fontsize=8,
            )
            title_artist.set_text(_param_math(param, value))
            return []

        update(0)
        fig.canvas.draw()
        pos = ax.get_position()
        cpos = cax.get_position()
        cax.set_position([cpos.x0, pos.y0, cpos.width, pos.height * 0.92])
        anim = FuncAnimation(
            fig,
            update,
            frames=len(frames),
            blit=False,
            cache_frame_data=False,
        )
        anim.save(
            path,
            writer=PillowWriter(fps=int(fps)),
            dpi=int(dpi),
            savefig_kwargs={"facecolor": "white"},
        )
        plt.close(fig)
    return path


def _threads():
    import torch

    torch.set_num_threads(2)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Visualize Grad–Shafranov equilibria and model comparisons")
    parser.add_argument("--out", default="outputs/visualize", help="directory for png and gif output")
    parser.add_argument("--R0", type=float, default=DEFAULT_SHAPE["R0"])
    parser.add_argument("--a", type=float, default=DEFAULT_SHAPE["a"])
    parser.add_argument("--kappa", type=float, default=DEFAULT_SHAPE["kappa"])
    parser.add_argument("--delta", type=float, default=DEFAULT_SHAPE["delta"])
    parser.add_argument("--Ip", type=float, default=DEFAULT_PROFILE["Ip"])
    parser.add_argument("--beta0", type=float, default=DEFAULT_PROFILE["beta0"])
    parser.add_argument("--alpha", type=float, default=DEFAULT_PROFILE["alpha"])
    parser.add_argument("--gamma", type=float, default=DEFAULT_PROFILE["gamma"])
    parser.add_argument("--B0", type=float, default=DEFAULT_PROFILE["B0"])
    parser.add_argument("--grid-n", type=int, default=129, help="points per side of the equilibrium mesh")
    parser.add_argument("--fps", type=int, default=10, help="shape-sweep frame rate (8–12 is comfortable)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.grid_n < 17:
        raise SystemExit("--grid-n must be at least 17")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shape = ShapeParams(args.R0, args.a, args.kappa, args.delta)
    profile = ProfileParams(args.Ip, args.beta0, args.alpha, args.gamma, args.B0)

    grid = covering_grid(shape, args.grid_n)
    print(
        f"solving equilibrium on {grid.nr}×{grid.nz} "
        f"R[{grid.R[0]:.3f}, {grid.R[-1]:.3f}] Z[{grid.Z[0]:.3f}, {grid.Z[-1]:.3f}]",
        flush=True,
    )
    eq = solve_fixed_boundary(shape, profile, grid)
    equilibrium_figure(eq, out / "equilibrium.png")
    summary = _summarize(eq)
    print(
        f"wrote {out / 'equilibrium.png'}  "
        f"psi_axis={eq.psi_axis:.4f}  shift={summary['shift']:+.4f} m  "
        f"q95={summary['q95']:.3f}  beta_p={summary['beta_p']:.3f}  "
        f"iterations={eq.n_iter}",
        flush=True,
    )

    surrogate = None
    surr_path = _find_file("outputs/surrogate/mlpcnn_pw0p1/checkpoint.pt")
    if surr_path is None:
        print("skip surrogate_vs_solver.png: outputs/surrogate/mlpcnn_pw0p1/checkpoint.pt not found", flush=True)
    else:
        _threads()
        from data.generate import fixed_grid
        from models.surrogate import load_surrogate, predict

        surrogate = load_surrogate(surr_path)
        _threads()
        g65 = fixed_grid(65, 65)
        # The published checkpoint is this mesh; refuse a silent mismatch.
        if surrogate.grid_R is None or not (
            np.allclose(surrogate.grid_R, g65.R) and np.allclose(surrogate.grid_Z, g65.Z)
        ):
            print(
                "warning: checkpoint grid differs from R[0.80, 2.76] × Z[-1.90, 1.90], 65×65; using the checkpoint grid",
                flush=True,
            )
            g65 = Grid(np.asarray(surrogate.grid_R, dtype=float), np.asarray(surrogate.grid_Z, dtype=float))
        print(f"solving surrogate comparison on {g65.nr}×{g65.nz}", flush=True)
        eq65 = solve_fixed_boundary(shape, profile, g65)
        pred = predict(surrogate, _param_vector(shape, profile))
        _, rel, axis_err = compare_figure(
            eq65,
            pred,
            "mlpcnn_pw0p1",
            out / "surrogate_vs_solver.png",
            note="Surrogate training grid  R [0.80, 2.76] m,  Z [−1.90, 1.90] m,  65×65",
        )
        print(
            f"wrote {out / 'surrogate_vs_solver.png'}  rel L2={rel:.3e}  axis error={_fmt_axis_error(axis_err)}",
            flush=True,
        )

    pinn_path = _find_file("outputs/pinn/nonlinear.pt")
    if pinn_path is None:
        print("skip pinn_vs_solver.png: outputs/pinn/nonlinear.pt not found", flush=True)
    else:
        _threads()
        from models.pinn import (
            NONLINEAR_EVAL_BOX,
            NONLINEAR_PROFILE,
            NONLINEAR_SHAPE,
            load_pinn,
            predict_grid,
        )

        pinn_shape = ShapeParams(**NONLINEAR_SHAPE)
        pinn_profile = ProfileParams(**NONLINEAR_PROFILE)
        pinn_grid = Grid.uniform(*NONLINEAR_EVAL_BOX, int(args.grid_n), int(args.grid_n))
        print(
            f"solving PINN reference on {pinn_grid.nr}×{pinn_grid.nz} "
            f"box {NONLINEAR_EVAL_BOX}",
            flush=True,
        )
        eq_pinn = solve_fixed_boundary(pinn_shape, pinn_profile, pinn_grid)
        model = load_pinn(pinn_path)
        _threads()
        pred = predict_grid(model, pinn_grid)
        _, rel, axis_err = compare_figure(
            eq_pinn,
            pred,
            "PINN nonlinear",
            out / "pinn_vs_solver.png",
            note=(
                f"Nonlinear PINN case  R0={pinn_shape.R0}, a={pinn_shape.a}, "
                f"κ={pinn_shape.kappa}, δ={pinn_shape.delta}, Ip={pinn_profile.Ip:.0f} A    "
                f"{pinn_grid.nr}×{pinn_grid.nz}"
            ),
        )
        print(
            f"wrote {out / 'pinn_vs_solver.png'}  rel L2={rel:.3e}  axis error={_fmt_axis_error(axis_err)}",
            flush=True,
        )

    print("kappa sweep", flush=True)
    shape_sweep_animation(
        out / "kappa_sweep.gif",
        param="kappa",
        shape=shape,
        profile=profile,
        surrogate=surrogate,
        fps=int(args.fps),
    )
    print(f"wrote {out / 'kappa_sweep.gif'}", flush=True)
    print("delta sweep", flush=True)
    shape_sweep_animation(
        out / "delta_sweep.gif",
        param="delta",
        shape=shape,
        profile=profile,
        surrogate=surrogate,
        fps=int(args.fps),
    )
    print(f"wrote {out / 'delta_sweep.gif'}", flush=True)


if __name__ == "__main__":
    main()
