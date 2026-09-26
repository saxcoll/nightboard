"""Explanatory diagrams for docs/REPORT.md.

Run from the fusion-gs directory:

    python docs/figures/make_diagrams.py

Both figures are drawn with matplotlib from the project's own solver and
diagnostic geometry. Nothing here is a measured experimental plot.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from eval.visualize import _lcfs_polyline
from gs.solver import Grid, ProfileParams, ShapeParams, solve_fixed_boundary
from models.diagnostics import SensorSet

OUT = Path(__file__).resolve().parent

# Showcase equilibrium used throughout the report.
SHAPE = ShapeParams(R0=1.7, a=0.6, kappa=1.7, delta=0.3)
PROFILE = ProfileParams(Ip=1.0e6, beta0=0.5, alpha=1.0, gamma=2.0, B0=2.0)


def _flux_and_field(eq):
    """Poloidal field from B_R = -(1/R) dpsi/dZ, B_Z = (1/R) dpsi/dR."""
    psi = np.asarray(eq.psi, dtype=float)
    dR = float(eq.grid.dR)
    dZ = float(eq.grid.dZ)
    dpsi_dR = np.gradient(psi, dR, axis=0)
    dpsi_dZ = np.gradient(psi, dZ, axis=1)
    R = eq.grid.RR
    br = -(1.0 / R) * dpsi_dZ
    bz = (1.0 / R) * dpsi_dR
    return br, bz


def cross_section() -> None:
    """Poloidal cross-section: coordinates, flux surfaces, axis, LCFS, sensors."""
    grid = Grid.uniform(0.80, 2.76, -1.90, 1.90, 129, 129)
    eq = solve_fixed_boundary(SHAPE, PROFILE, grid)
    sensors = SensorSet.for_grid(Grid.uniform(0.80, 2.76, -1.90, 1.90, 65, 65))
    lcfs = _lcfs_polyline(eq)
    br, bz = _flux_and_field(eq)
    pn = np.asarray(eq.psin(), dtype=float)
    pn_show = np.where(eq.mask, pn, np.nan)

    fig, ax = plt.subplots(figsize=(7.4, 7.6), constrained_layout=True)
    ax.set_facecolor("#f7f5f2")
    fill = ax.contourf(
        eq.grid.R,
        eq.grid.Z,
        pn_show.T,
        levels=np.linspace(0.0, 1.0, 21),
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        zorder=1,
    )
    ax.contour(
        eq.grid.R,
        eq.grid.Z,
        pn_show.T,
        levels=[0.2, 0.4, 0.6, 0.8],
        colors="white",
        linewidths=0.9,
        zorder=2,
    )
    if lcfs is not None:
        ax.plot(lcfs[0], lcfs[1], color="#111111", lw=2.0, zorder=4, label="LCFS")
    # Geometric centre versus magnetic axis: the outward Shafranov shift.
    ax.scatter([SHAPE.R0], [0.0], s=36, facecolors="none", edgecolors="#1d4e89", lw=1.4, zorder=5)
    ax.scatter([eq.R_axis], [eq.Z_axis], s=42, c="#9b2226", marker="+", lw=1.6, zorder=6)
    ax.annotate(
        "magnetic axis",
        xy=(eq.R_axis, eq.Z_axis),
        xytext=(eq.R_axis + 0.22, 0.28),
        fontsize=9,
        color="#9b2226",
        arrowprops=dict(arrowstyle="-", color="#9b2226", lw=0.7),
        zorder=7,
    )
    ax.annotate(
        r"geometric centre $(R_0, 0)$",
        xy=(SHAPE.R0, 0.0),
        xytext=(SHAPE.R0 - 0.85, -0.38),
        fontsize=9,
        color="#1d4e89",
        arrowprops=dict(arrowstyle="-", color="#1d4e89", lw=0.7),
        zorder=7,
    )

    # A few poloidal-field arrows. Current into the page gives a clockwise B_p.
    pts = [(2.15, 0.0), (1.85, 0.72), (1.45, 0.0), (1.85, -0.72)]
    for r0, z0 in pts:
        i = int(np.argmin(np.abs(eq.grid.R - r0)))
        j = int(np.argmin(np.abs(eq.grid.Z - z0)))
        if not eq.mask[i, j]:
            continue
        vx, vy = float(br[i, j]), float(bz[i, j])
        nrm = np.hypot(vx, vy)
        if nrm == 0.0:
            continue
        scale = 0.22
        ax.annotate(
            "",
            xy=(eq.grid.R[i] + scale * vx / nrm, eq.grid.Z[j] + scale * vy / nrm),
            xytext=(eq.grid.R[i], eq.grid.Z[j]),
            arrowprops=dict(arrowstyle="-|>", color="#0b6e4f", lw=1.3, mutation_scale=10),
            zorder=5,
        )
    ax.text(2.28, 0.08, r"$B_p$", color="#0b6e4f", fontsize=11)

    # Vessel and the synthetic diagnostics used by the inverse model.
    vr = np.append(sensors.vessel_R, sensors.vessel_R[0])
    vz = np.append(sensors.vessel_Z, sensors.vessel_Z[0])
    ax.plot(vr, vz, color="#3d5a80", lw=1.15, ls="--", zorder=3)
    ax.scatter(
        sensors.probe_R,
        sensors.probe_Z,
        s=14,
        c="white",
        edgecolors="#1b3a4b",
        linewidths=0.45,
        zorder=4,
    )
    ax.scatter(
        sensors.flux_R,
        sensors.flux_Z,
        s=18,
        marker="s",
        c="#90be6d",
        edgecolors="#1b3a4b",
        linewidths=0.4,
        zorder=4,
    )

    # Coordinate directions. In this view (R to the right, Z up) phi points into the page.
    ax.annotate(
        "",
        xy=(2.55, -1.55),
        xytext=(1.85, -1.55),
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.3),
    )
    ax.text(2.58, -1.55, r"$R$", va="center", fontsize=13)
    ax.annotate(
        "",
        xy=(1.85, -0.95),
        xytext=(1.85, -1.55),
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.3),
    )
    ax.text(1.90, -0.95, r"$Z$", fontsize=13)
    # Circle-with-a-cross: into the page. A centre dot would mean the opposite direction.
    ax.text(1.55, -1.45, r"$\otimes$", ha="center", va="center", fontsize=16, color="#111111", zorder=6)
    ax.text(1.68, -1.45, r"$\phi$ and $I_p$ into the page", va="center", fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlim(0.70, 2.90)
    ax.set_ylim(-2.05, 2.05)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Poloidal cross-section of the showcase equilibrium")
    cbar = fig.colorbar(fill, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"$\psi_n$  (0 on axis, 1 on the LCFS)")
    handles = [
        Line2D([0], [0], color="#111111", lw=2.0, label="last closed flux surface"),
        Line2D([0], [0], color="#3d5a80", lw=1.15, ls="--", label="vessel contour"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="#1b3a4b", markersize=6, label="pickup coils (40)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#90be6d",
               markeredgecolor="#1b3a4b", markersize=6, label="flux loops (20)"),
        Line2D([0], [0], color="#0b6e4f", lw=1.3, label="poloidal field"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=8, framealpha=0.92)
    fig.savefig(OUT / "tokamak_cross_section.png", dpi=130)
    plt.close(fig)


def _miller(R0, a, kappa, delta, n=400):
    tau = np.linspace(0.0, 2.0 * np.pi, n)
    alpha = np.arcsin(delta)
    R = R0 + a * np.cos(tau + alpha * np.sin(tau))
    Z = kappa * a * np.sin(tau)
    return R, Z


def profiles_and_shapes() -> None:
    """What beta0, alpha, gamma, kappa and delta actually change."""
    grid = Grid.uniform(0.90, 2.55, -1.25, 1.25, 97, 97)
    s = np.linspace(0.0, 1.0, 401)
    betas = (0.2, 0.5, 0.8)
    colors = {0.2: "#1d4e89", 0.5: "#9a3412", 0.8: "#0f6b4c"}
    solved = {}
    for b in betas:
        prof = ProfileParams(Ip=1.0e6, beta0=b, alpha=1.0, gamma=2.0, B0=2.0)
        solved[b] = solve_fixed_boundary(SHAPE, prof, grid)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    for b, eq in solved.items():
        p = np.asarray(eq.pressure(s), dtype=float) / 1e3
        ax.plot(s, p, color=colors[b], lw=1.8, label=rf"$\beta_0 = {b:.1f}$")
    ax.set_xlabel(r"$\psi_n$")
    ax.set_ylabel("pressure [kPa]")
    ax.set_title(r"(a)  pressure rises with $\beta_0$")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, ls=":", color="#cccccc")

    ax = axes[0, 1]
    for b, eq in solved.items():
        F = np.asarray(eq.F(s), dtype=float)
        ax.plot(s, F, color=colors[b], lw=1.8, label=rf"$\beta_0 = {b:.1f}$")
    ax.axhline(SHAPE.R0 * 2.0, color="#888888", lw=0.7, ls="--")
    ax.text(0.55, 3.41, r"$R_0 B_0 = 3.4$ T m", color="#666666", fontsize=8)
    ax.set_xlabel(r"$\psi_n$")
    ax.set_ylabel(r"$F = R B_\phi$  [T m]")
    ax.set_title(r"(b)  $F$ is larger on axis ($FF' > 0$)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, ls=":", color="#cccccc")

    ax = axes[1, 0]
    curves = [(1.0, 2.0, "showcase"), (0.8, 1.0, r"small $\alpha$, small $\gamma$"), (2.5, 3.0, r"large $\alpha$, large $\gamma$")]
    for alpha, gamma, label in curves:
        g = (1.0 - s**alpha) ** gamma
        ax.plot(s, g, lw=1.8, label=rf"$\alpha={alpha:.1f}$, $\gamma={gamma:.1f}$")
    ax.set_xlabel(r"$\psi_n$")
    ax.set_ylabel(r"$g(\psi_n) = (1 - \psi_n^{\alpha})^{\gamma}$")
    ax.set_title("(c)  current-profile shape")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, ls=":", color="#cccccc")

    ax = axes[1, 1]
    styles = [
        (1.2, 0.3, "#1d4e89", "-"),
        (1.7, 0.3, "#9a3412", "-"),
        (2.0, 0.3, "#0f6b4f", "-"),
        (1.7, 0.0, "#555555", "--"),
        (1.7, 0.5, "#555555", ":"),
    ]
    for kappa, delta, color, ls in styles:
        # Miller target used to place the Cerfon–Freidberg boundary conditions.
        Rb, Zb = _miller(1.7, 0.6, kappa, delta)
        ax.plot(Rb, Zb, color=color, lw=1.4, ls=ls, label=rf"$\kappa={kappa:.1f}$, $\delta={delta:.1f}$")
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("(d)  Miller target boundary")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.grid(True, ls=":", color="#cccccc")

    fig.savefig(OUT / "profiles_and_shapes.png", dpi=130)
    plt.close(fig)

    # Numbers quoted in the report.
    for b, eq in solved.items():
        F = np.asarray(eq.F(np.array([0.0, 1.0])), dtype=float)
        p = float(eq.pressure(0.0))
        print(f"beta0={b} psi_axis={eq.psi_axis:.6f} F_axis={F[0]:.4f} F_edge={F[1]:.4f} p_axis_kPa={p/1e3:.2f} shift={eq.R_axis-eq.shape.R0:.4f}")


if __name__ == "__main__":
    cross_section()
    profiles_and_shapes()
    print("wrote", OUT / "tokamak_cross_section.png")
    print("wrote", OUT / "profiles_and_shapes.png")
