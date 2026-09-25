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

MU0 = 4e-7 * np.pi

LevelSet = Callable[[np.ndarray, np.ndarray], np.ndarray]


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
        """Level-set function (negative inside) for the shaped fixed boundary."""
        raise NotImplementedError


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
        raise NotImplementedError

    def pprime(self, psin) -> np.ndarray:
        raise NotImplementedError

    def ffprime(self, psin) -> np.ndarray:
        raise NotImplementedError

    def F(self, psin) -> np.ndarray:
        """Toroidal field function F = R B_phi, with F(1) = R0 B0."""
        raise NotImplementedError

    def pressure(self, psin) -> np.ndarray:
        raise NotImplementedError


def delta_star_fd(psi: np.ndarray, grid: Grid) -> np.ndarray:
    """Second-order finite-difference Delta* psi. Edge rows/columns are NaN."""
    raise NotImplementedError


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
    raise NotImplementedError


def solve_fixed_boundary(
    shape: ShapeParams,
    profile: ProfileParams,
    grid: Grid,
    tol: float = 1e-8,
    max_iter: int = 200,
    relax: float = 0.5,
) -> Equilibrium:
    """Picard iteration for the nonlinear fixed-boundary equilibrium."""
    raise NotImplementedError
