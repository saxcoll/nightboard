"""Analytic Solov'ev equilibria (Cerfon and Freidberg, Phys. Plasmas 17, 032502, 2010).

Normalized coordinates x = R / R0, y = Z / R0. The normalized flux psi_bar satisfies

    x d/dx( (1/x) dpsi/dx ) + d2psi/dy2 = (1 - A) x^2 + A

and is written as psi_bar = psi_p + sum_{i=1..7} c_i psi_i (up-down symmetric basis),
where the particular solution is psi_p = x^4/8 + A (x^2 ln x / 2 - x^4/8). The seven
coefficients are fixed by requiring psi_bar = 0 at the outer/inner equatorial points and
the top point of the Miller-like boundary

    x = 1 + eps cos(tau + alpha sin tau),  y = eps kappa sin tau,  alpha = arcsin(delta)

plus the zero-slope condition at the top and the three curvature conditions.

Sign: psi_bar < 0 inside the plasma and psi_bar = 0 on the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SolovevEquilibrium:
    epsilon: float
    kappa: float
    delta: float
    A: float
    R0: float
    coeffs: np.ndarray

    def psi(self, R, Z) -> np.ndarray:
        """psi_bar(R/R0, Z/R0). Accepts broadcastable arrays in metres."""
        raise NotImplementedError

    def grad_psi(self, R, Z) -> tuple[np.ndarray, np.ndarray]:
        """(dpsi/dR, dpsi/dZ) of psi(R, Z) in physical coordinates."""
        raise NotImplementedError

    def gs_rhs(self, R, Z) -> np.ndarray:
        """Delta* psi in physical (R, Z) coordinates: ((1 - A) x^2 + A) / R0^2."""
        raise NotImplementedError

    def boundary(self, n: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """(R_b, Z_b) points of the Miller-like target boundary in metres."""
        raise NotImplementedError

    def level_set(self, R, Z) -> np.ndarray:
        """Negative inside the plasma, zero on the psi_bar = 0 surface, positive outside.

        Equal to psi_bar inside a bounding box around the plasma and forced positive outside
        it, so that spurious psi_bar < 0 regions far from the plasma are excluded.
        """
        raise NotImplementedError


def solovev(epsilon: float, kappa: float, delta: float, A: float, R0: float = 1.0) -> SolovevEquilibrium:
    """Build the Cerfon-Freidberg up-down symmetric Solov'ev equilibrium."""
    raise NotImplementedError
