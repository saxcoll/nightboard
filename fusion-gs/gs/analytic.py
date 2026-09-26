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
import sympy as sp

_X, _Y = sp.symbols("x y", positive=True)
_A = sp.symbols("A")
_C = sp.symbols("c1:8")
_LN = sp.log(_X)

_PSI_P = _X**4 / 8 + _A * (_X**2 * _LN / 2 - _X**4 / 8)
_BASIS = (
    sp.Integer(1),
    _X**2,
    _Y**2 - _X**2 * _LN,
    _X**4 - 4 * _X**2 * _Y**2,
    2 * _Y**4 - 9 * _Y**2 * _X**2 + 3 * _X**4 * _LN - 12 * _X**2 * _Y**2 * _LN,
    _X**6 - 12 * _X**4 * _Y**2 + 8 * _X**2 * _Y**4,
    8 * _Y**6
    - 140 * _Y**4 * _X**2
    + 75 * _Y**2 * _X**4
    - 15 * _X**6 * _LN
    + 180 * _X**4 * _Y**2 * _LN
    - 120 * _X**2 * _Y**4 * _LN,
)
_PSI_BAR = _PSI_P + sum(ci * pi for ci, pi in zip(_C, _BASIS))


def _lambdas(expr: sp.Expr, syms: tuple):
    return (
        sp.lambdify(syms, expr, modules="numpy"),
        sp.lambdify(syms, expr, modules="math"),
    )


# Particular solution and each homogeneous basis function, with derivatives
# needed by the seven boundary conditions. Built once at import.
def _component_funcs(expr: sp.Expr, with_A: bool):
    quants = (
        expr,
        sp.diff(expr, _X),
        sp.diff(expr, _Y),
        sp.diff(expr, _X, 2),
        sp.diff(expr, _Y, 2),
    )
    syms = (_X, _Y, _A) if with_A else (_X, _Y)
    return tuple(_lambdas(q, syms)[0] for q in quants)


_PART = _component_funcs(_PSI_P, with_A=True)
_HOMO = tuple(_component_funcs(p, with_A=False) for p in _BASIS)

_ARGS = (_X, _Y, _A, *_C)
_PSI_NP, _PSI_MATH = _lambdas(_PSI_BAR, _ARGS)
_DPSI_DX_NP, _DPSI_DX_MATH = _lambdas(sp.diff(_PSI_BAR, _X), _ARGS)
_DPSI_DY_NP, _DPSI_DY_MATH = _lambdas(sp.diff(_PSI_BAR, _Y), _ARGS)


def _as_float_eval(fn_np, fn_math, x, y, A, coeffs, scalar: bool):
    c = tuple(float(v) for v in coeffs)
    if scalar:
        return float(fn_math(float(x), float(y), float(A), *c))
    return np.asarray(fn_np(x, y, A, *c), dtype=float)


def _solve_coefficients(epsilon: float, kappa: float, delta: float, A: float) -> np.ndarray:
    """Seven up-down symmetric Cerfon-Freidberg boundary conditions."""
    alpha = float(np.arcsin(delta))
    n1 = -((1.0 + alpha) ** 2) / (epsilon * kappa**2)
    n2 = ((1.0 - alpha) ** 2) / (epsilon * kappa**2)
    n3 = -kappa / (epsilon * np.cos(alpha) ** 2)
    points = (
        (1.0 + epsilon, 0.0),
        (1.0 - epsilon, 0.0),
        (1.0 - delta * epsilon, kappa * epsilon),
    )

    def evaluate(funcs, with_A: bool):
        rows = []
        for xx, yy in points:
            if with_A:
                rows.append([float(funcs[k](xx, yy, A)) for k in range(5)])
            else:
                rows.append([float(funcs[k](xx, yy)) for k in range(5)])
        return rows

    def constraint(vals):
        outer, inner, top = vals
        # psi, psi_x, psi_y, psi_xx, psi_yy
        return np.array(
            [
                outer[0],
                inner[0],
                top[0],
                top[1],
                outer[4] + n1 * outer[1],
                inner[4] + n2 * inner[1],
                top[3] + n3 * top[2],
            ],
            dtype=float,
        )

    matrix = np.column_stack(
        [constraint(evaluate(funcs, with_A=False)) for funcs in _HOMO]
    )
    rhs = -constraint(evaluate(_PART, with_A=True))
    return np.linalg.solve(matrix, rhs)


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
        return self._eval_bar(R, Z, _PSI_NP, _PSI_MATH)

    def grad_psi(self, R, Z) -> tuple[np.ndarray, np.ndarray]:
        """(dpsi/dR, dpsi/dZ) of psi(R, Z) in physical coordinates."""
        dpsi_dx = self._eval_bar(R, Z, _DPSI_DX_NP, _DPSI_DX_MATH)
        dpsi_dy = self._eval_bar(R, Z, _DPSI_DY_NP, _DPSI_DY_MATH)
        scale = 1.0 / self.R0
        return dpsi_dx * scale, dpsi_dy * scale

    def gs_rhs(self, R, Z) -> np.ndarray:
        """Delta* psi in physical (R, Z) coordinates: ((1 - A) x^2 + A) / R0^2."""
        r = np.asarray(R, dtype=float)
        z = np.asarray(Z, dtype=float)
        rb, _zb = np.broadcast_arrays(r, z)
        x = rb / self.R0
        return ((1.0 - self.A) * x**2 + self.A) / self.R0**2

    def boundary(self, n: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """(R_b, Z_b) points of the Miller-like target boundary in metres."""
        tau = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
        alpha = float(np.arcsin(self.delta))
        x = 1.0 + self.epsilon * np.cos(tau + alpha * np.sin(tau))
        y = self.epsilon * self.kappa * np.sin(tau)
        return x * self.R0, y * self.R0

    def level_set(self, R, Z) -> np.ndarray:
        """Negative inside the plasma, zero on the psi_bar = 0 surface, positive outside.

        Equal to psi_bar inside a bounding box around the plasma and forced positive outside
        it, so that spurious psi_bar < 0 regions far from the plasma are excluded.
        """
        scalar = np.ndim(R) == 0 and np.ndim(Z) == 0
        if scalar:
            x = float(R) / self.R0
            y = float(Z) / self.R0
            if not self._in_box(x, y):
                return 1.0
            return float(_as_float_eval(_PSI_NP, _PSI_MATH, x, y, self.A, self.coeffs, True))

        r = np.asarray(R, dtype=float)
        z = np.asarray(Z, dtype=float)
        rb, zb = np.broadcast_arrays(r, z)
        x = rb / self.R0
        y = zb / self.R0
        xmin, xmax, ymin, ymax = self._box_norm()
        inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        out = np.ones(rb.shape, dtype=float)
        if np.any(inside):
            out[inside] = np.asarray(
                _PSI_NP(x[inside], y[inside], self.A, *self.coeffs), dtype=float
            )
        return out

    def _box_norm(self) -> tuple[float, float, float, float]:
        """Miller extent padded by 10% of the normalized minor radius on each side."""
        pad = 0.1 * self.epsilon
        xmin = 1.0 - self.epsilon - pad
        xmax = 1.0 + self.epsilon + pad
        ymin = -(self.kappa * self.epsilon + pad)
        ymax = self.kappa * self.epsilon + pad
        return xmin, xmax, ymin, ymax

    def _in_box(self, x: float, y: float) -> bool:
        xmin, xmax, ymin, ymax = self._box_norm()
        return xmin <= x <= xmax and ymin <= y <= ymax

    def _eval_bar(self, R, Z, fn_np, fn_math):
        scalar = np.ndim(R) == 0 and np.ndim(Z) == 0
        if scalar:
            x = float(R) / self.R0
            y = float(Z) / self.R0
            return _as_float_eval(fn_np, fn_math, x, y, self.A, self.coeffs, True)
        r = np.asarray(R, dtype=float)
        z = np.asarray(Z, dtype=float)
        rb, zb = np.broadcast_arrays(r, z)
        x = rb / self.R0
        y = zb / self.R0
        return _as_float_eval(fn_np, fn_math, x, y, self.A, self.coeffs, False)


def solovev(epsilon: float, kappa: float, delta: float, A: float, R0: float = 1.0) -> SolovevEquilibrium:
    """Build the Cerfon-Freidberg up-down symmetric Solov'ev equilibrium."""
    coeffs = _solve_coefficients(float(epsilon), float(kappa), float(delta), float(A))
    return SolovevEquilibrium(
        epsilon=float(epsilon),
        kappa=float(kappa),
        delta=float(delta),
        A=float(A),
        R0=float(R0),
        coeffs=np.asarray(coeffs, dtype=float),
    )
