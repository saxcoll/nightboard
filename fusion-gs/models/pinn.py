"""Physics-informed neural network for the fixed-boundary Grad-Shafranov equation.

The network is an MLP on affinely normalized ``(R, Z)`` (and optional scalar
conditioning inputs). It is trained on the strong form

    Delta* psi = d²psi/dR² - (1/R) dpsi/dR + d²psi/dZ²

with psi = 0 on the known plasma boundary. Boundary nodes are the level-set
zero contour located by ray bisection; the analytic flux is never used as a
target inside the domain.

Cases
-----
solovev
    Cerfon-Freidberg equilibrium (R0=1, eps=0.32, kappa=1.7, delta=0.33,
    A=-0.155). Residual is Delta* psi - gs_rhs.
nonlinear
    Fixed-boundary equilibrium with a peaked current profile. ``psi_axis`` and
    ``lam`` are lagged (Picard): a constant-current warm start, then several
    linear solves with a frozen right-hand side.
parametric
    Same ITER-like shape, conditioned on the Solov'ev parameter A in [-0.3, 0.1].

``train_pinn`` treats ``steps`` as an iteration budget split between Adam and
L-BFGS. Budgets below 30 steps run a single Adam phase only (unit tests).
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.optimize import brentq

from gs.analytic import SolovevEquilibrium, solovev
from gs.solver import (
    MU0,
    Grid,
    ProfileParams,
    ShapeParams,
    _magnetic_axis,  # same sub-grid axis locator as the finite-difference reference
    solve_fixed_boundary,
)

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

LevelSet = Callable[..., np.ndarray]

# ITER-like Cerfon-Freidberg equilibrium used by case "solovev".
SOLOVEV_SPEC = dict(epsilon=0.32, kappa=1.7, delta=0.33, A=-0.155, R0=1.0)
SOLOVEV_EVAL_BOX = (0.45, 1.55, -0.85, 0.85)
# Fixed-boundary nonlinear benchmark (matches the solver tests' coverage).
NONLINEAR_SHAPE = dict(R0=1.7, a=0.6, kappa=1.7, delta=0.3)
NONLINEAR_PROFILE = dict(Ip=1.0e6, beta0=0.5, alpha=1.0, gamma=2.0, B0=2.0)
NONLINEAR_EVAL_BOX = (0.90, 2.55, -1.25, 1.25)
PARAM_A_RANGE = (-0.3, 0.1)
PARAM_TRAIN_A = (-0.30, -0.22, -0.10, 0.00, 0.10)
PARAM_UNSEEN_A = (-0.27, -0.155, -0.05, 0.05)


def _seed_everything(seed: int) -> np.random.Generator:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _as_grad_leaf(t: torch.Tensor) -> torch.Tensor:
    """Leaf tensor that autograd can differentiate, without dropping an existing graph."""
    if t.requires_grad and not t.is_leaf:
        return t
    if t.requires_grad:
        return t
    return t.detach().requires_grad_(True)


def _scalar_level(level_set: LevelSet, R: float, Z: float) -> float:
    return float(np.asarray(level_set(R, Z), dtype=float).reshape(-1)[0])


def ray_boundary_points(
    level_set: LevelSet,
    origin: tuple[float, float],
    n: int,
    *,
    angles: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersection of rays from an interior point with the level_set = 0 contour.

    ``level_set`` is negative inside the plasma. Each ray is rooted with Brent's
    method, so the returned points lie on the true psi = 0 geometry rather than
    on the Miller parameterization (which is only an approximation of that contour).
    """
    n = int(n)
    if n < 3:
        raise ValueError("need at least 3 boundary rays")
    R0, Z0 = float(origin[0]), float(origin[1])
    f0 = _scalar_level(level_set, R0, Z0)
    if not np.isfinite(f0) or f0 >= 0.0:
        raise RuntimeError(f"ray origin ({R0}, {Z0}) is not inside the plasma (level={f0})")
    if angles is None:
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    else:
        angles = np.asarray(angles, dtype=float).reshape(-1)
        if angles.size != n:
            raise ValueError("angles must have length n")
    Rb = np.empty(n, dtype=float)
    Zb = np.empty(n, dtype=float)
    for i, ang in enumerate(angles):
        c = float(np.cos(ang))
        s = float(np.sin(ang))

        def func(t: float, c: float = c, s: float = s) -> float:
            return _scalar_level(level_set, R0 + t * c, Z0 + t * s)

        t_hi = 1.0e-4
        f_hi = func(t_hi)
        grows = 0
        while f_hi < 0.0 and grows < 50:
            t_hi *= 1.55
            f_hi = func(t_hi)
            grows += 1
        if f_hi < 0.0:
            raise RuntimeError(f"ray at angle {ang:.3f} never left the plasma")
        # func(0) < 0 and func(t_hi) >= 0. A root at exactly 0 is not expected.
        t_root = float(brentq(func, 0.0, t_hi, xtol=1e-14, rtol=1e-12))
        Rb[i] = R0 + t_root * c
        Zb[i] = Z0 + t_root * s
    return Rb, Zb


def _padded_bounds(Rb: np.ndarray, Zb: np.ndarray, frac: float = 0.18) -> tuple[tuple[float, float], tuple[float, float]]:
    r0, r1 = float(np.min(Rb)), float(np.max(Rb))
    z0, z1 = float(np.min(Zb)), float(np.max(Zb))
    pr = frac * (r1 - r0)
    pz = frac * (z1 - z0)
    return (r0 - pr, r1 + pr), (z0 - pz, z1 + pz)


def _sample_interior(
    level_set: LevelSet,
    r_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Rejection sample of points with level_set < 0 inside the bounding box."""
    n = int(n)
    got_r: list[np.ndarray] = []
    got_z: list[np.ndarray] = []
    total = 0
    r0, r1 = r_bounds
    z0, z1 = z_bounds
    # Safety cap: a star-shaped plasma fills a healthy fraction of the padded box.
    for _ in range(40):
        m = max(4 * n, 4096)
        R = rng.uniform(r0, r1, size=m)
        Z = rng.uniform(z0, z1, size=m)
        inside = np.asarray(level_set(R, Z), dtype=float) < 0.0
        got_r.append(np.asarray(R[inside], dtype=float))
        got_z.append(np.asarray(Z[inside], dtype=float))
        total += int(np.count_nonzero(inside))
        if total >= n:
            break
    if total < n:
        raise RuntimeError(f"only found {total} interior points (requested {n})")
    return np.concatenate(got_r)[:n], np.concatenate(got_z)[:n]


class PINN(torch.nn.Module):
    """MLP for psi(R, Z) with fixed input normalization and an output scale.

    Inputs are mapped affinely from the training box onto [-1, 1]. Optional
    conditioning channels (``n_cond``) are normalized with ``cond_bounds``.
    The raw network output is multiplied by ``output_scale`` so the head stays O(1)
    while psi itself carries the physical magnitude.
    """

    def __init__(
        self,
        r_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
        n_cond: int = 0,
        cond_bounds: Any = None,
        hidden: int = 64,
        n_layers: int = 5,
        output_scale: float = 0.05,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.n_cond = int(n_cond)
        self.activation_name = str(activation)
        try:
            act_cls = {"tanh": torch.nn.Tanh, "silu": torch.nn.SiLU}[self.activation_name]
        except KeyError as exc:
            raise ValueError("activation must be 'tanh' or 'silu'") from exc
        if self.n_layers < 1:
            raise ValueError("n_layers must be positive")
        d = 2 + self.n_cond
        blocks: list[torch.nn.Module] = []
        for _ in range(self.n_layers):
            blocks.append(torch.nn.Linear(d, self.hidden))
            blocks.append(act_cls())
            d = self.hidden
        blocks.append(torch.nn.Linear(d, 1))
        self.net = torch.nn.Sequential(*blocks)
        self._init_weights()

        if self.n_cond == 0:
            cmin = torch.zeros(0)
            cmax = torch.zeros(0)
        else:
            if cond_bounds is None:
                arr = np.stack([-np.ones(self.n_cond), np.ones(self.n_cond)], axis=1)
            else:
                arr = np.asarray(cond_bounds, dtype=float).reshape(self.n_cond, 2)
            cmin = torch.tensor(arr[:, 0], dtype=torch.float64)
            cmax = torch.tensor(arr[:, 1], dtype=torch.float64)
        self.register_buffer("r_min", torch.tensor(float(r_bounds[0]), dtype=torch.float64))
        self.register_buffer("r_max", torch.tensor(float(r_bounds[1]), dtype=torch.float64))
        self.register_buffer("z_min", torch.tensor(float(z_bounds[0]), dtype=torch.float64))
        self.register_buffer("z_max", torch.tensor(float(z_bounds[1]), dtype=torch.float64))
        self.register_buffer("cond_min", cmin)
        self.register_buffer("cond_max", cmax)
        self.register_buffer("output_scale", torch.tensor(float(output_scale), dtype=torch.float64))
        self.to(dtype=torch.float64)

    def _init_weights(self) -> None:
        for mod in self.net.modules():
            if isinstance(mod, torch.nn.Linear):
                torch.nn.init.xavier_normal_(mod.weight)
                torch.nn.init.zeros_(mod.bias)
        last = self.net[-1]
        assert isinstance(last, torch.nn.Linear)
        last.weight.data.mul_(0.1)

    @property
    def dtype(self) -> torch.dtype:
        return self.r_min.dtype

    def config(self) -> dict[str, Any]:
        cond_bounds = None
        if self.n_cond:
            cond_bounds = [
                [float(self.cond_min[i]), float(self.cond_max[i])] for i in range(self.n_cond)
            ]
        return {
            "r_bounds": [float(self.r_min), float(self.r_max)],
            "z_bounds": [float(self.z_min), float(self.z_max)],
            "n_cond": self.n_cond,
            "cond_bounds": cond_bounds,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
            "output_scale": float(self.output_scale),
            "activation": self.activation_name,
        }

    def _as_coord(self, v: Any) -> torch.Tensor:
        if torch.is_tensor(v):
            t = v
            if t.dtype != self.dtype or t.device != self.r_min.device:
                t = t.to(dtype=self.dtype, device=self.r_min.device)
            return t
        return torch.as_tensor(np.asarray(v), dtype=self.dtype, device=self.r_min.device)

    def _affine(self, v: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor) -> torch.Tensor:
        return 2.0 * (v - vmin) / (vmax - vmin) - 1.0

    def _cond_features(self, cond: Any, n: int, like: torch.Tensor) -> torch.Tensor:
        if self.n_cond == 0:
            raise RuntimeError("internal: cond features requested with n_cond=0")
        if cond is None:
            raise ValueError("this PINN expects conditioning inputs")
        c = self._as_coord(cond)
        # n_cond == 1 is the supported training path (parameter A). Also accept a
        # per-point vector or a single value broadcast across the batch.
        if self.n_cond == 1:
            flat = c.reshape(-1)
            if flat.numel() == 1:
                flat = flat.expand(n)
            elif flat.numel() != n:
                raise ValueError(f"cond has {flat.numel()} values, expected 1 or {n}")
            c = flat.reshape(n, 1)
        else:
            if c.ndim == 0 or c.numel() == self.n_cond:
                c = c.reshape(1, self.n_cond).expand(n, self.n_cond)
            elif c.shape == (n, self.n_cond):
                pass
            else:
                raise ValueError(f"cond shape {tuple(c.shape)} is not broadcastable to {(n, self.n_cond)}")
        return self._affine(c, self.cond_min, self.cond_max)

    def forward(self, R: Any, Z: Any, cond: Any = None) -> torch.Tensor:
        R_t = self._as_coord(R)
        Z_t = self._as_coord(Z)
        shape = torch.broadcast_shapes(R_t.shape, Z_t.shape)
        R_b = R_t.expand(shape)
        Z_b = Z_t.expand(shape)
        n = int(shape.numel()) if shape != torch.Size([]) else 1
        Rn = self._affine(R_b, self.r_min, self.r_max).reshape(n)
        Zn = self._affine(Z_b, self.z_min, self.z_max).reshape(n)
        x = torch.stack((Rn, Zn), dim=-1)
        if self.n_cond:
            cn = self._cond_features(cond, n, R_b)
            x = torch.cat((x, cn), dim=-1)
        elif cond is not None:
            raise ValueError("cond was passed to a PINN with n_cond=0")
        raw = self.net(x).squeeze(-1)
        out = self.output_scale * raw
        return out.reshape(shape)


def delta_star_autograd(
    model: torch.nn.Module,
    R: Any,
    Z: Any,
    cond: Any = None,
) -> torch.Tensor:
    """Delta* psi = d²psi/dR² - (1/R) dpsi/dR + d²psi/dZ² via torch.autograd.

    First and second derivatives are built with ``create_graph=True`` so the
    PDE residual can be differentiated with respect to the network weights.
    """
    if not torch.is_tensor(R):
        dtype = getattr(model, "dtype", torch.float64)
        device = getattr(getattr(model, "r_min", None), "device", torch.device("cpu"))
        R = torch.as_tensor(np.asarray(R), dtype=dtype, device=device)
    if not torch.is_tensor(Z):
        Z = torch.as_tensor(np.asarray(Z), dtype=R.dtype, device=R.device)
    elif Z.dtype != R.dtype or Z.device != R.device:
        Z = Z.to(dtype=R.dtype, device=R.device)
    R = _as_grad_leaf(R)
    Z = _as_grad_leaf(Z)
    psi = model(R, Z, cond)

    def deriv(out: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        if out is None or (torch.is_tensor(out) and not out.requires_grad):
            return torch.zeros_like(var)
        grad = torch.autograd.grad(
            out,
            var,
            grad_outputs=torch.ones_like(out),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            return torch.zeros_like(var)
        return grad

    dpsi_dR = deriv(psi, R)
    dpsi_dZ = deriv(psi, Z)
    d2psi_dR2 = deriv(dpsi_dR, R)
    d2psi_dZ2 = deriv(dpsi_dZ, Z)
    return d2psi_dR2 - dpsi_dR / R + d2psi_dZ2


def predict_grid(model: PINN, grid: Grid, cond: Any = None) -> np.ndarray:
    """Evaluate ``model`` on ``grid`` and return psi with shape ``(nr, nz)``."""
    was_training = model.training
    model.eval()
    dtype = model.dtype
    device = model.r_min.device
    R = torch.as_tensor(np.asarray(grid.RR), dtype=dtype, device=device)
    Z = torch.as_tensor(np.asarray(grid.ZZ), dtype=dtype, device=device)
    cond_t = None if cond is None else torch.as_tensor(np.asarray(cond), dtype=dtype, device=device)
    try:
        with torch.no_grad():
            psi = model(R, Z, cond_t)
        out = np.asarray(psi.detach().cpu().numpy(), dtype=float)
    finally:
        model.train(was_training)
    return out.reshape(grid.nr, grid.nz)


def load_pinn(path: str | Path) -> PINN:
    """Rebuild a PINN from a checkpoint written by ``train_pinn``."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model = PINN(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _tensor(arr: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(arr), dtype=torch.float64, device=device)


def _relative_l2(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    diff = np.asarray(pred, dtype=float)[mask] - np.asarray(ref, dtype=float)[mask]
    den = float(np.linalg.norm(np.asarray(ref, dtype=float)[mask]))
    num = float(np.linalg.norm(diff))
    if den == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return num / den


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class _Batch:
    """Collocation and boundary tensors for one linear residual fit."""

    def __init__(self) -> None:
        self.R: torch.Tensor | None = None
        self.Z: torch.Tensor | None = None
        self.rhs: torch.Tensor | None = None
        self.cond: torch.Tensor | None = None
        self.Rb: torch.Tensor | None = None
        self.Zb: torch.Tensor | None = None
        self.cond_b: torch.Tensor | None = None
        self.scale_r: float = 1.0
        self.scale_b: float = 1.0
        self.w_b: float = 100.0


def _loss_parts(model: PINN, batch: _Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert batch.R is not None and batch.Z is not None and batch.rhs is not None
    assert batch.Rb is not None and batch.Zb is not None
    R = batch.R.detach().requires_grad_(True)
    Z = batch.Z.detach().requires_grad_(True)
    ds = delta_star_autograd(model, R, Z, batch.cond)
    loss_r = (ds - batch.rhs).pow(2).mean() / batch.scale_r
    psi_b = model(batch.Rb, batch.Zb, batch.cond_b)
    loss_b = psi_b.pow(2).mean() / batch.scale_b
    loss = loss_r + batch.w_b * loss_b
    return loss, loss_r, loss_b


def _clone_state(model: PINN) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _restore_state(model: PINN, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)


def _fit_residual(
    model: PINN,
    batch: _Batch,
    *,
    adam_steps: int,
    lbfgs_steps: int,
    lr: float,
    verbose: bool,
    tag: str,
    resample: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Adam (optional point refresh) followed by L-BFGS on a frozen objective.

    The best weights of each phase are restored, measured by that phase's loss.
    """
    adam_steps = int(adam_steps)
    lbfgs_steps = int(lbfgs_steps)
    closures = 0
    last = {"loss": float("nan"), "pde": float("nan"), "bc": float("nan")}

    def eval_loss() -> tuple[torch.Tensor, float, float, float]:
        loss, loss_r, loss_b = _loss_parts(model, batch)
        return loss, float(loss.detach()), float(loss_r.detach()), float(loss_b.detach())

    if adam_steps > 0:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(adam_steps, 1), eta_min=min(lr, 1.0e-4)
        )
        best_val = float("inf")
        best_state = _clone_state(model)
        log_every = max(adam_steps // 5, 1)
        for step in range(1, adam_steps + 1):
            if resample is not None and step > 1 and (step - 1) % 250 == 0:
                resample()
                best_val = float("inf")
            opt.zero_grad(set_to_none=True)
            loss, loss_v, pde_v, bc_v = eval_loss()
            if not math.isfinite(loss_v):
                break
            loss.backward()
            opt.step()
            sched.step()
            last = {"loss": loss_v, "pde": pde_v, "bc": bc_v}
            if loss_v < best_val:
                best_val = loss_v
                best_state = _clone_state(model)
            if verbose and (step == 1 or step % log_every == 0 or step == adam_steps):
                print(
                    f"[{tag}] adam {step}/{adam_steps} loss {loss_v:.3e} "
                    f"pde {pde_v:.3e} bc {bc_v:.3e} w_b {batch.w_b:.1f}",
                    flush=True,
                )
        _restore_state(model, best_state)
        _, loss_v, pde_v, bc_v = eval_loss()
        last = {"loss": float(loss_v), "pde": float(pde_v), "bc": float(bc_v)}

    if lbfgs_steps > 0:
        # Fresh collocation set so L-BFGS is not scored on the Adam sample alone.
        if resample is not None:
            resample()

        def _new_lbfgs() -> torch.optim.LBFGS:
            return torch.optim.LBFGS(
                model.parameters(),
                lr=1.0,
                max_iter=20,
                max_eval=60,
                history_size=50,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-12,
                tolerance_change=1e-14,
            )

        opt_l = _new_lbfgs()
        # Plain `<` against +inf. Do not subtract a tolerance: inf - inf is NaN
        # and then every comparison fails, which discards the L-BFGS updates.
        best_val = float("inf")
        best_state = _clone_state(model)
        stall = 0
        half = max(lbfgs_steps // 2, 1)
        refreshed = False
        improved = False

        def closure() -> torch.Tensor:
            nonlocal closures, best_val, best_state, improved
            opt_l.zero_grad(set_to_none=True)
            loss, _, _ = _loss_parts(model, batch)
            val = float(loss.detach())
            if not math.isfinite(val):
                raise RuntimeError("non-finite PINN loss during L-BFGS")
            # Snapshot the weights that produced this value. Line search tries
            # several points; the lowest one is restored after the step.
            if val < best_val:
                best_val = val
                best_state = _clone_state(model)
                improved = True
            loss.backward()
            closures += 1
            return loss

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Converting a tensor with requires_grad=True to a scalar",
            )
            while closures < lbfgs_steps and stall < 6:
                improved = False
                before = closures
                try:
                    opt_l.step(closure)
                except RuntimeError as exc:
                    if verbose:
                        print(f"[{tag}] lbfgs stopped: {exc}", flush=True)
                    break
                if closures == before:
                    break
                stall = 0 if improved else stall + 1
                if verbose:
                    print(
                        f"[{tag}] lbfgs closures {closures}/{lbfgs_steps} best {best_val:.3e}",
                        flush=True,
                    )
                if (not refreshed) and resample is not None and closures >= half:
                    _restore_state(model, best_state)
                    resample()
                    opt_l = _new_lbfgs()
                    best_val = float("inf")
                    stall = 0
                    refreshed = True
        _restore_state(model, best_state)
        _, loss_v, pde_v, bc_v = eval_loss()
        last = {"loss": float(loss_v), "pde": float(pde_v), "bc": float(bc_v)}

    last["lbfgs_closures"] = float(closures)
    last["adam_steps"] = float(adam_steps)
    return last


def _make_grid(box: tuple[float, float, float, float], n: int) -> Grid:
    return Grid.uniform(box[0], box[1], box[2], box[3], n, n)


def _save_figure(
    path: Path,
    grid: Grid,
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
    Rb: np.ndarray,
    Zb: np.ndarray,
    rel_l2: float,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    pred_m = np.where(mask, pred, np.nan)
    ref_m = np.where(mask, ref, np.nan)
    err = np.where(mask, pred - ref, np.nan)
    vmin = float(np.nanmin([np.nanmin(ref_m), np.nanmin(pred_m)]))
    vmax = float(np.nanmax([np.nanmax(ref_m), np.nanmax(pred_m)]))
    elim = float(np.nanmax(np.abs(err)))
    if not np.isfinite(elim) or elim == 0.0:
        elim = 1e-6
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), constrained_layout=True)
    panels = (
        (axes[0], ref_m, "reference psi", vmin, vmax, "viridis"),
        (axes[1], pred_m, "PINN psi", vmin, vmax, "viridis"),
        (axes[2], err, f"error  rel L2 = {rel_l2:.3e}", -elim, elim, "coolwarm"),
    )
    Rb_c = np.r_[Rb, Rb[:1]]
    Zb_c = np.r_[Zb, Zb[:1]]
    for ax, data, label, v0, v1, cmap in panels:
        mesh = ax.pcolormesh(grid.RR, grid.ZZ, data, shading="auto", cmap=cmap, vmin=v0, vmax=v1)
        ax.plot(Rb_c, Zb_c, color="k", lw=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(label)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_outputs(
    out_dir: str | Path,
    case: str,
    model: PINN,
    metrics: dict[str, Any],
    grid: Grid,
    pred: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
    Rb: np.ndarray,
    Zb: np.ndarray,
    title: str | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / f"{case}_metrics.json"
    ckpt_path = out / f"{case}.pt"
    fig_path = out / f"{case}.png"
    metrics["checkpoint"] = str(ckpt_path)
    metrics["metrics_json"] = str(metrics_path)
    metrics["figure"] = str(fig_path)
    payload = _jsonify(metrics)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config(), "metrics": payload},
        ckpt_path,
    )
    rel = float(metrics.get("rel_l2", float("nan")))
    _save_figure(
        fig_path,
        grid,
        pred,
        ref,
        mask,
        Rb,
        Zb,
        rel,
        title=title or f"{case} PINN",
    )


def _budget(case: str, steps: int, n_interior: int | None, n_boundary: int | None, lbfgs_steps: int | None) -> dict[str, int]:
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if steps < 30:
        return {
            "adam": steps,
            "lbfgs": 0 if lbfgs_steps is None else int(lbfgs_steps),
            "n_interior": 160 if n_interior is None else int(n_interior),
            "n_boundary": 48 if n_boundary is None else int(n_boundary),
            "warm_adam": steps,
            "warm_lbfgs": 0,
            "n_picard": 0,
            "inner_adam": 0,
            "inner_lbfgs": 0,
            "quality": 0,
        }
    if n_interior is None:
        n_interior = 3840 if case == "parametric" else 3072
    if n_boundary is None:
        n_boundary = 512
    if case == "nonlinear":
        scale = 1.0 if steps >= 2000 else max(0.35, steps / 2000.0)
        return {
            "adam": 0,
            "lbfgs": 0,
            "n_interior": int(n_interior),
            "n_boundary": int(n_boundary),
            "warm_adam": int(round(450 * scale)),
            "warm_lbfgs": int(round(350 * scale)),
            "n_picard": max(4, int(round(6 * scale))),
            "inner_adam": max(20, int(round(40 * scale))),
            "inner_lbfgs": max(40, int(round(110 * scale))),
            "quality": 1,
        }
    if lbfgs_steps is None:
        adam = min(700, max(450, steps // 6))
        lbfgs = min(1100, max(400, steps // 3))
    else:
        adam = steps
        lbfgs = int(lbfgs_steps)
    return {
        "adam": adam,
        "lbfgs": lbfgs,
        "n_interior": int(n_interior),
        "n_boundary": int(n_boundary),
        "warm_adam": 0,
        "warm_lbfgs": 0,
        "n_picard": 0,
        "inner_adam": 0,
        "inner_lbfgs": 0,
        "quality": 1,
    }


def _boundary_max(model: PINN, Rb: np.ndarray, Zb: np.ndarray, cond_b: np.ndarray | None = None) -> float:
    model.eval()
    with torch.no_grad():
        psi = model(
            _tensor(Rb, model.r_min.device),
            _tensor(Zb, model.r_min.device),
            None if cond_b is None else _tensor(cond_b, model.r_min.device),
        )
    model.train()
    return float(psi.abs().max().detach())


def _train_solovev_family(
    *,
    case: str,
    steps: int,
    seed: int,
    n_interior: int | None,
    n_boundary: int | None,
    lbfgs_steps: int | None,
    hidden: int,
    n_layers: int,
    lr: float,
    w_b: float,
    verbose: bool,
    device: str,
) -> tuple[PINN, dict[str, Any], dict[str, Any]]:
    rng = _seed_everything(seed)
    bud = _budget(case, steps, n_interior, n_boundary, lbfgs_steps)
    if case == "solovev":
        eqs = [solovev(**SOLOVEV_SPEC)]
        labels = [float(SOLOVEV_SPEC["A"])]
    elif case == "parametric":
        labels = [float(a) for a in PARAM_TRAIN_A]
        eqs = [
            solovev(
                epsilon=SOLOVEV_SPEC["epsilon"],
                kappa=SOLOVEV_SPEC["kappa"],
                delta=SOLOVEV_SPEC["delta"],
                A=a,
                R0=SOLOVEV_SPEC["R0"],
            )
            for a in labels
        ]
    else:
        raise ValueError(case)

    origin = (float(SOLOVEV_SPEC["R0"]), 0.0)
    boundaries = [ray_boundary_points(eq.level_set, origin, bud["n_boundary"]) for eq in eqs]
    all_R = np.concatenate([b[0] for b in boundaries])
    all_Z = np.concatenate([b[1] for b in boundaries])
    r_bounds, z_bounds = _padded_bounds(all_R, all_Z)
    n_cond = 0 if case == "solovev" else 1
    cond_bounds = None if n_cond == 0 else [list(PARAM_A_RANGE)]
    # psi_bar is about -0.037 at the axis; 0.05 keeps the raw head O(1).
    model = PINN(
        r_bounds,
        z_bounds,
        n_cond=n_cond,
        cond_bounds=cond_bounds,
        hidden=hidden,
        n_layers=n_layers,
        output_scale=0.05,
        activation="tanh",
    ).to(device)
    model.train()

    n_each = max(32, bud["n_interior"] // len(eqs))
    # Characteristic |psi| used only to normalize the boundary penalty (not a target field).
    psi_scale = 0.02

    def pack(eqs_local: list[SolovevEquilibrium] = eqs) -> _Batch:
        rs = []
        zs = []
        rhs = []
        conds = []
        rbs = []
        zbs = []
        condbs = []
        for eq, A, (Rb, Zb) in zip(eqs_local, labels, boundaries):
            R, Z = _sample_interior(eq.level_set, r_bounds, z_bounds, n_each, rng)
            rs.append(R)
            zs.append(Z)
            rhs.append(np.asarray(eq.gs_rhs(R, Z), dtype=float))
            conds.append(np.full(R.shape, A, dtype=float))
            rbs.append(Rb)
            zbs.append(Zb)
            condbs.append(np.full(Rb.shape, A, dtype=float))
        batch = _Batch()
        dev = model.r_min.device
        batch.R = _tensor(np.concatenate(rs), dev)
        batch.Z = _tensor(np.concatenate(zs), dev)
        batch.rhs = _tensor(np.concatenate(rhs), dev)
        batch.Rb = _tensor(np.concatenate(rbs), dev)
        batch.Zb = _tensor(np.concatenate(zbs), dev)
        if n_cond:
            batch.cond = _tensor(np.concatenate(conds), dev)
            batch.cond_b = _tensor(np.concatenate(condbs), dev)
        batch.scale_r = float(batch.rhs.pow(2).mean().clamp_min(1e-30))
        batch.scale_b = psi_scale**2
        batch.w_b = float(w_b)
        return batch

    batch = pack()
    # Keep the PDE normalization fixed so Adam and L-BFGS see the same loss scale
    # when collocation points are refreshed.
    scale_r_fixed = batch.scale_r

    def resample() -> None:
        fresh = pack()
        batch.R, batch.Z, batch.rhs = fresh.R, fresh.Z, fresh.rhs
        batch.cond = fresh.cond
        batch.scale_r = scale_r_fixed

    # One-step tests should not pay for a second sample inside L-BFGS.
    do_resample = bool(bud["quality"])
    fit = _fit_residual(
        model,
        batch,
        adam_steps=bud["adam"],
        lbfgs_steps=bud["lbfgs"],
        lr=lr,
        verbose=verbose,
        tag=case,
        resample=resample if do_resample else None,
    )

    grid = _make_grid(SOLOVEV_EVAL_BOX, 129)
    if case == "solovev":
        eq = eqs[0]
        ref = np.asarray(eq.psi(grid.RR, grid.ZZ), dtype=float)
        mask = np.asarray(eq.level_set(grid.RR, grid.ZZ), dtype=float) < 0.0
        pred = predict_grid(model, grid)
        rel = _relative_l2(pred, ref, mask)
        max_err = float(np.max(np.abs(pred[mask] - ref[mask])))
        bmax = _boundary_max(model, boundaries[0][0], boundaries[0][1])
        metrics: dict[str, Any] = {
            "case": case,
            "rel_l2": rel,
            "max_err": max_err,
            "boundary_max_abs": bmax,
            "A": float(SOLOVEV_SPEC["A"]),
        }
        pack_eval = {
            "grid": grid,
            "pred": pred,
            "ref": ref,
            "mask": mask,
            "Rb": boundaries[0][0],
            "Zb": boundaries[0][1],
        }
    else:
        per_a: dict[str, Any] = {}
        worst_rel = -1.0
        worst_A = float(PARAM_UNSEEN_A[0])
        worst_pack: dict[str, Any] | None = None
        for A in PARAM_UNSEEN_A:
            eq = solovev(
                epsilon=SOLOVEV_SPEC["epsilon"],
                kappa=SOLOVEV_SPEC["kappa"],
                delta=SOLOVEV_SPEC["delta"],
                A=float(A),
                R0=SOLOVEV_SPEC["R0"],
            )
            ref = np.asarray(eq.psi(grid.RR, grid.ZZ), dtype=float)
            mask = np.asarray(eq.level_set(grid.RR, grid.ZZ), dtype=float) < 0.0
            pred = predict_grid(model, grid, cond=float(A))
            rel = _relative_l2(pred, ref, mask)
            max_err = float(np.max(np.abs(pred[mask] - ref[mask])))
            Rb, Zb = ray_boundary_points(eq.level_set, origin, 128)
            per_a[f"{A:.3f}"] = {
                "rel_l2": rel,
                "max_err": max_err,
                "boundary_max_abs": _boundary_max(model, Rb, Zb, np.full(Rb.shape, float(A))),
            }
            if rel > worst_rel:
                worst_rel = rel
                worst_A = float(A)
                worst_pack = {
                    "grid": grid,
                    "pred": pred,
                    "ref": ref,
                    "mask": mask,
                    "Rb": Rb,
                    "Zb": Zb,
                    "title": f"parametric PINN (unseen A = {float(A):.3f})",
                }
        rels = [float(v["rel_l2"]) for v in per_a.values()]
        metrics = {
            "case": case,
            "rel_l2": float(max(rels)),
            "rel_l2_mean": float(np.mean(rels)),
            "max_err": float(max(float(v["max_err"]) for v in per_a.values())),
            "unseen_A": per_a,
            "train_A": list(labels),
            "figure_A": worst_A,
            "boundary_max_abs": float(max(float(v["boundary_max_abs"]) for v in per_a.values())),
        }
        assert worst_pack is not None
        pack_eval = worst_pack

    metrics.update(
        {
            "final_loss": fit["loss"],
            "loss_pde": fit["pde"],
            "loss_bc": fit["bc"],
            "steps_requested": int(steps),
            "steps": int(fit["adam_steps"] + fit["lbfgs_closures"]),
            "adam_steps": int(fit["adam_steps"]),
            "lbfgs_closures": int(fit["lbfgs_closures"]),
            "seed": int(seed),
            "hidden": int(hidden),
            "n_layers": int(n_layers),
            "activation": "tanh",
            "n_interior": int(n_each * len(eqs)),
            "n_boundary": int(bud["n_boundary"] * len(eqs)),
            "w_b": float(w_b),
            "output_scale": float(model.output_scale),
            "psi_scale": psi_scale,
            "lr": float(lr),
            "dtype": "float64",
        }
    )
    return model, metrics, pack_eval


def _profile_factor(
    R: torch.Tensor,
    psi: torch.Tensor,
    psi_axis: float,
    shape: ShapeParams,
    profile: ProfileParams,
) -> torch.Tensor:
    """g = (beta0 R/R0 + (1-beta0) R0/R) (1 - psi_n^alpha)^gamma, detached from psi_axis."""
    axis = torch.as_tensor(psi_axis, dtype=psi.dtype, device=psi.device)
    pn = (psi - axis) / (0.0 - axis)
    pn = pn.clamp(0.0, 1.0)
    gpsi = (1.0 - pn.pow(profile.alpha)).clamp(min=0.0).pow(profile.gamma)
    geo = profile.beta0 * (R / shape.R0) + (1.0 - profile.beta0) * (shape.R0 / R)
    return geo * gpsi


def _nonlinear_state(
    model: PINN,
    shape: ShapeParams,
    profile: ProfileParams,
    quad: Grid,
    mask: np.ndarray,
    R_col: torch.Tensor,
    Z_col: torch.Tensor,
) -> dict[str, Any]:
    """Lagged psi_axis, lam and collocation RHS from the current network."""
    pred = predict_grid(model, quad)
    R_axis, Z_axis, psi_axis = _magnetic_axis(pred, mask, quad)
    with torch.no_grad():
        psi_c = model(R_col, Z_col)
    psi_axis = float(max(psi_axis, float(psi_c.max().detach())))
    if not np.isfinite(psi_axis) or abs(psi_axis) < 1e-8:
        raise RuntimeError(f"psi_axis is not usable ({psi_axis})")
    dev = R_col.device
    R_q = _tensor(quad.RR[mask], dev)
    Z_q = _tensor(quad.ZZ[mask], dev)
    with torch.no_grad():
        psi_q = model(R_q, Z_q)
        g_q = _profile_factor(R_q, psi_q, psi_axis, shape, profile)
        g_c = _profile_factor(R_col, psi_c, psi_axis, shape, profile)
    dA = float(quad.dR * quad.dZ)
    integral = float(g_q.sum().detach()) * dA
    if integral <= 0.0 or not np.isfinite(integral):
        raise RuntimeError("profile integral is not positive")
    lam = float(profile.Ip) / integral
    rhs = (-MU0 * lam) * R_col * g_c
    return {
        "rhs": rhs.detach(),
        "lam": lam,
        "psi_axis": psi_axis,
        "R_axis": float(R_axis),
        "Z_axis": float(Z_axis),
        "integral": integral,
    }


def _train_nonlinear(
    *,
    steps: int,
    seed: int,
    n_interior: int | None,
    n_boundary: int | None,
    lbfgs_steps: int | None,
    hidden: int,
    n_layers: int,
    lr: float,
    w_b: float,
    verbose: bool,
    device: str,
) -> tuple[PINN, dict[str, Any], dict[str, Any]]:
    rng = _seed_everything(seed)
    bud = _budget("nonlinear", steps, n_interior, n_boundary, lbfgs_steps)
    shape = ShapeParams(**NONLINEAR_SHAPE)
    profile = ProfileParams(**NONLINEAR_PROFILE)
    level = shape.level_set()
    origin = (float(shape.R0), 0.0)
    Rb_np, Zb_np = ray_boundary_points(level, origin, bud["n_boundary"])
    r_bounds, z_bounds = _padded_bounds(Rb_np, Zb_np)
    # psi_axis of the reference equilibrium is ~0.36; scale 0.5 keeps the head O(1).
    model = PINN(
        r_bounds,
        z_bounds,
        n_cond=0,
        hidden=hidden,
        n_layers=n_layers,
        output_scale=0.5,
        activation="tanh",
    ).to(device)
    model.train()

    quad = _make_grid(NONLINEAR_EVAL_BOX, 129 if bud["quality"] else 65)
    mask = np.asarray(level(quad.RR, quad.ZZ), dtype=float) < 0.0
    if int(mask.sum()) < 10:
        raise RuntimeError("quadrature grid does not cover the plasma")
    area = float(mask.sum()) * float(quad.dR * quad.dZ)
    J_const = float(profile.Ip) / area
    psi_scale = 0.20

    dev = model.r_min.device
    R_np, Z_np = _sample_interior(level, r_bounds, z_bounds, bud["n_interior"], rng)
    batch = _Batch()
    batch.R = _tensor(R_np, dev)
    batch.Z = _tensor(Z_np, dev)
    batch.Rb = _tensor(Rb_np, dev)
    batch.Zb = _tensor(Zb_np, dev)
    batch.rhs = (-MU0 * J_const) * batch.R
    batch.scale_r = float(batch.rhs.pow(2).mean().clamp_min(1e-30))
    batch.scale_b = psi_scale**2
    batch.w_b = float(w_b)
    scale_b = batch.scale_b

    def resample_const() -> None:
        R, Z = _sample_interior(level, r_bounds, z_bounds, bud["n_interior"], rng)
        batch.R = _tensor(R, dev)
        batch.Z = _tensor(Z, dev)
        batch.rhs = (-MU0 * J_const) * batch.R

    if verbose:
        print(
            f"[nonlinear] constant-current warm start  area={area:.4f} J={J_const:.4e}",
            flush=True,
        )
    fit = _fit_residual(
        model,
        batch,
        adam_steps=bud["warm_adam"],
        lbfgs_steps=bud["warm_lbfgs"],
        lr=lr,
        verbose=verbose,
        tag="nonlinear-warm",
        resample=resample_const if bud["quality"] else None,
    )
    adam_total = int(fit["adam_steps"])
    lbfgs_total = int(fit["lbfgs_closures"])
    history: list[dict[str, float]] = []
    prev_rhs: torch.Tensor | None = None

    for outer in range(bud["n_picard"]):
        state = _nonlinear_state(model, shape, profile, quad, mask, batch.R, batch.Z)
        rhs_new = state["rhs"]
        if prev_rhs is None:
            rel_change = 1.0
            rhs_target = rhs_new
        else:
            rel_change = float(
                (rhs_new - prev_rhs).pow(2).mean().sqrt()
                / rhs_new.pow(2).mean().sqrt().clamp_min(1e-30)
            )
            # Under-relax early Picard updates; last iterations use the true profile.
            relax = 1.0 if outer + 1 >= bud["n_picard"] - 1 else 0.6
            rhs_target = (1.0 - relax) * prev_rhs + relax * rhs_new
        prev_rhs = rhs_new.detach()
        batch.rhs = rhs_target.detach()
        batch.scale_r = float(batch.rhs.pow(2).mean().clamp_min(1e-30))
        batch.scale_b = scale_b

        if verbose:
            print(
                f"[nonlinear] picard {outer + 1}/{bud['n_picard']} "
                f"psi_axis={state['psi_axis']:.5f} R_axis={state['R_axis']:.4f} "
                f"lam={state['lam']:.5e} rhs_change={rel_change:.3e}",
                flush=True,
            )
        # Collocation is held fixed inside a Picard iteration so the lagged
        # (psi_axis, lam, g) target does not move under the inner solve.
        fit = _fit_residual(
            model,
            batch,
            adam_steps=bud["inner_adam"],
            lbfgs_steps=bud["inner_lbfgs"],
            lr=lr * 0.5,
            verbose=verbose,
            tag=f"nonlinear-p{outer + 1}",
            resample=None,
        )
        adam_total += int(fit["adam_steps"])
        lbfgs_total += int(fit["lbfgs_closures"])
        history.append(
            {
                "outer": float(outer + 1),
                "psi_axis": float(state["psi_axis"]),
                "R_axis": float(state["R_axis"]),
                "Z_axis": float(state["Z_axis"]),
                "lam": float(state["lam"]),
                "rhs_change": float(rel_change),
                "loss": float(fit["loss"]),
            }
        )
        if outer >= 2 and rel_change < 5e-4 and outer + 1 >= bud["n_picard"] - 1:
            break

    # Reference equilibrium on the same 129 grid used for the quadrature.
    eval_grid = quad if quad.nr == 129 else _make_grid(NONLINEAR_EVAL_BOX, 129)
    ref = solve_fixed_boundary(shape, profile, eval_grid)
    pred = predict_grid(model, eval_grid)
    mask_e = ref.mask
    rel = _relative_l2(pred, ref.psi, mask_e)
    max_err = float(np.max(np.abs(pred[mask_e] - ref.psi[mask_e])))
    R_axis, Z_axis, psi_axis = _magnetic_axis(pred, mask_e, eval_grid)
    bmax = _boundary_max(model, Rb_np, Zb_np)
    metrics = {
        "case": "nonlinear",
        "rel_l2": rel,
        "max_err": max_err,
        "boundary_max_abs": bmax,
        "psi_axis": float(psi_axis),
        "psi_axis_ref": float(ref.psi_axis),
        "psi_axis_error": float(abs(psi_axis - ref.psi_axis)),
        "R_axis": float(R_axis),
        "R_axis_ref": float(ref.R_axis),
        "R_axis_error": float(abs(R_axis - ref.R_axis)),
        "Z_axis": float(Z_axis),
        "Z_axis_ref": float(ref.Z_axis),
        "Z_axis_error": float(abs(Z_axis - ref.Z_axis)),
        "lam_ref": float(ref.lam),
        "final_loss": fit["loss"],
        "loss_pde": fit["pde"],
        "loss_bc": fit["bc"],
        "steps_requested": int(steps),
        "steps": int(adam_total + lbfgs_total),
        "adam_steps": adam_total,
        "lbfgs_closures": lbfgs_total,
        "seed": int(seed),
        "hidden": int(hidden),
        "n_layers": int(n_layers),
        "activation": "tanh",
        "n_interior": int(bud["n_interior"]),
        "n_boundary": int(bud["n_boundary"]),
        "w_b": float(w_b),
        "output_scale": float(model.output_scale),
        "psi_scale": psi_scale,
        "lr": float(lr),
        "dtype": "float64",
        "picard": history,
        "area": area,
    }
    # lam consistent with the predicted field and the same quadrature as the reference.
    try:
        final_state = _nonlinear_state(
            model,
            shape,
            profile,
            eval_grid,
            mask_e,
            _tensor(eval_grid.RR[mask_e], dev),
            _tensor(eval_grid.ZZ[mask_e], dev),
        )
        metrics["lam"] = float(final_state["lam"])
        metrics["lam_rel_error"] = float(abs(final_state["lam"] - ref.lam) / abs(ref.lam))
    except RuntimeError:
        metrics["lam"] = None
        metrics["lam_rel_error"] = None
    pack_eval = {
        "grid": eval_grid,
        "pred": pred,
        "ref": np.asarray(ref.psi, dtype=float),
        "mask": np.asarray(mask_e, dtype=bool),
        "Rb": Rb_np,
        "Zb": Zb_np,
    }
    return model, metrics, pack_eval


def train_pinn(
    case: str,
    steps: int,
    seed: int = 0,
    *,
    out_dir: str | Path | None = None,
    n_interior: int | None = None,
    n_boundary: int | None = None,
    lbfgs_steps: int | None = None,
    hidden: int = 64,
    n_layers: int = 5,
    lr: float = 1.0e-3,
    w_b: float | None = None,
    verbose: bool = False,
    device: str = "cpu",
) -> tuple[PINN, dict[str, Any]]:
    """Train a Grad-Shafranov PINN.

    Parameters
    ----------
    case:
        ``solovev``, ``nonlinear``, or ``parametric``.
    steps:
        Iteration budget. Values under 30 run that many Adam steps and skip
        L-BFGS (used by the fast tests). Larger budgets are split between Adam
        and L-BFGS; see the module docstring.
    seed:
        NumPy and PyTorch seed.

    Returns
    -------
    model, metrics
        The trained module and a JSON-serializable metrics dictionary. When
        ``out_dir`` is set, a checkpoint, metrics JSON, and comparison figure
        are written there.
    """
    torch.set_num_threads(2)
    case_name = str(case).strip().lower()
    if case_name not in {"solovev", "nonlinear", "parametric"}:
        raise ValueError("case must be 'solovev', 'nonlinear', or 'parametric'")
    if w_b is None:
        w_b = 200.0 if case_name == "nonlinear" else 100.0
    t0 = time.perf_counter()
    common = dict(
        steps=int(steps),
        seed=int(seed),
        n_interior=n_interior,
        n_boundary=n_boundary,
        lbfgs_steps=lbfgs_steps,
        hidden=int(hidden),
        n_layers=int(n_layers),
        lr=float(lr),
        w_b=float(w_b),
        verbose=bool(verbose),
        device=device,
    )
    if case_name == "nonlinear":
        model, metrics, pack = _train_nonlinear(**common)
    else:
        model, metrics, pack = _train_solovev_family(case=case_name, **common)
    metrics["train_time_s"] = float(time.perf_counter() - t0)
    model.eval()
    if out_dir is not None:
        _write_outputs(
            out_dir,
            case_name,
            model,
            metrics,
            pack["grid"],
            pack["pred"],
            pack["ref"],
            pack["mask"],
            pack["Rb"],
            pack["Zb"],
            title=pack.get("title"),
        )
    return model, metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a Grad-Shafranov PINN")
    parser.add_argument("--case", required=True, choices=["solovev", "nonlinear", "parametric"])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="outputs/pinn")
    args = parser.parse_args(argv)
    _model, metrics = train_pinn(
        args.case,
        args.steps,
        seed=args.seed,
        out_dir=args.out,
        verbose=True,
    )
    print(json.dumps(_jsonify(metrics), indent=2), flush=True)


if __name__ == "__main__":
    main()
