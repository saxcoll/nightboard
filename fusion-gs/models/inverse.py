"""Inverse equilibrium reconstruction from synthetic magnetic diagnostics.

``InverseNet`` maps a vector of pickup-probe, flux-loop and Rogowski signals
(and, for free-boundary data, the known PF coil currents) to the poloidal
flux ``psi(R, Z)`` plus a few scalar equilibrium quantities. It is an EFIT-like
regression inside the fixed-boundary and FreeGS families used in this project:
the network does not remove the fundamental non-uniqueness of a general
external-magnetics inverse problem, because the training distribution is a
low-dimensional profile family.

Conventions match ``gs.solver``: arrays are ``(nr, nz)`` with ``indexing="ij"``,
SI units, ``psi = 0`` on the boundary and ``psi_axis > 0``.

CLI::

    python -m models.inverse --data data/fixed_65.npz --noise 0.01 --epochs 150 --out outputs/inverse
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from eval.metrics import (
    find_magnetic_axis,
    flux_contour,
    gs_residual,
    lcfs_shape_error,
    q95,
    relative_l2,
)
from gs.solver import MU0, Grid
from models.diagnostics import SensorSet, channel_rms

torch.set_num_threads(1)

_AUX_FIXED = ("R0", "a", "kappa", "delta")
_AUX_FREEGS = ("R_geom", "a_minor", "kappa", "delta")
_NOISE_SWEEP = (0.0, 0.01, 0.03, 0.05)


def delta_star_torch(psi: torch.Tensor, R: torch.Tensor, dR: float, dZ: float) -> torch.Tensor:
    """Conservative ``Delta*`` matching ``gs.solver.delta_star_fd``.

    ``psi`` has shape ``(N, nr, nz)``. The result has shape ``(N, nr-2, nz-2)``
    and covers the interior nodes ``[1:-1, 1:-1]``.
    """
    r_half = 0.5 * (R[:-1] + R[1:])
    flux = (psi[:, 1:, :] - psi[:, :-1, :]) / (dR * r_half.view(1, -1, 1))
    radial = R[1:-1].view(1, -1, 1) * (flux[:, 1:, :] - flux[:, :-1, :]) / dR
    d2z = (psi[:, :, 2:] - 2.0 * psi[:, :, 1:-1] + psi[:, :, :-2]) / (dZ * dZ)
    return radial[:, :, 1:-1] + d2z[:, 1:-1, :]


class _ShapeDecoder(nn.Module):
    """Latent vector to the dimensionless shape ``psi / psi_axis``.

    Same skeleton as the params->psi surrogate: transposed convolutions from an
    8×8 latent volume, then a full-resolution mix with normalized ``(R, Z)`` and
    a broadcast of the latent code. The last convolution is zero-initialized so
    the shape starts at 0.
    """

    def __init__(self, latent: int, nr: int, nz: int, channels: int = 32, base: int = 8):
        super().__init__()
        self.nr = int(nr)
        self.nz = int(nz)
        self.channels = int(channels)
        self.base = 8 if int(base) >= 8 else 4
        self.fc = nn.Linear(latent, self.channels * self.base * self.base)
        self.up1 = nn.ConvTranspose2d(self.channels, 32, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.lat_proj = nn.Linear(latent, 16)
        self.ref1 = nn.Conv2d(8 + 2 + 16, 16, kernel_size=3, padding=1)
        self.ref2 = nn.Conv2d(16, 1, kernel_size=1)
        self.register_buffer("Rn", torch.linspace(-1.0, 1.0, self.nr))
        self.register_buffer("Zn", torch.linspace(-1.0, 1.0, self.nz))
        nn.init.xavier_uniform_(self.ref2.weight, gain=0.1)
        nn.init.zeros_(self.ref2.bias)
        # Exact zero keeps the untrained network at psi = 0 (tests and a stable start).
        nn.init.zeros_(self.ref2.weight)

    def set_grid(self, R, Z) -> None:
        R_t = torch.as_tensor(R, dtype=torch.float32, device=self.Rn.device).flatten()
        Z_t = torch.as_tensor(Z, dtype=torch.float32, device=self.Zn.device).flatten()
        span_r = (R_t[-1] - R_t[0]).clamp_min(1e-8)
        span_z = (Z_t[-1] - Z_t[0]).clamp_min(1e-8)
        self.Rn.copy_(2.0 * (R_t - R_t[0]) / span_r - 1.0)
        self.Zn.copy_(2.0 * (Z_t - Z_t[0]) / span_z - 1.0)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b = latent.shape[0]
        z = F.gelu(self.fc(latent)).view(b, self.channels, self.base, self.base)
        z = F.gelu(self.up1(z))
        z = F.gelu(self.up2(z))
        z = F.gelu(self.up3(z))
        z = F.interpolate(z, size=(self.nr, self.nz), mode="bilinear", align_corners=True)
        rr = self.Rn.view(1, 1, self.nr, 1).expand(b, 1, self.nr, self.nz)
        zz = self.Zn.view(1, 1, 1, self.nz).expand(b, 1, self.nr, self.nz)
        code = self.lat_proj(latent).view(b, 16, 1, 1).expand(b, 16, self.nr, self.nz)
        x = F.gelu(self.ref1(torch.cat([z, rr, zz, code], dim=1)))
        return self.ref2(x).squeeze(1)


class InverseNet(nn.Module):
    """MLP encoder on normalized diagnostics, then either a shape decoder or a surrogate.

    ``arch="direct"`` predicts the dimensionless shape ``psi / psi_axis`` and
    ``log(psi_axis)``. Physical flux is ``shape * exp(log psi_axis)``.

    ``arch="hybrid"`` predicts the equilibrium parameters and maps them through
    a frozen params->psi surrogate. ``B0`` is one of those parameters but does
    not affect the Grad-Shafranov flux; the head still emits it so the
    surrogate sees a complete vector.

    Inputs are physical signals. Outputs are physical ``psi`` ``(N, nr, nz)``
    and auxiliary scalars ``(N, n_aux)``.
    """

    def __init__(
        self,
        n_sensors: int,
        nr: int,
        nz: int,
        n_aux: int,
        n_coils: int = 0,
        latent: int = 160,
        hidden: int = 320,
        channels: int = 32,
        base: int = 8,
        arch: str = "direct",
        n_eq: int = 0,
    ):
        super().__init__()
        self.n_sensors = int(n_sensors)
        self.nr = int(nr)
        self.nz = int(nz)
        self.n_aux = int(n_aux)
        self.n_coils = int(n_coils)
        self.latent = int(latent)
        self.hidden = int(hidden)
        self.channels = int(channels)
        self.base = int(base)
        self.arch = str(arch)
        self.n_eq = int(n_eq)
        self.eq_names: list[str] = []
        self.surrogate_frozen = False
        self.skip_field = False
        self._log_s: torch.Tensor | None = None
        n_in = self.n_sensors + max(self.n_sensors - 1, 0) + 1 + self.n_coils
        self.encoder = nn.Sequential(
            nn.Linear(n_in, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.latent),
            nn.GELU(),
        )
        if self.arch == "direct":
            self.decoder: _ShapeDecoder | None = _ShapeDecoder(
                self.latent, self.nr, self.nz, channels=self.channels, base=self.base
            )
            self.log_axis: nn.Linear | None = nn.Linear(self.latent, 1)
            nn.init.zeros_(self.log_axis.weight)
            nn.init.zeros_(self.log_axis.bias)
        else:
            self.decoder = None
            self.log_axis = None
        self.aux_head = nn.Sequential(
            nn.Linear(self.latent, self.hidden // 2),
            nn.GELU(),
            nn.Linear(self.hidden // 2, self.n_aux),
        )
        nn.init.zeros_(self.aux_head[-1].weight)
        nn.init.zeros_(self.aux_head[-1].bias)
        if self.n_eq > 0:
            self.eq_head: nn.Sequential | None = nn.Sequential(
                nn.Linear(self.latent, self.hidden),
                nn.GELU(),
                nn.Linear(self.hidden, self.n_eq),
            )
            nn.init.zeros_(self.eq_head[-1].weight)
            nn.init.zeros_(self.eq_head[-1].bias)
        else:
            self.eq_head = None
        self.surrogate = None
        self.register_buffer("sensor_mean", torch.zeros(self.n_sensors))
        self.register_buffer("sensor_std", torch.ones(self.n_sensors))
        self.register_buffer("sensor_rms", torch.ones(self.n_sensors))
        self.register_buffer("coil_mean", torch.zeros(self.n_coils))
        self.register_buffer("coil_std", torch.ones(self.n_coils))
        self.register_buffer("psi_scale", torch.ones(()))
        self.register_buffer("aux_mean", torch.zeros(self.n_aux))
        self.register_buffer("aux_std", torch.ones(self.n_aux))
        self.register_buffer("eq_mean", torch.zeros(self.n_eq))
        self.register_buffer("eq_std", torch.ones(self.n_eq))
        # Ip-normalized magnetics: B and psi divided by the Rogowski channel.
        # Amplitude is removed so the shape of the vessel signal is visible.
        n_extra = max(self.n_sensors - 1, 0)
        self.register_buffer("extra_mean", torch.zeros(n_extra))
        self.register_buffer("extra_std", torch.ones(n_extra))
        self.register_buffer("logrog_mean", torch.zeros(()))
        self.register_buffer("logrog_std", torch.ones(()))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.surrogate is not None and self.surrogate_frozen:
            self.surrogate.eval()
        return self

    def attach_surrogate(self, surrogate, frozen: bool = True) -> None:
        self.surrogate = surrogate
        self.surrogate_frozen = bool(frozen)
        if frozen:
            for param in self.surrogate.parameters():
                param.requires_grad_(False)
            self.surrogate.eval()

    def set_normalization(
        self,
        sensor_mean: np.ndarray,
        sensor_std: np.ndarray,
        sensor_rms: np.ndarray,
        psi_scale: float,
        aux_mean: np.ndarray,
        aux_std: np.ndarray,
        coil_mean: np.ndarray | None = None,
        coil_std: np.ndarray | None = None,
    ) -> None:
        self.sensor_mean.copy_(torch.as_tensor(sensor_mean, dtype=torch.float32))
        self.sensor_std.copy_(torch.as_tensor(sensor_std, dtype=torch.float32))
        self.sensor_rms.copy_(torch.as_tensor(sensor_rms, dtype=torch.float32))
        self.psi_scale.copy_(torch.tensor(float(psi_scale), dtype=torch.float32))
        self.aux_mean.copy_(torch.as_tensor(aux_mean, dtype=torch.float32))
        self.aux_std.copy_(torch.as_tensor(aux_std, dtype=torch.float32))
        if self.n_coils:
            if coil_mean is None or coil_std is None:
                raise ValueError("coil normalization is required when n_coils > 0")
            self.coil_mean.copy_(torch.as_tensor(coil_mean, dtype=torch.float32))
            self.coil_std.copy_(torch.as_tensor(coil_std, dtype=torch.float32))

    def set_equilibrium_norm(self, mean: np.ndarray, std: np.ndarray, names: list[str]) -> None:
        if self.n_eq == 0:
            return
        self.eq_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.eq_std.copy_(torch.as_tensor(std, dtype=torch.float32))
        self.eq_names = list(names)

    def set_ip_feature_norm(self, extra_mean, extra_std, logrog_mean: float, logrog_std: float) -> None:
        self.extra_mean.copy_(torch.as_tensor(extra_mean, dtype=torch.float32))
        self.extra_std.copy_(torch.as_tensor(extra_std, dtype=torch.float32))
        self.logrog_mean.copy_(torch.tensor(float(logrog_mean), dtype=torch.float32))
        self.logrog_std.copy_(torch.tensor(float(logrog_std), dtype=torch.float32))

    def _latent(self, signals: torch.Tensor, coil_currents: torch.Tensor | None) -> torch.Tensor:
        x = (signals - self.sensor_mean) / self.sensor_std
        parts = [x]
        if signals.shape[-1] == self.n_sensors and self.n_sensors > 1:
            rog = signals[:, -1:].abs().clamp_min(1.0)
            extra = (signals[:, :-1] / rog - self.extra_mean) / self.extra_std.clamp_min(1e-8)
            logrog = (torch.log(rog.squeeze(-1)) - self.logrog_mean) / self.logrog_std.clamp_min(1e-8)
            parts.extend([extra, logrog.unsqueeze(-1)])
        if self.n_coils:
            if coil_currents is None:
                raise ValueError("InverseNet was trained with coil currents")
            coils = (coil_currents - self.coil_mean) / self.coil_std
            parts.append(coils)
        return self.encoder(torch.cat(parts, dim=-1))

    def equilibrium_params(
        self, signals: torch.Tensor, coil_currents: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Raw equilibrium parameters ``(N, n_eq)``, or None if this net has no head."""
        if self.eq_head is None:
            return None
        latent = self._latent(signals, coil_currents)
        return self.eq_head(latent) * self.eq_std + self.eq_mean

    def forward(
        self, signals: torch.Tensor, coil_currents: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self._latent(signals, coil_currents)
        self._log_s = None
        if self.arch == "hybrid" and self.surrogate is not None and not self.skip_field:
            if self.eq_head is None:
                raise RuntimeError("hybrid InverseNet requires an equilibrium-parameter head")
            params = self.eq_head(latent) * self.eq_std + self.eq_mean
            psi = self.surrogate(params)
        elif self.decoder is None or self.log_axis is None or self.skip_field:
            psi = signals.new_zeros(signals.shape[0], self.nr, self.nz)
        else:
            shape = self.decoder(latent)
            log_s = self.log_axis(latent).squeeze(-1)
            self._log_s = log_s
            scale = torch.exp(log_s.clamp(-6.0, 2.0))
            psi = shape * scale.unsqueeze(-1).unsqueeze(-1)
        aux = self.aux_head(latent) * self.aux_std + self.aux_mean
        return psi, aux


def _load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def _param_names(data: dict) -> list[str]:
    return [str(x) for x in np.asarray(data["param_names"]).tolist()]


def _aux_arrays(data: dict, index: np.ndarray) -> tuple[np.ndarray, list[str]]:
    names = _param_names(data)
    columns = [
        np.asarray(data["R_axis"], dtype=np.float64)[index],
        np.asarray(data["Z_axis"], dtype=np.float64)[index],
        np.asarray(data["psi_axis"], dtype=np.float64)[index],
    ]
    aux_names = ["R_axis", "Z_axis", "psi_axis"]
    if all(key in names for key in _AUX_FIXED):
        shape_keys = _AUX_FIXED
    elif all(key in names for key in _AUX_FREEGS):
        shape_keys = _AUX_FREEGS
    else:
        shape_keys = ()
    params = np.asarray(data["params"], dtype=np.float64)
    for key in shape_keys:
        columns.append(params[index, names.index(key)])
        aux_names.append(key)
    return np.column_stack(columns).astype(np.float32), aux_names


def _eq_matrix(data: dict, index: np.ndarray) -> tuple[np.ndarray | None, list[str]]:
    """Equilibrium parameters in surrogate column order, when the file has them."""
    from models.surrogate import PARAM_NAMES

    names = _param_names(data)
    if any(name not in names for name in PARAM_NAMES):
        return None, []
    params = np.asarray(data["params"], dtype=np.float64)
    columns = [params[np.asarray(index), names.index(name)] for name in PARAM_NAMES]
    return np.column_stack(columns).astype(np.float32), list(PARAM_NAMES)


def _surrogate_path(explicit) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    for candidate in (
        Path("outputs/surrogate/mlpcnn_pw0p1/checkpoint.pt"),
        Path("/workspace/fusion-gs/outputs/surrogate/mlpcnn_pw0p1/checkpoint.pt"),
    ):
        if candidate.is_file():
            return candidate
    return None


def _load_hybrid_surrogate(grid: Grid, data: dict, explicit):
    """Frozen surrogate when the grid and parameter vector match, else None."""
    path = _surrogate_path(explicit)
    if path is None or _geometry_for(data) != "dshape":
        return None
    from models.surrogate import load_surrogate

    surrogate = load_surrogate(path)
    r_s = surrogate.R.detach().cpu().numpy()
    z_s = surrogate.Z.detach().cpu().numpy()
    if r_s.shape != np.asarray(grid.R).shape or z_s.shape != np.asarray(grid.Z).shape:
        return None
    if not (np.allclose(r_s, grid.R) and np.allclose(z_s, grid.Z)):
        return None
    return surrogate


def _r2_columns(pred: np.ndarray, true: np.ndarray, names: list[str]) -> dict[str, float]:
    scores = {}
    for j, name in enumerate(names):
        pp = np.asarray(pred[:, j], dtype=float)
        tt = np.asarray(true[:, j], dtype=float)
        keep = np.isfinite(pp) & np.isfinite(tt)
        if int(keep.sum()) < 3:
            scores[name] = float("nan")
            continue
        tt = tt[keep]
        pp = pp[keep]
        ss_tot = float(np.sum((tt - tt.mean()) ** 2))
        ss_res = float(np.sum((tt - pp) ** 2))
        scores[name] = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    return scores


def _std_floor(values: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    std = np.asarray(values, dtype=np.float64)
    return np.where(std < floor, 1.0, std).astype(np.float32)


def _interior(mask: np.ndarray) -> np.ndarray:
    """Nodes whose four orthogonal neighbours lie in ``mask``. Shape (N, nr-2, nz-2)."""
    return (
        mask[:, 1:-1, 1:-1]
        & mask[:, :-2, 1:-1]
        & mask[:, 2:, 1:-1]
        & mask[:, 1:-1, :-2]
        & mask[:, 1:-1, 2:]
    )


def _batch_loss(
    model: InverseNet,
    signals: torch.Tensor,
    coils: torch.Tensor | None,
    psi: torch.Tensor,
    aux: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor,
    interior: torch.Tensor,
    R: torch.Tensor,
    dR: float,
    dZ: float,
    aux_weight: float,
    physics_weight: float,
    outside_weight: float,
    eq: torch.Tensor | None = None,
    eq_weight: float = 0.0,
    scale_weight: float = 0.1,
    psi_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-sample flux loss in units of ``psi / psi_axis``, plus scalar heads.

    ``aux[:, 2]`` is ``psi_axis``. Dividing by that value, rather than one global
    RMS, keeps low-current and high-current plasmas on the same scale.
    """
    psi_pred, aux_pred = model(signals, coils)
    axis = aux[:, 2].clamp_min(1e-8).unsqueeze(-1).unsqueeze(-1)
    err = ((psi_pred - psi) / axis).square()
    w = mask.to(dtype=err.dtype)
    w = w + (1.0 - w) * float(outside_weight)
    num = (err * w).flatten(1).sum(dim=1)
    den = w.flatten(1).sum(dim=1).clamp_min(1.0)
    loss_psi = (num / den).mean()
    aux_scale = model.aux_std.clamp(min=1e-8)
    aux_err = (aux_pred - aux) / aux_scale
    loss_aux = aux_err.square().mean()
    loss_scale = psi_pred.new_zeros(())
    if scale_weight > 0.0 and model._log_s is not None:
        target_log = torch.log(aux[:, 2].clamp_min(1e-8))
        loss_scale = F.mse_loss(model._log_s, target_log)
    loss_eq = psi_pred.new_zeros(())
    if eq is not None and eq_weight > 0.0 and model.eq_head is not None:
        eq_hat = model.equilibrium_params(signals, coils)
        eq_n = (eq_hat - model.eq_mean) / model.eq_std.clamp_min(1e-8)
        eq_t = (eq - model.eq_mean) / model.eq_std.clamp_min(1e-8)
        loss_eq = (eq_n - eq_t).square().mean()
    loss_phys = psi_pred.new_zeros(())
    if physics_weight > 0.0 and psi_weight > 0.0 and bool(interior.any()) and not model.skip_field:
        dstar = delta_star_torch(psi_pred, R, dR, dZ)
        rhs = -MU0 * R[1:-1].view(1, -1, 1) * J[:, 1:-1, 1:-1]
        weight = interior.to(dstar.dtype)
        resid = (dstar - rhs) * weight
        denom = (rhs.square() * weight).sum().clamp(min=1e-30)
        loss_phys = resid.square().sum() / denom
    total = (
        float(psi_weight) * loss_psi
        + float(scale_weight) * loss_scale
        + aux_weight * loss_aux
        + float(eq_weight) * loss_eq
        + physics_weight * loss_phys
    )
    parts = {
        "psi": float(loss_psi.detach()),
        "scale": float(loss_scale.detach()),
        "aux": float(loss_aux.detach()),
        "eq": float(loss_eq.detach()),
        "phys": float(loss_phys.detach()),
        "total": float(total.detach()),
    }
    return total, parts


def _run_epoch_eval(
    model: InverseNet,
    signals: torch.Tensor,
    coils: torch.Tensor | None,
    psi: torch.Tensor,
    aux: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor,
    interior: torch.Tensor,
    R: torch.Tensor,
    dR: float,
    dZ: float,
    aux_weight: float,
    physics_weight: float,
    outside_weight: float,
    batch_size: int,
    noise: torch.Tensor | None = None,
    eq: torch.Tensor | None = None,
    eq_weight: float = 0.0,
    scale_weight: float = 0.1,
    psi_weight: float = 1.0,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    n = signals.shape[0]
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            sig = signals[start:stop]
            if noise is not None:
                sig = sig + noise[start:stop]
            coil_b = None if coils is None else coils[start:stop]
            eq_b = None if eq is None else eq[start:stop]
            loss, _ = _batch_loss(
                model,
                sig,
                coil_b,
                psi[start:stop],
                aux[start:stop],
                J[start:stop],
                mask[start:stop],
                interior[start:stop],
                R,
                dR,
                dZ,
                aux_weight,
                physics_weight,
                outside_weight,
                eq=eq_b,
                eq_weight=eq_weight,
                scale_weight=scale_weight,
                psi_weight=psi_weight,
            )
            width = stop - start
            total += float(loss) * width
            count += width
    return total / max(count, 1)


def _indices(split: np.ndarray, value: int) -> np.ndarray:
    return np.flatnonzero(np.asarray(split) == value)


def _geometry_for(data: dict) -> str:
    if "coil_currents" in data:
        return "freegs"
    return "dshape"


def _coil_matrix(data: dict, index: np.ndarray) -> np.ndarray | None:
    if "coil_currents" not in data:
        return None
    return np.asarray(data["coil_currents"], dtype=np.float32)[index]


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(key): _jsonify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _mean_finite(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def make_F(params_row: np.ndarray, names: list[str], grid: Grid, psi: np.ndarray, mask: np.ndarray, psi_axis: float):
    """``F(psi_n) = R B_phi`` from the fixed-boundary profile in ``gs.solver``.

    ``lam`` is fixed by ``Ip = ∫ J dA`` with the same ``g(psi_n)`` and
    ``J_phi`` formula as ``solve_fixed_boundary``. ``F(1) = R0 B0``. Returns
    None when the row is not a fixed-boundary parameter vector.
    """
    need = ("R0", "Ip", "beta0", "alpha", "gamma", "B0")
    if any(key not in names for key in need):
        return None
    idx = {name: i for i, name in enumerate(names)}
    R0 = float(params_row[idx["R0"]])
    Ip = float(params_row[idx["Ip"]])
    beta0 = float(params_row[idx["beta0"]])
    alpha = float(params_row[idx["alpha"]])
    gamma = float(params_row[idx["gamma"]])
    B0 = float(params_row[idx["B0"]])
    psi_axis_f = float(psi_axis)
    if not np.isfinite(psi_axis_f) or psi_axis_f == 0.0:
        return None
    pn = np.clip((np.asarray(psi, dtype=float) - psi_axis_f) / (0.0 - psi_axis_f), 0.0, 1.0)
    g = (1.0 - pn ** alpha) ** gamma
    R = grid.RR
    bare = np.zeros_like(pn)
    bare[mask] = (beta0 * R[mask] / R0 + (1.0 - beta0) * R0 / R[mask]) * g[mask]
    area = float(np.sum(bare) * grid.dR * grid.dZ)
    if area == 0.0:
        return None
    lam = Ip / area
    s = np.linspace(0.0, 1.0, 4001)
    gs = (1.0 - s ** alpha) ** gamma
    ff = MU0 * lam * (1.0 - beta0) * R0 * gs
    ds = float(s[1] - s[0])
    cum = np.empty_like(s)
    cum[0] = 0.0
    cum[1:] = np.cumsum(0.5 * (ff[1:] + ff[:-1]) * ds)
    boundary_F2 = (R0 * B0) ** 2

    def F(psin):
        psin_a = np.asarray(psin, dtype=float)
        integ = np.interp(psin_a, s, cum) - cum[-1]
        f2 = boundary_F2 + 2.0 * (0.0 - psi_axis_f) * integ
        return np.sqrt(np.maximum(f2, 0.0))

    return F


def _predict_numpy(
    model: InverseNet,
    signals: np.ndarray,
    coil_currents: np.ndarray | None,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    outputs = []
    sig = np.asarray(signals, dtype=np.float32)
    coils = None if coil_currents is None else np.asarray(coil_currents, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, sig.shape[0], batch_size):
            stop = min(start + batch_size, sig.shape[0])
            sig_t = torch.as_tensor(sig[start:stop])
            coil_t = None if coils is None else torch.as_tensor(coils[start:stop])
            psi, _aux = model(sig_t, coil_t)
            outputs.append(psi.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _summary_metrics(raw: dict) -> dict:
    return {key: value for key, value in raw.items() if not key.startswith("_")}


def save_reconstruction_figure(
    path: Path,
    grid: Grid,
    sensors: SensorSet,
    psi_true: np.ndarray,
    psi_pred: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Sensor layout on one equilibrium, plus true / predicted / error maps."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    R = np.asarray(grid.R, dtype=float)
    Z = np.asarray(grid.Z, dtype=float)
    extent = [float(R[0]), float(R[-1]), float(Z[0]), float(Z[-1])]
    err = psi_pred - psi_true
    vmax = float(np.percentile(np.abs(psi_true), 99.5))
    vmax = max(vmax, 1e-6)
    emax = float(np.percentile(np.abs(err), 99.5))
    emax = max(emax, 1e-6)
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.2), constrained_layout=True)
    panels = [
        (psi_true, "inferno", 0.0, vmax, "true psi with sensors"),
        (psi_true, "inferno", 0.0, vmax, "true psi"),
        (psi_pred, "inferno", 0.0, vmax, "predicted psi"),
        (err, "coolwarm", -emax, emax, "pred − true"),
    ]
    for ax, (field, cmap, vmin, vhi, title) in zip(axes, panels):
        image = ax.imshow(
            field.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vhi,
            interpolation="nearest",
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
    ax0 = axes[0]
    ax0.plot(sensors.vessel_R, sensors.vessel_Z, color="cyan", lw=0.8, label="vessel")
    ax0.plot(sensors.vessel_R[0], sensors.vessel_Z[0], color="cyan", lw=0.8)
    # Close the vessel polyline visually.
    ax0.plot(
        [sensors.vessel_R[-1], sensors.vessel_R[0]],
        [sensors.vessel_Z[-1], sensors.vessel_Z[0]],
        color="cyan",
        lw=0.8,
    )
    ax0.scatter(sensors.probe_R, sensors.probe_Z, s=10, c="white", edgecolors="k", linewidths=0.3, label="B probes", zorder=3)
    scale = 0.06 * max(float(R[-1] - R[0]), float(Z[-1] - Z[0]))
    ax0.quiver(
        sensors.probe_R,
        sensors.probe_Z,
        sensors.tangent[:, 0],
        sensors.tangent[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0 / scale,
        width=0.004,
        color="white",
        zorder=2,
    )
    ax0.scatter(sensors.flux_R, sensors.flux_Z, s=16, marker="s", c="lime", edgecolors="k", linewidths=0.3, label="flux loops", zorder=3)
    ax0.legend(loc="upper right", fontsize=7, framealpha=0.85)
    # Keep the plasma outline visible on the true map.
    axes[1].contour(
        R,
        Z,
        np.where(mask, psi_true, np.nan).T,
        levels=[1e-3 * max(float(np.max(psi_true)), 1e-6)],
        colors="cyan",
        linewidths=0.6,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _build_signals(sensors: SensorSet, data: dict, grid: Grid) -> np.ndarray:
    J = np.asarray(data["J"], dtype=np.float64)
    coils = np.asarray(data["coil_currents"], dtype=np.float64) if "coil_currents" in data else None
    if coils is not None:
        names = data.get("coil_names")
        if names is None:
            sensors.coil_response_matrix(sensors.coil_names or _coil_names_from_freegs())
        else:
            sensors.coil_response_matrix(tuple(str(x) for x in np.asarray(names).tolist()))
    return sensors.measure(J, grid, coil_currents=coils).astype(np.float64)


def _coil_names_from_freegs() -> tuple[str, ...]:
    import freegs.machine as fm

    return tuple(label for label, _ in fm.TestTokamak().coils)


def train_inverse(
    dataset_path,
    noise: float = 0.01,
    epochs: int = 150,
    seed: int = 0,
    out_dir=None,
    batch_size: int = 64,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    physics_weight: float = 0.05,
    aux_weight: float = 0.05,
    outside_weight: float = 0.05,
    evaluate: bool = True,
    ood_path=None,
    num_threads: int = 1,
    verbose: bool = True,
    n_probes: int = 40,
    n_flux: int = 20,
    latent: int = 128,
    hidden: int = 256,
    channels: int = 32,
    base: int = 8,
    arch: str = "auto",
    surrogate_path=None,
    eq_weight: float = 0.35,
    scale_weight: float = 0.1,
):
    """Train on the dataset's train split and optionally evaluate test / OOD.

    Noise is Gaussian with standard deviation ``noise`` times each sensor's RMS
    over the whole file. It is resampled every training batch. Validation uses
    one fixed realization. Returns ``(model, metrics)``.
    """
    threads = max(1, min(int(num_threads), 2))
    torch.set_num_threads(threads)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    data = _load_npz(dataset_path)
    grid = Grid(np.asarray(data["R"], dtype=float), np.asarray(data["Z"], dtype=float))
    geometry = _geometry_for(data)
    if geometry == "freegs":
        # Plasmas in the FreeGS file cross the TestTokamak wall. Sit the vessel
        # one cell outside the union of every mask so the Rogowski still sees
        # the whole plasma and the PF coils stay outside the loop.
        union = np.asarray(data["mask"], dtype=bool).any(axis=0)
        sensors = SensorSet.around_plasmas(
            grid, union, n_probes=n_probes, n_flux=n_flux, include_normal=True, geometry="freegs"
        )
    else:
        sensors = SensorSet.for_grid(
            grid, n_probes=n_probes, n_flux=n_flux, include_normal=True, geometry=geometry
        )
    if verbose:
        print(
            f"sensors: {sensors.n_sensors} channels "
            f"({sensors.n_probes} Bt + {sensors.n_probes if sensors.include_normal else 0} Bn "
            f"+ {sensors.n_flux} flux + 1 Rogowski), geometry={geometry}",
            flush=True,
        )
    t_sig = time.perf_counter()
    signals = _build_signals(sensors, data, grid)
    if verbose:
        print(f"response matrix and signals in {time.perf_counter() - t_sig:.2f}s", flush=True)
    J_all = np.asarray(data["J"], dtype=np.float64)
    inside = sensors.inside_mask(grid)
    enclosed = (np.abs(J_all) * inside).sum(axis=(1, 2)) / np.maximum(np.abs(J_all).sum(axis=(1, 2)), 1e-30)
    if float(np.min(enclosed)) < 0.95:
        raise RuntimeError(
            f"vessel encloses only {float(np.min(enclosed)):.3f} of |J| on the worst sample"
        )
    split = np.asarray(data["split"])
    train_idx = _indices(split, 0)
    val_idx = _indices(split, 1)
    test_idx = _indices(split, 2)
    if train_idx.size == 0:
        raise RuntimeError("dataset has no training split (split == 0)")
    aux_all, aux_names = _aux_arrays(data, np.arange(signals.shape[0]))
    rms = channel_rms(signals).astype(np.float32)
    noise_std = (float(noise) * rms).astype(np.float32)
    sensor_mean = signals[train_idx].mean(axis=0).astype(np.float32)
    sensor_std = _std_floor(signals[train_idx].std(axis=0))
    psi_train = np.asarray(data["psi"], dtype=np.float64)[train_idx]
    mask_train = np.asarray(data["mask"])[train_idx]
    psi_scale = float(np.sqrt(np.mean(psi_train[mask_train] ** 2)))
    if not np.isfinite(psi_scale) or psi_scale <= 0.0:
        psi_scale = 1.0
    aux_mean = aux_all[train_idx].mean(axis=0).astype(np.float32)
    aux_std = _std_floor(aux_all[train_idx].std(axis=0), floor=1e-6)
    coils_all = _coil_matrix(data, np.arange(signals.shape[0]))
    n_coils = 0 if coils_all is None else int(coils_all.shape[1])
    coil_mean = coil_std = None
    if n_coils:
        coil_mean = coils_all[train_idx].mean(axis=0).astype(np.float32)
        coil_std = _std_floor(coils_all[train_idx].std(axis=0))
    eq_all, eq_names = _eq_matrix(data, np.arange(signals.shape[0]))
    resolved = str(arch)
    surrogate = None
    # Hybrid (sensors -> params -> frozen surrogate) ties the direct model on
    # flux error and does not recover a, kappa, or the profile exponents, so the
    # default stays the direct shape decoder.
    if resolved == "auto":
        resolved = "direct"
    if resolved == "hybrid":
        surrogate = _load_hybrid_surrogate(grid, data, surrogate_path)
        if surrogate is None or eq_all is None:
            if verbose:
                print("hybrid surrogate unavailable for this grid; using direct", flush=True)
            resolved = "direct"
            surrogate = None
    n_eq = 0 if eq_all is None else int(eq_all.shape[1])
    model = InverseNet(
        n_sensors=sensors.n_sensors,
        nr=grid.nr,
        nz=grid.nz,
        n_aux=len(aux_names),
        n_coils=n_coils,
        latent=latent,
        hidden=hidden,
        channels=channels,
        base=base,
        arch=resolved,
        n_eq=n_eq,
    )
    model.set_normalization(
        sensor_mean, sensor_std, rms, psi_scale, aux_mean, aux_std, coil_mean, coil_std
    )
    rog = np.maximum(np.abs(signals[:, -1:]), 1.0)
    extra = signals[:, :-1] / rog
    extra_mean = extra[train_idx].mean(axis=0).astype(np.float32)
    extra_std = _std_floor(extra[train_idx].std(axis=0))
    logrog = np.log(rog[:, 0])
    logrog_mean = float(logrog[train_idx].mean())
    logrog_std = float(logrog[train_idx].std())
    if logrog_std < 1e-8:
        logrog_std = 1.0
    model.set_ip_feature_norm(extra_mean, extra_std, logrog_mean, logrog_std)
    if model.decoder is not None:
        model.decoder.set_grid(grid.R, grid.Z)
    if n_eq and eq_all is not None:
        model.set_equilibrium_norm(
            eq_all[train_idx].mean(axis=0),
            _std_floor(eq_all[train_idx].std(axis=0)),
            eq_names,
        )
    if surrogate is not None:
        model.attach_surrogate(surrogate, frozen=True)
    if verbose:
        print(f"arch={model.arch} n_eq={model.n_eq}", flush=True)
    R_t = torch.as_tensor(np.asarray(grid.R, dtype=np.float32))
    dR = float(grid.dR)
    dZ = float(grid.dZ)
    psi_t = torch.as_tensor(np.asarray(data["psi"], dtype=np.float32))
    J_t = torch.as_tensor(np.asarray(data["J"], dtype=np.float32))
    mask_t = torch.as_tensor(np.asarray(data["mask"], dtype=bool))
    interior_t = torch.as_tensor(_interior(np.asarray(data["mask"], dtype=bool)))
    sig_t = torch.as_tensor(signals.astype(np.float32))
    aux_t = torch.as_tensor(aux_all)
    coil_t = None if coils_all is None else torch.as_tensor(coils_all)
    eq_t = None if eq_all is None else torch.as_tensor(eq_all)
    noise_std_t = torch.as_tensor(noise_std)

    def _subset(index: np.ndarray):
        idx = torch.as_tensor(index, dtype=torch.long)
        coils = None if coil_t is None else coil_t[idx]
        return (
            sig_t[idx],
            coils,
            psi_t[idx],
            aux_t[idx],
            J_t[idx],
            mask_t[idx],
            interior_t[idx],
        )

    train_pack = _subset(train_idx)
    val_pack = _subset(val_idx) if val_idx.size else None
    val_noise = None
    if val_pack is not None and float(noise) > 0.0:
        generator = torch.Generator()
        generator.manual_seed(int(seed) + 17)
        val_noise = torch.randn(val_pack[0].shape, generator=generator) * noise_std_t

    trainable = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(epochs), 1))
    # Hybrid spends a short prefix on the parameter head alone, then turns on
    # the frozen surrogate so the flux loss can move the identifiable parameters.
    # Parameter regression is an MLP and is cheap. Give it a long prefix so the
    # minor radius and profile exponents are fit before the surrogate flux loss,
    # which otherwise spends its steps correcting a bad ``a``.
    phase1 = min(40, max(0, int(epochs) - 10)) if model.arch == "hybrid" else 0
    eq_train = None if eq_t is None else eq_t[torch.as_tensor(train_idx, dtype=torch.long)]
    eq_val = None if eq_t is None else eq_t[torch.as_tensor(val_idx, dtype=torch.long)]

    def _weights(epoch: int) -> tuple[float, float, float, float]:
        use_psi = epoch >= phase1
        psi_w = 1.0 if use_psi else 0.0
        scale_w = float(scale_weight) if use_psi and model.arch == "direct" else 0.0
        if model.n_eq == 0:
            eq_w = 0.0
        elif use_psi and model.arch == "hybrid":
            eq_w = float(eq_weight)
        elif use_psi:
            eq_w = float(eq_weight)
        else:
            eq_w = 1.0
        phys_w = float(physics_weight) if use_psi and model.arch == "direct" else 0.0
        return psi_w, scale_w, eq_w, phys_w

    def _score(pack, eq_rows, noise, epoch: int) -> float:
        psi_w, scale_w, eq_w, phys_w = _weights(epoch)
        model.skip_field = psi_w == 0.0
        return _run_epoch_eval(
            model,
            *pack,
            R_t,
            dR,
            dZ,
            aux_weight,
            phys_w,
            outside_weight,
            batch_size,
            noise=noise,
            eq=eq_rows,
            eq_weight=eq_w,
            scale_weight=scale_w,
            psi_weight=psi_w,
        )

    loss_initial = _score(train_pack, eq_train, None, epoch=phase1)
    history = {"train_loss": [], "val_loss": [], "lr": []}
    model.meta = {
        "aux_names": aux_names,
        "grid_R": np.asarray(grid.R, dtype=np.float64),
        "grid_Z": np.asarray(grid.Z, dtype=np.float64),
        "sensor": sensors.to_dict(),
        "noise": float(noise),
        "param_names": _param_names(data),
        "coil_names": list(sensors.coil_names),
        "arch": model.arch,
        "eq_names": list(model.eq_names),
    }
    best_val = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stall = 0
    epochs_ran = 0
    t0 = time.perf_counter()
    n_train = int(train_idx.size)
    order_gen = np.random.default_rng(int(seed))
    for epoch in range(int(epochs)):
        if epoch == phase1:
            best_val = float("inf")
            stall = 0
        psi_w, scale_w, eq_w, phys_w = _weights(epoch)
        model.skip_field = psi_w == 0.0
        model.train()
        order = order_gen.permutation(n_train)
        for start in range(0, n_train, int(batch_size)):
            batch = order[start : start + int(batch_size)]
            b = torch.as_tensor(batch, dtype=torch.long)
            sig_b = train_pack[0][b]
            if float(noise) > 0.0:
                sig_b = sig_b + torch.randn_like(sig_b) * noise_std_t
            coil_b = None if train_pack[1] is None else train_pack[1][b]
            eq_b = None if eq_train is None else eq_train[b]
            opt.zero_grad(set_to_none=True)
            loss, _parts = _batch_loss(
                model,
                sig_b,
                coil_b,
                train_pack[2][b],
                train_pack[3][b],
                train_pack[4][b],
                train_pack[5][b],
                train_pack[6][b],
                R_t,
                dR,
                dZ,
                aux_weight,
                phys_w,
                outside_weight,
                eq=eq_b,
                eq_weight=eq_w,
                scale_weight=scale_w,
                psi_weight=psi_w,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        sched.step()
        # Full-train evaluation dominates the step time on the 65x65 set. Use it
        # when the split is small (unit tests); otherwise score a fixed subset.
        if n_train <= 512:
            train_loss = _score(train_pack, eq_train, None, epoch)
        else:
            monitor = np.random.default_rng(int(seed) + 91).choice(n_train, size=512, replace=False)
            mon = torch.as_tensor(np.sort(monitor), dtype=torch.long)
            train_loss = _score(
                (
                    train_pack[0][mon],
                    None if train_pack[1] is None else train_pack[1][mon],
                    train_pack[2][mon],
                    train_pack[3][mon],
                    train_pack[4][mon],
                    train_pack[5][mon],
                    train_pack[6][mon],
                ),
                None if eq_train is None else eq_train[mon],
                None,
                epoch,
            )
        if val_pack is not None:
            val_loss = _score(val_pack, eq_val, val_noise, epoch)
        else:
            val_loss = train_loss
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        epochs_ran = epoch + 1
        improved = val_loss < best_val - 1e-5
        if improved:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stall = 0
            if out_dir is not None and epoch >= phase1:
                held = {key: value.detach().clone() for key, value in model.state_dict().items()}
                model.load_state_dict(best_state)
                model.skip_field = False
                save_checkpoint(model, Path(out_dir) / "inverse.pt")
                model.load_state_dict(held)
        else:
            stall += 1
        if verbose and (epoch < 3 or epoch % 5 == 0 or improved or stall == 1):
            print(
                f"epoch {epoch + 1:03d}  train {train_loss:.5f}  val {val_loss:.5f}  "
                f"best {best_val:.5f}",
                flush=True,
            )
        if val_pack is not None and epoch >= phase1 and stall >= int(patience):
            if verbose:
                print(f"early stop at epoch {epoch + 1}", flush=True)
            break
    model.skip_field = False
    model.load_state_dict(best_state)
    model.eval()
    train_seconds = time.perf_counter() - t0
    metrics: dict = {
        "dataset": str(dataset_path),
        "arch": model.arch,
        "geometry": geometry,
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "noise": float(noise),
        "epochs_ran": int(epochs_ran),
        "epochs_requested": int(epochs),
        "best_val_loss": None if not np.isfinite(best_val) else float(best_val),
        "loss_initial": float(loss_initial),
        "train_loss": [float(v) for v in history["train_loss"]],
        "val_loss": [float(v) for v in history["val_loss"]],
        "train_seconds": float(train_seconds),
        "enclosed_current_fraction_min": float(np.min(enclosed)),
        "aux_names": aux_names,
        "n_sensors": int(sensors.n_sensors),
        "n_probes": int(sensors.n_probes),
        "n_flux": int(sensors.n_flux),
        "include_normal": True,
        "n_coils": int(n_coils),
        "hyperparameters": {
            "latent": int(latent),
            "hidden": int(hidden),
            "arch": model.arch,
            "channels": int(channels),
            "decoder_base": int(base),
            "eq_weight": float(eq_weight),
            "scale_weight": float(scale_weight),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "patience": int(patience),
            "physics_weight": float(physics_weight),
            "aux_weight": float(aux_weight),
            "outside_weight": float(outside_weight),
            "num_threads": int(threads),
        },
        "sensors": {
            "n_probes": int(sensors.n_probes),
            "n_normal": int(sensors.n_probes if sensors.include_normal else 0),
            "n_flux": int(sensors.n_flux),
            "n_rogowski": 1,
            "n_sensors": int(sensors.n_sensors),
            "geometry": geometry,
            "probe_R_min": float(np.min(sensors.probe_R)),
            "probe_R_max": float(np.max(sensors.probe_R)),
            "probe_Z_min": float(np.min(sensors.probe_Z)),
            "probe_Z_max": float(np.max(sensors.probe_Z)),
        },
    }
    model.meta = {
        "aux_names": aux_names,
        "grid_R": np.asarray(grid.R, dtype=np.float64),
        "grid_Z": np.asarray(grid.Z, dtype=np.float64),
        "sensor": sensors.to_dict(),
        "noise": float(noise),
        "param_names": _param_names(data),
        "coil_names": list(sensors.coil_names),
        "arch": model.arch,
        "eq_names": list(model.eq_names),
    }
    model.skip_field = False
    metrics.update(
        _fit_report(
            model,
            data,
            signals,
            eq_all,
            eq_names,
            train_idx,
            val_idx,
            test_idx,
            float(noise),
            rms,
            int(seed),
        )
    )
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(model, out / "inverse.pt")
    if evaluate and test_idx.size:
        metrics.update(
            _evaluate_splits(
                model,
                data,
                sensors,
                signals,
                rms,
                test_idx,
                noise=float(noise),
                seed=int(seed),
                ood_path=ood_path,
                out_dir=out_dir,
                verbose=verbose,
            )
        )
    if out_dir is not None:
        _write_metrics(Path(out_dir) / "metrics.json", metrics)
    return model, metrics


def _fit_report(
    model: InverseNet,
    data: dict,
    signals: np.ndarray,
    eq_all: np.ndarray | None,
    eq_names: list[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    noise: float,
    rms: np.ndarray,
    seed: int,
) -> dict:
    """Train/val flux error and per-parameter R^2. Caps the flux sample count."""
    model.eval()
    model.skip_field = False

    def _rel(index: np.ndarray) -> float:
        if index.size == 0:
            return float("nan")
        take = index if index.size <= 400 else index[:400]
        pred = _predict_numpy(model, signals[take], _coil_matrix(data, take))
        psi = np.asarray(data["psi"], dtype=float)[take]
        mask = np.asarray(data["mask"])[take]
        return _mean_finite([relative_l2(pred[i], psi[i], mask[i]) for i in range(take.size)])

    report: dict = {"train_rel_l2": _rel(train_idx), "val_rel_l2": _rel(val_idx)}
    if eq_all is None or model.eq_head is None or test_idx.size == 0:
        return report

    def _eq_hat(index: np.ndarray, noisy: bool) -> np.ndarray:
        sig = np.asarray(signals[index], dtype=np.float32)
        if noisy and noise > 0.0:
            unit = np.random.default_rng(seed + 11).normal(size=sig.shape)
            sig = sig + np.float32(noise) * np.asarray(rms, dtype=np.float32) * unit.astype(np.float32)
        coils = _coil_matrix(data, index)
        coil_t = None if coils is None else torch.as_tensor(np.asarray(coils, dtype=np.float32))
        with torch.inference_mode():
            hat = model.equilibrium_params(torch.as_tensor(sig), coil_t)
        return hat.detach().cpu().numpy()

    report["param_r2"] = _r2_columns(_eq_hat(test_idx, True), eq_all[test_idx], eq_names)
    report["param_r2_clean"] = _r2_columns(_eq_hat(test_idx, False), eq_all[test_idx], eq_names)
    return report


def _evaluate_splits(
    model,
    data,
    sensors,
    signals,
    rms,
    test_idx,
    noise: float,
    seed: int,
    ood_path,
    out_dir,
    verbose: bool,
) -> dict:
    grid = Grid(np.asarray(data["R"], dtype=float), np.asarray(data["Z"], dtype=float))
    clean = signals[test_idx]
    # One unit-normal draw is reused across the noise sweep.
    unit = np.random.default_rng(int(seed) + 1000).normal(size=clean.shape)
    sweep = {}
    full = {}
    pred_for_plot = None
    l2_for_plot = None
    for level in _NOISE_SWEEP:
        if level == 0.0:
            noisy = clean
        else:
            noisy = clean + level * np.asarray(rms, dtype=float) * unit
        headline = abs(level - noise) < 1e-12 or (noise == 0.0 and level == 0.0)
        raw = _evaluate_prepared(
            model,
            data,
            test_idx,
            noisy,
            n_q=50,
            seed=int(seed) + int(round(level * 1000)),
            detailed=headline,
        )
        sweep[f"{level:.2f}"] = {"relative_l2_psi": raw["relative_l2_psi"]}
        if headline:
            full = _summary_metrics(raw)
            pred_for_plot = raw["_pred"]
            l2_for_plot = raw["_l2_per_sample"]
    if not full:
        raw = _evaluate_prepared(model, data, test_idx, clean + noise * np.asarray(rms) * unit, n_q=50, seed=seed)
        full = _summary_metrics(raw)
        pred_for_plot = raw["_pred"]
        l2_for_plot = raw["_l2_per_sample"]
    result = {"test": full, "noise_sweep": sweep}
    if verbose:
        print(
            f"test  relL2 {full['relative_l2_psi']:.4f}  "
            f"axis {full['axis_error_m']:.4f} m  "
            f"lcfs {full['lcfs_shape_error_m']:.4f} m  "
            f"q95 {full['q95_abs_error']:.4f} (n={full['q95_n']})",
            flush=True,
        )
    ood_file = Path(ood_path) if ood_path else None
    if ood_file is not None and ood_file.is_file() and _same_grid(data, ood_file):
        ood = _load_npz(ood_file)
        ood_grid = Grid(np.asarray(ood["R"], dtype=float), np.asarray(ood["Z"], dtype=float))
        ood_signals = _build_signals(sensors, ood, ood_grid)
        ood_idx = np.arange(ood_signals.shape[0])
        ood_unit = np.random.default_rng(int(seed) + 2000).normal(size=ood_signals.shape)
        ood_sweep = {}
        ood_full = None
        for level in _NOISE_SWEEP:
            noisy = ood_signals if level == 0.0 else ood_signals + level * np.asarray(rms, dtype=float) * ood_unit
            headline = abs(level - noise) < 1e-12 or (noise == 0.0 and level == 0.0)
            raw = _evaluate_prepared(
                model, ood, ood_idx, noisy, n_q=50, seed=int(seed) + 7, detailed=headline
            )
            ood_sweep[f"{level:.2f}"] = {"relative_l2_psi": raw["relative_l2_psi"]}
            if headline:
                ood_full = _summary_metrics(raw)
        if ood_full is None:
            ood_full = _summary_metrics(
                _evaluate_prepared(
                    model,
                    ood,
                    ood_idx,
                    ood_signals + noise * np.asarray(rms) * ood_unit,
                    n_q=50,
                    seed=seed,
                )
            )
        result["ood"] = ood_full
        result["ood_noise_sweep"] = ood_sweep
        result["ood_dataset"] = str(ood_file)
        if verbose:
            print(
                f"ood   relL2 {ood_full['relative_l2_psi']:.4f}  "
                f"axis {ood_full['axis_error_m']:.4f} m  "
                f"lcfs {ood_full['lcfs_shape_error_m']:.4f} m  "
                f"q95 {ood_full['q95_abs_error']:.4f}",
                flush=True,
            )
    if out_dir is not None and pred_for_plot is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        finite = np.isfinite(l2_for_plot)
        order = np.argsort(np.where(finite, l2_for_plot, np.inf))
        mid = int(order[len(order) // 2])
        row = int(test_idx[mid])
        save_reconstruction_figure(
            out / "reconstruction.png",
            grid,
            sensors,
            np.asarray(data["psi"][row], dtype=float),
            pred_for_plot[mid],
            np.asarray(data["mask"][row], dtype=bool),
        )
        merged = dict(result)
        # Caller merges onto the training metrics. The file is rewritten by train_inverse
        # only for the eval keys; write a partial file here and let the caller overwrite.
        result["_figure"] = str(out / "reconstruction.png")
    return result


def _same_grid(data: dict, other: Path) -> bool:
    with np.load(other, allow_pickle=False) as archive:
        R = np.asarray(archive["R"], dtype=float)
        Z = np.asarray(archive["Z"], dtype=float)
    return np.allclose(R, np.asarray(data["R"], dtype=float)) and np.allclose(Z, np.asarray(data["Z"], dtype=float))


def _evaluate_prepared(model, data, index, signals, n_q: int, seed: int, detailed: bool = True) -> dict:
    """Metrics for signals that already include whatever noise is desired."""
    grid = Grid(np.asarray(data["R"], dtype=float), np.asarray(data["Z"], dtype=float))
    psi_true = np.asarray(data["psi"], dtype=np.float64)[index]
    J = np.asarray(data["J"], dtype=np.float64)[index]
    mask = np.asarray(data["mask"], dtype=bool)[index]
    coils = _coil_matrix(data, index)
    pred = _predict_numpy(model, signals, coils)
    return _score_prediction(
        pred, psi_true, J, mask, data, index, grid, n_q=n_q, seed=seed, detailed=detailed
    )


def _score_prediction(pred, psi_true, J, mask, data, index, grid, n_q: int, seed: int, detailed: bool = True) -> dict:
    l2 = []
    axis_err = []
    shape_err = []
    shape_fail = 0
    gs_err = []
    names = _param_names(data)
    q_err = []
    q_fail = 0
    n = pred.shape[0]
    q_slots = min(int(n_q), n) if detailed else 0
    if q_slots:
        q_pick = np.random.default_rng(seed + 3).choice(n, size=q_slots, replace=False)
        q_set = set(int(i) for i in q_pick)
    else:
        q_set = set()
    for i in range(n):
        l2.append(relative_l2(pred[i], psi_true[i], mask[i]))
        if not detailed:
            continue
        try:
            r_p, z_p, psi_p = find_magnetic_axis(pred[i], grid, mask[i])
            axis_err.append(
                float(
                    np.hypot(
                        r_p - float(data["R_axis"][index[i]]),
                        z_p - float(data["Z_axis"][index[i]]),
                    )
                )
            )
        except (ValueError, np.linalg.LinAlgError):
            r_p = z_p = psi_p = float("nan")
            axis_err.append(float("nan"))
        psi_axis_true = float(data["psi_axis"][index[i]])
        try:
            r_t, z_t = flux_contour(
                psi_true[i],
                grid,
                1e-3 * psi_axis_true,
                float(data["R_axis"][index[i]]),
                float(data["Z_axis"][index[i]]),
            )
            if not np.isfinite(psi_p) or psi_p <= 0.0:
                raise ValueError("predicted axis flux is not positive")
            r_hat, z_hat = flux_contour(pred[i], grid, 1e-3 * float(psi_p), float(r_p), float(z_p))
            shape_err.append(lcfs_shape_error(r_hat, z_hat, r_t, z_t))
        except (ValueError, np.linalg.LinAlgError):
            shape_fail += 1
        try:
            rhs = -MU0 * grid.RR * J[i]
            gs_err.append(gs_residual(pred[i], grid, rhs, mask[i]))
        except (ValueError, np.linalg.LinAlgError):
            gs_err.append(float("nan"))
        if i in q_set:
            try:
                F = make_F(
                    np.asarray(data["params"][index[i]], dtype=float),
                    names,
                    grid,
                    psi_true[i],
                    mask[i],
                    psi_axis_true,
                )
                if F is None or not np.isfinite(psi_p) or psi_p <= 0.0:
                    raise ValueError("F profile is not available")
                q_true = q95(
                    psi_true[i],
                    grid,
                    psi_axis_true,
                    0.0,
                    float(data["R_axis"][index[i]]),
                    float(data["Z_axis"][index[i]]),
                    F,
                )
                q_pred = q95(pred[i], grid, float(psi_p), 0.0, float(r_p), float(z_p), F)
                q_err.append(abs(float(q_pred) - float(q_true)))
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                q_fail += 1
    l2_arr = np.asarray(l2, dtype=float)
    return {
        "n": int(n),
        "relative_l2_psi": _mean_finite(l2),
        "relative_l2_psi_median": float(np.nanmedian(l2_arr)) if l2_arr.size else float("nan"),
        "axis_error_m": _mean_finite(axis_err),
        "lcfs_shape_error_m": _mean_finite(shape_err),
        "lcfs_n": int(len(shape_err)),
        "lcfs_fail": int(shape_fail),
        "gs_residual": _mean_finite(gs_err),
        "q95_abs_error": _mean_finite(q_err),
        "q95_n": int(len(q_err)),
        "q95_fail": int(q_fail),
        "_l2_per_sample": l2_arr,
        "_pred": pred,
    }


def _write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in metrics.items() if not str(key).startswith("_")}
    path.write_text(json.dumps(_jsonify(public), indent=2) + "\n")


def save_checkpoint(model: InverseNet, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = getattr(model, "meta", {})
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_sensors": model.n_sensors,
            "nr": model.nr,
            "nz": model.nz,
            "n_aux": model.n_aux,
            "n_coils": model.n_coils,
            "latent": model.latent,
            "hidden": model.hidden,
            "channels": model.channels,
            "base": model.base,
            "aux_names": list(meta.get("aux_names", [])),
            "grid_R": np.asarray(meta.get("grid_R", []), dtype=np.float64),
            "grid_Z": np.asarray(meta.get("grid_Z", []), dtype=np.float64),
            "sensor": meta.get("sensor", {}),
            "noise": float(meta.get("noise", 0.0)),
            "param_names": list(meta.get("param_names", [])),
            "coil_names": list(meta.get("coil_names", [])),
            "arch": model.arch,
            "n_eq": int(model.n_eq),
            "eq_names": list(getattr(model, "eq_names", [])),
            "surrogate_config": None if model.surrogate is None else dict(model.surrogate.config),
            "surrogate_frozen": bool(model.surrogate_frozen),
        },
        path,
    )


def load_inverse(path) -> InverseNet:
    """Load a checkpoint written by ``train_inverse`` / ``save_checkpoint``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = str(ckpt.get("arch", "direct"))
    n_eq = int(ckpt.get("n_eq", 0))
    model = InverseNet(
        n_sensors=int(ckpt["n_sensors"]),
        nr=int(ckpt["nr"]),
        nz=int(ckpt["nz"]),
        n_aux=int(ckpt["n_aux"]),
        n_coils=int(ckpt["n_coils"]),
        latent=int(ckpt["latent"]),
        hidden=int(ckpt["hidden"]),
        channels=int(ckpt.get("channels", 32)),
        base=int(ckpt.get("base", 8)),
        arch=arch,
        n_eq=n_eq,
    )
    if arch == "hybrid" and ckpt.get("surrogate_config"):
        from models.surrogate import TrainedSurrogate

        model.attach_surrogate(
            TrainedSurrogate(ckpt["surrogate_config"]),
            frozen=bool(ckpt.get("surrogate_frozen", True)),
        )
    model.load_state_dict(ckpt["state_dict"])
    model.eq_names = [str(x) for x in ckpt.get("eq_names", [])]
    model.skip_field = False
    model.meta = {
        "aux_names": list(ckpt.get("aux_names", [])),
        "grid_R": np.asarray(ckpt.get("grid_R", []), dtype=np.float64),
        "grid_Z": np.asarray(ckpt.get("grid_Z", []), dtype=np.float64),
        "sensor": ckpt.get("sensor", {}),
        "noise": float(ckpt.get("noise", 0.0)),
        "param_names": list(ckpt.get("param_names", [])),
        "coil_names": list(ckpt.get("coil_names", [])),
    }
    model.eval()
    return model


def reconstruct(model: InverseNet, signals, coil_currents=None) -> np.ndarray:
    """Map diagnostic signals to ``psi`` with shape ``(N, nr, nz)``."""
    sig = np.asarray(signals, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[None, :]
    coils = None
    if coil_currents is not None:
        coils = np.asarray(coil_currents, dtype=np.float32)
        if coils.ndim == 1:
            coils = coils[None, :]
    return _predict_numpy(model, sig, coils)


def _default_ood(dataset_path: Path) -> Path | None:
    name = dataset_path.name
    if "ood" in name:
        return None
    candidate = dataset_path.with_name(name.replace(".npz", "_ood.npz"))
    if candidate.is_file():
        return candidate
    return None


def _default_freegs(dataset_path: Path) -> Path | None:
    candidate = dataset_path.with_name("freegs_65.npz")
    if candidate.is_file() and candidate.resolve() != dataset_path.resolve():
        return candidate
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train inverse equilibrium reconstruction")
    parser.add_argument("--data", required=True)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--out", default="outputs/inverse")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--arch", default="auto", choices=["auto", "direct", "hybrid"])
    parser.add_argument("--surrogate", default=None, help="frozen params->psi checkpoint for --arch hybrid")
    parser.add_argument("--ood", default=None, help="optional OOD npz; default is a sibling *_ood.npz")
    parser.add_argument("--freegs", default=None, help="optional FreeGS npz; trains a second model")
    parser.add_argument("--skip-freegs", action="store_true")
    args = parser.parse_args(argv)
    data_path = Path(args.data)
    ood = Path(args.ood) if args.ood else _default_ood(data_path)
    freegs = Path(args.freegs) if args.freegs else _default_freegs(data_path)
    model, metrics = train_inverse(
        data_path,
        noise=args.noise,
        epochs=args.epochs,
        seed=args.seed,
        out_dir=args.out,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        physics_weight=args.physics_weight,
        num_threads=max(1, min(int(args.threads), 2)),
        ood_path=ood,
        evaluate=True,
        verbose=True,
        arch=args.arch,
        surrogate_path=args.surrogate,
    )
    if freegs is not None and freegs.is_file() and not args.skip_freegs:
        print(f"training a second model on {freegs}", flush=True)
        _free_model, free_metrics = train_inverse(
            freegs,
            noise=args.noise,
            epochs=min(int(args.epochs), 40),
            seed=args.seed,
            out_dir=str(Path(args.out) / "freegs"),
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            physics_weight=args.physics_weight,
            num_threads=max(1, min(int(args.threads), 2)),
            arch="direct",
            ood_path=None,
            evaluate=True,
            verbose=True,
        )
        metrics["freegs"] = free_metrics.get("test")
        metrics["freegs_dataset"] = str(freegs)
        metrics["freegs_noise_sweep"] = free_metrics.get("noise_sweep")
    else:
        metrics["freegs"] = None
    # Drop non-serializable leftovers and rewrite the combined metrics.
    out = Path(args.out)
    _write_metrics(out / "metrics.json", metrics)
    print(f"wrote {out / 'metrics.json'}", flush=True)
    _ = model


if __name__ == "__main__":
    main()
