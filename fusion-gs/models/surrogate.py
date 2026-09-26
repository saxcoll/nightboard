"""Supervised surrogate: equilibrium parameters -> poloidal flux psi(R, Z).

The map is trained on fixed-boundary Grad-Shafranov solutions that share one
(R, Z) grid (see ``data/generate.py``). Two architectures are provided.

Normalization
-------------
Inputs are standardized with the training-split mean and standard deviation of
each parameter. Columns are ordered

    R0, a, kappa, delta, Ip, beta0, alpha, gamma, B0

so a current in amperes and an elongation of order one both enter the network
as O(1) numbers.

Outputs use a per-sample scale. A small MLP head predicts ``log(psi_axis)`` and
the spatial network predicts the dimensionless shape ``psi / psi_axis``. The
physical flux is ``shape * exp(log psi_axis)``. On this dataset that shape lies
in ``[0, 1]`` inside the plasma and is identically 0 outside, while ``psi_axis``
itself spans about an order of magnitude because ``Ip`` and the plasma size
vary. Dividing by the per-sample axis value keeps the spatial regression on a
common scale; a single global standard deviation would fold that amplitude into
the field and inflate the relative error of low-current plasmas. The degeneracy
(shape doubling while the scale halves) is broken by an MSE penalty on
``log psi_axis``.

The data term is a weighted MSE of the physical prediction divided by the true
``psi_axis`` (weight 1 inside the plasma mask, ``outside_weight`` outside).

Physics term
------------
``physics_weight * ||Delta* psi_pred + mu0 R J_true||^2 / ||mu0 R J_true||^2``
on nodes whose four orthogonal neighbours are inside the mask. ``Delta*`` is the
conservative five-point stencil of ``gs.solver.delta_star_fd`` (arithmetic
midpoints ``R_{i+1/2}``). The ratio is the square of the relative GS residual,
so ``physics_weight`` is dimensionless. It is held at 0 for the first
``physics_warmup`` epochs and then set to the requested value, which keeps the
early updates on the flux fit.

API
---
``train_surrogate``, ``load_surrogate``, ``predict``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

# Keep BLAS/OpenMP from grabbing every core; this process shares the machine.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gs.solver import MU0, Grid, ShapeParams

torch.set_num_threads(1)

PARAM_NAMES = ["R0", "a", "kappa", "delta", "Ip", "beta0", "alpha", "gamma", "B0"]

# Clamp on the log-axis head. The training range is about [-2.4, 0.1].
_LOG_AXIS_MIN = -6.0
_LOG_AXIS_MAX = 2.0


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def delta_star_fd_torch(
    psi: torch.Tensor, R: torch.Tensor, dR: float, dZ: float
) -> torch.Tensor:
    """Second-order conservative ``Delta*`` matching ``gs.solver.delta_star_fd``.

    ``psi`` is ``(nr, nz)`` or ``(batch, nr, nz)``. The radial piece is

        flux_{i+1/2} = (psi_{i+1} - psi_i) / (R_{i+1/2} dR)
        radial_i = R_i (flux_{i+1/2} - flux_{i-1/2}) / dR

    with ``R_{i+1/2} = (R_i + R_{i+1}) / 2``, plus the standard second difference
    in Z. Edge rows and columns are 0 (the NumPy routine stores NaN there); only
    the interior ``[1:-1, 1:-1]`` is the stencil.
    """
    single = psi.ndim == 2
    if single:
        psi = psi.unsqueeze(0)
    nr, nz = psi.shape[-2], psi.shape[-1]
    out = torch.zeros_like(psi)
    if nr < 3 or nz < 3:
        return out[0] if single else out
    r_half = 0.5 * (R[:-1] + R[1:])
    flux = (psi[:, 1:, :] - psi[:, :-1, :]) / (float(dR) * r_half.view(1, -1, 1))
    radial = R[1:-1].view(1, -1, 1) * (flux[:, 1:, :] - flux[:, :-1, :]) / float(dR)
    d2z = (psi[:, :, 2:] - 2.0 * psi[:, :, 1:-1] + psi[:, :, :-2]) / float(dZ) ** 2
    out[:, 1:-1, 1:-1] = radial[:, :, 1:-1] + d2z[:, 1:-1, :]
    return out[0] if single else out


def interior_mask(mask: torch.Tensor) -> torch.Tensor:
    """Nodes where ``mask`` and its four orthogonal neighbours are true.

    Array edges are false. ``mask`` is ``(nr, nz)`` or ``(batch, nr, nz)``.
    """
    single = mask.ndim == 2
    if single:
        mask = mask.unsqueeze(0)
    out = torch.zeros_like(mask, dtype=torch.bool)
    if mask.shape[-2] >= 3 and mask.shape[-1] >= 3:
        out[:, 1:-1, 1:-1] = (
            mask[:, 1:-1, 1:-1]
            & mask[:, :-2, 1:-1]
            & mask[:, 2:, 1:-1]
            & mask[:, 1:-1, :-2]
            & mask[:, 1:-1, 2:]
        )
    return out[0] if single else out


def physics_loss(
    psi: torch.Tensor,
    J: torch.Tensor,
    R: torch.Tensor,
    mask: torch.Tensor,
    dR: float,
    dZ: float,
) -> torch.Tensor:
    """Mean over the batch of ``||Delta* psi + mu0 R J||^2 / ||mu0 R J||^2``.

    The norm is restricted to interior plasma nodes (four in-mask neighbours).
    Samples with no such nodes contribute 0.
    """
    delta = delta_star_fd_torch(psi, R, dR, dZ)
    Rb = R[1:-1].view(1, -1, 1)
    Jb = J[:, 1:-1, 1:-1]
    resid = delta[:, 1:-1, 1:-1] + float(MU0) * Rb * Jb
    inter = interior_mask(mask)[:, 1:-1, 1:-1].to(dtype=psi.dtype)
    num = (resid * resid * inter).sum(dim=(1, 2))
    den = ((float(MU0) * Rb * Jb) ** 2 * inter).sum(dim=(1, 2)).clamp_min(1e-12)
    has = inter.sum(dim=(1, 2)) > 0
    per = torch.where(has, num / den, torch.zeros_like(num))
    return per.mean()


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, outside_weight: float
) -> torch.Tensor:
    """Per-sample weighted MSE, then a batch mean.

    Weight is 1 on mask nodes and ``outside_weight`` elsewhere.
    """
    err = (pred - target) ** 2
    w = mask.to(dtype=pred.dtype)
    w = w + (1.0 - w) * float(outside_weight)
    num = (err * w).flatten(1).sum(dim=1)
    den = w.flatten(1).sum(dim=1).clamp_min(1.0)
    return (num / den).mean()


def _init_scale_head(head: nn.Sequential, log_mean: float) -> None:
    last = head[-1]
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, float(log_mean))


def _scale_head(n_params: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_params, 64),
        nn.GELU(),
        nn.Linear(64, 64),
        nn.GELU(),
        nn.Linear(64, 1),
    )


class MLPCNNSurrogate(nn.Module):
    """MLP encoder, transposed-convolution decoder, coordinate refinement.

    Parameters ``(B, P)`` are mapped to a ``latent_ch × 8 × 8`` feature volume,
    decoded by three stride-2 transposed convolutions to ``64 × 64``, then
    bilinearly resampled to ``(nr, nz)``. Odd sizes such as 65 are produced by
    that final interpolate. Normalized ``(R, Z)`` and the parameters are
    concatenated at full resolution and mixed by a 3×3 convolution so the flux
    can sit at an absolute position (a pure CNN is translation equivariant; the
    tokamak is not).

    ``forward`` returns physical ``psi`` of shape ``(B, nr, nz)``.
    ``shape_and_log_scale`` returns the dimensionless shape and ``log psi_axis``.
    """

    def __init__(
        self,
        n_params: int = 9,
        nr: int = 65,
        nz: int = 65,
        hidden: int = 256,
        latent_ch: int = 32,
    ):
        super().__init__()
        self.n_params = int(n_params)
        self.nr = int(nr)
        self.nz = int(nz)
        self.hidden = int(hidden)
        self.latent_ch = int(latent_ch)
        self.grid0 = 8
        self.fc1 = nn.Linear(self.n_params, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.latent_ch * self.grid0 * self.grid0)
        self.up1 = nn.ConvTranspose2d(self.latent_ch, 32, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        refine_in = 8 + 2 + self.n_params
        self.ref1 = nn.Conv2d(refine_in, 16, kernel_size=3, padding=1)
        self.ref2 = nn.Conv2d(16, 1, kernel_size=1)
        self.scale_head = _scale_head(self.n_params)
        self.register_buffer("Rn", torch.linspace(-1.0, 1.0, self.nr))
        self.register_buffer("Zn", torch.linspace(-1.0, 1.0, self.nz))
        _init_scale_head(self.scale_head, math.log(0.37))
        nn.init.xavier_uniform_(self.ref2.weight, gain=0.1)
        nn.init.zeros_(self.ref2.bias)

    def set_grid(self, R: torch.Tensor | np.ndarray, Z: torch.Tensor | np.ndarray) -> None:
        """Store normalized physical coordinates used by the refinement head."""
        R_t = torch.as_tensor(R, dtype=torch.float32, device=self.Rn.device).flatten()
        Z_t = torch.as_tensor(Z, dtype=torch.float32, device=self.Zn.device).flatten()
        if R_t.numel() != self.nr or Z_t.numel() != self.nz:
            raise ValueError(f"grid {(R_t.numel(), Z_t.numel())} != {(self.nr, self.nz)}")
        span_r = (R_t[-1] - R_t[0]).clamp_min(1e-8)
        span_z = (Z_t[-1] - Z_t[0]).clamp_min(1e-8)
        self.Rn.copy_(2.0 * (R_t - R_t[0]) / span_r - 1.0)
        self.Zn.copy_(2.0 * (Z_t - Z_t[0]) / span_z - 1.0)

    def shape_and_log_scale(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.fc1(params))
        h = F.gelu(self.fc2(h))
        z = F.gelu(self.fc3(h))
        b = params.shape[0]
        z = z.view(b, self.latent_ch, self.grid0, self.grid0)
        z = F.gelu(self.up1(z))
        z = F.gelu(self.up2(z))
        z = F.gelu(self.up3(z))
        z = F.interpolate(z, size=(self.nr, self.nz), mode="bilinear", align_corners=True)
        rr = self.Rn.view(1, 1, self.nr, 1).expand(b, 1, self.nr, self.nz)
        zz = self.Zn.view(1, 1, 1, self.nz).expand(b, 1, self.nr, self.nz)
        pp = params.view(b, self.n_params, 1, 1).expand(b, self.n_params, self.nr, self.nz)
        x = torch.cat([z, rr, zz, pp], dim=1)
        x = F.gelu(self.ref1(x))
        shape = self.ref2(x).squeeze(1)
        log_s = self.scale_head(params).squeeze(-1)
        return shape, log_s

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        shape, log_s = self.shape_and_log_scale(params)
        scale = torch.exp(log_s.clamp(_LOG_AXIS_MIN, _LOG_AXIS_MAX)).unsqueeze(-1).unsqueeze(-1)
        return shape * scale


class SpectralConv2d(nn.Module):
    """Fourier layer keeping a corner of low modes, weights stored as real/imag."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / (self.in_channels * self.out_channels)
        shape = (self.in_channels, self.out_channels, self.modes1, self.modes2)
        self.wr1 = nn.Parameter(scale * torch.randn(shape))
        self.wi1 = nn.Parameter(scale * torch.randn(shape))
        self.wr2 = nn.Parameter(scale * torch.randn(shape))
        self.wi2 = nn.Parameter(scale * torch.randn(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        # Cap so the positive and negative R-mode blocks do not overlap.
        m1 = min(self.modes1, h // 2)
        m2 = min(self.modes2, w // 2 + 1)
        if m1 > 0 and m2 > 0:
            w1 = torch.complex(self.wr1[:, :, :m1, :m2], self.wi1[:, :, :m1, :m2])
            w2 = torch.complex(self.wr2[:, :, :m1, :m2], self.wi2[:, :, :m1, :m2])
            out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], w1)
            out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], w2)
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes, modes)
        self.w = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.w(x))


class FNOSurrogate(nn.Module):
    """Small Fourier Neural Operator on a parameter-and-coordinate field.

    Input channels are the standardized parameters broadcast over the grid,
    normalized ``(R, Z)``, and the fixed-boundary mask from
    ``ShapeParams.level_set`` (1 inside the plasma). A pointwise lift, a stack
    of Fourier layers, and a pointwise projection produce the dimensionless
    flux shape. ``log psi_axis`` comes from an MLP on the parameters alone.

    ``forward(params, R, Z, mask)`` returns physical ``psi`` ``(B, nr, nz)``.
    """

    def __init__(
        self,
        n_params: int = 9,
        nr: int = 65,
        nz: int = 65,
        width: int = 16,
        modes: int = 8,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n_params = int(n_params)
        self.nr = int(nr)
        self.nz = int(nz)
        self.width = int(width)
        self.modes = int(modes)
        self.n_layers = int(n_layers)
        in_ch = self.n_params + 3  # params, R, Z, mask
        self.lift = nn.Conv2d(in_ch, self.width, kernel_size=1)
        self.blocks = nn.ModuleList(FNOBlock(self.width, self.modes) for _ in range(self.n_layers))
        self.proj = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.width, 1, kernel_size=1),
        )
        self.scale_head = _scale_head(self.n_params)
        _init_scale_head(self.scale_head, math.log(0.37))
        last = self.proj[-1]
        nn.init.xavier_uniform_(last.weight, gain=0.1)
        nn.init.zeros_(last.bias)

    def shape_and_log_scale(
        self,
        params: torch.Tensor,
        R: torch.Tensor,
        Z: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = params.shape[0]
        nr, nz = int(R.shape[0]), int(Z.shape[0])
        span_r = (R[-1] - R[0]).clamp_min(1e-8)
        span_z = (Z[-1] - Z[0]).clamp_min(1e-8)
        rn = 2.0 * (R - R[0]) / span_r - 1.0
        zn = 2.0 * (Z - Z[0]) / span_z - 1.0
        rr = rn.view(1, 1, nr, 1).expand(b, 1, nr, nz)
        zz = zn.view(1, 1, 1, nz).expand(b, 1, nr, nz)
        pp = params.view(b, self.n_params, 1, 1).expand(b, self.n_params, nr, nz)
        mm = mask.to(dtype=params.dtype).unsqueeze(1)
        x = self.lift(torch.cat([pp, rr, zz, mm], dim=1))
        for block in self.blocks:
            x = block(x)
        shape = self.proj(x).squeeze(1)
        log_s = self.scale_head(params).squeeze(-1)
        return shape, log_s

    def forward(
        self,
        params: torch.Tensor,
        R: torch.Tensor,
        Z: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        shape, log_s = self.shape_and_log_scale(params, R, Z, mask)
        scale = torch.exp(log_s.clamp(_LOG_AXIS_MIN, _LOG_AXIS_MAX)).unsqueeze(-1).unsqueeze(-1)
        return shape * scale


def masks_from_params(params: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Plasma mask from ``ShapeParams.level_set`` (True inside), shape ``(N, nr, nz)``."""
    params = np.asarray(params, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    rr, zz = np.meshgrid(R, Z, indexing="ij")
    out = np.empty((params.shape[0], R.size, Z.size), dtype=bool)
    for i, row in enumerate(params):
        level = ShapeParams(float(row[0]), float(row[1]), float(row[2]), float(row[3])).level_set()
        out[i] = np.asarray(level(rr, zz), dtype=np.float64) < 0.0
    return out


def _build_net(config: dict) -> nn.Module:
    name = config["model"]
    if name == "mlpcnn":
        return MLPCNNSurrogate(
            n_params=int(config["n_params"]),
            nr=int(config["nr"]),
            nz=int(config["nz"]),
            hidden=int(config.get("hidden", 256)),
            latent_ch=int(config.get("latent_ch", 32)),
        )
    if name == "fno":
        return FNOSurrogate(
            n_params=int(config["n_params"]),
            nr=int(config["nr"]),
            nz=int(config["nz"]),
            width=int(config.get("fno_width", 16)),
            modes=int(config.get("fno_modes", 8)),
            n_layers=int(config.get("fno_layers", 3)),
        )
    raise ValueError(f"unknown model {name!r}")


class TrainedSurrogate(nn.Module):
    """Network plus the normalization and grid needed to reload without the dataset.

    Buffers ``param_mean``, ``param_std``, ``R`` and ``Z`` travel with
    ``state_dict``. ``predict`` consumes raw parameters in ``PARAM_NAMES`` order.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.model_name = str(config["model"])
        self.net = _build_net(config)
        n_params = int(config["n_params"])
        nr = int(config["nr"])
        nz = int(config["nz"])
        self.register_buffer("param_mean", torch.zeros(n_params, dtype=torch.float32))
        self.register_buffer("param_std", torch.ones(n_params, dtype=torch.float32))
        self.register_buffer("R", torch.zeros(nr, dtype=torch.float32))
        self.register_buffer("Z", torch.zeros(nz, dtype=torch.float32))
        # Full-precision grid used to rebuild the boundary mask at inference.
        # Not a buffer: it is restored from the checkpoint payload.
        self.grid_R: np.ndarray | None = None
        self.grid_Z: np.ndarray | None = None

    def set_norm(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        R: np.ndarray,
        Z: np.ndarray,
    ) -> None:
        self.param_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.param_std.copy_(torch.as_tensor(std, dtype=torch.float32))
        self.grid_R = np.asarray(R, dtype=np.float64).copy()
        self.grid_Z = np.asarray(Z, dtype=np.float64).copy()
        self.R.copy_(torch.as_tensor(self.grid_R, dtype=torch.float32))
        self.Z.copy_(torch.as_tensor(self.grid_Z, dtype=torch.float32))
        if isinstance(self.net, MLPCNNSurrogate):
            self.net.set_grid(self.grid_R, self.grid_Z)

    def normalize(self, params: torch.Tensor) -> torch.Tensor:
        return (params - self.param_mean) / self.param_std

    def shape_and_log_scale(
        self, params_raw: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.normalize(params_raw)
        if self.model_name == "fno":
            if mask is None:
                raise ValueError("FNOSurrogate requires a boundary mask")
            return self.net.shape_and_log_scale(x, self.R, self.Z, mask)
        return self.net.shape_and_log_scale(x)

    def forward(self, params_raw: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        shape, log_s = self.shape_and_log_scale(params_raw, mask=mask)
        scale = torch.exp(log_s.clamp(_LOG_AXIS_MIN, _LOG_AXIS_MAX)).unsqueeze(-1).unsqueeze(-1)
        return shape * scale


def _load_npz(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def _order_params(params: np.ndarray, names) -> np.ndarray:
    names = [str(x) for x in list(names)]
    missing = [n for n in PARAM_NAMES if n not in names]
    if missing:
        raise ValueError(f"dataset is missing parameters {missing}")
    index = [names.index(n) for n in PARAM_NAMES]
    return np.ascontiguousarray(params[:, index])


def _split_indices(split: np.ndarray, value: int) -> np.ndarray:
    return np.flatnonzero(np.asarray(split) == value).astype(np.int64)


def _slice_samples(data: dict, idx: np.ndarray) -> dict:
    """Row-slice per-sample arrays. Grid vectors and parameter names stay intact."""
    n = int(np.asarray(data["split"]).shape[0])
    keep_full = {"R", "Z", "param_names"}
    out = {}
    for key, value in data.items():
        arr = np.asarray(value)
        if key not in keep_full and arr.ndim >= 1 and arr.shape[0] == n:
            out[key] = arr[idx]
        else:
            out[key] = arr
    return out


def _batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator | None):
    if rng is not None:
        indices = indices[rng.permutation(indices.size)]
    for start in range(0, int(indices.size), int(batch_size)):
        yield indices[start : start + int(batch_size)]


def _as_float(value) -> float:
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def _train_objective(
    model: TrainedSurrogate,
    params: torch.Tensor,
    psi: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor,
    psi_axis: torch.Tensor,
    R: torch.Tensor,
    dR: float,
    dZ: float,
    *,
    physics_weight: float,
    outside_weight: float,
    scale_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shape, log_s = model.shape_and_log_scale(params, mask=mask)
    scale = torch.exp(log_s.clamp(_LOG_AXIS_MIN, _LOG_AXIS_MAX))
    psi_pred = shape * scale.unsqueeze(-1).unsqueeze(-1)
    axis = psi_axis.clamp_min(1e-12).unsqueeze(-1).unsqueeze(-1)
    loss_psi = masked_mse(psi_pred / axis, psi / axis, mask, outside_weight)
    loss_scale = F.mse_loss(log_s, torch.log(psi_axis.clamp_min(1e-12)))
    if physics_weight > 0.0:
        loss_phys = physics_loss(psi_pred, J, R, mask, dR, dZ)
    else:
        loss_phys = psi_pred.new_zeros(())
    loss = loss_psi + float(scale_coef) * loss_scale + float(physics_weight) * loss_phys
    return loss, loss_psi, loss_scale, loss_phys


@torch.no_grad()
def _mean_over(
    model: TrainedSurrogate,
    idx_all: np.ndarray,
    tensors: dict,
    batch_size: int,
    *,
    physics_weight: float,
    outside_weight: float,
    scale_coef: float,
    dR: float,
    dZ: float,
) -> dict[str, float]:
    """Mean training objective and relative L2 over ``idx_all`` (no shuffle)."""
    model.eval()
    totals = {k: 0.0 for k in ("loss", "psi", "scale", "phys", "rel")}
    n = 0
    rel_num = 0.0
    for idx in _batches(idx_all, batch_size, rng=None):
        sl = torch.from_numpy(idx)
        params = tensors["params"][sl]
        psi = tensors["psi"][sl]
        J = tensors["J"][sl]
        mask = tensors["mask"][sl]
        axis = tensors["psi_axis"][sl]
        loss, loss_psi, loss_scale, loss_phys = _train_objective(
            model,
            params,
            psi,
            J,
            mask,
            axis,
            tensors["R"],
            dR,
            dZ,
            physics_weight=physics_weight,
            outside_weight=outside_weight,
            scale_coef=scale_coef,
        )
        pred = model(params, mask=mask)
        m = mask.to(dtype=pred.dtype)
        diff = (pred - psi) * m
        ref = psi * m
        num = torch.linalg.vector_norm(diff.flatten(1), dim=1)
        den = torch.linalg.vector_norm(ref.flatten(1), dim=1).clamp_min(1e-12)
        b = int(idx.size)
        totals["loss"] += _as_float(loss) * b
        totals["psi"] += _as_float(loss_psi) * b
        totals["scale"] += _as_float(loss_scale) * b
        totals["phys"] += _as_float(loss_phys) * b
        rel_num += float(num.div(den).sum())
        n += b
    if n == 0:
        return {k: float("nan") for k in ("loss", "psi", "scale", "phys", "rel")}
    out = {k: totals[k] / n for k in ("loss", "psi", "scale", "phys")}
    out["rel"] = rel_num / n
    return out


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")


def _default_ood_path(dataset_path: Path) -> Path | None:
    candidate = dataset_path.with_name(dataset_path.stem + "_ood" + dataset_path.suffix)
    if candidate.is_file():
        return candidate
    return None


def train_surrogate(
    dataset_path,
    model: str = "mlpcnn",
    epochs: int = 200,
    physics_weight: float = 0.0,
    seed: int = 0,
    out_dir="outputs/surrogate",
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 20,
    weight_decay: float = 1e-6,
    outside_weight: float = 0.1,
    scale_coef: float = 0.1,
    physics_warmup: int = 8,
    ood_path=None,
    evaluate: bool = True,
    q95_samples: int = 50,
    num_threads: int = 1,
    hidden: int = 256,
    latent_ch: int = 32,
    fno_width: int = 16,
    fno_modes: int = 8,
    fno_layers: int = 3,
):
    """Train a surrogate and write ``checkpoint.pt`` plus metrics under ``out_dir``.

    Returns ``(model, metrics)``. With ``evaluate=True`` the metrics include the
    test split and, when the file exists, the OOD split. Early stopping watches
    the validation objective (data loss + scale loss + physics), and patience
    starts after ``physics_warmup`` so a physics run is not frozen on the
    pre-physics weights. The best weights are restored before the checkpoint
    is written.
    """
    torch.set_num_threads(max(1, min(int(num_threads), 2)))
    set_seed(seed)
    dataset_path = Path(dataset_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_npz(dataset_path)
    params_np = _order_params(data["params"], data["param_names"]).astype(np.float32)
    split = np.asarray(data["split"])
    train_idx = _split_indices(split, 0)
    val_idx = _split_indices(split, 1)
    if train_idx.size == 0:
        raise ValueError("dataset has no training rows (split == 0)")
    mean = params_np[train_idx].mean(axis=0).astype(np.float64)
    std = params_np[train_idx].std(axis=0).astype(np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    R64 = np.asarray(data["R"], dtype=np.float64)
    Z64 = np.asarray(data["Z"], dtype=np.float64)
    dR = float(R64[1] - R64[0])
    dZ = float(Z64[1] - Z64[0])
    nr, nz = int(R64.size), int(Z64.size)
    config = {
        "model": str(model),
        "n_params": int(params_np.shape[1]),
        "nr": nr,
        "nz": nz,
        "hidden": int(hidden),
        "latent_ch": int(latent_ch),
        "fno_width": int(fno_width),
        "fno_modes": int(fno_modes),
        "fno_layers": int(fno_layers),
        "param_names": list(PARAM_NAMES),
        "outside_weight": float(outside_weight),
        "scale_coef": float(scale_coef),
        "physics_weight": float(physics_weight),
    }
    net = TrainedSurrogate(config)
    net.set_norm(mean, std, R64, Z64)
    log_mean = float(np.log(np.clip(data["psi_axis"][train_idx].astype(np.float64), 1e-8, None)).mean())
    _init_scale_head(net.net.scale_head, log_mean)
    n_param = sum(p.numel() for p in net.parameters())
    tensors = {
        "params": torch.from_numpy(np.ascontiguousarray(params_np)),
        "psi": torch.from_numpy(np.ascontiguousarray(data["psi"].astype(np.float32))),
        "J": torch.from_numpy(np.ascontiguousarray(data["J"].astype(np.float32))),
        "mask": torch.from_numpy(np.ascontiguousarray(data["mask"].astype(bool))),
        "psi_axis": torch.from_numpy(np.ascontiguousarray(data["psi_axis"].astype(np.float32))),
        "R": net.R.detach(),
    }
    opt = torch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(int(epochs), 1), eta_min=float(lr) * 0.01
    )
    rng = np.random.default_rng(int(seed) + 17)
    batch_size = max(1, min(int(batch_size), int(train_idx.size)))
    score_idx = val_idx if val_idx.size else train_idx
    before = _mean_over(
        net,
        train_idx,
        tensors,
        batch_size,
        physics_weight=0.0,
        outside_weight=outside_weight,
        scale_coef=scale_coef,
        dR=dR,
        dZ=dZ,
    )
    history = []
    best_score = float("inf")
    best_epoch = -1
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    bad = 0
    epochs_ran = 0
    t_train = time.perf_counter()
    print(
        f"train {model} N={params_np.shape[0]} train={train_idx.size} val={val_idx.size} "
        f"grid={nr}x{nz} params={n_param} physics_weight={physics_weight} lr={lr}",
        flush=True,
    )
    for epoch in range(int(epochs)):
        net.train()
        # Physics stays off during the warmup so the flux fit is established
        # before the stiff finite-difference term is added.
        if float(physics_weight) > 0.0 and epoch < int(physics_warmup):
            pw = 0.0
        else:
            pw = float(physics_weight)
        running = 0.0
        seen = 0
        for idx in _batches(train_idx, batch_size, rng):
            sl = torch.from_numpy(idx)
            opt.zero_grad(set_to_none=True)
            loss, loss_psi, loss_scale, loss_phys = _train_objective(
                net,
                tensors["params"][sl],
                tensors["psi"][sl],
                tensors["J"][sl],
                tensors["mask"][sl],
                tensors["psi_axis"][sl],
                tensors["R"],
                dR,
                dZ,
                physics_weight=pw,
                outside_weight=outside_weight,
                scale_coef=scale_coef,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
            running += _as_float(loss) * int(idx.size)
            seen += int(idx.size)
        sched.step()
        train_loss = running / max(seen, 1)
        val_stats = _mean_over(
            net,
            score_idx,
            tensors,
            batch_size,
            physics_weight=pw,
            outside_weight=outside_weight,
            scale_coef=scale_coef,
            dR=dR,
            dZ=dZ,
        )
        epochs_ran = epoch + 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_stats["loss"],
            "val_psi": val_stats["psi"],
            "val_scale": val_stats["scale"],
            "val_phys": val_stats["phys"],
            "val_rel_l2": val_stats["rel"],
            "physics_weight": pw,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} train {train_loss:.4e} val {val_stats['loss']:.4e} "
            f"psi {val_stats['psi']:.4e} scale {val_stats['scale']:.4e} "
            f"phys {val_stats['phys']:.4e} relL2 {val_stats['rel']:.4f} pw {pw:.3g}",
            flush=True,
        )
        # Patience starts once the physics weight has reached its target so a
        # physics run cannot early-stop on the warmup weights.
        warmed = epoch >= int(physics_warmup) or float(physics_weight) == 0.0
        if warmed and val_stats["loss"] < best_score:
            best_score = float(val_stats["loss"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        elif warmed:
            bad += 1
            if bad >= int(patience):
                print(f"early stop at epoch {epoch} (best {best_epoch})", flush=True)
                break
    net.load_state_dict(best_state)
    after = _mean_over(
        net,
        train_idx,
        tensors,
        batch_size,
        physics_weight=0.0,
        outside_weight=outside_weight,
        scale_coef=scale_coef,
        dR=dR,
        dZ=dZ,
    )
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "format": 1,
            "config": config,
            "state_dict": net.state_dict(),
            "grid_R": R64,
            "grid_Z": Z64,
        },
        ckpt_path,
    )
    metrics: dict = {
        "dataset": str(dataset_path),
        "model": str(model),
        "physics_weight": float(physics_weight),
        "seed": int(seed),
        "epochs_requested": int(epochs),
        "epochs_ran": int(epochs_ran),
        "best_epoch": int(best_epoch),
        "best_val_loss": None if best_epoch < 0 else float(best_score),
        "n_parameters": int(n_param),
        "train_mse_before": before["psi"],
        "train_mse_after": after["psi"],
        "train_seconds": time.perf_counter() - t_train,
        "normalization": {
            "input": "per-parameter mean/std on the training split",
            "output": "psi = shape * exp(log_psi_axis); MSE on psi/psi_axis with outside_weight, plus MSE on log psi_axis",
            "param_mean": mean.tolist(),
            "param_std": std.tolist(),
            "param_names": list(PARAM_NAMES),
            "log_psi_axis_train_mean": log_mean,
            "outside_weight": float(outside_weight),
            "scale_coef": float(scale_coef),
        },
        "hyperparameters": {
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "patience": int(patience),
            "physics_warmup": int(physics_warmup),
            "hidden": int(hidden),
            "latent_ch": int(latent_ch),
            "fno_width": int(fno_width),
            "fno_modes": int(fno_modes),
            "fno_layers": int(fno_layers),
            "grad_clip": 1.0,
            "scheduler": "cosine",
        },
        "history": history,
        "checkpoint": str(ckpt_path),
        "grid": {
            "Rmin": float(R64[0]),
            "Rmax": float(R64[-1]),
            "Zmin": float(Z64[0]),
            "Zmax": float(Z64[-1]),
            "nr": nr,
            "nz": nz,
        },
    }
    net.eval()
    if evaluate:
        if ood_path is None:
            ood_path = _default_ood_path(dataset_path)
        report = evaluate_surrogate(
            net,
            dataset_path,
            ood_path=ood_path,
            out_dir=out_dir,
            q95_samples=q95_samples,
            seed=seed,
        )
        metrics["evaluation"] = report
        _print_eval(report)
    _write_json(out_dir / "metrics.json", metrics)
    print(
        f"saved {ckpt_path} best_epoch={best_epoch} "
        f"train_mse {before['psi']:.4e} -> {after['psi']:.4e}",
        flush=True,
    )
    return net, metrics


def load_surrogate(path) -> TrainedSurrogate:
    """Load a checkpoint written by ``train_surrogate`` (normalization included)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TrainedSurrogate(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    if "grid_R" in ckpt and "grid_Z" in ckpt:
        model.grid_R = np.asarray(ckpt["grid_R"], dtype=np.float64)
        model.grid_Z = np.asarray(ckpt["grid_Z"], dtype=np.float64)
    else:
        model.grid_R = model.R.detach().cpu().numpy().astype(np.float64)
        model.grid_Z = model.Z.detach().cpu().numpy().astype(np.float64)
    model.eval()
    return model


@torch.no_grad()
def predict(model: TrainedSurrogate, params: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Physical psi, ``float32`` array of shape ``(N, nr, nz)``.

    ``params`` is ``(N, P)`` or ``(P,)`` in ``PARAM_NAMES`` order, un-normalized.
    """
    model.eval()
    raw = np.asarray(params, dtype=np.float32)
    single = raw.ndim == 1
    if single:
        raw = raw.reshape(1, -1)
    if raw.ndim != 2:
        raise ValueError(f"params must have shape (N, P), got {raw.shape}")
    nr = int(model.R.shape[0])
    nz = int(model.Z.shape[0])
    out = np.empty((raw.shape[0], nr, nz), dtype=np.float32)
    need_mask = model.model_name == "fno"
    if getattr(model, "grid_R", None) is not None:
        R_np = np.asarray(model.grid_R, dtype=np.float64)
        Z_np = np.asarray(model.grid_Z, dtype=np.float64)
    else:
        R_np = model.R.detach().cpu().numpy()
        Z_np = model.Z.detach().cpu().numpy()
    step = max(1, int(batch_size))
    for start in range(0, raw.shape[0], step):
        chunk = np.ascontiguousarray(raw[start : start + step])
        pt = torch.from_numpy(chunk)
        mask = None
        if need_mask:
            mask_np = masks_from_params(chunk, R_np, Z_np)
            mask = torch.from_numpy(np.ascontiguousarray(mask_np))
        pred = model(pt, mask=mask)
        out[start : start + step] = pred.detach().cpu().numpy().astype(np.float32, copy=False)
    return out[0] if single else out


def _summarize(values) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "median": None, "n": 0}
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "n": int(arr.size)}


def _save_figure(path: Path, R, Z, psi_true, psi_pred, indices, rel) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(10.2, 3.15 * n), constrained_layout=True)
    if n == 1:
        axes = np.asarray([axes])
    extent = [float(R[0]), float(R[-1]), float(Z[0]), float(Z[-1])]
    for row, index in enumerate(indices):
        true = np.asarray(psi_true[index], dtype=np.float64)
        pred = np.asarray(psi_pred[index], dtype=np.float64)
        err = pred - true
        vmax = max(float(true.max()), float(pred.max()), 1e-8)
        panels = (
            (axes[row, 0], true, f"true  [{index}]", "viridis", 0.0, vmax),
            (axes[row, 1], pred, f"pred  relL2={rel[index]:.3f}", "viridis", 0.0, vmax),
            (axes[row, 2], err, "pred − true", "coolwarm", None, None),
        )
        for ax, img, title, cmap, vmin, vmax_ in panels:
            kw = {"cmap": cmap, "origin": "lower", "extent": extent, "aspect": "equal"}
            if vmin is None:
                lim = max(float(np.max(np.abs(img))), 1e-8)
                kw["vmin"] = -lim
                kw["vmax"] = lim
            else:
                kw["vmin"] = vmin
                kw["vmax"] = vmax_
            im = ax.imshow(img.T, **kw)
            ax.set_title(title)
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _evaluate_split(
    model: TrainedSurrogate,
    data: dict,
    *,
    q95_samples: int,
    seed: int,
    figure_path: Path | None,
    time_inference: bool,
) -> dict:
    from eval.metrics import (
        find_magnetic_axis,
        flux_contour,
        gs_residual,
        lcfs_shape_error,
        q95,
        relative_l2,
        time_call,
    )
    from gs.solver import ProfileParams, solve_fixed_boundary

    params = _order_params(data["params"], data["param_names"]).astype(np.float32)
    psi_true = np.asarray(data["psi"], dtype=np.float32)
    J = np.asarray(data["J"], dtype=np.float64)
    mask = np.asarray(data["mask"], dtype=bool)
    psi_axis = np.asarray(data["psi_axis"], dtype=np.float64)
    R_axis = np.asarray(data["R_axis"], dtype=np.float64)
    Z_axis = np.asarray(data["Z_axis"], dtype=np.float64)
    R = np.asarray(data["R"], dtype=np.float64)
    Z = np.asarray(data["Z"], dtype=np.float64)
    grid = Grid(R, Z)
    rr = grid.RR
    n = int(params.shape[0])
    pred = predict(model, params)
    rel = np.empty(n, dtype=np.float64)
    gs = np.empty(n, dtype=np.float64)
    axis_err = np.empty(n, dtype=np.float64)
    axis_r = np.empty(n, dtype=np.float64)
    axis_z = np.empty(n, dtype=np.float64)
    lcfs = np.empty(n, dtype=np.float64)
    lcfs_fail = 0
    gs_fail = 0
    axis_fail = 0
    for i in range(n):
        rel[i] = relative_l2(pred[i], psi_true[i], mask[i])
        rhs = -float(MU0) * rr * J[i]
        try:
            gs[i] = gs_residual(pred[i], grid, rhs, mask[i])
        except ValueError:
            gs[i] = np.nan
            gs_fail += 1
        try:
            r_hat, z_hat, psi_hat = find_magnetic_axis(pred[i], grid, mask[i])
            axis_r[i] = r_hat - R_axis[i]
            axis_z[i] = z_hat - Z_axis[i]
            axis_err[i] = math.hypot(axis_r[i], axis_z[i])
        except ValueError:
            axis_r[i] = axis_z[i] = axis_err[i] = np.nan
            r_hat = z_hat = psi_hat = None
            axis_fail += 1
        if r_hat is None or not np.isfinite(psi_hat) or psi_hat == 0.0:
            lcfs[i] = np.nan
            lcfs_fail += 1
            continue
        level_true = 1e-3 * float(psi_axis[i])
        level_pred = 1e-3 * float(psi_hat)
        try:
            rt, zt = flux_contour(psi_true[i], grid, level_true, float(R_axis[i]), float(Z_axis[i]))
            rp, zp = flux_contour(pred[i], grid, level_pred, float(r_hat), float(z_hat))
            lcfs[i] = lcfs_shape_error(rp, zp, rt, zt)
        except ValueError:
            lcfs[i] = np.nan
            lcfs_fail += 1
    rng = np.random.default_rng(int(seed))
    take = min(int(q95_samples), n)
    choice = np.sort(rng.choice(n, size=take, replace=False)) if take else np.empty(0, dtype=int)
    q_abs = []
    q_rel = []
    q_fail = 0
    for i in choice:
        row = params[i]
        try:
            eq = solve_fixed_boundary(
                ShapeParams(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                ProfileParams(
                    float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])
                ),
                grid,
            )
            q_true = q95(
                psi_true[i].astype(np.float64),
                grid,
                float(psi_axis[i]),
                0.0,
                float(R_axis[i]),
                float(Z_axis[i]),
                eq.F,
            )
            r_hat, z_hat, psi_hat = find_magnetic_axis(pred[i], grid, mask[i])
            q_pred = q95(
                pred[i].astype(np.float64),
                grid,
                float(psi_hat),
                0.0,
                float(r_hat),
                float(z_hat),
                eq.F,
            )
        except (ValueError, RuntimeError):
            q_fail += 1
            continue
        q_abs.append(abs(q_pred - q_true))
        denom = abs(q_true)
        q_rel.append(abs(q_pred - q_true) / denom if denom > 0 else float("nan"))
    if figure_path is not None and n:
        finite = np.flatnonzero(np.isfinite(rel))
        if finite.size:
            order = finite[np.argsort(rel[finite])]
            picks = [order[0], order[len(order) // 2], order[(3 * len(order)) // 4], order[-1]]
            # Unique, stable, at most 4.
            seen = []
            for p in picks:
                if int(p) not in seen:
                    seen.append(int(p))
            _save_figure(figure_path, R, Z, psi_true, pred, seen, rel)
    timing = {}
    if time_inference and n:
        timing["infer_batch_s"] = time_call(lambda: predict(model, params), repeats=3)
        timing["infer_batch_s_per_sample"] = timing["infer_batch_s"] / n
        timing["infer_single_s"] = time_call(lambda: predict(model, params[:1]), repeats=5)
        row0 = params[0]

        def _solve_one():
            return solve_fixed_boundary(
                ShapeParams(float(row0[0]), float(row0[1]), float(row0[2]), float(row0[3])),
                ProfileParams(
                    float(row0[4]), float(row0[5]), float(row0[6]), float(row0[7]), float(row0[8])
                ),
                grid,
            )

        timing["solver_s"] = time_call(_solve_one, repeats=3)
        timing["speedup_batch"] = timing["solver_s"] / timing["infer_batch_s_per_sample"]
        timing["speedup_single"] = timing["solver_s"] / timing["infer_single_s"]
    return {
        "n": n,
        "rel_l2": _summarize(rel),
        "gs_residual": _summarize(gs),
        "axis_error_m": _summarize(axis_err),
        "axis_error_R_m": _summarize(axis_r),
        "axis_error_Z_m": _summarize(axis_z),
        "lcfs_error_m": _summarize(lcfs),
        "lcfs_n_failed": int(lcfs_fail),
        "gs_n_failed": int(gs_fail),
        "axis_n_failed": int(axis_fail),
        "q95_abs_error": _summarize(q_abs),
        "q95_rel_error": _summarize(q_rel),
        "q95_n_failed": int(q_fail),
        "q95_n_attempted": int(take),
        "timing": timing,
        "lcfs_level": "1e-3 * psi_axis (true axis for the reference contour, predicted axis for the prediction)",
    }


def evaluate_surrogate(
    model: TrainedSurrogate,
    dataset_path,
    ood_path=None,
    out_dir=None,
    q95_samples: int = 50,
    seed: int = 0,
) -> dict:
    """Test-split metrics and, if ``ood_path`` is a file, the same metrics on it.

    Figures are written when ``out_dir`` is set. The in-distribution file is
    scored on rows with ``split == 2``; an OOD file is scored on every row.
    """
    data = _load_npz(dataset_path)
    test_idx = _split_indices(np.asarray(data["split"]), 2)
    if test_idx.size == 0:
        raise ValueError("dataset has no test rows (split == 2)")
    test = _slice_samples(data, test_idx)
    fig = None if out_dir is None else Path(out_dir) / "samples_test.png"
    report = {
        "test": _evaluate_split(
            model,
            test,
            q95_samples=q95_samples,
            seed=seed,
            figure_path=fig,
            time_inference=True,
        )
    }
    if fig is not None:
        report["test"]["figure"] = str(fig)
    if ood_path is not None and Path(ood_path).is_file():
        ood = _load_npz(ood_path)
        fig_o = None if out_dir is None else Path(out_dir) / "samples_ood.png"
        report["ood"] = _evaluate_split(
            model,
            ood,
            q95_samples=q95_samples,
            seed=seed + 1,
            figure_path=fig_o,
            time_inference=False,
        )
        report["ood"]["dataset"] = str(ood_path)
        if fig_o is not None:
            report["ood"]["figure"] = str(fig_o)
    else:
        report["ood"] = None
    return report


def _print_eval(report: dict) -> None:
    for name in ("test", "ood"):
        block = report.get(name)
        if not block:
            continue
        rel = block["rel_l2"]
        gs = block["gs_residual"]
        axis = block["axis_error_m"]
        lcfs = block["lcfs_error_m"]
        qrel = block["q95_rel_error"]
        qabs = block["q95_abs_error"]
        print(
            f"  {name}: relL2 mean={rel['mean']:.4f} median={rel['median']:.4f} "
            f"gs mean={gs['mean']:.4f} median={gs['median']:.4f} "
            f"axis mean={axis['mean']:.4e} m lcfs mean={lcfs['mean']:.4e} m "
            f"q95 abs={qabs['mean']} rel={qrel['mean']} "
            f"(lcfs fail {block['lcfs_n_failed']}, q95 fail {block['q95_n_failed']})",
            flush=True,
        )
        timing = block.get("timing") or {}
        if timing:
            print(
                f"  timing: batch {timing['infer_batch_s_per_sample']:.4e} s/sample, "
                f"single {timing['infer_single_s']:.4e} s, solver {timing['solver_s']:.4e} s, "
                f"speedup batch {timing['speedup_batch']:.1f}x single {timing['speedup_single']:.1f}x",
                flush=True,
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a params -> psi Grad-Shafranov surrogate")
    parser.add_argument("--data", required=True, help="npz dataset (fixed-boundary contract)")
    parser.add_argument("--ood", default=None, help="optional OOD npz; default is <data>_ood.npz if present")
    parser.add_argument("--model", default="mlpcnn", choices=["mlpcnn", "fno"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--physics-weight", type=float, default=0.0)
    parser.add_argument("--out", default="outputs/surrogate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--physics-warmup", type=int, default=8)
    parser.add_argument("--q95-samples", type=int, default=50)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args(argv)
    train_surrogate(
        args.data,
        model=args.model,
        epochs=args.epochs,
        physics_weight=args.physics_weight,
        seed=args.seed,
        out_dir=args.out,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        physics_warmup=args.physics_warmup,
        ood_path=args.ood,
        evaluate=not args.no_eval,
        q95_samples=args.q95_samples,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
