"""Generate equilibrium datasets for training the ML models.

Dataset file format (``.npz``), shared by all models:

    R, Z           (nr,), (nz,)       shared uniform grid in metres
    psi            (N, nr, nz) f32    poloidal flux, Wb/rad (0 on the plasma boundary)
    J              (N, nr, nz) f32    toroidal current density, A/m^2
    mask           (N, nr, nz) bool   True inside the plasma
    params         (N, P) f32         equilibrium parameters, columns named by param_names
    param_names    (P,) str           ["R0", "a", "kappa", "delta", "Ip", "beta0", "alpha", "gamma", "B0"]
    psi_axis, R_axis, Z_axis  (N,) f32
    converged      (N,) bool
    split          (N,) i1            0 = train, 1 = val, 2 = test

Free-boundary (FreeGS) datasets use the same keys and add ``coil_names`` and
``coil_currents`` (N, n_coils); there ``mask`` is the region inside the separatrix/LCFS
and psi is shifted so that psi = 0 on the LCFS.

CLI: ``python -m data.generate --n 5000 --nr 65 --nz 65 --out data/fixed_65.npz``
"""
from __future__ import annotations

PARAM_NAMES = ["R0", "a", "kappa", "delta", "Ip", "beta0", "alpha", "gamma", "B0"]


def load_dataset(path: str) -> dict:
    """Load a dataset file into a dict of numpy arrays."""
    raise NotImplementedError
