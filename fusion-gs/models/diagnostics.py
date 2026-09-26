"""Synthetic magnetic diagnostics for fixed- and free-boundary equilibria.

Poloidal flux per radian at ``(R, Z)`` due to a unit toroidal filament at
``(Rc, Zc)`` is the standard axisymmetric Green's function

    psi = mu0 / (2 pi) * sqrt(R Rc) * ((2 - k^2) K(k^2) - 2 E(k^2)) / k

    k^2 = 4 R Rc / ((R + Rc)^2 + (Z - Zc)^2)

with ``K`` and ``E`` the complete elliptic integrals (``scipy.special.ellipk``
and ``ellipe``, both called with parameter ``k^2``). This matches
``freegs.machine.Greens`` including the clip of ``k^2`` away from 0 and 1.

``B_R = -(1/R) dpsi/dZ`` and ``B_Z = (1/R) dpsi/dR`` are the same central
differences FreeGS uses (step ``1e-3`` m), so ``greens_br`` / ``greens_bz``
agree with ``freegs.machine.GreensBr`` / ``GreensBz`` to roundoff. That step
is a few parts in 10^5 relative to the analytic derivative at typical
probe-to-filament separations; it is far below the 1% sensor noise used in
training.

Fixed-boundary equilibria in this project have ``psi = 0`` outside the plasma,
so the external field is computed from the plasma current alone. A
``SensorSet`` places pickup probes, flux loops and one Rogowski coil on a
vessel contour and precomputes the response matrix ``G`` with

    signals = G @ (J.ravel() * dR * dZ).

Free-boundary signals add coil contributions from ``freegs.machine.TestTokamak``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path
from scipy.special import ellipe, ellipk

from gs.solver import MU0, Grid

# Same step as freegs.gradshafranov.GreensBr / GreensBz.
GREENS_FD_EPS = 1e-3
_K2_EPS = 1e-10

# TestTokamak wall (freegs.machine.TestTokamak), CCW from the inboard lower corner.
FREEGS_WALL_R = np.array([0.75, 0.75, 1.5, 1.8, 1.8, 1.5], dtype=float)
FREEGS_WALL_Z = np.array([-0.85, 0.85, 0.85, 0.25, -0.25, -0.85], dtype=float)


def greens_psi(Rc, Zc, R, Z) -> np.ndarray:
    """Poloidal flux per radian at (R, Z) due to a unit current at (Rc, Zc).

    Broadcasts like NumPy. A fully scalar input returns a Python float.
    """
    Rc_a = np.asarray(Rc, dtype=float)
    Zc_a = np.asarray(Zc, dtype=float)
    R_a = np.asarray(R, dtype=float)
    Z_a = np.asarray(Z, dtype=float)
    scalar = (
        Rc_a.ndim == 0 and Zc_a.ndim == 0 and R_a.ndim == 0 and Z_a.ndim == 0
    )
    k2 = 4.0 * R_a * Rc_a / ((R_a + Rc_a) ** 2 + (Z_a - Zc_a) ** 2)
    k2 = np.clip(k2, _K2_EPS, 1.0 - _K2_EPS)
    k = np.sqrt(k2)
    out = (
        (MU0 / (2.0 * np.pi))
        * np.sqrt(R_a * Rc_a)
        * ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2))
        / k
    )
    if scalar:
        return float(out)
    return out


def greens_br(Rc, Zc, R, Z, eps: float = GREENS_FD_EPS) -> np.ndarray:
    """Radial field at (R, Z) due to a unit filament at (Rc, Zc).

    ``B_R = -(1/R) dpsi/dZ``, central difference with the FreeGS step.
    """
    R_a = np.asarray(R, dtype=float)
    Z_a = np.asarray(Z, dtype=float)
    scalar = (
        np.ndim(Rc) == 0 and np.ndim(Zc) == 0 and R_a.ndim == 0 and Z_a.ndim == 0
    )
    out = (greens_psi(Rc, Zc, R_a, Z_a - eps) - greens_psi(Rc, Zc, R_a, Z_a + eps)) / (
        2.0 * eps * R_a
    )
    if scalar:
        return float(out)
    return out


def greens_bz(Rc, Zc, R, Z, eps: float = GREENS_FD_EPS) -> np.ndarray:
    """Vertical field at (R, Z) due to a unit filament at (Rc, Zc).

    ``B_Z = (1/R) dpsi/dR``, central difference with the FreeGS step.
    The denominator uses the unshifted ``R``, matching FreeGS.
    """
    R_a = np.asarray(R, dtype=float)
    Z_a = np.asarray(Z, dtype=float)
    scalar = (
        np.ndim(Rc) == 0 and np.ndim(Zc) == 0 and R_a.ndim == 0 and Z_a.ndim == 0
    )
    out = (greens_psi(Rc, Zc, R_a + eps, Z_a) - greens_psi(Rc, Zc, R_a - eps, Z_a)) / (
        2.0 * eps * R_a
    )
    if scalar:
        return float(out)
    return out


def _resample_closed(R: np.ndarray, Z: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points spaced uniformly in arc length along a closed polyline."""
    R = np.asarray(R, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    dR = np.diff(R, append=R[0])
    dZ = np.diff(Z, append=Z[0])
    seg = np.hypot(dR, dZ)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0.0:
        raise ValueError("vessel contour has zero length")
    query = np.linspace(0.0, total, int(n), endpoint=False)
    return (
        np.interp(query, s, np.append(R, R[0])),
        np.interp(query, s, np.append(Z, Z[0])),
    )


def _tangents(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Unit tangents of a closed loop from a centred difference, shape (n, 2)."""
    dR = np.roll(R, -1) - np.roll(R, 1)
    dZ = np.roll(Z, -1) - np.roll(Z, 1)
    nrm = np.hypot(dR, dZ)
    nrm = np.maximum(nrm, 1e-30)
    return np.column_stack([dR / nrm, dZ / nrm])


def _outward_normals(R: np.ndarray, Z: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Unit normals pointing out of the polygon, shape (n, 2)."""
    area = 0.5 * float(np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z))
    tR = tangent[:, 0]
    tZ = tangent[:, 1]
    if area >= 0.0:
        nR, nZ = tZ, -tR
    else:
        nR, nZ = -tZ, tR
    cR = float(np.mean(R))
    cZ = float(np.mean(Z))
    if float(np.mean(nR * (R - cR) + nZ * (Z - cZ))) < 0.0:
        nR = -nR
        nZ = -nZ
    return np.column_stack([nR, nZ])


def dshape_vessel(
    rmin: float,
    rmax: float,
    zmin: float,
    zmax: float,
    n: int = 720,
    delta: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Up-down symmetric D-shaped vessel inset inside a rectangular box.

    The fractional margins are chosen so that on the shared fixed-boundary box
    ``R in [0.80, 2.76]``, ``Z in [-1.90, 1.90]`` the contour stays inside the
    grid and strictly outside every plasma in ``fixed_65.npz`` and
    ``fixed_65_ood.npz`` (Miller D with ``delta = 0.2``). The same fractions
    are used on any other uniform box, including the 33x33 test grid.
    """
    width = float(rmax - rmin)
    height = float(zmax - zmin)
    margin_r = 0.005102 * width
    margin_z = 0.015 * height
    r_in = float(rmin) + margin_r
    r_out = float(rmax) - margin_r
    z_lo = float(zmin) + margin_z
    z_hi = float(zmax) - margin_z
    a = 0.5 * (r_out - r_in)
    R0 = 0.5 * (r_out + r_in)
    z0 = 0.5 * (z_lo + z_hi)
    b = 0.5 * (z_hi - z_lo)
    if a <= 0.0 or b <= 0.0:
        raise ValueError("grid box is too small for a vessel contour")
    kappa = b / a
    tau = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
    alpha = float(np.arcsin(delta))
    R = R0 + a * np.cos(tau + alpha * np.sin(tau))
    Z = z0 + kappa * a * np.sin(tau)
    return R, Z


def freegs_wall_vessel(n: int = 480) -> tuple[np.ndarray, np.ndarray]:
    """TestTokamak wall, densified. PF coils of that machine lie outside it."""
    R = FREEGS_WALL_R
    Z = FREEGS_WALL_Z
    parts_r = []
    parts_z = []
    per = max(8, int(n) // R.size)
    for i in range(R.size):
        r0, z0 = float(R[i]), float(Z[i])
        r1, z1 = float(R[(i + 1) % R.size]), float(Z[(i + 1) % Z.size])
        t = np.linspace(0.0, 1.0, per, endpoint=False)
        parts_r.append(r0 + t * (r1 - r0))
        parts_z.append(z0 + t * (z1 - z0))
    return np.concatenate(parts_r), np.concatenate(parts_z)


def channel_rms(signals: np.ndarray) -> np.ndarray:
    """Per-channel RMS of ``signals`` with shape (N, n_sensors)."""
    sig = np.asarray(signals, dtype=float)
    if sig.ndim != 2:
        raise ValueError(f"signals must have shape (N, n_sensors), got {sig.shape}")
    return np.sqrt(np.mean(sig * sig, axis=0))


def add_noise(signals: np.ndarray, noise_std: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with the given per-channel standard deviation."""
    sig = np.asarray(signals, dtype=float)
    std = np.asarray(noise_std, dtype=float).reshape(1, -1)
    if std.shape[1] != sig.shape[1]:
        raise ValueError("noise_std length does not match n_sensors")
    return sig + rng.normal(0.0, 1.0, size=sig.shape) * std


@dataclass
class SensorSet:
    """Pickup probes, flux loops and one Rogowski coil on a closed vessel.

    Channel order of ``measure`` / ``response_matrix``:

    * ``Bt`` (n_probes): tangential poloidal field ``B · t``
    * ``Bn`` (n_probes): outward normal field ``B · n``, if ``include_normal``
    * ``flux`` (n_flux): poloidal flux per radian
    * ``rogowski`` (1): total toroidal current inside the vessel

    ``positions`` has shape ``(n_probes + n_flux, 2)`` and lists probe
    locations first, then flux loops. The Rogowski coil is the vessel contour
    itself (``vessel_R``, ``vessel_Z``), not a point.
    """

    probe_R: np.ndarray
    probe_Z: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    flux_R: np.ndarray
    flux_Z: np.ndarray
    vessel_R: np.ndarray
    vessel_Z: np.ndarray
    include_normal: bool = True
    geometry: str = "dshape"
    _response: np.ndarray | None = field(default=None, repr=False)
    _response_key: tuple | None = field(default=None, repr=False)
    _coil_matrix: np.ndarray | None = field(default=None, repr=False)
    coil_names: tuple[str, ...] = ()

    @property
    def n_probes(self) -> int:
        return int(self.probe_R.size)

    @property
    def n_flux(self) -> int:
        return int(self.flux_R.size)

    @property
    def n_sensors(self) -> int:
        n = self.n_probes + self.n_flux + 1
        if self.include_normal:
            n += self.n_probes
        return n

    @property
    def positions(self) -> np.ndarray:
        """``(n_probes + n_flux, 2)`` array of ``(R, Z)`` for point sensors."""
        return np.column_stack(
            [
                np.concatenate([self.probe_R, self.flux_R]),
                np.concatenate([self.probe_Z, self.flux_Z]),
            ]
        )

    def channel_names(self) -> list[str]:
        names = [f"Bt{i:02d}" for i in range(self.n_probes)]
        if self.include_normal:
            names += [f"Bn{i:02d}" for i in range(self.n_probes)]
        names += [f"flux{i:02d}" for i in range(self.n_flux)]
        names.append("rogowski")
        return names

    @classmethod
    def _from_vessel(
        cls,
        vessel_R: np.ndarray,
        vessel_Z: np.ndarray,
        n_probes: int,
        n_flux: int,
        include_normal: bool,
        geometry: str,
    ) -> "SensorSet":
        if n_probes < 4 or n_flux < 4:
            raise ValueError("need at least 4 probes and 4 flux loops")
        probe_R, probe_Z = _resample_closed(vessel_R, vessel_Z, int(n_probes))
        # Half-step offset so flux loops do not sit on the probes.
        flux_src_R, flux_src_Z = _resample_closed(vessel_R, vessel_Z, int(n_flux) * 8)
        shift = max(1, flux_src_R.size // (2 * int(n_flux)))
        flux_R, flux_Z = _resample_closed(
            np.roll(flux_src_R, shift), np.roll(flux_src_Z, shift), int(n_flux)
        )
        tangent = _tangents(probe_R, probe_Z)
        normal = _outward_normals(probe_R, probe_Z, tangent)
        return cls(
            probe_R=np.asarray(probe_R, dtype=float),
            probe_Z=np.asarray(probe_Z, dtype=float),
            tangent=np.asarray(tangent, dtype=float),
            normal=np.asarray(normal, dtype=float),
            flux_R=np.asarray(flux_R, dtype=float),
            flux_Z=np.asarray(flux_Z, dtype=float),
            vessel_R=np.asarray(vessel_R, dtype=float),
            vessel_Z=np.asarray(vessel_Z, dtype=float),
            include_normal=bool(include_normal),
            geometry=str(geometry),
        )

    @classmethod
    def for_grid(
        cls,
        grid: Grid,
        n_probes: int = 40,
        n_flux: int = 20,
        include_normal: bool = True,
        geometry: str = "dshape",
    ) -> "SensorSet":
        """Build a diagnostic set around ``grid``.

        ``geometry="dshape"`` is the fixed-boundary vessel. ``geometry="freegs"``
        uses the TestTokamak wall, which leaves the PF coils outside the Rogowski.
        Free-boundary datasets whose plasmas cross that wall should use
        ``around_plasmas`` instead.
        """
        if geometry == "freegs":
            vessel_R, vessel_Z = freegs_wall_vessel()
        elif geometry == "dshape":
            vessel_R, vessel_Z = dshape_vessel(
                float(grid.R[0]),
                float(grid.R[-1]),
                float(grid.Z[0]),
                float(grid.Z[-1]),
            )
        else:
            raise ValueError(f"unknown vessel geometry {geometry!r}")
        return cls._from_vessel(vessel_R, vessel_Z, n_probes, n_flux, include_normal, geometry)

    @classmethod
    def around_plasmas(
        cls,
        grid: Grid,
        mask: np.ndarray,
        n_probes: int = 40,
        n_flux: int = 20,
        include_normal: bool = True,
        margin_cells: int = 1,
        geometry: str = "freegs",
    ) -> "SensorSet":
        """Vessel on the contour of ``mask`` dilated by ``margin_cells``.

        Used for free-boundary equilibria that are not contained by the
        TestTokamak wall. One cell of margin keeps every probe off the plasma
        current while leaving the PF coils outside the Rogowski on the
        FreeGS scans in this project.
        """
        import contourpy
        from scipy import ndimage

        mask_a = np.asarray(mask, dtype=bool)
        if mask_a.shape != (grid.nr, grid.nz):
            raise ValueError(f"mask shape {mask_a.shape} != {(grid.nr, grid.nz)}")
        if mask_a.ndim != 2:
            raise ValueError("mask must be a single (nr, nz) union")
        dilated = ndimage.binary_dilation(mask_a, iterations=int(margin_cells))
        if not np.any(dilated):
            raise ValueError("mask is empty")
        dR = float(grid.dR)
        dZ = float(grid.dZ)
        padded = np.pad(dilated.astype(float), 1)
        R = np.asarray(grid.R, dtype=float)
        Z = np.asarray(grid.Z, dtype=float)
        Rext = np.concatenate([[R[0] - dR], R, [R[-1] + dR]])
        Zext = np.concatenate([[Z[0] - dZ], Z, [Z[-1] + dZ]])
        generator = contourpy.contour_generator(
            x=Rext, y=Zext, z=np.ascontiguousarray(padded.T)
        )
        lines = generator.lines(0.5)
        if not lines:
            raise RuntimeError("dilation has no closed contour")

        def _area(pts: np.ndarray) -> float:
            return float(
                0.5
                * abs(
                    np.dot(pts[:, 0], np.roll(pts[:, 1], -1))
                    - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))
                )
            )

        line = max((np.asarray(item, dtype=float) for item in lines), key=_area)
        if line.shape[0] < 8:
            raise RuntimeError("vessel contour is too short")
        gap = float(np.hypot(line[0, 0] - line[-1, 0], line[0, 1] - line[-1, 1]))
        if gap <= 1e-8 * max(dR, dZ, 1e-6):
            line = line[:-1]
        return cls._from_vessel(
            line[:, 0], line[:, 1], n_probes, n_flux, include_normal, geometry
        )

    def to_dict(self) -> dict:
        return {
            "probe_R": np.asarray(self.probe_R, dtype=np.float64),
            "probe_Z": np.asarray(self.probe_Z, dtype=np.float64),
            "tangent": np.asarray(self.tangent, dtype=np.float64),
            "normal": np.asarray(self.normal, dtype=np.float64),
            "flux_R": np.asarray(self.flux_R, dtype=np.float64),
            "flux_Z": np.asarray(self.flux_Z, dtype=np.float64),
            "vessel_R": np.asarray(self.vessel_R, dtype=np.float64),
            "vessel_Z": np.asarray(self.vessel_Z, dtype=np.float64),
            "include_normal": bool(self.include_normal),
            "geometry": str(self.geometry),
            "coil_names": np.asarray(self.coil_names, dtype="U32"),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SensorSet":
        names = payload.get("coil_names", ())
        coil_names = tuple(str(x) for x in np.asarray(names).tolist())
        sensors = cls(
            probe_R=np.asarray(payload["probe_R"], dtype=float),
            probe_Z=np.asarray(payload["probe_Z"], dtype=float),
            tangent=np.asarray(payload["tangent"], dtype=float),
            normal=np.asarray(payload["normal"], dtype=float),
            flux_R=np.asarray(payload["flux_R"], dtype=float),
            flux_Z=np.asarray(payload["flux_Z"], dtype=float),
            vessel_R=np.asarray(payload["vessel_R"], dtype=float),
            vessel_Z=np.asarray(payload["vessel_Z"], dtype=float),
            include_normal=bool(payload["include_normal"]),
            geometry=str(payload.get("geometry", "dshape")),
            coil_names=coil_names,
        )
        return sensors

    def _vessel_path(self) -> Path:
        R = np.append(self.vessel_R, self.vessel_R[0])
        Z = np.append(self.vessel_Z, self.vessel_Z[0])
        return Path(np.column_stack([R, Z]))

    def inside_mask(self, grid: Grid) -> np.ndarray:
        """Boolean mask of grid nodes strictly inside the vessel, shape (nr, nz)."""
        pts = np.column_stack([grid.RR.ravel(), grid.ZZ.ravel()])
        return self._vessel_path().contains_points(pts).reshape(grid.nr, grid.nz)

    def _cache_key(self, grid: Grid) -> tuple:
        return (
            int(grid.nr),
            int(grid.nz),
            float(grid.R[0]),
            float(grid.R[-1]),
            float(grid.Z[0]),
            float(grid.Z[-1]),
            float(grid.dR),
            float(grid.dZ),
        )

    def response_matrix(self, grid: Grid) -> np.ndarray:
        """Plasma response ``G`` with shape ``(n_sensors, nr * nz)``.

        Column ``i * nz + j`` is the sensor vector of a unit-current filament
        at ``(R[i], Z[j])``. ``measure`` multiplies by ``J * dR * dZ``.
        The Rogowski row is 1 on nodes inside the vessel and 0 outside.
        """
        key = self._cache_key(grid)
        if self._response is not None and self._response_key == key:
            return self._response
        nr, nz = grid.nr, grid.nz
        Rg = np.broadcast_to(np.asarray(grid.R, dtype=float)[:, None], (nr, nz)).ravel()
        Zg = np.broadcast_to(np.asarray(grid.Z, dtype=float)[None, :], (nr, nz)).ravel()
        Br = greens_br(
            Rg[None, :], Zg[None, :], self.probe_R[:, None], self.probe_Z[:, None]
        )
        Bz = greens_bz(
            Rg[None, :], Zg[None, :], self.probe_R[:, None], self.probe_Z[:, None]
        )
        blocks = [Br * self.tangent[:, 0:1] + Bz * self.tangent[:, 1:2]]
        if self.include_normal:
            blocks.append(Br * self.normal[:, 0:1] + Bz * self.normal[:, 1:2])
        blocks.append(
            greens_psi(
                Rg[None, :], Zg[None, :], self.flux_R[:, None], self.flux_Z[:, None]
            )
        )
        blocks.append(self.inside_mask(grid).astype(float).ravel()[None, :])
        matrix = np.vstack(blocks)
        if matrix.shape != (self.n_sensors, nr * nz):
            raise RuntimeError(
                f"response matrix shape {matrix.shape} != {(self.n_sensors, nr * nz)}"
            )
        self._response = matrix
        self._response_key = key
        return matrix

    def coil_response_matrix(self, coil_names: list[str] | tuple[str, ...]) -> np.ndarray:
        """Signal per ampere of each TestTokamak coil, shape ``(n_sensors, n_coils)``.

        Point and shaped coils use ``controlPsi`` / ``controlBr`` / ``controlBz``.
        The Rogowski entry is the fraction of that coil's cross-section inside
        the vessel (0 for the TestTokamak PF coils, which sit outside the wall).
        """
        names = tuple(str(x) for x in coil_names)
        if (
            self._coil_matrix is not None
            and self.coil_names == names
            and self._coil_matrix.shape == (self.n_sensors, len(names))
        ):
            return self._coil_matrix
        import freegs.machine as fm
        from shapely.geometry import Polygon

        tokamak = fm.TestTokamak()
        by_name = {label: coil for label, coil in tokamak.coils}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise KeyError(f"TestTokamak has no coils {missing}")
        polygon = Polygon(np.column_stack([self.vessel_R, self.vessel_Z]))
        matrix = np.zeros((self.n_sensors, len(names)), dtype=float)
        for ic, name in enumerate(names):
            coil = by_name[name]
            br = np.asarray(coil.controlBr(self.probe_R, self.probe_Z), dtype=float)
            bz = np.asarray(coil.controlBz(self.probe_R, self.probe_Z), dtype=float)
            parts = [br * self.tangent[:, 0] + bz * self.tangent[:, 1]]
            if self.include_normal:
                parts.append(br * self.normal[:, 0] + bz * self.normal[:, 1])
            parts.append(np.asarray(coil.controlPsi(self.flux_R, self.flux_Z), dtype=float))
            parts.append(np.array([float(coil.inShape(polygon))], dtype=float))
            matrix[:, ic] = np.concatenate(parts)
        self._coil_matrix = matrix
        self.coil_names = names
        return matrix

    def measure(
        self,
        J: np.ndarray,
        grid: Grid,
        coil_currents: np.ndarray | None = None,
    ) -> np.ndarray:
        """Synthetic signals, shape ``(N, n_sensors)``.

        ``J`` is toroidal current density in A/m^2 with shape ``(nr, nz)`` or
        ``(N, nr, nz)``. The plasma part is ``G @ (J.ravel() * dR * dZ)``.
        ``coil_currents`` with shape ``(N, n_coils)`` or ``(n_coils,)`` adds the
        TestTokamak coil field. Coil order must match ``coil_response_matrix``.
        """
        J_a = np.asarray(J, dtype=float)
        single = J_a.ndim == 2
        if single:
            J_a = J_a[None, ...]
        if J_a.ndim != 3 or J_a.shape[1:] != (grid.nr, grid.nz):
            raise ValueError(
                f"J shape {np.asarray(J).shape} is not (nr, nz) or (N, nr, nz) "
                f"with nr, nz = {grid.nr}, {grid.nz}"
            )
        G = self.response_matrix(grid)
        dA = float(grid.dR) * float(grid.dZ)
        signals = (J_a.reshape(J_a.shape[0], -1) * dA) @ G.T
        if coil_currents is not None:
            currents = np.asarray(coil_currents, dtype=float)
            if currents.ndim == 1:
                currents = np.broadcast_to(currents, (J_a.shape[0], currents.shape[0])).copy()
            if currents.shape[0] != J_a.shape[0]:
                raise ValueError(
                    f"coil_currents has {currents.shape[0]} rows for {J_a.shape[0]} equilibria"
                )
            C = self.coil_response_matrix(self.coil_names if self.coil_names else _default_coil_names(currents.shape[1]))
            if currents.shape[1] != C.shape[1]:
                raise ValueError(
                    f"coil_currents has {currents.shape[1]} coils, response has {C.shape[1]}"
                )
            signals = signals + currents @ C.T
        if single:
            return signals
        return signals


def _default_coil_names(n_coils: int) -> tuple[str, ...]:
    import freegs.machine as fm

    tokamak = fm.TestTokamak()
    names = tuple(label for label, _ in tokamak.coils)
    if len(names) != n_coils:
        raise ValueError(
            f"expected {len(names)} TestTokamak coils, got {n_coils} currents"
        )
    return names
