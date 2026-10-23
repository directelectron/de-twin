"""Axial aberrations in CEOS notation, shared by the TEM CTF and the coherent STEM probe.

The aberration function (phase error of a beam at complex scattering angle
``w = lambda * (kx + i*ky)``) is::

    chi(w) = (2 pi / lambda) * Re{ 1/2 wb w  C1 + 1/2 wb^2 A1
                                 + wb^2 w B2 + 1/3 wb^3 A2
                                 + 1/4 (wb w)^2 C3 + wb^3 w S3 + 1/4 wb^4 A3
                                 + wb^3 w^2 B4 + wb^4 w D4 + 1/5 wb^5 A4
                                 + 1/6 (wb w)^3 C5 + wb^4 w^2 S5 + wb^5 w R5 + 1/6 wb^6 A5 }

with ``wb`` the complex conjugate of ``w``. Round aberrations (C1, C3, C5) are real;
the others are complex, magnitude and azimuth as used by CEOS software.
Units: every coefficient is in **nanometres** (C3 = 1.2 mm is 1.2e6 nm).

Sign convention: C1 is the twin's defocus (negative = underfocus), so
``chi = pi * lambda * C1 * k^2 + pi/2 * C3 * lambda^3 * k^4 + ...``, which is
the CTF convention already used by the TEM renderer.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

# name -> (power of wb, power of w, prefactor, round?)
TERMS: dict[str, tuple[int, int, float, bool]] = {
    "C1": (1, 1, 1 / 2, True),
    "A1": (2, 0, 1 / 2, False),
    "B2": (2, 1, 1.0, False),
    "A2": (3, 0, 1 / 3, False),
    "C3": (2, 2, 1 / 4, True),
    "S3": (3, 1, 1.0, False),
    "A3": (4, 0, 1 / 4, False),
    "B4": (3, 2, 1.0, False),
    "D4": (4, 1, 1.0, False),
    "A4": (5, 0, 1 / 5, False),
    "C5": (3, 3, 1 / 6, True),
    "S5": (4, 2, 1.0, False),
    "R5": (5, 1, 1.0, False),
    "A5": (6, 0, 1 / 6, False),
}
ORDER = {n: TERMS[n][0] + TERMS[n][1] - 1 for n in TERMS}  # wave-aberration order (C1 -> 1, C3 -> 3)
NAMES = tuple(TERMS)


@dataclass
class Aberrations:
    """A set of aberration coefficients (complex, nm). Missing names are zero."""

    coeffs: dict[str, complex] = field(default_factory=dict)

    def __post_init__(self):
        clean = {}
        for k, v in dict(self.coeffs).items():
            if k not in TERMS:
                raise KeyError(f"unknown aberration {k!r}; known: {', '.join(NAMES)}")
            v = complex(v)
            if TERMS[k][3]:
                v = complex(v.real, 0.0)
            if v != 0:
                clean[k] = v
        self.coeffs = clean

    # ---------------------------------------------------------- construction
    @classmethod
    def from_polar(cls, **mag_angle) -> Aberrations:
        """``Aberrations.from_polar(A1=(5.0, 30.0), C3=1.2e6)``: magnitude nm, azimuth deg."""
        out = {}
        for k, v in mag_angle.items():
            if isinstance(v, (tuple, list)):
                mag, ang = v
                n = TERMS[k][0] - TERMS[k][1]  # symmetry order of the term
                phi = math.radians(ang) * max(abs(n), 1)
                out[k] = mag * complex(math.cos(phi), math.sin(phi))
            else:
                out[k] = complex(v)
        return cls(out)

    def copy(self) -> Aberrations:
        return Aberrations(dict(self.coeffs))

    def __getitem__(self, name: str) -> complex:
        if name not in TERMS:
            raise KeyError(name)
        return self.coeffs.get(name, 0j)

    def __add__(self, other: Aberrations | Mapping[str, complex]) -> Aberrations:
        o = other.coeffs if isinstance(other, Aberrations) else dict(other)
        out = dict(self.coeffs)
        for k, v in o.items():
            out[k] = out.get(k, 0j) + complex(v)
        return Aberrations(out)

    def __sub__(self, other: Aberrations | Mapping[str, complex]) -> Aberrations:
        o = other.coeffs if isinstance(other, Aberrations) else dict(other)
        return self + {k: -complex(v) for k, v in o.items()}

    def scaled(self, factor: float) -> Aberrations:
        return Aberrations({k: v * factor for k, v in self.coeffs.items()})

    def only(self, names: Iterable[str]) -> Aberrations:
        names = set(names)
        return Aberrations({k: v for k, v in self.coeffs.items() if k in names})

    def polar(self) -> dict[str, tuple[float, float]]:
        """name -> (magnitude nm, azimuth deg) in CEOS convention."""
        out = {}
        for k, v in self.coeffs.items():
            n = max(abs(TERMS[k][0] - TERMS[k][1]), 1)
            out[k] = (abs(v), math.degrees(math.atan2(v.imag, v.real)) / n if not TERMS[k][3] else 0.0)
        return out

    def key(self) -> tuple:
        """Hashable, rounded representation (cache keys)."""
        return tuple(sorted((k, round(v.real, 6), round(v.imag, 6)) for k, v in self.coeffs.items()))

    def __bool__(self) -> bool:
        return bool(self.coeffs)

    # --------------------------------------------------------------- physics
    def chi(self, kx, ky, wavelength_nm: float) -> np.ndarray:
        """Aberration phase chi(k) in radians; kx, ky in 1/nm (broadcastable)."""
        lam = float(wavelength_nm)
        w = lam * (np.asarray(kx, np.float64) + 1j * np.asarray(ky, np.float64))
        wb = np.conj(w)
        acc = np.zeros(np.broadcast(w, wb).shape, np.complex128)
        for name, c in self.coeffs.items():
            p, q, pre, _ = TERMS[name]
            acc += pre * c * wb**p * w**q
        return (2.0 * math.pi / lam) * acc.real

    def gradient(self, kx, ky, wavelength_nm: float) -> tuple[np.ndarray, np.ndarray]:
        """(d chi/d kx, d chi/d ky) in rad*nm; used for spatial-coherence envelopes and
        for the probe/image shift produced by beam tilt."""
        lam = float(wavelength_nm)
        w = lam * (np.asarray(kx, np.float64) + 1j * np.asarray(ky, np.float64))
        wb = np.conj(w)
        gx = np.zeros(np.broadcast(w, wb).shape, np.complex128)
        gy = np.zeros_like(gx)
        for name, c in self.coeffs.items():
            p, q, pre, _ = TERMS[name]
            a = p * wb ** (p - 1) * w**q if p else 0.0
            b = q * wb**p * w ** (q - 1) if q else 0.0
            gx += pre * c * lam * (a + b)
            gy += pre * c * lam * (-1j * a + 1j * b)
        s = 2.0 * math.pi / lam
        return s * gx.real, s * gy.real

    def phase_error_at(self, angle_mrad: float, wavelength_nm: float, n: int = 72) -> float:
        """Largest ``abs(chi)`` (rad) on a ring of the given semi-angle (e.g. the pi/4 criterion)."""
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        k = angle_mrad * 1e-3 / wavelength_nm
        c = self.chi(k * np.cos(th), k * np.sin(th), wavelength_nm)
        c0 = float(self.chi(0.0, 0.0, wavelength_nm))
        return float(np.max(np.abs(c - c0)))

    def flat_angle_mrad(
        self, wavelength_nm: float, limit_rad: float = math.pi / 4, max_mrad: float = 60.0
    ) -> float:
        """Largest aperture semi-angle whose phase error stays below ``limit_rad``
        (what a corrector tableau quotes as the 'pi/4 angle')."""
        lo, hi = 0.0, max_mrad
        if self.phase_error_at(hi, wavelength_nm) < limit_rad:
            return hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.phase_error_at(mid, wavelength_nm) < limit_rad:
                lo = mid
            else:
                hi = mid
        return lo


def uncorrected(cs_mm: float = 1.2) -> Aberrations:
    """A typical uncorrected 200 kV objective: C3 only (C5 negligible)."""
    return Aberrations({"C3": cs_mm * 1e6})
