"""CEOS conventions: the aberration list and string formats, the Zemlin tableau measurement model, and drift."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..optics.aberrations import TERMS, Aberrations
from ..optics.physics import electron_wavelength_nm

# ---------------------------------------------------------------------------
# CEOS constants
# ---------------------------------------------------------------------------

#: The CEOS reference server's aberration list, in the order it reports (and
#: truncates at ``maxFit``). We is image shift (m), WD the beam tilt (rad).
CEOS_ORDER = ("C1", "A1", "A2", "B2", "C3", "A3", "S3", "A4", "D4", "B4", "C5", "A5", "R5", "S5", "We", "WD")
FIT_NAMES = tuple(n for n in CEOS_ORDER if n in TERMS)
#: tabType -> (default outer angle mrad, default maxFit)
TABLEAU_TYPES = {"Fast": (9.0, "B2"), "Standard": (18.0, "A4"), "Enhanced": (34.0, "A5")}
#: getInfo measurementRange (metres): the most one image can measure.
MEASUREMENT_RANGE_M = {"A1": 4.7e-07, "C1": 4.7e-07, "We": 2.56e-08}
PROTOCOL_VERSION = 4
PROBE_PORT, IMAGE_PORT = 7072, 7071
CONFIG_DEFAULTS = {"detilt": True, "use2ndorder": True}

#: CorrectorService states (== DE-TEM-Channel OperationState for 0..3).
IDLE, RUNNING, DONE, FAILED, ABSENT = 0, 1, 2, 3, 6

# Durations on the twin clock (seconds): the reference server's sleeps.
T_MEASURE_C1A1 = 2.0
T_TABLEAU_OVERHEAD = 1.0
T_TABLEAU_PER_IMAGE = 0.4
T_CORRECT = 0.5
T_ALIGNMENT = 1.0
T_CONFIG = 0.05
T_READBACK = 0.2
T_RECONNECT = 1.0

#: Tableau tilt rings: (fraction of the outer angle, azimuths); plus one untilted image.
TABLEAU_RINGS = {
    "Fast": ((1.0, 8),),
    "Standard": ((1.0, 12), (0.5, 6)),
    "Enhanced": ((1.0, 18), (2 / 3, 12), (1 / 3, 6)),
}

# ---------------------------------------------------------------------------
# Model parameters (nm)
# ---------------------------------------------------------------------------

#: Uncorrected lens at 200 kV: round terms signed, others |value| (random azimuth).
NATIVE_NOMINAL = {"C3": 1.2e6, "C5": 2.0e6, "B2": 150.0, "A2": 300.0, "S3": 3000.0, "A3": 3000.0}
NATIVE_SPREAD = {"C3": 0.03, "C5": 0.25, "B2": 0.5, "A2": 0.5, "S3": 0.5, "A3": 0.5}
#: A good tune at 200 kV (|value| drawn in [0.5, 1.5] x nominal, random azimuth / sign).
#: Gives a pi/4 angle (excluding C1/A1, as CEOS quotes it) of ~24-30 mrad with C5 ~2 mm.
TUNED_NOMINAL = {"C1": 1.5, "A1": 1.5, "B2": 7.0, "A2": 14.0, "C3": 400.0, "S3": 130.0, "A3": 400.0}
#: Out of tune (cold start, after an HT change): added to the tuned offset.
#: (C1/A1 stay mostly inside the 470 nm measurement range, so a Fast tableau or a
#: C1A1 measurement still works; a Standard tableau loses its outer images.)
DETUNE_HT = {"C1": 300.0, "A1": 150.0, "B2": 1000.0, "A2": 500.0, "C3": 15000.0, "S3": 5000.0, "A3": 5000.0}
#: Probe mode / function mode / TEM-STEM change.
DETUNE_MODE = {"C1": 20.0, "A1": 10.0, "B2": 25.0, "A2": 10.0}
#: Random-walk rates, nm per sqrt(second) (rms of |increment|).
DRIFT_RATES = {"C1": 0.30, "A1": 0.20, "B2": 0.60, "A2": 0.25, "C3": 8.0, "S3": 3.0, "A3": 4.0}

#: Per-image measurement noise of the induced C1 / A1 (nm, 1 sigma) and relative part.
IMAGE_SIGMA_NM = 0.5
IMAGE_SIGMA_REL = 0.01
#: Tilt calibration error (relative, fixed per unit) and jitter (mrad).
TILT_CAL_ERROR = 0.01
TILT_JITTER_MRAD = 0.05

#: Correction: (fixed gain error sigma, fixed rotation error sigma in deg, jitter fraction).
CORRECT_ERRORS = {
    "C1": (0.02, 0.0, 0.01),
    "A1": (0.03, 2.0, 0.01),
    "B2": (0.05, 3.0, 0.02),
    "A2": (0.05, 3.0, 0.02),
    "C3": (0.03, 0.0, 0.01),
    "S3": (0.08, 5.0, 0.03),
    "A3": (0.08, 5.0, 0.03),
}
CORRECT_ERRORS_DEFAULT = (0.15, 8.0, 0.05)
#: Parasitic coupling: correcting X by |d| adds k*|d| (random azimuth) to each Y.
COUPLING = {
    "A1": {"C1": 0.005},
    "B2": {"C1": 0.02, "A1": 0.03},
    "A2": {"A1": 0.03, "B2": 0.02},
    "C3": {"S3": 0.01, "A3": 0.01, "B2": 0.002},
    "S3": {"C3": 0.02, "B2": 0.005},
    "A3": {"C3": 0.02, "A2": 0.005},
    "C5": {"C3": 0.05, "S3": 0.02, "A3": 0.02},
}
SELECT_ERROR_SCALE = {"fine": 0.5, "coarse": 2.0}


class CorrectorError(RuntimeError):
    """A corrector command was refused or failed (``getCorrectorError`` text)."""


def _server_error(detail: str) -> str:
    """How CeosRpcClient joins a JSON-RPC error: message, data.message, code."""
    return f"Server error: {detail} (-32000)"


def _round(name: str) -> bool:
    return bool(TERMS[name][3])


def _symmetry(name: str) -> int:
    p, q = TERMS[name][0], TERMS[name][1]
    return abs(p - q)


def _random_coeff(rng: np.random.Generator, name: str, magnification: float) -> complex:
    if _round(name):
        return complex(magnification * (1.0 if rng.random() < 0.5 else -1.0), 0.0)
    phi = rng.uniform(0.0, 2.0 * math.pi)
    return complex(magnification * math.cos(phi), magnification * math.sin(phi))


def native_aberrations(seed: int = 0, side: str = "probe") -> Aberrations:
    """The uncorrected objective of one side (deterministic from ``seed``)."""
    rng = np.random.default_rng([int(seed), 7072 if side == "probe" else 7071, 1])
    out = {}
    for name, nom in NATIVE_NOMINAL.items():
        spread = NATIVE_SPREAD[name]
        magnification = nom * (1.0 + spread * rng.uniform(-1.0, 1.0))
        out[name] = complex(magnification, 0.0) if _round(name) else _random_coeff(rng, name, magnification)
    return Aberrations(out)


def tuned_residual(rng: np.random.Generator, scale: float = 1.0) -> Aberrations:
    """A residual after a good tune (without C5, which stays native)."""
    return Aberrations(
        {n: _random_coeff(rng, n, v * scale * rng.uniform(0.5, 1.5)) for n, v in TUNED_NOMINAL.items()}
    )


def operator_c1a1(state, side: str, stig_nm_per_unit: float = 1000.0) -> tuple[float, complex]:
    """The operator's focus (C1) and stigmator (A1) of a MicroscopeState, nm, as the
    optics adds them: C1 = (defocus + stage z) * 1000; A1 = condenser stig (probe) /
    objective stig (image) x ``stig_nm_per_unit`` (OpticsConfig default 1000)."""
    c1 = (float(state.defocus_um) + float(state.stage.z_um)) * 1000.0
    stig = state.condenser_stig if side == "probe" else state.objective_stig
    return c1, complex(stig.x * stig_nm_per_unit, stig.y * stig_nm_per_unit)


def pi4_angle_mrad(ab: Aberrations, ht_kv: float = 200.0, *, include_c1a1: bool = False) -> float:
    """The pi/4 angle a CEOS tableau quotes: C1 and A1 excluded (the operator sets
    focus and stigmation), everything else in phase (worst azimuth)."""
    lam = electron_wavelength_nm(ht_kv)
    a = ab if include_c1a1 else Aberrations({k: v for k, v in ab.coeffs.items() if k not in ("C1", "A1")})
    return a.flat_angle_mrad(lam)


# ---------------------------------------------------------------------------
# Tableau physics
# ---------------------------------------------------------------------------


def induced_c1a1(coeffs: dict[str, complex], tilt: complex) -> tuple[float, complex]:
    """Effective defocus C1 and two-fold astigmatism A1 (nm) seen with the beam tilted
    by ``tilt`` (complex angle, rad): the second derivatives of chi about the tilt.

    With chi = Re sum pre*c*wb^p*w^q: C1 = 2 Re f_{w wb}, A1 = f_{wb wb} + conj(f_{ww}).
    (C3 alone gives the textbook 2*C3*|t|^2 and C3*t^2.)
    """
    t = complex(tilt)
    tb = t.conjugate()
    c1 = 0.0
    a1 = 0j
    for name, c in coeffs.items():
        p, q, pre, _ = TERMS[name]
        k = pre * complex(c)
        if p >= 1 and q >= 1:
            c1 += 2.0 * (k * p * q * tb ** (p - 1) * t ** (q - 1)).real
        if p >= 2:
            a1 += k * p * (p - 1) * tb ** (p - 2) * t**q
        if q >= 2:
            a1 += (k * q * (q - 1) * tb**p * t ** (q - 2)).conjugate()
    return c1, a1


def tableau_tilts(tab_type: str, angle_mrad: float) -> list[complex]:
    tilts = [0j]
    for frac, n in TABLEAU_RINGS[tab_type]:
        r = angle_mrad * 1e-3 * frac
        off = math.pi / n if frac < 1.0 else 0.0
        tilts += [
            r * complex(math.cos(off + 2 * math.pi * i / n), math.sin(off + 2 * math.pi * i / n))
            for i in range(n)
        ]
    return tilts


def _params(names: Iterable[str]) -> list[tuple[str, complex]]:
    """Real unknowns of a fit: (name, basis unit 1 or 1j)."""
    out = []
    for n in names:
        out.append((n, 1.0 + 0j))
        if not _round(n):
            out.append((n, 1j))
    return out


def _design_rows(params, tilt: complex) -> np.ndarray:
    """3 x P: rows = induced C1, Re A1, Im A1 per unit parameter."""
    m = np.zeros((3, len(params)))
    for j, (n, unit) in enumerate(params):
        c1, a1 = induced_c1a1({n: unit}, tilt)
        m[:, j] = (c1, a1.real, a1.imag)
    return m


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class MeasurementResult:
    """A tableau / C1A1 measurement, as the corrector reports it (plus confidence)."""

    kind: str  # "C1A1" | "tableau"
    aberrations: Aberrations  # measured, complex nm (TOTAL: incl. operator focus/stig)
    sigma_nm: dict[str, float]  # 1-sigma confidence of |coefficient|, nm
    tab_type: str = ""
    angle_mrad: float = 0.0
    max_fit: str = ""
    images: int = 1
    images_lost: int = 0
    t_start: float = 0.0
    t_done: float = 0.0
    side: str = "probe"
    names: tuple = ()

    def ceos(self) -> list[tuple[str, tuple[float, float]]]:
        """[(name, (x_m, y_m))] in CEOS order (the AberrationSet on the wire)."""
        out = []
        for n in self.names:
            v = self.aberrations[n]
            out.append((n, (v.real * 1e-9, 0.0 if _round(n) else v.imag * 1e-9)))
        return out

    def to_string(self) -> str:
        """``AberrationSet::toString``: ``C1=x,y;A1=x,y;...`` metres, ``%.9g``."""
        return ";".join(f"{n}={x:.9g},{y:.9g}" for n, (x, y) in self.ceos())

    def pi4_angle_mrad(self, ht_kv: float = 200.0) -> float:
        return pi4_angle_mrad(self.aberrations, ht_kv)


def format_aberration_set(items: Iterable[tuple[str, tuple[float, float]]]) -> str:
    return ";".join(f"{n}={x:.9g},{y:.9g}" for n, (x, y) in items)


def parse_aberration_set(text: str) -> list[tuple[str, tuple[float, float]]]:
    """Inverse of ``AberrationSet::toString`` (getAberrations / getCorrectorResult)."""
    out = []
    for part in (text or "").split(";"):
        name, sep, val = part.partition("=")
        if not sep:
            continue
        xs = val.split(",")
        try:
            x = float(xs[0])
            y = float(xs[1]) if len(xs) > 1 else 0.0
        except ValueError:
            continue
        out.append((name.strip(), (x, y)))
    return out


def ceos_to_nm(items: Iterable[tuple[str, tuple[float, float]]]) -> dict[str, complex]:
    """CEOS (x, y) metres -> {name: complex nm} for the names chi knows (We/WD dropped)."""
    return {n: complex(x * 1e9, 0.0 if _round(n) else y * 1e9) for n, (x, y) in items if n in TERMS}


def parse_info(text: str) -> dict[str, str]:
    """``CorrectorInfo::toString`` -> dict (measurementRange kept as its string)."""
    out = {}
    for part in (text or "").split(";"):
        k, sep, v = part.partition("=")
        if sep:
            out[k.strip()] = v
    return out


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


class _Drift:
    """Deterministic random walk sampled on a fixed time grid, independent of how
    often (or when) it is read. Piecewise constant between grid points (``step_s``),
    so render caches keyed on the aberrations stay valid within a step. Time must
    not go backwards (it is clamped)."""

    def __init__(self, rates: dict[str, float], seed, t0: float, step_s: float = 1.0):
        self.names = [n for n, r in rates.items() if r > 0]
        self.rates = np.array([float(rates[n]) for n in self.names])
        self.round = np.array([_round(n) for n in self.names])
        self.step = float(step_s)
        self.t0 = float(t0)
        self._rng = np.random.default_rng(seed)
        self._k = 0  # grid index of self._cur
        self._cur = np.zeros(len(self.names), np.complex128)
        self._next = self._cur + self._increments(1)[0]

    def _increments(self, n: int) -> np.ndarray:
        z = self._rng.standard_normal((n, len(self.names), 2))
        s = self.rates * math.sqrt(self.step)
        re = np.where(self.round, z[..., 0], z[..., 0] / math.sqrt(2.0)) * s
        im = np.where(self.round, 0.0, z[..., 1] / math.sqrt(2.0)) * s
        return re + 1j * im

    def at(self, t: float) -> dict[str, complex]:
        if not self.names:
            return {}
        x = max(0.0, (float(t) - self.t0) / self.step)
        k = math.floor(x)
        while self._k < k:  # advance in chunks
            n = min(k - self._k, 100_000)
            inc = self._increments(n)
            path = self._next + np.cumsum(inc, axis=0)
            self._k += n
            self._cur = path[-2] if n >= 2 else self._next
            self._next = path[-1]
        v = self._cur  # value at the grid point at or before t (clamped: no going back)
        return {n: complex(val) for n, val in zip(self.names, v)}
