"""Kinematic Kikuchi master patterns: one per (phase, beam energy), built once and cached.

The master pattern of a phase is its Kikuchi intensity over every crystal direction ``n``::

    M(n) = sum_g  |F_g|^2  P(|n . g_hat| / sin(theta_B,g)),     sin(theta_B) = lambda |g| / 2

summed over the reflections of the crystal library (:class:`de_twin.crystal.CrystalLibrary`:
Lobato scattering factors and the material's Debye-Waller factor), one per +-g pair. Each
band is the band of lattice planes ``g`` — the directions within ``theta_B`` of the plane —
so it has the true width ``2 theta_B`` and its kinematic weight ``|F_g|^2``; higher orders of
a plane (2g, 3g, ...) nest inside it, narrower and weaker. The profile ``P`` is a top hat
whose edges are blurred by the energy spread of the backscattered electrons
(``sigma_t = (dE / 2E)``: the band width goes as lambda) and the screen's angular blur, with
an excess line just inside and a deficiency just outside each edge, which is how dynamical
bands look.

It is then standardised (zero mean, unit variance over the sphere), its bright tail softly
compressed (``3 log(1 + m / 3)``, so zone axes stay bright without dominating), and stored as
uint8 (``value = offset + scale * code``, 256 levels over the range: well below the shot noise
of any pattern) on a square-Lambert northern hemisphere (kinematic patterns are
centrosymmetric): 0.9 MB per phase and energy at the default 0.15 deg.

What this does not have, compared with an EMsoft dynamical master pattern: no dynamical
intensities (band intensities are |F|^2, not Bloch-wave; no channelling contrast reversals,
no thickness/depth integration over the Monte-Carlo depth and energy distribution, no HOLZ
rings or fine lines inside zone axes, no anomalous absorption), and the excess/deficiency
edges are symmetric and fixed in the crystal, where in reality they depend on the
scattering geometry.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import kernels

#: Reflections weaker than this fraction of the strongest |F|^2 are left out.
MIN_F2_FRACTION = 0.01
#: At most this many bands (strongest first).
MAX_BANDS = 400
#: The library's reciprocal-lattice extent for master patterns (1/nm).
MAX_G_INV_NM = 20.0
#: Bytes of master patterns kept (least recently used dropped first).
CACHE_BUDGET_BYTES = 64 * 1024 * 1024
#: Bright tail compression m -> c log(1 + m / c) (m > 0): zone axes stay the brightest points
#: without dominating the pattern the way a plain kinematic sum makes them.
COMPRESS = 3.0


@dataclass
class MasterPattern:
    material: int
    kv: float
    data: np.ndarray  # (N, N) uint8 codes, square-Lambert northern hemisphere
    offset: float  # standardised value = offset + scale * code
    scale: float
    n_bands: int
    build_s: float
    resolution_deg: float
    #: Southern hemisphere (same coding), read at Lambert(x, y, |z|) for z < 0 as EMsoft stores
    #: it; None for a centrosymmetric pattern (the north read at -direction).
    south: Optional[np.ndarray] = None
    source: str = "kinematic"  # or the file / "array" a user master came from
    #: The cube map the renderer reads ((3 | 6, Nf, Nf) uint8, same coding; see kernels).
    cube: Optional[np.ndarray] = None

    def values(self, hemisphere: str = "north") -> np.ndarray:
        """The standardised master pattern as float32 (N, N)."""
        d = (
            self.data
            if hemisphere == "north"
            else (self.south if self.south is not None else self.data[::-1, ::-1])
        )
        return np.float32(self.offset) + np.float32(self.scale) * d.astype(np.float32)

    @property
    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in (self.data, self.south, self.cube) if a is not None))

    def value(self, directions) -> np.ndarray:
        """Bilinear master values at crystal directions (..., 3) (numpy, for checks)."""
        v = np.asarray(directions, np.float64)
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        south = v[..., 2] < 0
        if self.south is not None:  # EMsoft: the southern array at Lambert(x, y, |z|)
            v = np.where(south[..., None], v * np.array([1.0, 1.0, -1.0]), v)
        X, Y = kernels.lambert_xy(v)  # folds z < 0 to -v (the centrosymmetric case)
        n = self.data.shape[0]
        s = (n - 1) / (2.0 * kernels.LAMBERT_L)
        fx, fy = (X + kernels.LAMBERT_L) * s, (Y + kernels.LAMBERT_L) * s
        ix = np.clip(np.floor(fx).astype(int), 0, n - 2)
        iy = np.clip(np.floor(fy).astype(int), 0, n - 2)
        tx, ty = np.clip(fx - ix, 0, 1), np.clip(fy - iy, 0, 1)
        out = np.empty(v.shape[:-1], np.float32)
        for hemi, sel in (
            (("north", ~south), ("south", south)) if self.south is not None else (("north", None),)
        ):
            d = self.values(hemi)
            val = (d[iy, ix] * (1 - tx) + d[iy, ix + 1] * tx) * (1 - ty) + (
                d[iy + 1, ix] * (1 - tx) + d[iy + 1, ix + 1] * tx
            ) * ty
            if sel is None:
                out = val.astype(np.float32)
            else:
                out[sel] = val[sel]
        return out


def bands(
    material: int,
    kv: float,
    max_g_inv_nm: float = MAX_G_INV_NM,
    min_f2_fraction: float = MIN_F2_FRACTION,
    max_bands: int = MAX_BANDS,
):
    """(hkl (K, 3) int, unit plane normals (K, 3), sin(theta_B) (K,), weight |F|^2 / max (K,))
    of the Kikuchi bands of ``material`` at ``kv``, one per +-g pair, strongest first."""
    from ..crystal import library_for
    from ..optics.physics import electron_wavelength_nm

    lib = library_for(int(material), max_g_inv_nm)
    if lib is None:
        raise ValueError(f"material {material} is not crystalline")
    hkl = lib.hkl
    # one of each +-g pair: the first non-zero index positive
    first = np.where(hkl[:, 0] != 0, hkl[:, 0], np.where(hkl[:, 1] != 0, hkl[:, 1], hkl[:, 2]))
    keep = first > 0
    g, f2, hkl = lib.g[keep], lib.f2[keep], hkl[keep]
    order = np.argsort(-f2, kind="stable")
    g, f2, hkl = g[order], f2[order], hkl[order]
    sel = f2 >= min_f2_fraction * f2.max()
    g, f2, hkl = g[sel][:max_bands], f2[sel][:max_bands], hkl[sel][:max_bands]
    glen = np.linalg.norm(g, axis=1)
    lam = electron_wavelength_nm(float(kv))
    return hkl, g / glen[:, None], 0.5 * lam * glen, f2 / f2.max()


def cube_side(resolution_deg: float) -> int:
    """Nodes per side of a cube-map face whose (largest, face-centre) pixel is ``resolution_deg``."""
    return int(math.ceil(2.0 / math.radians(max(resolution_deg, 0.01)))) | 1


def quantise(*arrays):
    """(offset, scale, uint8 codes of each array) over their joint range."""
    lo = min(float(a.min()) for a in arrays)
    hi = max(float(a.max()) for a in arrays)
    scale = (hi - lo) / 255.0 if hi > lo else 1.0
    return lo, scale, [np.clip(np.rint((a - lo) / scale), 0, 255).astype(np.uint8) for a in arrays]


def lambert_sample(north, south, dirs) -> np.ndarray:
    """Bilinear values of float Lambert hemispheres (``south`` None: centrosymmetric) at unit
    directions (..., 3), EMsoft's convention (south read at Lambert(x, y, |z|))."""
    v = np.asarray(dirs, np.float64)
    neg = v[..., 2] < 0
    if south is not None:
        v = np.where(neg[..., None], v * np.array([1.0, 1.0, -1.0]), v)
    X, Y = kernels.lambert_xy(v)
    n = north.shape[0]
    s = (n - 1) / (2.0 * kernels.LAMBERT_L)
    fx, fy = (X + kernels.LAMBERT_L) * s, (Y + kernels.LAMBERT_L) * s
    ix = np.clip(np.floor(fx).astype(int), 0, n - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, n - 2)
    tx, ty = np.clip(fx - ix, 0, 1), np.clip(fy - iy, 0, 1)

    def bil(d):
        return (d[iy, ix] * (1 - tx) + d[iy, ix + 1] * tx) * (1 - ty) + (
            d[iy + 1, ix] * (1 - tx) + d[iy + 1, ix + 1] * tx
        ) * ty

    out = bil(north)
    if south is not None:
        out = np.where(neg, bil(south), out)
    return out


def lambert_side(resolution_deg: float) -> int:
    """Nodes per side of a square-Lambert hemisphere whose pixel is ``resolution_deg``."""
    n = int(math.ceil(math.sqrt(2.0 * math.pi) / math.radians(max(resolution_deg, 0.01))))
    return n | 1


def build_master(
    material: int,
    kv: float,
    *,
    energy_spread_frac: float = 0.08,
    angular_blur_deg: float = 0.1,
    resolution_deg: float = 0.15,
    max_bands: int = MAX_BANDS,
    hkl_subset=None,
) -> MasterPattern:
    """Build the master pattern (see the module docstring). ``hkl_subset``: only these planes
    (and their higher orders' weights as given; for tests)."""
    t0 = time.perf_counter()
    hkl, normals, sin_tb, amp = bands(material, kv, max_bands=max_bands)
    if hkl_subset is not None:
        want = {tuple(int(v) for v in h) for h in hkl_subset}
        want |= {tuple(-v for v in h) for h in want}
        m = np.array([tuple(h) in want for h in hkl.tolist()], bool)
        hkl, normals, sin_tb, amp = hkl[m], normals[m], sin_tb[m], amp[m]
    theta = np.arcsin(np.clip(sin_tb, 0, 1))
    sig_e = float(energy_spread_frac) / 2.355 / 2.0  # edge shift dt = dE / (2E)
    sig_a = math.radians(float(angular_blur_deg)) / np.maximum(theta, 1e-9)
    sigma = np.maximum(np.sqrt(sig_e**2 + sig_a**2), 0.02)
    sigma_edge = np.maximum(sigma, kernels.EDGE_MIN_SIGMA)
    reach = 1.0 + kernels.EDGE_OFFSET + 5.0 * np.maximum(sigma, sigma_edge)
    n = lambert_side(resolution_deg)
    nf = cube_side(resolution_deg)
    args = (
        np.ascontiguousarray(normals, np.float64),
        sin_tb.astype(np.float64),
        amp.astype(np.float64),
        sigma.astype(np.float64),
        sigma_edge.astype(np.float64),
        reach.astype(np.float64),
    )

    def evaluate(dirs):
        out = np.empty(len(dirs), np.float64)
        kernels.accumulate_bands(np.ascontiguousarray(dirs), *args, out)
        return out

    lam = evaluate(kernels.lambert_directions(n).reshape(-1, 3))
    cub = evaluate(kernels.cube_directions(nf, 3).reshape(-1, 3))
    mu, sd = lam.mean(), lam.std()  # equal-area grid: plain moments are moments over the sphere
    sd = sd if sd > 0 else 1.0

    def finish(a):
        a = (a - mu) / sd
        return np.where(a > 0, COMPRESS * np.log1p(np.maximum(a, 0.0) / COMPRESS), a)

    lo, scale, (data, cube) = quantise(finish(lam), finish(cub))
    return MasterPattern(
        int(material),
        float(kv),
        data.reshape(n, n),
        lo,
        scale,
        int(len(hkl)),
        time.perf_counter() - t0,
        float(resolution_deg),
        cube=cube.reshape(3, nf, nf),
    )


class MasterCache:
    """Byte-budgeted LRU of master patterns, keyed by everything that shapes one."""

    def __init__(self, budget_bytes: int = CACHE_BUDGET_BYTES):
        self.budget = int(budget_bytes)
        self._d: "OrderedDict[tuple, MasterPattern]" = OrderedDict()
        self._lock = threading.Lock()
        self.builds = 0

    def get(
        self,
        material: int,
        kv: float,
        energy_spread_frac: float,
        angular_blur_deg: float,
        resolution_deg: float,
    ) -> MasterPattern:
        key = (
            int(material),
            round(float(kv), 2),
            round(float(energy_spread_frac), 4),
            round(float(angular_blur_deg), 4),
            round(float(resolution_deg), 4),
        )
        with self._lock:
            hit = self._d.get(key)
            if hit is not None:
                self._d.move_to_end(key)
                return hit
            mp = build_master(
                material,
                kv,
                energy_spread_frac=energy_spread_frac,
                angular_blur_deg=angular_blur_deg,
                resolution_deg=resolution_deg,
            )
            self.builds += 1
            self._d[key] = mp
            while len(self._d) > 1 and self.nbytes > self.budget:
                self._d.popitem(last=False)
            return mp

    @property
    def nbytes(self) -> int:
        return sum(m.nbytes for m in self._d.values())

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        return len(self._d)


#: Process-wide master patterns (they depend only on the phase and the energy settings).
MASTERS = MasterCache()


def master_for(material: int, kv: float, geometry=None) -> MasterPattern:
    """The cached master pattern of ``material`` at ``kv`` with ``geometry``'s energy spread,
    angular blur and resolution (an :class:`~de_twin.ebsd.EbsdGeometry`; default settings
    when None)."""
    from .geometry import EbsdGeometry

    g = geometry or EbsdGeometry()
    return MASTERS.get(material, kv, g.energy_spread_frac, g.angular_blur_deg, g.master_resolution_deg)
