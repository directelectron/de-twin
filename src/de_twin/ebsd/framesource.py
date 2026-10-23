"""A scan's EBSD camera frames made by the fused CPU kernel (:mod:`de_twin.ebsd.frames`).

:class:`EbsdFrameSource` holds everything a frame needs but its scan point and its noise
serial, worked out once per scan and column state: the scan's plan (per-point grain,
orientation, pattern centre, ...) and master bank from :class:`~de_twin.ebsd.EbsdRenderer`,
the detector's :class:`~de_twin.detector.detector.GridExposure`, and the normal table. A
frame is then one call of the kernel on one worker thread (:mod:`de_twin.workers`), writing
the raw uint16 frame wherever the caller says.

The frames are the ones ``DigitalTwin.frames`` makes through ``render_ebsd_frame`` and
``Detector.expose``: the same physics and noise streams, so they differ only where float
rounding tips a draw (see :mod:`de_twin.ebsd.frames`).
"""

from __future__ import annotations

import threading
from typing import Optional

import numba as nb
import numpy as np

from ..detector import fast
from . import frames as K
from .render import ebsd_geometry

#: The extended normal table is made for rows up to this wide.
_TABLE_WIDTH = 16384
_TABLES: dict = {}
_TABLES_LOCK = threading.Lock()


def _extended(table: np.ndarray) -> np.ndarray:
    with _TABLES_LOCK:
        hit = _TABLES.get(id(table))
        if hit is None or hit[0] is not table:
            hit = _TABLES[id(table)] = (table, K.extended_table(table, _TABLE_WIDTH))
        return hit[1]


@nb.njit(cache=True, nogil=True)
def _mix(x):
    x = (x ^ (x >> np.uint64(30))) * np.uint64(fast._M1)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(fast._M2)
    return x ^ (x >> np.uint64(31))


@nb.njit(cache=True, nogil=True)
def _noise(seed, serials, keys, zs, ns):
    """``detector.fast.frame_key`` / ``offsets`` for each serial, without the interpreter."""
    s = np.uint64(seed & 0xFFFFFFFF) << np.uint64(32)
    for i in range(serials.shape[0]):
        f = np.uint64(serials[i] & 0xFFFFFFF) << np.uint64(4)
        k0 = _mix(s ^ f)  # stage 0
        keys[i] = _mix(s ^ f ^ np.uint64(1))  # stage 1
        zs[i] = np.int64(_mix(k0 ^ np.uint64(0xA5A5)) % np.uint64(fast.NORMALS))
        ns[i] = np.int64(_mix(k0 ^ np.uint64(0x5A5A)) % np.uint64(fast.NORMALS))


class EbsdFrameSource:
    """Raw EBSD camera frames of one scan (see the module docstring). ``plan``: the scan's
    ``EbsdPlan``; ``grid``: ``EbsdRenderer.grid(optics)``; ``exposure``: the detector's
    ``GridExposure`` for the frame time."""

    def __init__(self, plan, grid, exposure, kappa_lobe: Optional[tuple] = None):
        h, w, ox, oy, bx, by = grid
        if (h, w) != tuple(exposure.shape) or w > _TABLE_WIDTH:
            raise ValueError("the EBSD grid and the detector's output grid differ")
        self.plan = plan
        self.shape = (h, w)
        self.exposure = exposure
        self.sensor_pixels = h * w * round(1.0 / exposure.inv_nb)
        lobe, kappa = float(plan.lobe), float(plan.kappa)
        self._projection = (
            np.ascontiguousarray(plan.bank).ravel(), int(plan.bank.shape[1]), plan.face0, plan.full, plan.offs,
            plan.scales, plan.matrices, plan.o1, plan.o2, plan.ph1, plan.ph2, plan.mix, plan.contrast, plan.es,
            plan.finv, bool(plan.has_f), plan.dx, plan.dy, plan.pc_lin, plan.s2dT, float(ox), float(oy),
            float(bx), float(by), plan.normal, plan.fwd, lobe, kappa, K._vmf(kappa), float(exposure.t_nb))
        e = exposure
        self._detector = (h, e.gain, e.offset, e.hot_ptr, e.hot_col, e.hot_val, float(e.adu), float(e.k),
                          float(e.a), float(e.inv_nb), float(e.sigma), float(e.scale), float(e.vmax),
                          _extended(e.table))

    # ------------------------------------------------------------------ noise
    def noise(self, seed: int, serials) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each frame's (deposit key, gamma/Poisson table start, read-noise table start), as
        ``Detector._expose_fast`` derives them from (seed, serial)."""
        serials = np.asarray(serials, np.int64)
        keys = np.empty(len(serials), np.uint64)
        zs = np.empty(len(serials), np.int64)
        ns = np.empty(len(serials), np.int64)
        _noise(int(seed), serials, keys, zs, ns)
        return keys, zs, ns

    # ------------------------------------------------------------------ frames
    def render_into(self, local: dict, out: np.ndarray, ks, seed: int, serials) -> tuple[np.ndarray, np.ndarray]:
        """Frames of flat scan points *ks* with noise (seed, serials[i]) into ``out[i]`` (a
        (>= len(ks), h, w) uint16 array), on the calling thread, in scratch kept in *local*.
        Returns (electrons, clipped pixels) per frame."""
        h, w = self.shape
        scratch = local.get(("ebsd-frames", w))
        if scratch is None:
            s32, su32 = K.scratch(1, w)
            scratch = local[("ebsd-frames", w)] = (s32[0], su32[0])
        d32, dui = scratch
        ks = np.asarray(ks, np.int64)
        n = len(ks)
        keys, zs, ns = self.noise(seed, serials)
        dose = np.zeros(n)
        sats = np.zeros(n, np.int64)
        K.raw_frames_serial(*self._projection, *self._detector, ks, keys, zs, ns, out.reshape(-1),
                            np.arange(n, dtype=np.int64) * (h * w), dose, sats, d32, dui)
        return dose, sats

    def render_one(self, local: dict, out: np.ndarray, k: int, seed: int, serial: int) -> tuple[float, int]:
        """One frame into the (h, w) *out* (see :meth:`render_into`)."""
        dose, sats = self.render_into(local, out, [k], seed, [serial])
        return float(dose[0]), int(sats[0])

    def submit(self, workers, out: np.ndarray, ks, seed: int, serials):
        """Frames of flat scan points *ks* into ``out[i]`` (a (>= len(ks), h, w) uint16 array)
        as one job for *workers* (:class:`de_twin.workers.Workers`), the interpreter taken once
        for them all; returns its `Batch`, whose one result is (electrons, clipped) per frame."""
        return workers.submit(self.render_into, [(out, np.asarray(ks, np.int64), int(seed),
                                                  np.asarray(serials, np.int64))])

    def render(self, ks, seed: int, serials, out: Optional[np.ndarray] = None, band: Optional[int] = None):
        """Frames of flat scan points *ks* in one call on numba's threads, a band of rows a task
        (the way to make a few frames quickly): ``(frames (B, h, w) uint16, electrons, clipped)``."""
        import numba

        h, w = self.shape
        ks = np.asarray(ks, np.int64)
        B = len(ks)
        out = np.empty((B, h, w), np.uint16) if out is None else out
        threads = numba.get_num_threads()
        band = band or max(8, -(-h * B // (2 * threads)))
        band = min(band, h)
        nbands = -(-h // band)
        keys, zs, ns = self.noise(seed, serials)
        dose = np.zeros((B, nbands))
        sats = np.zeros((B, nbands), np.int64)
        s32, su32 = _parallel_scratch(w)
        K.raw_frames(*self._projection, *self._detector, ks, keys, zs, ns, band, out.reshape(-1),
                     np.arange(B, dtype=np.int64) * h * w, dose, sats, s32, su32)
        return out, dose.sum(axis=1), sats.sum(axis=1)

    def means(self, ks) -> np.ndarray:
        """The Poisson means of scan points *ks*: (B, h, w) float32, electrons per grid pixel
        in a frame (``render_frame``'s ``lam``)."""
        h, w = self.shape
        lam = np.empty((len(ks), h, w), np.float32)
        K.means(*self._projection, np.asarray(ks, np.int64), lam)
        return lam


_SCRATCH: dict = {}
_SCRATCH_LOCK = threading.Lock()


def _parallel_scratch(width: int):
    import numba

    with _SCRATCH_LOCK:
        hit = _SCRATCH.get(width)
        if hit is None:
            hit = _SCRATCH[width] = K.scratch(numba.config.NUMBA_NUM_THREADS, width)
        return hit


def usable(optics) -> bool:
    """Whether the fused kernel renders this EBSD geometry (one sample per pixel)."""
    return int(max(1, ebsd_geometry(optics).supersample)) == 1
