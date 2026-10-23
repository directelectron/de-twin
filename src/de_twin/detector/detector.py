"""Detector model: turns an electron flux into a raw camera frame.

Integrating cameras (DE16, DE64, Celeritas, DESim, ...)::

    electrons  N ~ Poisson(flux * t)                     (per sensor pixel)
    deposit    S | N ~ Gamma(N / cv^2, mu * cv^2)        (Landau-like; mu = ADU/e at the HT,
                                                          Wilson-Hilferty sampling)
    charge     S <- S * [a, 1-2a, a] (x) [a, 1-2a, a]    (separable charge sharing / MTF,
                                                          sum-preserving)
    analogue   x = S * gain + offset + fpn + dark_current * t + read_noise * z
    defects    hot pixels (large dark current), dead pixels (gain 0), bad columns
    ADC        clip to [0, 2^bits - 1], hardware binning (average), round -> uint16

Counting cameras (Apollo family) emit counted events per pixel (uint8):
each delivered frame sums ``round(t / cycle)`` sensor cycles, a pixel counts
at most one event per cycle, and events closer than ``cluster_area_px``
coincide (paralysable loss, ``p = (1 - exp(-lam * A)) / A`` per cycle).
With ``request.super_resolution`` (unbinned) the frame is 2x the ROI each way and every
event lands in one of its pixel's 2x2 sub-pixels, drawn uniformly (`fast.count_super`).

Speed: the frame is processed in fixed bands of rows on a thread pool (numpy
releases the GIL) using per-thread scratch buffers. Each band draws from its
own generator ``rng_for(seed, SeedKind.SHOT_NOISE, frame_index, salt)``, so a
frame is bit-identical for a given (seed, frame_index) whatever the thread
count. Poisson draws below a mean of 12 use an exact vectorised inversion
sampler (about 2x faster than numpy's); 12..40 use numpy; above 40 a rounded
Gaussian.

When ``flux`` is given at the *binned* output shape instead of the ROI shape,
the detector simulates directly on the binned grid (the electron, read-noise
and map statistics are those of the binned pixel). This is much faster for
binned acquisitions; per-sensor-pixel clipping inside a bin is not modelled.
Dark frames (``flux=None``) always take this path.
"""

from __future__ import annotations

import dataclasses
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from ..hashing import SeedKind, rng_for
from ..state import AcquisitionRequest, Roi
from .models import CameraModel, camera

_BAND_PIXELS = 1 << 18  # ~1 MB of float32 per scratch array per band
_POISSON_INV_CUT = 12.0  # below this mean: exact inversion sampling
_POISSON_GAUSS_CUT = 40.0  # at or above this mean: rounded Gaussian
_POOL: Optional[ThreadPoolExecutor] = None
_POOL_LOCK = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            n = int(os.environ.get("DE_TWIN_DETECTOR_THREADS", "0")) or min(12, os.cpu_count() or 2)
            _POOL = ThreadPoolExecutor(max_workers=max(1, n), thread_name_prefix="de-twin-detector")
    return _POOL


def _binning(request: Optional[AcquisitionRequest]) -> tuple[int, int]:
    """(bx, by) from ``request.hw_binning`` (DE-Server DeSize order: w, h)."""
    if request is None or request.hw_binning is None:
        return 1, 1
    bx, by = request.hw_binning
    return max(1, int(bx)), max(1, int(by))


class Detector:
    """A camera: sensor maps (dark, gain, defects) plus the per-frame noise pipeline.

    ``overrides`` replace any :class:`CameraModel` field, e.g.
    ``Detector("DE16", read_noise_adu=4, hot_pixel_fraction=0)``.
    ``expose`` is serialised by an internal lock (it reuses buffers).
    """

    def __init__(self, model: "CameraModel | str" = "DE16", seed: int = 0, *,
                 ht_kv: float = 300.0, **overrides):
        model = camera(model)
        if overrides:
            known = {f.name for f in dataclasses.fields(CameraModel)}
            bad = set(overrides) - known
            if bad:
                raise TypeError(f"unknown detector parameter(s): {sorted(bad)}")
            model = model.with_(**overrides)
        self.model: CameraModel = model
        self.seed = int(seed)
        self.ht_kv = float(ht_kv)
        self._maps_built = False
        self._binned_cache: dict = {}
        self._S: Optional[np.ndarray] = None  # reused deposit buffer
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"Detector({self.model.name!r}, seed={self.seed}, ht_kv={self.ht_kv})"

    def set_gain_mode(self, mode: str) -> None:
        """Read out at ``"high"`` or ``"low"`` gain (``CameraModel.gain_mode``) from the next frame."""
        if mode not in ("high", "low"):
            raise ValueError(f"unknown gain mode {mode!r} (high | low)")
        self.model = self.model.with_(gain_mode=mode)

    def _adu_at(self, ht: float) -> float:
        """``model.adu_per_electron_at``, remembered for the last energy (a per-frame cost)."""
        hit = getattr(self, "_adu_memo", None)
        if hit is None or hit[0] != (self.model, ht):
            hit = self._adu_memo = ((self.model, ht), float(self.model.adu_per_electron_at(ht)))
        return hit[1]

    # ------------------------------------------------------------ geometry
    def roi(self, request: Optional[AcquisitionRequest] = None) -> Roi:
        """Resolved hardware ROI: the requested ROI clamped to the sensor (0 size = full).

        No DE alignment is re-applied: DE-Server sends an already aligned ROI and
        its ``hw_frame`` must be reproduced exactly.
        """
        H, W = self.model.sensor_shape
        r = request.hw_roi if request is not None else None
        if r is None:
            return Roi(0, 0, W, H)
        x = min(max(int(r.x), 0), W - 1)
        y = min(max(int(r.y), 0), H - 1)
        w = int(r.w) if int(r.w) > 0 else W - x
        h = int(r.h) if int(r.h) > 0 else H - y
        return Roi(x, y, max(1, min(w, W - x)), max(1, min(h, H - y)))

    def output_shape(self, request: Optional[AcquisitionRequest] = None) -> tuple[int, int]:
        """(h, w) of the hardware frame: ROI, then binning (integer division, like DE-Server);
        twice that each way for a super-resolution counting frame."""
        h, w = self._grid_shape(request)
        return (2 * h, 2 * w) if self.super_resolution(request) else (h, w)

    def _grid_shape(self, request: Optional[AcquisitionRequest] = None) -> tuple[int, int]:
        r = self.roi(request)
        bx, by = _binning(request)
        return max(1, r.h // by), max(1, r.w // bx)

    def super_resolution(self, request: Optional[AcquisitionRequest] = None) -> bool:
        """Whether *request* reads this camera out at super-resolution: a counting camera,
        asked for it, unbinned (binning and super-resolution do not combine)."""
        return bool(request is not None and getattr(request, "super_resolution", False)
                    and self.model.hardware_counting and _binning(request) == (1, 1))

    # ------------------------------------------------------- ground truth
    def _build_maps(self) -> None:
        if self._maps_built:
            return
        m = self.model
        H, W = m.sensor_shape
        yy = np.arange(H, dtype=np.float32)[:, None]
        xx = np.arange(W, dtype=np.float32)[None, :]

        # dark offset + fixed pattern (GrabberSim Image::GenCol) + column FPN + DSNU
        rng = rng_for(self.seed, SeedKind.DARK, 0)
        off = np.full((H, W), m.dark_offset_adu, np.float32)
        if m.fpn_pattern:
            xi = np.arange(W)
            seg_w = max(1, m.segment_shape[1])
            col = (((xi // seg_w + 1) % 2) * 10 + np.where(xi < seg_w, 20, 0)
                   + np.where((xi >= seg_w) & (xi < 2 * seg_w), -20, 0)).astype(np.float32)
            off += (np.arange(H) % 8).astype(np.float32)[:, None] * (xi % 8).astype(np.float32)[None, :]
            off += col[None, :]
        if m.fpn_column_adu > 0:
            off += (rng.standard_normal(W, dtype=np.float32) * np.float32(m.fpn_column_adu))[None, :]
        if m.dsnu_adu > 0:
            off += rng.standard_normal((H, W), dtype=np.float32) * np.float32(m.dsnu_adu)

        # gain: radial fall-off, stitch-block gradient, column and pixel PRNU; mean 1
        rng = rng_for(self.seed, SeedKind.GAIN, 0)
        r2 = ((xx - (W - 1) / 2) / (W / 2)) ** 2 + ((yy - (H - 1) / 2) / (H / 2)) ** 2
        gain = 1.0 - np.float32(m.gain_radial / 2.0) * r2  # r2 = 2 at the corners
        seg_h = max(1, m.segment_shape[0])
        gain = gain * (1.0 + np.float32(m.gain_stitch) * ((yy % seg_h) / seg_h - 0.5))
        gain = gain.astype(np.float32, copy=False)
        if m.gain_column_sigma > 0:
            gain *= 1.0 + (rng.standard_normal(W, dtype=np.float32) * np.float32(m.gain_column_sigma))[None, :]
        if m.gain_pixel_sigma > 0:
            gain *= 1.0 + rng.standard_normal((H, W), dtype=np.float32) * np.float32(m.gain_pixel_sigma)
        gain /= np.float32(gain.mean(dtype=np.float64))

        # defects
        rng = rng_for(self.seed, SeedKind.BAD_PIXELS, 0)
        bad = np.zeros((H, W), bool)
        n = H * W
        n_hot = int(round(m.hot_pixel_fraction * n))
        n_dead = int(round(m.dead_pixel_fraction * n))
        n_cols = int(round(m.bad_column_fraction * W))
        picks = (rng.choice(n, size=n_hot + n_dead, replace=False) if n_hot + n_dead
                 else np.zeros(0, np.int64))
        hot_idx, dead_idx = picks[:n_hot], picks[n_hot:]
        lo, hi = m.hot_dark_current_adu_per_s
        hot_rate = np.exp(rng.uniform(np.log(lo), np.log(hi), n_hot)).astype(np.float32)
        gflat, oflat, bflat = gain.reshape(-1), off.reshape(-1), bad.reshape(-1)
        gflat[dead_idx] = 0.0
        oflat[dead_idx[: len(dead_idx) // 2]] = 0.0  # half of the dead pixels read ~0
        bflat[hot_idx] = True
        bflat[dead_idx] = True
        bad_cols = (np.sort(rng.choice(W, size=n_cols, replace=False)) if n_cols
                    else np.zeros(0, np.int64))
        for c in bad_cols:
            gain[:, c] *= 0.3
            off[:, c] += 150.0
            bad[:, c] = True

        self._offset = off
        self._gain = gain
        self._bad = bad
        self._hot_y, self._hot_x = np.divmod(hot_idx.astype(np.int64), W)
        self._hot_rate = hot_rate
        self.bad_columns = bad_cols
        self._maps_built = True

    @property
    def dark_map(self) -> np.ndarray:
        """Exposure-independent dark level (offset + fixed pattern), full sensor, ADU."""
        self._build_maps()
        return self._offset

    @property
    def gain_map(self) -> np.ndarray:
        """Relative gain (mean 1; 0 for dead pixels), full sensor."""
        self._build_maps()
        return self._gain

    @property
    def bad_pixel_mask(self) -> np.ndarray:
        """True for hot, dead and bad-column pixels, full sensor."""
        self._build_maps()
        return self._bad

    @property
    def hot_pixels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(y, x, dark current ADU/s) of the hot pixels (counting cameras: false counts/s)."""
        self._build_maps()
        return self._hot_y, self._hot_x, self._hot_rate

    def dark_current_map(self, exposure_s: float) -> np.ndarray:
        """Dark-current contribution (ADU) for ``exposure_s``, full sensor (hot pixels included)."""
        self._build_maps()
        d = np.full(self.model.sensor_shape, self.model.dark_current_adu_per_s * exposure_s, np.float32)
        d[self._hot_y, self._hot_x] += self._hot_rate * np.float32(exposure_s)
        return d

    def expected_dark(self, exposure_s: float, request: Optional[AcquisitionRequest] = None) -> np.ndarray:
        """Noise-free expected dark frame (float64, output shape); integrating cameras."""
        r = self.roi(request)
        bx, by = _binning(request)
        oh, ow = self.output_shape(request)
        d = (self.dark_map + self.dark_current_map(exposure_s))[r.y:r.y + oh * by, r.x:r.x + ow * bx]
        d = np.clip(d, 0, self.model.max_value)
        return d.reshape(oh, by, ow, bx).mean(axis=(1, 3), dtype=np.float64)

    def expected_gain(self, request: Optional[AcquisitionRequest] = None) -> np.ndarray:
        """Gain map binned to the output shape (float64)."""
        r = self.roi(request)
        bx, by = _binning(request)
        oh, ow = self.output_shape(request)
        g = self.gain_map[r.y:r.y + oh * by, r.x:r.x + ow * bx]
        return g.reshape(oh, by, ow, bx).mean(axis=(1, 3), dtype=np.float64)

    # --------------------------------------------------------------- expose
    def expose(self, flux, exposure_s: float, request: Optional[AcquisitionRequest] = None,
               frame_index: int = 0, *, ht_kv: Optional[float] = None,
               seed: Optional[int] = None, means=None) -> tuple[np.ndarray, dict]:
        """Simulate one raw frame.

        ``flux``: electrons / sensor pixel / second, a float array of the ROI
        shape (or of the binned output shape, or a scalar), or ``None`` for no
        beam (dark reference, blanked, retracted...).
        Returns ``(frame, info)``; the frame has :meth:`output_shape` and the
        model's dtype (uint16, or uint8 for counting cameras). ``info`` holds
        ``dose_e_per_px`` (mean electrons per sensor pixel), ``saturated``,
        ``saturated_pixels``, ``blanked`` and the resolved geometry.
        ``seed`` replaces the detector's own for this frame's noise (a pinned acquisition).
        ``means``: ``(lam, p0, dose, t_nb[, row_totals])``, the exposure's Poisson means already
        made by the renderer (``lam = flux * t_nb`` per grid pixel, ``p0 = exp(-lam)``, their sum,
        the ``t_nb`` they were made for and optionally each row's sum), as ``detector.fast.prepare``
        would make them; used by the fused integrating path when they match this frame (else
        ignored). The row totals are what an electron-by-electron (sparse) path needs.
        """
        self._build_maps()
        m = self.model
        request = request if request is not None else AcquisitionRequest(camera_model=m.name)
        ht = float(ht_kv if ht_kv is not None else self.ht_kv)
        t = max(0.0, float(exposure_s))
        r = self.roi(request)
        bx, by = _binning(request)
        oh, ow = self._grid_shape(request)
        sr = self.super_resolution(request)
        blanked = flux is None

        # --- choose the simulation grid
        if flux is not None and np.ndim(flux) == 0:
            flux = np.full((oh * by, ow * bx), float(flux), np.float32)
        if flux is None:
            direct = True
        else:
            flux = np.asarray(flux)
            if flux.ndim != 2:
                raise ValueError(f"flux must be 2-D, got shape {flux.shape}")
            if flux.shape == (oh, ow) and (bx, by) != (1, 1):
                direct = True
            elif flux.shape[0] >= oh * by and flux.shape[1] >= ow * bx:
                direct = False
                flux = flux[: oh * by, : ow * bx]
            else:
                raise ValueError(f"flux shape {flux.shape} matches neither the ROI "
                                 f"({r.h}, {r.w}) nor the output shape ({oh}, {ow})")
            if flux.dtype != np.float32:
                flux = flux.astype(np.float32)

        sl = (slice(r.y, r.y + oh * by), slice(r.x, r.x + ow * bx))
        if direct:
            gain, off = self._binned_maps(r, bx, by)
            nb, gbx, gby = bx * by, 1, 1
        else:
            gain, off = self._gain[sl], self._offset[sl]
            nb, gbx, gby = 1, bx, by

        # hot pixels inside the ROI, in grid coordinates
        hm = ((self._hot_y >= sl[0].start) & (self._hot_y < sl[0].stop)
              & (self._hot_x >= sl[1].start) & (self._hot_x < sl[1].stop))
        hy = (self._hot_y[hm] - r.y) // (by if direct else 1)
        hx = (self._hot_x[hm] - r.x) // (bx if direct else 1)
        hrate = self._hot_rate[hm] / np.float32(nb)

        gh, gw = gain.shape
        out = np.empty((2 * oh, 2 * ow) if sr else (oh, ow), m.dtype)
        band = max(8, _BAND_PIXELS // max(gw, 1))
        band = gby * max(1, band // gby)
        bands = [(i, r0, min(r0 + band, gh)) for i, r0 in enumerate(range(0, gh, band))]
        bd = request.bit_depth
        scale = 2.0 ** (min(bd, m.bit_depth) - m.bit_depth) if bd else 1.0
        vmax = float(min(m.max_value, (1 << bd) - 1) if bd else m.max_value)

        if means is not None and (flux is None or tuple(np.shape(means[0])) != tuple(flux.shape)
                                  or abs(float(means[3]) - t * nb) > 1e-9 * max(t * nb, 1e-30)):
            means = None  # made for another grid or exposure
        ctx = _Ctx(model=m, seed=self.seed if seed is None else int(seed), frame_index=int(frame_index), t=t,
                   flux=flux, means=means,
                   nb=nb, gain=gain, off=off, gbx=gbx, gby=gby, out=out, vmax=vmax, scale=scale,
                   hy=hy, hx=hx, hrate=hrate, S=None,
                   adu=self._adu_at(ht),
                   a=m.charge_spread_at(ht) / (bx if direct else 1))

        with self._lock:
            if m.hardware_counting and (sr or _fast_detector(m)):
                res = [self._expose_counting_fast(ctx, gh, gw, sr)]
            elif _fast_detector(m):
                res = [self._expose_fast(ctx, gh, gw)]
            elif m.hardware_counting:
                res = list(_pool().map(ctx.counting_band, bands))
            else:
                doses = [0.0] * len(bands)
                if flux is not None:
                    if self._S is None or self._S.shape != (gh, gw):
                        self._S = np.empty((gh, gw), np.float32)
                    ctx.S = self._S
                    doses = list(_pool().map(ctx.deposit_band, bands))
                sats = list(_pool().map(ctx.analog_band, bands))
                res = list(zip(doses, sats))

        dose_sum = sum(d for d, _ in res)
        sat = int(sum(s for _, s in res))
        dose = dose_sum / (gh * gw * nb) if flux is not None else 0.0
        info = {
            "dose_e_per_px": float(dose),
            "saturated": sat > 0,
            "saturated_pixels": sat,
            "blanked": bool(blanked),
            "frame_index": int(frame_index),
            "exposure_s": t,
            "roi": r,
            "binning": (bx, by),
            "ht_kv": ht,
            "counting": bool(m.hardware_counting),
            "super_resolution": sr,
        }
        return out, info

    def grid_exposure(self, exposure_s: float, request: Optional[AcquisitionRequest] = None, *,
                      ht_kv: Optional[float] = None) -> Optional["GridExposure"]:
        """What :meth:`expose` works out for an integrating frame whose flux (or Poisson means)
        is given on the output grid, independent of the flux: the maps, the noise parameters
        and the clip. A fused frame kernel (``de_twin.ebsd.frames``) makes the same frames from
        it. None where :meth:`expose` would take another path (a counting camera,
        super-resolution, the numpy reference detector, not 16-bit)."""
        from . import fast

        m = self.model
        request = request if request is not None else AcquisitionRequest(camera_model=m.name)
        if m.hardware_counting or self.super_resolution(request) or not _fast_detector(m) \
                or np.dtype(m.dtype) != np.uint16:
            return None
        self._build_maps()
        ht = float(ht_kv if ht_kv is not None else self.ht_kv)
        t = max(0.0, float(exposure_s))
        r = self.roi(request)
        bx, by = _binning(request)
        bd = request.bit_depth
        key = (m, self.seed, r.x, r.y, r.w, r.h, bx, by, t, bd, ht)
        held = getattr(self, "_grid_exposure", None)
        if held is not None and held.key == key:
            return held
        oh, ow = self._grid_shape(request)
        gain, off = self._binned_maps(r, bx, by)
        nb = bx * by
        scale = 2.0 ** (min(bd, m.bit_depth) - m.bit_depth) if bd else 1.0
        vmax = float(min(m.max_value, (1 << bd) - 1) if bd else m.max_value)
        hm = ((self._hot_y >= r.y) & (self._hot_y < r.y + oh * by)
              & (self._hot_x >= r.x) & (self._hot_x < r.x + ow * bx))
        hot = np.zeros((oh, ow), np.float32)  # as _hot_grid makes it
        if hm.any():
            np.add.at(hot, ((self._hot_y[hm] - r.y) // by, (self._hot_x[hm] - r.x) // bx),
                      self._hot_rate[hm] / np.float32(nb) * np.float32(t * scale))
        rows, cols = np.nonzero(hot)
        hot_ptr = np.zeros(oh + 1, np.int64)
        np.add.at(hot_ptr, rows + 1, 1)
        self._grid_exposure = GridExposure(
            key=key, shape=(oh, ow),
            gain=np.ascontiguousarray(gain, np.float32).ravel(),
            offset=(np.asarray(off, np.float32) + np.float32(m.dark_current_adu_per_s * t)).ravel(),
            hot_ptr=np.cumsum(hot_ptr), hot_col=cols.astype(np.int64), hot_val=hot[rows, cols],
            adu=self._adu_at(ht), k=1.0 / max(m.adu_cv, 1e-3) ** 2, a=float(m.charge_spread_at(ht) / bx),
            inv_nb=1.0 / nb, sigma=float(m.read_noise_adu / math.sqrt(nb)) if m.read_noise_adu > 0 else 0.0,
            scale=float(scale), vmax=vmax, t_nb=t * nb, seed=self.seed, table=fast.normal_table())
        return self._grid_exposure

    def _expose_fast(self, ctx: "_Ctx", gh: int, gw: int) -> tuple[float, int]:
        """One integrating frame through the fused numba kernels (`detector.fast`).
        Returns ``(dose, saturated pixels)`` like the band workers do."""
        from . import fast

        m = ctx.model
        table = fast.normal_table()
        zstart, nstart = fast.offsets(fast.frame_key(ctx.seed, ctx.frame_index, 0))
        dose = 0.0
        has = ctx.flux is not None
        if has:
            if self._S is None or self._S.shape != (gh, gw):
                self._S = np.empty((gh, gw), np.float32)
            k = 1.0 / max(m.adu_cv, 1e-3) ** 2
            if getattr(ctx, "means", None) is not None:
                lam, p0, dose = ctx.means[0], ctx.means[1], float(ctx.means[2])
            else:
                lam, p0, dose = self._exposure_means(ctx.flux, float(ctx.t * ctx.nb))
            lut = getattr(self, "_gamma_lut", None)
            if lut is None or lut[0] != k:
                lut = self._gamma_lut = (k, fast.gamma_lut(k))
            fast.deposit(lam, p0, float(ctx.adu), float(k),
                         np.uint64(fast.frame_key(ctx.seed, ctx.frame_index, 1)), self._S,
                         table, zstart, lut[1])
        S = self._S if has else ctx.gain  # unread without signal
        base = float(m.dark_current_adu_per_s * ctx.t)
        sigma = float(m.read_noise_adu / math.sqrt(ctx.nb)) if m.read_noise_adu > 0 else 0.0
        if ctx.gbx == 1 and ctx.gby == 1:
            hot = self._hot_grid(ctx, gh, gw)
            sat = fast.analog_out(S, has, float(ctx.a), 1.0 / ctx.nb, ctx.gain, ctx.off, base,
                                  sigma, float(ctx.scale), table, nstart, hot,
                                  float(ctx.vmax), ctx.out)
            return float(dose), int(sat)
        x = getattr(self, "_X", None)
        if x is None or x.shape != (gh, gw):
            x = self._X = np.empty((gh, gw), np.float32)
        fast.analog(S, has, float(ctx.a), 1.0 / ctx.nb, ctx.gain, ctx.off, base, sigma,
                    float(ctx.scale), table, nstart, x)
        if ctx.hy.size:
            np.add.at(x, (ctx.hy, ctx.hx), ctx.hrate * np.float32(ctx.t * ctx.scale))
        sat = fast.finish(x, float(ctx.vmax), int(ctx.gbx), int(ctx.gby), ctx.out)
        return float(dose), int(sat)

    def _expose_counting_fast(self, ctx: "_Ctx", gh: int, gw: int, sr: bool) -> tuple[float, int]:
        """One counting frame through the numba kernels (`fast.count`, `fast.count_super`)."""
        from . import fast

        m = ctx.model
        ncyc = max(1, int(round(ctx.t / m.full_frame_time_s)))
        cap = ncyc * ctx.nb
        mean, p0, dose = self._counting_means(ctx, gh, gw, ncyc)
        table = fast.normal_table()
        zstart, _ = fast.offsets(fast.frame_key(ctx.seed, ctx.frame_index, 0))
        key = np.uint64(fast.frame_key(ctx.seed, ctx.frame_index, 3))
        if sr:
            sat = fast.count_super(mean, p0, key, table, zstart, int(cap), float(ctx.vmax), ctx.out)
            return dose, int(sat)
        if ctx.gbx == 1 and ctx.gby == 1:
            sat = fast.count(mean, p0, key, table, zstart, int(cap), 1.0 / ctx.nb,
                             float(ctx.vmax), ctx.out, mean, False)
            return dose, int(sat)
        x = getattr(self, "_X", None)
        if x is None or x.shape != (gh, gw):
            x = self._X = np.empty((gh, gw), np.float32)
        fast.count(mean, p0, key, table, zstart, int(cap), 1.0 / ctx.nb, float(ctx.vmax),
                   ctx.out, x, True)
        sat = fast.finish(x, float(ctx.vmax), int(ctx.gbx), int(ctx.gby), ctx.out)
        return dose, int(sat)

    def _counting_means(self, ctx: "_Ctx", gh: int, gw: int, ncyc: int):
        """Each pixel's mean counts per frame and exp(-mean), made once per exposure."""
        from . import fast

        m = ctx.model
        flux = ctx.flux
        if flux is not None:
            owner = flux.base if flux.base is not None else flux
            fkey = (id(owner), flux.__array_interface__["data"][0], flux.shape, flux.strides)
        else:
            owner, fkey = None, None
        # `expose` slices the gain map too: a new view each frame, of the same memory
        gain = ctx.gain
        gowner = gain.base if gain.base is not None else gain
        gkey = (id(gowner), gain.__array_interface__["data"][0], gain.shape, gain.strides)
        key = (fkey, float(ctx.t), ctx.nb, ncyc, gkey, (gh, gw), ctx.hy.tobytes(), ctx.hx.tobytes())
        cached = getattr(self, "_cmeans", None)
        if cached is not None and cached[0] == key and cached[1] is owner and cached[2] is gowner:
            return cached[3], cached[4], cached[5]
        hot = np.zeros((gh, gw), np.float32)
        if ctx.hy.size:  # spurious counts per cycle
            np.add.at(hot, (ctx.hy, ctx.hx), ctx.hrate * np.float32(ctx.nb * m.full_frame_time_s))
        mean = np.empty((gh, gw), np.float32)
        p0 = np.empty((gh, gw), np.float32)
        dose = fast.counting_prepare(flux if flux is not None else ctx.gain, flux is not None,
                                     ctx.gain, float(ctx.t * ctx.nb), float(m.cluster_area_px),
                                     float(ncyc * ctx.nb), float(m.false_event_rate), hot,
                                     mean, p0)
        self._cmeans = (key, owner, gowner, mean, p0, float(dose))
        return mean, p0, float(dose)

    def _exposure_means(self, flux: np.ndarray, t_nb: float):
        """The per-pixel mean electrons and exp(-mean) of an exposure, shared by all its
        frames: the flux of an exposure is one array, so they are made once."""
        from . import fast

        # `expose` slices the flux, so each frame gets a new VIEW of the same array: key
        # on the array underneath (held, so its id cannot be reused) and the window.
        owner = flux.base if flux.base is not None else flux
        key = (id(owner), flux.__array_interface__["data"][0], flux.shape, flux.strides, t_nb)
        cached = getattr(self, "_means", None)
        if cached is not None and cached[0] == key and cached[1] is owner:
            return cached[2], cached[3], cached[4]
        lam = np.empty(flux.shape, np.float32)
        p0 = np.empty(flux.shape, np.float32)
        dose = fast.prepare(flux, t_nb, lam, p0)
        self._means = (key, owner, lam, p0, float(dose))
        return lam, p0, float(dose)

    def _hot_grid(self, ctx: "_Ctx", gh: int, gw: int) -> np.ndarray:
        """The hot pixels' ADU for this exposure as a grid (zero almost everywhere),
        reused between frames of the same geometry and exposure."""
        key = (gh, gw, float(ctx.t), float(ctx.scale), ctx.hy.tobytes(), ctx.hx.tobytes())
        cached = getattr(self, "_hot_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        hot = np.zeros((gh, gw), np.float32)
        if ctx.hy.size:
            np.add.at(hot, (ctx.hy, ctx.hx), ctx.hrate * np.float32(ctx.t * ctx.scale))
        self._hot_cache = (key, hot)
        return hot

    def _binned_maps(self, r: Roi, bx: int, by: int):
        key = (r.x, r.y, r.w, r.h, bx, by)
        hit = self._binned_cache.get(key)
        if hit is None:
            oh, ow = r.h // by, r.w // bx
            sl = (slice(r.y, r.y + oh * by), slice(r.x, r.x + ow * bx))
            if (bx, by) == (1, 1):
                hit = (self._gain[sl], self._offset[sl])
            else:
                g = self._gain[sl].reshape(oh, by, ow, bx).mean(axis=(1, 3), dtype=np.float32)
                o = self._offset[sl].reshape(oh, by, ow, bx).mean(axis=(1, 3), dtype=np.float32)
                hit = (g, o)
            if len(self._binned_cache) > 8:
                self._binned_cache.clear()
            self._binned_cache[key] = hit
        return hit


@dataclasses.dataclass(eq=False)
class GridExposure:
    """:meth:`Detector.grid_exposure`: one exposure's detector, on the output grid."""

    key: tuple
    shape: tuple  # (h, w) of the grid
    gain: np.ndarray  # flat float32, mean 1 (binned)
    offset: np.ndarray  # flat float32: dark offset + dark current of the exposure, ADU
    hot_ptr: np.ndarray  # hot pixels by grid row (CSR): row r's are hot_ptr[r] .. hot_ptr[r + 1]
    hot_col: np.ndarray
    hot_val: np.ndarray  # their dark current in the frame, ADU (x the bit-depth scale)
    adu: float  # ADU per electron
    k: float  # the deposit's gamma shape per electron, 1 / cv^2
    a: float  # charge sharing on the grid
    inv_nb: float  # 1 / sensor pixels per grid pixel
    sigma: float  # read noise per grid pixel, ADU
    scale: float  # bit-depth scale
    vmax: float  # clip
    t_nb: float  # exposure x sensor pixels per grid pixel: Poisson mean = flux x t_nb
    seed: int  # the detector's noise seed
    table: np.ndarray  # detector.fast.normal_table()


class _Ctx:
    """Per-frame state shared by the band workers.

    Large temporaries come from per-thread scratch buffers: fresh multi-MB
    allocations page-fault serially (notably on Windows) and kill thread scaling.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def _rng(self, band: int, stage: int) -> np.random.Generator:
        return rng_for(self.seed, SeedKind.SHOT_NOISE, self.frame_index, (band << 4) | stage)

    # ---- phase 1: electrons -> deposited ADU (before charge sharing)
    def deposit_band(self, b) -> float:
        i, r0, r1 = b
        m = self.model
        rng = self._rng(i, 1)
        src = self.flux[r0:r1]
        lam = _buf("lam", src.shape)
        np.multiply(src, np.float32(self.t * self.nb), out=lam)
        np.maximum(lam, np.float32(0.0), out=lam)
        dose = float(lam.sum(dtype=np.float64))
        n = _poisson(rng, lam, _buf("n", src.shape))
        # S | N ~ Gamma(N k, mu / k) by Wilson-Hilferty:
        #   S = N mu (1 - 1/(9a) + z / (3 sqrt a))^3,  a = N k,  k = 1 / cv^2
        k = np.float32(1.0 / max(m.adu_cv, 1e-3) ** 2)
        inv = _buf("inv", src.shape)
        np.maximum(n, np.float32(1.0), out=inv)
        inv *= k
        np.reciprocal(inv, out=inv)
        w = _buf("w", src.shape)
        np.sqrt(inv, out=w)
        z = _buf("z", src.shape)
        rng.standard_normal(dtype=np.float32, out=z)
        w *= z
        w *= np.float32(1.0 / 3.0)
        inv *= np.float32(1.0 / 9.0)
        w -= inv
        w += np.float32(1.0)
        np.maximum(w, np.float32(0.0), out=w)
        s = self.S[r0:r1]
        np.multiply(w, w, out=s)
        s *= w
        n *= np.float32(self.adu)
        s *= n
        return dose

    # ---- phase 2: charge sharing, gain, dark, read noise, clip, bin, quantise
    def analog_band(self, b) -> int:
        i, r0, r1 = b
        m = self.model
        rng = self._rng(i, 2)
        gain, off = self.gain[r0:r1], self.off[r0:r1]
        h, w = gain.shape
        x = _buf("x", (h, w))
        if self.S is not None:
            S = self.S
            a = float(self.a)
            if a > 0:
                ca = np.float32((1.0 - 2.0 * a) / a)
                # vertical: v = a * (up + down + S (1-2a)/a), nearest-edge
                v = _buf("v", (h, w))
                v[0] = S[r0 - 1] if r0 > 0 else S[r0]
                if h > 1:
                    v[1:] = S[r0:r1 - 1]
                    v[:-1] += S[r0 + 1:r1]
                v[-1] += S[r1] if r1 < S.shape[0] else S[r1 - 1]
                tmp = _buf("tmp", (h, w))
                np.multiply(S[r0:r1], ca, out=tmp)
                v += tmp
                v *= np.float32(a)
                # horizontal, same kernel
                x[:, 0] = v[:, 0]
                if w > 1:
                    x[:, 1:] = v[:, :-1]
                    x[:, :-1] += v[:, 1:]
                x[:, -1] += v[:, -1]
                np.multiply(v, ca, out=tmp)
                x += tmp
                x *= np.float32(a)
            else:
                np.copyto(x, S[r0:r1])
            if self.nb > 1:
                x *= np.float32(1.0 / self.nb)
            x *= gain
            x += off
        else:
            np.copyto(x, off)
        base = np.float32(m.dark_current_adu_per_s * self.t)
        if base:
            x += base
        sel = (self.hy >= r0) & (self.hy < r1)
        if sel.any():
            np.add.at(x, (self.hy[sel] - r0, self.hx[sel]), self.hrate[sel] * np.float32(self.t))
        if m.read_noise_adu > 0:
            z = _buf("z", (h, w))
            rng.standard_normal(dtype=np.float32, out=z)
            z *= np.float32(m.read_noise_adu / math.sqrt(self.nb))
            x += z
        if self.scale != 1.0:
            x *= np.float32(self.scale)
        return self._finish(x, r0, r1)

    def _finish(self, x: np.ndarray, r0: int, r1: int) -> int:
        """Clip, bin (average), round into the output; returns the clipped-pixel count."""
        h, w = x.shape
        ge = _buf("ge", (h, w), bool)
        np.greater_equal(x, np.float32(self.vmax), out=ge)
        sat = int(np.count_nonzero(ge))
        np.clip(x, np.float32(0.0), np.float32(self.vmax), out=x)
        bx, by = self.gbx, self.gby
        if bx > 1 or by > 1:
            x = x.reshape(h // by, by, w // bx, bx).mean(axis=(1, 3), dtype=np.float32)
        x += np.float32(0.5)
        self.out[r0 // by:r1 // by] = x  # truncation after +0.5 == rounding (x >= 0)
        return sat

    # ---- counting cameras: counted events per pixel
    def counting_band(self, b) -> tuple[float, int]:
        i, r0, r1 = b
        m = self.model
        rng = self._rng(i, 3)
        gain = self.gain[r0:r1]
        h, w = gain.shape
        ncyc = max(1, int(round(self.t / m.full_frame_time_s)))
        dose = 0.0
        p = _buf("p_c", (h, w))
        if self.flux is not None:
            np.multiply(self.flux[r0:r1], np.float32(self.t * self.nb), out=p)
            np.maximum(p, np.float32(0.0), out=p)
            dose = float(p.sum(dtype=np.float64))
            p *= gain
            A = np.float32(m.cluster_area_px)
            p *= np.float32(-A / (ncyc * self.nb))  # -lam A, lam per sensor pixel per cycle
            np.expm1(p, out=p)
            p *= np.float32(-1.0 / A)  # (1 - exp(-lam A)) / A
        else:
            p.fill(0.0)
        p += np.float32(m.false_event_rate)
        sel = (self.hy >= r0) & (self.hy < r1)
        if sel.any():
            np.add.at(p, (self.hy[sel] - r0, self.hx[sel]),
                      self.hrate[sel] * np.float32(self.nb * m.full_frame_time_s))
        np.minimum(p, np.float32(1.0), out=p)
        p *= np.float32(ncyc * self.nb)
        counts = _poisson(rng, p, _buf("n", (h, w)))
        np.minimum(counts, np.float32(ncyc * self.nb), out=counts)
        if self.nb > 1:  # binned grid: average of the summed sensor pixels
            counts *= np.float32(1.0 / self.nb)
        return dose, self._finish(counts, r0, r1)


_TLS = threading.local()


def _fast_detector(model) -> bool:
    """The fused numba kernels, unless numba is missing or ``DE_TWIN_NUMPY_DETECTOR=1``
    asks for the numpy path (a reference for comparisons). Super-resolution frames are
    numba only."""
    if os.environ.get("DE_TWIN_NUMPY_DETECTOR") == "1":
        return False
    from . import fast

    return fast.AVAILABLE


def _buf(name: str, shape, dtype=np.float32) -> np.ndarray:
    """A per-thread reusable scratch array (contiguous view of a growable flat buffer)."""
    d = getattr(_TLS, "bufs", None)
    if d is None:
        d = _TLS.bufs = {}
    size = int(np.prod(shape))
    key = (name, np.dtype(dtype).str)
    flat = d.get(key)
    if flat is None or flat.size < size:
        flat = d[key] = np.empty(size, dtype)
    return flat[:size].reshape(shape)


def _poisson(rng: np.random.Generator, lam: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Poisson draws as float32 (see module docstring for the regimes)."""
    if out is None:
        out = np.empty(lam.shape, np.float32)
    lmax = float(lam.max()) if lam.size else 0.0
    if lmax < _POISSON_INV_CUT:
        return _poisson_inversion(rng, lam, out)
    lo = lam < _POISSON_INV_CUT
    if lo.any():
        out[lo] = _poisson_inversion(rng, lam[lo], None)
    mid = ~lo & (lam < _POISSON_GAUSS_CUT)
    if mid.any():
        out[mid] = rng.poisson(lam[mid])
    hi = lam >= _POISSON_GAUSS_CUT
    if hi.any():
        lh = lam[hi]
        out[hi] = np.maximum(np.rint(lh + np.sqrt(lh) * rng.standard_normal(lh.shape, dtype=np.float32)), 0)
    return out


def _poisson_inversion(rng: np.random.Generator, lam: np.ndarray, out: Optional[np.ndarray]) -> np.ndarray:
    """Exact inversion sampler N = #{k >= 1 : U > F(k-1)}; compacts the still-active set."""
    shape, size = lam.shape, lam.size
    lam = lam.reshape(-1)
    n_all = (out if out is not None else np.empty(shape, np.float32)).reshape(-1)
    n_all.fill(0.0)
    u = _buf("pi_u", (size,))
    rng.random(dtype=np.float32, out=u)
    p = _buf("pi_p", (size,))
    np.negative(lam, out=p)
    np.exp(p, out=p)
    f = _buf("pi_f", (size,))
    np.copyto(f, p)
    mbuf = _buf("pi_m", (size,), bool)
    n, idx, lsub = n_all, None, lam
    for k in range(1, 128):
        mm = mbuf[: u.size]
        np.greater(u, f, out=mm)
        cnt = int(np.count_nonzero(mm))
        if cnt == 0:
            break
        if cnt < 0.3 * mm.size and mm.size > 2048:
            sel = np.flatnonzero(mm)
            if idx is None:
                n, idx = n_all[sel], sel
            else:
                n_all[idx] = n
                n, idx = n[sel], idx[sel]
            u, f, p, lsub = u[sel], f[sel], p[sel], lsub[sel]
            n += np.float32(1.0)
        else:
            n += mm
        p *= lsub
        p *= np.float32(1.0 / k)
        f += p
    if idx is not None:
        n_all[idx] = n
    return n_all.reshape(shape)
