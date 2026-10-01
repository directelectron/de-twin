"""``Renderer``: dispatch on ``OpticsState.render_mode`` and own every cache.

Caches
------
* FieldMaps, keyed by (view, layers, specimen generation, time bucket). A
  specimen is treated as time dependent when it has a truthy ``time_dependent``
  attribute (or ``is_time_dependent()``); time is then bucketed by
  ``RenderConfig.time_quantum_s``. ``generation`` (attribute) and
  ``FieldMap.generation`` are both honoured. The renderer never calls
  ``Specimen.update`` - the twin's clock does.
* TEM: exit-wave spectra per FieldMap and transfer functions per aberration
  state, plus a one-frame output cache (C++ frame cache).
* Diffraction: the byte-budgeted :class:`DiffractionCache` (STEM patterns),
  per-FieldMap STEM key tables and SAED composites. The reciprocal-lattice libraries
  (:func:`de_twin.crystal.library_for`) are process-wide.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np

from ..specimen.fieldmap import LAYER_DESCAN, LAYER_STRAIN, FieldMap, GrainTable
from ..state import RenderMode
from .coherent import CoherentStem, choose_model
from .config import RenderConfig
from .diffraction import DiffractionCache
from .saed import SaedRenderer
from .stem import StemRenderer, resolve_scan_point
from .tem import TransferCache, render_tem

TEM_LAYERS = frozenset()
SAED_LAYERS = frozenset()
STEM_LAYERS = frozenset({LAYER_DESCAN, LAYER_STRAIN})


#: Precession phases are quantised to this many steps of the cone, so frames that cover
#: only part of it share a handful of cached patterns.
PRECESSION_PHASES = 24


def _precession_frame(optics, time_s: float):
    """The optics of one frame under precession: the sweep's phase at its start and the
    arc it covers in the exposure. A frame shorter than a precession period sees part of
    the cone, as a real one does. (`Renderer.datacube`, `virtual_image` and `ground_truth`
    are noiseless summaries: they show the whole cone.)"""
    theta = float(getattr(optics, "precession_mrad", 0.0))
    hz = float(getattr(optics, "precession_hz", 0.0))
    if theta <= 0.0 or hz <= 0.0:
        return optics
    exposure = float(optics.extras.get("exposure_s", 0.0) or 0.0)
    arc = 2.0 * math.pi * hz * exposure if exposure > 0 else 2.0 * math.pi
    if arc >= 2.0 * math.pi:
        return dataclasses.replace(optics, precession_phase_rad=0.0, precession_arc_rad=2.0 * math.pi)
    step = 2.0 * math.pi / PRECESSION_PHASES
    phase = (2.0 * math.pi * hz * float(time_s)) % (2.0 * math.pi)
    arc = max(step, round(arc / step) * step)
    return dataclasses.replace(optics, precession_phase_rad=round(phase / step) * step, precession_arc_rad=arc)


def _lattice(view, x: float, y: float) -> tuple[float, float]:
    """(column, row) of world point (x, y) on the view's world-fixed pixel lattice: pixel
    units along the view's own (rotated, flipped, foreshortened) axes from the origin."""
    rot = float(getattr(view, "rotation_rad", 0.0))
    if rot:
        c, s = math.cos(rot), math.sin(rot)
        x, y = c * x + s * y, -s * x + c * y
    sx = -1.0 if getattr(view, "flip_x", False) else 1.0
    sy = -1.0 if getattr(view, "flip_y", False) else 1.0
    pu = view.pixel_um
    return sx * x * max(view.cos_beta, 0.1) / pu, sy * y * max(view.cos_alpha, 0.1) / pu


def _from_lattice(view, lc: float, lr: float) -> tuple[float, float]:
    """The world point at lattice (column, row) ``(lc, lr)`` (inverse of `_lattice`)."""
    sx = -1.0 if getattr(view, "flip_x", False) else 1.0
    sy = -1.0 if getattr(view, "flip_y", False) else 1.0
    pu = view.pixel_um
    x = sx * lc * pu / max(view.cos_beta, 0.1)
    y = sy * lr * pu / max(view.cos_alpha, 0.1)
    rot = float(getattr(view, "rotation_rad", 0.0))
    if rot:
        c, s = math.cos(rot), math.sin(rot)
        x, y = c * x - s * y, s * x + c * y
    return (float(x), float(y))


def _axis_aligned(view) -> bool:
    """Whether the view's rows run along a world axis. Only then is a raster exactly the
    rasters of its pieces: the specimen paints overlapping particles window by window, and
    a rotated view's windows, clipped at a piece's edge, change which particle claims a
    pixel where two overlap (a few pixels in a thousand), so rotated views are rasterised
    whole."""
    rot = float(getattr(view, "rotation_rad", 0.0))
    return abs(math.sin(2.0 * rot)) < 1e-9


def _pad_len(n: int, margin: float, margin_px: int | None = None) -> int:
    """``n`` plus ``margin`` of it (or ``margin_px`` pixels) on both sides, rounded up
    (keeping the parity of ``n``, so the padded raster's pixels sit on the view's lattice) to
    a fast FFT length."""
    from scipy.fft import next_fast_len

    p = n + 2 * (int(margin_px) if margin_px is not None else int(round(margin * n)))
    while next_fast_len(p) != p:
        p += 2
    return p


#: Guard bands (pixels a side) the adaptive padding picks from: a few coarse sizes, so a focus or
#: magnification change mostly keeps the padded raster's shape (and its field map).
GUARD_LEVELS_PX = (64, 128, 256, 512)


#: The guard band holds all but this fraction of the objective's point-spread energy, with
#: a safety factor (calibrated on Au on carbon, the strongest contrast, against a 1024-pixel
#: guard: <5e-4 RMS at 3k-25k, <1e-3 to 100k; see tests/test_render_caches.py).
GUARD_LEAK = 1e-4
GUARD_FACTOR = 1.3
_SPREAD_CACHE: "OrderedDict[tuple, float]" = OrderedDict()
_RADII: dict = {}


def transfer_spread_px(optics, leak: float = GUARD_LEAK, N: int = 512) -> float:
    """The radius (raster pixels) holding all but ``leak`` of the energy of the objective
    transfer's point-spread function |F^-1 H|^2 on this sampling (measured on an N^2 grid, so
    at most ~0.4 N): how far the image of a point spreads, i.e. how wide a guard band the
    periodic FFT needs for the view not to see the far edge of the raster wrapped in.
    Cached per transfer function."""
    from scipy import fft as sfft

    from ..optics.state import image_aberrations_of
    from .tem import _raster_tilt, transfer_function

    view = optics.view
    key = (view.pixel_um, optics.wavelength_nm, image_aberrations_of(optics).key(), _raster_tilt(optics),
           optics.illumination_mrad, optics.focal_spread_nm, optics.objective_aperture_mrad, float(leak), N)
    hit = _SPREAD_CACHE.get(key)
    if hit is not None:
        return hit
    H = transfer_function((N, N), view.pixel_um * 1000.0, optics)
    r = 0.0
    if H is not None:
        rad = _RADII.get(N)
        if rad is None:
            yy = np.minimum(np.arange(N), N - np.arange(N))
            rad = _RADII[N] = np.hypot(yy[:, None], yy[None, :]).astype(np.int64).ravel()
        psf = np.abs(sfft.ifft2(H, workers=-1)).ravel() ** 2
        cum = np.cumsum(np.bincount(rad, weights=psf))
        r = float(np.searchsorted(cum / cum[-1], 1.0 - leak))
    _SPREAD_CACHE[key] = r
    while len(_SPREAD_CACHE) > 64:
        _SPREAD_CACHE.popitem(last=False)
    return r


def _guard_px(optics, minimum: int, maximum: int) -> int:
    need = max(float(minimum), GUARD_FACTOR * transfer_spread_px(optics))
    for g in GUARD_LEVELS_PX:
        if g >= need or g >= maximum:
            return min(g, maximum)
    return maximum


def _sub_view(view, r0: int, r1: int, c0: int, c1: int):
    """Rows r0:r1, columns c0:c1 of ``view`` as a view of their own."""
    x, y = view.pixel_to_world(np.array([(r0 + r1 - 1) / 2.0]), np.array([(c0 + c1 - 1) / 2.0]))
    return dataclasses.replace(view, center_um=(float(x[0]), float(y[0])), shape=(r1 - r0, c1 - c0))


#: A crop whose offset is this close to a whole number of raster pixels is not resampled.
SUBPIXEL_TOLERANCE_PX = 0.02


def _cubic_weights(t: float) -> np.ndarray:
    """Keys cubic convolution (a = -1/2) weights of the samples at -1, 0, 1, 2 for a point
    a fraction ``t`` in [0, 1) past sample 0: interpolating, and exact for quadratics."""
    return np.array([((-0.5 * t + 1.0) * t - 0.5) * t,
                     (1.5 * t - 2.5) * t * t + 1.0,
                     ((-1.5 * t + 2.0) * t + 0.5) * t,
                     (0.5 * t - 0.5) * t * t], np.float32)


def _crop(raster, center_um, view):
    """The view's pixels out of a padded raster whose middle shows ``center_um``, or None
    when the view is not inside it. An offset of a fraction of a pixel is interpolated
    (separable cubic convolution), so a stage move inside the margin moves the image as
    far as a fresh render would, not to the nearest raster pixel."""
    from .util import cubic_resample

    ny, nx = view.shape
    py, px = raster.shape
    lc, lr = _lattice(view, *view.center_um)
    oc, orow = _lattice(view, *center_um)
    fc = (px - nx) / 2 + (lc - oc)
    fr = (py - ny) / 2 + (lr - orow)
    c0, r0 = int(round(fc)), int(round(fr))
    if r0 < 0 or c0 < 0 or r0 + ny > py or c0 + nx > px:
        return None
    if abs(fc - c0) < SUBPIXEL_TOLERANCE_PX and abs(fr - r0) < SUBPIXEL_TOLERANCE_PX:
        return raster[r0:r0 + ny, c0:c0 + nx]
    # output pixel j samples the raster at f + j: taps at floor(f) - 1 .. floor(f) + 2
    br, bc = math.floor(fr), math.floor(fc)
    wr, wc = _cubic_weights(fr - br), _cubic_weights(fc - bc)
    # the rows and columns the taps need (edge-clamped where the view touches the raster edge)
    if br >= 1 and bc >= 1 and br + ny + 2 <= py and bc + nx + 2 <= px:
        src = raster[br - 1:br + ny + 2, bc - 1:bc + nx + 2]
    else:
        src = raster[np.clip(np.arange(br - 1, br + ny + 2), 0, py - 1)][
            :, np.clip(np.arange(bc - 1, bc + nx + 2), 0, px - 1)]
    return cubic_resample(src, wr, wc, (ny, nx))


class Renderer:
    def __init__(self, specimen, config: Optional[RenderConfig] = None):
        self.specimen = specimen
        self.config = config or RenderConfig()
        self.cache = DiffractionCache(self.config.diffraction_cache_mb)
        self._fieldmaps: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._preview_fieldmaps: "OrderedDict[tuple, tuple]" = OrderedDict()
        # the coherent STEM engine's fine views: their own store, so the transmission tiles of a
        # sparse scan (one per point) do not push out the scan's own field map
        self._fine_fieldmaps: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._tokens = itertools.count(1)
        self._tem_cache = TransferCache()
        self._frame: Optional[tuple] = None
        self._saed = SaedRenderer(self.config)
        self._stem = StemRenderer(self.cache, self.config, self.seed)
        self._coherent = CoherentStem(self.cache, self.config, self.seed)
        self._grains_fallback: Optional[GrainTable] = None
        self._lock = threading.RLock()  # render() and a prefetching thread share the caches
        self._prefetch_tem = TransferCache(size=1)
        self._inflight = None  # (optics, threading.Event) of the view being prefetched
        self.prefetched = 0
        self.rasters_built = 0
        self.rasters_reused = 0  # of rasters_built: assembled from a cached field map + new strips
        self.frames_from_cache = 0

    # ----------------------------------------------------------- specimen
    @property
    def seed(self) -> int:
        cfg = getattr(self.specimen, "config", None)
        return int(getattr(cfg, "seed", 42) or 42)

    @property
    def grains(self) -> Optional[GrainTable]:
        g = getattr(self.specimen, "grains", None)
        if g is None:
            if self._grains_fallback is None:
                self._grains_fallback = GrainTable.generate(self.seed)
            g = self._grains_fallback
        return g

    @property
    def crystallinity(self):
        fn = getattr(self.specimen, "crystallinity", None)
        return fn if callable(fn) else None

    def _time_dependent(self) -> bool:
        td = getattr(self.specimen, "time_dependent", None)
        if callable(td):
            td = td()
        if td is None:
            fn = getattr(self.specimen, "is_time_dependent", None)
            td = fn() if callable(fn) else False
        return bool(td)

    def field_map(self, optics, layers: frozenset = frozenset(), time_s: float = 0.0, store=None):
        """(FieldMap, token): rasterise only when view / layers / generation / time change.
        ``store``: the cache to use (default the renderer's; previews keep their own, so
        they do not push out the full-resolution field maps a stage move reuses)."""
        store = self._fieldmaps if store is None else store
        tq = self.config.time_quantum_s
        tkey = round(time_s / tq) if (self._time_dependent() and tq > 0) else 0
        gen = getattr(self.specimen, "generation", 0)
        key = (optics.view, frozenset(layers), gen, tkey)
        hit = store.get(key)
        if hit is not None:
            fm, token, fgen = hit
            if getattr(fm, "generation", 0) == fgen:
                store.move_to_end(key)
                return fm, token
        fm = self._rasterize(optics.view, frozenset(layers))
        self.rasters_built += 1
        token = (next(self._tokens), getattr(fm, "generation", 0))
        self._keep_field_map(store, key, fm, token)
        return fm, token

    def _rasterize(self, view, layers: frozenset, span_px=None):
        """``specimen.rasterize``: in parallel strips (``RenderConfig.raster_threads``) and,
        for a piece of a larger view, drawn as that view (``span_px``), when the specimen's
        raster is pixel-local."""
        if getattr(self.specimen, "raster_is_local", False) and _axis_aligned(view):
            return self.specimen.rasterize(view, layers, span_px=span_px,
                                           threads=max(1, int(self.config.raster_threads)))
        return self.specimen.rasterize(view, layers)

    def _keep_field_map(self, store, key, fm, token, size: Optional[int] = None) -> None:
        store[key] = (fm, token, getattr(fm, "generation", 0))
        store.move_to_end(key)
        while len(store) > max(1, self.config.fieldmap_cache_size if size is None else size):
            store.popitem(last=False)

    def _render_tem_panned(self, optics, time_s: float) -> np.ndarray:
        """TEM imaging through a padded raster, cropped to the view.

        The raster is rendered ``pan_margin`` of the field larger on every side, centred on
        the pixel-lattice point nearest the view's centre, and kept (the last
        ``pan_cache_size``), keyed by everything about the view EXCEPT where it is; a later
        view at the same magnification, focus and so on whose centre is within the margin
        is a crop of it, to the nearest raster pixel. Every raster sits on the same
        world-fixed pixel lattice, so the crop is the one a fresh render of that view gives
        (up to the transfer's periodic edges), and a new raster takes the part of the
        specimen it shares with a recent one from that one's field map (see
        `_panned_field_map`). The illumination disc and the upsampling are applied after
        the crop — they are fixed on the detector, not the specimen.
        """
        from .tem import finish_tem, render_tem_raster

        tq = self.config.time_quantum_s
        tkey = round(time_s / tq) if (self._time_dependent() and tq > 0) else 0
        gen = getattr(self.specimen, "generation", 0)
        last = getattr(self, "_pan_out", None)
        if last is not None and last[0] == optics and last[1] == (tkey, gen):
            self.frames_from_cache += 1
            return last[2]  # the same view again: every frame of an exposure
        view = optics.view
        base = dataclasses.replace(optics, view=dataclasses.replace(view, center_um=(0.0, 0.0)), beam_offset_px=(0.0, 0.0))
        key = (base, self.config.tem_model, gen, tkey)
        img = self._crop_pan(key, view)
        if img is None and self.config.interactive:
            import time as _time

            state = (optics, tkey, gen)
            now = _time.monotonic()
            last = getattr(self, "_last_req", None)
            if last is None:
                self._last_req, self._changed_at = state, -1e9  # first view: full
            elif last != state:
                self._last_req, self._changed_at = state, now
            if now - self._changed_at < self.config.settle_s:
                return self._render_preview(optics, time_s, tkey, gen)
        if img is None:
            poptics, shift = self._padded_optics(optics)
            fm, token = self._panned_field_map(poptics, time_s)
            raster = render_tem_raster(fm, poptics, self.grains, self.crystallinity, self.config,
                                       self.seed, self._tem_cache, token, shift)
            self._keep_pan("_pans", key, view.center_um, raster, poptics.view)
            img = self._crop_pan(key, view)
        else:
            self.frames_from_cache += 1
        out = finish_tem(img, optics, self.config)
        out.flags.writeable = False  # shared by every frame of an exposure; see render()
        self._pan_out = (optics, (tkey, gen), out)
        return out

    def _render_preview(self, optics, time_s: float, tkey, gen) -> np.ndarray:
        """The view at up to ``preview_side`` squared — the frame shown while it is still
        moving — with its own padded crop cache, so a drag between previews is a crop."""
        from .tem import finish_tem, render_tem_raster

        view = optics.view
        ny, nx = view.shape
        f = 1
        while (ny // (2 * f)) * (nx // (2 * f)) >= self.config.preview_side ** 2 \
                and ny % (2 * f) == 0 and nx % (2 * f) == 0:
            f *= 2
        if f == 1:
            self._changed_at = -1e9  # already small: the full render is the preview
            return self._render_tem_panned(optics, time_s)
        pview = dataclasses.replace(view, shape=(ny // f, nx // f), pixel_um=view.pixel_um * f)
        poptics = dataclasses.replace(
            optics, view=pview, raster_downsample=max(1, optics.raster_downsample) * f,
            blur_sigma_px=optics.blur_sigma_px / f, fresnel_sigma_px=optics.fresnel_sigma_px / f,
            beam_offset_px=tuple(v / f for v in getattr(optics, "beam_offset_px", (0.0, 0.0))))
        pbase = dataclasses.replace(poptics, view=dataclasses.replace(pview, center_um=(0.0, 0.0)),
                                    beam_offset_px=(0.0, 0.0))
        pkey = ("preview", pbase, self.config.tem_model, gen, tkey)
        img = self._crop_pan(pkey, pview, "_pan_previews")
        if img is None:
            pad_optics, shift = self._padded_optics(poptics)
            fm, token = self._panned_field_map(pad_optics, time_s, self._preview_fieldmaps)
            raster = render_tem_raster(fm, pad_optics, self.grains, self.crystallinity,
                                       self.config, self.seed, self._tem_cache, token, shift)
            self._keep_pan("_pan_previews", pkey, pview.center_um, raster, pad_optics.view)
            img = self._crop_pan(pkey, pview, "_pan_previews")
        self.previews_rendered = getattr(self, "previews_rendered", 0) + 1
        out = finish_tem(img, poptics, self.config)
        out.flags.writeable = False
        return out

    def _padded_optics(self, optics):
        """(``optics`` over the padded raster, sub-pixel shift): ``pan_margin`` more on every
        side (rounded up to a fast FFT size of the same parity), centred on the lattice
        point nearest the view's centre, and the (rows, cols) translation that puts the
        view's own centre back in the middle of it."""
        view = optics.view
        if self.config.adaptive_margin:  # a guard band as wide as the transfer delocalises
            g = _guard_px(optics, self.config.min_margin_px, self.config.max_margin_px)
            py, px = (_pad_len(n, 0.0, g) for n in view.shape)
        else:
            py, px = (_pad_len(n, self.config.pan_margin) for n in view.shape)
        lc, lr = _lattice(view, *view.center_um)
        kc, kr = round(lc), round(lr)
        pview = dataclasses.replace(view, shape=(py, px), center_um=_from_lattice(view, kc, kr))
        d = max(1, optics.raster_downsample)
        return (dataclasses.replace(optics, view=pview, output_shape=(py * d, px * d)),
                (-(lr - kr), -(lc - kc)))

    def _panned_field_map(self, optics, time_s: float, store=None):
        """`field_map` of a padded (lattice-centred) view that takes the part it shares with
        a cached field map of the same sampling from that one and rasterises only the rest:
        a stage move rasterises the strips of specimen it brings into view. Only for a
        specimen whose raster is pixel-local (``raster_is_local``), for which the pieces
        are the pixels a whole raster has."""
        store = self._fieldmaps if store is None else store
        if (not getattr(self.specimen, "raster_is_local", False) or not self.config.reuse_rasters
                or not _axis_aligned(optics.view)):
            return self.field_map(optics, TEM_LAYERS, time_s, store)
        tq = self.config.time_quantum_s
        tkey = round(time_s / tq) if (self._time_dependent() and tq > 0) else 0
        gen = getattr(self.specimen, "generation", 0)
        view = optics.view
        if (view, frozenset(TEM_LAYERS), gen, tkey) in store:
            return self.field_map(optics, TEM_LAYERS, time_s, store)
        ny, nx = view.shape
        base = dataclasses.replace(view, center_um=(0.0, 0.0), shape=(1, 1))
        lc, lr = _lattice(view, *view.center_um)
        best, best_area = None, 0
        for (v, layers, g, t), (fm, _, fgen) in store.items():
            if layers or g != gen or t != tkey or getattr(fm, "generation", 0) != fgen:
                continue
            if dataclasses.replace(v, center_um=(0.0, 0.0), shape=(1, 1)) != base:
                continue
            oy, ox = v.shape  # a cached raster of another size (another guard band) serves too
            if (oy - ny) % 2 or (ox - nx) % 2:
                continue
            oc, orow = _lattice(view, *v.center_um)
            dc, dr = lc - oc, lr - orow
            if abs(dc - round(dc)) > 1e-3 or abs(dr - round(dr)) > 1e-3:
                continue  # not on the same lattice
            dc = int(round(dc)) + (ox - nx) // 2  # new pixel (r, c) is old pixel (r + dr, c + dc)
            dr = int(round(dr)) + (oy - ny) // 2
            area = (max(0, min(ny, oy - dr) - max(0, -dr))) * (max(0, min(nx, ox - dc) - max(0, -dc)))
            if area > best_area:
                best, best_area = (fm, dr, dc, oy, ox), area
        if best is None or best_area < self.config.reuse_min_overlap * ny * nx:
            return self.field_map(optics, TEM_LAYERS, time_s, store)
        old, dr, dc, oy, ox = best
        r_lo, r_hi = max(0, -dr), min(ny, oy - dr)
        c_lo, c_hi = max(0, -dc), min(nx, ox - dc)
        parts = [((r_lo, r_hi, c_lo, c_hi), old, (r_lo + dr, c_lo + dc))]
        rects = ((0, r_lo, 0, nx), (r_hi, ny, 0, nx), (r_lo, r_hi, 0, c_lo), (r_lo, r_hi, c_hi, nx))
        for r0, r1, c0, c1 in rects:
            if r1 > r0 and c1 > c0:
                sub = self._rasterize(_sub_view(view, r0, r1, c0, c1), frozenset(TEM_LAYERS), max(ny, nx))
                parts.append(((r0, r1, c0, c1), sub, (0, 0)))
        last = parts[-1][1]
        from .. import buffers

        fm = FieldMap(view=view, material_id=buffers.zeros((ny, nx), np.uint8),
                      thickness_nm=buffers.zeros((ny, nx), np.float32),
                      grain_id=buffers.full((ny, nx), -1, np.int32),
                      generation=getattr(last, "generation", 0), time_s=getattr(last, "time_s", 0.0))
        if any(p.under_thickness_nm is not None for _, p, _ in parts):
            fm.under_material = buffers.zeros((ny, nx), np.uint8)
            fm.under_thickness_nm = buffers.zeros((ny, nx), np.float32)
        for (r0, r1, c0, c1), src, (sr, sc) in parts:
            h, w = r1 - r0, c1 - c0
            for name in ("material_id", "thickness_nm", "grain_id", "under_material", "under_thickness_nm"):
                a = getattr(src, name)
                if a is not None:
                    getattr(fm, name)[r0:r1, c0:c1] = a[sr:sr + h, sc:sc + w]
        self.rasters_built += 1
        self.rasters_reused += 1
        token = (next(self._tokens), fm.generation)
        self._keep_field_map(store, (view, frozenset(TEM_LAYERS), gen, tkey), fm, token)
        return fm, token

    def _keep_pan(self, slot: str, key, center_um, raster, padded_view=None) -> None:
        pans = getattr(self, slot, None) or []
        pans.insert(0, (key, center_um, raster, padded_view))
        setattr(self, slot, pans[:max(1, self.config.pan_cache_size)])

    def _crop_pan(self, key, view, slot: str = "_pans"):
        pans = getattr(self, slot, None) or []
        for i, (k, c0, raster, pview) in enumerate(pans):
            if k != key:
                continue
            img = _crop(raster, c0, view)
            if img is not None:
                pans.insert(0, pans.pop(i))
                # the raster's field map is in use again: keep it (a focus or tilt change of
                # this view re-renders from it without rasterising)
                for store in (self._fieldmaps, self._preview_fieldmaps):
                    for fk in [fk for fk in store if fk[0] == pview]:
                        store.move_to_end(fk)
                return img
        return None

    def invalidate(self) -> None:
        """Drop every cached raster, frame and pattern (e.g. after the specimen changed)."""
        with self._lock:
            self._invalidate()

    def _invalidate(self) -> None:
        self._pans = []
        self._pan_out = None
        self._pan_previews = []
        self._last_req = None
        self._fieldmaps.clear()
        self._preview_fieldmaps.clear()
        self._fine_fieldmaps.clear()
        self._tem_cache = TransferCache()
        self._frame = None
        self.cache.clear()
        self.cache.set_budget_mb(self.config.diffraction_cache_mb)
        self._saed = SaedRenderer(self.config)
        self._stem = StemRenderer(self.cache, self.config, self.seed)
        self._coherent = CoherentStem(self.cache, self.config, self.seed)
        self._grains_fallback = None

    # -------------------------------------------------------------- render
    def render(self, optics, *, frame_index: int = 0, scan_point=None, time_s: float = 0.0) -> np.ndarray:
        """float32 (ny, nx) = optics.output_shape, electrons / detector pixel / second.

        Returns zeros when the beam is blanked (the caller may map that to ``flux=None``).
        """
        inflight = self._inflight
        if inflight is not None and inflight[0] == optics:
            inflight[1].wait(10.0)  # being prefetched: its result is a crop in a moment
        with self._lock:
            return self._render(optics, frame_index=frame_index, scan_point=scan_point, time_s=time_s)

    def prefetch(self, optics, time_s: float = 0.0) -> bool:
        """Render the TEM view ``optics`` into the caches, off the caller's thread: the padded
        raster and its field map, so asking for that view later is a crop. The heavy work
        runs without the renderer's lock (a live frame is not held up); only looking up and
        storing take it. Coherent 4D-STEM: the probe, transmission tile and first block of
        patterns. Returns whether anything was rendered."""
        if (not optics.beam_blanked and optics.render_mode in (RenderMode.STEM_4D, RenderMode.STEM_PARKED)
                and self.stem_model(optics) == "coherent"):
            return self._prefetch_coherent(optics, time_s)
        if (optics.beam_blanked or optics.render_mode != RenderMode.TEM_IMAGING or self.config.pan_margin <= 0
                or self._time_dependent()):
            return False
        tkey = 0  # not time dependent
        gen = getattr(self.specimen, "generation", 0)
        view = optics.view
        base = dataclasses.replace(optics, view=dataclasses.replace(view, center_um=(0.0, 0.0)),
                                   beam_offset_px=(0.0, 0.0))
        key = (base, self.config.tem_model, gen, tkey)
        poptics, shift = self._padded_optics(optics)
        fkey = (poptics.view, frozenset(TEM_LAYERS), gen, tkey)
        with self._lock:
            if any(k == key and _crop(raster, c0, view) is not None
                   for k, c0, raster, _ in getattr(self, "_pans", None) or []):
                return False
            hit = self._fieldmaps.get(fkey)
            if hit is not None:  # the live view's exit-wave spectra (a focus step reuses one)
                self._prefetch_tem = TransferCache(size=len(self._tem_cache.spectra) + 1)
                for k, v in self._tem_cache.spectra.items():
                    self._prefetch_tem.spectra[k] = v
            done = threading.Event()
            self._inflight = (optics, done)
        try:
            return self._prefetch(optics, key, view, poptics, shift, fkey, hit)
        finally:
            done.set()
            self._inflight = None

    def _prefetch(self, optics, key, view, poptics, shift, fkey, hit) -> bool:
        from .tem import render_tem_raster

        if hit is not None and getattr(hit[0], "generation", 0) == hit[2]:
            fm, token = hit[0], hit[1]
        else:
            fm = self._rasterize(poptics.view, frozenset(TEM_LAYERS))
            token = (next(self._tokens), getattr(fm, "generation", 0))
        raster = render_tem_raster(fm, poptics, self.grains, self.crystallinity, self.config, self.seed,
                                   self._prefetch_tem, token, shift)
        with self._lock:
            self._keep_field_map(self._fieldmaps, fkey, fm, token)
            pans = getattr(self, "_pans", None) or []
            pans.insert(1 if pans else 0, (key, view.center_um, raster, poptics.view))  # behind the live view
            self._pans = pans[:max(1, self.config.pan_cache_size)]
            # its exit-wave spectrum too: a focus step of the prefetched view is a multiply + FFT
            for k, v in list(self._prefetch_tem.spectra.items()):
                self._tem_cache._put(self._tem_cache.spectra, k, v)
            self.prefetched += 1
        return True

    def _render(self, optics, *, frame_index: int = 0, scan_point=None, time_s: float = 0.0) -> np.ndarray:
        h, w = optics.output_shape
        if optics.beam_blanked:
            return np.zeros((h, w), np.float32)
        mode = optics.render_mode
        if mode != RenderMode.TEM_IMAGING:  # precession only changes diffraction patterns
            optics = _precession_frame(optics, time_s)
        if mode == RenderMode.TEM_IMAGING and self.config.pan_margin > 0:
            return self._render_tem_panned(optics, time_s)
        if mode == RenderMode.TEM_IMAGING:
            fm, token = self.field_map(optics, TEM_LAYERS, time_s)
            fkey = (token, optics, self.config.tem_model)
            if self._frame is not None and self._frame[0] == fkey:
                self.frames_from_cache += 1
                return self._frame[1]
            img = render_tem(fm, optics, self.grains, self.crystallinity, self.config, self.seed,
                             self._tem_cache, token)
            # Shared, not copied: every frame of an exposure is the SAME array (a 4096² copy
            # per frame was 20 ms, and it defeated the detector's per-exposure cache).
            # Read-only, so a caller that would scribble on the cache fails loudly instead.
            img.flags.writeable = False
            self._frame = (fkey, img)
            return img
        if mode == RenderMode.TEM_DIFFRACTION:
            fm, token = self.field_map(optics, SAED_LAYERS, time_s)
            return self._saed.render(fm, token, optics, self.grains, self.crystallinity)
        fm, token = self.field_map(optics, STEM_LAYERS, time_s)
        if self.stem_model(optics) == "coherent":
            return self._coherent.render(self._ctx(optics, time_s), fm, token, optics, scan_point,
                                         frame_index, self.specimen)
        return self._stem.render(fm, token, optics, self.grains, self.crystallinity, scan_point,
                                 frame_index, self.specimen)

    # ------------------------------------------------------ coherent STEM
    def _ctx(self, optics, time_s: float):
        """What the coherent engine needs from the renderer: a cached fine-view rasteriser."""
        import dataclasses
        from types import SimpleNamespace

        def fine(view):
            return self._fine_field_map(dataclasses.replace(optics, view=view), time_s)
        return SimpleNamespace(field_map=fine, grains=self.grains, crystallinity=self.crystallinity)

    def _fine_field_map(self, optics, time_s: float):
        """``field_map`` of a coherent-STEM fine view in its own store, rasterised without the
        renderer's lock (a prefetch thread does not hold up live frames)."""
        store = self._fine_fieldmaps
        tq = self.config.time_quantum_s
        tkey = round(time_s / tq) if (self._time_dependent() and tq > 0) else 0
        gen = getattr(self.specimen, "generation", 0)
        key = (optics.view, frozenset(), gen, tkey)
        with self._lock:
            hit = store.get(key)
            if hit is not None and getattr(hit[0], "generation", 0) == hit[2]:
                store.move_to_end(key)
                return hit[0], hit[1]
        fm = self._rasterize(optics.view, frozenset())
        with self._lock:
            hit = store.get(key)
            if hit is not None and getattr(hit[0], "generation", 0) == hit[2]:
                return hit[0], hit[1]
            self.rasters_built += 1
            token = (next(self._tokens), getattr(fm, "generation", 0))
            self._keep_field_map(store, key, fm, token, size=2)  # as many as the engine keeps tiles
            return fm, token

    def _prefetch_coherent(self, optics, time_s: float) -> bool:
        """The coherent STEM engine's probe, transmission tile and first block of ``optics``,
        off the render thread (the renderer's lock only for the small scan raster)."""
        optics = _precession_frame(optics, time_s)
        with self._lock:
            fm, token = self.field_map(optics, STEM_LAYERS, time_s)
            engine = self._coherent
        done = engine.prefetch(self._ctx(optics, time_s), optics, fm, token)
        if done:
            with self._lock:
                self.prefetched += 1
        return done

    def stem_model(self, optics) -> str:
        """The STEM model used for these optics: "coherent" or "kinematic"
        (``RenderConfig.stem_model``, with "auto" resolved; see :mod:`de_twin.render.coherent`)."""
        if optics.render_mode not in (RenderMode.STEM_4D, RenderMode.STEM_PARKED):
            return "kinematic"
        return choose_model(optics, self.config, lambda: self._coherent.sampling(optics))

    def coherent_sampling(self, optics):
        """The coherent simulation's :class:`~de_twin.render.coherent.Sampling` for these optics."""
        return self._coherent.sampling(optics)

    def datacube(self, optics, *, time_s: float = 0.0, model: Optional[str] = None) -> np.ndarray:
        """Noiseless 4D-STEM data: float32 (scan ny, nx, Hd, Wd) at the *binned* detector shape,
        electrons per binned pixel per second."""
        if optics.render_mode not in (RenderMode.STEM_4D, RenderMode.STEM_PARKED):
            raise ValueError("datacube needs STEM optics")
        fm, token = self.field_map(optics, STEM_LAYERS, time_s)
        if (model or self.stem_model(optics)) == "coherent":
            return self._coherent.datacube(self._ctx(optics, time_s), optics, fm, token)
        ny, nx = optics.view.shape
        bx, by = (max(1, int(v)) for v in optics.extras.get("hw_binning", (1, 1)))
        h, w = optics.output_shape
        hd, wd = h // by, w // bx
        out = np.empty((ny, nx, hd, wd), np.float32)
        for iy in range(ny):
            for ix in range(nx):
                p = self._stem.render(fm, token, optics, self.grains, self.crystallinity, (ix, iy), 0,
                                      self.specimen)
                out[iy, ix] = p[:hd * by, :wd * bx].reshape(hd, by, wd, bx).sum(axis=(1, 3))
        return out

    def ground_truth_ptychography(self, optics, *, time_s: float = 0.0) -> dict:
        """Complex object, probe (modes), scan positions and aberrations of the coherent
        simulation (see :meth:`de_twin.render.coherent.CoherentStem.ground_truth`)."""
        if optics.render_mode not in (RenderMode.STEM_4D, RenderMode.STEM_PARKED):
            raise ValueError("ptychography ground truth needs STEM optics")
        return self._coherent.ground_truth(self._ctx(optics, time_s), optics)

    def virtual_image(self, optics, inner_mrad: float, outer_mrad: float, *,
                      time_s: float = 0.0) -> np.ndarray:
        """Annular-detector image over the whole scan, float32 (scan ny, nx),
        electrons per second collected at each probe position.

        BF: ``(0, convergence)``; ADF/HAADF: e.g. ``(40, 200)``. Uses the field map
        and the pattern specs directly - no per-point CBED is rendered.
        """
        if optics.render_mode not in (RenderMode.STEM_4D, RenderMode.STEM_PARKED):
            raise ValueError("virtual_image needs STEM optics (scan raster); got "
                             f"{RenderMode(optics.render_mode).name}")
        ny, nx = optics.view.shape
        if optics.beam_blanked:
            return np.zeros((ny, nx), np.float32)
        fm, token = self.field_map(optics, STEM_LAYERS, time_s)
        if self.stem_model(optics) == "coherent":
            return self._coherent.virtual_image(self._ctx(optics, time_s), optics, float(inner_mrad),
                                                float(outer_mrad), fm)
        return self._stem.virtual_image(fm, token, optics, self.grains, self.crystallinity,
                                        float(inner_mrad), float(outer_mrad))

    def warm_up(self, optics, max_patterns: Optional[int] = None, budget_s: float = 0.5,
                time_s: float = 0.0) -> int:
        """Pre-render the most used STEM patterns of the current view (C++ WarmUpCache)."""
        if optics.render_mode not in (RenderMode.STEM_4D, RenderMode.STEM_PARKED):
            return 0
        if self.stem_model(optics) == "coherent":
            return 0
        fm, token = self.field_map(optics, STEM_LAYERS, time_s)
        return self._stem.warm_up(fm, token, optics, self.grains, self.crystallinity, max_patterns, budget_s)

    def ground_truth(self, optics, time_s: float = 0.0) -> dict:
        """Per raster pixel (TEM view) or scan point (STEM): ``material_id``, ``grain_id`` and the
        effective orientation ``quaternion`` (orix/diffsims rotation incl. the stage tilt;
        identity where there is no crystal), plus ``thickness_nm``."""
        from ..crystal import effective_matrices, rotation_from_crystal_to_lab
        from ..specimen.materials import GRAINS_PER_MATERIAL

        fm, _ = self.field_map(optics, frozenset(), time_s)
        grains = self.grains
        gid = fm.grain_id.astype(np.int64)
        ok = (gid >= 0) & (gid < len(grains)) & (gid // GRAINS_PER_MATERIAL == fm.material_id)
        q = np.zeros(gid.shape + (4,))
        q[..., 0] = 1.0
        uniq, inv = np.unique(gid[ok], return_inverse=True)
        if len(uniq):
            m = effective_matrices(grains.matrices[uniq], optics.alpha_rad, optics.beta_rad)
            q[ok] = rotation_from_crystal_to_lab(m)[inv]
        extra = {}
        if fm.under_thickness_nm is not None:
            extra = {"under_material": fm.under_material.copy(),
                     "under_thickness_nm": fm.under_thickness_nm * np.float32(optics.thickness_tilt_factor)}
        return {**extra, "material_id": fm.material_id.copy(), "grain_id": np.where(ok, gid, -1),
                "quaternion": q, "thickness_nm": fm.thickness_nm * np.float32(optics.thickness_tilt_factor),
                "view": fm.view}

    def scan_point_for(self, optics, frame_index: int = 0, scan_point=None) -> tuple[int, int]:
        return resolve_scan_point(optics, scan_point, frame_index, self.specimen, self.config)

    @property
    def stats(self) -> dict:
        s = self.cache.stats
        return {"rasters_built": self.rasters_built, "frames_from_cache": self.frames_from_cache,
                "saed_exact_grains": self._saed.exact_grains,
                "dp_hits": s.hits, "dp_misses": s.misses, "dp_evictions": s.evictions,
                "dp_entries": s.entries, "dp_bytes": s.bytes, "dp_last_render_ms": s.last_render_ms}
