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
from collections import OrderedDict
from typing import Optional

import numpy as np

from ..specimen.fieldmap import LAYER_DESCAN, LAYER_STRAIN, GrainTable
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


class Renderer:
    def __init__(self, specimen, config: Optional[RenderConfig] = None):
        self.specimen = specimen
        self.config = config or RenderConfig()
        self.cache = DiffractionCache(self.config.diffraction_cache_mb)
        self._fieldmaps: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._tokens = itertools.count(1)
        self._tem_cache = TransferCache()
        self._frame: Optional[tuple] = None
        self._saed = SaedRenderer(self.config)
        self._stem = StemRenderer(self.cache, self.config, self.seed)
        self._coherent = CoherentStem(self.cache, self.config, self.seed)
        self._grains_fallback: Optional[GrainTable] = None
        self.rasters_built = 0
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

    def field_map(self, optics, layers: frozenset = frozenset(), time_s: float = 0.0):
        """(FieldMap, token): rasterise only when view / layers / generation / time change."""
        tq = self.config.time_quantum_s
        tkey = round(time_s / tq) if (self._time_dependent() and tq > 0) else 0
        gen = getattr(self.specimen, "generation", 0)
        key = (optics.view, frozenset(layers), gen, tkey)
        hit = self._fieldmaps.get(key)
        if hit is not None:
            fm, token, fgen = hit
            if getattr(fm, "generation", 0) == fgen:
                self._fieldmaps.move_to_end(key)
                return fm, token
        fm = self.specimen.rasterize(optics.view, frozenset(layers))
        self.rasters_built += 1
        token = (next(self._tokens), getattr(fm, "generation", 0))
        self._fieldmaps[key] = (fm, token, getattr(fm, "generation", 0))
        while len(self._fieldmaps) > max(1, self.config.fieldmap_cache_size):
            self._fieldmaps.popitem(last=False)
        return fm, token

    def _render_tem_panned(self, optics, time_s: float) -> np.ndarray:
        """TEM imaging through a padded raster, cropped to the view.

        The raster is rendered ``pan_margin`` of the field larger on every side and kept,
        keyed by everything about the view EXCEPT where it is; a later view at the same
        magnification, focus and so on whose centre is within the margin is a crop of it,
        to the nearest raster pixel. The illumination disc and the upsampling are applied
        after the crop — they are fixed on the detector, not the specimen.
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
        ny, nx = view.shape
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
            m = self.config.pan_margin
            py, px = ny + 2 * int(round(m * ny)), nx + 2 * int(round(m * nx))
            pview = dataclasses.replace(view, shape=(py, px))
            d = max(1, optics.raster_downsample)
            poptics = dataclasses.replace(optics, view=pview, output_shape=(py * d, px * d))
            fm, token = self.field_map(poptics, TEM_LAYERS, time_s)
            raster = render_tem_raster(fm, poptics, self.grains, self.crystallinity, self.config,
                                       self.seed, self._tem_cache, token)
            self.rasters_built += 0  # counted by field_map
            self._pan = (key, view.center_um, raster)
            img = self._crop_pan(key, view)
        else:
            self.frames_from_cache += 1
        out = finish_tem(img, optics, self.config)
        out.flags.writeable = False  # shared by every frame of an exposure; see render()
        self._pan_out = (optics, (tkey, gen), out)
        return out

    def _render_preview(self, optics, time_s: float, tkey, gen) -> np.ndarray:
        """The view at up to ``preview_side``² — the frame shown while it is still
        moving — with its own padded crop cache, so a drag between previews is a crop."""
        from .tem import finish_tem, render_tem_raster

        view = optics.view
        ny, nx = view.shape
        f = 1
        while (ny // (2 * f)) * (nx // (2 * f)) >= self.config.preview_side ** 2                 and ny % (2 * f) == 0 and nx % (2 * f) == 0:
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
        img = self._crop_pan(pkey, pview, "_pan_preview")
        if img is None:
            m = self.config.pan_margin
            qy, qx = pview.shape
            py, px = qy + 2 * int(round(m * qy)), qx + 2 * int(round(m * qx))
            padded = dataclasses.replace(pview, shape=(py, px))
            d = poptics.raster_downsample
            pad_optics = dataclasses.replace(poptics, view=padded, output_shape=(py * d, px * d))
            fm, token = self.field_map(pad_optics, TEM_LAYERS, time_s)
            raster = render_tem_raster(fm, pad_optics, self.grains, self.crystallinity,
                                       self.config, self.seed, self._tem_cache, token)
            self._pan_preview = (pkey, pview.center_um, raster)
            img = self._crop_pan(pkey, pview, "_pan_preview")
        self.previews_rendered = getattr(self, "previews_rendered", 0) + 1
        out = finish_tem(img, poptics, self.config)
        out.flags.writeable = False
        return out

    def _crop_pan(self, key, view, slot: str = "_pan"):
        pan = getattr(self, slot, None)
        if pan is None or pan[0] != key:
            return None
        _, (cx0, cy0), raster = pan
        ny, nx = view.shape
        py, px = raster.shape
        pu = view.pixel_um
        # ViewWindow pixel (row, col) centres sit at centre + ((i + 0.5) - n / 2) * pixel,
        # divided by the tilt foreshortening along each axis.
        sx = -1.0 if getattr(view, "flip_x", False) else 1.0
        sy = -1.0 if getattr(view, "flip_y", False) else 1.0
        # the centre's move, in the view's own (rotated) frame
        wx, wy = view.center_um[0] - cx0, view.center_um[1] - cy0
        rot = float(getattr(view, "rotation_rad", 0.0))
        if rot:
            c, s = math.cos(rot), math.sin(rot)
            wx, wy = c * wx + s * wy, -s * wx + c * wy
        dc = sx * wx * max(view.cos_beta, 0.1) / pu
        dr = sy * wy * max(view.cos_alpha, 0.1) / pu
        c0 = int(round((px - nx) / 2 + dc))
        r0 = int(round((py - ny) / 2 + dr))
        if r0 < 0 or c0 < 0 or r0 + ny > py or c0 + nx > px:
            return None
        return raster[r0:r0 + ny, c0:c0 + nx]

    def invalidate(self) -> None:
        """Drop every cached raster, frame and pattern (e.g. after the specimen changed)."""
        self._pan = None
        self._pan_out = None
        self._pan_preview = None
        self._last_req = None
        self._fieldmaps.clear()
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
            return self.field_map(dataclasses.replace(optics, view=view), frozenset(), time_s)
        return SimpleNamespace(field_map=fine, grains=self.grains, crystallinity=self.crystallinity)

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
        return {"material_id": fm.material_id.copy(), "grain_id": np.where(ok, gid, -1),
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
