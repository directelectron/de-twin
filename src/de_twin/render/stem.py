"""STEM: parked and 4D nano-beam / CBED patterns and fast virtual images.

Per scan point (one raster pixel per scan point, pitch = scan step):

* the pattern of the grain under the probe (own grain, or a 25 nm world-cell fallback grain
  for grain-less crystalline pixels) at its effective orientation (grain x stage tilt), at
  the pixel's thickness bin and crystallinity bin; disks have the probe convergence;
* weight ``T = exp(-t/Lambda)``, plus the ``(1 - T)`` screened-Rutherford diffuse background;
* five-sample probe-footprint blend (centre 1/3, four arms 1/6 at +-round(d/2 / step))
  when the probe is wider than one step;
* descan: the instrument ramp (optional) plus the FieldMap ``descan`` layer, and a
  centred affine from the ``strain`` layer;
* per-point intensity fluctuation 0.9..1.1 (hashed).

Rendered patterns are cached per (grain, thickness bin, crystallinity bin, tilt, geometry).
``virtual_image`` integrates annular detectors analytically from the pattern descriptions
per distinct key, so a whole-scan HAADF/BF is one vectorised pass. It ignores descan/strain.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..hashing import SeedKind, hash_seed, uniform_from_hash
from ..specimen.materials import MATERIALS
from ..state import RenderMode
from .diffraction import (CRYSTAL_BINS, THICKNESS_BIN_NM, DiffractionCache, PatternOptions, annulus_fractions,
                          bucket_patterns, crystal_bin_for, diffuse_annulus_fraction, render_diffuse,
                          render_pattern)
from .samples import Samples, describe
from .util import shift_bilinear

BLEND_CENTRE_WEIGHT = 1.0 / 3.0
BLEND_ARM_WEIGHT = 1.0 / 6.0
BLEND_MIN_OFFSET_PX = 0.5


@dataclass
class StemTables:
    samples: Samples
    key_index: np.ndarray  # (ny, nx) index into keys
    keys: np.ndarray  # (K, 4) int64: material, grain, thickness bin, crystallinity bin
    jitter: np.ndarray  # (ny, nx) float32 fluctuation factor (1 when disabled)
    all_specs: dict = None  # _tilt_key -> PatternSet of every key, built on first use


def build_tables(fm, optics, grains, crystallinity, cfg, seed: int) -> StemTables:
    s = describe(fm, optics, grains, crystallinity, fallback_grains=True, diffuse=cfg.diffuse_scattering)
    codes = ((s.mat.astype(np.int64) << 44) | ((s.gid + 1) << 16) | (s.thickness_bin << 4)
             | crystal_bin_for(s.crystallinity))
    uniq, inv = np.unique(codes, return_inverse=True)
    keys = np.stack([uniq >> 44, ((uniq >> 16) & 0x0FFFFFFF) - 1, (uniq >> 4) & 0xFFF, uniq & 0xF], 1)
    ny, nx = s.mat.shape
    if cfg.intensity_jitter:
        idx = np.arange(ny * nx, dtype=np.uint64).reshape(ny, nx)
        jit = (0.9 + 0.2 * uniform_from_hash(hash_seed(seed, SeedKind.INTENSITY_JITTER, idx))).astype(np.float32)
    else:
        jit = np.ones((ny, nx), np.float32)
    return StemTables(s, inv.reshape(ny, nx), keys, jit)


def resolve_scan_point(optics, scan_point, frame_index: int, specimen=None, cfg=None) -> tuple[int, int]:
    """(ix, iy) on the scan raster for this frame."""
    ny, nx = optics.view.shape
    if scan_point is not None:
        return int(np.clip(int(scan_point[0]), 0, nx - 1)), int(np.clip(int(scan_point[1]), 0, ny - 1))
    if optics.render_mode == RenderMode.STEM_4D:
        i = int(frame_index) % (nx * ny)
        return i % nx, i // nx
    if optics.extras.get("scan_park"):
        px, py = optics.extras.get("park_position", (nx // 2, ny // 2))
        if 0 <= px < nx and 0 <= py < ny:
            return int(px), int(py)
    if cfg is not None and cfg.park_on_feature and specimen is not None:
        try:
            cx, cy = optics.view.center_um
            f = specimen.nearest_feature(cx, cy)
            pos = getattr(f, "center_um", None) or (getattr(f, "x_um", None), getattr(f, "y_um", None))
            if f is not None and pos[0] is not None:
                r, c = optics.view.world_to_pixel(pos[0], pos[1])
                r, c = int(round(float(r))), int(round(float(c)))
                if 0 <= c < nx and 0 <= r < ny:
                    return c, r
        except Exception:  # noqa: BLE001 - a missing/odd feature API just parks on the axis
            pass
    return nx // 2, ny // 2


def _blend_samples(optics, ix, iy, enabled: bool):
    ny, nx = optics.view.shape
    off = float(optics.extras.get("probe_blend_offset_px", 0.0))
    if not enabled or optics.scan_step_um <= 0 or off < BLEND_MIN_OFFSET_PX:
        return [(ix, iy, 1.0)]
    o = int(round(off))
    pts = [(ix, iy, BLEND_CENTRE_WEIGHT), (ix - o, iy, BLEND_ARM_WEIGHT), (ix + o, iy, BLEND_ARM_WEIGHT),
           (ix, iy - o, BLEND_ARM_WEIGHT), (ix, iy + o, BLEND_ARM_WEIGHT)]
    return [(int(np.clip(x, 0, nx - 1)), int(np.clip(y, 0, ny - 1)), wt) for x, y, wt in pts]


def _tilt_key(optics) -> tuple:
    """What of the beam tilt and precession a pattern depends on."""
    return (tuple(optics.beam_tilt_mrad), getattr(optics, "precession_mrad", 0.0),
            getattr(optics, "precession_descan", True), getattr(optics, "precession_phase_rad", 0.0),
            getattr(optics, "precession_arc_rad", 0.0))


class StemRenderer:
    def __init__(self, cache: DiffractionCache, cfg, seed: int):
        self.cache = cache
        self.cfg = cfg
        self.seed = seed
        self._tables: "OrderedDict[tuple, StemTables]" = OrderedDict()

    def tables(self, fm, fm_token, optics, grains, crystallinity) -> StemTables:
        key = (fm_token, optics.ht_kv, optics.thickness_tilt_factor, optics.alpha_rad, optics.beta_rad,
               optics.convergence_mrad, self.cfg.diffuse_scattering, self.cfg.intensity_jitter)
        t = self._tables.get(key)
        if t is None:
            t = build_tables(fm, optics, grains, crystallinity, self.cfg, self.seed)
            self._tables[key] = t
            while len(self._tables) > 2:
                self._tables.popitem(last=False)
        else:
            self._tables.move_to_end(key)
        return t

    def _bucket(self, keys, optics, grains):
        return bucket_patterns(keys[:, 0], keys[:, 1], keys[:, 2] * THICKNESS_BIN_NM, keys[:, 3] / CRYSTAL_BINS,
                               grains, optics, PatternOptions.from_config(self.cfg))

    def all_specs(self, tab: StemTables, optics, grains):
        """PatternSet of every key of the table (one vectorised batch, then kept)."""
        if tab.all_specs is None:
            tab.all_specs = {}
        tk = _tilt_key(optics)
        if tk not in tab.all_specs:
            if len(tab.all_specs) >= 48:
                tab.all_specs.pop(next(iter(tab.all_specs)))
            tab.all_specs[tk] = self._bucket(tab.keys, optics, grains)
        return tab.all_specs[tk]

    def spec(self, tab: StemTables, i: int, optics, grains):
        specs = (tab.all_specs or {}).get(_tilt_key(optics))
        if specs is not None:
            return specs[i]
        return self._bucket(tab.keys[i:i + 1], optics, grains)[0]

    def _pattern(self, tab, i, optics, grains) -> np.ndarray:
        h, w = optics.output_shape
        key = ("dp", tuple(int(v) for v in tab.keys[i]), (h, w), optics.recip_pixel_inv_nm, optics.disk_radius_px,
               optics.convergence_mrad, optics.alpha_rad, optics.beta_rad, optics.ht_kv,
               PatternOptions.from_config(self.cfg), _tilt_key(optics))
        return self.cache.get(key, lambda _k: render_pattern(self.spec(tab, i, optics, grains), (h, w),
                                                             optics.recip_pixel_inv_nm, optics.disk_radius_px))

    # ------------------------------------------------------------ CBED
    def render(self, fm, fm_token, optics, grains, crystallinity, scan_point, frame_index,
               specimen=None) -> np.ndarray:
        cfg = self.cfg
        h, w = optics.output_shape
        tab = self.tables(fm, fm_token, optics, grains, crystallinity)
        s = tab.samples
        ix, iy = resolve_scan_point(optics, scan_point, frame_index, specimen, cfg)
        terms: dict = {}
        dterms: dict = {}
        for x, y, wt in _blend_samples(optics, ix, iy, cfg.probe_footprint_blend):
            k = int(tab.key_index[y, x])
            terms[k] = terms.get(k, 0.0) + wt * float(s.T[y, x])
            if s.diffuse_w[y, x] > 0:
                m = int(s.mat[y, x])
                dterms[m] = dterms.get(m, 0.0) + wt * float(s.diffuse_w[y, x])
        out = np.zeros((h, w), np.float32)
        for k, wt in terms.items():
            out += np.float32(wt) * self._pattern(tab, k, optics, grains)
        for m, wt in dterms.items():
            key = ("diffuse", m, (h, w), optics.recip_pixel_inv_nm, optics.ht_kv)
            out += np.float32(wt) * self.cache.get(key, lambda _k: render_diffuse(m, (h, w), optics))

        # descan (instrument ramp + layer) and strain, about the pattern centre
        dx = dy = 0.0
        ny, nx = optics.view.shape
        if cfg.add_descan and nx > 0 and ny > 0:
            xs, ys, dg = cfg.descan_ramp_px
            dx += cfg.descan_ramp_scale * (2.0 * (ix - nx / 2.0) / nx * xs + xs + iy * dg / ny)
            dy += cfg.descan_ramp_scale * (2.0 * (iy - ny / 2.0) / ny * ys + ys)
        if fm.descan is not None:
            dx += float(fm.descan[0, iy, ix])
            dy += float(fm.descan[1, iy, ix])
        cx0, cy0 = float(w // 2), float(h // 2)
        dcx, dcy = optics.diffraction_center_px
        tx, ty = dcx - cx0 + dx, dcy - cy0 + dy
        if fm.strain is not None and np.abs(fm.strain[:, iy, ix] - [1.0, 0.0, 1.0]).max() > 1e-6:
            exx, exy, eyy = (float(v) for v in fm.strain[:, iy, ix])
            # forward map p' = M (p - c) + c + t  ->  ndimage wants the inverse (row, col order)
            minv = np.linalg.inv(np.array([[eyy, exy], [exy, exx]]))
            c = np.array([cy0, cx0])
            out = ndimage.affine_transform(out, minv, offset=c - minv @ (c + np.array([ty, tx])), order=1,
                                           mode="constant", cval=0.0)
        else:
            out = shift_bilinear(out, tx, ty)
        out *= np.float32(optics.pattern_e_per_s / float(tab.jitter[iy, ix]))
        return out

    # --------------------------------------------------- virtual detector
    def virtual_image(self, fm, fm_token, optics, grains, crystallinity, inner_mrad: float,
                      outer_mrad: float) -> np.ndarray:
        tab = self.tables(fm, fm_token, optics, grains, crystallinity)
        s = tab.samples
        lam = optics.wavelength_nm
        frac = annulus_fractions(self.all_specs(tab, optics, grains), optics.convergence_mrad,
                                 lam, inner_mrad, outer_mrad)
        dfrac = np.array([diffuse_annulus_fraction(m, lam, inner_mrad, outer_mrad) for m in range(len(MATERIALS))])
        sig = (s.T * frac[tab.key_index] + s.diffuse_w * dfrac[s.mat]).astype(np.float32)
        if len(_blend_samples(optics, 0, 0, self.cfg.probe_footprint_blend)) > 1:
            o = int(round(float(optics.extras.get("probe_blend_offset_px", 0.0))))
            p = np.pad(sig, o, mode="edge")
            ny, nx = sig.shape
            sig = (BLEND_CENTRE_WEIGHT * p[o:o + ny, o:o + nx]
                   + BLEND_ARM_WEIGHT * (p[o:o + ny, 0:nx] + p[o:o + ny, 2 * o:2 * o + nx]
                                         + p[0:ny, o:o + nx] + p[2 * o:2 * o + ny, o:o + nx]))
        return (sig * np.float32(optics.pattern_e_per_s) / tab.jitter).astype(np.float32)

    def warm_up(self, fm, fm_token, optics, grains, crystallinity, max_patterns=None,
                budget_s: float = 0.5) -> int:
        import time
        tab = self.tables(fm, fm_token, optics, grains, crystallinity)
        order = np.argsort(-np.bincount(tab.key_index.ravel(), minlength=len(tab.keys)), kind="stable")
        cap = len(order) if max_patterns is None else min(len(order), int(max_patterns))
        self.all_specs(tab, optics, grains)
        t0 = time.perf_counter()
        before = self.cache.stats.misses
        for i in order[:cap]:
            if time.perf_counter() - t0 > budget_s:
                break
            self._pattern(tab, int(i), optics, grains)
        return self.cache.stats.misses - before
