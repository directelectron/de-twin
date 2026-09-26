"""Selected-area electron diffraction.

The selected disk of the SAED raster is grouped into (material, grain, thickness-bin)
buckets carrying their summed transmission ``T``. Every grain bucket contributes its exact
pattern at the grain's effective orientation (grain x stage tilt, :mod:`de_twin.crystal`);
beyond ``MAX_EXACT_GRAINS`` buckets the weakest ones are folded per material into one
Debye-Scherrer powder pattern computed from the same library. The composite is scaled to
electrons with ``optics.pattern_e_per_s``, moved to the diffraction centre (diffraction
shift), gets the ``(1 - T)`` screened-Rutherford background, and is cut by the beam stop.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..specimen.materials import MATERIALS
from .diffraction import (THICKNESS_BIN_NM, Pattern, PatternOptions, bucket_patterns, render_pattern,
                          screening_angle_mrad)
from .samples import describe
from .util import shift_bilinear

MAX_EXACT_GRAINS = 4000


def buckets(fm, optics, grains, crystallinity, cfg):
    """(mats, gids, t_nm, cryst, weight) of the inscribed disk + per-material diffuse weight."""
    s = describe(fm, optics, grains, crystallinity, fallback_grains=False, diffuse=cfg.diffuse_scattering)
    ny, nx = s.mat.shape
    yy = (np.arange(ny) + 0.5 - ny / 2.0)[:, None]
    xx = (np.arange(nx) + 0.5 - nx / 2.0)[None, :]
    disk = (xx * xx + yy * yy) <= (0.5 * min(nx, ny)) ** 2
    mat = s.mat[disk].astype(np.int64)
    key = (mat << 40) | ((s.own_gid[disk] + 1) << 12) | s.thickness_bin[disk]
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    n = max(int(disk.sum()), 1)
    weight = np.bincount(inv, weights=s.T[disk].astype(np.float64), minlength=len(uniq)) / n
    cryst = np.bincount(inv, weights=s.crystallinity[disk], minlength=len(uniq)) / counts
    diffuse = np.bincount(mat, weights=s.diffuse_w[disk].astype(np.float64), minlength=len(MATERIALS)) / n
    return (uniq >> 40, ((uniq >> 12) & 0x0FFFFFFF) - 1, (uniq & 0xFFF) * THICKNESS_BIN_NM, cryst, weight), diffuse


def composite(bk, optics, grains, options: PatternOptions) -> tuple[Pattern, int]:
    """Weighted sum of the bucket patterns; the weakest grains beyond the cap become powder."""
    mats, gids, t, cryst, weight = (np.array(v) for v in bk)
    grain = np.flatnonzero(gids >= 0)
    folded = grain[np.argsort(-weight[grain], kind="stable")][MAX_EXACT_GRAINS:]
    if len(folded):
        keep = np.setdiff1d(np.arange(len(mats)), folded)
        rows = []
        for m in np.unique(mats[folded]):
            f = folded[mats[folded] == m]
            rows.append((m, -1, np.average(t[f], weights=weight[f]), np.average(cryst[f], weights=weight[f]),
                         weight[f].sum()))
        fm, fg, ft, fc, fw = (np.array(v) for v in zip(*rows))
        mats, gids = np.r_[mats[keep], fm], np.r_[gids[keep], fg]
        t, cryst, weight = np.r_[t[keep], ft], np.r_[cryst[keep], fc], np.r_[weight[keep], fw]
    return (bucket_patterns(mats, gids, t, cryst, grains, optics, options).combine(weight),
            int(min(len(grain), MAX_EXACT_GRAINS)))


def apply_beam_stop(img: np.ndarray, cx: float, cy: float, radius: float) -> None:
    """Disk of radius R plus a shaft of half-width R/2 to the +y edge."""
    if radius <= 0:
        return
    h, w = img.shape
    y = np.arange(h, dtype=np.float32)[:, None] - np.float32(cy)
    x = np.arange(w, dtype=np.float32)[None, :] - np.float32(cx)
    stop = (x * x + y * y) <= radius * radius
    stop |= (y >= 0) & (np.abs(x) <= 0.5 * radius)
    img[stop] = 0.0


#: Memory for the drawn patterns of precession tilts (a frame shorter than a precession
#: period sums a few of them instead of drawing every grain's disks again).
PRECESSION_CACHE_BYTES = 256 * 1024 * 1024


class SaedRenderer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._composites: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._tilts: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self.exact_grains = 0

    def render(self, fm, fm_token, optics, grains, crystallinity) -> np.ndarray:
        """The SAED pattern; under precession the average of the cone's tilts in the frame,
        each tilt's pattern drawn once and kept (`PRECESSION_CACHE_BYTES`)."""
        import dataclasses

        from .diffraction import tilt_samples

        samples = tilt_samples(optics)
        if len(samples) == 1 and samples[0][2] == (0.0, 0.0):
            return self._render_one(fm, fm_token, optics, grains, crystallinity)
        out = None
        for (tx, ty), wt, (sx, sy) in samples:
            key = (fm_token, dataclasses.replace(optics, precession_mrad=0.0, precession_phase_rad=0.0,
                                                 precession_arc_rad=6.283185307179586,
                                                 beam_tilt_mrad=(tx * 1e3, ty * 1e3)),
                   round(sx, 9), round(sy, 9))
            img = self._tilts.get(key)
            if img is None:
                img = self._render_one(fm, fm_token, key[1], grains, crystallinity, keep=False)
                if sx or sy:  # no descan: this tilt's pattern sits off the centre
                    img = shift_bilinear(img, sx / optics.recip_pixel_inv_nm, sy / optics.recip_pixel_inv_nm)
                self._tilts[key] = img
                while sum(v.nbytes for v in self._tilts.values()) > PRECESSION_CACHE_BYTES and len(self._tilts) > 1:
                    self._tilts.popitem(last=False)
            else:
                self._tilts.move_to_end(key)
            out = img * np.float32(wt) if out is None else out + img * np.float32(wt)
        return out

    def _render_one(self, fm, fm_token, optics, grains, crystallinity, keep: bool = True) -> np.ndarray:
        cfg = self.cfg
        h, w = optics.output_shape
        options = PatternOptions.from_config(cfg)
        sig = (fm_token, optics, options, cfg.beam_stop_radius_px, cfg.diffuse_scattering)
        img = self._composites.get(sig)
        if img is not None:
            self._composites.move_to_end(sig)
            return img.copy()
        bk, diffuse = buckets(fm, optics, grains, crystallinity, cfg)
        if not len(bk[0]):
            return np.zeros((h, w), np.float32)
        spec, self.exact_grains = composite(bk, optics, grains, options)
        mrad_px = optics.recip_pixel_inv_nm * 1000.0 * optics.wavelength_nm
        dif = [(screening_angle_mrad(m, optics.wavelength_nm), float(diffuse[m]))
               for m in range(len(diffuse)) if diffuse[m] > 0]
        cx0, cy0 = float(w // 2), float(h // 2)
        pat = render_pattern(spec, (h, w), optics.recip_pixel_inv_nm, optics.disk_radius_px,
                             center=(cx0, cy0), diffuse=dif, mrad_per_px=mrad_px)
        dcx, dcy = optics.diffraction_center_px
        pat = shift_bilinear(pat, dcx - cx0, dcy - cy0)
        pat *= np.float32(optics.pattern_e_per_s)
        ax, ay = optics.extras.get("axis_center_px", (cx0, cy0))
        apply_beam_stop(pat, ax, ay, cfg.beam_stop_radius_px)
        if not keep:
            return pat
        self._composites[sig] = pat
        while len(self._composites) > 2:
            self._composites.popitem(last=False)
        return pat.copy()
