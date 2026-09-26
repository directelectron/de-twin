"""Diffraction patterns from the crystal library, drawn at a detector geometry.

A :class:`Pattern` is a list of disks (direct beam + Bragg reflections at reciprocal
positions in 1/nm) and radial rings (amorphous halos, powder rings), with weights that
sum to 1. :func:`bucket_patterns` builds one per "bucket" (material, grain, thickness,
crystallinity) from :mod:`de_twin.crystal`; :func:`render_pattern` draws disks of the
illumination convergence (anti-aliased, PSF-limited when small) and rings;
:func:`annulus_fractions` integrates a virtual STEM detector analytically.

Normalisation per bucket (before the mass-thickness transmission ``T = exp(-t/Lambda)``,
whose ``1 - T`` goes to the screened-Rutherford diffuse background)::

    P      = sum_g (pi t / xi_g)^2 <sinc^2(pi t s_g)>     kinematic Bragg sum at the effective
                                                           orientation (grain x stage tilt)
    D      = 0.75 (1 - exp(-P / 0.75))                     Bragg fraction, saturating
    A      = 0.75 min(1, t / t_sat) sum(halo weights)      glassy/amorphous halo fraction
    spots  = c D I_g / P        halo = (1 - c) A           c: crystallinity (in-situ)
    direct = 1 - c D - (1 - c) A, of which 8 % goes to the support-film halo

Crystalline pixels without a grain (unresolved polycrystal) and folded SAED grains use
the orientation average ``<P> = t sum_g pi^2 / (2 g xi_g^2)`` as Debye-Scherrer rings.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import ndimage
from scipy.special import erf, erfc

from ..crystal import effective_matrices, library_for
from ..specimen.materials import MaterialId, material

MAX_BRAGG_FRACTION = 0.75
MAX_DIFFRACTED_FRACTION = 0.75
FULL_DIFFRACTION_THICKNESS_NM = 20.0
FILM_HALO_WEIGHT = 0.08
RING_SIGMA_INV_NM = 0.8
POWDER_RING_SIGMA_INV_NM = 0.15
MIN_DISK_RADIUS_PX = 0.75
BLUR_FRACTION = 0.12
MIN_BLUR_SIGMA_PX = 1.5
MAX_BLUR_SIGMA_PX = 4.0
THICKNESS_BIN_NM = 2.0
MAX_THICKNESS_BIN = 250
CRYSTAL_BINS = 8
MIN_SPOT_WEIGHT = 1e-7
_S2 = math.sqrt(2.0)


@dataclass(frozen=True)
class PatternOptions:
    film_halo: bool = True  # support-film halo under every crystalline pattern
    # the amorphous fraction saturates at 20 nm * Lambda(mat) / Lambda(Au) (False: 20 nm for all)
    scaled_saturation: bool = True
    max_g_inv_nm: float = 25.0

    @classmethod
    def from_config(cls, cfg) -> "PatternOptions":
        return cls(bool(cfg.film_halo), bool(cfg.scaled_diffraction_saturation), float(cfg.max_g_inv_nm))


@dataclass
class Pattern:
    """Disks at reciprocal positions and radial rings; weights sum to 1."""

    spots: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # gx, gy [1/nm], weight
    rings: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # g [1/nm], half width, weight

    @property
    def total(self) -> float:
        return float(self.spots[:, 2].sum() + self.rings[:, 2].sum())


@dataclass
class PatternSet:
    """Patterns of N buckets as owner-sorted arrays (``set[i]`` is one :class:`Pattern`)."""

    n: int
    spots: np.ndarray  # (K, 3)
    spot_owner: np.ndarray  # (K,) sorted
    rings: np.ndarray  # (R, 3)
    ring_owner: np.ndarray  # (R,) sorted

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        return (self[i] for i in range(self.n))

    def __getitem__(self, i: int) -> Pattern:
        if not -self.n <= i < self.n:
            raise IndexError(i)
        i %= self.n
        a, b = np.searchsorted(self.spot_owner, [i, i + 1])
        c, d = np.searchsorted(self.ring_owner, [i, i + 1])
        return Pattern(self.spots[a:b], self.rings[c:d])

    def combine(self, weights) -> Pattern:
        """Weighted sum of all patterns, as one :class:`Pattern`."""
        w = np.asarray(weights, float)
        sp, rg = self.spots.copy(), self.rings.copy()
        sp[:, 2] *= w[self.spot_owner]
        rg[:, 2] *= w[self.ring_owner]
        return Pattern(sp, rg)


def thickness_bin_for(t_nm):
    return np.clip(np.rint(np.asarray(t_nm, np.float64) / THICKNESS_BIN_NM), 0, MAX_THICKNESS_BIN).astype(np.int64)


def crystal_bin_for(c):
    return np.clip(np.floor(np.clip(np.asarray(c, np.float64), 0, 1) * CRYSTAL_BINS + 1e-9), 0,
                   CRYSTAL_BINS).astype(np.int64)


def saturation_thickness_nm(material_id: int, options: PatternOptions = PatternOptions()) -> float:
    lam = material(material_id).absorption_length_200kv_nm
    if not options.scaled_saturation or lam <= 0:
        return FULL_DIFFRACTION_THICKNESS_NM
    return FULL_DIFFRACTION_THICKNESS_NM * lam / material(MaterialId.GOLD).absorption_length_200kv_nm


def amorphous_fraction(material_id: int, t_nm, options: PatternOptions = PatternOptions()):
    m = material(material_id)
    tw = np.clip(np.asarray(t_nm, float) / saturation_thickness_nm(material_id, options), 0.0, 1.0)
    return MAX_DIFFRACTED_FRACTION * tw * sum(w for _, w in m.halos)


def bragg_fraction(p):
    return MAX_BRAGG_FRACTION * (1.0 - np.exp(-np.asarray(p, float) / MAX_BRAGG_FRACTION))


def _rings(owner, weight, g, gw, width):
    """Rings at radii ``g`` (relative weights ``gw``) for every owner, scaled by ``weight``."""
    gw = np.asarray(gw, float) / max(float(np.sum(gw)), 1e-300)
    sel = weight > 0
    o, w = owner[sel], weight[sel]
    return (np.repeat(o, len(g)),
            np.stack([np.tile(g, len(o)), np.full(len(o) * len(g), width), (w[:, None] * gw).ravel()], 1))


def beam_tilt_rotation(tilt_rad) -> np.ndarray:
    """The lab rotation equivalent to tilting the incident beam by *tilt_rad* = (tx, ty):
    the Ewald sphere tilts with the beam, which is the crystal tilting the other way.
    Tilting the beam towards -g by the Bragg angle excites +g."""
    tx, ty = (float(v) for v in tilt_rad)
    if tx == 0.0 and ty == 0.0:
        return np.eye(3)
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec([-ty, tx, 0.0]).as_matrix()


def tilt_samples(optics, n_cone: int = 24) -> list:
    """The beam tilts one exposure averages over, as ``(tilt_rad, weight, shift_inv_nm)``:
    the static beam tilt, or with precession `n_cone` tilts on the cone swept during the
    frame (its arc, from its phase). ``shift`` is where the pattern lands relative to the
    static diffraction centre: nothing with descan, the tilt itself without. (Without
    descan only the Bragg spots and the direct beam sweep; rings and the diffuse
    background stay centred, a simplification.)"""
    bx, by = (v * 1e-3 for v in getattr(optics, "beam_tilt_mrad", (0.0, 0.0)))
    theta = float(getattr(optics, "precession_mrad", 0.0)) * 1e-3
    if theta <= 0.0:
        return [((bx, by), 1.0, (0.0, 0.0))]
    arc = float(getattr(optics, "precession_arc_rad", 2.0 * math.pi))
    phase = float(getattr(optics, "precession_phase_rad", 0.0))
    n = n_cone if arc >= 2.0 * math.pi else max(2, int(math.ceil(n_cone * arc / (2.0 * math.pi))))
    step = min(arc, 2.0 * math.pi) / n
    lam = optics.wavelength_nm
    descan = bool(getattr(optics, "precession_descan", True))
    out = []
    for j in range(n):
        phi = phase + (j + 0.5) * step
        px, py = theta * math.cos(phi), theta * math.sin(phi)
        shift = (0.0, 0.0) if descan else (px / lam, py / lam)
        out.append(((bx + px, by + py), 1.0 / n, shift))
    return out


def _excite_buckets(lib, grains, gids, t_nm, optics, tilt_rad=(0.0, 0.0)):
    """Excitation of (grain, thickness) buckets: each distinct grain is evaluated once over the
    distinct thicknesses, then expanded to its buckets. Returns (owner bucket, gx, gy, I, P)."""
    ug, ginv = np.unique(gids, return_inverse=True)
    ut, tinv = np.unique(t_nm, return_inverse=True)
    m = effective_matrices(grains.matrices[ug], optics.alpha_rad, optics.beta_rad)
    if tilt_rad[0] != 0.0 or tilt_rad[1] != 0.0:
        m = np.einsum("ij,njk->nik", beam_tilt_rotation(tilt_rad), m)
    if len(ug) * len(ut) > 4 * len(gids) + 64:  # scattered thicknesses: one orientation per bucket
        ex = lib.excite(m[ginv], optics.wavelength_nm, t_nm, optics.ht_kv, optics.convergence_mrad)
        return ex.owner, ex.gx, ex.gy, ex.intensity, ex.total
    ex = lib.excite(m, optics.wavelength_nm, np.broadcast_to(ut, (len(ug), len(ut))), optics.ht_kv,
                    optics.convergence_mrad)
    order = np.argsort(ginv, kind="stable")
    counts = np.bincount(ginv, minlength=len(ug))
    starts = np.cumsum(counts) - counts
    rep = counts[ex.owner]
    k = np.repeat(np.arange(len(ex.owner)), rep)
    b = order[starts[ex.owner[k]] + np.arange(len(k)) - np.repeat(np.cumsum(rep) - rep, rep)]
    return b, ex.gx[k], ex.gy[k], ex.intensity[k, tinv[b]], ex.total[ginv, tinv]


def bucket_patterns(mats, gids, t_nm, cryst, grains, optics,
                    options: PatternOptions = PatternOptions()) -> PatternSet:
    """Normalised patterns of N buckets (material, grain, thickness, crystallinity). Crystalline
    buckets with a grain use the exact excitation at the grain's effective orientation (grain x
    stage tilt x beam tilt); crystalline buckets without one the powder (orientation) average.
    With precession, the average over the tilts of the cone (`tilt_samples`)."""
    samples = tilt_samples(optics)
    if len(samples) == 1 and samples[0][2] == (0.0, 0.0):
        return _bucket_patterns_at(mats, gids, t_nm, cryst, grains, optics, options, samples[0][0])
    base = (np.asarray(mats, np.int64).tobytes(), np.asarray(gids, np.int64).tobytes(),
            np.asarray(t_nm, float).tobytes(), np.asarray(cryst, float).tobytes(), id(grains),
            float(optics.alpha_rad), float(optics.beta_rad), float(optics.wavelength_nm),
            float(optics.ht_kv), float(optics.convergence_mrad), options)

    def at(tilt):
        key = (base, round(tilt[0], 12), round(tilt[1], 12))
        hit = _CONE_CACHE.get(key)
        if hit is not None and hit[0] is grains:
            _CONE_CACHE.move_to_end(key)
            return hit[1]
        ps = _bucket_patterns_at(mats, gids, t_nm, cryst, grains, optics, options, tilt)
        _CONE_CACHE[key] = (grains, ps)
        while len(_CONE_CACHE) > CONE_CACHE_SIZE:
            _CONE_CACHE.popitem(last=False)
        return ps

    parts = [(at(tilt), w, sh) for tilt, w, sh in samples]
    n = parts[0][0].n
    spots = np.concatenate([p.spots * [1.0, 1.0, w] + [sh[0], sh[1], 0.0] for p, w, sh in parts])
    s_own = np.concatenate([p.spot_owner for p, _, _ in parts])
    rings = np.concatenate([p.rings * [1.0, 1.0, w] for p, w, _ in parts])
    r_own = np.concatenate([p.ring_owner for p, _, _ in parts])
    si, ri = np.argsort(s_own, kind="stable"), np.argsort(r_own, kind="stable")
    return PatternSet(n, spots[si], s_own[si], rings[ri], r_own[ri])


#: Pattern sets per precession tilt, so a frame that covers part of the cone re-weights
#: cached tilts instead of exciting every grain again (two full cones' worth).
CONE_CACHE_SIZE = 48
_CONE_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()


def _bucket_patterns_at(mats, gids, t_nm, cryst, grains, optics, options, tilt_rad) -> PatternSet:
    """`bucket_patterns` for one incident-beam tilt (radians)."""
    mats = np.asarray(mats, np.int64)
    gids = np.asarray(gids, np.int64)
    t_nm = np.asarray(t_nm, float)
    cryst = np.clip(np.asarray(cryst, float), 0.0, 1.0)
    n = len(mats)
    n_grains = len(grains) if grains is not None else 0
    direct = np.ones(n)
    halo = np.zeros(n)  # into the material's own (amorphous / glassy) halos
    spots, s_own, rings, r_own = [], [], [], []
    for mid in np.unique(mats):
        sel = np.flatnonzero(mats == mid)
        m = material(mid)
        if m.id == MaterialId.VACUUM:
            continue
        a = amorphous_fraction(mid, t_nm[sel], options)
        if not m.crystalline:
            halo[sel] = a
            direct[sel] = 1.0 - a
        else:
            c = cryst[sel]
            halo[sel] = (1.0 - c) * a
            direct[sel] = 1.0 - halo[sel]
            has = (gids[sel] >= 0) & (gids[sel] < n_grains)
            lib = library_for(int(mid), options.max_g_inv_nm)
            if has.any() and lib is not None:  # an amorphous material scatters no Bragg beams
                own = sel[has]
                owner, gx, gy, inten, total = _excite_buckets(lib, grains, gids[own], t_nm[own], optics,
                                                              tilt_rad)
                bragg = cryst[own] * bragg_fraction(total)
                w = inten * np.where(total > 0, bragg / np.maximum(total, 1e-300), 0.0)[owner]
                keep = w > MIN_SPOT_WEIGHT  # the dropped remainder stays in the direct beam
                direct[own] -= np.bincount(owner, weights=np.where(keep, w, 0.0), minlength=len(own))
                spots.append(np.stack([gx, gy, w], 1)[keep])
                s_own.append(own[owner[keep]])
            if (~has).any():
                g, per_nm = lib.rings(optics.ht_kv, optics.wavelength_nm)
                pw = sel[~has]
                bragg = cryst[pw] * bragg_fraction(t_nm[pw] * per_nm.sum())
                direct[pw] -= bragg
                o, r = _rings(pw, bragg, g, per_nm, POWDER_RING_SIGMA_INV_NM)
                keep = r[:, 2] > MIN_SPOT_WEIGHT
                direct[pw] += np.bincount(o[~keep], weights=r[~keep, 2], minlength=n)[pw]
                rings.append(r[keep])
                r_own.append(o[keep])
            if options.film_halo:
                film = direct[sel] * FILM_HALO_WEIGHT
                direct[sel] -= film
                cg = material(MaterialId.AMORPHOUS_CARBON).halos
                o, r = _rings(sel, film, [g for g, _ in cg], [w for _, w in cg], RING_SIGMA_INV_NM)
                rings.append(r)
                r_own.append(o)
        if m.halos:
            o, r = _rings(sel, halo[sel], [g for g, _ in m.halos], [w for _, w in m.halos], RING_SIGMA_INV_NM)
            rings.append(r)
            r_own.append(o)
    spots = np.concatenate([np.stack([np.zeros(n), np.zeros(n), direct], 1)] + spots)
    s_own = np.concatenate([np.arange(n)] + s_own)
    rings = np.concatenate(rings) if rings else np.zeros((0, 3))
    r_own = np.concatenate(r_own).astype(np.int64) if r_own else np.zeros(0, np.int64)
    si, ri = np.argsort(s_own, kind="stable"), np.argsort(r_own, kind="stable")
    return PatternSet(n, spots[si], s_own[si], rings[ri], r_own[ri])


def powder_pattern(mid: int, t_nm: float, crystallinity: float, optics,
                   options: PatternOptions = PatternOptions()) -> Pattern:
    """Orientation-averaged (Debye-Scherrer) pattern of a crystalline material."""
    return bucket_patterns([mid], [-1], [t_nm], [crystallinity], None, optics, options)[0]


# ------------------------------------------------------------ rendering
def blur_sigma_px(disk_radius_px: float) -> float:
    return float(np.clip(BLUR_FRACTION * max(disk_radius_px, MIN_DISK_RADIUS_PX), MIN_BLUR_SIGMA_PX,
                         MAX_BLUR_SIGMA_PX))


_RADIUS_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def radius_map(shape: tuple[int, int], cx: float, cy: float) -> np.ndarray:
    key = (int(shape[0]), int(shape[1]), float(cx), float(cy))
    r = _RADIUS_CACHE.get(key)
    if r is None:
        y = (np.arange(shape[0], dtype=np.float32) - np.float32(cy))[:, None]
        x = (np.arange(shape[1], dtype=np.float32) - np.float32(cx))[None, :]
        r = np.sqrt(x * x + y * y)
        _RADIUS_CACHE[key] = r
        while len(_RADIUS_CACHE) > 3:
            _RADIUS_CACHE.popitem(last=False)
    else:
        _RADIUS_CACHE.move_to_end(key)
    return r


#: Disks at least this large are drawn over the detector only (`disks.draw_disks`) rather
#: than stamped: a STEM probe's disks can outgrow the detector.
DRAW_DISKS_ABOVE_PX = 24.0


def stamp_disks(out: np.ndarray, x, y, w, radius: float, sigma: float) -> None:
    """Add soft disks (uniform disk ~convolved with a Gaussian PSF), each integrating to its
    weight over the infinite plane (what falls off the detector is lost). Vectorised."""
    h, wd = out.shape
    half = radius + 4.0 * sigma + 1.0
    n = int(math.ceil(2.0 * half)) + 1
    x, y, w = (np.asarray(v, np.float64) for v in (x, y, w))
    on = (w != 0) & (x + half > 0) & (y + half > 0) & (x - half < wd) & (y - half < h)
    x, y, w = x[on], y[on], w[on]
    if radius >= DRAW_DISKS_ABOVE_PX and radius >= sigma:
        from . import disks

        if disks.AVAILABLE:
            disks.draw_disks(out, x, y, w, radius, sigma)
            return
    step = max(1, (1 << 22) // (n * n))
    ar = np.arange(n)
    for c0 in range(0, len(x), step):
        xs, ys, ws = x[c0:c0 + step], y[c0:c0 + step], w[c0:c0 + step]
        ix0 = np.floor(xs - half).astype(np.int64)
        iy0 = np.floor(ys - half).astype(np.int64)
        dx = (ix0[:, None] + ar) - xs[:, None]
        dy = (iy0[:, None] + ar) - ys[:, None]
        d = np.sqrt(dx[:, None, :] ** 2 + dy[:, :, None] ** 2)
        if radius < sigma:  # small disk: disk (x) Gaussian ~ Gaussian of the combined variance
            v = np.exp(-0.5 * d * d / (sigma * sigma + radius * radius / 4.0))
        else:
            v = 0.5 * erfc((d - radius) / (_S2 * sigma))
        v *= (ws / v.sum(axis=(1, 2)))[:, None, None]
        rows = np.broadcast_to((iy0[:, None] + ar)[:, :, None], v.shape)
        cols = np.broadcast_to((ix0[:, None] + ar)[:, None, :], v.shape)
        ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < wd)
        np.add.at(out, (rows[ok], cols[ok]), v[ok].astype(np.float32))


def _merge(rows: np.ndarray) -> np.ndarray:
    """Sum the weights (column 2) of rows with identical (column 0, column 1)."""
    if len(rows) < 2:
        return rows
    q = np.rint(rows[:, :2] * 1e7).astype(np.int64)  # 1e-7 1/nm grid, packed into one int64 key
    u, first, inv = np.unique(q[:, 0] * (1 << 31) + q[:, 1], return_index=True, return_inverse=True)
    return np.c_[rows[first, :2], np.bincount(inv.ravel(), weights=rows[:, 2], minlength=len(u))]


def splat_blur(shape, x, y, w, sigma: float) -> np.ndarray:
    """Bilinear splat of point weights, blurred by a Gaussian (sum kept, off-detector lost)."""
    h, wd = shape
    pad = int(math.ceil(4 * sigma)) + 2
    img = np.zeros((h + 2 * pad, wd + 2 * pad), np.float64)  # margin: spots just off the edge
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - ix, y - iy
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            r, c = iy + dy + pad, ix + dx + pad
            m = (r >= 0) & (r < h + 2 * pad) & (c >= 0) & (c < wd + 2 * pad)
            np.add.at(img, (r[m], c[m]), (w * wx * wy)[m])
    s = math.sqrt(max(sigma * sigma - 1.0 / 6.0, 0.25))  # the splat itself adds ~1/6 px^2
    return ndimage.gaussian_filter(img, s, mode="constant", truncate=4.0)[pad:-pad, pad:-pad].astype(np.float32)


def _ring_table(rings_px: list, rmax: float, step: float = 0.25):
    """Radial profile (value per pixel) of a set of normalised Gaussian rings."""
    rho = np.arange(0.0, rmax + 2.0, step)
    prof = np.zeros_like(rho)
    for radius, sigma, weight in rings_px:
        if weight <= 0 or sigma <= 0:
            continue
        grid = np.linspace(max(0.0, radius - 6.0 * sigma), radius + 6.0 * sigma, 2049)
        integ = 2.0 * math.pi * np.trapezoid(grid * np.exp(-0.5 * ((grid - radius) / sigma) ** 2), grid)
        if integ > 0:
            prof += (weight / integ) * np.exp(-0.5 * ((rho - radius) / sigma) ** 2)
    return rho, prof


def render_pattern(p: Pattern, shape: tuple[int, int], recip_px: float, disk_radius_px: float,
                   center: Optional[tuple[float, float]] = None, diffuse=(),
                   mrad_per_px: float = 0.0) -> np.ndarray:
    """Draw a pattern at a detector geometry. Components integrate to their weights.

    ``diffuse`` is an optional list of ``(theta0_mrad, weight)`` screened-Rutherford terms
    (needs ``mrad_per_px``), folded into the same radial table as the rings.
    """
    h, w = int(shape[0]), int(shape[1])
    cx, cy = center if center is not None else (float(w // 2), float(h // 2))
    out = np.zeros((h, w), np.float32)
    if not recip_px > 0:
        return out
    radius = max(float(disk_radius_px), MIN_DISK_RADIUS_PX)
    sigma = blur_sigma_px(radius)
    spots = _merge(p.spots)
    if len(spots):
        x, y = cx + spots[:, 0] / recip_px, cy + spots[:, 1] / recip_px
        if radius < sigma:  # PSF-limited spots: bilinear splat + one Gaussian (disk (x) PSF)
            out += splat_blur((h, w), x, y, spots[:, 2], math.sqrt(sigma * sigma + radius * radius / 4.0))
        else:
            stamp_disks(out, x, y, spots[:, 2], radius, sigma)
    diffuse = [(t0, wgt) for t0, wgt in diffuse if wgt > 0] if mrad_per_px > 0 else []
    if len(p.rings) or diffuse:
        rings_px = [(g / recip_px, math.sqrt((hw / recip_px) ** 2 / 3.0 + radius * radius / 4.0 + sigma * sigma),
                     float(wt)) for g, hw, wt in _merge(p.rings)]
        r = radius_map((h, w), cx, cy)
        rho, prof = _ring_table(rings_px, float(math.hypot(max(cx, w - cx), max(cy, h - cy))))
        for t0, wgt in diffuse:
            th = rho * mrad_per_px
            prof = prof + wgt * (t0 * t0 / math.pi) * mrad_per_px ** 2 / (th * th + t0 * t0) ** 2
        out += np.interp(r, rho, prof).astype(np.float32)
    return out


def screening_angle_mrad(material_id: int, wavelength_nm: float) -> float:
    """Characteristic angle of screened-Rutherford (Wentzel) elastic scattering."""
    a_nm = 0.885 * 0.0529177 * material(material_id).z_eff ** (-1.0 / 3.0)
    return 1000.0 * wavelength_nm / (2.0 * math.pi * a_nm)


def render_diffuse(material_id: int, shape: tuple[int, int], optics,
                   center: Optional[tuple[float, float]] = None) -> np.ndarray:
    """Normalised (plane integral 1) screened-Rutherford background:
    p(theta) = theta0^2 / (pi (theta^2 + theta0^2)^2) per mrad^2."""
    h, w = int(shape[0]), int(shape[1])
    cx, cy = center if center is not None else (float(w // 2), float(h // 2))
    mrad_px = optics.recip_pixel_inv_nm * 1000.0 * optics.wavelength_nm
    t0 = screening_angle_mrad(material_id, optics.wavelength_nm)
    r = radius_map((h, w), cx, cy) * np.float32(mrad_px)
    return np.float32(t0 * t0 / math.pi * mrad_px * mrad_px) / (r * r + np.float32(t0 * t0)) ** 2


# ----------------------------------------------------- virtual detectors
def _overlap(d, r: float, big_r: float):
    """Area of the intersection of a disk of radius ``r`` at distance ``d`` from the origin with
    the disk of radius ``big_r`` centred on the origin (vectorised over ``d``)."""
    if not np.isfinite(big_r):
        return np.full(np.shape(d), math.pi * r * r)
    if big_r <= 0:
        return np.zeros(np.shape(d))
    d = np.maximum(np.asarray(d, float), 1e-12)
    c1 = np.clip((d * d + r * r - big_r * big_r) / (2 * d * r), -1, 1)
    c2 = np.clip((d * d + big_r * big_r - r * r) / (2 * d * big_r), -1, 1)
    k = np.maximum((-d + r + big_r) * (d + r - big_r) * (d - r + big_r) * (d + r + big_r), 0.0)
    lens = r * r * np.arccos(c1) + big_r * big_r * np.arccos(c2) - 0.5 * np.sqrt(k)
    return np.where(d >= r + big_r, 0.0, np.where(d <= abs(big_r - r), math.pi * min(r, big_r) ** 2, lens))


def annulus_fractions(ps: PatternSet, convergence_mrad: float, wavelength_nm: float,
                      inner_mrad: float, outer_mrad: float) -> np.ndarray:
    """Fraction of each pattern of a set (weights, not truncated by a camera) collected by an
    annular detector [inner, outer) centred on the direct beam. Vectorised."""
    k = 1000.0 * wavelength_nm
    out = np.zeros(ps.n)
    if len(ps.spots):
        s = ps.spots
        d = np.hypot(s[:, 0], s[:, 1]) * k
        a = float(convergence_mrad)
        if a < 1e-9:
            frac = ((d >= inner_mrad) & (d < outer_mrad)).astype(float)
        else:
            frac = (_overlap(d, a, outer_mrad) - _overlap(d, a, max(inner_mrad, 0.0))) / (math.pi * a * a)
        out += np.bincount(ps.spot_owner, weights=frac * s[:, 2], minlength=ps.n)
    if len(ps.rings):
        r = ps.rings
        rad = r[:, 0] * k
        sig = np.sqrt((r[:, 1] * k) ** 2 / 3.0 + convergence_mrad ** 2 / 4.0)

        def cdf(x):
            return 0.5 * (1.0 + erf((min(x, 1e30) - rad) / (_S2 * sig)))
        frac = (cdf(outer_mrad) - cdf(max(inner_mrad, 0.0))) / np.maximum(1.0 - cdf(0.0), 1e-12)
        out += np.bincount(ps.ring_owner, weights=np.clip(frac, 0, 1) * r[:, 2], minlength=ps.n)
    return out


def annulus_fraction(p: Pattern, convergence_mrad: float, wavelength_nm: float,
                     inner_mrad: float, outer_mrad: float) -> float:
    ps = PatternSet(1, p.spots, np.zeros(len(p.spots), np.int64), p.rings, np.zeros(len(p.rings), np.int64))
    return float(annulus_fractions(ps, convergence_mrad, wavelength_nm, inner_mrad, outer_mrad)[0])


def diffuse_annulus_fraction(material_id: int, wavelength_nm: float, inner_mrad: float, outer_mrad: float) -> float:
    t0 = screening_angle_mrad(material_id, wavelength_nm)

    def cdf(t):
        return t * t / (t * t + t0 * t0) if np.isfinite(t) else 1.0
    return float(cdf(outer_mrad) - cdf(max(inner_mrad, 0.0)))


# ---------------------------------------------------------------- cache
@dataclass
class DiffractionCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes: int = 0
    budget_bytes: int = 256 * 1024 * 1024
    entries: int = 0
    last_render_ms: float = 0.0


class DiffractionCache:
    """Byte-budgeted LRU of rendered patterns, any hashable key (never evicts the newest)."""

    def __init__(self, budget_mb: int = 256):
        self._lru: "OrderedDict[object, np.ndarray]" = OrderedDict()
        self.stats = DiffractionCacheStats(budget_bytes=max(1, int(budget_mb)) * 1024 * 1024)

    def set_budget_mb(self, mb: int) -> None:
        self.stats.budget_bytes = max(1, int(mb)) * 1024 * 1024
        self._evict()

    def __len__(self) -> int:
        return len(self._lru)

    def __contains__(self, key) -> bool:
        return key in self._lru

    def get(self, key, render: Callable[[object], np.ndarray]) -> np.ndarray:
        arr = self._lru.get(key)
        if arr is not None:
            self._lru.move_to_end(key)
            self.stats.hits += 1
            return arr
        t0 = time.perf_counter()
        arr = render(key)
        arr.setflags(write=False)
        self.stats.last_render_ms = (time.perf_counter() - t0) * 1000.0
        self._lru[key] = arr
        self.stats.bytes += arr.nbytes
        self.stats.misses += 1
        self._evict()
        return arr

    def clear(self) -> None:
        self._lru.clear()
        self.stats.bytes = 0
        self.stats.entries = 0

    def _evict(self) -> None:
        while self.stats.bytes > self.stats.budget_bytes and len(self._lru) > 1:
            _, victim = self._lru.popitem(last=False)
            self.stats.bytes -= victim.nbytes
            self.stats.evictions += 1
        self.stats.entries = len(self._lru)
