"""TEM imaging.

Two models share the amplitude term, mass-thickness plus diffraction contrast::

    T = exp(-t * f_tilt / Lambda(material, HT))       (f_tilt = 1/(cos a cos b))
    I_amp = clamp(T * (1 - s * c * D_out), 0, 1)

``D_out`` is the fraction of the beam Bragg-scattered *outside the objective aperture*
(all of it when there is none) by the pixel's grain at its effective orientation (grain x
stage tilt) and thickness, from :mod:`de_twin.crystal` (the same kinematic excitation as
the diffraction patterns); ``c`` is the crystallinity and ``s`` ``diffraction_contrast_scale``.
Tilting the stage therefore moves grains in and out of Bragg conditions (bend/thickness
contours included).

``legacy``: Gaussian defocus blur of ``I_amp`` (sigma = |df| theta_ill / px, floored at
0.5 px), the signed Fresnel unsharp mask, gold lattice fringes as an intensity modulation.

``physical`` (default): a proper exit wave propagated through the objective transfer function::

    psi(r)   = sqrt(I_amp) * exp(i phi - kappa phi_tex)
    phi      = sigma V0(material) t                    mean inner potential (Fresnel fringes)
             + phi_tex                                 amorphous granularity  (Thon rings)
             + phi_lat                                 lattice fringes of the excited beams
    phi_tex  = s * sigma V0 sqrt(t / n) / p * N(r)     N: world-locked unit white noise, band-limited
                                                       by the atomic form factor (0.07 nm Gaussian);
                                                       PSD = sigma^2 V0^2 t / n  (proportional to t)
    I(r)     = | F^-1[ F[psi] * exp(-i chi(k)) * E_s * E_t * A_obj ] |^2

    chi(k)   = pi lambda (df |k|^2 + a1x (kx^2 - ky^2) + a1y 2 kx ky) + pi/2 Cs lambda^3 |k|^4
               evaluated at k + k_tilt minus chi(k_tilt)  (beam tilt: image shift df * tilt, axial coma)
    E_s      = exp(-(alpha_ill / (2 lambda))^2 |grad chi|^2)       spatial coherence
    E_t      = exp(-0.5 (pi lambda Delta (|k+kt|^2 - |kt|^2))^2)    temporal coherence (focal spread)

df < 0 is underfocus. Intensity is |psi|^2 so it is never negative. Weak-phase
theory gives the familiar CTF ``sin chi - kappa cos chi`` for the texture, so the
power spectrum of a carbon film shows Thon rings at ``chi = atan(kappa) + n pi``.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

from ..crystal import effective_matrices, library_for
from ..hashing import SeedKind, hash_seed, uniform_from_hash
from ..optics.physics import interaction_constant
from ..optics.state import image_aberrations_of
from ..specimen.materials import (GRAINS_PER_MATERIAL, MATERIALS, MaterialId, absorption_lengths_nm,
                                  material_array)
from . import fastpath as _fp
from .diffraction import MAX_THICKNESS_BIN, THICKNESS_BIN_NM, bragg_fraction, thickness_bin_for
from .samples import DIFFUSE_CUTOFF_LENGTHS
from .util import (chi_and_gradient, disk_profile, gaussian_filter_threaded, parallel_rows, pool, upsample_to,
                   world_normal_noise)

GOLD_LATTICE_CONTRAST = 0.12
LATTICE_MIN_SAMPLES = 2.25
FRINGE_BEAM_FRACTION = 0.02  # a beam carrying this fraction of the electrons gives full-strength fringes
MAX_FRINGE_BEAMS = 3
MAX_COHERENT_BEAMS = 48  # coherent STEM: Friedel pairs per grain put into the phase grating


# ------------------------------------------------------ diffraction contrast
@dataclass
class BraggContrast:
    loss: np.ndarray  # (ny, nx) float32 fraction of the beam removed from the bright field
    gid: np.ndarray  # (ny, nx) int64 own grain (-1 none)
    fringes: dict  # grain id -> (kx, ky [rad/nm], amplitude) of its strongest resolved transmitted beams
    # coherent STEM (``coherent_k_max``): fringes are a phase grating sum_w a_w cos(g_w r + p_w)
    # with a_w = sqrt(2 f_w) (f_w: kinematic electron fraction of the Friedel pair) at the grain
    # typical thickness ``t_typ[gid]``, scaled by t / t_typ
    phase_grating: bool = False
    t_typ: dict = None
    kinematic: bool = False  # fringe amplitudes are kinematic (a phase grating scaled by t / t_typ)


class _BraggMemo:
    """Per-grain Bragg excitation for one tilt / beam / aperture, kept across views.

    ``geom[gid]`` is the grain's reflections near the Ewald sphere (index, gx, gy, s, |g|)
    and ``pair[(gid, thickness bin)]`` = (loss per unit crystallinity, total, per-reflection
    intensity, electrons per unit I_g): exactly what a dense evaluation gives, but only for
    the pairs a view contains, and each only once — a stage move or a magnification step
    mostly sees grains it has seen before."""

    MAX_PAIRS = 400_000

    def __init__(self, grains):
        self.grains = grains
        self.geom: dict = {}
        self.pair: dict = {}
        self._table = None
        self._done = None
        self.min_gxy = math.inf

    def _ensure_geometry(self, lib, grains, gids, optics) -> None:
        new_g = [int(g) for g in gids if int(g) not in self.geom]
        if not new_g:
            return
        m = effective_matrices(grains.matrices[np.asarray(new_g)], optics.alpha_rad, optics.beta_rad)

        def geo(r0, r1):
            own, idx, gx, gy, s = lib.geometry(m[r0:r1], optics.wavelength_nm, optics.convergence_mrad)
            cuts = np.searchsorted(own, np.arange(r1 - r0 + 1))
            gxy = np.hypot(gx, gy)
            for j in range(r1 - r0):
                a, b = cuts[j], cuts[j + 1]
                self.geom[new_g[r0 + j]] = (idx[a:b], gx[a:b], gy[a:b], s[a:b], gxy[a:b])
        parallel_rows(geo, len(new_g), min_chunk=16)

    def table(self, lib, grains, mid: int, optics, need=None) -> np.ndarray:
        """(GRAINS_PER_MATERIAL, MAX_THICKNESS_BIN + 1) loss per unit crystallinity of the
        grains of material ``mid`` at the thickness bins (:mod:`.bragg_table`). Filled lazily:
        ``need[k]`` is the largest bin wanted for grain ``k`` of the material (-1 none; None:
        every grain, every bin); rows and bins already computed are kept."""
        from .bragg_table import loss_table

        nbins = MAX_THICKNESS_BIN + 1
        if self._table is None:
            self._table = np.zeros((GRAINS_PER_MATERIAL, nbins))
            self._done = np.full(GRAINS_PER_MATERIAL, -1, np.int64)
            self.min_gxy = math.inf
        g0 = mid * GRAINS_PER_MATERIAL
        count = max(0, min(GRAINS_PER_MATERIAL, len(grains) - g0))
        need = (np.full(count, nbins - 1, np.int64) if need is None
                else np.asarray(need, np.int64)[:count])
        todo = np.flatnonzero(need > self._done[:count])
        if todo.size:
            top = int(need[todo].max())
            ids = [g0 + int(k) for k in todo]
            self._ensure_geometry(lib, grains, ids, optics)
            geom = [self.geom[g] for g in ids]
            tab, _ = loss_table(lib, geom, optics, self._g_obj, top + 1)
            self._table[todo, :top + 1] = tab
            self._done[todo] = top
            gxy = np.concatenate([g[4] for g in geom]) if geom else np.zeros(0)
            gxy = gxy[gxy > 1e-6]
            if gxy.size:
                self.min_gxy = min(self.min_gxy, float(gxy.min()))
        return self._table

    def values(self, lib, grains, gids, ps, tbs, optics) -> np.ndarray:
        """Loss per unit crystallinity (``frac * p_out``) of the pairs (gids[ps], tbs)."""
        if len(self.pair) > self.MAX_PAIRS:
            self.pair.clear()
        keys = [(int(gids[i]), int(tb)) for i, tb in zip(ps, tbs)]
        missing = [k for k in keys if k not in self.pair]
        if missing:
            self._ensure_geometry(lib, grains, sorted({g for g, _ in missing}), optics)
            lens = np.array([len(self.geom[g][0]) for g, _ in missing], np.int64)
            idx = np.concatenate([self.geom[g][0] for g, _ in missing])
            s = np.concatenate([self.geom[g][3] for g, _ in missing])
            t = np.repeat(np.array([tb for _, tb in missing], np.float64) * THICKNESS_BIN_NM, lens)
            inten = np.empty(len(idx))

            def work(r0, r1):
                inten[r0:r1] = lib.intensity_at(idx[r0:r1], s[r0:r1], t[r0:r1], optics.ht_kv,
                                                optics.wavelength_nm, optics.convergence_mrad)
            parallel_rows(work, len(idx), min_chunk=8192)
            out = np.concatenate([self.geom[g][4] for g, _ in missing]) > self._g_obj
            owner = np.repeat(np.arange(len(missing)), lens)
            total = np.bincount(owner, weights=inten, minlength=len(missing))
            p_out = np.bincount(owner, weights=inten * out, minlength=len(missing))
            frac = bragg_fraction(total) / np.maximum(total, 1e-300)
            cuts = np.concatenate([[0], np.cumsum(lens)])
            for j, k in enumerate(missing):
                self.pair[k] = (float(frac[j] * p_out[j]), float(total[j]), inten[cuts[j]:cuts[j + 1]],
                                float(frac[j]))
        return np.array([self.pair[k][0] for k in keys], np.float64)


_BRAGG_MEMOS: "OrderedDict[tuple, _BraggMemo]" = OrderedDict()
#: Tilt / beam states whose Bragg tables are kept (a tilt series back and forth, a wobbler):
#: ~1 MB each per crystalline material.
BRAGG_TABLES_KEPT = 16


def _bragg_memo(grains, mid: int, max_g: float, optics, g_obj: float) -> _BraggMemo:
    key = (id(grains), mid, float(max_g), float(optics.alpha_rad), float(optics.beta_rad),
           float(optics.wavelength_nm), float(optics.ht_kv), float(optics.convergence_mrad), float(g_obj))
    memo = _BRAGG_MEMOS.get(key)
    if memo is None or memo.grains is not grains:
        memo = _BraggMemo(grains)
        memo._g_obj = float(g_obj)
        _BRAGG_MEMOS[key] = memo
        while len(_BRAGG_MEMOS) > BRAGG_TABLES_KEPT:
            _BRAGG_MEMOS.popitem(last=False)
    _BRAGG_MEMOS.move_to_end(key)
    return memo


#: The Bragg memos are shared by every renderer (and a prefetching thread): one at a time.
_BRAGG_LOCK = threading.RLock()


def bragg_contrast(fm, optics, grains, crystallinity, cfg, coherent_k_max: float = 0.0,
                   _scratch: bool = False) -> BraggContrast:
    with _BRAGG_LOCK:
        return _bragg_contrast(fm, optics, grains, crystallinity, cfg, coherent_k_max, _scratch)


def _bragg_contrast(fm, optics, grains, crystallinity, cfg, coherent_k_max: float = 0.0,
                    _scratch: bool = False) -> BraggContrast:
    """Per-pixel Bragg loss from the grain at its effective orientation and thickness bin,
    and the lattice fringes (strongest excited, resolvable, transmitted beams) per grain.

    ``coherent_k_max > 0`` (coherent STEM): beams with ``|g| <= coherent_k_max`` (and resolvable
    on the raster) are not a loss but a phase grating of kinematic strength (all of them, up to
    ``MAX_COHERENT_BEAMS`` Friedel pairs); only the beams beyond are removed (the renderer adds
    them back incoherently)."""
    mat = fm.material_id
    f_t = np.float32(optics.thickness_tilt_factor)
    n = len(grains) if grains is not None else 0
    nb = MAX_THICKNESS_BIN + 1
    # a render's temporaries (not kept by the caller): reused buffers, see fastpath.scratch
    alloc = _fp.scratch if (_scratch and _fp.AVAILABLE) else (lambda name, shape, dt: np.empty(shape, dt))
    gid = alloc("bragg_gid", mat.shape, np.int64)
    # (grain, thickness bin) of every crystal pixel as one integer (-1 none); the pairs are few
    pix_all = alloc("bragg_pix", mat.shape, np.int64)

    def pixel_rows(r0, r1):
        g = fm.grain_id[r0:r1].astype(np.int64)
        ok = (g >= 0) & (g < n) & (g // GRAINS_PER_MATERIAL == mat[r0:r1])
        gid[r0:r1] = np.where(ok, g, -1)
        tb = thickness_bin_for(fm.thickness_nm[r0:r1].astype(np.float32) * f_t)
        pix_all[r0:r1] = np.where(ok, g * nb + tb, -1)
    maxbin = None  # the largest thickness bin of each grain in view (numba path)
    if _fp.AVAILABLE and n > 0:
        maxb = np.full((max(1, min(64, mat.shape[0] // 16)), n), -1, np.int16)
        _fp.bragg_pixels(fm.grain_id, mat, fm.thickness_nm, f_t, n, nb, GRAINS_PER_MATERIAL, gid, pix_all,
                         maxb)
        maxbin = maxb.max(axis=0)
    else:
        parallel_rows(pixel_rows, mat.shape[0])
    loss = alloc("bragg_loss", mat.shape, np.float32)
    loss.fill(0.0)
    fringes: dict = {}
    present = np.flatnonzero(maxbin >= 0) if maxbin is not None else None
    if (present.size == 0) if present is not None else not (gid >= 0).any():
        return BraggContrast(loss, gid, fringes)
    lam = optics.wavelength_nm
    g_obj = optics.objective_aperture_mrad / (1000.0 * lam) if optics.objective_aperture_mrad > 0 else 0.0
    pitch_nm = optics.view.pixel_um * 1000.0 / min(optics.view.cos_alpha, optics.view.cos_beta)
    g_res = 1.0 / (LATTICE_MIN_SAMPLES * pitch_nm)
    coherent = coherent_k_max > 0
    if coherent:
        g_obj = min(coherent_k_max, g_res)
    t_typ: dict = {}
    # the loss: a gather from each material's (grain, thickness bin) table for this tilt / beam
    lut = alloc("bragg_lut", (n * nb + 1,), np.float32)  # the last entry: no crystal (-1)
    lut.fill(0.0)
    fringe_mids = []
    mids = (np.unique(present // GRAINS_PER_MATERIAL) if present is not None
            else np.flatnonzero(np.bincount(gid[gid >= 0] // GRAINS_PER_MATERIAL)))
    for mid in mids:
        lib = library_for(int(mid), cfg.max_g_inv_nm)
        if lib is None:
            continue  # an amorphous material: no Bragg contrast (the FIB-liftout pattern's "auto" post)
        memo = _bragg_memo(grains, int(mid), cfg.max_g_inv_nm, optics, g_obj)
        g0 = int(mid) * GRAINS_PER_MATERIAL
        g1 = min(g0 + GRAINS_PER_MATERIAL, n)
        tab = memo.table(lib, grains, int(mid), optics, None if maxbin is None else maxbin[g0:g1])
        ids = np.arange(g0, g1)
        crm = np.ones(len(ids)) if crystallinity is None else np.asarray(crystallinity(ids), float).reshape(-1)
        lut[g0 * nb:g1 * nb] = (cfg.diffraction_contrast_scale * (crm[:, None] * tab[:g1 - g0])).ravel()
        # fringes need beams the raster resolves (and the aperture passes)
        if coherent or (cfg.lattice_fringes and memo.min_gxy <= g_res):
            fringe_mids.append(int(mid))
    if _fp.AVAILABLE:
        _fp.gather(lut, pix_all, loss)
    else:
        parallel_rows(lambda r0, r1: loss.__setitem__(slice(r0, r1), lut[pix_all[r0:r1]]), mat.shape[0])
    if not fringe_mids:
        return BraggContrast(loss, gid, fringes, coherent, t_typ,
                             coherent or (cfg.lattice_fringes and cfg.lattice_fringe_model == "kinematic"))
    pix = pix_all[pix_all >= 0]
    pix = pix[np.isin(pix // (nb * GRAINS_PER_MATERIAL), fringe_mids)]
    pair_count = np.bincount(pix, minlength=n * nb)
    nz = np.flatnonzero(pair_count)
    pg, ptb = np.divmod(nz, nb)
    g_u = np.unique(pg)
    tb_u = np.unique(ptb)
    pgi, pti = np.searchsorted(g_u, pg), np.searchsorted(tb_u, ptb)
    cr = np.ones(len(g_u)) if crystallinity is None else np.asarray(crystallinity(g_u), float).reshape(-1)
    counts = np.zeros((len(g_u), len(tb_u)), np.int64)
    counts[pgi, pti] = pair_count[nz]
    typical = counts.argmax(axis=1)  # most common thickness bin of each grain (for the fringes)
    for mid in np.unique(g_u // GRAINS_PER_MATERIAL):
        sel = np.flatnonzero(g_u // GRAINS_PER_MATERIAL == mid)
        lib = library_for(int(mid), cfg.max_g_inv_nm)
        memo = _bragg_memo(grains, int(mid), cfg.max_g_inv_nm, optics, g_obj)
        # the per-beam intensities at each grain's typical thickness (exact, few)
        memo.values(lib, grains, g_u[sel], np.arange(len(sel)), tb_u[typical[sel]], optics)
        for j, gi in enumerate(sel):
            geo = memo.geom[int(g_u[gi])]
            inten, frac = memo.pair[(int(g_u[gi]), int(tb_u[typical[gi]]))][2:]
            gx_all, gy_all, gxy = geo[1], geo[2], geo[4]
            if coherent:
                k = np.flatnonzero((gxy <= g_obj) & (gxy > 1e-6))
                f = inten[k] * frac * cr[gi]
                pairs: dict = {}
                for i, fi in zip(k, f):
                    gx_, gy_ = float(gx_all[i]), float(gy_all[i])
                    if gx_ < -1e-9 or (abs(gx_) <= 1e-9 and gy_ < 0):
                        gx_, gy_ = -gx_, -gy_
                    key = (round(gx_, 6), round(gy_, 6))
                    pairs[key] = pairs.get(key, 0.0) + float(fi)
                top = sorted(pairs.items(), key=lambda kv: -kv[1])[:MAX_COHERENT_BEAMS]
                top = [(gk, fv) for gk, fv in top if fv > 1e-7]
                if top:
                    w = np.array([(gk[0], gk[1], math.sqrt(2.0 * fv)) for gk, fv in top])
                    fringes[int(g_u[gi])] = (2 * math.pi * w[:, 0], 2 * math.pi * w[:, 1], w[:, 2])
                    t_typ[int(g_u[gi])] = max(float(tb_u[typical[gi]]) * THICKNESS_BIN_NM, THICKNESS_BIN_NM)
            else:
                # beams that reach the image (inside the aperture, or all without one) and are resolved
                k = np.flatnonzero((gxy <= g_res) & (gxy > 1e-6) & ((gxy <= g_obj) | (g_obj == 0)))
                f = inten[k] * frac
                waves: list = []
                kin = cfg.lattice_fringe_model == "kinematic"
                for i, fi in zip(k[np.argsort(-f)], np.sort(f)[::-1]):
                    if len(waves) == MAX_FRINGE_BEAMS:
                        break
                    if not any(abs(gx_all[i] + wx) < 1e-6 and abs(gy_all[i] + wy) < 1e-6
                               for wx, wy, _ in waves):
                        # the Friedel pair (g, -g) carries 2 f: a phase grating 2 sqrt(f) cos(g r)
                        a = (cfg.lattice_fringe_efficiency * math.sqrt(2.0 * 2.0 * fi) if kin
                             else min(1.0, math.sqrt(fi / FRINGE_BEAM_FRACTION)))
                        waves.append((gx_all[i], gy_all[i], a * cr[gi]))
                if waves:
                    w = np.array(waves)
                    fringes[int(g_u[gi])] = (2 * math.pi * w[:, 0], 2 * math.pi * w[:, 1], w[:, 2])
                    t_typ[int(g_u[gi])] = max(float(tb_u[typical[gi]]) * THICKNESS_BIN_NM, THICKNESS_BIN_NM)
    kinematic = coherent or (cfg.lattice_fringes and cfg.lattice_fringe_model == "kinematic")
    return BraggContrast(loss, gid, fringes, coherent, t_typ, kinematic)


def amplitude_rows(fm, optics, r0, r1, loss) -> tuple:
    """(I_amp, t_eff) for raster rows r0:r1."""
    mat = fm.material_id[r0:r1]
    f = np.float32(optics.thickness_tilt_factor)
    t = fm.thickness_nm[r0:r1].astype(np.float32) * f
    lam = absorption_lengths_nm(optics.ht_kv)
    e = t / lam[mat]
    if fm.under_thickness_nm is not None:  # the amorphous layer under it absorbs too
        e = e + fm.under_thickness_nm[r0:r1] * f / lam[fm.under_material[r0:r1]]
    T = np.exp(-e)
    return np.clip(T * (1.0 - loss[r0:r1]), 0.0, 1.0).astype(np.float32), t


# ------------------------------------------------------------- lattice
def lattice_field(fm, optics, bc: BraggContrast, t, gold_only: bool = False):
    """sum_w a_w cos(g_w . r + phase_w) / n_w over each grain's strongest resolved beams,
    times min(1, t / 10 nm). Returns ((rows, cols), weighted, unweighted) or None."""
    gids = [g for g in bc.fringes if not gold_only or g // GRAINS_PER_MATERIAL == MaterialId.GOLD]
    mask = np.isin(bc.gid, gids) if gids else None
    if mask is None or not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    g = bc.gid[rows, cols]
    acc = np.zeros(len(rows))
    raw = np.zeros(len(rows))
    grating = bool(getattr(bc, "phase_grating", False) or getattr(bc, "kinematic", False))
    weight = np.zeros(len(rows)) if grating else np.minimum(1.0, t[rows, cols] / 10.0)
    # each grain's pixels as slices of a grain-sorted order (not a mask over all of them),
    # in chunks on the shared pool
    order = np.argsort(g, kind="stable")
    gs = g[order]
    jobs = []
    for gg in gids:
        a, b = int(np.searchsorted(gs, gg)), int(np.searchsorted(gs, gg, side="right"))
        jobs += [(gg, c0, min(b, c0 + 65536)) for c0 in range(a, b, 65536)]

    def one(job):
        gg, a, b = job
        sel = order[a:b]
        x_um, y_um = optics.view.pixel_to_world(rows[sel], cols[sel])
        xs, ys = x_um * 1000.0, y_um * 1000.0
        kx, ky, amp = bc.fringes[gg]
        norm = 1.0 if grating else 1.0 / len(kx)
        acc_g = np.zeros(len(sel))
        raw_g = np.zeros(len(sel))
        for w in range(len(kx)):
            ph = 2.0 * math.pi * uniform_from_hash(hash_seed(int(gg), SeedKind.GRAIN, w, 0xA111))
            c = np.cos(kx[w] * xs + ky[w] * ys + ph) * norm
            acc_g += amp[w] * c
            raw_g += c
        acc[sel] = acc_g
        raw[sel] = raw_g
        if grating:  # kinematic amplitude grows with the projected thickness
            weight[sel] = np.clip(t[rows[sel], cols[sel]] / bc.t_typ[gg], 0.0, 2.0)
    list(pool().map(one, jobs))
    return (rows, cols), (acc * weight).astype(np.float32), raw.astype(np.float32)


# --------------------------------------------------------------- legacy
def render_legacy(fm, optics, grains, crystallinity, cfg) -> np.ndarray:
    bc = bragg_contrast(fm, optics, grains, crystallinity, cfg)
    I, t = amplitude_rows(fm, optics, 0, fm.material_id.shape[0], bc.loss)
    lat = lattice_field(fm, optics, bc, t, gold_only=True)
    if lat is not None:
        (rows, cols), _, raw = lat
        I[rows, cols] *= 1.0 + GOLD_LATTICE_CONTRAST * raw
        np.clip(I, 0.0, 1.0, out=I)
    unblurred = I.copy()
    sigma = max(optics.blur_sigma_px, 1e-3)
    radius = min(31, int(math.ceil(3.0 * sigma - 1e-6)))
    out = ndimage.gaussian_filter(I, sigma, truncate=max(1.0, radius / sigma), mode="nearest")
    if optics.fresnel_gain > 0:
        fs = optics.fresnel_sigma_px
        fr = min(31, int(math.ceil(3.0 * fs - 1e-6)))
        low = ndimage.gaussian_filter(unblurred, fs, truncate=max(1.0, fr / fs), mode="nearest")
        out = out + np.float32(optics.fresnel_gain * optics.fresnel_sign) * (unblurred - low)
        np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32)


# ------------------------------------------------------------- physical
class TransferCache:
    """Caches for the wave-optical model: exit-wave spectra and transfer functions."""

    def __init__(self, size: int = 4):
        self.spectra: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self.transfer: "OrderedDict[tuple, np.ndarray | None]" = OrderedDict()
        self.size = size

    def _put(self, d, k, v):
        d[k] = v
        d.move_to_end(k)
        while len(d) > self.size:
            d.popitem(last=False)

    def buffer(self, shape, dtype) -> np.ndarray:
        """An array for a new spectrum (`de_twin.buffers`: recycled when one is free)."""
        from .. import buffers

        return buffers.empty(shape, dtype)


def _kgrid(shape, pitch_nm):
    ny, nx = shape
    kx = sfft.fftfreq(nx, d=pitch_nm)
    ky = sfft.fftfreq(ny, d=pitch_nm)
    return kx, ky


def _raster_tilt(optics) -> tuple[float, float]:
    """The beam tilt (1/nm) in the raster's frame: the tilt coils sit above the specimen, so
    a tilt is a specimen-plane (world) direction, turned by the view's rotation and flips
    like everything else on the camera."""
    lam = optics.wavelength_nm
    wx, wy = (v / 1000.0 / lam for v in optics.beam_tilt_mrad)
    view = optics.view
    rot = float(getattr(view, "rotation_rad", 0.0))
    if rot:
        c, s = math.cos(rot), math.sin(rot)
        wx, wy = c * wx + s * wy, -s * wx + c * wy
    if getattr(view, "flip_x", False):
        wx = -wx
    if getattr(view, "flip_y", False):
        wy = -wy
    return wx, wy


def transfer_function(shape, pitch_nm, optics, transposed: bool = False) -> np.ndarray | None:
    """Objective transfer H(k) (complex64, FFT layout) or None when it is ~identity.

    chi comes from ``optics.image_aberrations`` (CEOS set incl. C1 = defocus and A1 = objective
    stigmator, see :mod:`de_twin.optics.aberrations`); beam tilt enters as
    ``chi(k + k_t) - chi(k_t)`` and the envelopes use ``grad chi(k + k_t) - grad chi(k_t)``."""
    lam = optics.wavelength_nm
    ab = image_aberrations_of(optics)
    tx, ty = _raster_tilt(optics)
    alpha = optics.illumination_mrad / 1000.0
    delta = optics.focal_spread_nm
    kap = optics.objective_aperture_mrad / 1000.0 / lam if optics.objective_aperture_mrad > 0 else 0.0
    kx1, ky1 = _kgrid(shape, pitch_nm)
    # E_s = exp(-(alpha/(2 lambda))^2 |grad chi|^2), grad chi in rad nm (2 pi lambda df k for
    # pure defocus); E_t = exp(-0.5 (pi lambda Delta (|k+kt|^2 - |kt|^2))^2)
    es = (alpha / (2.0 * lam)) ** 2
    et = 0.5 * (math.pi * lam * delta) ** 2
    kt2 = tx * tx + ty * ty
    c0 = float(ab.chi(tx, ty, lam))
    gx0, gy0 = (float(v) for v in ab.gradient(tx, ty, lam))

    # cheap identity test on the grid edge / corners
    kmx, kmy = float(np.abs(kx1).max()), float(np.abs(ky1).max())
    px = np.array([kmx, 0, kmx, kmx, -kmx, -kmx, -kmx, 0])
    py = np.array([0, kmy, kmy, -kmy, kmy, -kmy, 0, -kmy])
    c = ab.chi(px + tx, py + ty, lam)
    gxp, gyp = ab.gradient(px + tx, py + ty, lam)
    k2p = (px + tx) ** 2 + (py + ty) ** 2
    envp = np.exp(-es * ((gxp - gx0) ** 2 + (gyp - gy0) ** 2) - et * (k2p - kt2) ** 2)
    if (np.abs(c - c0).max() < 2e-3 and envp.min() > 0.999 and kap == 0.0 and tx == 0 and ty == 0):
        return None

    ny, nx = shape
    if _fp.AVAILABLE and set(ab.coeffs) <= {"C1", "A1", "C3", "C5"}:
        from .. import buffers

        H = buffers.empty((nx, ny) if transposed else (ny, nx), np.complex64)
        f = np.float32
        s = 2.0 * math.pi / lam
        l2 = lam * lam
        c1, c3, c5 = (ab[n].real for n in ("C1", "C3", "C5"))
        a1 = ab["A1"]
        _fp.transfer_c1a1c3c5(
            kx1.astype(np.float64), ky1.astype(np.float64), float(tx), float(ty),
            f(s * l2 * c1 / 2), f(s * l2 * l2 * c3 / 4), f(s * l2 ** 3 * c5 / 6),
            f(s * l2 * a1.real / 2), f(s * l2 * a1.imag), f(s * l2 * c1), f(s * l2 * l2 * c3),
            f(s * l2 ** 3 * c5), f(s * l2 * a1.real), f(s * l2 * a1.imag), f(c0), f(gx0), f(gy0),
            f(es), f(et), f(kt2), f(kap * kap), H, transposed)
        return H
    if transposed:
        H = transfer_function(shape, pitch_nm, optics)
        return None if H is None else np.ascontiguousarray(H.T)
    H = np.empty((ny, nx), np.complex64)
    KX = (kx1 + tx).astype(np.float32)[None, :]
    kyf = (ky1 + ty).astype(np.float32)

    def work(r0, r1):
        KY = kyf[r0:r1, None]
        chi, gx, gy = chi_and_gradient(ab, KX, KY, lam, dtype=np.float32)
        chi -= np.float32(c0)
        gx -= np.float32(gx0)
        gy -= np.float32(gy0)
        k2t = KX * KX + KY * KY - np.float32(kt2)
        arg = np.empty(chi.shape, np.complex64)
        arg.real = -(np.float32(es) * (gx * gx + gy * gy) + np.float32(et) * k2t * k2t)
        arg.imag = -chi
        h = np.exp(arg)
        if kap > 0:
            h[(KX * KX + KY * KY) > np.float32(kap * kap)] = 0
        H[r0:r1] = h

    parallel_rows(work, ny)
    return H


def _world_locked_noise(view, seed: int, salt: int, _scratch: bool = False) -> np.ndarray:
    """Unit white noise on the raster of *view*, each pixel the value of the world cell under
    it (cells of pitch (px / cos_beta, px / cos_alpha)), so it moves with the specimen. An
    unrotated view reads a block of cells directly; a rotated one (a realistic column's
    image rotation) takes each pixel's nearest cell."""
    ny, nx = view.shape
    pitch_x = view.pixel_um / view.cos_beta
    pitch_y = view.pixel_um / view.cos_alpha
    if not getattr(view, "rotation_rad", 0.0) and not getattr(view, "flip_x", False)             and not getattr(view, "flip_y", False):
        # the nearest cell, ties (a view centred on the pixel lattice) broken upwards
        ix0 = math.floor(view.center_um[0] / pitch_x + (0.5 - nx / 2.0) + 0.5 + 1e-6)
        iy0 = math.floor(view.center_um[1] / pitch_y + (0.5 - ny / 2.0) + 0.5 + 1e-6)
        return world_normal_noise(ix0, iy0, ny, nx, seed, salt,
                                  _fp.scratch("noise", (ny, nx), np.float32) if (_scratch and _fp.AVAILABLE) else None)
    rows, cols = np.mgrid[0:ny, 0:nx]
    x, y = view.pixel_to_world(rows.ravel(), cols.ravel())
    ix = np.floor(np.asarray(x) / pitch_x).astype(np.int64)
    iy = np.floor(np.asarray(y) / pitch_y).astype(np.int64)
    x0, y0 = int(ix.min()), int(iy.min())
    block = world_normal_noise(x0, y0, int(iy.max()) - y0 + 1, int(ix.max()) - x0 + 1, seed, salt)
    return block[iy - y0, ix - x0].reshape(ny, nx)


def texture_noise(optics, cfg, seed: int, _scratch: bool = False) -> np.ndarray:
    """World-locked unit white noise on the raster, band-limited by the atomic form factor."""
    view = optics.view
    p_nm = view.pixel_um * 1000.0
    salt = cfg.texture_seed_salt ^ (int(round(math.log2(max(p_nm, 1e-6)) * 64)) & 0xFFFF)
    noise = _world_locked_noise(view, seed, salt, _scratch)
    sb = cfg.texture_bandlimit_nm / p_nm
    if sb > 0.3:
        noise = gaussian_filter_threaded(noise, sb, mode="wrap")
    return noise


def diffuse_phase(fm, optics, cfg, seed: int, weights: dict, k_max: float) -> np.ndarray | None:
    """Coherent diffuse (elastic) scattering as a world-locked random phase field.

    For each material ``m``: unit white noise filtered to the screened-Rutherford power
    spectrum ``PSD(k) ~ 1 / (k^2 + k0^2)^2`` (``k0 = theta0 / lambda``) for ``|k| <= k_max``,
    normalised to unit variance, times ``sqrt(w_m(r))``: a weak phase whose scattered fraction is
    ``w_m`` with the kinematic diffuse angular distribution (a single frozen configuration)."""
    from .diffraction import screening_angle_mrad

    view = fm.view
    ny, nx = view.shape
    p_nm = view.pixel_um * 1000.0
    ky = sfft.fftfreq(ny, p_nm).astype(np.float32)[:, None]
    kx = sfft.rfftfreq(nx, p_nm).astype(np.float32)[None, :]  # real noise: the half spectrum
    k2 = kx * kx + ky * ky
    # the mean of the PSD over the full spectrum, from the half one (columns 1..(nx-1)//2
    # stand for two)
    cw = np.full(kx.shape[1], 2.0, np.float32)
    cw[0] = 1.0
    if nx % 2 == 0:
        cw[-1] = 1.0
    out = None
    for m, w in weights.items():
        if w is None or not (w > 0).any():
            continue
        k0 = screening_angle_mrad(m, optics.wavelength_nm) / (1000.0 * optics.wavelength_nm)
        salt = (cfg.texture_seed_salt * 31 + 7919 * int(m)) ^ (int(round(math.log2(max(p_nm, 1e-6)) * 64)) & 0xFFFF)
        noise = _world_locked_noise(view, seed, salt)
        psd = np.float32(1.0) / (k2 + np.float32(k0 * k0)) ** 2
        psd[k2 > np.float32(k_max * k_max)] = 0
        mean = float((psd * cw).sum(dtype=np.float64)) / (ny * nx)
        filt = np.sqrt(psd / np.float32(mean)).astype(np.float32)  # unit-variance white noise stays unit variance
        spec = sfft.rfft2(noise, workers=-1)
        spec *= filt
        field = sfft.irfft2(spec, s=(ny, nx), workers=-1, overwrite_x=True).astype(np.float32, copy=False)
        phi = np.sqrt(w).astype(np.float32) * field
        out = phi if out is None else out + phi
    return out


def transmission_function(fm, optics, grains, crystallinity, cfg, seed: int, *,
                          return_contrast: bool = False, coherent_k_max: float = 0.0, _scratch: bool = False):
    """The specimen's complex transmission ``t(r) = A(r) exp(i phi(r))`` on ``fm.view``
    (complex64, built in threaded row chunks), shared by TEM imaging and coherent STEM::

        A   = sqrt(I_amp) exp(-kappa phi_tex) [x refraction loss]
        phi = sigma V0 t (edge-tapered) + phi_tex + phi_lat

    The sampling is the FieldMap's view (any pixel size / window: the TEM raster, or the
    fine real-space grid of a coherent 4D-STEM simulation). ``optics`` supplies the beam
    (HT, tilt, objective aperture for the Bragg loss); its ``view`` is replaced by
    ``fm.view``. With ``return_contrast`` it returns ``(t, BraggContrast)``.

    ``coherent_k_max`` (1/nm, coherent STEM): scattering that a coherent simulation band-limited
    to ``coherent_k_max`` can represent stays in the wave instead of being an amplitude loss:
    Bragg beams inside it become a kinematic-strength phase grating (:func:`bragg_contrast`),
    and of the mass-thickness diffuse part ``dw = (1 - T) exp(-t / 10 Lambda)`` only the
    screened-Rutherford fraction beyond ``theta_c = lambda coherent_k_max``,
    ``f_b = theta0^2 / (theta_c^2 + theta0^2)``, is removed:
    ``I_amp = (T + dw (1 - f_b)) (1 - Bragg loss beyond k_max)``; the kept part ``dw (1 - f_b)``
    is scattered coherently by a screened-Rutherford phase field (:func:`diffuse_phase`). The
    renderer adds the removed electrons back incoherently, so totals and the angular
    distribution match the kinematic model.
    """
    if optics.view != fm.view:
        import dataclasses
        optics = dataclasses.replace(optics, view=fm.view)
    view = optics.view
    ny, nx = view.shape
    p_nm = view.pixel_um * 1000.0
    sigma = np.float32(interaction_constant(optics.ht_kv))
    bc = bragg_contrast(fm, optics, grains, crystallinity, cfg, coherent_k_max,
                        _scratch=_scratch and not return_contrast)
    amorphous = np.array([m.amorphous for m in MATERIALS])
    keep_diffuse = None
    if coherent_k_max > 0 and cfg.diffuse_scattering:
        from .diffraction import screening_angle_mrad
        theta_c = 1000.0 * optics.wavelength_nm * coherent_k_max
        kd = [0.0]
        for m in range(1, len(MATERIALS)):
            t0 = screening_angle_mrad(m, optics.wavelength_nm)
            kd.append(1.0 - t0 * t0 / (theta_c * theta_c + t0 * t0))
        keep_diffuse = np.array(kd, np.float32)
        lam_abs = absorption_lengths_nm(optics.ht_kv)
        dkeep = {}
    mip_v = material_array("mean_inner_potential_v")
    use_tex = cfg.phase_texture_scale > 0 and (bool(amorphous[np.unique(fm.material_id[::7, ::7])].any())
                                               or fm.under_thickness_nm is not None)
    noise = texture_noise(optics, cfg, seed, _scratch and not return_contrast) if use_tex else None
    tex_k = np.float32(sigma * cfg.phase_texture_scale / p_nm)
    w = cfg.amplitude_contrast
    kappa = np.float32(w / math.sqrt(max(1e-9, 1.0 - w * w))) if w > 0 else np.float32(0.0)
    inv_dens = (1.0 / material_array("scatterer_density_nm3")).astype(np.float32)
    psi = np.empty((ny, nx), np.complex64)
    t_full = np.empty((ny, nx), np.float32)
    mip = None
    loss = None
    taper_px = cfg.edge_taper_nm / p_nm
    if _fp.AVAILABLE:
        return _transmission_fast(fm, optics, cfg, bc, noise, sigma, mip_v, amorphous, tex_k, kappa,
                                  inv_dens, p_nm, taper_px, return_contrast, _scratch and not return_contrast,
                                  keep_diffuse, seed, coherent_k_max)
    if cfg.mip_phase:
        # mean-inner-potential phase with rounded (not ideal-step) edges
        f_t = np.float32(optics.thickness_tilt_factor)

        def mip_rows(r0, r1):
            m = sigma * mip_v[fm.material_id[r0:r1]] * (fm.thickness_nm[r0:r1].astype(np.float32) * f_t)
            if fm.under_thickness_nm is not None:
                m = m + sigma * mip_v[fm.under_material[r0:r1]] * (fm.under_thickness_nm[r0:r1] * f_t)
            return m
        mip = np.empty((ny, nx), mip_rows(0, 1).dtype)
        parallel_rows(lambda r0, r1: mip.__setitem__(slice(r0, r1), mip_rows(r0, r1)), ny)
        if taper_px > 0.3:
            mip = gaussian_filter_threaded(mip, min(taper_px, 16.0), mode="nearest", truncate=3.0)
        if cfg.refraction_loss:
            # Electrons refracted by a steep phase gradient beyond the imaging band (objective
            # aperture, or half the raster Nyquist) are lost: amplitude *= exp(-(g/g_c)^4),
            # g = |grad phi| in rad/px. This also stops a 30 rad particle rim from aliasing.
            g_c = 0.5 * math.pi
            if optics.objective_aperture_mrad > 0:
                g_c = min(g_c, 2.0 * math.pi * p_nm * optics.objective_aperture_mrad * 1e-3
                          / optics.wavelength_nm)
            q = np.empty_like(mip)
            inv_gc2 = np.float32(1.0 / (g_c * g_c))

            def q_rows(r0, r1):
                gy = np.zeros((r1 - r0, nx), mip.dtype)
                gx = np.zeros((r1 - r0, nx), mip.dtype)
                a, b = max(r0, 1), min(r1, ny - 1)
                if b > a:
                    gy[a - r0:b - r0] = 0.5 * (mip[a + 1:b + 1] - mip[a - 1:b - 1])
                gx[:, 1:-1] = 0.5 * (mip[r0:r1, 2:] - mip[r0:r1, :-2])
                q[r0:r1] = (gx * gx + gy * gy) * inv_gc2
            parallel_rows(q_rows, ny)
            if float(q.max()) > 0.01:
                loss = np.empty((ny, nx), np.float32)
                parallel_rows(lambda r0, r1: loss.__setitem__(
                    slice(r0, r1), np.exp(-q[r0:r1] * q[r0:r1]).astype(np.float32)), ny)

    def work(r0, r1):
        I, t = amplitude_rows(fm, optics, r0, r1, bc.loss)
        t_full[r0:r1] = t
        mat = fm.material_id[r0:r1]
        if keep_diffuse is not None:
            T = np.exp(-t / lam_abs[mat])
            dw = (1.0 - T) * np.exp(-t / (DIFFUSE_CUTOFF_LENGTHS * lam_abs[mat]))
            kept = (dw * keep_diffuse[mat]).astype(np.float32)
            I = np.clip((T + kept) * (1.0 - bc.loss[r0:r1]), 0.0, 1.0).astype(np.float32)
            dkeep[(r0, r1)] = kept
        v0 = mip_v[mat]
        if mip is not None:
            phi = mip[r0:r1].copy()
        else:
            phi = np.zeros_like(t)
        amp = np.sqrt(I)
        if loss is not None:
            amp *= loss[r0:r1]
        if noise is not None:
            var = np.where(amorphous[mat], (tex_k * v0) ** 2 * t * inv_dens[mat], np.float32(0.0))
            if fm.under_thickness_nm is not None:
                um = fm.under_material[r0:r1]
                tu = fm.under_thickness_nm[r0:r1] * np.float32(optics.thickness_tilt_factor)
                var = var + (tex_k * mip_v[um]) ** 2 * tu * inv_dens[um]
            tex = (np.sqrt(var) * noise[r0:r1]).astype(np.float32)
            phi += tex
            if kappa:
                amp *= np.exp(-kappa * tex)
        psi.real[r0:r1] = amp * np.cos(phi)
        psi.imag[r0:r1] = amp * np.sin(phi)

    parallel_rows(work, ny)
    if keep_diffuse is not None and dkeep:
        kept = np.empty((ny, nx), np.float32)
        for (r0, r1), v in dkeep.items():
            kept[r0:r1] = v
        weights = {int(m): np.where(fm.material_id == m, kept, np.float32(0.0))
                   for m in np.unique(fm.material_id) if m != 0}
        phi_d = diffuse_phase(fm, optics, cfg, seed, weights, coherent_k_max)
        if phi_d is not None:
            psi *= np.exp(1j * phi_d).astype(np.complex64)
    if bc.phase_grating or bc.kinematic or (cfg.lattice_fringes and cfg.lattice_phase_rad > 0):
        lat = lattice_field(fm, optics, bc, t_full)
        if lat is not None:
            (rows, cols), val, _ = lat
            scale = np.float32(1.0 if (bc.phase_grating or bc.kinematic) else cfg.lattice_phase_rad)

            def fringe(a, b):
                r, c = rows[a:b], cols[a:b]
                psi[r, c] *= np.exp(1j * scale * val[a:b]).astype(np.complex64)
            parallel_rows(fringe, len(rows), 16384)
    return (psi, bc) if return_contrast else psi


def _transmission_fast(fm, optics, cfg, bc, noise, sigma, mip_v, amorphous, tex_k, kappa, inv_dens, p_nm,
                       taper_px, return_contrast, use_scratch=False, keep_diffuse=None, seed=0,
                       coherent_k_max=0.0):
    """`transmission_function` through :mod:`.fastpath` (``keep_diffuse``: the coherent-STEM
    diffuse part, see there)."""
    ny, nx = fm.material_id.shape
    has_keep = keep_diffuse is not None
    kept = np.empty((ny, nx), np.float32) if has_keep else None
    f_t = np.float32(optics.thickness_tilt_factor)
    has_under = fm.under_thickness_nm is not None
    umat = fm.under_material if has_under else fm.material_id
    uthick = fm.under_thickness_nm if has_under else fm.thickness_nm
    mip = np.zeros((1, 1), np.float32)
    refraction = False
    inv_gc2 = np.float32(0.0)
    alloc = _fp.scratch if use_scratch else (lambda name, shape, dt: np.empty(shape, dt))
    if cfg.mip_phase:
        mip = alloc("mip", (ny, nx), np.float32)
        _fp.mip_phase(fm.material_id, fm.thickness_nm, umat, uthick, has_under, f_t, np.float32(sigma),
                      mip_v, mip)
        if taper_px > 0.3:
            mip = _fp.gaussian_nearest(mip, min(taper_px, 16.0), truncate=3.0)
        if cfg.refraction_loss:
            g_c = 0.5 * math.pi
            if optics.objective_aperture_mrad > 0:
                g_c = min(g_c, 2.0 * math.pi * p_nm * optics.objective_aperture_mrad * 1e-3
                          / optics.wavelength_nm)
            inv_gc2 = np.float32(1.0 / (g_c * g_c))
            refraction = bool(_fp.refraction_qmax(mip, inv_gc2) > 0.01)
    psi = alloc("psi", (ny, nx), np.complex64)
    t_full = alloc("t_full", (ny, nx), np.float32)
    lam_abs = absorption_lengths_nm(optics.ht_kv)
    _fp.exit_wave(fm.material_id, fm.thickness_nm, umat, uthick, has_under, f_t, lam_abs, bc.loss, mip,
                  bool(cfg.mip_phase), refraction, inv_gc2,
                  noise if noise is not None else np.zeros((1, 1), np.float32), noise is not None,
                  amorphous, np.float32(tex_k), mip_v, inv_dens, np.float32(kappa), psi, t_full,
                  keep_diffuse if has_keep else np.zeros(1, np.float32), has_keep, DIFFUSE_CUTOFF_LENGTHS,
                  kept if has_keep else np.zeros((1, 1), np.float32))
    if has_keep:
        weights = {int(m): np.where(fm.material_id == m, kept, np.float32(0.0))
                   for m in np.flatnonzero(np.bincount(fm.material_id.ravel(), minlength=256)) if m != 0}
        phi_d = diffuse_phase(fm, optics, cfg, seed, weights, coherent_k_max)
        if phi_d is not None:
            _fp.mul_phase(psi, phi_d)
    if bc.phase_grating or bc.kinematic or (cfg.lattice_fringes and cfg.lattice_phase_rad > 0):
        lat = lattice_field(fm, optics, bc, t_full)
        if lat is not None:
            (rows, cols), val, _ = lat
            scale = np.float32(1.0 if (bc.phase_grating or bc.kinematic) else cfg.lattice_phase_rad)

            def fringe(a, b):
                r, c = rows[a:b], cols[a:b]
                psi[r, c] *= np.exp(1j * scale * val[a:b]).astype(np.complex64)
            parallel_rows(fringe, len(rows), 16384)
    return (psi, bc) if return_contrast else psi


def exit_wave(fm, optics, grains, crystallinity, cfg, seed: int) -> np.ndarray:
    """TEM exit wave under plane-wave illumination: the transmission function itself."""
    return transmission_function(fm, optics, grains, crystallinity, cfg, seed)


def _shift_ramps(shape, shift_px) -> tuple[np.ndarray, np.ndarray] | None:
    """Row and column phase ramps (complex64, FFT layout) that move an image's content by
    ``shift_px = (rows, cols)`` pixels (a sub-pixel, band-limited translation), or None."""
    dy, dx = (float(v) for v in shift_px)
    if dy == 0.0 and dx == 0.0:
        return None
    ny, nx = shape
    ry = np.exp(-2j * np.pi * sfft.fftfreq(ny) * dy).astype(np.complex64)
    rx = np.exp(-2j * np.pi * sfft.fftfreq(nx) * dx).astype(np.complex64)
    return ry, rx


def render_physical(fm, optics, grains, crystallinity, cfg, seed: int, cache: TransferCache,
                    fm_token, shift_px=(0.0, 0.0)) -> np.ndarray:
    """|psi|^2 on the raster; ``shift_px`` (rows, cols) translates the image content by a
    fraction of a pixel (a phase ramp on the spectrum), so a raster on a world-fixed pixel
    lattice can show a view centred between its pixels."""
    view = optics.view
    shape = view.shape
    p_nm = view.pixel_um * 1000.0
    skey = (fm_token, round(optics.ht_kv, 6), round(optics.thickness_tilt_factor, 12), optics.alpha_rad,
            optics.beta_rad, optics.objective_aperture_mrad, optics.convergence_mrad, cfg.max_g_inv_nm,
            cfg.diffraction_contrast_scale, cfg.mip_phase, cfg.edge_taper_nm, cfg.refraction_loss, cfg.phase_texture_scale,
            cfg.texture_bandlimit_nm, cfg.amplitude_contrast, cfg.lattice_fringes,
            cfg.lattice_phase_rad, cfg.lattice_fringe_model, cfg.lattice_fringe_efficiency, seed)
    hkey = (shape, p_nm, optics.wavelength_nm, image_aberrations_of(optics).key(), _raster_tilt(optics),
            optics.illumination_mrad, optics.focal_spread_nm, optics.objective_aperture_mrad)
    if _fp.AVAILABLE:  # spectra and transfer kept TRANSPOSED (fastpath.fft2_t), buffers reused
        skey = skey + ("T",)
        spec = cache.spectra.get(skey)
        if spec is None:
            psi = transmission_function(fm, optics, grains, crystallinity, cfg, seed, _scratch=True)
            spec = _fp.fft2_t(psi, out=cache.buffer((shape[1], shape[0]), np.complex64))
            cache._put(cache.spectra, skey, spec)
        else:
            cache.spectra.move_to_end(skey)
        hkey = hkey + ("T",)
        if hkey in cache.transfer:
            HT = cache.transfer[hkey]
            cache.transfer.move_to_end(hkey)
        else:
            HT = transfer_function(shape, p_nm, optics, transposed=True)
            cache._put(cache.transfer, hkey, HT)
        ramps = _shift_ramps(shape, shift_px)
        prod = _fp.scratch("prod", spec.shape, np.complex64)
        one = np.ones(1, np.complex64)
        _fp.spectrum_product(spec, HT if HT is not None else np.ones((1, 1), np.complex64), HT is not None,
                             ramps[1] if ramps else one, ramps[0] if ramps else one, ramps is not None, prod)
        img = _fp.ifft2_t(prod, out=_fp.scratch("img", shape, np.complex64))
        from .. import buffers

        out = buffers.empty(shape, np.float32)
        _fp.intensity(img, out)
        return out
    spec = cache.spectra.get(skey)
    if spec is None:
        psi = exit_wave(fm, optics, grains, crystallinity, cfg, seed)
        spec = sfft.fft2(psi, workers=-1, overwrite_x=True)
        cache._put(cache.spectra, skey, spec)
    else:
        cache.spectra.move_to_end(skey)
    if hkey in cache.transfer:
        H = cache.transfer[hkey]
        cache.transfer.move_to_end(hkey)
    else:
        H = transfer_function(shape, p_nm, optics)
        cache._put(cache.transfer, hkey, H)
    ramps = _shift_ramps(shape, shift_px)
    if H is None and ramps is None:
        psi = sfft.ifft2(spec, workers=-1)
    else:
        prod = np.empty_like(spec)

        def mult(r0, r1):
            if H is None:
                prod[r0:r1] = spec[r0:r1]
            else:
                np.multiply(spec[r0:r1], H[r0:r1], out=prod[r0:r1])
            if ramps is not None:
                prod[r0:r1] *= ramps[0][r0:r1, None]
                prod[r0:r1] *= ramps[1][None, :]
        parallel_rows(mult, shape[0])
        psi = sfft.ifft2(prod, workers=-1, overwrite_x=True)
    out = np.empty(shape, np.float32)

    def mag2(r0, r1):
        re, im = psi.real[r0:r1], psi.imag[r0:r1]
        np.multiply(re, re, out=out[r0:r1])
        out[r0:r1] += im * im
    parallel_rows(mag2, shape[0])
    return out


# --------------------------------------------------------------- driver
def render_tem(fm, optics, grains, crystallinity, cfg, seed: int, cache: TransferCache,
               fm_token) -> np.ndarray:
    """float32 electrons / detector pixel / s at ``optics.output_shape``."""
    return finish_tem(render_tem_raster(fm, optics, grains, crystallinity, cfg, seed, cache,
                                        fm_token), optics, cfg)


def render_tem_raster(fm, optics, grains, crystallinity, cfg, seed: int, cache: TransferCache,
                      fm_token, shift_px=(0.0, 0.0)) -> np.ndarray:
    """The specimen's image in RASTER space, before the illumination disc and upsampling —
    what moves rigidly with the stage (see `Renderer` panning). ``shift_px`` (rows, cols)
    translates it by a fraction of a pixel."""
    if cfg.tem_model == "legacy":
        img = render_legacy(fm, optics, grains, crystallinity, cfg)
        ramps = _shift_ramps(img.shape, shift_px)
        if ramps is None:
            return img
        spec = sfft.fft2(img, workers=-1) * ramps[0][:, None] * ramps[1][None, :]
        return np.clip(sfft.ifft2(spec, workers=-1).real, 0.0, None).astype(np.float32)
    return render_physical(fm, optics, grains, crystallinity, cfg, seed, cache, fm_token, shift_px)


def finish_tem(img, optics, cfg) -> np.ndarray:
    """The illumination disc (fixed on the detector, not the specimen) and the upsampling
    to ``optics.output_shape``."""
    view = optics.view
    if cfg.illumination_profile and optics.illuminated_diameter_um > 0:
        ny, nx = view.shape
        d = max(1, optics.raster_downsample)
        off = optics.extras.get("roi_offset_px", (0.0, 0.0))
        bo = getattr(optics, "beam_offset_px", (0.0, 0.0))
        cx = (nx - 1) / 2.0 - off[0] / d + bo[0]
        cy = (ny - 1) / 2.0 - off[1] / d + bo[1]
        radius = optics.illuminated_diameter_um * 1000.0 / 2.0 / (view.pixel_um * 1000.0)
        prof = disk_profile(view.shape, cx, cy, radius, max(0.5, 0.01 * radius))
        if prof is not None:
            img = img * prof
    return upsample_to(img, optics.raster_downsample, optics.output_shape, optics.dose_e_per_px_s)
