"""TEM amplitude: mass-thickness and Bragg diffraction contrast, and lattice fringes."""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from ..crystal import effective_matrices, library_for
from ..hashing import SeedKind, hash_seed, uniform_from_hash
from ..specimen.materials import GRAINS_PER_MATERIAL, absorption_lengths_nm
from . import fastpath as _fp
from .diffraction import MAX_THICKNESS_BIN, THICKNESS_BIN_NM, bragg_fraction, thickness_bin_for
from .util import parallel_rows, pool

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


_BRAGG_MEMOS: OrderedDict[tuple, _BraggMemo] = OrderedDict()
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


def bragg_contrast(fm, optics, grains, crystallinity, config, coherent_k_max: float = 0.0,
                   _scratch: bool = False) -> BraggContrast:
    with _BRAGG_LOCK:
        return _bragg_contrast(fm, optics, grains, crystallinity, config, coherent_k_max, _scratch)


def _bragg_contrast(fm, optics, grains, crystallinity, config, coherent_k_max: float = 0.0,
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
        lib = library_for(int(mid), config.max_g_inv_nm)
        if lib is None:
            continue  # an amorphous material: no Bragg contrast (the FIB-liftout pattern's "auto" post)
        memo = _bragg_memo(grains, int(mid), config.max_g_inv_nm, optics, g_obj)
        g0 = int(mid) * GRAINS_PER_MATERIAL
        g1 = min(g0 + GRAINS_PER_MATERIAL, n)
        tab = memo.table(lib, grains, int(mid), optics, None if maxbin is None else maxbin[g0:g1])
        ids = np.arange(g0, g1)
        crm = np.ones(len(ids)) if crystallinity is None else np.asarray(crystallinity(ids), float).reshape(-1)
        lut[g0 * nb:g1 * nb] = (config.diffraction_contrast_scale * (crm[:, None] * tab[:g1 - g0])).ravel()
        # fringes need beams the raster resolves (and the aperture passes)
        if coherent or (config.lattice_fringes and memo.min_gxy <= g_res):
            fringe_mids.append(int(mid))
    if _fp.AVAILABLE:
        _fp.gather(lut, pix_all, loss)
    else:
        parallel_rows(lambda r0, r1: loss.__setitem__(slice(r0, r1), lut[pix_all[r0:r1]]), mat.shape[0])
    if not fringe_mids:
        return BraggContrast(loss, gid, fringes, coherent, t_typ,
                             coherent or (config.lattice_fringes and config.lattice_fringe_model == "kinematic"))
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
        lib = library_for(int(mid), config.max_g_inv_nm)
        memo = _bragg_memo(grains, int(mid), config.max_g_inv_nm, optics, g_obj)
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
                kin = config.lattice_fringe_model == "kinematic"
                for i, fi in zip(k[np.argsort(-f)], np.sort(f)[::-1]):
                    if len(waves) == MAX_FRINGE_BEAMS:
                        break
                    if not any(abs(gx_all[i] + wx) < 1e-6 and abs(gy_all[i] + wy) < 1e-6
                               for wx, wy, _ in waves):
                        # the Friedel pair (g, -g) carries 2 f: a phase grating 2 sqrt(f) cos(g r)
                        a = (config.lattice_fringe_efficiency * math.sqrt(2.0 * 2.0 * fi) if kin
                             else min(1.0, math.sqrt(fi / FRINGE_BEAM_FRACTION)))
                        waves.append((gx_all[i], gy_all[i], a * cr[gi]))
                if waves:
                    w = np.array(waves)
                    fringes[int(g_u[gi])] = (2 * math.pi * w[:, 0], 2 * math.pi * w[:, 1], w[:, 2])
                    t_typ[int(g_u[gi])] = max(float(tb_u[typical[gi]]) * THICKNESS_BIN_NM, THICKNESS_BIN_NM)
    kinematic = coherent or (config.lattice_fringes and config.lattice_fringe_model == "kinematic")
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
def lattice_field(optics, bc: BraggContrast, t):
    """sum_w a_w cos(g_w . r + phase_w) / n_w over each grain's strongest resolved beams,
    times min(1, t / 10 nm). Returns ((rows, cols), weighted, unweighted) or None."""
    gids = list(bc.fringes)
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
