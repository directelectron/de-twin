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
from .diffraction import THICKNESS_BIN_NM, bragg_fraction, thickness_bin_for
from .samples import DIFFUSE_CUTOFF_LENGTHS
from .util import chi_and_gradient, disk_profile, parallel_rows, upsample_to, world_normal_noise

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


def bragg_contrast(fm, optics, grains, crystallinity, cfg, coherent_k_max: float = 0.0) -> BraggContrast:
    """Per-pixel Bragg loss from the grain at its effective orientation and thickness bin,
    and the lattice fringes (strongest excited, resolvable, transmitted beams) per grain.

    ``coherent_k_max > 0`` (coherent STEM): beams with ``|g| <= coherent_k_max`` (and resolvable
    on the raster) are not a loss but a phase grating of kinematic strength (all of them, up to
    ``MAX_COHERENT_BEAMS`` Friedel pairs); only the beams beyond are removed (the renderer adds
    them back incoherently)."""
    mat = fm.material_id
    t = fm.thickness_nm.astype(np.float32) * np.float32(optics.thickness_tilt_factor)
    gid = fm.grain_id.astype(np.int64)
    n = len(grains) if grains is not None else 0
    gid = np.where((gid >= 0) & (gid < n) & (gid // GRAINS_PER_MATERIAL == mat), gid, -1)
    loss = np.zeros(mat.shape, np.float32)
    fringes: dict = {}
    if not (gid >= 0).any():
        return BraggContrast(loss, gid, fringes)
    lam = optics.wavelength_nm
    g_obj = optics.objective_aperture_mrad / (1000.0 * lam) if optics.objective_aperture_mrad > 0 else 0.0
    pitch_nm = optics.view.pixel_um * 1000.0 / min(optics.view.cos_alpha, optics.view.cos_beta)
    g_res = 1.0 / (LATTICE_MIN_SAMPLES * pitch_nm)
    coherent = coherent_k_max > 0
    if coherent:
        g_obj = min(coherent_k_max, g_res)
    t_typ: dict = {}
    rows, cols = np.nonzero(gid >= 0)
    g_u, g_inv = np.unique(gid[rows, cols], return_inverse=True)
    tb_u, t_inv = np.unique(thickness_bin_for(t[rows, cols]), return_inverse=True)
    cr = np.ones(len(g_u)) if crystallinity is None else np.asarray(crystallinity(g_u), float).reshape(-1)
    counts = np.zeros((len(g_u), len(tb_u)), np.int64)
    np.add.at(counts, (g_inv, t_inv), 1)
    typical = counts.argmax(axis=1)  # most common thickness bin of each grain (for the fringes)
    table = np.zeros((len(g_u), len(tb_u)))
    for mid in np.unique(g_u // GRAINS_PER_MATERIAL):
        sel = np.flatnonzero(g_u // GRAINS_PER_MATERIAL == mid)
        lib = library_for(int(mid), cfg.max_g_inv_nm)
        if lib is None:
            continue  # an amorphous material: no Bragg contrast (the FIB-liftout pattern's "auto" post)
        m = effective_matrices(grains.matrices[g_u[sel]], optics.alpha_rad, optics.beta_rad)
        ex = lib.excite(
            m, lam, np.broadcast_to(tb_u * THICKNESS_BIN_NM, (len(sel), len(tb_u))), optics.ht_kv,
            optics.convergence_mrad)
        gxy = np.hypot(ex.gx, ex.gy)
        out = gxy > g_obj
        p_out = np.stack([np.bincount(ex.owner, weights=ex.intensity[:, j] * out, minlength=len(sel))
                          for j in range(len(tb_u))], 1)
        frac = bragg_fraction(ex.total) / np.maximum(ex.total, 1e-300)  # electrons per unit I_g
        table[sel] = cr[sel, None] * frac * p_out
        if coherent:
            ok = (gxy <= g_obj) & (gxy > 1e-6)
            for j, gi in enumerate(sel):
                k = np.flatnonzero(ok & (ex.owner == j))
                f = ex.intensity[k, typical[gi]] * frac[j, typical[gi]] * cr[gi]
                pairs: dict = {}
                for i, fi in zip(k, f):
                    gx_, gy_ = float(ex.gx[i]), float(ex.gy[i])
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
        elif cfg.lattice_fringes:
            # beams that reach the image (inside the aperture, or all without one) and are resolved
            ok = (gxy <= g_res) & (gxy > 1e-6) & ((gxy <= g_obj) | (g_obj == 0))
            for j, gi in enumerate(sel):
                k = np.flatnonzero(ok & (ex.owner == j))
                f = ex.intensity[k, typical[gi]] * frac[j, typical[gi]]
                waves: list = []
                for i, fi in zip(k[np.argsort(-f)], np.sort(f)[::-1]):
                    if len(waves) == MAX_FRINGE_BEAMS:
                        break
                    if not any(abs(ex.gx[i] + wx) < 1e-6 and abs(ex.gy[i] + wy) < 1e-6 for wx, wy, _ in waves):
                        waves.append((ex.gx[i], ex.gy[i], min(1.0, math.sqrt(fi / FRINGE_BEAM_FRACTION)) * cr[gi]))
                if waves:
                    w = np.array(waves)
                    fringes[int(g_u[gi])] = (2 * math.pi * w[:, 0], 2 * math.pi * w[:, 1], w[:, 2])
    loss[rows, cols] = (cfg.diffraction_contrast_scale * table[g_inv, t_inv]).astype(np.float32)
    return BraggContrast(loss, gid, fringes, coherent, t_typ)


def amplitude_rows(fm, optics, r0, r1, loss) -> tuple:
    """(I_amp, t_eff) for raster rows r0:r1."""
    mat = fm.material_id[r0:r1]
    t = fm.thickness_nm[r0:r1].astype(np.float32) * np.float32(optics.thickness_tilt_factor)
    T = np.exp(-t / absorption_lengths_nm(optics.ht_kv)[mat])
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
    x_um, y_um = optics.view.pixel_to_world(rows, cols)
    X, Y = x_um * 1000.0, y_um * 1000.0
    g = bc.gid[rows, cols]
    acc = np.zeros(len(rows))
    raw = np.zeros(len(rows))
    grating = bool(getattr(bc, "phase_grating", False))
    weight = np.zeros(len(rows)) if grating else np.minimum(1.0, t[rows, cols] / 10.0)
    for gg in gids:
        sel = g == gg
        kx, ky, amp = bc.fringes[gg]
        norm = 1.0 if grating else 1.0 / len(kx)
        for w in range(len(kx)):
            ph = 2.0 * math.pi * uniform_from_hash(hash_seed(int(gg), SeedKind.GRAIN, w, 0xA111))
            c = np.cos(kx[w] * X[sel] + ky[w] * Y[sel] + ph) * norm
            acc[sel] += amp[w] * c
            raw[sel] += c
        if grating:  # kinematic amplitude grows with the projected thickness
            weight[sel] = np.clip(t[rows[sel], cols[sel]] / bc.t_typ[gg], 0.0, 2.0)
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

    def __init__(self, size: int = 2):
        self.spectra: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self.transfer: "OrderedDict[tuple, np.ndarray | None]" = OrderedDict()
        self.size = size

    def _put(self, d, k, v):
        d[k] = v
        d.move_to_end(k)
        while len(d) > self.size:
            d.popitem(last=False)


def _kgrid(shape, pitch_nm):
    ny, nx = shape
    kx = sfft.fftfreq(nx, d=pitch_nm)
    ky = sfft.fftfreq(ny, d=pitch_nm)
    return kx, ky


def transfer_function(shape, pitch_nm, optics) -> np.ndarray | None:
    """Objective transfer H(k) (complex64, FFT layout) or None when it is ~identity.

    chi comes from ``optics.image_aberrations`` (CEOS set incl. C1 = defocus and A1 = objective
    stigmator, see :mod:`de_twin.optics.aberrations`); beam tilt enters as
    ``chi(k + k_t) - chi(k_t)`` and the envelopes use ``grad chi(k + k_t) - grad chi(k_t)``."""
    lam = optics.wavelength_nm
    ab = image_aberrations_of(optics)
    tx, ty = (v / 1000.0 / lam for v in optics.beam_tilt_mrad)
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


def _world_locked_noise(view, seed: int, salt: int) -> np.ndarray:
    """Unit white noise on the raster of *view*, each pixel the value of the world cell under
    it (cells of pitch (px / cos_beta, px / cos_alpha)), so it moves with the specimen. An
    unrotated view reads a block of cells directly; a rotated one (a realistic column's
    image rotation) takes each pixel's nearest cell."""
    ny, nx = view.shape
    pitch_x = view.pixel_um / view.cos_beta
    pitch_y = view.pixel_um / view.cos_alpha
    if not getattr(view, "rotation_rad", 0.0) and not getattr(view, "flip_x", False)             and not getattr(view, "flip_y", False):
        ix0 = int(round(view.center_um[0] / pitch_x + (0.5 - nx / 2.0)))
        iy0 = int(round(view.center_um[1] / pitch_y + (0.5 - ny / 2.0)))
        return world_normal_noise(ix0, iy0, ny, nx, seed, salt)
    rows, cols = np.mgrid[0:ny, 0:nx]
    x, y = view.pixel_to_world(rows.ravel(), cols.ravel())
    ix = np.floor(np.asarray(x) / pitch_x).astype(np.int64)
    iy = np.floor(np.asarray(y) / pitch_y).astype(np.int64)
    x0, y0 = int(ix.min()), int(iy.min())
    block = world_normal_noise(x0, y0, int(iy.max()) - y0 + 1, int(ix.max()) - x0 + 1, seed, salt)
    return block[iy - y0, ix - x0].reshape(ny, nx)


def texture_noise(optics, cfg, seed: int) -> np.ndarray:
    """World-locked unit white noise on the raster, band-limited by the atomic form factor."""
    view = optics.view
    p_nm = view.pixel_um * 1000.0
    salt = cfg.texture_seed_salt ^ (int(round(math.log2(max(p_nm, 1e-6)) * 64)) & 0xFFFF)
    noise = _world_locked_noise(view, seed, salt)
    sb = cfg.texture_bandlimit_nm / p_nm
    if sb > 0.3:
        noise = ndimage.gaussian_filter(noise, sb, mode="wrap")
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
    kx = sfft.fftfreq(nx, p_nm).astype(np.float32)[None, :]
    k2 = kx * kx + ky * ky
    out = None
    for m, w in weights.items():
        if w is None or not (w > 0).any():
            continue
        k0 = screening_angle_mrad(m, optics.wavelength_nm) / (1000.0 * optics.wavelength_nm)
        salt = (cfg.texture_seed_salt * 31 + 7919 * int(m)) ^ (int(round(math.log2(max(p_nm, 1e-6)) * 64)) & 0xFFFF)
        noise = _world_locked_noise(view, seed, salt)
        psd = np.float32(1.0) / (k2 + np.float32(k0 * k0)) ** 2
        psd[k2 > np.float32(k_max * k_max)] = 0
        filt = np.sqrt(psd / psd.mean()).astype(np.float32)  # unit-variance white noise stays unit variance
        field = sfft.ifft2(sfft.fft2(noise, workers=-1) * filt, workers=-1).real.astype(np.float32)
        phi = np.sqrt(w).astype(np.float32) * field
        out = phi if out is None else out + phi
    return out


def transmission_function(fm, optics, grains, crystallinity, cfg, seed: int, *,
                          return_contrast: bool = False, coherent_k_max: float = 0.0):
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
    bc = bragg_contrast(fm, optics, grains, crystallinity, cfg, coherent_k_max)
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
    use_tex = cfg.phase_texture_scale > 0 and bool(amorphous[np.unique(fm.material_id[::7, ::7])].any())
    noise = texture_noise(optics, cfg, seed) if use_tex else None
    tex_k = np.float32(sigma * cfg.phase_texture_scale / p_nm)
    w = cfg.amplitude_contrast
    kappa = np.float32(w / math.sqrt(max(1e-9, 1.0 - w * w))) if w > 0 else np.float32(0.0)
    inv_dens = (1.0 / material_array("scatterer_density_nm3")).astype(np.float32)
    psi = np.empty((ny, nx), np.complex64)
    t_full = np.empty((ny, nx), np.float32)
    mip = None
    loss = None
    taper_px = cfg.edge_taper_nm / p_nm
    if cfg.mip_phase:
        # mean-inner-potential phase with rounded (not ideal-step) edges
        t_all = fm.thickness_nm.astype(np.float32) * np.float32(optics.thickness_tilt_factor)
        mip = sigma * mip_v[fm.material_id] * t_all
        if taper_px > 0.3:
            mip = ndimage.gaussian_filter(mip, min(taper_px, 16.0), mode="nearest", truncate=3.0)
        if cfg.refraction_loss:
            # Electrons refracted by a steep phase gradient beyond the imaging band (objective
            # aperture, or half the raster Nyquist) are lost: amplitude *= exp(-(g/g_c)^4),
            # g = |grad phi| in rad/px. This also stops a 30 rad particle rim from aliasing.
            g_c = 0.5 * math.pi
            if optics.objective_aperture_mrad > 0:
                g_c = min(g_c, 2.0 * math.pi * p_nm * optics.objective_aperture_mrad * 1e-3
                          / optics.wavelength_nm)
            gy = np.zeros_like(mip)
            gx = np.zeros_like(mip)
            gy[1:-1] = 0.5 * (mip[2:] - mip[:-2])
            gx[:, 1:-1] = 0.5 * (mip[:, 2:] - mip[:, :-2])
            q = (gx * gx + gy * gy) * np.float32(1.0 / (g_c * g_c))
            if float(q.max()) > 0.01:
                loss = np.exp(-q * q).astype(np.float32)

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
            phi = sigma * v0 * t if cfg.mip_phase else np.zeros_like(t)
        amp = np.sqrt(I)
        if loss is not None:
            amp *= loss[r0:r1]
        if noise is not None:
            std = tex_k * v0 * np.sqrt(t * inv_dens[mat])
            tex = np.where(amorphous[mat], std * noise[r0:r1], np.float32(0.0))
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
    if bc.phase_grating or (cfg.lattice_fringes and cfg.lattice_phase_rad > 0):
        lat = lattice_field(fm, optics, bc, t_full)
        if lat is not None:
            (rows, cols), val, _ = lat
            scale = 1.0 if bc.phase_grating else cfg.lattice_phase_rad
            psi[rows, cols] *= np.exp(1j * np.float32(scale) * val).astype(np.complex64)
    return (psi, bc) if return_contrast else psi


def exit_wave(fm, optics, grains, crystallinity, cfg, seed: int) -> np.ndarray:
    """TEM exit wave under plane-wave illumination: the transmission function itself."""
    return transmission_function(fm, optics, grains, crystallinity, cfg, seed)


def render_physical(fm, optics, grains, crystallinity, cfg, seed: int, cache: TransferCache,
                    fm_token) -> np.ndarray:
    view = optics.view
    shape = view.shape
    p_nm = view.pixel_um * 1000.0
    skey = (fm_token, round(optics.ht_kv, 6), round(optics.thickness_tilt_factor, 12), optics.alpha_rad,
            optics.beta_rad, optics.objective_aperture_mrad, optics.convergence_mrad, cfg.max_g_inv_nm,
            cfg.diffraction_contrast_scale, cfg.mip_phase, cfg.edge_taper_nm, cfg.refraction_loss, cfg.phase_texture_scale,
            cfg.texture_bandlimit_nm, cfg.amplitude_contrast, cfg.lattice_fringes,
            cfg.lattice_phase_rad, seed)
    spec = cache.spectra.get(skey)
    if spec is None:
        psi = exit_wave(fm, optics, grains, crystallinity, cfg, seed)
        spec = sfft.fft2(psi, workers=-1, overwrite_x=True)
        cache._put(cache.spectra, skey, spec)
    else:
        cache.spectra.move_to_end(skey)
    hkey = (shape, p_nm, optics.wavelength_nm, image_aberrations_of(optics).key(), optics.beam_tilt_mrad,
            optics.illumination_mrad, optics.focal_spread_nm, optics.objective_aperture_mrad)
    if hkey in cache.transfer:
        H = cache.transfer[hkey]
        cache.transfer.move_to_end(hkey)
    else:
        H = transfer_function(shape, p_nm, optics)
        cache._put(cache.transfer, hkey, H)
    if H is None:
        psi = sfft.ifft2(spec, workers=-1)
    else:
        prod = np.empty_like(spec)
        parallel_rows(lambda r0, r1: np.multiply(spec[r0:r1], H[r0:r1], out=prod[r0:r1]), shape[0])
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
                      fm_token) -> np.ndarray:
    """The specimen's image in RASTER space, before the illumination disc and upsampling —
    what moves rigidly with the stage (see `Renderer` panning)."""
    if cfg.tem_model == "legacy":
        return render_legacy(fm, optics, grains, crystallinity, cfg)
    return render_physical(fm, optics, grains, crystallinity, cfg, seed, cache, fm_token)


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
