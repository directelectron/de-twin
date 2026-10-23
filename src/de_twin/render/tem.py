"""TEM imaging.

The image starts from the amplitude term, mass-thickness plus diffraction contrast::

    T = exp(-t * f_tilt / Lambda(material, HT))       (f_tilt = 1/(cos a cos b))
    I_amp = clamp(T * (1 - s * c * D_out), 0, 1)

``D_out`` is the fraction of the beam Bragg-scattered *outside the objective aperture*
(all of it when there is none) by the pixel's grain at its effective orientation (grain x
stage tilt) and thickness, from :mod:`de_twin.crystal` (the same kinematic excitation as
the diffraction patterns); ``c`` is the crystallinity and ``s`` ``diffraction_contrast_scale``.
Tilting the stage therefore moves grains in and out of Bragg conditions (bend/thickness
contours included).

A proper exit wave propagated through the objective transfer function::

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

import numpy as np
from scipy import fft as sfft

from ..optics.physics import interaction_constant
from ..optics.state import image_aberrations_of
from ..specimen.materials import MATERIALS, absorption_lengths_nm, material_array
from . import fastpath as _fp
from .samples import DIFFUSE_CUTOFF_LENGTHS
from .tem_contrast import amplitude_rows, bragg_contrast, lattice_field
from .util import (
    chi_and_gradient,
    disk_profile,
    gaussian_filter_threaded,
    parallel_rows,
    upsample_to,
    world_normal_noise,
)


# ------------------------------------------------------------- physical
class TransferCache:
    """Caches for the wave-optical model: exit-wave spectra and transfer functions."""

    def __init__(self, size: int = 4):
        self.spectra: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self.transfer: OrderedDict[tuple, np.ndarray | None] = OrderedDict()
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
    if np.abs(c - c0).max() < 2e-3 and envp.min() > 0.999 and kap == 0.0 and tx == 0 and ty == 0:
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
            kx1.astype(np.float64),
            ky1.astype(np.float64),
            float(tx),
            float(ty),
            f(s * l2 * c1 / 2),
            f(s * l2 * l2 * c3 / 4),
            f(s * l2**3 * c5 / 6),
            f(s * l2 * a1.real / 2),
            f(s * l2 * a1.imag),
            f(s * l2 * c1),
            f(s * l2 * l2 * c3),
            f(s * l2**3 * c5),
            f(s * l2 * a1.real),
            f(s * l2 * a1.imag),
            f(c0),
            f(gx0),
            f(gy0),
            f(es),
            f(et),
            f(kt2),
            f(kap * kap),
            H,
            transposed,
        )
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
    if (
        not getattr(view, "rotation_rad", 0.0)
        and not getattr(view, "flip_x", False)
        and not getattr(view, "flip_y", False)
    ):
        # the nearest cell, ties (a view centred on the pixel lattice) broken upwards
        ix0 = math.floor(view.center_um[0] / pitch_x + (0.5 - nx / 2.0) + 0.5 + 1e-6)
        iy0 = math.floor(view.center_um[1] / pitch_y + (0.5 - ny / 2.0) + 0.5 + 1e-6)
        return world_normal_noise(
            ix0,
            iy0,
            ny,
            nx,
            seed,
            salt,
            _fp.scratch("noise", (ny, nx), np.float32) if (_scratch and _fp.AVAILABLE) else None,
        )
    rows, cols = np.mgrid[0:ny, 0:nx]
    x, y = view.pixel_to_world(rows.ravel(), cols.ravel())
    ix = np.floor(np.asarray(x) / pitch_x).astype(np.int64)
    iy = np.floor(np.asarray(y) / pitch_y).astype(np.int64)
    x0, y0 = int(ix.min()), int(iy.min())
    block = world_normal_noise(x0, y0, int(iy.max()) - y0 + 1, int(ix.max()) - x0 + 1, seed, salt)
    return block[iy - y0, ix - x0].reshape(ny, nx)


def texture_noise(optics, config, seed: int, _scratch: bool = False) -> np.ndarray:
    """World-locked unit white noise on the raster, band-limited by the atomic form factor."""
    view = optics.view
    p_nm = view.pixel_um * 1000.0
    salt = config.texture_seed_salt ^ (round(math.log2(max(p_nm, 1e-6)) * 64) & 0xFFFF)
    noise = _world_locked_noise(view, seed, salt, _scratch)
    sb = config.texture_bandlimit_nm / p_nm
    if sb > 0.3:
        noise = gaussian_filter_threaded(noise, sb, mode="wrap")
    return noise


def diffuse_phase(field_map, optics, config, seed: int, weights: dict, k_max: float) -> np.ndarray | None:
    """Coherent diffuse (elastic) scattering as a world-locked random phase field.

    For each material ``m``: unit white noise filtered to the screened-Rutherford power
    spectrum ``PSD(k) ~ 1 / (k^2 + k0^2)^2`` (``k0 = theta0 / lambda``) for ``|k| <= k_max``,
    normalised to unit variance, times ``sqrt(w_m(r))``: a weak phase whose scattered fraction is
    ``w_m`` with the kinematic diffuse angular distribution (a single frozen configuration)."""
    from .diffraction import screening_angle_mrad

    view = field_map.view
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
        salt = (config.texture_seed_salt * 31 + 7919 * int(m)) ^ (
            round(math.log2(max(p_nm, 1e-6)) * 64) & 0xFFFF
        )
        noise = _world_locked_noise(view, seed, salt)
        psd = np.float32(1.0) / (k2 + np.float32(k0 * k0)) ** 2
        psd[k2 > np.float32(k_max * k_max)] = 0
        mean = float((psd * cw).sum(dtype=np.float64)) / (ny * nx)
        filt = np.sqrt(psd / np.float32(mean)).astype(
            np.float32
        )  # unit-variance white noise stays unit variance
        spec = sfft.rfft2(noise, workers=-1)
        spec *= filt
        field = sfft.irfft2(spec, s=(ny, nx), workers=-1, overwrite_x=True).astype(np.float32, copy=False)
        phi = np.sqrt(w).astype(np.float32) * field
        out = phi if out is None else out + phi
    return out


def transmission_function(
    field_map,
    optics,
    grains,
    crystallinity,
    config,
    seed: int,
    *,
    return_contrast: bool = False,
    coherent_k_max: float = 0.0,
    _scratch: bool = False,
):
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
    if optics.view != field_map.view:
        import dataclasses

        optics = dataclasses.replace(optics, view=field_map.view)
    view = optics.view
    ny, nx = view.shape
    p_nm = view.pixel_um * 1000.0
    sigma = np.float32(interaction_constant(optics.ht_kv))
    bc = bragg_contrast(
        field_map,
        optics,
        grains,
        crystallinity,
        config,
        coherent_k_max,
        _scratch=_scratch and not return_contrast,
    )
    amorphous = np.array([m.amorphous for m in MATERIALS])
    keep_diffuse = None
    if coherent_k_max > 0 and config.diffuse_scattering:
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
    use_tex = config.phase_texture_scale > 0 and (
        bool(amorphous[np.unique(field_map.material_id[::7, ::7])].any())
        or field_map.under_thickness_nm is not None
    )
    noise = texture_noise(optics, config, seed, _scratch and not return_contrast) if use_tex else None
    tex_k = np.float32(sigma * config.phase_texture_scale / p_nm)
    w = config.amplitude_contrast
    kappa = np.float32(w / math.sqrt(max(1e-9, 1.0 - w * w))) if w > 0 else np.float32(0.0)
    inv_dens = (1.0 / material_array("scatterer_density_nm3")).astype(np.float32)
    psi = np.empty((ny, nx), np.complex64)
    t_full = np.empty((ny, nx), np.float32)
    mip = None
    loss = None
    taper_px = config.edge_taper_nm / p_nm
    if _fp.AVAILABLE:
        return _transmission_fast(
            field_map,
            optics,
            config,
            bc,
            noise,
            sigma,
            mip_v,
            amorphous,
            tex_k,
            kappa,
            inv_dens,
            p_nm,
            taper_px,
            return_contrast,
            _scratch and not return_contrast,
            keep_diffuse,
            seed,
            coherent_k_max,
        )
    if config.mip_phase:
        # mean-inner-potential phase with rounded (not ideal-step) edges
        f_t = np.float32(optics.thickness_tilt_factor)

        def mip_rows(r0, r1):
            m = (
                sigma
                * mip_v[field_map.material_id[r0:r1]]
                * (field_map.thickness_nm[r0:r1].astype(np.float32) * f_t)
            )
            if field_map.under_thickness_nm is not None:
                m = m + sigma * mip_v[field_map.under_material[r0:r1]] * (
                    field_map.under_thickness_nm[r0:r1] * f_t
                )
            return m

        mip = np.empty((ny, nx), mip_rows(0, 1).dtype)
        parallel_rows(lambda r0, r1: mip.__setitem__(slice(r0, r1), mip_rows(r0, r1)), ny)
        if taper_px > 0.3:
            mip = gaussian_filter_threaded(mip, min(taper_px, 16.0), mode="nearest", truncate=3.0)
        if config.refraction_loss:
            # Electrons refracted by a steep phase gradient beyond the imaging band (objective
            # aperture, or half the raster Nyquist) are lost: amplitude *= exp(-(g/g_c)^4),
            # g = |grad phi| in rad/px. This also stops a 30 rad particle rim from aliasing.
            g_c = 0.5 * math.pi
            if optics.objective_aperture_mrad > 0:
                g_c = min(
                    g_c, 2.0 * math.pi * p_nm * optics.objective_aperture_mrad * 1e-3 / optics.wavelength_nm
                )
            q = np.empty_like(mip)
            inv_gc2 = np.float32(1.0 / (g_c * g_c))

            def q_rows(r0, r1):
                gy = np.zeros((r1 - r0, nx), mip.dtype)
                gx = np.zeros((r1 - r0, nx), mip.dtype)
                a, b = max(r0, 1), min(r1, ny - 1)
                if b > a:
                    gy[a - r0 : b - r0] = 0.5 * (mip[a + 1 : b + 1] - mip[a - 1 : b - 1])
                gx[:, 1:-1] = 0.5 * (mip[r0:r1, 2:] - mip[r0:r1, :-2])
                q[r0:r1] = (gx * gx + gy * gy) * inv_gc2

            parallel_rows(q_rows, ny)
            if float(q.max()) > 0.01:
                loss = np.empty((ny, nx), np.float32)
                parallel_rows(
                    lambda r0, r1: loss.__setitem__(
                        slice(r0, r1), np.exp(-q[r0:r1] * q[r0:r1]).astype(np.float32)
                    ),
                    ny,
                )

    def work(r0, r1):
        I, t = amplitude_rows(field_map, optics, r0, r1, bc.loss)
        t_full[r0:r1] = t
        mat = field_map.material_id[r0:r1]
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
            if field_map.under_thickness_nm is not None:
                um = field_map.under_material[r0:r1]
                tu = field_map.under_thickness_nm[r0:r1] * np.float32(optics.thickness_tilt_factor)
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
        weights = {
            int(m): np.where(field_map.material_id == m, kept, np.float32(0.0))
            for m in np.unique(field_map.material_id)
            if m != 0
        }
        phi_d = diffuse_phase(field_map, optics, config, seed, weights, coherent_k_max)
        if phi_d is not None:
            psi *= np.exp(1j * phi_d).astype(np.complex64)
    if bc.phase_grating or bc.kinematic or (config.lattice_fringes and config.lattice_phase_rad > 0):
        lat = lattice_field(optics, bc, t_full)
        if lat is not None:
            (rows, cols), val, _ = lat
            scale = np.float32(1.0 if (bc.phase_grating or bc.kinematic) else config.lattice_phase_rad)

            def fringe(a, b):
                r, c = rows[a:b], cols[a:b]
                psi[r, c] *= np.exp(1j * scale * val[a:b]).astype(np.complex64)

            parallel_rows(fringe, len(rows), 16384)
    return (psi, bc) if return_contrast else psi


def _transmission_fast(
    field_map,
    optics,
    config,
    bc,
    noise,
    sigma,
    mip_v,
    amorphous,
    tex_k,
    kappa,
    inv_dens,
    p_nm,
    taper_px,
    return_contrast,
    use_scratch=False,
    keep_diffuse=None,
    seed=0,
    coherent_k_max=0.0,
):
    """`transmission_function` through :mod:`.fastpath` (``keep_diffuse``: the coherent-STEM
    diffuse part, see there)."""
    ny, nx = field_map.material_id.shape
    has_keep = keep_diffuse is not None
    kept = np.empty((ny, nx), np.float32) if has_keep else None
    f_t = np.float32(optics.thickness_tilt_factor)
    has_under = field_map.under_thickness_nm is not None
    umat = field_map.under_material if has_under else field_map.material_id
    uthick = field_map.under_thickness_nm if has_under else field_map.thickness_nm
    mip = np.zeros((1, 1), np.float32)
    refraction = False
    inv_gc2 = np.float32(0.0)
    alloc = _fp.scratch if use_scratch else (lambda name, shape, dt: np.empty(shape, dt))
    if config.mip_phase:
        mip = alloc("mip", (ny, nx), np.float32)
        _fp.mip_phase(
            field_map.material_id,
            field_map.thickness_nm,
            umat,
            uthick,
            has_under,
            f_t,
            np.float32(sigma),
            mip_v,
            mip,
        )
        if taper_px > 0.3:
            mip = _fp.gaussian_nearest(mip, min(taper_px, 16.0), truncate=3.0)
        if config.refraction_loss:
            g_c = 0.5 * math.pi
            if optics.objective_aperture_mrad > 0:
                g_c = min(
                    g_c, 2.0 * math.pi * p_nm * optics.objective_aperture_mrad * 1e-3 / optics.wavelength_nm
                )
            inv_gc2 = np.float32(1.0 / (g_c * g_c))
            refraction = bool(_fp.refraction_qmax(mip, inv_gc2) > 0.01)
    psi = alloc("psi", (ny, nx), np.complex64)
    t_full = alloc("t_full", (ny, nx), np.float32)
    lam_abs = absorption_lengths_nm(optics.ht_kv)
    _fp.exit_wave(
        field_map.material_id,
        field_map.thickness_nm,
        umat,
        uthick,
        has_under,
        f_t,
        lam_abs,
        bc.loss,
        mip,
        bool(config.mip_phase),
        refraction,
        inv_gc2,
        noise if noise is not None else np.zeros((1, 1), np.float32),
        noise is not None,
        amorphous,
        np.float32(tex_k),
        mip_v,
        inv_dens,
        np.float32(kappa),
        psi,
        t_full,
        keep_diffuse if has_keep else np.zeros(1, np.float32),
        has_keep,
        DIFFUSE_CUTOFF_LENGTHS,
        kept if has_keep else np.zeros((1, 1), np.float32),
    )
    if has_keep:
        weights = {
            int(m): np.where(field_map.material_id == m, kept, np.float32(0.0))
            for m in np.flatnonzero(np.bincount(field_map.material_id.ravel(), minlength=256))
            if m != 0
        }
        phi_d = diffuse_phase(field_map, optics, config, seed, weights, coherent_k_max)
        if phi_d is not None:
            _fp.mul_phase(psi, phi_d)
    if bc.phase_grating or bc.kinematic or (config.lattice_fringes and config.lattice_phase_rad > 0):
        lat = lattice_field(optics, bc, t_full)
        if lat is not None:
            (rows, cols), val, _ = lat
            scale = np.float32(1.0 if (bc.phase_grating or bc.kinematic) else config.lattice_phase_rad)

            def fringe(a, b):
                r, c = rows[a:b], cols[a:b]
                psi[r, c] *= np.exp(1j * scale * val[a:b]).astype(np.complex64)

            parallel_rows(fringe, len(rows), 16384)
    return (psi, bc) if return_contrast else psi


def exit_wave(field_map, optics, grains, crystallinity, config, seed: int) -> np.ndarray:
    """TEM exit wave under plane-wave illumination: the transmission function itself."""
    return transmission_function(field_map, optics, grains, crystallinity, config, seed)


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


def render_physical(
    field_map,
    optics,
    grains,
    crystallinity,
    config,
    seed: int,
    cache: TransferCache,
    fm_token,
    shift_px=(0.0, 0.0),
) -> np.ndarray:
    """|psi|^2 on the raster; ``shift_px`` (rows, cols) translates the image content by a
    fraction of a pixel (a phase ramp on the spectrum), so a raster on a world-fixed pixel
    lattice can show a view centred between its pixels."""
    view = optics.view
    shape = view.shape
    p_nm = view.pixel_um * 1000.0
    skey = (
        fm_token,
        round(optics.ht_kv, 6),
        round(optics.thickness_tilt_factor, 12),
        optics.alpha_rad,
        optics.beta_rad,
        optics.objective_aperture_mrad,
        optics.convergence_mrad,
        config.max_g_inv_nm,
        config.diffraction_contrast_scale,
        config.mip_phase,
        config.edge_taper_nm,
        config.refraction_loss,
        config.phase_texture_scale,
        config.texture_bandlimit_nm,
        config.amplitude_contrast,
        config.lattice_fringes,
        config.lattice_phase_rad,
        config.lattice_fringe_model,
        config.lattice_fringe_efficiency,
        seed,
    )
    hkey = (
        shape,
        p_nm,
        optics.wavelength_nm,
        image_aberrations_of(optics).key(),
        _raster_tilt(optics),
        optics.illumination_mrad,
        optics.focal_spread_nm,
        optics.objective_aperture_mrad,
    )
    if _fp.AVAILABLE:  # spectra and transfer kept TRANSPOSED (fastpath.fft2_t), buffers reused
        skey = skey + ("T",)
        spec = cache.spectra.get(skey)
        if spec is None:
            psi = transmission_function(field_map, optics, grains, crystallinity, config, seed, _scratch=True)
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
        _fp.spectrum_product(
            spec,
            HT if HT is not None else np.ones((1, 1), np.complex64),
            HT is not None,
            ramps[1] if ramps else one,
            ramps[0] if ramps else one,
            ramps is not None,
            prod,
        )
        image = _fp.ifft2_t(prod, out=_fp.scratch("img", shape, np.complex64))
        from .. import buffers

        out = buffers.empty(shape, np.float32)
        _fp.intensity(image, out)
        return out
    spec = cache.spectra.get(skey)
    if spec is None:
        psi = exit_wave(field_map, optics, grains, crystallinity, config, seed)
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
def render_tem(
    field_map, optics, grains, crystallinity, config, seed: int, cache: TransferCache, fm_token
) -> np.ndarray:
    """float32 electrons / detector pixel / s at ``optics.output_shape``."""
    return finish_tem(
        render_tem_raster(field_map, optics, grains, crystallinity, config, seed, cache, fm_token),
        optics,
        config,
    )


def render_tem_raster(
    field_map,
    optics,
    grains,
    crystallinity,
    config,
    seed: int,
    cache: TransferCache,
    fm_token,
    shift_px=(0.0, 0.0),
) -> np.ndarray:
    """The specimen's image in RASTER space, before the illumination disc and upsampling —
    what moves rigidly with the stage (see `Renderer` panning). ``shift_px`` (rows, cols)
    translates it by a fraction of a pixel."""
    return render_physical(field_map, optics, grains, crystallinity, config, seed, cache, fm_token, shift_px)


def finish_tem(image, optics, config) -> np.ndarray:
    """The illumination disc (fixed on the detector, not the specimen) and the upsampling
    to ``optics.output_shape``."""
    view = optics.view
    if config.illumination_profile and optics.illuminated_diameter_um > 0:
        ny, nx = view.shape
        d = max(1, optics.raster_downsample)
        off = optics.extras.get("roi_offset_px", (0.0, 0.0))
        bo = getattr(optics, "beam_offset_px", (0.0, 0.0))
        cx = (nx - 1) / 2.0 - off[0] / d + bo[0]
        cy = (ny - 1) / 2.0 - off[1] / d + bo[1]
        radius = optics.illuminated_diameter_um * 1000.0 / 2.0 / (view.pixel_um * 1000.0)
        prof = disk_profile(view.shape, cx, cy, radius, max(0.5, 0.01 * radius))
        if prof is not None:
            image = image * prof
    return upsample_to(image, optics.raster_downsample, optics.output_shape, optics.dose_e_per_px_s)
