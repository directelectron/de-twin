"""Coherent 4D-STEM: probe wavefunction x specimen transmission, far field on the detector.

The kinematic STEM renderer (:mod:`de_twin.render.stem`) draws each scan point's pattern
from a cache of disk patterns; its bright-field disk has no interference, so neither
ptychography nor tilt-corrected bright field (tcBF / parallax) can work on it. This module
simulates the wave-optical 4D-STEM experiment instead.

Model (per scan position ``r_p``)
---------------------------------
::

    probe     psi_p(k) = A(|k|) exp(-i chi(k))            A: condenser aperture |k| <= alpha/lambda
                                                          (edge softened over one simulation pixel)
    chi(k)    from OpticsState.probe_aberrations (CEOS set, de_twin.optics.aberrations):
              C1 = effective defocus (negative = underfocus, the twin's TEM/CTF convention),
              A1 = condenser stigmator (OpticsConfig.condenser_stig_nm_per_unit, 1000 nm/unit),
              C3 (1.2 mm uncorrected, or what a probe corrector leaves: C3 ~ 0, C5, ...),
              beam tilt k_t: chi(k + k_t) - chi(k_t) (off-axis coma/astigmatism, and a probe
              shift = grad chi(k_t) / 2pi, e.g. defocus x tilt).
    exit wave psi(r)   = psi_p(r - r_p) t(r)              projection approximation, single slice
    pattern   I(k)     = |F[psi](k)|^2                    (norm: sum = 1 for a vacuum pattern)
    detector  electrons per binned pixel = pattern_e_per_s x integral of I over the pixel

``t(r) = A(r) exp(i phi(r))`` is :func:`de_twin.render.tem.transmission_function`, the same
object the TEM imaging model uses (mass-thickness amplitude, mean-inner-potential phase,
world-locked amorphous phase texture, grains at their effective orientation incl. stage tilt),
called with ``coherent_k_max = K_t`` (the object band limit, default ``k_det + alpha/lambda``:
everything that can reach the detector). Scattering a band-limited coherent simulation can
represent stays coherent:

* Bragg beams with ``|g| <= K_t`` form a phase grating of kinematic strength
  (``a_g = sqrt(2 f_g)`` per Friedel pair, scaled with the local thickness), so crystal disks
  interfere with the direct beam;
* the mass-thickness diffuse part below ``theta_c = lambda K_t`` is a world-locked random phase
  field with the screened-Rutherford power spectrum (elastic scattering of the amorphous
  potential, one frozen configuration).

Only what lies beyond ``K_t`` is removed from ``t`` and added back incoherently (Bragg disks of the
high-order reflections, the Wentzel tail beyond ``theta_c``), weighted by the probe intensity
over the window. Totals, BF and ADF signals therefore match the kinematic model (tested to a
few %), while the part of the pattern ptychography and tcBF use is coherent.

Sampling
--------
The detector delivers ``(Hd, Wd) = output_shape / hw_binning`` pixels of
``dk_b = recip_pixel_inv_nm * binning`` 1/nm. The simulation uses a square grid of ``n``
pixels with reciprocal pitch ``dk = recip_pixel_inv_nm * g / m`` (``g = gcd(bx, by)``), so
each binned detector pixel integrates ``(m bx/g) x (m by/g)`` simulation pixels, and::

    real-space window  W  = 1/dk = m / (recip * g)   >= 2 (1.1 R_geo + tail lambda/alpha + 3 sigma_s)
    real-space pixel   dx = 1 / (n dk)
    k Nyquist          n dk / 2 >= max(k_det, (k_det + alpha/lambda + K_t) / 2)
                       (t is band-limited to K_t: nothing wraps around onto the detector)

``R_geo = max |grad chi| / 2pi`` over the aperture is the geometric probe radius (``|C1| alpha``
for a defocused probe; ``C3 alpha^3`` for an uncorrected one), so a tcBF-style probe at 1 um
defocus and 20 mrad needs a ~50 nm window. ``m`` (the oversampling factor) is the smallest
integer giving that window; the grid is capped by ``RenderConfig.coherent_max_grid``
(``Sampling.truncated`` reports a window that had to be cut). Diffraction shift and
descan place the optic axis anywhere on the detector: the integer part moves the crop of the
simulation grid, the fractional part is a phase ramp on the probe (an exact k-shift).

Partial coherence
-----------------
* temporal: the focal spread ``Delta`` (1 sigma, from Cc and the energy spread) as a Gauss-Hermite
  quadrature over defocus (1, 3 or 5 samples, by the phase ``pi lambda Delta (alpha/lambda)^2``);
* spatial: a Gaussian effective source (``OpticsState.source_size_nm`` FWHM, from
  ``OpticsConfig.stem_source_size_nm`` scaled by 1/spot) as a 3x3 Gauss-Hermite quadrature of
  probe positions.
The weighted sample probes are decomposed into orthogonal modes (eigenvectors of their Gram
matrix, i.e. the mixed-state / coherent-mode decomposition of the partially coherent probe);
modes are kept until ``coherent_mode_power`` of the intensity (at most ``coherent_max_modes``)
and every pattern is the incoherent sum ``sum_m w_m |F[phi_m t]|^2``. At the defaults (200 kV,
Cc 1.4 mm, 0.7 eV, 0.04 nm source at spot 3) a 20 mrad probe keeps ~66 % in its first mode
(the source is comparable to the diffraction limit): single-mode ePIE still reaches a phase
correlation ~0.85, mixed-state ePIE with the ground-truth modes ~0.9. A larger spot number
shrinks the source (and the current).

Limits (documented)
-------------------
Projection approximation (one slice): valid while the specimen is thinner than the probe's
depth of field ``~lambda/alpha^2`` (6 nm at 20 mrad, 25 nm at 10 mrad) - thick particles are
approximated (no channelling / dynamical thickness dependence beyond ``exp(i phi)``, no HOLZ);
Bragg phases per beam are hashed, not the crystal's structure-factor phases. Reflections beyond
K_t and the Wentzel tail beyond ``lambda K_t`` are incoherent. No inelastic (plasmon) scattering
model, no strain affine (the ``strain`` layer is ignored here), no per-point intensity jitter,
crystalline pixels without their own grain have no Bragg scattering. Scan positions
snap to the simulation grid (``dx``, ~0.02-0.05 nm; the true positions used are reported by
:meth:`CoherentStem.ground_truth`).
"""

from __future__ import annotations


import dataclasses
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import fft as sfft

from ..optics.state import probe_aberrations_of
from ..specimen.fieldmap import ViewWindow
from ..specimen.materials import GRAINS_PER_MATERIAL, MATERIALS, absorption_lengths_nm
from .diffraction import (THICKNESS_BIN_NM, Pattern, PatternOptions, PatternSet, annulus_fractions,
                          bucket_patterns, diffuse_annulus_fraction, render_pattern, screening_angle_mrad,
                          thickness_bin_for)
from .samples import DIFFUSE_CUTOFF_LENGTHS
from .stem import _tilt_key, resolve_scan_point
from .tem import transmission_function
from .util import chi_and_gradient, pool, shift_bilinear

FIELD_MARGIN_PX = 64  # transmission tiles are built this much larger (edge filters, texture band limit)
_GH3 = (np.array([-math.sqrt(3.0), 0.0, math.sqrt(3.0)]), np.array([1 / 6, 2 / 3, 1 / 6]))
_GH5 = (np.array([-2.856970013872806, -1.355626179974266, 0.0, 1.355626179974266, 2.856970013872806]),
        np.array([0.011257411327721, 0.222075922005613, 0.533333333333333, 0.222075922005613,
                  0.011257411327721]))


# ------------------------------------------------------------------ sampling
@dataclass(frozen=True)
class Sampling:
    det_shape: tuple[int, int]  # (Hd, Wd) binned detector pixels
    binning: tuple[int, int]  # (bx, by)
    recip_px: float  # 1/nm per unbinned detector pixel
    n: int  # simulation grid (square)
    dk: float  # 1/nm per simulation k pixel
    sub: tuple[int, int]  # (fx, fy) simulation pixels per binned detector pixel
    center_px: tuple[float, float]  # optic axis on the (unbinned) output, (x, y)
    k_alpha: float  # aperture radius, 1/nm
    probe_size_nm: float  # geometric + diffraction-limited probe diameter, 2 R_geo + 1.22 lambda/alpha
    window_required_nm: float
    truncated: bool  # window or k range had to be cut to respect coherent_max_grid
    object_bandwidth: float = 0.0  # 1/nm: the transmission function is band-limited to this

    @property
    def dx_nm(self) -> float:
        return 1.0 / (self.n * self.dk)

    @property
    def window_nm(self) -> float:
        return 1.0 / self.dk

    @property
    def oversampling(self) -> int:
        return int(round(self.sub[0] * math.gcd(*self.binning) / self.binning[0]))

    @property
    def det_recip_px(self) -> tuple[float, float]:
        """1/nm per binned detector pixel (x, y)."""
        return self.recip_px * self.binning[0], self.recip_px * self.binning[1]

    @property
    def det_center_px(self) -> tuple[float, float]:
        """Optic axis in binned detector pixel coordinates (pixel centres are integers)."""
        cx, cy = self.center_px
        bx, by = self.binning
        return (cx + 0.5) / bx - 0.5, (cy + 0.5) / by - 0.5


def _tilt_k(optics, cfg) -> tuple[float, float]:
    if not getattr(cfg, "coherent_beam_tilt", True):
        return 0.0, 0.0
    lam = optics.wavelength_nm
    return tuple(v / 1000.0 / lam for v in optics.beam_tilt_mrad)


def probe_phase(ab, kx, ky, lam: float, tilt=(0.0, 0.0), gradient: bool = False):
    """chi(k + k_t) - chi(k_t) (and its gradient) for a probe with beam tilt ``k_t``."""
    tx, ty = tilt
    chi, gx, gy = chi_and_gradient(ab, np.asarray(kx) + tx, np.asarray(ky) + ty, lam, gradient=gradient)
    if tx or ty:
        chi = chi - chi_and_gradient(ab, tx, ty, lam, gradient=False)[0]
    return (chi, gx, gy) if gradient else chi


def geometric_probe_radius_nm(ab, lam: float, k_alpha: float, tilt=(0.0, 0.0)) -> float:
    """Largest ray displacement ``|grad chi| / 2pi`` over the aperture (nm)."""
    kr = np.linspace(0.0, k_alpha, 33)[:, None]
    th = np.linspace(0.0, 2 * math.pi, 72, endpoint=False)[None, :]
    _, gx, gy = probe_phase(ab, kr * np.cos(th), kr * np.sin(th), lam, tilt, gradient=True)
    return float(np.hypot(gx, gy).max()) / (2.0 * math.pi)


def plan_sampling(optics, cfg) -> Sampling:
    lam = optics.wavelength_nm
    alpha = optics.convergence_mrad / 1000.0
    ka = alpha / lam
    h, w = (int(v) for v in optics.output_shape)
    bx, by = (max(1, int(v)) for v in optics.extras.get("hw_binning", (1, 1)))
    hd, wd = max(1, h // by), max(1, w // bx)
    recip = float(optics.recip_pixel_inv_nm)
    g = math.gcd(bx, by)
    cx, cy = optics.diffraction_center_px
    kd = recip * max(cx + 0.5, w - 0.5 - cx, cy + 0.5, h - 0.5 - cy, 1.0)
    ab = probe_aberrations_of(optics)
    tilt = _tilt_k(optics, cfg)
    r_geo = geometric_probe_radius_nm(ab, lam, ka, tilt)
    sig_s = optics.source_size_nm / 2.3548
    w_req = 2.0 * (1.1 * r_geo + cfg.coherent_probe_tail * lam / max(alpha, 1e-6) + 3.0 * sig_s)
    dk1 = recip * g
    m = max(1, int(math.ceil(w_req * dk1 - 1e-9)))

    # no wrap-around onto the detector: exit-wave frequencies reach ka + K_t (K_t: the object's
    # band limit, enforced on the transmission tile), aliased ones land at k - 2 k_nyq
    kt = float(cfg.coherent_object_bandwidth_inv_nm)
    kt = kd + ka if kt <= 0 else min(kt, kd + ka)
    k_nyq = max(kd, 0.5 * (kd + ka + kt))

    def grid(m):
        dk = dk1 / m
        fx, fy = m * bx // g, m * by // g
        n = int(math.ceil(2.0 * k_nyq / dk)) + 2
        n = max(n, hd * fy, wd * fx)
        n = sfft.next_fast_len(n + (n & 1))
        while n & 1:
            n = sfft.next_fast_len(n + 1)
        return n, dk, fx, fy

    n, dk, fx, fy = grid(m)
    truncated = False
    cap = int(cfg.coherent_max_grid)
    # over the cap: shrink the window toward the hard minimum (geometric disk + 2 lambda/alpha);
    # only a window below that minimum counts as truncated
    w_min = 2.0 * (r_geo + 2.0 * lam / max(alpha, 1e-6) + 2.0 * sig_s)
    while n > cap and m > 1:
        m -= 1
        n, dk, fx, fy = grid(m)
    if m / dk1 < w_min:
        truncated = True
    if n > cap:
        crop = max(hd * fy, wd * fx)
        n2 = max(cap, crop + (crop & 1))
        truncated = truncated or n2 < n
        n = n2
    return Sampling((hd, wd), (bx, by), recip, int(n), dk, (fx, fy), (float(cx), float(cy)), ka,
                    2.0 * r_geo + 1.22 * lam / max(alpha, 1e-6), w_req, bool(truncated), kt)


# --------------------------------------------------------------------- probe
@dataclass
class Probe:
    sampling: Sampling
    modes: np.ndarray  # (M, n, n) complex64, real space, centred at (n//2, n//2), sum |.|^2 = 1 each
    work: np.ndarray  # modes x placement ramp x (-1)^(i+j) (so the FFT puts k = 0 at n/2)
    weights: np.ndarray  # (M,) float64, sum 1
    intensity: np.ndarray  # (n, n) float32 sum_m w_m |phi_m|^2
    crop_start: tuple[int, int]  # (i0x, i0y) signed frequency index of the first detector sub-pixel
    frac: tuple[float, float]  # (eps_x, eps_y) fractional k shift baked into the modes
    aberrations: object
    tilt_k: tuple[float, float]
    focal_samples: int
    source_samples: int
    coherent_fraction: float  # weight of the first mode before truncation


def _placement(s: Sampling, center_px) -> tuple[int, float, int, float]:
    """(i0x, eps_x, i0y, eps_y): the first detector sub-pixel sits at frequency index
    ``i0 + eps`` for an optic axis at ``center_px`` (unbinned output pixel coordinates)."""
    (fx, fy), (bx, by) = s.sub, s.binning
    cx, cy = center_px
    ux = (-0.5 - cx) * fx / bx + 0.5
    uy = (-0.5 - cy) * fy / by + 0.5
    ix, iy = math.floor(ux), math.floor(uy)
    return ix, ux - ix, iy, uy - iy


def _ramp(n: int, eps: float) -> np.ndarray:
    return np.exp(-2j * math.pi * eps * np.arange(n) / n).astype(np.complex64)


def build_probe(optics, cfg, s: Optional[Sampling] = None, *, center_px=None) -> Probe:
    s = s or plan_sampling(optics, cfg)
    lam = optics.wavelength_nm
    n, dk, ka = s.n, s.dk, s.k_alpha
    f = sfft.fftfreq(n, d=1.0 / (n * dk))
    KX, KY = f[None, :], f[:, None]
    kr = np.hypot(KX, KY)
    A = np.clip((ka - kr) / dk + 0.5, 0.0, 1.0)
    ab = probe_aberrations_of(optics)
    tilt = _tilt_k(optics, cfg)
    chi0 = probe_phase(ab, KX, KY, lam, tilt)
    k2t = (KX + tilt[0]) ** 2 + (KY + tilt[1]) ** 2 - (tilt[0] ** 2 + tilt[1] ** 2)

    # temporal coherence: Gauss-Hermite over the focal spread
    delta = float(optics.focal_spread_nm) if cfg.coherent_focal_spread else 0.0
    phase_edge = math.pi * lam * delta * ka * ka
    nf = int(cfg.coherent_focal_samples)
    if nf <= 0:
        nf = 1 if phase_edge < 0.15 else (3 if phase_edge < 2.5 else 5)
    fnodes, fweights = ((np.zeros(1), np.ones(1)) if nf == 1 else (_GH3 if nf == 3 else _GH5))
    fnodes = fnodes * delta
    # spatial coherence: Gauss-Hermite 3x3 over the effective source
    sig = float(optics.source_size_nm) / 2.3548 if cfg.coherent_source_size else 0.0
    use_src = sig > 0.02 * lam / max(ka * lam, 1e-9)  # sigma > 2 % of lambda/alpha
    snodes = [(0.0, 0.0, 1.0)]
    if use_src:
        x3, w3 = _GH3
        snodes = [(sig * a, sig * b, wa * wb) for a, wa in zip(x3, w3) for b, wb in zip(x3, w3)]

    samples = []
    wts = []
    for df, wf in zip(fnodes, fweights):
        base = A * np.exp(-1j * (chi0 + math.pi * lam * df * k2t))
        for sx, sy, ws in snodes:
            P = base if (sx == 0 and sy == 0) else base * np.exp(-2j * math.pi * (KX * sx + KY * sy))
            psi = sfft.fftshift(sfft.ifft2(P, norm="ortho", workers=-1))
            psi /= math.sqrt(float(np.vdot(psi, psi).real))
            samples.append(psi.astype(np.complex64).ravel())
            wts.append(wf * ws)
    wts = np.asarray(wts) / np.sum(wts)
    if len(samples) == 1:
        modes = samples[0].reshape(1, n, n)
        mw = np.ones(1)
        first = 1.0
    else:
        M = np.stack(samples) * np.sqrt(wts)[:, None].astype(np.float32)
        G = (M @ M.conj().T).astype(np.complex128)
        ev, V = np.linalg.eigh(G)
        order = np.argsort(ev)[::-1]
        ev, V = np.clip(ev[order], 0.0, None), V[:, order]
        frac = ev / ev.sum()
        first = float(frac[0])
        keep = int(np.searchsorted(np.cumsum(frac), cfg.coherent_mode_power) + 1)
        keep = max(1, min(keep, int(cfg.coherent_max_modes), len(ev)))
        # The symmetric source sampling gives degenerate eigenvalues (e.g. an x/y pair). Within
        # a degenerate group the eigenvectors are an arbitrary basis that LAPACK picks
        # differently with BLAS threading, so cutting through a group makes the patterns
        # machine-dependent. Keep whole groups: their summed intensity is basis-independent.
        while keep < len(ev) and ev[keep] > 0 and abs(ev[keep - 1] - ev[keep]) <= 1e-3 * ev[keep - 1]:
            keep += 1
        modes = (V[:, :keep].conj().T.astype(np.complex64) @ M) / np.sqrt(ev[:keep])[:, None].astype(np.float32)
        modes = modes.reshape(keep, n, n)
        mw = frac[:keep] / frac[:keep].sum()
    center = s.center_px if center_px is None else center_px
    ix, ex, iy, ey = _placement(s, center)
    cb = np.where((np.arange(n)[:, None] + np.arange(n)[None, :]) % 2 == 0, 1.0, -1.0)
    work = modes * (_ramp(n, ey)[:, None] * _ramp(n, ex)[None, :] * cb).astype(np.complex64)[None]
    inten = np.tensordot(mw, (modes.real ** 2 + modes.imag ** 2), axes=1).astype(np.float32)
    return Probe(s, modes.astype(np.complex64), work.astype(np.complex64), mw, inten, (ix, iy), (ex, ey),
                 ab, tilt, nf, len(snodes), first)


# ------------------------------------------------------------ specimen tile
@dataclass
class Tile:
    view: ViewWindow  # fine grid, rotation 0, lab-aligned (x = detector columns)
    t: np.ndarray  # (Ny, Nx) complex64 transmission
    bragg: Optional[np.ndarray]  # (Ny, Nx) float32 T x Bragg loss (None: no crystal in the tile)
    gid: np.ndarray  # (Ny, Nx) int64 own grain (-1 none)
    tbin: np.ndarray  # (Ny, Nx) int64 thickness bin
    mat: np.ndarray  # (Ny, Nx) uint8
    diffuse: dict  # material id -> (Ny, Nx) float32 diffuse weight (1 - T) exp(-t / 10 Lambda)
    cache: dict = dataclasses.field(default_factory=dict)  # probe-weighted add-on maps


def field_view(xs_um, ys_um, s: Sampling, like: ViewWindow, margin_px: int = FIELD_MARGIN_PX) -> ViewWindow:
    """A lab-aligned fine view (pixel dx) covering probe windows at the given world positions.
    Pixel centres sit on a world-anchored grid, so world-locked textures agree between tiles."""
    dx_um = s.dx_nm / 1000.0
    cb, ca = max(like.cos_beta, 0.1), max(like.cos_alpha, 0.1)
    px_x, px_y = dx_um / cb, dx_um / ca  # world pitch of a lab pixel
    half = s.n // 2 + margin_px
    c0 = int(math.floor(float(np.min(xs_um)) / px_x)) - half
    c1 = int(math.ceil(float(np.max(xs_um)) / px_x)) + half
    r0 = int(math.floor(float(np.min(ys_um)) / px_y)) - half
    r1 = int(math.ceil(float(np.max(ys_um)) / px_y)) + half
    nx, ny = c1 - c0 + 1, r1 - r0 + 1
    cx = (c0 - 0.5 + nx / 2.0) * px_x
    cy = (r0 - 0.5 + ny / 2.0) * px_y
    return ViewWindow(center_um=(cx, cy), pixel_um=dx_um, shape=(ny, nx), rotation_rad=0.0,
                      cos_alpha=like.cos_alpha, cos_beta=like.cos_beta)


def band_limit(t: np.ndarray, dx_nm: float, k_max: float) -> np.ndarray:
    """Low-pass a complex field to |k| <= k_max (soft 5 % edge), in place where possible."""
    ny, nx = t.shape
    if k_max <= 0 or k_max >= 0.5 / dx_nm * math.sqrt(2.0):
        return t
    F = sfft.fft2(t, workers=-1)
    ky = sfft.fftfreq(ny, dx_nm)[:, None].astype(np.float32)
    kx = sfft.fftfreq(nx, dx_nm)[None, :].astype(np.float32)
    k = np.sqrt(kx * kx + ky * ky)
    edge = np.float32(0.05 * k_max)
    F *= np.clip((np.float32(k_max) - k) / edge + np.float32(0.5), 0, 1).astype(np.float32)
    return sfft.ifft2(F, workers=-1, overwrite_x=True).astype(np.complex64)


def diffuse_beyond_fraction(mid: int, wavelength_nm: float, theta_c_mrad: float) -> float:
    """Screened-Rutherford fraction scattered beyond ``theta_c``: theta0^2 / (theta_c^2 + theta0^2)."""
    if mid == 0:
        return 0.0
    t0 = screening_angle_mrad(mid, wavelength_nm)
    return t0 * t0 / (theta_c_mrad * theta_c_mrad + t0 * t0)


def build_tile(fm, optics, grains, crystallinity, cfg, seed: int, bandwidth: float = 0.0) -> Tile:
    """Transmission function (band-limited to ``bandwidth``) plus the incoherent remainder:
    Bragg beams beyond ``bandwidth`` and the diffuse part beyond ``lambda bandwidth``."""
    stem_optics = dataclasses.replace(optics, view=fm.view, objective_aperture_mrad=0.0)
    kmax = bandwidth if (bandwidth > 0 and cfg.coherent_incoherent_scattering) else 0.0
    t, bc = transmission_function(fm, stem_optics, grains, crystallinity, cfg, seed, return_contrast=True,
                                  coherent_k_max=kmax)
    if bandwidth > 0:
        t = band_limit(t, fm.view.pixel_um * 1000.0, bandwidth)
    thick = fm.thickness_nm.astype(np.float32) * np.float32(optics.thickness_tilt_factor)
    mat = fm.material_id
    lam_px = absorption_lengths_nm(optics.ht_kv)[mat]
    T = np.exp(-thick / lam_px).astype(np.float32)
    dw = ((1.0 - T) * np.exp(-thick / (DIFFUSE_CUTOFF_LENGTHS * lam_px))).astype(np.float32)
    theta_c = 1000.0 * optics.wavelength_nm * kmax
    fb = np.array([diffuse_beyond_fraction(m, optics.wavelength_nm, theta_c) if kmax > 0 else (1.0 if m else 0.0)
                   for m in range(len(MATERIALS))], np.float32)
    kept = T + (dw * (1.0 - fb[mat]) if (kmax > 0 and cfg.diffuse_scattering) else 0.0)
    bragg = None
    if (bc.loss > 0).any():
        bragg = (kept * bc.loss).astype(np.float32)
    diffuse = {}
    if cfg.diffuse_scattering and cfg.coherent_incoherent_scattering:
        dwb = dw * fb[mat]
        for m in np.unique(mat):
            if m == 0:
                continue
            sel = mat == m
            if dwb[sel].max() > 0:
                diffuse[int(m)] = np.where(sel, dwb, np.float32(0.0))
    return Tile(fm.view, t, bragg if cfg.coherent_incoherent_scattering else None, bc.gid,
                thickness_bin_for(thick), mat, diffuse)


# -------------------------------------------------------------- the engine
class CoherentStem:
    """Coherent 4D-STEM engine; owned by :class:`~de_twin.render.renderer.Renderer`.

    ``field_map(view) -> (FieldMap, token)`` rasterises the specimen on a fine view (the
    renderer's cached rasteriser)."""

    def __init__(self, cache, cfg, seed: int):
        self.cache = cache  # DiffractionCache (Bragg / diffuse add-on patterns)
        self.cfg = cfg
        self.seed = seed
        self._probes: "OrderedDict[tuple, Probe]" = OrderedDict()
        self._samplings: "OrderedDict[tuple, Sampling]" = OrderedDict()
        self._tiles: "OrderedDict[tuple, Tile]" = OrderedDict()
        self._blocks: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._block_bytes = 0
        self.blocks_computed = 0
        self.points_computed = 0

    # ------------------------------------------------------------ keys
    def _cfg_key(self) -> tuple:
        c = self.cfg
        return (c.coherent_max_grid, c.coherent_probe_tail, c.coherent_mode_power, c.coherent_max_modes,
                c.coherent_focal_samples, c.coherent_focal_spread, c.coherent_source_size, c.coherent_beam_tilt,
                c.coherent_incoherent_scattering, c.diffuse_scattering, c.max_g_inv_nm, c.mip_phase,
                c.edge_taper_nm, c.refraction_loss, c.phase_texture_scale, c.texture_bandlimit_nm,
                c.amplitude_contrast, c.lattice_fringes, c.lattice_phase_rad, c.diffraction_contrast_scale,
                c.add_descan, c.descan_ramp_scale, tuple(c.descan_ramp_px), c.film_halo,
                c.coherent_object_bandwidth_inv_nm)

    def _optics_key(self, optics) -> tuple:
        return (optics, probe_aberrations_of(optics).key(), tuple(optics.extras.get("hw_binning", (1, 1))))

    def sampling(self, optics) -> Sampling:
        """The simulation sampling (cheap: no probe is built)."""
        key = (self._optics_key(optics), self._cfg_key())
        p = self._probes.get(key)
        if p is not None:
            return p.sampling
        hit = self._samplings.get(key)
        if hit is None:
            hit = plan_sampling(optics, self.cfg)
            self._samplings[key] = hit
            while len(self._samplings) > 8:
                self._samplings.popitem(last=False)
        return hit

    def probe(self, optics) -> Probe:
        key = (self._optics_key(optics), self._cfg_key())
        p = self._probes.get(key)
        if p is None:
            p = build_probe(optics, self.cfg, self.sampling(optics))
            self._probes[key] = p
            while len(self._probes) > 2:
                self._probes.popitem(last=False)
        else:
            self._probes.move_to_end(key)
        return p

    # ------------------------------------------------------------ tiles
    def _scan_world(self, optics, points) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points, np.int64).reshape(-1, 2)
        x, y = optics.view.pixel_to_world(pts[:, 1], pts[:, 0])
        return np.asarray(x, float), np.asarray(y, float)

    def _full_scan_points(self, optics) -> np.ndarray:
        ny, nx = optics.view.shape
        iy, ix = np.mgrid[0:ny, 0:nx]
        return np.stack([ix.ravel(), iy.ravel()], 1)

    def tile_for(self, ctx, optics, s: Sampling, points) -> Tile:
        """Transmission tile covering the probe windows of ``points``: the whole scan field when
        it fits ``coherent_field_max_px``, else just these points."""
        allx, ally = self._scan_world(optics, self._full_scan_points(optics))
        full = field_view(allx, ally, s, optics.view)
        if full.shape[0] * full.shape[1] <= self.cfg.coherent_field_max_px:
            view = full
        else:
            xs, ys = self._scan_world(optics, points)
            view = field_view(xs, ys, s, optics.view)
        fm, token = ctx.field_map(view)
        key = (token, view, optics.ht_kv, optics.thickness_tilt_factor, optics.alpha_rad, optics.beta_rad,
               optics.convergence_mrad, self.seed, self._cfg_key())
        tile = self._tiles.get(key)
        if tile is None:
            tile = build_tile(fm, optics, ctx.grains, ctx.crystallinity, self.cfg, self.seed,
                              s.object_bandwidth)
            self._tiles[key] = tile
            while len(self._tiles) > 2:
                self._tiles.popitem(last=False)
        else:
            self._tiles.move_to_end(key)
        return tile

    # ------------------------------------------------------------ descan
    def _descan_px(self, optics, fm, ix: int, iy: int) -> tuple[float, float]:
        cfg = self.cfg
        dx = dy = 0.0
        ny, nx = optics.view.shape
        if cfg.add_descan and nx > 0 and ny > 0:
            xs, ys, dg = cfg.descan_ramp_px
            dx += cfg.descan_ramp_scale * (2.0 * (ix - nx / 2.0) / nx * xs + xs + iy * dg / ny)
            dy += cfg.descan_ramp_scale * (2.0 * (iy - ny / 2.0) / ny * ys + ys)
        if fm is not None and fm.descan is not None:
            dx += float(fm.descan[0, iy, ix])
            dy += float(fm.descan[1, iy, ix])
        return dx, dy

    # ------------------------------------------------------------ core
    def _weights(self, tile: Tile, probe: Probe) -> dict:
        """Probe-intensity-weighted Bragg / diffuse weights for every window position of the
        tile (valid correlation: element (r0, c0) is the window starting there)."""
        from scipy.signal import correlate

        key = id(probe)
        hit = tile.cache.get(key)
        if hit is not None and hit[0] is probe:
            return hit[1]
        out = {}
        if tile.bragg is not None:
            out["bragg"] = correlate(tile.bragg, probe.intensity, mode="valid", method="fft")
        for m, dw in tile.diffuse.items():
            out[m] = correlate(dw, probe.intensity, mode="valid", method="fft")
        tile.cache.clear()
        tile.cache[key] = (probe, out)
        return out

    def _compute(self, ctx, optics, probe: Probe, tile: Tile, points, fm_scan=None, *,
                 annuli_mrad=None, want_patterns=True) -> tuple:
        s = probe.sampling
        n, h2 = s.n, s.n // 2
        (hd, wd), (fx, fy) = s.det_shape, s.sub
        H, W = hd * fy, wd * fx
        xs, ys = self._scan_world(optics, points)
        rows, cols = tile.view.world_to_pixel(xs, ys)
        rows = np.rint(rows).astype(np.int64)
        cols = np.rint(cols).astype(np.int64)
        if (rows.min() - h2 < 0 or cols.min() - h2 < 0 or rows.max() + h2 > tile.t.shape[0]
                or cols.max() + h2 > tile.t.shape[1]):
            raise RuntimeError("coherent STEM: probe window outside the transmission tile")
        pts = np.asarray(points, np.int64).reshape(-1, 2)
        npts = len(pts)
        shifts = [self._descan_px(optics, fm_scan, int(ix), int(iy)) for ix, iy in pts]
        # crop start (array index of the first detector sub-pixel; k = 0 sits at n/2) per point
        place = []
        for sx, sy in shifts:
            ix, ex, iy, ey = _placement(s, (s.center_px[0] + sx, s.center_px[1] + sy))
            place.append((ix + h2, ex - probe.frac[0], iy + h2, ey - probe.frac[1]))
        place = np.array(place)
        any_ramp = bool(np.abs(place[:, [1, 3]]).max() > 1e-9)
        amask = None
        if annuli_mrad:
            f = (np.arange(n) - h2) * s.dk * (1000.0 * optics.wavelength_nm)
            kk = np.hypot(f[None, :], f[:, None])
            amask = [((kk >= a) & (kk < b)).astype(np.float32) for a, b in annuli_mrad]
        windows = np.lib.stride_tricks.sliding_window_view(tile.t, (n, n))
        work, mw = probe.work, probe.weights

        pats = np.zeros((npts, hd, wd), np.float32) if want_patterns else None
        anns = np.zeros((npts, len(amask)), np.float64) if amask is not None else None

        def crop_into(acc, I, k, x0, y0, w):
            # add w * I[y0:y0+H, x0:x0+W] into acc[k] (zero outside the simulated k range)
            ya, yb = max(y0, 0), min(y0 + H, n)
            xa, xb = max(x0, 0), min(x0 + W, n)
            if ya < yb and xa < xb:
                acc[k, ya - y0:yb - y0, xa - x0:xb - x0] += w * I[ya:yb, xa:xb]

        def sq(G):
            v = G.view(np.float32)
            v = v.reshape(v.shape[:-1] + (v.shape[-1] // 2, 2))
            return np.einsum("...i,...i->...", v, v)

        def run(lo, hi):
            B = hi - lo
            rr, cc = rows[lo:hi] - h2, cols[lo:hi] - h2
            ramp = None
            if any_ramp:
                ar = np.arange(n)
                ry = np.exp(-2j * np.pi * place[lo:hi, 3, None] * ar / n).astype(np.complex64)
                rx = np.exp(-2j * np.pi * place[lo:hi, 1, None] * ar / n).astype(np.complex64)
                ramp = ry[:, :, None] * rx[:, None, :]
            acc = np.zeros((B, H, W), np.float32) if want_patterns else None
            full = np.zeros((B, n, n), np.float32) if amask is not None else None
            x0s = place[lo:hi, 0].astype(int)
            y0s = place[lo:hi, 2].astype(int)
            simple = (want_patterns and (x0s == x0s[0]).all() and (y0s == y0s[0]).all()
                      and y0s[0] >= 0 and x0s[0] >= 0 and y0s[0] + H <= n and x0s[0] + W <= n)
            e = np.empty((B, n, n), np.complex64)
            for mode, w in zip(work, mw):
                for k in range(B):
                    np.multiply(windows[rr[k], cc[k]], mode, out=e[k])
                if ramp is not None:
                    e *= ramp
                F = sfft.fft2(e, norm="ortho", workers=1, overwrite_x=True)
                wf = np.float32(w)
                if simple:
                    I = sq(F[:, y0s[0]:y0s[0] + H, x0s[0]:x0s[0] + W])
                    if len(mw) == 1:
                        acc = I
                    else:
                        acc += wf * I
                    if full is not None:
                        full += wf * sq(F)
                    continue
                I = sq(F)
                if full is not None:
                    full += wf * I
                if want_patterns:
                    for k in range(B):
                        crop_into(acc, I[k], k, x0s[k], y0s[k], wf)
            if want_patterns:
                if fx == 1 and fy == 1:
                    pats[lo:hi] = acc
                else:
                    pats[lo:hi] = acc.reshape(B, hd, fy, wd, fx).sum(axis=(2, 4), dtype=np.float32)
            if full is not None:
                for j, m in enumerate(amask):
                    anns[lo:hi, j] = np.tensordot(full, m, axes=([1, 2], [0, 1]))

        nw = pool()._max_workers
        bs = int(max(1, min(64, (24 << 20) // (n * n * 8), -(-npts // (2 * nw)))))
        edges = list(range(0, npts, bs)) + [npts]
        jobs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        if len(jobs) == 1:
            run(*jobs[0])
        else:
            list(pool().map(lambda j: run(*j), jobs))

        # incoherent add-on weights (probe-intensity weighted over the window)
        wts = self._weights(tile, probe)
        r0, c0 = rows - h2, cols - h2
        bragg_w = np.zeros(npts)
        keys: list = [None] * npts
        if "bragg" in wts:
            bragg_w = np.maximum(wts["bragg"][r0, c0], 0.0).astype(np.float64)
            for k in np.flatnonzero(bragg_w > 1e-12):
                r, c = int(rows[k]), int(cols[k])
                g = int(tile.gid[r, c])
                if g < 0:  # probe centre off the grain: the grain dominating the window
                    prod = probe.intensity * tile.bragg[r - h2:r + h2, c - h2:c + h2]
                    jr, jc = divmod(int(np.argmax(prod)), n)
                    r, c = r - h2 + jr, c - h2 + jc
                    g = int(tile.gid[r, c])
                if g >= 0:
                    keys[k] = (g // GRAINS_PER_MATERIAL, g, int(tile.tbin[r, c]))
                else:
                    bragg_w[k] = 0.0
        diffuse = {m: np.maximum(v[r0, c0], 0.0).astype(np.float64) for m, v in wts.items() if m != "bragg"}
        self.points_computed += npts
        return pats, anns, bragg_w, keys, diffuse, shifts

    # ------------------------------------------------------ add-on patterns
    def _bragg_pattern(self, ctx, optics, s: Sampling, key, center_b) -> Optional[np.ndarray]:
        hd, wd = s.det_shape
        rb = s.det_recip_px[0]
        ck = ("coh-bragg", key, (hd, wd), rb, round(center_b[0], 3), round(center_b[1], 3), s.object_bandwidth,
              optics.convergence_mrad, optics.alpha_rad, optics.beta_rad, optics.ht_kv, self.cfg.max_g_inv_nm,
              _tilt_key(optics))

        def render(_k):
            mid, gid, tb = key
            ps = bucket_patterns([mid], [gid], [tb * THICKNESS_BIN_NM], [1.0], ctx.grains, optics,
                                 PatternOptions(film_halo=False, max_g_inv_nm=self.cfg.max_g_inv_nm))
            sp = ps[0].spots
            sp = sp[np.hypot(sp[:, 0], sp[:, 1]) > max(1e-9, s.object_bandwidth)]
            tot = float(sp[:, 2].sum()) if len(sp) else 0.0
            if tot <= 0:
                return np.zeros((hd, wd), np.float32)
            sp = sp.copy()
            sp[:, 2] /= tot
            disk = optics.convergence_mrad / 1000.0 / optics.wavelength_nm / rb
            return render_pattern(Pattern(sp), (hd, wd), rb, disk, center=center_b)
        return self.cache.get(ck, render)

    def diffuse_cut_mrad(self, optics) -> float:
        """Inner angle of the incoherent diffuse background in coherent mode: the scattering
        angle ``lambda K_t`` beyond which the band-limited transmission function has none."""
        if not self.cfg.coherent_incoherent_scattering:
            return 0.0
        return 1000.0 * optics.wavelength_nm * self.sampling(optics).object_bandwidth

    def _diffuse_pattern(self, optics, s: Sampling, mid: int, center_b) -> np.ndarray:
        """Screened-Rutherford background beyond ``diffuse_cut_mrad`` (plane integral 1): the
        mass-thickness electrons are scattered to large angles; inside the bright-field cone the
        coherent wave already carries the specimen's (elastic) scattering."""
        hd, wd = s.det_shape
        rb = s.det_recip_px[0]
        cut = self.diffuse_cut_mrad(optics)
        ck = ("coh-diffuse", mid, (hd, wd), rb, round(center_b[0], 3), round(center_b[1], 3), optics.ht_kv,
              round(cut, 6))

        def render(_k):
            from scipy.special import erfc

            mrad_px = rb * 1000.0 * optics.wavelength_nm
            t0 = screening_angle_mrad(mid, optics.wavelength_nm)
            y = (np.arange(hd, dtype=np.float32) - np.float32(center_b[1]))[:, None]
            x = (np.arange(wd, dtype=np.float32) - np.float32(center_b[0]))[None, :]
            r = np.sqrt(x * x + y * y) * np.float32(mrad_px)
            p = np.float32(t0 * t0 / math.pi * mrad_px * mrad_px) / (r * r + np.float32(t0 * t0)) ** 2
            if cut > 0:
                p = p * (0.5 * erfc((cut - r) / (math.sqrt(2.0) * mrad_px))).astype(np.float32)
                p /= np.float32(t0 * t0 / (cut * cut + t0 * t0))
            return p.astype(np.float32)
        return self.cache.get(ck, render)

    # ------------------------------------------------------------- blocks
    def _block_size(self, optics, s: Sampling) -> int:
        ny, nx = optics.view.shape
        total = max(1, ny * nx)
        per = s.det_shape[0] * s.det_shape[1] * 4
        rows = max(1, int(math.ceil(1024 / max(nx, 1))))  # whole rows, >= 1024 points (all threads busy)
        b = min(total, rows * nx)
        b = min(b, max(1, (128 << 20) // per))
        return int(b)

    def patterns(self, ctx, optics, points, fm_scan=None) -> np.ndarray:
        """(P, Hd, Wd) float32 binned detector patterns (electrons / binned pixel / s)."""
        probe = self.probe(optics)
        s = probe.sampling
        tile = self.tile_for(ctx, optics, s, points)
        out, _, bragg_w, keys, diffuse, shifts = self._compute(ctx, optics, probe, tile, points, fm_scan)
        cb = s.det_center_px
        bx, by = s.binning
        for k in range(len(out)):
            add = None
            if keys[k] is not None and bragg_w[k] > 0:
                add = np.float32(bragg_w[k]) * self._bragg_pattern(ctx, optics, s, keys[k], cb)
            for m, wm in diffuse.items():
                if wm[k] > 0:
                    d = np.float32(wm[k]) * self._diffuse_pattern(optics, s, m, cb)
                    add = d if add is None else add + d
            if add is not None:
                sx, sy = shifts[k]
                if sx or sy:
                    add = shift_bilinear(add, sx / bx, sy / by)
                out[k] += add
        out *= np.float32(optics.pattern_e_per_s)
        return out

    def _block(self, ctx, optics, fm_scan, fm_token, index: int) -> tuple[np.ndarray, int]:
        s = self.sampling(optics)
        ny, nx = optics.view.shape
        bsz = self._block_size(optics, s)
        b = index // bsz
        key = (self._optics_key(optics), self._cfg_key(), fm_token, bsz, b)
        blk = self._blocks.get(key)
        if blk is None:
            idx = np.arange(b * bsz, min(ny * nx, (b + 1) * bsz))
            pts = np.stack([idx % nx, idx // nx], 1)
            blk = self.patterns(ctx, optics, pts, fm_scan)
            blk.setflags(write=False)
            self.blocks_computed += 1
            self._blocks[key] = blk
            self._block_bytes += blk.nbytes
            budget = int(self.cfg.coherent_cache_mb) << 20
            while self._block_bytes > budget and len(self._blocks) > 1:
                _, v = self._blocks.popitem(last=False)
                self._block_bytes -= v.nbytes
        else:
            self._blocks.move_to_end(key)
        return blk, index - b * bsz

    def binned(self, ctx, fm_scan, fm_token, optics, scan_point, frame_index, specimen=None) -> np.ndarray:
        ix, iy = resolve_scan_point(optics, scan_point, frame_index, specimen, self.cfg)
        nx = optics.view.shape[1]
        blk, j = self._block(ctx, optics, fm_scan, fm_token, iy * nx + ix)
        return blk[j]

    def render(self, ctx, fm_scan, fm_token, optics, scan_point, frame_index, specimen=None) -> np.ndarray:
        """float32 (h, w) = optics.output_shape, electrons / unbinned pixel / s."""
        pat = self.binned(ctx, fm_scan, fm_token, optics, scan_point, frame_index, specimen)
        return expand_binned(pat, optics)

    def datacube(self, ctx, optics, fm_scan=None, fm_token=None) -> np.ndarray:
        """(ny, nx, Hd, Wd) float32: every scan point's binned pattern (electrons / pixel / s)."""
        ny, nx = optics.view.shape
        s = self.sampling(optics)
        out = np.empty((ny * nx,) + s.det_shape, np.float32)
        i = 0
        while i < ny * nx:
            blk, j = self._block(ctx, optics, fm_scan, fm_token, i)
            k = min(len(blk) - j, ny * nx - i)
            out[i:i + k] = blk[j:j + k]
            i += k
        return out.reshape(ny, nx, *s.det_shape)

    # ------------------------------------------------------ virtual images
    def virtual_image(self, ctx, optics, inner_mrad: float, outer_mrad: float, fm_scan=None) -> np.ndarray:
        """Annular detector image over the scan: the coherent part integrated on the full
        simulation grid, the incoherent Bragg / diffuse parts analytically (not truncated by
        the camera, like the kinematic model)."""
        probe = self.probe(optics)
        s = probe.sampling
        ny, nx = optics.view.shape
        pts = self._full_scan_points(optics)
        out = np.zeros(ny * nx, np.float64)
        lam = optics.wavelength_nm
        bsz = max(1, self._block_size(optics, s))
        cut = self.diffuse_cut_mrad(optics)
        dfrac = {}
        for m in range(len(MATERIALS)):
            t0 = screening_angle_mrad(m, lam) if m else 1.0
            beyond = t0 * t0 / (cut * cut + t0 * t0)
            dfrac[m] = (diffuse_annulus_fraction(m, lam, max(inner_mrad, cut), max(outer_mrad, cut)) / beyond
                        if m and beyond > 0 else 0.0)
        bfrac: dict = {}
        for b0 in range(0, len(pts), bsz):
            chunk = pts[b0:b0 + bsz]
            tile = self.tile_for(ctx, optics, s, chunk)
            _, anns, bragg_w, keys, diffuse, _ = self._compute(
                ctx, optics, probe, tile, chunk, fm_scan, annuli_mrad=[(inner_mrad, outer_mrad)],
                want_patterns=False)
            v = anns[:, 0].copy()
            for k, key in enumerate(keys):
                if key is None or bragg_w[k] <= 0:
                    continue
                if key not in bfrac:
                    mid, gid, tb = key
                    ps = bucket_patterns([mid], [gid], [tb * THICKNESS_BIN_NM], [1.0], ctx.grains, optics,
                                         PatternOptions(film_halo=False, max_g_inv_nm=self.cfg.max_g_inv_nm))
                    sp = ps[0].spots
                    sp = sp[np.hypot(sp[:, 0], sp[:, 1]) > max(1e-9, s.object_bandwidth)]
                    tot = float(sp[:, 2].sum()) if len(sp) else 0.0
                    bfrac[key] = 0.0
                    if tot > 0:
                        sp = sp.copy()
                        sp[:, 2] /= tot
                        one = PatternSet(1, sp, np.zeros(len(sp), np.int64), np.zeros((0, 3)),
                                         np.zeros(0, np.int64))
                        bfrac[key] = float(annulus_fractions(one, optics.convergence_mrad, lam,
                                                             inner_mrad, outer_mrad)[0])
                v[k] += bragg_w[k] * bfrac[key]
            for m, wm in diffuse.items():
                v += wm * dfrac[m]
            out[b0:b0 + len(chunk)] = v
        return (out.reshape(ny, nx) * optics.pattern_e_per_s).astype(np.float32)

    # ---------------------------------------------------------- ground truth
    def ground_truth(self, ctx, optics) -> dict:
        probe = self.probe(optics)
        s = probe.sampling
        pts = self._full_scan_points(optics)
        allx, ally = self._scan_world(optics, pts)
        view = field_view(allx, ally, s, optics.view)
        fm, token = ctx.field_map(view)
        key = (token, view, optics.ht_kv, optics.thickness_tilt_factor, optics.alpha_rad, optics.beta_rad,
               optics.convergence_mrad, self.seed, self._cfg_key())
        tile = self._tiles.get(key)
        if tile is None:
            tile = build_tile(fm, optics, ctx.grains, ctx.crystallinity, self.cfg, self.seed,
                              s.object_bandwidth)
        rows, cols = tile.view.world_to_pixel(allx, ally)
        ny, nx = optics.view.shape
        pos = np.stack([np.rint(rows), np.rint(cols)], 1).astype(np.float64)  # snapped: exactly as simulated
        ab = probe.aberrations
        modes = probe.modes
        return {
            "object": tile.t, "dx_nm": s.dx_nm, "view": tile.view,
            "positions_px": pos, "positions_nm": pos * s.dx_nm, "scan_shape": (ny, nx),
            "probe": modes[0], "probe_modes": modes, "mode_weights": probe.weights,
            "coherent_fraction": probe.coherent_fraction,
            "wavelength_nm": optics.wavelength_nm, "convergence_mrad": optics.convergence_mrad,
            "alpha_mrad": optics.convergence_mrad,
            "aberrations": ab, "aberrations_polar": ab.polar(),
            "C1_nm": float(ab["C1"].real), "A1_nm": complex(ab["A1"]), "C3_nm": float(ab["C3"].real),
            "B2_nm": complex(ab["B2"]), "C5_nm": float(ab["C5"].real),
            "defocus_nm": float(optics.defocus_um * 1000.0),
            "beam_tilt_mrad": tuple(optics.beam_tilt_mrad), "focal_spread_nm": float(optics.focal_spread_nm),
            "source_size_nm": float(optics.source_size_nm),
            "scan_step_nm": optics.scan_step_um * 1000.0, "scan_rotation_rad": optics.scan_rotation_rad,
            "sampling": s,
            "detector_shape": s.det_shape, "detector_recip_px_inv_nm": s.det_recip_px,
            "detector_center_px": s.det_center_px,
        }


def expand_binned(pat: np.ndarray, optics) -> np.ndarray:
    """Binned (Hd, Wd) pattern -> unbinned output (h, w), each sensor pixel an equal share."""
    h, w = (int(v) for v in optics.output_shape)
    bx, by = (max(1, int(v)) for v in optics.extras.get("hw_binning", (1, 1)))
    if bx == 1 and by == 1 and pat.shape == (h, w):
        return np.array(pat, np.float32, copy=True)
    up = np.repeat(np.repeat(pat, by, axis=0), bx, axis=1) * np.float32(1.0 / (bx * by))
    out = np.zeros((h, w), np.float32)
    hh, ww = min(h, up.shape[0]), min(w, up.shape[1])
    out[:hh, :ww] = up[:hh, :ww]
    return out


def resample_object(obj: np.ndarray, dx_nm: float, new_dx_nm: float) -> np.ndarray:
    """Fourier-resample a complex object to a coarser pixel (band-limited; same field of view)."""
    ny, nx = obj.shape
    my, mx = int(round(ny * dx_nm / new_dx_nm)), int(round(nx * dx_nm / new_dx_nm))
    F = sfft.fftshift(sfft.fft2(obj, workers=-1))
    cy, cx = ny // 2, nx // 2
    G = F[cy - my // 2:cy - my // 2 + my, cx - mx // 2:cx - mx // 2 + mx]
    return (sfft.ifft2(sfft.ifftshift(G), workers=-1) * (my * mx) / (ny * nx)).astype(np.complex64)


def choose_model(optics, cfg, sampling_fn: Callable[[], Sampling]) -> str:
    """``RenderConfig.stem_model`` resolved for these optics ("coherent" | "kinematic")."""
    m = str(cfg.stem_model).lower()
    if m in ("coherent", "kinematic"):
        return m
    try:
        s = sampling_fn()
    except Exception:  # noqa: BLE001 - degenerate geometry: fall back to the robust model
        return "kinematic"
    if s.truncated or s.n > cfg.coherent_max_grid:
        return "kinematic"
    hd, wd = s.det_shape
    if hd * wd <= cfg.coherent_auto_max_pattern_px:
        return "coherent"
    # larger patterns: when the scan oversamples the probe (ptychography-style overlap)
    if 0 < optics.scan_step_um * 1000.0 < 2.0 * s.probe_size_nm:
        return "coherent"
    return "kinematic"
