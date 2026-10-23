"""The coherent 4D-STEM simulation's setup: sampling grid, probe wave and specimen transmission tile."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft

from ..optics.state import probe_aberrations_of
from ..specimen.fieldmap import ViewWindow
from ..specimen.materials import MATERIALS, absorption_lengths_nm
from . import stem_nb as _snb
from .diffraction import screening_angle_mrad, thickness_bin_for
from .samples import DIFFUSE_CUTOFF_LENGTHS
from .tem import transmission_function
from .util import chi_and_gradient

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
        return round(self.sub[0] * math.gcd(*self.binning) / self.binning[0])

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


def _tilt_k(optics, config) -> tuple[float, float]:
    if not getattr(config, "coherent_beam_tilt", True):
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


def plan_sampling(optics, config) -> Sampling:
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
    tilt = _tilt_k(optics, config)
    r_geo = geometric_probe_radius_nm(ab, lam, ka, tilt)
    sig_s = optics.source_size_nm / 2.3548
    w_req = 2.0 * (1.1 * r_geo + config.coherent_probe_tail * lam / max(alpha, 1e-6) + 3.0 * sig_s)
    dk1 = recip * g
    m = max(1, math.ceil(w_req * dk1 - 1e-9))

    # no wrap-around onto the detector: exit-wave frequencies reach ka + K_t (K_t: the object's
    # band limit, enforced on the transmission tile), aliased ones land at k - 2 k_nyq
    kt = float(config.coherent_object_bandwidth_inv_nm)
    kt = kd + ka if kt <= 0 else min(kt, kd + ka)
    k_nyq = max(kd, 0.5 * (kd + ka + kt))

    def grid(m):
        dk = dk1 / m
        fx, fy = m * bx // g, m * by // g
        n = math.ceil(2.0 * k_nyq / dk) + 2
        n = max(n, hd * fy, wd * fx)
        n = sfft.next_fast_len(n + (n & 1))
        while n & 1:
            n = sfft.next_fast_len(n + 1)
        return n, dk, fx, fy

    n, dk, fx, fy = grid(m)
    truncated = False
    cap = int(config.coherent_max_grid)
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


def build_probe(optics, config, s: Sampling | None = None, *, center_px=None) -> Probe:
    s = s or plan_sampling(optics, config)
    lam = optics.wavelength_nm
    n, dk, ka = s.n, s.dk, s.k_alpha
    f = sfft.fftfreq(n, d=1.0 / (n * dk))
    # The probe lives on the aperture: every sample (focal / source node) is evaluated on the
    # k pixels with A > 0 only, the mode decomposition is done there (the FFT is unitary, so
    # the Gram matrix of the k-space samples is that of the real-space ones), and only the
    # kept modes are transformed to real space.
    iy, ix = np.nonzero(np.hypot(f[None, :], f[:, None]) < ka + 0.5 * dk)
    KX, KY = f[ix], f[iy]
    A = np.clip((ka - np.hypot(KX, KY)) / dk + 0.5, 0.0, 1.0)
    sel = A > 0
    iy, ix, KX, KY, A = iy[sel], ix[sel], KX[sel], KY[sel], A[sel]
    ab = probe_aberrations_of(optics)
    tilt = _tilt_k(optics, config)
    chi0 = probe_phase(ab, KX, KY, lam, tilt)
    k2t = (KX + tilt[0]) ** 2 + (KY + tilt[1]) ** 2 - (tilt[0] ** 2 + tilt[1] ** 2)

    # temporal coherence: Gauss-Hermite over the focal spread
    delta = float(optics.focal_spread_nm) if config.coherent_focal_spread else 0.0
    phase_edge = math.pi * lam * delta * ka * ka
    nf = int(config.coherent_focal_samples)
    if nf <= 0:
        nf = 1 if phase_edge < 0.15 else (3 if phase_edge < 2.5 else 5)
    fnodes, fweights = ((np.zeros(1), np.ones(1)) if nf == 1 else (_GH3 if nf == 3 else _GH5))
    fnodes = fnodes * delta
    # spatial coherence: Gauss-Hermite 3x3 over the effective source
    sig = float(optics.source_size_nm) / 2.3548 if config.coherent_source_size else 0.0
    use_src = sig > 0.02 * lam / max(ka * lam, 1e-9)  # sigma > 2 % of lambda/alpha
    snodes = [(0.0, 0.0, 1.0)]
    if use_src:
        x3, w3 = _GH3
        snodes = [(sig * a, sig * b, wa * wb) for a, wa in zip(x3, w3) for b, wb in zip(x3, w3)]

    if _snb.AVAILABLE:
        sn = np.asarray(snodes, np.float64)
        samples = np.empty((len(fnodes) * len(sn), len(A)), np.complex64)
        wts = np.empty(len(samples))
        _snb.probe_samples(np.ascontiguousarray(chi0, np.float64), k2t, KX, KY, A,
                           np.asarray(fnodes, np.float64), np.asarray(fweights, np.float64),
                           np.ascontiguousarray(sn[:, 0]), np.ascontiguousarray(sn[:, 1]),
                           np.ascontiguousarray(sn[:, 2]), math.pi * lam, samples, wts)
        samples = list(samples)
    else:
        samples = []
        wts = []
        for df, wf in zip(fnodes, fweights):
            base = A * np.exp(-1j * (chi0 + math.pi * lam * df * k2t))
            for sx, sy, ws in snodes:
                P = base if (sx == 0 and sy == 0) else base * np.exp(-2j * math.pi * (KX * sx + KY * sy))
                P = P / math.sqrt(float(np.vdot(P, P).real))  # Parseval: the real-space norm
                samples.append(P.astype(np.complex64))
                wts.append(wf * ws)
    wts = np.asarray(wts) / np.sum(wts)
    # real space, centred at (n/2, n/2): fftshift of the ifft = ifft of (-1)^(i+j) x the k grid
    # (n even); ortho norm
    ksign = np.where((iy + ix) % 2 == 0, 1.0, -1.0).astype(np.float32)

    def to_real(Pk):
        full = np.zeros((len(Pk), n, n), np.complex64)
        full[:, iy, ix] = Pk * ksign
        return sfft.ifft2(full, norm="ortho", workers=-1, overwrite_x=True).astype(np.complex64, copy=False)

    if len(samples) == 1:
        modes = to_real(samples[0][None])
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
        keep = int(np.searchsorted(np.cumsum(frac), config.coherent_mode_power) + 1)
        keep = max(1, min(keep, int(config.coherent_max_modes), len(ev)))
        # The symmetric source sampling gives degenerate eigenvalues (e.g. an x/y pair). Within
        # a degenerate group the eigenvectors are an arbitrary basis that LAPACK picks
        # differently with BLAS threading, so cutting through a group makes the patterns
        # machine-dependent. Keep whole groups: their summed intensity is basis-independent.
        while keep < len(ev) and ev[keep] > 0 and abs(ev[keep - 1] - ev[keep]) <= 1e-3 * ev[keep - 1]:
            keep += 1
        # (a null mode, possible when every mode is kept, is 0 rather than 0 / 0)
        norm = np.sqrt(np.where(ev[:keep] > 0, ev[:keep], np.inf))[:, None].astype(np.float32)
        modes = to_real((V[:, :keep].conj().T.astype(np.complex64) @ M) / norm)
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
    bragg: np.ndarray | None  # (Ny, Nx) float32 T x Bragg loss (None: no crystal in the tile)
    gid: np.ndarray  # (Ny, Nx) int32 own grain (-1 none)
    tbin: np.ndarray  # (Ny, Nx) int16 thickness bin
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
    c0 = math.floor(float(np.min(xs_um)) / px_x) - half
    c1 = math.ceil(float(np.max(xs_um)) / px_x) + half
    r0 = math.floor(float(np.min(ys_um)) / px_y) - half
    r1 = math.ceil(float(np.max(ys_um)) / px_y) + half
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


def build_tile(fm, optics, grains, crystallinity, config, seed: int, bandwidth: float = 0.0) -> Tile:
    """Transmission function (band-limited to ``bandwidth``) plus the incoherent remainder:
    Bragg beams beyond ``bandwidth`` and the diffuse part beyond ``lambda bandwidth``."""
    stem_optics = dataclasses.replace(optics, view=fm.view, objective_aperture_mrad=0.0)
    kmax = bandwidth if (bandwidth > 0 and config.coherent_incoherent_scattering) else 0.0
    t, bc = transmission_function(fm, stem_optics, grains, crystallinity, config, seed, return_contrast=True,
                                  coherent_k_max=kmax)
    if bandwidth > 0:
        t = band_limit(t, fm.view.pixel_um * 1000.0, bandwidth)
    mat = fm.material_id
    lam_all = absorption_lengths_nm(optics.ht_kv)
    theta_c = 1000.0 * optics.wavelength_nm * kmax
    fb = np.array([diffuse_beyond_fraction(m, optics.wavelength_nm, theta_c) if kmax > 0 else (1.0 if m else 0.0)
                   for m in range(len(MATERIALS))], np.float32)
    if _snb.AVAILABLE:
        return _tile_fast(fm, optics, config, t, bc, mat, lam_all, fb, kmax)
    thick = fm.thickness_nm.astype(np.float32) * np.float32(optics.thickness_tilt_factor)
    lam_px = lam_all[mat]
    T = np.exp(-thick / lam_px).astype(np.float32)
    dw = ((1.0 - T) * np.exp(-thick / (DIFFUSE_CUTOFF_LENGTHS * lam_px))).astype(np.float32)
    under = None
    if fm.under_thickness_nm is not None:  # absorbed with the primary; its diffuse part added below
        umat = fm.under_material
        tu = fm.under_thickness_nm.astype(np.float32) * np.float32(optics.thickness_tilt_factor)
        Tu = np.exp(-tu / lam_all[umat]).astype(np.float32)
        under = (umat, ((1.0 - Tu) * np.exp(-tu / (DIFFUSE_CUTOFF_LENGTHS * lam_all[umat]))).astype(np.float32))
        T = T * Tu
        dw = dw * Tu
    kept = T + (dw * (1.0 - fb[mat]) if (kmax > 0 and config.diffuse_scattering) else 0.0)
    bragg = None
    if (bc.loss > 0).any():
        bragg = (kept * bc.loss).astype(np.float32)
    diffuse = {}
    if config.diffuse_scattering and config.coherent_incoherent_scattering:
        dwb = dw * fb[mat]
        for m in np.unique(mat):
            if m == 0:
                continue
            sel = mat == m
            if dwb[sel].max() > 0:
                diffuse[int(m)] = np.where(sel, dwb, np.float32(0.0))
        if under is not None:
            umat, uw = under
            for m in np.unique(umat):
                if m == 0:
                    continue
                add = np.where(umat == m, uw * fb[m], np.float32(0.0))
                diffuse[int(m)] = diffuse[int(m)] + add if int(m) in diffuse else add
    return Tile(fm.view, t, bragg if config.coherent_incoherent_scattering else None, bc.gid.astype(np.int32),
                thickness_bin_for(thick).astype(np.int16), mat, diffuse)


def _tile_fast(fm, optics, config, t, bc, mat, lam_all, fb, kmax) -> Tile:
    """`build_tile`'s add-on maps in one numba pass (:func:`de_twin.render.stem_nb.tile_maps`)."""
    ny, nx = mat.shape
    has_under = fm.under_thickness_nm is not None
    umat = fm.under_material if has_under else mat
    uthick = fm.under_thickness_nm if has_under else fm.thickness_nm
    diffuse_on = bool(config.diffuse_scattering and config.coherent_incoherent_scattering)
    has_loss = bool(config.coherent_incoherent_scattering and (bc.loss > 0).any())
    present = np.bincount(mat.ravel(), minlength=256)
    upresent = np.bincount(umat.ravel(), minlength=256) if has_under else np.zeros(256, np.int64)
    mids = [m for m in range(1, 256) if present[m] or upresent[m]] if diffuse_on else []
    slot = np.zeros(256, np.int64)
    for k, m in enumerate(mids):
        slot[m] = k
    dmaps = np.zeros((max(1, len(mids)), ny, nx) if diffuse_on else (1, 1, 1), np.float32)
    dmax = np.zeros((ny, max(1, len(mids))) if diffuse_on else (1, 1), np.float32)
    bragg = np.empty((ny, nx), np.float32) if has_loss else np.zeros((1, 1), np.float32)
    tbin = np.empty((ny, nx), np.int16)
    _snb.tile_maps(mat, fm.thickness_nm, np.float32(optics.thickness_tilt_factor), np.asarray(lam_all, np.float64),
                   has_under, umat, uthick, float(DIFFUSE_CUTOFF_LENGTHS), fb,
                   bool(kmax > 0 and config.diffuse_scattering), bc.loss if has_loss else bragg, has_loss,
                   diffuse_on, slot, bragg, dmaps, dmax, tbin)
    diffuse = {}
    if diffuse_on:
        mx = dmax.max(axis=0)
        # the reference's order: the primary layer's materials (with a diffuse part), then the
        # under layer's others
        for m in mids:
            if present[m] and mx[slot[m]] > 0:
                diffuse[m] = dmaps[slot[m]]
        for m in mids:
            if upresent[m] and m not in diffuse:
                diffuse[m] = dmaps[slot[m]]
    return Tile(fm.view, t, bragg if has_loss else None, bc.gid.astype(np.int32), tbin, mat, diffuse)
