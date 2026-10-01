"""Fused numba kernels for the TEM imaging path.

Each kernel does in one pass over the raster what the NumPy reference in
:mod:`de_twin.render.tem` does in several: the Bragg pixel keys and loss gather, the
mean-inner-potential phase, the exit wave (amplitude, absorption, texture, refraction loss,
phase), the transfer function of the usual aberration set, and the spectrum product and
``|psi|^2``. Same arithmetic in float32; they differ from the reference by float rounding
(tested). ``AVAILABLE`` is False without numba, and the reference is used.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import numba as nb

    AVAILABLE = True
except Exception:  # noqa: BLE001
    AVAILABLE = False


def _njit(**kw):
    if not AVAILABLE:
        return lambda f: f
    return nb.njit(cache=True, nogil=True, **kw)


_prange = nb.prange if AVAILABLE else range


# ------------------------------------------------------------------ Bragg pixels
@_njit()
def _tbin(t):
    """``thickness_bin_for``: rint(t / 2 nm) (half to even), clipped to 0..250."""
    x = t / 2.0
    r = math.floor(x + 0.5)
    if r - x == 0.5 and r % 2.0 != 0.0:
        r -= 1.0
    if r < 0.0:
        r = 0.0
    if r > 250.0:
        r = 250.0
    return int(r)


@_njit(parallel=True)
def bragg_pixels(grain_id, material_id, thickness, f_t, n, nb_, gpm, gid, pix, maxbin):
    """Per pixel: own grain (-1 none) and the key ``grain * nb_ + thickness bin`` (-1);
    ``maxbin[c, g]``: the largest thickness bin of grain g in row chunk c."""
    ny, nx = grain_id.shape
    nc = maxbin.shape[0]
    for c in _prange(nc):
        r0 = ny * c // nc
        r1 = ny * (c + 1) // nc
        for i in range(r0, r1):
            for j in range(nx):
                g = np.int64(grain_id[i, j])
                if g >= 0 and g < n and g // gpm == material_id[i, j]:
                    tb = _tbin(np.float64(np.float32(thickness[i, j]) * f_t))
                    gid[i, j] = g
                    pix[i, j] = g * nb_ + tb
                    if tb > maxbin[c, g]:
                        maxbin[c, g] = tb
                else:
                    gid[i, j] = -1
                    pix[i, j] = -1


@_njit(parallel=True)
def gather(lut, pix, out):
    ny, nx = pix.shape
    for i in _prange(ny):
        for j in range(nx):
            p = pix[i, j]
            out[i, j] = lut[p] if p >= 0 else np.float32(0.0)


# ------------------------------------------------------------------ exit wave
@_njit(parallel=True)
def mip_phase(mat, thick, umat, uthick, has_under, f_t, sigma, mip_v, out):
    ny, nx = mat.shape
    for i in _prange(ny):
        for j in range(nx):
            v = sigma * mip_v[mat[i, j]] * (np.float32(thick[i, j]) * f_t)
            if has_under:
                v = v + sigma * mip_v[umat[i, j]] * (uthick[i, j] * f_t)
            out[i, j] = v


@_njit(parallel=True)
def refraction_qmax(mip, inv_gc2):
    ny, nx = mip.shape
    rowmax = np.zeros(ny, np.float32)
    for i in _prange(ny):
        m = np.float32(0.0)
        for j in range(nx):
            gy = np.float32(0.5) * (mip[i + 1, j] - mip[i - 1, j]) if 0 < i < ny - 1 else np.float32(0.0)
            gx = np.float32(0.5) * (mip[i, j + 1] - mip[i, j - 1]) if 0 < j < nx - 1 else np.float32(0.0)
            q = (gx * gx + gy * gy) * inv_gc2
            if q > m:
                m = q
        rowmax[i] = m
    return rowmax.max()


@_njit(parallel=True)
def exit_wave(mat, thick, umat, uthick, has_under, f_t, lam_abs, bragg, mip, has_mip, refraction,
              inv_gc2, noise, has_noise, amorphous, tex_k, mip_v, inv_dens, kappa, psi, t_out,
              keep, has_keep, cutoff_lengths, kept_out):
    """``has_keep`` (coherent STEM): the amplitude keeps the diffuse part a band-limited coherent
    simulation represents, ``I = (T + kept) (1 - bragg)`` with ``kept = (1 - T)
    exp(-t / (cutoff_lengths Lambda)) keep[m]`` (``T`` of the primary layer), written to
    ``kept_out``."""
    ny, nx = mat.shape
    one = np.float32(1.0)
    zero = np.float32(0.0)
    for i in _prange(ny):
        for j in range(nx):
            m = mat[i, j]
            t = np.float32(thick[i, j]) * f_t
            t_out[i, j] = t
            e = t / lam_abs[m]
            if has_under:
                e = e + uthick[i, j] * f_t / lam_abs[umat[i, j]]
            if has_keep:
                tl = t / lam_abs[m]
                T64 = math.exp(-tl)
                kp = np.float32((1.0 - T64) * math.exp(-tl / cutoff_lengths) * keep[m])
                kept_out[i, j] = kp
                I64 = (T64 + kp) * (1.0 - bragg[i, j])
                I = np.float32(min(max(I64, 0.0), 1.0))
            else:
                T = np.float32(math.exp(-e))
                I = T * (one - bragg[i, j])
                if I < zero:
                    I = zero
                elif I > one:
                    I = one
            phi = mip[i, j] if has_mip else zero
            amp = np.float32(math.sqrt(I))
            if refraction:
                gy = np.float32(0.5) * (mip[i + 1, j] - mip[i - 1, j]) if 0 < i < ny - 1 else zero
                gx = np.float32(0.5) * (mip[i, j + 1] - mip[i, j - 1]) if 0 < j < nx - 1 else zero
                q = (gx * gx + gy * gy) * inv_gc2
                amp = amp * np.float32(math.exp(-q * q))
            if has_noise:
                var = zero
                if amorphous[m]:
                    a = tex_k * mip_v[m]
                    var = a * a * t * inv_dens[m]
                if has_under:
                    um = umat[i, j]
                    a = tex_k * mip_v[um]
                    var = var + a * a * (uthick[i, j] * f_t) * inv_dens[um]
                tex = np.float32(math.sqrt(var)) * noise[i, j]
                phi = phi + tex
                if kappa != zero:
                    amp = amp * np.float32(math.exp(-kappa * tex))
            psi[i, j] = complex(amp * math.cos(phi), amp * math.sin(phi))


# ------------------------------------------------------------------ transfer
@_njit(parallel=True)
def transfer_c1a1c3c5(kx1, ky1, tx, ty, radial, poly3, poly5, a_re, a_im, g1, g3, g5, ga_re, ga_im,
                      c0, gx0, gy0, es, et, kt2, kap2, H, transposed=False):
    """H(k) of an aberration set of C1, A1, C3 and C5 (`tem.transfer_function`); with
    ``transposed``, H.T (shape (nx, ny), for the transposed spectra of :func:`fft2_t`)."""
    ny = ky1.shape[0]
    nx = kx1.shape[0]
    if transposed:
        for j in _prange(nx):
            for i in range(ny):
                H[j, i] = _h1(kx1[j], ky1[i], tx, ty, radial, poly3, poly5, a_re, a_im, g1, g3, g5, ga_re, ga_im,
                              c0, gx0, gy0, es, et, kt2, kap2)
    else:
        for i in _prange(ny):
            for j in range(nx):
                H[i, j] = _h1(kx1[j], ky1[i], tx, ty, radial, poly3, poly5, a_re, a_im, g1, g3, g5, ga_re, ga_im,
                              c0, gx0, gy0, es, et, kt2, kap2)


@_njit()
def _h1(kx, ky, tx, ty, radial, poly3, poly5, a_re, a_im, g1, g3, g5, ga_re, ga_im, c0, gx0, gy0, es, et, kt2,
        kap2):
    KY = np.float32(ky + ty)
    KX = np.float32(kx + tx)
    k2 = KX * KX + KY * KY
    chi = k2 * (radial + k2 * (poly3 + poly5 * k2)) + a_re * (KX * KX - KY * KY) + a_im * (KX * KY)
    rad = g1 + k2 * (g3 + g5 * k2)
    gx = rad * KX + ga_re * KX + ga_im * KY - gx0
    gy = rad * KY - ga_re * KY + ga_im * KX - gy0
    chi = chi - c0
    k2t = k2 - kt2
    if kap2 > 0.0 and k2 > kap2:
        return complex(0.0, 0.0)
    env = math.exp(-(es * (gx * gx + gy * gy) + et * k2t * k2t))
    return complex(env * math.cos(chi), -env * math.sin(chi))


@_njit(parallel=True)
def spectrum_product(spec, H, has_h, ry, rx, has_ramp, out):
    ny, nx = spec.shape
    for i in _prange(ny):
        for j in range(nx):
            v = spec[i, j]
            if has_h:
                v = v * H[i, j]
            if has_ramp:
                v = v * ry[i] * rx[j]
            out[i, j] = v


@_njit(parallel=True)
def intensity(psi, out):
    ny, nx = psi.shape
    for i in _prange(ny):
        for j in range(nx):
            v = psi[i, j]
            out[i, j] = v.real * v.real + v.imag * v.imag


# ------------------------------------------------------------------ gaussian (mode "nearest")
@_njit(parallel=True)
def _gauss_rows(a, w, out):
    ny, nx = a.shape
    r = (w.shape[0] - 1) // 2
    for i in _prange(ny):
        for j in range(nx):
            acc = np.float32(0.0)
            for k in range(-r, r + 1):
                jj = min(max(j + k, 0), nx - 1)
                acc += w[k + r] * a[i, jj]
            out[i, j] = acc


@_njit(parallel=True)
def _gauss_cols(a, w, out):
    ny, nx = a.shape
    r = (w.shape[0] - 1) // 2
    for i in _prange(ny):
        for j in range(nx):
            out[i, j] = np.float32(0.0)
        for k in range(-r, r + 1):
            ii = min(max(i + k, 0), ny - 1)
            wk = w[k + r]
            for j in range(nx):
                out[i, j] += wk * a[ii, j]


def gaussian_nearest(a: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """``scipy.ndimage.gaussian_filter(a, sigma, mode="nearest", truncate=truncate)`` of a
    float32 image (the same kernel weights; float32 sums)."""
    radius = int(truncate * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / float(sigma)) ** 2)
    w = (w / w.sum()).astype(np.float32)
    a = np.ascontiguousarray(a, np.float32)
    tmp = scratch("gauss_tmp", a.shape, np.float32)
    out = scratch("gauss_out", a.shape, np.float32)
    _gauss_cols(a, w, tmp)
    _gauss_rows(tmp, w, out)
    return out


# ------------------------------------------------------------------ 2-D FFTs, threaded
@_njit(parallel=True)
def transpose(x, out):
    """``out[:] = x.T`` in cache-sized blocks."""
    n0, n1 = x.shape
    B = 32
    for bi in _prange((n0 + B - 1) // B):
        i0 = bi * B
        i1 = min(i0 + B, n0)
        for j0 in range(0, n1, B):
            j1 = min(j0 + B, n1)
            for i in range(i0, i1):
                for j in range(j0, j1):
                    out[j, i] = x[i, j]


def _rows(x, inverse: bool) -> None:
    """1-D FFTs of every row of ``x``, in place, in row chunks on the shared pool (scipy's own
    ``workers`` barely helps a single 2-D transform: its column pass is memory-bound)."""
    from scipy import fft as sfft

    from .util import pool

    f = sfft.ifft if inverse else sfft.fft
    n = x.shape[0]
    k = max(1, min(pool()._max_workers, n // 32))
    e = np.linspace(0, n, k + 1).astype(int)

    def one(i):
        a, b = int(e[i]), int(e[i + 1])
        r = f(x[a:b], axis=1, overwrite_x=True)
        if r.ctypes.data != x[a:b].ctypes.data:
            x[a:b] = r
    list(pool().map(one, range(k)))


_SCRATCH = __import__("threading").local()


def scratch(name: str, shape, dtype) -> np.ndarray:
    """A per-thread, reused array for a temporary of one render (uninitialised). Fresh large
    arrays cost their page faults on first touch (~4 ms per 14 MB here), which for the
    imaging path's 1344^2 complex temporaries was most of an FFT's time."""
    d = getattr(_SCRATCH, "d", None)
    if d is None:
        d = _SCRATCH.d = {}
    key = (name, tuple(int(v) for v in shape), np.dtype(dtype).str)
    a = d.get(key)
    if a is None:
        for k in [k for k in d if k[0] == name]:  # one size per name: a view change frees the old
            del d[k]
        a = d[key] = np.empty(shape, dtype)
    return a


def fft2_t(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """The 2-D FFT of complex64 ``x`` (overwritten), returned TRANSPOSED (into ``out`` if
    given): rows, transpose, rows. Keeping the spectrum transposed saves a transpose each way."""
    _rows(x, False)
    t = out if out is not None else np.empty((x.shape[1], x.shape[0]), x.dtype)
    transpose(x, t)
    _rows(t, False)
    return t


def ifft2_t(t: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Inverse of :func:`fft2_t`: from a transposed spectrum (overwritten) to the image."""
    _rows(t, True)
    x = out if out is not None else np.empty((t.shape[1], t.shape[0]), t.dtype)
    transpose(t, x)
    _rows(x, True)
    return x


@_njit(parallel=True)
def mul_phase(psi, phi):
    """``psi *= exp(i phi)`` (the factor rounded to complex64 first, like the NumPy form)."""
    ny, nx = psi.shape
    for i in _prange(ny):
        for j in range(nx):
            p = float(phi[i, j])
            f = np.complex64(complex(math.cos(p), math.sin(p)))
            psi[i, j] = psi[i, j] * f
