"""Fused numba kernels for coherent 4D-STEM (:mod:`de_twin.render.coherent`).

* :func:`probe_samples`: the partially coherent probe's focal / source samples on the
  aperture's k pixels (one pass: phase, cos / sin, weight);
* :func:`fill_exit`: exit waves ``t(window) x probe mode (x placement ramp)`` of a batch of
  scan points;
* :func:`tile_maps`: a transmission tile's incoherent add-on maps (Bragg, diffuse) and
  thickness bins in one pass;
* :func:`accumulate`: ``w |F|^2`` of the detector crop, binned onto the detector pixels and
  added into the batch's patterns (no full-grid intensity array);
* :func:`annulus_sums`: ``w |F|^2`` summed over an annular detector's k pixels;
* :func:`window_dots`: probe-intensity-weighted sums of a map over each point's window (the
  incoherent Bragg / diffuse weights of the points actually rendered).

Same arithmetic as the NumPy formulation, in float32 / complex64 (sums in float64 where they
run over a whole window); results differ from it by float rounding only. The per-batch
kernels are serial and release the GIL: the caller runs batches on its thread pool.
``AVAILABLE`` is False without numba (the NumPy formulation is used).
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


@_njit(parallel=True)
def probe_samples(chi0, k2t, kx, ky, a, fdf, fw, sx, sy, sw, pi_lam, out, wts):
    """Sample ``(f, s)`` = ``a exp(-i (chi0 + pi lam df_f k2t + 2 pi (kx sx_s + ky sy_s))) / |a|``
    into ``out[f * S + s]`` (complex64) and its weight ``fw_f sw_s`` into ``wts``."""
    m = chi0.shape[0]
    nf = fdf.shape[0]
    ns = sx.shape[0]
    norm = 0.0
    for p in range(m):
        norm += a[p] * a[p]
    inv = 1.0 / math.sqrt(norm)
    for q in _prange(nf * ns):
        f = q // ns
        s = q % ns
        wts[q] = fw[f] * sw[s]
        c = pi_lam * fdf[f]
        tx = 2.0 * math.pi * sx[s]
        ty = 2.0 * math.pi * sy[s]
        for p in range(m):
            ph = chi0[p] + c * k2t[p] + tx * kx[p] + ty * ky[p]
            amp = a[p] * inv
            out[q, p] = complex(amp * math.cos(ph), -amp * math.sin(ph))


@_njit()
def fill_exit(t, r0s, c0s, mode, ry, rx, has_ramp, e):
    """``e[k] = t[r0:r0+n, c0:c0+n] * mode (* ry[k][:, None] * rx[k][None, :])``."""
    nb_ = e.shape[0]
    n = mode.shape[0]
    for k in range(nb_):
        r0 = r0s[k]
        c0 = c0s[k]
        for i in range(n):
            ti = t[r0 + i]
            mi = mode[i]
            if has_ramp:
                fy = ry[k, i]
                for j in range(n):
                    e[k, i, j] = ti[c0 + j] * mi[j] * (fy * rx[k, j])
            else:
                for j in range(n):
                    e[k, i, j] = ti[c0 + j] * mi[j]


@_njit()
def accumulate(F, y0s, x0s, H, W, fy, fx, w, acc):
    """``acc[k, i // fy, j // fx] += w |F[k, y0 + i, x0 + j]|^2`` over the detector crop
    (``H`` x ``W`` simulation pixels from ``(y0s[k], x0s[k])``); pixels outside the simulated
    k range add nothing."""
    nb_, n, m = F.shape
    for k in range(nb_):
        y0 = y0s[k]
        x0 = x0s[k]
        ja = max(0, -x0)
        jb = min(W, m - x0)
        for i in range(H):
            ya = y0 + i
            if ya < 0 or ya >= n:
                continue
            bi = i // fy
            Fr = F[k, ya]
            ar = acc[k, bi]
            for j in range(ja, jb):
                v = Fr[x0 + j]
                ar[j // fx] += w * (v.real * v.real + v.imag * v.imag)


@_njit()
def annulus_sums(F, iy, ix, w, out, col):
    """``out[k, col] += w sum_p |F[k, iy[p], ix[p]]|^2``."""
    for k in range(F.shape[0]):
        s = 0.0
        for p in range(iy.shape[0]):
            v = F[k, iy[p], ix[p]]
            s += v.real * v.real + v.imag * v.imag
        out[k, col] += w * s


@_njit(parallel=True)
def window_dots(a, r0s, c0s, inten, out):
    """``out[k] = sum a[r0:r0+n, c0:c0+n] * inten`` (float64 sums)."""
    n = inten.shape[0]
    for k in _prange(r0s.shape[0]):
        r0 = r0s[k]
        c0 = c0s[k]
        s = 0.0
        for i in range(n):
            ai = a[r0 + i]
            wi = inten[i]
            for j in range(n):
                s += ai[c0 + j] * wi[j]
        out[k] = s


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
    return r


@_njit(parallel=True)
def tile_maps(mat, thick, f_t, lam, has_under, umat, uthick, cutoff, fb, keep_on, loss, has_loss,
              diffuse_on, slot, bragg, dmaps, dmax, tbin):
    """The incoherent add-on maps of a coherent-STEM tile in one pass (``coherent.build_tile``):
    ``bragg = (T + dw (1 - fb)) loss``, the diffuse weight beyond the band limit
    ``dw fb`` (and the under layer's) into ``dmaps[slot[m]]``, ``dmax[row, slot]`` its row
    maxima, and the thickness bins."""
    ny, nx = mat.shape
    for i in _prange(ny):
        for j in range(nx):
            m = mat[i, j]
            t = np.float32(thick[i, j]) * f_t
            lp = lam[m]
            T = np.float32(math.exp(-t / lp))
            dw = np.float32((np.float32(1.0) - T) * math.exp(-t / (cutoff * lp)))
            uw = np.float32(0.0)
            um = m
            if has_under:
                um = umat[i, j]
                tu = np.float32(uthick[i, j]) * f_t
                lu = lam[um]
                Tu = np.float32(math.exp(-tu / lu))
                uw = np.float32((np.float32(1.0) - Tu) * math.exp(-tu / (cutoff * lu)))
                T = T * Tu
                dw = dw * Tu
            if has_loss:
                kept = T + dw * (np.float32(1.0) - fb[m]) if keep_on else T
                bragg[i, j] = kept * loss[i, j]
            if diffuse_on:
                if m != 0:
                    v = dw * fb[m]
                    dmaps[slot[m], i, j] = v
                    if v > dmax[i, slot[m]]:
                        dmax[i, slot[m]] = v
                if has_under and um != 0:
                    dmaps[slot[um], i, j] += uw * fb[um]
            tbin[i, j] = np.int16(_tbin(t))
