"""EBSD camera frames in one fused CPU pass: scan point -> raw uint16 frame.

The general path makes a camera frame in three threaded passes with Python between them:
the projection writes the Poisson means (:func:`de_twin.ebsd.kernels.render`), the detector
draws the electrons and their deposits (:func:`de_twin.detector.fast.deposit`), then charge
sharing and the analogue chain (:func:`de_twin.detector.fast.analog_out`). Each pass is
parallel over the rows of ONE frame, so every frame pays three fork/joins of the thread
pool, three trips of its intermediates through memory and the Python in between.

:func:`raw_frames` does the same physics for a BATCH of frames in one call. A task is a band
of rows of one frame (the whole frame when there are frames enough to go round), so the
cores are kept busy by several frames at once, and it goes from the scan point to the raw
values of its rows in row-sized scratch: per row it projects the master pattern, draws the
electrons and their deposit, and once the rows above and below are there (charge sharing
reads them) runs the analogue chain and writes the raw row wherever the caller says. A band
recomputes the deposit of the row either side of it; the noise is a function of (frame,
pixel), so it is the same row.

Each row is a few loops written so the compiler vectorises them:

* no data-dependent branches, and no NaN-aware reductions (``min`` over a row does not
  vectorise; a count does);
* arrays indexed from zero or by unsigned offsets: a signed index of unknown sign costs a
  negative-index check per access, which stops vectorisation;
* no transcendental calls inside a row (numba has no vector ``exp``): the backscatter lobe's
  ``exp`` is a polynomial, and the master pattern lookup splits each row into runs on one
  cube face, so a run is straight-line arithmetic and then one gather pass.

The exact Poisson draw below ``POISSON_EXACT_BELOW`` is a sequential search per pixel (a
chain of dependent multiplies), so a row's dim pixels are searched together, one electron
count at a time across all of them. Scratch is per thread and made once: an allocation per
task contends for the heap, and views of one shared array contend on its reference count.

Same physics and noise streams as the general path (``detector.fast``: the normal table at
the frame's offsets, the SplitMix64 counter per pixel), so a frame matches the general
path's except where float rounding tips a draw: here the projection's directions,
background and lobe are float32, and the deposit's Wilson-Hilferty factor is computed
rather than looked up.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from ..detector.fast import _GOLD, _M1, _M2, _U53, NORMALS, POISSON_EXACT_BELOW
from ..parallel import PARALLEL
from .kernels import LOBE_STEP, _detector_to_crystal

_MASK = NORMALS - 1
#: The second normal stream of a frame (the Poisson draw's) starts this far after the first.
_SECOND_STREAM = 2654435761
#: The exact Poisson draw stops counting here (``detector.fast.deposit``'s cap).
_MAX_COUNT = 200
_LOG2E = 1.4426950408889634
_LN2_HI = 0.693145751953125  # ln 2 split so k * hi is exact in float32
_LN2_LO = 1.428606765330187e-06
#: 1/k! for k = 7 .. 0: exp on |r| <= ln 2 / 2 to ~1e-8, below float32's resolution.
_EXP_C = tuple(1.0 / math.factorial(k) for k in range(7, -1, -1))
U64 = np.uint64
U32 = np.uint32
F32 = np.float32
_CHUNK = 8  # the dim-pixel search works in whole vectors of this many; scratch rows are padded to it
_ROWS32 = 20
_ROWSU32 = 3


def _njit(**kw):
    if kw.get("parallel"):
        kw["parallel"] = PARALLEL
    return nb.njit(cache=True, nogil=True, fastmath=True, error_model="numpy", **kw)


def extended_table(table: np.ndarray, width: int) -> np.ndarray:
    """The normal table followed by its first *width* values again, so a row of up to *width*
    normals from any start (``& mask``) is contiguous: ``ext[s + c] == table[(s + c) & mask]``."""
    return np.ascontiguousarray(np.concatenate([table, table[:width]]))


def scratch(threads: int, width: int) -> tuple:
    """Per-thread scratch for :func:`raw_frames` on frames *width* wide: two typed lists of
    arrays, one array per thread, so a thread's views count references on a cache line no
    other thread touches."""
    from numba.typed import List

    width = max(width, (width - 1) // LOBE_STEP + 2, _CHUNK)
    width = (width + _CHUNK - 1) // _CHUNK * _CHUNK
    s32, su32 = List(), List()
    for _ in range(threads):
        s32.append(np.zeros(_ROWS32 * width, np.float32))
        su32.append(np.zeros(_ROWSU32 * width, np.uint32))
    return s32, su32


@_njit()
def _exp(x, bits, scale, n):
    """``x[:n] = exp(x[:n])`` in float32 for moderate arguments, vectorised: a polynomial on
    ``|r| <= ln 2 / 2`` and 2^k made from its bits (``bits``, int32 scratch; ``scale``, the
    same memory as float32)."""
    log2e = F32(_LOG2E)
    hi = F32(_LN2_HI)
    lo = F32(_LN2_LO)
    half = F32(0.5)
    for i in range(n):
        t = x[i]
        k = np.floor(t * log2e + half)
        r = (t - k * hi) - k * lo
        p = ((((((F32(_EXP_C[0]) * r + F32(_EXP_C[1])) * r + F32(_EXP_C[2])) * r + F32(_EXP_C[3])) * r
                + F32(_EXP_C[4])) * r + F32(_EXP_C[5])) * r + F32(_EXP_C[6])) * r + F32(_EXP_C[7])
        x[i] = p
        bits[i] = (np.int32(k) + np.int32(127)) << np.int32(23)
    for i in range(n):
        x[i] *= scale[i]


@_njit()
def _point(k, mats, o1, o2, ph1, ph2, mix, contrast, es, finv, has_f, dx, dy, pc_lin, s2dT, M1, M2):
    """Per scan point: fills M1, M2 (detector -> crystal), returns
    (w2, x0, y0, dd, c, e, p1, p2, g1)."""
    g1 = o1[k]
    g2 = o2[k]
    w2 = mix[k] if (g2 >= 0 and g1 >= 0) else 0.0
    if g1 >= 0:
        _detector_to_crystal(mats[g1], finv[k if has_f else 0], has_f, s2dT, M1)
    if w2 > 0.0:
        _detector_to_crystal(mats[g2], finv[k if has_f else 0], has_f, s2dT, M2)
    x0 = pc_lin[0] + pc_lin[1] * dx[k] + pc_lin[2] * dy[k]
    y0 = pc_lin[3] + pc_lin[4] * dx[k] + pc_lin[5] * dy[k]
    dd = pc_lin[6] + pc_lin[7] * dx[k] + pc_lin[8] * dy[k]
    c = contrast[k] if g1 >= 0 else 0.0
    return w2, x0, y0, dd, c, es[k], ph1[k], ph2[k], g1


@_njit(inline="always")
def _face(a, b, c, full):
    """The cube face of direction (a, b, c), as ``kernels._cube_lookup`` picks it."""
    ax, ay, az = abs(a), abs(b), abs(c)
    if ax >= ay and ax >= az:
        f, m = 0, a
    elif ay >= az:
        f, m = 1, b
    else:
        f, m = 2, c
    if m < 0.0 and full:
        f += 3
    return f


@_njit()
def _face_run(n, plane, m0, ms, u0, us, v0, vs, j0, j1, iota, q, tx, ty):
    """Bilinear nodes of pixels j0 .. j1 of a row on one cube face (direction ``(m, u, v) + j
    (ms, us, vs)``, j = ``iota[j]``): ``q`` the flat index of the top-left node, ``tx``, ``ty``
    its fractions."""
    s = F32((n - 1) * 0.5)
    one = F32(1.0)
    top = np.int32(n - 2)
    zero = np.int32(0)
    nn = U32(n)
    base = U32(plane * n * n)
    m0f, msf, u0f, usf, v0f, vsf = F32(m0), F32(ms), F32(u0), F32(us), F32(v0), F32(vs)
    for t in range(j1 - j0):
        jt = U64(j0) + U64(t)
        jj = iota[jt]
        r = one / (m0f + jj * msf)
        fx = ((u0f + jj * usf) * r + one) * s
        fy = ((v0f + jj * vsf) * r + one) * s
        ix = min(max(np.int32(fx), zero), top)
        iy = min(max(np.int32(fy), zero), top)
        tx[jt] = fx - F32(ix)
        ty[jt] = fy - F32(iy)
        q[jt] = base + U32(iy) * nn + U32(ix)


@_njit()
def _nodes(n, face0, full, a0, sa, b0, sb, c0, sc, iota, q, tx, ty):
    """:func:`_face_run` over a row along crystal direction ``(a0, b0, c0) + j (sa, sb, sc)``
    (any length: the cube lookup is scale-free). A straight line crosses each face of the cube
    once, so the row is a few runs on one face each, found by bisection."""
    w = iota.shape[0]
    j = 0
    while j < w:
        f = _face(a0 + j * sa, b0 + j * sb, c0 + j * sc, full)
        end = w
        if _face(a0 + (w - 1) * sa, b0 + (w - 1) * sb, c0 + (w - 1) * sc, full) != f:
            lo, hi = j, w - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if _face(a0 + mid * sa, b0 + mid * sb, c0 + mid * sc, full) == f:
                    lo = mid
                else:
                    hi = mid
            end = lo + 1
        axis = f % 3
        if axis == 0:
            _face_run(n, face0 + f, a0, sa, b0, sb, c0, sc, j, end, iota, q, tx, ty)
        elif axis == 1:
            _face_run(n, face0 + f, b0, sb, a0, sa, c0, sc, j, end, iota, q, tx, ty)
        else:
            _face_run(n, face0 + f, c0, sc, a0, sa, b0, sb, j, end, iota, q, tx, ty)
        j = end


@_njit()
def _gather(bank, n, offset, scale, q, tx, ty, val):
    """``val[j]`` = the standardised master value at the bilinear nodes ``q, tx, ty``."""
    one = U32(1)
    nn = U32(n)
    of = F32(offset)
    sf = F32(scale)
    for j in range(val.shape[0]):
        k = q[j]
        v00 = F32(bank[k])
        v01 = F32(bank[k + one])
        v10 = F32(bank[k + nn])
        v11 = F32(bank[k + nn + one])
        top = v00 + (v01 - v00) * tx[j]
        bot = v10 + (v11 - v10) * tx[j]
        val[j] = of + sf * (top + (bot - top) * ty[j])


@_njit()
def _project_row(i, bank, n, face0, full, offs, scales, M1, M2, w2, x0, y0, dd, c, e, p1, p2, g1,
                 ox, oy, bx, by, normal, fwd, lobe, kappa, vmf, t_nb, iota, lob, bits, bitsf, ex, bgf, val,
                 val2, q, tx, ty, lam):
    """Row *i* of a point's pattern as Poisson means (``lam``, electrons per grid pixel in the
    frame); returns their sum. ``kernels.render`` with ``ss = 1``."""
    w = lam.shape[0]
    Y = oy + (i + 0.5) * by - y0
    X0 = ox + 0.5 * bx - x0
    one = F32(1.0)
    zero = F32(0.0)
    if kappa > 0.0:  # the lobe is smooth: exact every LOBE_STEP pixels, linear between
        na = (w - 1) // LOBE_STEP + 2
        kf = F32(kappa)
        Xa0 = F32(X0)
        step = F32(LOBE_STEP * bx)
        last = F32(ox + (w - 1 + 0.5) * bx - x0)
        yy = F32(Y * Y + dd * dd)
        yf = F32(Y * fwd[1] + dd * fwd[2])
        f0 = F32(fwd[0])
        for a in range(na):
            Xa = min(Xa0 + F32(a) * step, last)
            lob[a] = kf * ((Xa * f0 + yf) / math.sqrt(Xa * Xa + yy) - one)
        _exp(lob, bits, bitsf, na)
        quarter = F32(1.0 / LOBE_STEP)
        for a in range(w // LOBE_STEP):
            l0 = lob[a]
            d = (lob[a + 1] - l0) * quarter
            for s in range(LOBE_STEP):
                ex[a * LOBE_STEP + s] = l0 + F32(s) * d
        j0 = ((w - 1) // LOBE_STEP) * LOBE_STEP  # the last segment may be shorter
        j1 = min(j0 + LOBE_STEP, w - 1)
        a = j0 // LOBE_STEP
        for j in range(j0, w):
            t = F32((j - j0) / (j1 - j0)) if j1 > j0 else zero
            ex[j] = lob[a] + (lob[a + 1] - lob[a]) * t
    else:
        for j in range(w):
            ex[j] = one
    # the backscatter background and solid angle
    X0f = F32(X0)
    bxf = F32(bx)
    ddf = F32(dd)
    yy = F32(Y * Y + dd * dd)
    yn = F32(Y * normal[1] + dd * normal[2])
    n0 = F32(normal[0])
    tf = F32(t_nb)
    k_cos = F32(t_nb * e * (1.0 - lobe) / math.pi / (dd * dd))
    k_lobe = F32(t_nb * e * lobe * vmf / (dd * dd))
    inv_rim = F32(1.0 / 0.03)
    for j in range(w):
        X = X0f + iota[j] * bxf
        inv = one / math.sqrt(X * X + yy)
        rz = ddf * inv
        rn = (X * n0 + yn) * inv
        shadow = min(rn * inv_rim, one)
        f = shadow * (k_cos * rn + k_lobe * ex[j]) * rz * rz * rz
        bgf[j] = f if rn > zero else zero
    # the master pattern and the band modulation
    acc = 0.0
    if g1 >= 0:
        _nodes(n, face0[p1], full[p1], M1[0, 0] * X0 + M1[0, 1] * Y + M1[0, 2] * dd, M1[0, 0] * bx,
               M1[1, 0] * X0 + M1[1, 1] * Y + M1[1, 2] * dd, M1[1, 0] * bx,
               M1[2, 0] * X0 + M1[2, 1] * Y + M1[2, 2] * dd, M1[2, 0] * bx, iota, q, tx, ty)
        _gather(bank, n, offs[p1], scales[p1], q, tx, ty, val)
        if w2 > 0.0:
            _nodes(n, face0[p2], full[p2], M2[0, 0] * X0 + M2[0, 1] * Y + M2[0, 2] * dd, M2[0, 0] * bx,
                   M2[1, 0] * X0 + M2[1, 1] * Y + M2[1, 2] * dd, M2[1, 0] * bx,
                   M2[2, 0] * X0 + M2[2, 1] * Y + M2[2, 2] * dd, M2[2, 0] * bx, iota, q, tx, ty)
            _gather(bank, n, offs[p2], scales[p2], q, tx, ty, val2)
            wa = F32(1.0 - w2)
            wb = F32(w2)
            for j in range(w):
                val[j] = wa * val[j] + wb * val2[j]
        cf = F32(c)
        for j in range(w):
            v = max(bgf[j] * (one + cf * val[j]), zero)
            lam[j] = v
            acc += v
    else:
        for j in range(w):
            v = bgf[j]
            lam[j] = v
            acc += v
    return acc


@_njit()
def _deposit_row(r, lam, S, so, adu, k, key, table, zstart, lane, ln, bits, bitsf, u, p, f, cnt, Ls):
    """``detector.fast.deposit`` for one row: deposited ADU per grid pixel, into
    ``S[so : so + w]``. ``table``: :func:`extended_table`; the rest scratch."""
    w = lam.shape[0]
    base = r * w
    zg = U64((zstart + base) & _MASK)
    zp = U64((zstart + _SECOND_STREAM + base) & _MASK)
    s0 = U64(so)
    adu32 = F32(adu)
    k32 = F32(k)
    half = F32(0.5)
    one = F32(1.0)
    zero = F32(0.0)
    ninth = F32(1.0 / 9.0)
    third = F32(1.0 / 3.0)
    cut = F32(POISSON_EXACT_BELOW)
    # every pixel by the normal approximation (right from POISSON_EXACT_BELOW up), and its deposit
    for c in range(w):
        L = lam[c]
        cu = U64(c)
        n = max(np.floor(L + math.sqrt(L) * table[zp + cu] + half), zero)
        ri = one / math.sqrt(max(n, one) * k32)
        g = max(one - ninth * ri * ri + third * table[zg + cu] * ri, zero)
        S[s0 + cu] = n * adu32 * g * g * g
    dim = 0  # (`&`, not `and`: a short-circuit is a branch, and a branch stops vectorisation)
    for c in range(w):
        dim += 1 if ((lam[c] < cut) & (lam[c] > zero)) else 0
    if dim == 0:
        return
    # the dim pixels again, exactly
    m = 0
    for c in range(w):
        lane[m] = U32(c)
        m += 1 if ((lam[c] < cut) & (lam[c] > zero)) else 0
    for i in range(m):
        L = lam[lane[i]]
        Ls[i] = L
        ln[i] = -L
    _exp(ln, bits, bitsf, m)  # P(0)
    ukey = U64(key)
    for i in range(m):
        x = ukey + U64(base + lane[i]) * U64(_GOLD)
        x = (x ^ (x >> U64(30))) * U64(_M1)
        x = (x ^ (x >> U64(27))) * U64(_M2)
        x = x ^ (x >> U64(31))
        u[i] = F32(((x >> U64(11)) + U64(1)) * _U53)
        p[i] = ln[i]
        f[i] = ln[i]
        cnt[i] = zero
    mp = (m + _CHUNK - 1) // _CHUNK * _CHUNK  # whole vectors: no scalar remainder per count
    for i in range(m, mp):  # padding lanes, never drawn
        u[i] = zero
        p[i] = zero
        f[i] = one
        Ls[i] = zero
        cnt[i] = zero
    # inversion, n = #{k >= 1 : u > F(k - 1)}: the row's dim pixels a count at a time
    for kk in range(1, _MAX_COUNT + 1):
        inv_k = F32(1.0 / kk)
        more = zero  # a float sum vectorises with the rest; an integer one halves the width
        for i in range(mp):
            up = u[i] > f[i]
            cnt[i] += one if up else zero
            more += one if up else zero
            p[i] *= Ls[i] * inv_k
            f[i] += p[i]
        if more == zero:
            break
    for i in range(m):
        c = lane[i]
        n = cnt[i]
        ri = one / math.sqrt(max(n, one) * k32)
        g = max(one - ninth * ri * ri + third * table[zg + U64(c)] * ri, zero)
        S[s0 + U64(c)] = n * adu32 * g * g * g


@_njit()
def _analog_row(q, S, su, sr, sd, a, inv_nb, gain, offb, hot_ptr, hot_col, hot_val, sigma, scale, vmax,
                table, nstart, v, out, o0):
    """``detector.fast.analog_out`` for row *q* (the deposits of rows q-1, q, q+1): charge
    sharing, gain, offset + dark, hot pixels, read noise, clip, round, into ``out[o0:o0+w]``.
    Returns the clipped-pixel count."""
    w = v.shape[0]
    a32 = F32(a)
    ca = F32((1.0 - 2.0 * a) / a) if a > 0 else F32(0.0)
    inb = F32(inv_nb)
    s32 = F32(sigma)
    sc = F32(scale)
    vm = F32(vmax)
    zero = F32(0.0)
    half = F32(0.5)
    m0 = U64(q * w)  # gain, offb: flat (h * w)
    z0 = U64((nstart + q * w) & _MASK)
    oo = U64(o0)
    u_, r_, d_ = U64(su), U64(sr), U64(sd)
    if a > 0:
        for c in range(w):
            cu = U64(c)
            v[c] = a32 * (S[u_ + cu] + S[d_ + cu] + S[r_ + cu] * ca)
    else:
        for c in range(w):
            v[c] = S[r_ + U64(c)]
    sat = 0
    if a > 0 and w > 2:
        for c in range(1, w - 1):
            cu = U64(c)
            y = (a32 * (v[c - 1] + v[c + 1] + v[c] * ca) * inb * gain[m0 + cu] + offb[m0 + cu]
                 + s32 * table[z0 + cu]) * sc
            sat += 1 if y >= vm else 0
            out[oo + cu] = np.uint16(min(max(y, zero), vm) + half)
        sat += _analog_pixel(0, v, a, inv_nb, gain, offb, sigma, scale, vmax, table, m0, z0, zero, out, oo)
        sat += _analog_pixel(w - 1, v, a, inv_nb, gain, offb, sigma, scale, vmax, table, m0, z0, zero, out, oo)
    else:
        for c in range(w):
            sat += _analog_pixel(c, v, a, inv_nb, gain, offb, sigma, scale, vmax, table, m0, z0, zero, out, oo)
    for hh in range(hot_ptr[q], hot_ptr[q + 1]):  # hot pixels: again, with their dark current
        c = hot_col[hh]
        cu = U64(c)
        y = _analog_value(c, v, a, inv_nb, gain, offb, sigma, scale, table, m0, z0, hot_val[hh])
        old = _analog_value(c, v, a, inv_nb, gain, offb, sigma, scale, table, m0, z0, zero)
        sat += (1 if y >= vm else 0) - (1 if old >= vm else 0)
        out[oo + cu] = np.uint16(min(max(y, zero), vm) + half)
    return sat


@_njit(inline="always")
def _analog_value(c, v, a, inv_nb, gain, offb, sigma, scale, table, m0, z0, hot):
    w = v.shape[0]
    a32 = F32(a)
    if a > 0:
        ca = F32((1.0 - 2.0 * a) / a)
        x = a32 * (v[c - 1 if c > 0 else 0] + v[c + 1 if c < w - 1 else w - 1] + v[c] * ca)
    else:
        x = v[c]
    cu = U64(c)
    return (x * F32(inv_nb) * gain[m0 + cu] + offb[m0 + cu] + hot + F32(sigma) * table[z0 + cu]) * F32(scale)


@_njit(inline="always")
def _analog_pixel(c, v, a, inv_nb, gain, offb, sigma, scale, vmax, table, m0, z0, zero, out, oo):
    y = _analog_value(c, v, a, inv_nb, gain, offb, sigma, scale, table, m0, z0, zero)
    vm = F32(vmax)
    out[oo + U64(c)] = np.uint16(min(max(y, zero), vm) + F32(0.5))
    return 1 if y >= vm else 0


@_njit()
def _band(b, r0, r1, d32, dui, bank, bank_n, face0, full, offs, scales, mats, o1, o2, ph1, ph2, mix, contrast, es,
          finv, has_f, dx, dy, pc_lin, s2dT, ox, oy, bx, by, normal, fwd, lobe, kappa, vmf, t_nb,
          h, gain, offb, hot_ptr, hot_col, hot_val, adu, k, a, inv_nb, sigma, scale, vmax, table,
          ks, keys, zstarts, nstarts, out, offsets):
    """Rows r0 .. r1 of frame b (see :func:`raw_frames`), in the scratch ``d32``, ``dui``.
    Returns (electrons, clipped pixels).

    The rows' projection, deposit and analogue stage are written out here rather than called:
    a call with this many arrays costs as much as one of the row's loops."""
    w = gain.shape[0] // h
    na = (w - 1) // LOBE_STEP + 2
    W = d32.shape[0] // _ROWS32  # scratch row length (>= w)
    iota = d32[0:W]  # (W >= w and >= the lobe's anchors)
    iw = d32[0:w]
    ex = d32[W:W + w]
    bgf = d32[2 * W:2 * W + w]
    val = d32[3 * W:3 * W + w]
    val2 = d32[4 * W:4 * W + w]
    tx = d32[5 * W:5 * W + w]
    ty = d32[6 * W:6 * W + w]
    lam = d32[7 * W:7 * W + w]
    vrow = d32[8 * W:8 * W + w]
    S3 = d32[9 * W:12 * W]
    lu = d32[12 * W:13 * W]
    lp = d32[13 * W:14 * W]
    lf = d32[14 * W:15 * W]
    lcnt = d32[15 * W:16 * W]
    ln = d32[16 * W:17 * W]
    lob = d32[17 * W:17 * W + na]
    Ls = d32[18 * W:19 * W]
    q = dui[0:w]
    lane = dui[W:2 * W]
    bits = dui[2 * W:3 * W].view(np.int32)
    bitsf = dui[2 * W:3 * W].view(np.float32)
    for j in range(W):
        iota[j] = F32(j)
    M1 = np.zeros((3, 3))
    M2 = np.zeros((3, 3))
    w2, x0, y0, dd, c, e, p1, p2, g1 = _point(ks[b], mats, o1, o2, ph1, ph2, mix, contrast, es, finv, has_f,
                                              dx, dy, pc_lin, s2dT, M1, M2)
    one = F32(1.0)
    zero = F32(0.0)
    half = F32(0.5)
    # per point: the projection's constants
    X0 = ox + 0.5 * bx - x0
    kf = F32(kappa)
    Xa0 = F32(X0)
    astep = F32(LOBE_STEP * bx)
    alast = F32(ox + (w - 1 + 0.5) * bx - x0)
    f0 = F32(fwd[0])
    quarter = F32(1.0 / LOBE_STEP)
    jl0 = ((w - 1) // LOBE_STEP) * LOBE_STEP  # the last lobe segment may be shorter
    jl1 = min(jl0 + LOBE_STEP, w - 1)
    X0f = F32(X0)
    bxf = F32(bx)
    ddf = F32(dd)
    n0 = F32(normal[0])
    k_cos = F32(t_nb * e * (1.0 - lobe) / math.pi / (dd * dd))
    k_lobe = F32(t_nb * e * lobe * vmf / (dd * dd))
    inv_rim = F32(1.0 / 0.03)
    cf = F32(c)
    wa = F32(1.0 - w2)
    wb = F32(w2)
    # per frame: the detector's constants
    adu32 = F32(adu)
    k32 = F32(k)
    ninth = F32(1.0 / 9.0)
    third = F32(1.0 / 3.0)
    cut = F32(POISSON_EXACT_BELOW)
    ukey = U64(keys[b])
    zstart = zstarts[b]
    nstart = nstarts[b]
    a32 = F32(a)
    ca = F32((1.0 - 2.0 * a) / a) if a > 0 else F32(0.0)
    inb = F32(inv_nb)
    s32 = F32(sigma)
    sc = F32(scale)
    vm = F32(vmax)

    acc = 0.0
    sat = 0
    first = max(r0 - 1, 0)
    last = min(r1, h - 1)
    o_frame = offsets[b]
    for r in range(first, last + 1):
        # ---- projection: row r's Poisson means (kernels.render, ss = 1)
        Y = oy + (r + 0.5) * by - y0
        yy = F32(Y * Y + dd * dd)
        if kappa > 0.0:  # the lobe is smooth: exact every LOBE_STEP pixels, linear between
            yf = F32(Y * fwd[1] + dd * fwd[2])
            for ia in range(na):
                Xa = min(Xa0 + iota[ia] * astep, alast)  # (iota: a float index converts without a scalar step)
                lob[ia] = kf * ((Xa * f0 + yf) / math.sqrt(Xa * Xa + yy) - one)
            _exp(lob, bits, bitsf, na)
            for ia in range(w // LOBE_STEP):
                l0 = lob[ia]
                dl = (lob[ia + 1] - l0) * quarter
                for s in range(LOBE_STEP):
                    ex[ia * LOBE_STEP + s] = l0 + F32(s) * dl
            ia = jl0 // LOBE_STEP
            for j in range(jl0, w):
                t = F32((j - jl0) / (jl1 - jl0)) if jl1 > jl0 else zero
                ex[j] = lob[ia] + (lob[ia + 1] - lob[ia]) * t
        else:
            for j in range(w):
                ex[j] = one
        yn = F32(Y * normal[1] + dd * normal[2])
        for j in range(w):  # the backscatter background and solid angle
            X = X0f + iota[j] * bxf
            inv = one / math.sqrt(X * X + yy)
            rz = ddf * inv
            rn = (X * n0 + yn) * inv
            shadow = min(rn * inv_rim, one)
            fb = shadow * (k_cos * rn + k_lobe * ex[j]) * rz * rz * rz
            bgf[j] = fb if rn > zero else zero
        rd = 0.0
        if g1 >= 0:  # the master pattern and the band modulation
            _nodes(bank_n, face0[p1], full[p1], M1[0, 0] * X0 + M1[0, 1] * Y + M1[0, 2] * dd, M1[0, 0] * bx,
                   M1[1, 0] * X0 + M1[1, 1] * Y + M1[1, 2] * dd, M1[1, 0] * bx,
                   M1[2, 0] * X0 + M1[2, 1] * Y + M1[2, 2] * dd, M1[2, 0] * bx, iw, q, tx, ty)
            _gather(bank, bank_n, offs[p1], scales[p1], q, tx, ty, val)
            if w2 > 0.0:
                _nodes(bank_n, face0[p2], full[p2], M2[0, 0] * X0 + M2[0, 1] * Y + M2[0, 2] * dd, M2[0, 0] * bx,
                       M2[1, 0] * X0 + M2[1, 1] * Y + M2[1, 2] * dd, M2[1, 0] * bx,
                       M2[2, 0] * X0 + M2[2, 1] * Y + M2[2, 2] * dd, M2[2, 0] * bx, iw, q, tx, ty)
                _gather(bank, bank_n, offs[p2], scales[p2], q, tx, ty, val2)
                for j in range(w):
                    val[j] = wa * val[j] + wb * val2[j]
            for j in range(w):
                lv = max(bgf[j] * (one + cf * val[j]), zero)
                lam[j] = lv
                rd += lv
        else:
            for j in range(w):
                lv = bgf[j]
                lam[j] = lv
                rd += lv
        if r >= r0 and r < r1:
            acc += rd

        # ---- deposit: detector.fast.deposit for row r, into S3 at (r % 3) * W
        base = r * w
        zg = U64((zstart + base) & _MASK)
        zp = U64((zstart + _SECOND_STREAM + base) & _MASK)
        s0 = U64((r % 3) * W)
        dim = zero  # (`&`, not `and`: a short-circuit is a branch, and a branch stops vectorisation)
        for cc in range(w):  # every pixel by the normal approximation (right from the cut up)
            L = lam[cc]
            cu = U64(cc)
            nn = max(np.floor(L + math.sqrt(L) * table[zp + cu] + half), zero)
            ri = one / math.sqrt(max(nn, one) * k32)
            g = max(one - ninth * ri * ri + third * table[zg + cu] * ri, zero)
            S3[s0 + cu] = nn * adu32 * g * g * g
            dim += one if ((L < cut) & (L > zero)) else zero
        if dim > zero:  # the dim pixels again, exactly
            m = 0
            for cc in range(w):
                lane[m] = U32(cc)
                m += 1 if ((lam[cc] < cut) & (lam[cc] > zero)) else 0
            for i in range(m):
                L = lam[lane[i]]
                Ls[i] = L
                ln[i] = -L
            _exp(ln, bits, bitsf, m)  # P(0)
            for i in range(m):
                xh = ukey + U64(base + lane[i]) * U64(_GOLD)
                xh = (xh ^ (xh >> U64(30))) * U64(_M1)
                xh = (xh ^ (xh >> U64(27))) * U64(_M2)
                xh = xh ^ (xh >> U64(31))
                lu[i] = F32(((xh >> U64(11)) + U64(1)) * _U53)
                lp[i] = ln[i]
                lf[i] = ln[i]
                lcnt[i] = zero
            mp = (m + _CHUNK - 1) // _CHUNK * _CHUNK  # whole vectors: no scalar remainder per count
            for i in range(m, mp):  # padding lanes, never drawn
                lu[i] = zero
                lp[i] = zero
                lf[i] = one
                Ls[i] = zero
                lcnt[i] = zero
            # inversion, n = #{k >= 1 : u > F(k - 1)}: the row's dim pixels a count at a time
            for kk in range(1, _MAX_COUNT + 1):
                inv_k = F32(1.0 / kk)
                more = zero  # a float sum vectorises with the rest; an integer one halves the width
                for i in range(mp):
                    up = lu[i] > lf[i]
                    lcnt[i] += one if up else zero
                    more += one if up else zero
                    lp[i] *= Ls[i] * inv_k
                    lf[i] += lp[i]
                if more == zero:
                    break
            for i in range(m):
                cc = lane[i]
                nn = lcnt[i]
                ri = one / math.sqrt(max(nn, one) * k32)
                g = max(one - ninth * ri * ri + third * table[zg + U64(cc)] * ri, zero)
                S3[s0 + U64(cc)] = nn * adu32 * g * g * g

        # ---- analogue: the row above, now that the deposits around it are there
        qr = r - 1
        if r == h - 1 and r1 == h:
            nq = 2  # the last row too: below it is itself
        else:
            nq = 1
        for iq in range(nq):
            if iq == 1:
                qr = h - 1
            if qr < r0 or qr >= r1:
                continue
            pu = qr - 1 if qr > 0 else qr
            pd = qr + 1 if qr < h - 1 else qr
            su = U64((pu % 3) * W)
            sr = U64((qr % 3) * W)
            sd = U64((pd % 3) * W)
            m0 = U64(qr * w)  # gain, offb: flat (h * w)
            z0 = U64((nstart + qr * w) & _MASK)
            oo = U64(o_frame + qr * w)
            if a > 0:
                for cc in range(w):
                    cu = U64(cc)
                    vrow[cc] = a32 * (S3[su + cu] + S3[sd + cu] + S3[sr + cu] * ca)
            else:
                for cc in range(w):
                    vrow[cc] = S3[sr + U64(cc)]
            if a > 0 and w > 2:
                for cc in range(1, w - 1):
                    cu = U64(cc)
                    y = (a32 * (vrow[cc - 1] + vrow[cc + 1] + vrow[cc] * ca) * inb * gain[m0 + cu] + offb[m0 + cu]
                         + s32 * table[z0 + cu]) * sc
                    sat += 1 if y >= vm else 0
                    out[oo + cu] = np.uint16(min(max(y, zero), vm) + half)
                sat += _analog_pixel(0, vrow, a, inv_nb, gain, offb, sigma, scale, vmax, table, m0, z0, zero, out, oo)
                sat += _analog_pixel(w - 1, vrow, a, inv_nb, gain, offb, sigma, scale, vmax, table, m0, z0, zero, out,
                                     oo)
            else:
                for cc in range(w):
                    sat += _analog_pixel(cc, vrow, a, inv_nb, gain, offb, sigma, scale, vmax, table, m0, z0, zero,
                                         out, oo)
            for hh in range(hot_ptr[qr], hot_ptr[qr + 1]):  # hot pixels: again, with their dark current
                cc = hot_col[hh]
                y = _analog_value(cc, vrow, a, inv_nb, gain, offb, sigma, scale, table, m0, z0, hot_val[hh])
                old = _analog_value(cc, vrow, a, inv_nb, gain, offb, sigma, scale, table, m0, z0, zero)
                sat += (1 if y >= vm else 0) - (1 if old >= vm else 0)
                out[oo + U64(cc)] = np.uint16(min(max(y, zero), vm) + half)
    return acc, sat


def _vmf(kappa: float) -> float:
    """The forward lobe's von Mises-Fisher normalisation (``kernels.render``'s ``vmf``)."""
    return kappa / (2.0 * math.pi * (1.0 - math.exp(-2.0 * kappa))) if kappa > 0 else 1.0 / (4.0 * math.pi)


@_njit(parallel=True)
def raw_frames(bank, bank_n, face0, full, offs, scales, mats, o1, o2, ph1, ph2, mix, contrast, es, finv, has_f,
               dx, dy, pc_lin, s2dT, ox, oy, bx, by, normal, fwd, lobe, kappa, vmf, t_nb,
               height, gain, offb, hot_ptr, hot_col, hot_val, adu, k, a, inv_nb, sigma, scale, vmax, table,
               ks, keys, zstarts, nstarts, band, out, offsets, dose, sats, s32, su32):
    """Raw frames of scan points ``ks`` (flat), frame b into ``out[offsets[b] : + h * w]`` (a
    flat uint16 array: a batch, or shared-memory slots), on numba's threads. ``bank``: the
    master cube maps, flat, faces ``bank_n`` square. ``keys``, ``zstarts``, ``nstarts``: each
    frame's noise (``detector.fast.frame_key(seed, serial, 1)`` and
    ``offsets(frame_key(seed, serial, 0))``). ``gain``, ``offb`` (offset + dark): flat maps of
    the grid; hot pixels by row (CSR, ADU already scaled to the frame). ``s32``, ``su32``:
    per-thread scratch (:func:`scratch`). Writes each frame's electrons (``dose``) and clipped
    pixels (``sats``) per band of ``band`` rows: ``(B, bands)``."""
    B = ks.shape[0]
    nbands = (height + band - 1) // band
    for task in nb.prange(B * nbands):
        b = task // nbands
        bi = task - b * nbands
        r0 = bi * band
        tid = nb.get_thread_id()
        acc, sat = _band(b, r0, min(r0 + band, height), s32[tid], su32[tid], bank, bank_n, face0, full, offs,
                         scales, mats, o1, o2, ph1, ph2, mix, contrast, es, finv, has_f, dx, dy, pc_lin, s2dT,
                         ox, oy, bx, by, normal, fwd, lobe, kappa, vmf, t_nb, height, gain, offb, hot_ptr,
                         hot_col, hot_val, adu, k, a, inv_nb, sigma, scale, vmax, table, ks, keys, zstarts,
                         nstarts, out, offsets)
        dose[b, bi] = acc
        sats[b, bi] = sat


@_njit()
def raw_frames_serial(bank, bank_n, face0, full, offs, scales, mats, o1, o2, ph1, ph2, mix, contrast, es, finv,
                      has_f, dx, dy, pc_lin, s2dT, ox, oy, bx, by, normal, fwd, lobe, kappa, vmf, t_nb,
                      height, gain, offb, hot_ptr, hot_col, hot_val, adu, k, a, inv_nb, sigma, scale, vmax, table,
                      ks, keys, zstarts, nstarts, out, offsets, dose, sats, d32, dui):
    """:func:`raw_frames` on the calling thread, whole frames, in its own scratch ``d32``,
    ``dui`` (one array each of :func:`scratch`'s lists); ``dose``, ``sats``: ``(B,)``."""
    for b in range(ks.shape[0]):
        acc, sat = _band(b, 0, height, d32, dui, bank, bank_n, face0, full, offs, scales, mats, o1, o2, ph1, ph2,
                         mix, contrast, es, finv, has_f, dx, dy, pc_lin, s2dT, ox, oy, bx, by, normal, fwd, lobe,
                         kappa, vmf, t_nb, height, gain, offb, hot_ptr, hot_col, hot_val, adu, k, a, inv_nb,
                         sigma, scale, vmax, table, ks, keys, zstarts, nstarts, out, offsets)
        dose[b] = acc
        sats[b] = sat


@_njit(parallel=True)
def means(bank, bank_n, face0, full, offs, scales, mats, o1, o2, ph1, ph2, mix, contrast, es, finv, has_f,
          dx, dy, pc_lin, s2dT, ox, oy, bx, by, normal, fwd, lobe, kappa, vmf, t_nb, ks, lam):
    """The Poisson means :func:`raw_frames` draws from, ``lam[b]`` for scan point ``ks[b]``
    (``kernels.render``'s ``lam`` with ``ss = 1``)."""
    B, h, w = lam.shape
    na = (w - 1) // LOBE_STEP + 2
    for task in nb.prange(B * h):
        b = task // h
        r = task - b * h
        M1 = np.zeros((3, 3))
        M2 = np.zeros((3, 3))
        w2, x0, y0, dd, c, e, p1, p2, g1 = _point(ks[b], mats, o1, o2, ph1, ph2, mix, contrast, es, finv, has_f,
                                                  dx, dy, pc_lin, s2dT, M1, M2)
        iota = np.arange(w).astype(np.float32)
        lob = np.empty(na, np.float32)
        bits = np.empty(na, np.int32)
        bitsf = bits.view(np.float32)
        f32 = np.empty((6, w), np.float32)
        q = np.empty(w, np.uint32)
        _project_row(r, bank, bank_n, face0, full, offs, scales, M1, M2, w2, x0, y0, dd, c, e, p1, p2, g1,
                     ox, oy, bx, by, normal, fwd, lobe, kappa, vmf, t_nb, iota, lob, bits, bitsf, f32[0], f32[1], f32[2],
                     f32[3], q, f32[4], f32[5], lam[b, r])
