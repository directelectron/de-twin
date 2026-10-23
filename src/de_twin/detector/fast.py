"""Fused numba kernels for the integrating detector: one frame in two parallel passes.

The numpy path (`detector._Ctx`) makes ~20 whole-image passes per frame, each a memory
round trip; at 4096² that is ~170 ms a frame. These kernels do the same physics per pixel
in registers, row by row:

0. per EXPOSURE (`prepare`): each pixel's mean electrons and exp(-mean), which every frame
   of the exposure starts its Poisson draw from;
1. `deposit`: electrons ~ Poisson(mean) (exact inversion below 30, normal above), then the
   ADU they deposit, S | N ~ Gamma(N k, adu / k) by Wilson-Hilferty (the numpy path's own
   approximation) with its factors from a table for small N;
2. `analog_out` / `analog`: 3x3 separable charge sharing of S (nearest edge), gain,
   offset, dark current, hot pixels, read noise, then clip and round (`finish` bins).

Corners cut, all invisible in an image:

* Uniforms come from a counter hash (SplitMix64 of seed, frame, stage, pixel), so a frame
  is the same whatever the thread schedule and a seed still reproduces a sequence.
* Normals come from a table of 4M, read from a random offset per frame and stage: a value
  recurs only 4M pixels apart (1024 rows on a 4096-wide sensor) and never in the same
  place in two frames. It removes a Box-Muller log/cos/sqrt from every pixel.

The streams differ from the numpy path's, so a frame is statistically, not bitwise, the
numpy one; `DE_TWIN_NUMPY_DETECTOR=1` selects that path for comparison.

Everything is written inline, with row views: numba does not reliably inline calls between
jitted functions, and the calls alone cost 3x.
"""

from __future__ import annotations

import math

import numpy as np

from ..parallel import numba_jit

try:
    import numba as nb

    AVAILABLE = True
except Exception:  # noqa: BLE001 - numba missing: the numpy path remains
    nb = None
    AVAILABLE = False

#: Below this mean the Poisson draw is exact inversion; above, the normal approximation.
POISSON_EXACT_BELOW = 30.0
#: Electrons up to which the gamma factor comes from a table rather than a sqrt.
GAMMA_LUT_N = 256
#: The normal table's size (a power of two, so an index is a mask).
NORMALS = 1 << 22
_MASK = NORMALS - 1

_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_GOLD = 0x9E3779B97F4A7C15
_U53 = 1.0 / 9007199254740994.0


_njit = numba_jit


_TABLE = None


def normal_table() -> np.ndarray:
    """The shared table of standard normals, drawn once."""
    global _TABLE
    if _TABLE is None:
        _TABLE = np.random.default_rng(0x5EED).standard_normal(NORMALS).astype(np.float32)
    return _TABLE


def gamma_lut(k: float) -> np.ndarray:
    """``[1 - 1/(9 n k), 1/(3 sqrt(n k))]`` for n = 0..GAMMA_LUT_N-1 (Wilson-Hilferty)."""
    n = np.maximum(np.arange(GAMMA_LUT_N, dtype=np.float64), 1.0) * k
    return np.stack([1.0 - 1.0 / (9.0 * n), 1.0 / (3.0 * np.sqrt(n))], 1).astype(np.float32)


@_njit(parallel=True)
def prepare(flux, t_nb, lam, p0):
    """Per exposure: mean electrons per grid pixel and exp(-mean). Returns the total."""
    h, w = flux.shape
    t32 = np.float32(t_nb)
    cut = np.float32(POISSON_EXACT_BELOW)
    rows = np.zeros(h)
    for r in nb.prange(h):
        fr = flux[r]
        lr = lam[r]
        pr = p0[r]
        acc = 0.0
        for c in range(w):
            v = fr[c] * t32
            if v < 0:
                v = np.float32(0.0)
            lr[c] = v
            pr[c] = math.exp(-v) if v < cut else np.float32(0.0)
            acc += v
        rows[r] = acc
    return rows.sum()


@_njit(parallel=True)
def deposit(lam, p0, adu, k, key, S, table, zstart, lut):
    """Phase 1: S = deposited ADU per grid pixel, from the exposure's *lam* and *p0*."""
    h, w = lam.shape
    adu32 = np.float32(adu)
    k32 = np.float32(k)
    cut = np.float32(POISSON_EXACT_BELOW)
    key = np.uint64(key)
    for r in nb.prange(h):
        lr = lam[r]
        pr = p0[r]
        sr = S[r]
        base = r * w
        for c in range(w):
            L = lr[c]
            if L <= 0:
                sr[c] = np.float32(0.0)
                continue
            i = base + c
            if L < cut:
                # SplitMix64 of the pixel's counter -> a uniform in (0, 1)
                x = key + np.uint64(i) * np.uint64(_GOLD)
                x = (x ^ (x >> np.uint64(30))) * np.uint64(_M1)
                x = (x ^ (x >> np.uint64(27))) * np.uint64(_M2)
                x = x ^ (x >> np.uint64(31))
                u = np.float32(((x >> np.uint64(11)) + np.uint64(1)) * _U53)
                p = pr[c]
                f = p
                n = 0
                while u > f and n < 200:
                    n += 1
                    p *= L / np.float32(n)
                    f += p
            else:
                zz = table[(zstart + 2654435761 + i) & _MASK]
                n = int(math.floor(L + math.sqrt(L) * zz + np.float32(0.5)))
                if n < 0:
                    n = 0
            if n == 0:
                sr[c] = np.float32(0.0)
                continue
            z = table[(zstart + i) & _MASK]
            if n < GAMMA_LUT_N:
                g = lut[n, 0] + z * lut[n, 1]
            else:
                a = np.float32(n) * k32
                g = np.float32(1.0) - np.float32(1.0) / (np.float32(9.0) * a) \
                    + z / (np.float32(3.0) * math.sqrt(a))
            if g < 0:
                g = np.float32(0.0)
            sr[c] = np.float32(n) * adu32 * g * g * g


@_njit(parallel=True)
def analog_out(S, has_signal, a, inv_nb, gain, off, base, read_sigma, scale, table, nstart,
               hot, vmax, out):
    """Phase 2 for an UNBINNED frame, straight into the output: charge sharing, gain,
    offset, dark, hot pixels (*hot*: ADU to add, zero almost everywhere), read noise,
    clip, round. Returns the clipped-pixel count."""
    h, w = gain.shape
    a32 = np.float32(a)
    ca = np.float32((1.0 - 2.0 * a) / a) if a > 0 else np.float32(0.0)
    inb = np.float32(inv_nb)
    b32 = np.float32(base)
    s32 = np.float32(read_sigma)
    sc = np.float32(scale)
    vm = np.float32(vmax)
    sats = np.zeros(h, np.int64)
    for r in nb.prange(h):
        g = gain[r]
        o = off[r]
        hr = hot[r]
        orow = out[r]
        v = np.empty(w, np.float32)
        if has_signal:
            Sr = S[r]
            if a > 0:
                Su = S[r - 1 if r > 0 else 0]
                Sd = S[r + 1 if r < h - 1 else h - 1]
                for c in range(w):
                    v[c] = a32 * (Su[c] + Sd[c] + Sr[c] * ca)
            else:
                for c in range(w):
                    v[c] = Sr[c]
        idx0 = nstart + r * w
        sat = 0
        for c in range(w):
            if has_signal:
                if a > 0:
                    x = a32 * (v[c - 1 if c > 0 else 0] + v[c + 1 if c < w - 1 else w - 1]
                               + v[c] * ca)
                else:
                    x = v[c]
                x = x * inb * g[c] + o[c]
            else:
                x = o[c]
            x = (x + b32 + hr[c] + s32 * table[(idx0 + c) & _MASK]) * sc
            if x >= vm:
                sat += 1
                x = vm
            elif x < 0:
                x = np.float32(0.0)
            orow[c] = x + np.float32(0.5)
        sats[r] = sat
    return sats.sum()


@_njit(parallel=True)
def analog(S, has_signal, a, inv_nb, gain, off, base, read_sigma, scale, table, nstart, x):
    """Phase 2 up to (not including) clipping, into the float grid *x* (binned frames)."""
    h, w = gain.shape
    a32 = np.float32(a)
    ca = np.float32((1.0 - 2.0 * a) / a) if a > 0 else np.float32(0.0)
    inb = np.float32(inv_nb)
    b32 = np.float32(base)
    s32 = np.float32(read_sigma)
    sc = np.float32(scale)
    for r in nb.prange(h):
        g = gain[r]
        o = off[r]
        xr = x[r]
        v = np.empty(w, np.float32)
        if has_signal:
            Sr = S[r]
            if a > 0:
                Su = S[r - 1 if r > 0 else 0]
                Sd = S[r + 1 if r < h - 1 else h - 1]
                for c in range(w):
                    v[c] = a32 * (Su[c] + Sd[c] + Sr[c] * ca)
            else:
                for c in range(w):
                    v[c] = Sr[c]
        idx0 = nstart + r * w
        for c in range(w):
            if has_signal:
                if a > 0:
                    y = a32 * (v[c - 1 if c > 0 else 0] + v[c + 1 if c < w - 1 else w - 1]
                               + v[c] * ca)
                else:
                    y = v[c]
                y = y * inb * g[c] + o[c]
            else:
                y = o[c]
            xr[c] = (y + b32 + s32 * table[(idx0 + c) & _MASK]) * sc


@_njit(parallel=True)
def finish(x, vmax, bx, by, out):
    """Clip, bin (average) and round into *out*; returns the clipped-pixel count."""
    oh, ow = out.shape
    vm = np.float32(vmax)
    inv = np.float32(1.0 / (bx * by))
    sats = np.zeros(oh, np.int64)
    for orow in nb.prange(oh):
        o = out[orow]
        sat = 0
        for oc in range(ow):
            acc = np.float32(0.0)
            for dy in range(by):
                xr = x[orow * by + dy]
                for dx in range(bx):
                    v = xr[oc * bx + dx]
                    if v >= vm:
                        sat += 1
                        v = vm
                    elif v < 0:
                        v = np.float32(0.0)
                    acc += v
            o[oc] = acc * inv + np.float32(0.5)
        sats[orow] = sat
    return sats.sum()


def _mix_py(x: int) -> int:
    m = (1 << 64) - 1
    x = (x ^ (x >> 30)) * _M1 & m
    x = (x ^ (x >> 27)) * _M2 & m
    return x ^ (x >> 31)


def frame_key(seed: int, frame_index: int, stage: int) -> int:
    """The counter key of one frame's stage."""
    k = ((int(seed) & 0xFFFFFFFF) << 32) ^ ((int(frame_index) & 0xFFFFFFF) << 4) ^ (stage & 0xF)
    return _mix_py(k)


def offsets(key: int) -> tuple[int, int]:
    """This frame's two starting points in the normal table (gamma, read noise)."""
    return int(_mix_py(key ^ 0xA5A5) % NORMALS), int(_mix_py(key ^ 0x5A5A) % NORMALS)


# ---------------------------------------------------------------- counting cameras
# A counting camera sums ``ncyc`` sensor cycles per frame; a pixel counts at most one
# event per cycle and nearby events coincide (paralysable, ``cluster_area_px``). Per
# EXPOSURE (`counting_prepare`) that is one Poisson mean per pixel; per FRAME it is one
# Poisson draw, capped at the cycles summed.


@_njit(parallel=True)
def counting_prepare(flux, has_flux, gain, t_nb, area, cyc_nb, false_rate, hot, mean, p0):
    """Per exposure: each pixel's mean counts over the frame and exp(-mean).
    Returns the electrons delivered (the dose)."""
    h, w = gain.shape
    t32 = np.float32(t_nb)
    cut = np.float32(POISSON_EXACT_BELOW)
    kk = np.float32(area / cyc_nb)  # lam A per cycle, over lam per frame
    inv_a = np.float32(1.0 / area)
    fr = np.float32(false_rate)
    cyc = np.float32(cyc_nb)
    rows = np.zeros(h)
    for r in nb.prange(h):
        g = gain[r]
        hr = hot[r]
        mr = mean[r]
        pr = p0[r]
        acc = 0.0
        for c in range(w):
            p = fr + hr[c]
            if has_flux:
                lam = flux[r, c] * t32
                if lam < 0:
                    lam = np.float32(0.0)
                acc += lam
                p += -math.expm1(-lam * g[c] * kk) * inv_a
            if p > 1:
                p = np.float32(1.0)
            m = p * cyc
            mr[c] = m
            pr[c] = math.exp(-m) if m < cut else np.float32(0.0)
        rows[r] = acc
    return rows.sum()


@_njit(parallel=True)
def count(mean, p0, key, table, zstart, cap, inv_nb, vmax, out, xout, to_float):
    """Per frame: counts ~ Poisson(mean), capped at *cap* cycles, averaged over the
    *inv_nb* binned pixels. Into the uint8 *out* (clipped, rounded; returns the clipped
    count) or, with *to_float*, into the float grid *xout* for `finish` to bin."""
    h, w = mean.shape
    cut = np.float32(POISSON_EXACT_BELOW)
    key = np.uint64(key)
    inb = np.float32(inv_nb)
    vm = np.float32(vmax)
    sats = np.zeros(h, np.int64)
    for r in nb.prange(h):
        mr = mean[r]
        pr = p0[r]
        base = r * w
        sat = 0
        for c in range(w):
            L = mr[c]
            n = 0
            if L > 0:
                i = base + c
                if L < cut:
                    x = key + np.uint64(i) * np.uint64(_GOLD)
                    x = (x ^ (x >> np.uint64(30))) * np.uint64(_M1)
                    x = (x ^ (x >> np.uint64(27))) * np.uint64(_M2)
                    x = x ^ (x >> np.uint64(31))
                    u = np.float32(((x >> np.uint64(11)) + np.uint64(1)) * _U53)
                    p = pr[c]
                    f = p
                    while u > f and n < 200:
                        n += 1
                        p *= L / np.float32(n)
                        f += p
                else:
                    zz = table[(zstart + i) & _MASK]
                    n = int(math.floor(L + math.sqrt(L) * zz + np.float32(0.5)))
                    if n < 0:
                        n = 0
                if n > cap:
                    n = cap
            v = np.float32(n) * inb
            if to_float:
                xout[r, c] = v
                continue
            if v >= vm:
                sat += 1
                v = vm
            out[r, c] = v + np.float32(0.5)
        sats[r] = sat
    return sats.sum()


@_njit(parallel=True)
def count_super(mean, p0, key, table, zstart, cap, vmax, out):
    """Per frame, super-resolution: as `count`, and each event lands in one of its
    pixel's 2x2 sub-pixels, uniformly at random. *out* is (2h, 2w) uint8.

    The corner cut: where in the pixel an event landed is drawn, not modelled. The
    rendered flux carries no detail finer than a sensor pixel, so a uniform draw is
    what a real super-resolution readout of the same image would show."""
    h, w = mean.shape
    cut = np.float32(POISSON_EXACT_BELOW)
    key = np.uint64(key)
    sub = key ^ np.uint64(0x5B5B5B5B5B5B5B5B)
    vm = int(vmax)
    sats = np.zeros(h, np.int64)
    for r in nb.prange(h):
        o0 = out[2 * r]
        o1 = out[2 * r + 1]
        for c in range(2 * w):
            o0[c] = 0
            o1[c] = 0
        mr = mean[r]
        pr = p0[r]
        base = r * w
        sat = 0
        for c in range(w):
            L = mr[c]
            if L <= 0:
                continue
            i = base + c
            n = 0
            if L < cut:
                x = key + np.uint64(i) * np.uint64(_GOLD)
                x = (x ^ (x >> np.uint64(30))) * np.uint64(_M1)
                x = (x ^ (x >> np.uint64(27))) * np.uint64(_M2)
                x = x ^ (x >> np.uint64(31))
                u = np.float32(((x >> np.uint64(11)) + np.uint64(1)) * _U53)
                p = pr[c]
                f = p
                while u > f and n < 200:
                    n += 1
                    p *= L / np.float32(n)
                    f += p
            else:
                zz = table[(zstart + i) & _MASK]
                n = int(math.floor(L + math.sqrt(L) * zz + np.float32(0.5)))
                if n < 0:
                    n = 0
            if n > cap:
                n = cap
            if n == 0:
                continue
            y = sub + np.uint64(i) * np.uint64(_GOLD)
            y = (y ^ (y >> np.uint64(30))) * np.uint64(_M1)
            y = (y ^ (y >> np.uint64(27))) * np.uint64(_M2)
            y = y ^ (y >> np.uint64(31))
            for j in range(n):
                q = int((y >> np.uint64((2 * j) & 63)) & np.uint64(3))
                row = o0 if (q >> 1) == 0 else o1
                cc = 2 * c + (q & 1)
                if row[cc] >= vm:
                    sat += 1
                else:
                    row[cc] += 1
        sats[r] = sat
    return sats.sum()
