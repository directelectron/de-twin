"""numba kernels that paint drawn primitives (patches) onto the specimen rasters."""

from __future__ import annotations

import math

import numpy as np

from .raster_nb import ELLIPSE, FLAT, RECT, RING, ROUGH_ELLIPSE, SPHERICAL, njit, prange

try:
    AVAILABLE = True
except Exception:
    AVAILABLE = False


# ------------------------------------------------------------------ drawn primitives
@njit()
def _inside_t(lx, ly, shape, profile, t0, inner, fn, fj, period, pin2, pout2):
    """(inside, thickness) of one primitive at a pixel (`raster._inside`, `raster._thickness`)."""
    r2 = lx * lx + ly * ly
    if shape == RECT:
        ins = abs(lx) <= 1.0 and abs(ly) <= 1.0
    elif shape == ELLIPSE:
        ins = r2 <= 1.0
    elif shape == RING:
        ins = r2 <= 1.0 and r2 >= inner * inner
    elif shape == ROUGH_ELLIPSE:
        ins = r2 <= pin2
        if not ins and r2 <= pout2:
            th = math.atan2(ly, lx)
            rm = 1.0 + 1.0 * (
                math.sin((th + 0.43) * period) / 15.0 + math.sin((th + 0.14) * period * 5) / 60.0
            )
            ins = math.sqrt(r2) <= rm
    else:  # FACETED star polygon
        n = max(fn, 3)
        phi = math.atan2(ly, lx) % (2.0 * np.pi)
        k = min(int(phi * n / (2.0 * np.pi)), n - 1)
        k1 = (k + 1) % n
        jk = fj[k]
        jk1 = fj[k1]
        tk = 2.0 * np.pi * k / n
        tk1 = 2.0 * np.pi * k1 / n
        ax, ay = jk * math.cos(tk), jk * math.sin(tk)
        bx, by = jk1 * math.cos(tk1), jk1 * math.sin(tk1)
        ins = (bx - ax) * (ly - ay) - (by - ay) * (lx - ax) >= 0.0
    if profile == FLAT:
        t = t0
    elif profile == SPHERICAL:
        t = t0 * math.sqrt(max(0.0, 1.0 - r2))
    else:  # WEDGE
        t = t0 * min(max(0.5 * (ly + 1.0), 0.0), 1.0)
    return ins, t


@njit(parallel=True)
def _patch_contributions(
    nx,
    co,
    r0s,
    c0s,
    hs,
    ws,
    shape,
    profile,
    t0s,
    inner,
    fn,
    fj,
    period,
    pin2,
    pout2,
    item_k,
    item_a,
    counts,
    starts,
    flat,
    tt,
    owner,
    count_only,
):
    """Every (primitive, window row) item in parallel: its inside pixels, in the order of the
    NumPy mask (primitive, row, column)."""
    for it in prange(item_k.shape[0]):
        k = item_k[it]
        a = item_a[it]
        A0, Ac, Ar, B0, Bc, Br = co[k, 0], co[k, 1], co[k, 2], co[k, 3], co[k, 4], co[k, 5]
        q = starts[it]
        cnt = 0
        rr = r0s[k] + a
        for b in range(ws[k]):
            cc = c0s[k] + b
            lx = A0 + Ac * float(cc) + Ar * float(rr)
            ly = B0 + Bc * float(cc) + Br * float(rr)
            ins, t = _inside_t(lx, ly, shape, profile, t0s[k], inner[k], fn[k], fj[k], period, pin2, pout2)
            if ins and t > 0.0:
                if not count_only:
                    flat[q + cnt] = rr * nx + cc
                    tt[q + cnt] = t
                    owner[q + cnt] = k
                cnt += 1
        counts[it] = cnt


@njit()
def _apply_sorted(thick, material, grain, flat, tt, owner, mats, grains, ows):
    """`RasterContext.apply` with pixels hit more than once: per pixel the contributions
    sorted by thickness (stable, as np.lexsort((t, flat))), summed in that order, the
    thickest (the last) claiming."""
    o1 = np.argsort(tt, kind="mergesort")
    o2 = np.argsort(flat[o1], kind="mergesort")
    order = o1[o2]
    n = order.shape[0]
    i = 0
    while i < n:
        f = flat[order[i]]
        j = i
        s = tt[order[i]]
        while j + 1 < n and flat[order[j + 1]] == f:
            j += 1
            s = s + tt[order[j]]
        w = order[j]
        twin = tt[w]
        existing = np.float64(thick[f])
        wasvac = material[f] == 0
        thick[f] = np.float32(min(max(existing + s, 0.0), 65535.0))
        k = owner[w]
        if wasvac or (ows[k] and twin >= existing):
            material[f] = mats[k]
        if grains[k] >= 0:
            grain[f] = grains[k]
        i = j + 1


@njit(parallel=True)
def _apply_unique(thick, material, grain, flat, tt, owner, mats, grains, ows):
    for i in prange(flat.shape[0]):
        f = flat[i]
        k = owner[i]
        t = tt[i]
        existing = np.float64(thick[f])
        wasvac = material[f] == 0
        thick[f] = np.float32(min(max(existing + t, 0.0), 65535.0))
        if wasvac or (ows[k] and t >= existing):
            material[f] = mats[k]
        if grains[k] >= 0:
            grain[f] = grains[k]


@njit(parallel=True)
def _all_unique(flat, mark):
    n = flat.shape[0]
    for i in prange(n):
        mark[flat[i]] = i
    bad = 0
    for i in prange(n):
        if mark[flat[i]] != i:
            bad += 1
    return bad == 0


def paint_patch(
    context, co, r0s, c0s, hs, ws, shape, profile, t0s, inner, fn, fj, mats, grains, ows, period, pin2, pout2
):
    """One `raster.iter_patches` stack of primitives, rasterised and composited like
    `paint_drawn` + `RasterContext.apply` (the same contributions, the same claim rule)."""
    item_k = np.repeat(np.arange(co.shape[0], dtype=np.int64), hs)
    item_a = (
        (np.arange(item_k.shape[0], dtype=np.int64) - np.repeat(np.cumsum(hs) - hs, hs))
        if item_k.size
        else np.zeros(0, np.int64)
    )
    n = item_k.shape[0]
    if n == 0:
        return
    counts = np.zeros(n, np.int64)
    starts = np.zeros(n, np.int64)
    dummy_i = np.zeros(1, np.int64)
    dummy_f = np.zeros(1)
    _patch_contributions(
        context.nx,
        co,
        r0s,
        c0s,
        hs,
        ws,
        shape,
        profile,
        t0s,
        inner,
        fn,
        fj,
        period,
        pin2,
        pout2,
        item_k,
        item_a,
        counts,
        starts,
        dummy_i,
        dummy_f,
        dummy_i,
        True,
    )
    starts[1:] = np.cumsum(counts)[:-1]
    total = int(counts.sum())
    if total == 0:
        return
    flat = np.empty(total, np.int64)
    tt = np.empty(total)
    owner = np.empty(total, np.int64)
    _patch_contributions(
        context.nx,
        co,
        r0s,
        c0s,
        hs,
        ws,
        shape,
        profile,
        t0s,
        inner,
        fn,
        fj,
        period,
        pin2,
        pout2,
        item_k,
        item_a,
        counts,
        starts,
        flat,
        tt,
        owner,
        False,
    )
    if _all_unique(flat, context._mark):
        _apply_unique(context.thick, context.material, context.grain, flat, tt, owner, mats, grains, ows)
    else:
        _apply_sorted(context.thick, context.material, context.grain, flat, tt, owner, mats, grains, ows)
