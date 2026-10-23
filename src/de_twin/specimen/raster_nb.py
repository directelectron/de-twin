"""numba kernels for the specimen rasteriser (:mod:`.raster` and the structures' fills).

Each kernel evaluates, pixel by pixel, the same float64 expressions in the same order as
the NumPy code it replaces, so point-sampled geometry comes out identical (transcendental
functions may differ from NumPy's vectorised ones in the last bit). The NumPy code stays as
the reference and the fallback (``AVAILABLE`` False without numba); tests compare the two.
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np

from ..hashing import splitmix64_kernel
from ..parallel import numba_jit

try:
    import numba as nb

    AVAILABLE = True
except Exception:  # noqa: BLE001
    AVAILABLE = False


njit = partial(numba_jit, fastmath=False, error_model="numpy")


prange = nb.prange if AVAILABLE else range

# Shape / Profile codes (raster.Shape, raster.Profile)
ELLIPSE, ROUGH_ELLIPSE, FACETED, RECT, RING = 0, 1, 2, 3, 4
FLAT, SPHERICAL, WEDGE, CURTAIN = 0, 1, 2, 3


@njit()
def _span(A0, Ac, Ar, B0, Bc, Br, row, rect, nx):
    """`raster.row_spans` for one row: inclusive column bounds (hi < lo: empty)."""
    a = A0 + Ar * row
    b = B0 + Br * row
    if rect:
        lo = -np.inf
        hi = np.inf
        for k in range(2):
            k0 = a if k == 0 else b
            kc = Ac if k == 0 else Bc
            if abs(kc) < 1e-300:
                if abs(k0) > 1.0:
                    lo = np.inf
            else:
                c1 = (-1.0 - k0) / kc
                c2 = (1.0 - k0) / kc
                lo = max(lo, min(c1, c2))
                hi = min(hi, max(c1, c2))
    else:
        qa = Ac * Ac + Bc * Bc
        qb = 2.0 * (a * Ac + b * Bc)
        qc = a * a + b * b - 1.0
        disc = qb * qb - 4.0 * qa * qc
        sq = math.sqrt(max(disc, 0.0))
        if disc >= 0:
            lo = (-qb - sq) / (2.0 * qa)
            hi = (-qb + sq) / (2.0 * qa)
        else:
            lo = np.inf
            hi = -np.inf
    lo = int(min(max(np.ceil(lo - 1e-9), 0.0), float(nx)))
    hi = int(min(max(np.floor(hi + 1e-9), -1.0), float(nx - 1)))
    return lo, hi


@njit(parallel=True)
def paint_large(thick, material, grain, nx, r0, r1, co, rect, ring_inner2, profile, t0, mat, gid, overwrite):
    """`raster.paint_large` for FLAT / SPHERICAL / WEDGE profiles: every row of the window's
    span, the primitive's thickness added and the material / grain claimed (`apply`)."""
    A0, Ac, Ar, B0, Bc, Br = co[0], co[1], co[2], co[3], co[4], co[5]
    for r in prange(r0, r1):
        lo, hi = _span(A0, Ac, Ar, B0, Bc, Br, float(r), rect, nx)
        for c in range(lo, hi + 1):
            lx = A0 + Ac * float(c) + Ar * float(r)
            ly = B0 + Bc * float(c) + Br * float(r)
            r2 = lx * lx + ly * ly
            if profile == FLAT:
                t = t0
            elif profile == SPHERICAL:
                t = t0 * math.sqrt(max(0.0, 1.0 - r2))
            else:
                t = t0 * min(max(0.5 * (ly + 1.0), 0.0), 1.0)
            if not t > 0.0:
                continue
            if ring_inner2 >= 0.0 and not r2 >= ring_inner2:
                continue
            f = r * nx + c
            existing = np.float64(thick[f])
            wasvac = material[f] == 0
            thick[f] = np.float32(min(max(existing + t, 0.0), 65535.0))
            if wasvac or (overwrite and t >= existing):
                material[f] = mat
            if gid >= 0:
                grain[f] = gid


# ------------------------------------------------------------------ shared scalar helpers
@njit()
def interp1(x, xp, fp):
    """``np.interp(x, xp, fp)`` for one x (NumPy's algorithm: last xp[j] <= x, exact nodes)."""
    n = xp.shape[0]
    if x < xp[0]:
        return fp[0]
    if x > xp[n - 1]:
        return fp[n - 1]
    lo = 0
    hi = n
    while lo < hi:  # last index with xp[j] <= x
        mid = lo + ((hi - lo) >> 1)
        if x >= xp[mid]:
            lo = mid + 1
        else:
            hi = mid
    j = lo - 1
    if j == n - 1:
        return fp[j]
    if xp[j] == x:
        return fp[j]
    slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    res = slope * (x - xp[j]) + fp[j]
    if np.isnan(res):
        res = slope * (x - xp[j + 1]) + fp[j + 1]
        if np.isnan(res) and fp[j] == fp[j + 1]:
            res = fp[j]
    return res


@njit()
def rint(x):
    """``np.rint``: to the nearest integer, halves to even."""
    r = math.floor(x + 0.5)
    if r - x == 0.5 and r % 2.0 != 0.0:
        r -= 1.0
    return r


@njit()
def grain_from_hash(material, h, gpm):
    """`fieldmap.grain_id_from_hash` for one hash: hash_seed(h, GRAIN, 0)."""
    k = splitmix64_kernel(splitmix64_kernel(splitmix64_kernel(splitmix64_kernel(h) ^ np.uint64(6)) ^ np.uint64(0)) ^ np.uint64(0))
    return np.int32(material * gpm + np.int64(k % np.uint64(gpm)))


@njit()
def add_thickness1(thick, f, nm):
    thick[f] = np.float32(min(max(np.float64(thick[f]) + nm, 0.0), 65535.0))


@njit()
def add_under1(uthick, umat, f, material, nm):
    v = np.float32(nm)
    uthick[f] = max(uthick[f] + v, np.float32(0.0))
    if v > 0:
        umat[f] = material


@njit()
def level_offset1(c, cov1, covh, covs, lvx, levels, table, offsets):
    """`stamps._level_offset` for one coverage: (family level, threshold offset). ``covs`` is
    ``cov[1:half + 1]``, ``lvx`` ``arange(1, half + 1)``, ``cov1, covh`` = ``cov[1], cov[half]``."""
    half = lvx.shape[0]
    nlev = levels.shape[0]
    lv = interp1(min(max(c, cov1), covh), covs, lvx)
    row = int(min(max(rint((lv - 1.0) / (half - 1.0) * (nlev - 1)), 0.0), float(nlev - 1)))
    return levels[row], interp1(c, table[row], offsets)


# ------------------------------------------------------------------ jittered-lattice labels
@njit()
def _label1(x, y, SX, SY, x0, y0, L, gw, small):
    """`JitteredLattice._label` at one point: the flat site-table index of the nearest site."""
    u = (x - x0) * (1.0 / L)
    v = (y - y0) * (1.0 / L)
    ci = math.floor(u)
    cj = math.floor(v)
    if small:
        if (u - ci) < 0.5:
            ci -= 1.0
        if (v - cj) < 0.5:
            cj -= 1.0
        n = 2
    else:
        ci -= 1.0
        cj -= 1.0
        n = 3
    base = np.int32(cj * gw + ci)
    xf = np.float32(u * L)
    yf = np.float32(v * L)
    best = np.float32(np.inf)
    bf = base
    for dj in range(n):
        for di in range(n):
            f = base + np.int32(dj * gw + di)
            ddx = SX[f] - xf
            ddy = SY[f] - yf
            d2 = ddx * ddx + ddy * ddy
            if d2 < best:
                best = d2
                bf = f
    return bf


@njit(parallel=True)
def block_labels(W, offx, offy, r0, r1, c0, c1, s, nr_n, nc_n, SX, SY, x0, y0, L, gw, small, out):
    """`JitteredLattice.block_labels`: labels on a node grid of stride ``s``; a block whose four
    corner nodes agree takes that label, the others are labelled per pixel (s < 2: all)."""
    H = r1 - r0
    Wd = c1 - c0
    coarse = s >= 2 and H >= 3 and Wd >= 3
    Ln = np.empty((max(nr_n, 1), max(nc_n, 1)), np.int32)
    if coarse:
        for a in prange(nr_n):
            r = float(r0 + s * a)
            for b in range(nc_n):
                c = float(c0 + s * b)
                X = W[0] + W[2] * c + W[4] * r - offx
                Y = W[1] + W[3] * c + W[5] * r - offy
                Ln[a, b] = _label1(X, Y, SX, SY, x0, y0, L, gw, small)
    for i in prange(H):
        r = float(r0 + i)
        for j in range(Wd):
            if coarse:
                a = i // s
                b = j // s
                lab = Ln[a, b]
                if Ln[a + 1, b] == lab and Ln[a, b + 1] == lab and Ln[a + 1, b + 1] == lab:
                    out[i, j] = lab
                    continue
            c = float(c0 + j)
            X = W[0] + W[2] * c + W[4] * r - offx
            Y = W[1] + W[3] * c + W[5] * r - offy
            out[i, j] = _label1(X, Y, SX, SY, x0, y0, L, gw, small)


# ------------------------------------------------------------------ grid bars
@njit(parallel=True)
def mesh_bulk(r0, r1, nx, LC, W, pitch, half, corner, disk2, usable2, bar_mat, bar_t, origin, ncell, cell_area,
              mat, thick, area):
    """`Holder._sample_bulk` of a mesh / waffle grid: bars, rounded holes and the placement
    area of each hole, for rows r0:r1."""
    inv = 1.0 / pitch
    for i in prange(r1 - r0):
        r = float(r0 + i)
        for c in range(nx):
            cf = float(c)
            lx = LC[0] + LC[1] * cf + LC[2] * r
            ly = LC[3] + LC[4] * cf + LC[5] * r
            qx = math.floor(lx * inv + 0.5)
            qy = math.floor(ly * inv + 0.5)
            u = abs(lx - qx * pitch)
            v = abs(ly - qy * pitch)
            hole = u < half and v < half
            if corner > 0:
                ic = half - corner
                du = max(u - ic, 0.0)
                dv = max(v - ic, 0.0)
                hole = hole and (du * du + dv * dv) < corner ** 2
            X = W[0] + W[2] * cf + W[4] * r
            Y = W[1] + W[3] * cf + W[5] * r
            r2 = X * X + Y * Y
            in_disk = r2 <= disk2
            hole = hole and r2 <= usable2
            bar = in_disk and not hole
            mat[i, c] = bar_mat if bar else np.uint8(0)
            thick[i, c] = bar_t if bar else np.float32(0.0)
            ci = qx - origin
            cj = qy - origin
            if hole and ci >= 0 and ci < ncell and cj >= 0 and cj < ncell:
                area[i, c] = cell_area[int(cj), int(ci)]
            else:
                area[i, c] = -1


# ------------------------------------------------------------------ protein fields
@njit(parallel=True)
def protein_block(thick, material, grain, nx, ra, rb, c0, c1, W, O, labels, SX, SY, x0, y0, diam, stained,
                  has_stain, thickness, fade, stain_nm, inner, maxr, mat_id, stain_mat):
    """`ProteinFieldStructure.fill` for one row block (no crystal patches): each pixel's
    nearest blob (``labels`` into the site table), its projected sphere, the negative-stain
    rim, and the claim."""
    for r in prange(ra, rb):
        i = r - ra
        for c in range(c0, c1):
            X = W[0] + W[2] * float(c) + W[4] * float(r)
            Y = W[1] + W[3] * float(c) + W[5] * float(r)
            dx0 = X - O[0]
            dy0 = Y - O[1]
            lx = dx0 * O[2] - dy0 * O[3]
            ly = dx0 * O[3] + dy0 * O[2]
            if not (abs(lx) <= O[4] and abs(ly) <= O[5]):
                continue
            site = labels[i, c - c0]
            dx = (X - x0) - np.float64(SX[site])
            dy = (Y - y0) - np.float64(SY[site])
            d2 = dx * dx + dy * dy
            rad = 0.5 * diam[site] / 1000.0
            m = min(rad, maxr)
            if not d2 < m * m:
                continue
            rn = math.sqrt(d2) / rad
            add = thickness * math.sqrt(max(0.0, 1.0 - rn * rn)) * fade
            mat = mat_id
            if has_stain and rn > inner and stained[site]:
                add = add + stain_nm
                mat = stain_mat
            if not add > 0:
                continue
            f = r * nx + c
            existing = np.float64(thick[f])
            wasvac = material[f] == 0
            add_thickness1(thick, f, add)
            if wasvac or add >= existing:
                grain[f] = -1
                material[f] = mat


# ------------------------------------------------------------------ support films
@njit(parallel=True)
def film_block(thick, material, area, nx, r0, r1, same, film_ok, film_nm, removed, has_removed, holey, LC, p,
               hole_r2, ice, ahx, ahy, acx, acy, W, ice_grad, noise_mode, lodg, G, gix, gtx, giy, gty, narr,
               film_mat):
    """`Holder._film_block` over rows r0:r1: the support film's thickness added where the
    placement area carries film (holey perforations, the ice meniscus and the film's
    granularity noise), vacuum claimed by the film material."""
    for r in prange(r0, r1):
        for c in range(nx):
            f = r * nx + c
            if same >= 0:
                ai = same
            else:
                a = area[f]
                if a < 0:
                    continue
                ai = a
                if not film_ok[ai]:
                    continue
                if has_removed and removed[(r - r0) * nx + c]:
                    continue
            t = film_nm[ai]
            if holey:
                lx = LC[0] + LC[1] * float(c) + LC[2] * float(r)
                ly = LC[3] + LC[4] * float(c) + LC[5] * float(r)
                hu = lx - math.floor(lx / p + 0.5) * p
                hv = ly - math.floor(ly / p + 0.5) * p
                if not (hu * hu + hv * hv >= hole_r2):
                    continue
            if ice:
                X = W[0] + W[2] * float(c) + W[4] * float(r)
                Y = W[1] + W[3] * float(c) + W[5] * float(r)
                hx = ahx[ai] if ahx[ai] > 0 else 1.0
                hy = ahy[ai] if ahy[ai] > 0 else 1.0
                d = min(max(math.hypot((X - acx[ai]) / hx, (Y - acy[ai]) / hy), 0.0), 1.0)
                t = t * (1.0 + ice_grad * d * d)
            if noise_mode == 1:  # value_noise_separable: along y on the lattice rows, then x
                i = r - r0
                j0 = giy[i]
                k = gix[c]
                g0a = G[j0, k]
                g1a = G[j0 + 1, k]
                ra = g0a + (g1a - g0a) * gty[i]
                g0b = G[j0, k + 1]
                g1b = G[j0 + 1, k + 1]
                rb = g0b + (g1b - g0b) * gty[i]
                t = t + lodg * ((ra + (rb - ra) * gtx[c]) - 0.5)
            elif noise_mode == 2:
                t = t + lodg * (narr[(r - r0) * nx + c] - 0.5)
            t = max(t, 0.0)
            add_thickness1(thick, f, t)
            if material[f] == 0:
                material[f] = film_mat


# ------------------------------------------------------------------ polycrystalline films
@njit(parallel=True)
def polycrystal_film(thick, material, grain, nx, ra, rb, c0, c1, W, O, S1, C1, N1, A1, S2, C2, N2, A2,
                     L, amp, cseed, groove, grain_nm, groove_w, base_material, gpm):
    """`PolycrystalFilmStructure.fill`: warped cells, grooved boundaries, one grain each."""
    for r in prange(ra, rb):
        for c in range(c0, c1):
            X = W[0] + W[2] * float(c) + W[4] * float(r)
            Y = W[1] + W[3] * float(c) + W[5] * float(r)
            dx = X - O[0]
            dy = Y - O[1]
            lx = dx * O[2] - dy * O[3]
            ly = dx * O[3] + dy * O[2]
            if not (abs(lx) <= O[4] and abs(ly) <= O[5]):
                continue
            f = r * nx + c
            wx = lx + amp * fbm1(lx, ly, S1, C1, N1, A1)
            wy = ly + amp * fbm1(lx, ly, S2, C2, N2, A2)
            _, gap, h = cell1(wx / L, wy / L, cseed, 0.0, 0.8)
            g = gap * grain_nm / groove_w
            add_thickness1(thick, f, -(groove * math.exp(-0.5 * (g * g))))
            grain[f] = grain_from_hash(base_material, h, gpm)
            material[f] = base_material



# ------------------------------------------------------------------ fast replica helpers
@njit()
def _lookup_pow2(tile, u, v, mask):
    """`stamps._lookup` for a power-of-two tile: masks instead of integer modulo."""
    fu = math.floor(u)
    fv = math.floor(v)
    tu = u - fu
    tv = v - fv
    i0 = np.int64(fu) & mask
    j0 = np.int64(fv) & mask
    i1 = (i0 + 1) & mask
    j1 = (j0 + 1) & mask
    a = tile[j0, i0] + (tile[j0, i1] - tile[j0, i0]) * tu
    b = tile[j1, i0] + (tile[j1, i1] - tile[j1, i0]) * tu
    return a + (b - a) * tv


@njit()
def periodic4(stack, fu, fv):
    """The four channels of `stamps.periodic1` at one point, the indices computed once."""
    n = stack.shape[1]
    u = fu * n - 0.5
    v = fv * n - 0.5
    a = math.floor(u)
    b = math.floor(v)
    tu = u - a
    tv = v - b
    i0 = int(a) % n
    j0 = int(b) % n
    i1 = i0 + 1 if i0 + 1 < n else 0
    j1 = j0 + 1 if j0 + 1 < n else 0
    out = np.empty(4, np.float32)
    for c in range(4):
        top = stack[c, j0, i0] + (stack[c, j0, i1] - stack[c, j0, i0]) * tu
        bot = stack[c, j1, i0] + (stack[c, j1, i1] - stack[c, j1, i0]) * tu
        out[c] = np.float32(top + (bot - top) * tv)
    return out[0], out[1], out[2], out[3]


@njit()
def field1_fast(xk, yk, levk, tiles, seed, px_per_um, cells_per_um):
    """`stamps.field1` with sharp = 6 as products and masked lookups (power-of-two tiles)."""
    nlev = tiles.shape[0]
    nvar = tiles.shape[1]
    mask = tiles.shape[2] - 1
    cx = xk * cells_per_um
    cy = yk * cells_per_um
    ci = math.floor(cx - 0.5)
    cj = math.floor(cy - 0.5)
    fx = cx - 0.5 - ci
    fy = cy - 0.5 - cj
    lv = min(max(levk, 0.0), nlev - 1.0)
    l0 = int(math.floor(lv))
    l1 = min(l0 + 1, nlev - 1)
    tl = lv - l0
    acc = 0.0
    wsum = 0.0
    uu = xk * px_per_um
    vv = yk * px_per_um
    for dj in range(2):
        for di in range(2):
            w = (fx if di else 1.0 - fx) * (fy if dj else 1.0 - fy)
            w2 = w * w
            w = w2 * w2 * w2
            h = _stamp_hash(seed, ci + di, cj + dj)
            var = int(h % np.uint64(nvar))
            d8 = int((h >> np.uint64(8)) & np.uint64(7))
            ou = float((h >> np.uint64(16)) & np.uint64(0xFFFF)) / 65536.0 * 256.0
            ov = float((h >> np.uint64(32)) & np.uint64(0xFFFF)) / 65536.0 * 256.0
            u = uu
            v = vv
            if d8 & 1:
                u = -u
            if d8 & 2:
                u, v = v, u
            if d8 & 4:
                u, v = -u, -v
            f0 = _lookup_pow2(tiles[l0, var], u + ou, v + ov, mask)
            f1 = _lookup_pow2(tiles[l1, var], u + ou, v + ov, mask)
            acc += w * (f0 + (f1 - f0) * tl)
            wsum += w
    return acc / wsum


@njit()
def nearest_site_hash3(xv, yv, seed, jitter):
    """`noise.nearest_site_hash` over the 3 x 3 cells around the point (the 5 x 5 search's
    answer except in rare corner configurations of a 0.8 jitter)."""
    ci = np.int64(math.floor(xv))
    cj = np.int64(math.floor(yv))
    best = 1e30
    bh = np.uint64(0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            sx, sy, h = _site(seed, ci + di, cj + dj, jitter)
            d = (sx - xv) ** 2 + (sy - yv) ** 2
            if d < best:
                best, bh = d, h
    return bh


def level_offset_lut(cov, half, levels, table, offsets, n=8192):
    """(c grid, family level, threshold offset) of `stamps._level_offset` on n coverages, for
    linear lookup (:func:`level_offset_fast`)."""
    c = np.linspace(1e-4, 1.0 - 1e-4, n)
    lv = np.interp(np.clip(c, cov[1], cov[half]), cov[1:half + 1], np.arange(1, half + 1, dtype=np.float64))
    nlev = len(levels)
    row = np.clip(np.rint((lv - 1.0) / (half - 1.0) * (nlev - 1)).astype(np.int64), 0, nlev - 1)
    off = np.array([np.interp(ci, table[r], offsets) for ci, r in zip(c, row)])
    return c, levels[row].astype(np.float64), off


@njit()
def level_offset_fast(c, lut_c, lut_lev, lut_off):
    n = lut_c.shape[0]
    x = (c - lut_c[0]) / (lut_c[n - 1] - lut_c[0]) * (n - 1)
    i = int(x)
    if i < 0:
        i = 0
    if i > n - 2:
        i = n - 2
    t = x - i
    lev = lut_lev[i] if t < 0.5 else lut_lev[i + 1]
    return lev, lut_off[i] + (lut_off[i + 1] - lut_off[i]) * t


# ------------------------------------------------------------------ shadowed replicas
if AVAILABLE:
    from .noise import _site, cell1, fbm1, fbm1_grad, nearest_site_hash
    from .spheres import occlusion1, top_chord1
    from .stamps import _hash_nb as _stamp_hash
    from .stamps import field1, periodic1


@njit()
def _h_texture(xx, yy, FS, FC, FSN, FA, FN, crumple_on, rough_term, crumple_nm, rough_nm):
    out = 0.0
    if crumple_on:
        out = out + crumple_nm * fbm1(xx, yy, FS[4, :FN[4]], FC[4, :FN[4]], FSN[4, :FN[4]], FA[4, :FN[4]])
    if rough_term:
        out = out + rough_nm * fbm1(xx, yy, FS[5, :FN[5]], FC[5, :FN[5]], FSN[5, :FN[5]], FA[5, :FN[5]])
    return out


@njit()
def _node_lxly(W, O, r, c):
    X = W[0] + W[2] * float(c) + W[4] * float(r)
    Y = W[1] + W[3] * float(c) + W[5] * float(r)
    dx = X - O[0]
    dy = Y - O[1]
    return dx * O[2] - dy * O[3], dx * O[3] + dy * O[2]


@njit(parallel=True)
def replica_nodes(ra, rb, c0, c1, W, O, FS, FC, FSN, FA, FN, R, TX, ST, OFF, NW, NE, NC, NR):
    """The replica's aperiodic noise (the lines' waviness and edge jitter, the crumple, the
    roughness with its slope) on node grids of strides ``ST`` (raster pixels, one per field,
    each a fraction of its finest octave's wavelength), for bilinear lookup per pixel: the
    noise is smooth on those scales, so a pixel need not evaluate every octave itself."""
    wavy, edge, across = R[0], R[1], R[3]
    crumple_nm, rough_nm = TX[0], TX[1]
    for q in range(4):
        s = ST[q]
        grid = NW if q == 0 else (NE if q == 1 else (NC if q == 2 else NR))
        nr = grid.shape[0]
        nc = grid.shape[1]
        for a in prange(nr):
            r = ra - OFF[q, 0] + s * a  # nodes on the world-fixed pixel lattice's multiples of s
            for b in range(nc):
                lx, ly = _node_lxly(W, O, r, c0 - OFF[q, 1] + s * b)
                if q == 0:
                    grid[a, b, 0] = wavy * fbm1(across * lx, ly, FS[0, :FN[0]], FC[0, :FN[0]], FSN[0, :FN[0]], FA[0, :FN[0]])
                    grid[a, b, 1] = wavy * fbm1(lx, across * ly, FS[1, :FN[1]], FC[1, :FN[1]], FSN[1, :FN[1]], FA[1, :FN[1]])
                elif q == 1:
                    grid[a, b, 0] = edge * fbm1(0.3 * lx, ly, FS[2, :FN[2]], FC[2, :FN[2]], FSN[2, :FN[2]], FA[2, :FN[2]])
                    grid[a, b, 1] = edge * fbm1(lx, 0.3 * ly, FS[3, :FN[3]], FC[3, :FN[3]], FSN[3, :FN[3]], FA[3, :FN[3]])
                else:
                    k = 4 if q == 2 else 5
                    amp = crumple_nm if q == 2 else rough_nm
                    v_, dx_, dy_ = fbm1_grad(lx, ly, FS[k, :FN[k]], FC[k, :FN[k]], FSN[k, :FN[k]], FA[k, :FN[k]])
                    grid[a, b, 0] = amp * v_
                    grid[a, b, 1] = amp * dx_
                    grid[a, b, 2] = amp * dy_


@njit()
def _bilin(grid, s, i, j, ch):
    """Bilinear lookup of channel ``ch`` of a node grid of stride ``s`` at pixel offset (i, j)."""
    a = i // s
    b = j // s
    if s == 1:
        return grid[a, b, ch]
    fa = (i - a * s) / s
    fb = (j - b * s) / s
    top = grid[a, b, ch] + (grid[a, b + 1, ch] - grid[a, b, ch]) * fb
    bot = grid[a + 1, b, ch] + (grid[a + 1, b + 1, ch] - grid[a + 1, b, ch]) * fb
    return top + (bot - top) * fa


@njit(parallel=True)
def replica_block(thick, material, grain, uthick, umat, used, nx, ra, rb, c0, c1, W, O, FL, R, stack,
                  FS, FC, FSN, FA, FN, TX, scx, scy, sr, sstart_all, smem, SPB, SNB, BSO, bstep, D, M, I, tiles,
                  fseed, cseed, covs, lvx, levels, table, offsets, ST, OFF, NW, NE, NC, NR, LUTC, LUTL, LUTO):
    """`ShadowedReplicaStructure.fill` for the raster rows ra:rb, columns c0:c1 of one owner
    (the relief, the replica's texture, latex spheres, the angled metal deposit and its
    island film; see the NumPy code in `standards.replica` for the physics)."""
    has_relief, crumple_on, rough_term, has_spheres, shadowed, metal_on, islands, wavy_on, edge_on, analytic = (
        FL[0], FL[1], FL[2], FL[3], FL[4], FL[5], FL[6], FL[7], FL[8], FL[9])
    wavy, edge, P, across = R[0], R[1], R[2], R[3]
    crumple_nm, rough_nm, e = TX[0], TX[1], TX[2]
    metal_nm, sin_e, cos_e, ux, uy, flat, leak, base, mean_c, ceq = (
        D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8], D[9])
    metal, sphere_mat, carbon_id, gpm = M[0], M[1], M[2], M[3]
    coverage, px_per_um, cells_per_um, sharp, G, cov1, covh = I[0], I[1], I[2], I[3], I[4], I[5], I[6]
    for r in prange(ra, rb):
        for c in range(c0, c1):
            X = W[0] + W[2] * float(c) + W[4] * float(r)
            Y = W[1] + W[3] * float(c) + W[5] * float(r)
            dx = X - O[0]
            dy = Y - O[1]
            lx = dx * O[2] - dy * O[3]
            ly = dx * O[3] + dy * O[2]
            if not (abs(lx) <= O[4] and abs(ly) <= O[5]):
                continue
            f = r * nx + c
            # the relief (looked up) and the replica's own surface (noise)
            ni = r - ra
            nj = c - c0
            if has_relief and analytic:  # the warp from its node grids (stride 1: evaluated here)
                u = lx
                v = ly
                if wavy_on:
                    if ST[0] > 1:
                        u = u + _bilin(NW, ST[0], ni + OFF[0, 0], nj + OFF[0, 1], 0)
                        v = v + _bilin(NW, ST[0], ni + OFF[0, 0], nj + OFF[0, 1], 1)
                    else:
                        u = u + wavy * fbm1(across * lx, ly, FS[0, :FN[0]], FC[0, :FN[0]], FSN[0, :FN[0]], FA[0, :FN[0]])
                        v = v + wavy * fbm1(lx, across * ly, FS[1, :FN[1]], FC[1, :FN[1]], FSN[1, :FN[1]], FA[1, :FN[1]])
                if edge_on:
                    if ST[1] > 1:
                        u = u + _bilin(NE, ST[1], ni + OFF[1, 0], nj + OFF[1, 1], 0)
                        v = v + _bilin(NE, ST[1], ni + OFF[1, 0], nj + OFF[1, 1], 1)
                    else:
                        u = u + edge * fbm1(0.3 * lx, ly, FS[2, :FN[2]], FC[2, :FN[2]], FSN[2, :FN[2]], FA[2, :FN[2]])
                        v = v + edge * fbm1(lx, 0.3 * ly, FS[3, :FN[3]], FC[3, :FN[3]], FSN[3, :FN[3]], FA[3, :FN[3]])
                h32, gx32, gy32, lit32 = periodic4(stack, u / P, v / P)
            elif has_relief:
                u = lx
                v = ly
                if wavy_on:
                    u = u + wavy * fbm1(across * lx, ly, FS[0, :FN[0]], FC[0, :FN[0]], FSN[0, :FN[0]], FA[0, :FN[0]])
                    v = v + wavy * fbm1(lx, across * ly, FS[1, :FN[1]], FC[1, :FN[1]], FSN[1, :FN[1]], FA[1, :FN[1]])
                if edge_on:
                    u = u + edge * fbm1(0.3 * lx, ly, FS[2, :FN[2]], FC[2, :FN[2]], FSN[2, :FN[2]], FA[2, :FN[2]])
                    v = v + edge * fbm1(lx, 0.3 * ly, FS[3, :FN[3]], FC[3, :FN[3]], FSN[3, :FN[3]], FA[3, :FN[3]])
                fu = u / P
                fv = v / P
                h32 = periodic1(stack, 0, fu, fv)
                gx32 = periodic1(stack, 1, fu, fv)
                gy32 = periodic1(stack, 2, fu, fv)
                lit32 = periodic1(stack, 3, fu, fv)
            else:
                h32 = np.float32(0.0)
                gx32 = np.float32(0.0)
                gy32 = np.float32(0.0)
                lit32 = np.float32(1.0)
            if analytic:  # the texture's slope from the noise's own derivative
                h0 = 0.0
                tgx = 0.0
                tgy = 0.0
                if crumple_on:
                    if ST[2] > 1:
                        h0 += _bilin(NC, ST[2], ni + OFF[2, 0], nj + OFF[2, 1], 0)
                        tgx += _bilin(NC, ST[2], ni + OFF[2, 0], nj + OFF[2, 1], 1)
                        tgy += _bilin(NC, ST[2], ni + OFF[2, 0], nj + OFF[2, 1], 2)
                    else:
                        v_, dx_, dy_ = fbm1_grad(lx, ly, FS[4, :FN[4]], FC[4, :FN[4]], FSN[4, :FN[4]], FA[4, :FN[4]])
                        h0 += crumple_nm * v_
                        tgx += crumple_nm * dx_
                        tgy += crumple_nm * dy_
                if rough_term:
                    if ST[3] > 1:
                        h0 += _bilin(NR, ST[3], ni + OFF[3, 0], nj + OFF[3, 1], 0)
                        tgx += _bilin(NR, ST[3], ni + OFF[3, 0], nj + OFF[3, 1], 1)
                        tgy += _bilin(NR, ST[3], ni + OFF[3, 0], nj + OFF[3, 1], 2)
                    else:
                        v_, dx_, dy_ = fbm1_grad(lx, ly, FS[5, :FN[5]], FC[5, :FN[5]], FSN[5, :FN[5]], FA[5, :FN[5]])
                        h0 += rough_nm * v_
                        tgx += rough_nm * dx_
                        tgy += rough_nm * dy_
                h = np.float64(h32) + h0
                gx = np.float64(gx32) + tgx / 1000.0  # nm per um of x -> nm / nm
                gy = np.float64(gy32) + tgy / 1000.0
            else:  # finite differences over e (the NumPy reference)
                h0 = _h_texture(lx, ly, FS, FC, FSN, FA, FN, crumple_on, rough_term, crumple_nm, rough_nm)
                hx = _h_texture(lx + e, ly, FS, FC, FSN, FA, FN, crumple_on, rough_term, crumple_nm, rough_nm)
                hy = _h_texture(lx, ly + e, FS, FC, FSN, FA, FN, crumple_on, rough_term, crumple_nm, rough_nm)
                h = np.float64(h32) + h0
                gx = np.float64(gx32) + (hx - h0) / (e * 1000.0)
                gy = np.float64(gy32) + (hy - h0) / (e * 1000.0)
            top = 0.0
            chord = 0.0
            if has_spheres:  # the sphere set of this pixel's row block (as the NumPy fill builds them)
                kb = (r - ra) // bstep
                SP = SPB[kb]
                SN = SNB[kb]
                sstart = sstart_all[BSO[kb, 0]:BSO[kb, 1]]
                t_, c_ = top_chord1(lx, ly, scx, scy, sr, sstart, smem, SP[0], SP[1], SP[2], SN[0], SN[1])
                top = t_ * 1000.0
                chord = c_ * 1000.0
                if top > 0 and shadowed:
                    step, dn = SP[3], SP[4]
                    tx = top_chord1(lx + step, ly, scx, scy, sr, sstart, smem, SP[0], SP[1], SP[2], SN[0], SN[1])[0] * 1000.0
                    ty = top_chord1(lx, ly + step, scx, scy, sr, sstart, smem, SP[0], SP[1], SP[2], SN[0], SN[1])[0] * 1000.0
                    gx = min(max((tx - top) / dn, -8.0), 8.0)
                    gy = min(max((ty - top) / dn, -8.0), 8.0)
                    h = h + top
            dep = 0.0
            if metal_on:
                dep = metal_nm * max(sin_e - cos_e * (gx * ux + gy * uy), 0.0)
                dep = min(dep, 12.0 * flat)
                if has_spheres and shadowed:
                    kb = (r - ra) // bstep
                    SP = SPB[kb]
                    SN = SNB[kb]
                    sstart = sstart_all[BSO[kb, 0]:BSO[kb, 1]]
                    occ = occlusion1(lx, ly, h / 1000.0, scx, scy, sr, sstart, smem, SP[0], SP[1], SP[2], SN[0],
                                     SN[1], SP[5], SP[6], SP[7], SP[8], SP[9], SP[10])
                    lit = min(np.float64(lit32), 1.0 - occ)
                    dep = dep * lit + leak * flat * (1.0 - lit)
                else:  # lit is float32 here, as in the NumPy code
                    term = np.float32(leak * flat) * (np.float32(1.0) - lit32)
                    dep = dep * np.float64(lit32) + np.float64(term)
            # layers: the film (carbon) with latex on it, the metal on top
            add_thickness1(thick, f, base - mean_c)
            latex = chord > 0
            if latex:
                add_thickness1(thick, f, chord - base)
                material[f] = sphere_mat
                add_under1(uthick, umat, f, carbon_id, base)
                used[0] = 1
            if not metal_on:
                continue
            gid = np.int32(-1)
            if islands and analytic:
                rel = dep / max(flat, 1e-9)
                cc = min(max(coverage * rel ** 0.8, 0.0), 0.9)
                lev, off = level_offset_fast(min(max(cc, 1e-4), 1.0 - 1e-4), LUTC, LUTL, LUTO)
                fval = field1_fast(lx, ly, lev, tiles, fseed, px_per_um, cells_per_um) + off
                inside = fval > 0.0 and dep > 0
                t_m = dep / max(cc, 1e-3) if inside else 0.0
                if inside:
                    gid = grain_from_hash(metal, nearest_site_hash3(lx / G, ly / G, cseed, 0.8), gpm)
            elif islands:
                rel = dep / max(flat, 1e-9)
                cc = min(max(coverage * rel ** 0.8, 0.0), 0.9)
                lev, off = level_offset1(min(max(cc, 1e-4), 1.0 - 1e-4), cov1, covh, covs, lvx, levels, table,
                                         offsets)
                fval = field1(lx, ly, lev, tiles, fseed, px_per_um, cells_per_um, sharp) + off
                inside = fval > 0.0 and dep > 0
                t_m = dep / max(cc, 1e-3) if inside else 0.0
                if inside:
                    gid = grain_from_hash(metal, nearest_site_hash(lx / G, ly / G, cseed, 0.8), gpm)
            else:
                inside = dep > 0
                t_m = dep
            if inside:
                below = np.float64(thick[f])
                add_thickness1(thick, f, t_m - below)
                if not latex:
                    add_under1(uthick, umat, f, carbon_id, below)
                else:
                    add_under1(uthick, umat, f, sphere_mat, chord + base * ceq)
                used[0] = 1
                grain[f] = gid
                material[f] = metal


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
            rm = 1.0 + 1.0 * (math.sin((th + 0.43) * period) / 15.0 + math.sin((th + 0.14) * period * 5) / 60.0)
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
def _patch_contributions(nx, co, r0s, c0s, hs, ws, shape, profile, t0s, inner, fn, fj, period, pin2, pout2,
                         item_k, item_a, counts, starts, flat, tt, owner, count_only):
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
        tw = tt[w]
        existing = np.float64(thick[f])
        wasvac = material[f] == 0
        thick[f] = np.float32(min(max(existing + s, 0.0), 65535.0))
        k = owner[w]
        if wasvac or (ows[k] and tw >= existing):
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


def paint_patch(ctx, co, r0s, c0s, hs, ws, shape, profile, t0s, inner, fn, fj, mats, grains, ows, period, pin2,
                pout2):
    """One `raster.iter_patches` stack of primitives, rasterised and composited like
    `paint_drawn` + `RasterContext.apply` (the same contributions, the same claim rule)."""
    item_k = np.repeat(np.arange(co.shape[0], dtype=np.int64), hs)
    item_a = (np.arange(item_k.shape[0], dtype=np.int64)
              - np.repeat(np.cumsum(hs) - hs, hs)) if item_k.size else np.zeros(0, np.int64)
    n = item_k.shape[0]
    if n == 0:
        return
    counts = np.zeros(n, np.int64)
    starts = np.zeros(n, np.int64)
    dummy_i = np.zeros(1, np.int64)
    dummy_f = np.zeros(1)
    _patch_contributions(ctx.nx, co, r0s, c0s, hs, ws, shape, profile, t0s, inner, fn, fj, period, pin2, pout2,
                         item_k, item_a, counts, starts, dummy_i, dummy_f, dummy_i, True)
    starts[1:] = np.cumsum(counts)[:-1]
    total = int(counts.sum())
    if total == 0:
        return
    flat = np.empty(total, np.int64)
    tt = np.empty(total)
    owner = np.empty(total, np.int64)
    _patch_contributions(ctx.nx, co, r0s, c0s, hs, ws, shape, profile, t0s, inner, fn, fj, period, pin2, pout2,
                         item_k, item_a, counts, starts, flat, tt, owner, False)
    if _all_unique(flat, ctx._mark):
        _apply_unique(ctx.thick, ctx.material, ctx.grain, flat, tt, owner, mats, grains, ows)
    else:
        _apply_sorted(ctx.thick, ctx.material, ctx.grain, flat, tt, owner, mats, grains, ows)
