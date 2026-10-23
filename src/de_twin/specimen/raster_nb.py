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

if AVAILABLE:
    from .noise import cell1, fbm1

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