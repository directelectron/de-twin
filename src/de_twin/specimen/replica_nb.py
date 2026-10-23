"""numba kernels for the shadowed-replica calibration standards."""

from __future__ import annotations

import math
import numpy as np
from .raster_nb import add_thickness1, add_under1, grain_from_hash, level_offset1, njit, prange
try:

    AVAILABLE = True
except Exception:  # noqa: BLE001
    AVAILABLE = False


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
    from .noise import _site, fbm1, fbm1_grad, nearest_site_hash
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
