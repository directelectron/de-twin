"""numba kernels for EBSD: the kinematic master-pattern sum and the per-pattern projection.

Masters are stored (and exchanged) on the square-Lambert projection below, and rendered from a
**cube map**: the gnomonic projection of the sphere onto the faces of a cube (face ``a, s``:
``c_a = s``, ``u = c_b / c_a``, ``v = c_c / c_a`` with ``(b, c)`` the other two axes in
order, ``Nf x Nf`` nodes over ``[-1, 1]^2``). Finding a direction's cube node takes one
reciprocal where the Lambert point takes a square root, a division and an arctangent, which
halves the lookup. A centrosymmetric master needs the three positive faces (a direction is
flipped onto them), a non-centrosymmetric one all six.

Square Lambert projection (Rosca-Lambert, as EMsoft stores master patterns) of the northern
hemisphere ``z >= 0`` onto ``[-L, L]^2``, ``L = sqrt(pi / 2)``, sampled at ``N x N`` nodes
(``linspace(-L, L, N)``, so the equator is the square's edge). Kinematic Kikuchi patterns are
centrosymmetric (``|F_g| = |F_-g|`` and every band is symmetric about its plane), so a
direction with ``z < 0`` reads the master at ``-direction``: one hemisphere holds the sphere.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from ..parallel import PARALLEL

SQRT_PI = math.sqrt(math.pi)
LAMBERT_L = math.sqrt(math.pi / 2.0)

#: The forward-scatter lobe is evaluated exactly every LOBE_STEP pixels along a row and
#: interpolated linearly between (relative error ~1e-4 at 512^2: the lobe is ~20 deg wide).
LOBE_STEP = 4

#: Band profile: an excess line just inside each edge and a deficiency just outside, at
#: ``EDGE_OFFSET`` band half-widths from it (dynamical-looking edges on a kinematic band).
EDGE_EXCESS = 0.25
EDGE_DEFICIENCY = 0.35
EDGE_OFFSET = 0.08
EDGE_MIN_SIGMA = 0.05


def _njit(**kw):
    if kw.get("parallel"):
        kw["parallel"] = PARALLEL
    return nb.njit(cache=True, nogil=True, fastmath=True, error_model="numpy", **kw)


# ---------------------------------------------------------------------- Lambert (numpy)
def lambert_directions(n: int) -> np.ndarray:
    """(n, n, 3) unit directions at the nodes of the square-Lambert northern hemisphere, row =
    Lambert Y, column = Lambert X."""
    L = LAMBERT_L
    c = np.linspace(-L, L, n)
    X, Y = np.meshgrid(c, c)
    out = np.empty((n, n, 3))
    a = np.abs(Y) <= np.abs(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        # |Y| <= |X|
        q = 2.0 * X / math.pi * np.sqrt(np.maximum(math.pi - X * X, 0.0))
        ang = np.where(X != 0, math.pi * Y / (4.0 * np.where(X != 0, X, 1.0)), 0.0)
        xa, ya, za = q * np.cos(ang), q * np.sin(ang), 1.0 - 2.0 * X * X / math.pi
        # |X| < |Y|
        q = 2.0 * Y / math.pi * np.sqrt(np.maximum(math.pi - Y * Y, 0.0))
        ang = np.where(Y != 0, math.pi * X / (4.0 * np.where(Y != 0, Y, 1.0)), 0.0)
        xb, yb, zb = q * np.sin(ang), q * np.cos(ang), 1.0 - 2.0 * Y * Y / math.pi
    out[..., 0] = np.where(a, xa, xb)
    out[..., 1] = np.where(a, ya, yb)
    out[..., 2] = np.where(a, za, zb)
    out[(X == 0) & (Y == 0)] = (0.0, 0.0, 1.0)
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out


def lambert_xy(v) -> tuple[np.ndarray, np.ndarray]:
    """Square-Lambert (X, Y) of unit directions (..., 3) (folded to z >= 0)."""
    v = np.asarray(v, np.float64)
    v = np.where(v[..., 2:3] < 0, -v, v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    q = np.sqrt(np.maximum(2.0 * (1.0 - z), 0.0))
    a = np.abs(y) <= np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = np.where(x >= 0, 1.0, -1.0)
        sy = np.where(y >= 0, 1.0, -1.0)
        Xa = sx * q * SQRT_PI / 2.0
        Ya = sx * q * 2.0 / SQRT_PI * np.arctan(np.where(x != 0, y / np.where(x != 0, x, 1.0), 0.0))
        Xb = sy * q * 2.0 / SQRT_PI * np.arctan(np.where(y != 0, x / np.where(y != 0, y, 1.0), 0.0))
        Yb = sy * q * SQRT_PI / 2.0
    return np.where(a, Xa, Xb), np.where(a, Ya, Yb)


# ---------------------------------------------------------------------- master sum
@_njit(parallel=True)
def accumulate_bands(dirs, normals, half_width, amp, sigma, sigma_edge, reach, out):
    """``out[p] = sum_k amp_k * P(|dirs_p . normals_k| / half_width_k)``: every band ``k`` a
    smoothed top hat of half-width ``sin(theta_B)`` (edge blur ``sigma``, in half-widths) with
    an excess line just inside each edge and a deficiency just outside (``sigma_edge``).
    ``reach``: |t| beyond which band k contributes nothing (skipped)."""
    P = dirs.shape[0]
    K = normals.shape[0]
    r2 = 1.0 / math.sqrt(2.0)
    for p in nb.prange(P):
        x, y, z = dirs[p, 0], dirs[p, 1], dirs[p, 2]
        acc = 0.0
        for k in range(K):
            u = abs(x * normals[k, 0] + y * normals[k, 1] + z * normals[k, 2])
            t = u / half_width[k]
            if t > reach[k]:
                continue
            s = sigma[k]
            band = 0.5 * (math.erf((1.0 - t) * r2 / s) + math.erf((1.0 + t) * r2 / s))
            se = sigma_edge[k]
            a = t - 1.0 + EDGE_OFFSET
            b = t - 1.0 - EDGE_OFFSET
            edge = EDGE_EXCESS * math.exp(-0.5 * a * a / (se * se)) - EDGE_DEFICIENCY * math.exp(
                -0.5 * b * b / (se * se)
            )
            acc += amp[k] * (band + edge)
        out[p] = acc


# ---------------------------------------------------------------------- projection
#: atan(x) = x P(x^2) on |x| <= 1, least-squares fit, max error 1.1e-8 rad (libm's scalar atan
#: was half the lookup's time).
_ATAN = (
    0.0028340989710117373,
    -0.016005224935549546,
    0.04258803010402406,
    -0.07495492338117586,
    0.10636782943784438,
    -0.14202580379049168,
    0.19992485341385735,
    -0.33333066920402804,
    0.9999999842746156,
)


@_njit(inline="always")
def _atan1(x):
    t = x * x
    p = _ATAN[0]
    for c in _ATAN[1:]:
        p = p * t + c
    return x * p


@_njit(inline="always")
def _lookup(bank, plane, scale, cx, cy, cz):
    """Bilinear code of ``bank[plane]`` at the square-Lambert point of unit direction
    (cx, cy, cz), cz >= 0 (indexed in place: no per-pixel array views)."""
    q = math.sqrt(max(2.0 * (1.0 - cz), 0.0))
    if abs(cy) <= abs(cx):
        if cx == 0.0:
            X, Y = 0.0, 0.0
        else:
            sg = 1.0 if cx > 0.0 else -1.0
            X = sg * q * (SQRT_PI * 0.5)
            Y = sg * q * (2.0 / SQRT_PI) * _atan1(cy / cx)
    else:
        sg = 1.0 if cy > 0.0 else -1.0
        X = sg * q * (2.0 / SQRT_PI) * _atan1(cx / cy)
        Y = sg * q * (SQRT_PI * 0.5)
    n = bank.shape[1]
    fx = (X + LAMBERT_L) * scale
    fy = (Y + LAMBERT_L) * scale
    ix = int(fx)
    iy = int(fy)
    if ix < 0:
        ix = 0
    elif ix > n - 2:
        ix = n - 2
    if iy < 0:
        iy = 0
    elif iy > n - 2:
        iy = n - 2
    tx = min(max(fx - ix, 0.0), 1.0)
    ty = min(max(fy - iy, 0.0), 1.0)
    v00 = np.float32(bank[plane, iy, ix])
    v01 = np.float32(bank[plane, iy, ix + 1])
    v10 = np.float32(bank[plane, iy + 1, ix])
    v11 = np.float32(bank[plane, iy + 1, ix + 1])
    return (v00 * (1.0 - tx) + v01 * tx) * (1.0 - ty) + (v10 * (1.0 - tx) + v11 * tx) * ty


@_njit(inline="always")
def _cube_lookup(bank, face0, full, cx, cy, cz):
    """Bilinear cube-map code at direction (cx, cy, cz) (any length): faces ``face0 ..`` of
    ``bank``; ``full``: six faces, else three and the direction is flipped onto them."""
    ax, ay, az = abs(cx), abs(cy), abs(cz)
    if ax >= ay and ax >= az:
        a, m, u, v = 0, cx, cy, cz
    elif ay >= az:
        a, m, u, v = 1, cy, cx, cz
    else:
        a, m, u, v = 2, cz, cx, cy
    f = face0 + a
    if m < 0.0 and full:
        f += 3
    r = 1.0 / m  # u / m is the same for the direction and its flip
    n = bank.shape[1]
    s = (n - 1) * 0.5
    fx = (u * r + 1.0) * s
    fy = (v * r + 1.0) * s
    ix = min(max(int(fx), 0), n - 2)
    iy = min(max(int(fy), 0), n - 2)
    tx = fx - ix
    ty = fy - iy
    v00 = np.float32(bank[f, iy, ix])
    v01 = np.float32(bank[f, iy, ix + 1])
    v10 = np.float32(bank[f, iy + 1, ix])
    v11 = np.float32(bank[f, iy + 1, ix + 1])
    return (v00 * (1.0 - tx) + v01 * tx) * (1.0 - ty) + (v10 * (1.0 - tx) + v11 * tx) * ty


def cube_directions(n: int, faces: int = 3) -> np.ndarray:
    """(faces, n, n, 3) unit directions at the cube-map nodes (row = v, column = u); faces
    0-2 are +x, +y, +z, 3-5 (``faces=6``) -x, -y, -z."""
    c = np.linspace(-1.0, 1.0, n)
    U, V = np.meshgrid(c, c)
    out = np.empty((faces, n, n, 3))
    for f in range(faces):
        a, sg = f % 3, (1.0 if f < 3 else -1.0)
        b, cc = [k for k in range(3) if k != a]
        out[f, ..., a] = sg
        out[f, ..., b] = U * sg
        out[f, ..., cc] = V * sg
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


@_njit(inline="always")
def _value(bank, face0, full, offs, scales, p, M, rx, ry, rz):
    """Standardised master value of phase ``p`` along detector direction r (unit) for the
    detector -> crystal matrix M (need not be orthogonal: the cube lookup is scale-free)."""
    cx = M[0, 0] * rx + M[0, 1] * ry + M[0, 2] * rz
    cy = M[1, 0] * rx + M[1, 1] * ry + M[1, 2] * rz
    cz = M[2, 0] * rx + M[2, 1] * ry + M[2, 2] * rz
    return offs[p] + scales[p] * _cube_lookup(bank, face0[p], full[p], cx, cy, cz)


@_njit()
def _detector_to_crystal(g, finv, has_f, s2dT, out):
    """out = g^T @ finv @ s2dT (g crystal -> sample, finv the inverse deformation gradient)."""
    a = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += (finv[i, k] if has_f else (1.0 if i == k else 0.0)) * s2dT[k, j]
            a[i, j] = acc
    for i in range(3):
        for j in range(3):
            acc = 0.0
            for k in range(3):
                acc += g[k, i] * a[k, j]
            out[i, j] = acc


@_njit(parallel=True)
def render(
    bank,
    face0,
    full,
    offs,
    scales,
    mats,
    o1,
    o2,
    ph1,
    ph2,
    mix,
    contrast,
    es,
    finv,
    has_f,
    dx,
    dy,
    pc_lin,
    s2dT,
    ks,
    ox,
    oy,
    bx,
    by,
    ss,
    raw,
    normal,
    fwd,
    lobe,
    kappa,
    t_nb,
    out,
    lam,
    p0,
    want_means,
):
    """EBSD patterns of scan points ``ks`` on a binned detector grid, in one threaded pass.

    Per point k: grain orientation ``mats[o1[k]]`` (crystal -> sample) of phase ``ph1[k]``, its
    neighbour ``mats[o2[k]]`` / ``ph2[k]`` mixed in with weight ``mix[k]``, the inverse
    deformation gradient ``finv[k]`` (sample frame; when ``has_f``), the beam offset
    ``(dx[k], dy[k])`` (µm, sample frame) that moves the pattern centre linearly
    (``pc_lin``: x0, dx0/ddx, dx0/ddy, y0, ..., dd, ... in sensor pixels), band modulation
    ``contrast[k]`` and backscattered electrons per second ``es[k]``.

    Pixel (i, j) of point m is the sensor area ``[ox + j bx, ox + (j + 1) bx) x [oy + i by, ...)``
    sampled at ``ss x ss`` points (supersampled binning). Its value (``raw``: the blended master
    value only) is electrons per sensor pixel per second::

        es ((1 - lobe) (r.n)/pi + lobe vMF(r; fwd, kappa)) dOmega (1 + contrast m)

    with ``dOmega = cos^3 / dd^2`` per sensor pixel, shadowed below the sample surface. With
    ``want_means`` it also writes the detector's Poisson means ``lam = out * t_nb`` and
    ``p0 = exp(-lam)`` (``detector.fast.prepare``'s contract) and returns each row's dose.
    """
    m_pts = ks.shape[0]
    h, w = out.shape[1], out.shape[2]
    rows = np.zeros(m_pts * h)
    vmf = kappa / (2.0 * math.pi * (1.0 - math.exp(-2.0 * kappa))) if kappa > 0 else 1.0 / (4.0 * math.pi)
    cut = 30.0
    inv_ss = 1.0 / ss
    for row in nb.prange(m_pts * h):
        mi = row // h
        i = row - mi * h
        k = ks[mi]
        M1 = np.empty((3, 3))
        M2 = np.empty((3, 3))
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
        inv_dd2 = 1.0 / (dd * dd)
        c = contrast[k] if g1 >= 0 else 0.0
        e = es[k]
        p1 = ph1[k]
        p2 = ph2[k]
        acc_row = 0.0
        anchors = ss == 1 and kappa > 0.0 and not raw
        if anchors:  # the lobe is smooth: exact every LOBE_STEP pixels, linear between
            na = (w - 1) // LOBE_STEP + 2
            lob = np.empty(na)
            Ya = oy + (i + 0.5) * by - y0
            for a in range(na):
                ja = min(a * LOBE_STEP, w - 1)
                Xa = ox + (ja + 0.5) * bx - x0
                ia = 1.0 / math.sqrt(Xa * Xa + Ya * Ya + dd * dd)
                rfa = (Xa * fwd[0] + Ya * fwd[1] + dd * fwd[2]) * ia
                lob[a] = math.exp(kappa * (rfa - 1.0))
        for j in range(w):
            tot = 0.0
            for sa in range(ss):
                Y = oy + (i + (sa + 0.5) * inv_ss) * by - y0
                for sb in range(ss):
                    X = ox + (j + (sb + 0.5) * inv_ss) * bx - x0
                    inv = 1.0 / math.sqrt(X * X + Y * Y + dd * dd)
                    rx, ry, rz = X * inv, Y * inv, dd * inv
                    val = 0.0
                    if g1 >= 0:
                        val = (1.0 - w2) * _value(bank, face0, full, offs, scales, p1, M1, rx, ry, rz)
                        if w2 > 0.0:
                            val += w2 * _value(bank, face0, full, offs, scales, p2, M2, rx, ry, rz)
                    if raw:
                        tot += val
                        continue
                    rn = rx * normal[0] + ry * normal[1] + rz * normal[2]
                    if rn <= 0.0:
                        continue
                    shadow = min(rn / 0.03, 1.0)
                    if anchors:
                        a = j // LOBE_STEP
                        j0 = a * LOBE_STEP
                        j1 = min(j0 + LOBE_STEP, w - 1)
                        t = (j - j0) / (j1 - j0) if j1 > j0 else 0.0
                        ex = lob[a] + (lob[a + 1] - lob[a]) * t
                    else:
                        rf = rx * fwd[0] + ry * fwd[1] + rz * fwd[2]
                        ex = math.exp(kappa * (rf - 1.0))
                    bg = (1.0 - lobe) * rn / math.pi + lobe * vmf * ex
                    v = e * shadow * bg * rz * rz * rz * inv_dd2 * (1.0 + c * val)
                    if v > 0.0:
                        tot += v
            v = tot * inv_ss * inv_ss
            out[mi, i, j] = v
            if want_means:
                lv = v * t_nb
                lam[mi, i, j] = lv
                p0[mi, i, j] = math.exp(np.float32(-lv)) if lv < cut else 0.0
                acc_row += lv
        rows[row] = acc_row
    return rows
