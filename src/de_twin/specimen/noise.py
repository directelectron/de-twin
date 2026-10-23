"""Smooth fractal noise for specimen textures (numba, with a NumPy fallback giving the same
values).

`fbm` sums octaves of smoothstep-interpolated value noise, each octave turned so the
lattices do not line up. Thresholded, it gives rounded organic shapes (island films); as a
height it gives a crumpled surface. Values are world-locked: a function of (seed, x, y) only.
"""

from __future__ import annotations

import math

import numpy as np

from ..parallel import numba_jit

try:
    import numba as nb

    AVAILABLE = True
except Exception:  # noqa: BLE001 - numba missing: the NumPy path
    nb = None
    AVAILABLE = False

_prange = nb.prange if AVAILABLE else range

#: Each octave is turned by this much (radians), cycling, so octave lattices do not align.
OCTAVE_TURN = np.array([0.0, 0.61, 1.37, 2.11, 2.83])
#: Each octave is this much finer than the one before.
LACUNARITY = 2.3

_KI = 0x9E3779B97F4A7C15
_KJ = 0xC2B2AE3D27D4EB4F
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_INV53 = 1.0 / 9007199254740992.0


_njit = numba_jit


@_njit()
def _corner(seed, i, j):
    z = seed ^ (np.uint64(i) * np.uint64(_KI)) ^ (np.uint64(j) * np.uint64(_KJ))
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
    z = z ^ (z >> np.uint64(31))
    return float(z >> np.uint64(11)) * _INV53


@_njit()
def fbm1(xv, yv, seeds, cs, sn, amps):
    """`fbm` at one point, from the octave parameters of :func:`fbm_params`."""
    acc = 0.0
    for o in range(seeds.size):
        u = xv * cs[o] - yv * sn[o]
        v = xv * sn[o] + yv * cs[o]
        fu, fv = math.floor(u), math.floor(v)
        tu, tv = u - fu, v - fv
        tu = tu * tu * (3.0 - 2.0 * tu)
        tv = tv * tv * (3.0 - 2.0 * tv)
        i, j = np.int64(fu), np.int64(fv)
        s = seeds[o]
        a = _corner(s, i, j)
        b = _corner(s, i + 1, j)
        c = _corner(s, i, j + 1)
        d = _corner(s, i + 1, j + 1)
        ab = a + (b - a) * tu
        cd = c + (d - c) * tu
        acc += amps[o] * (2.0 * (ab + (cd - ab) * tv) - 1.0)
    return acc


@_njit()
def fbm1_grad(xv, yv, seeds, cs, sn, amps):
    """`fbm1` and its analytic gradient (d/dx, d/dy, per unit of x and y): the same octaves,
    the smoothstep's derivative instead of finite differences (one evaluation, not three)."""
    acc = 0.0
    gx = 0.0
    gy = 0.0
    for o in range(seeds.size):
        u = xv * cs[o] - yv * sn[o]
        v = xv * sn[o] + yv * cs[o]
        fu, fv = math.floor(u), math.floor(v)
        tu, tv = u - fu, v - fv
        su = tu * tu * (3.0 - 2.0 * tu)
        sv = tv * tv * (3.0 - 2.0 * tv)
        du = 6.0 * tu * (1.0 - tu)
        dv = 6.0 * tv * (1.0 - tv)
        i, j = np.int64(fu), np.int64(fv)
        s = seeds[o]
        a = _corner(s, i, j)
        b = _corner(s, i + 1, j)
        c = _corner(s, i, j + 1)
        d = _corner(s, i + 1, j + 1)
        ab = a + (b - a) * su
        cd = c + (d - c) * su
        acc += amps[o] * (2.0 * (ab + (cd - ab) * sv) - 1.0)
        k = a - b - c + d
        dVu = ((b - a) + k * sv) * du
        dVv = ((c - a) + k * su) * dv
        # u = x cs - y sn, v = x sn + y cs
        gx += 2.0 * amps[o] * (dVu * cs[o] + dVv * sn[o])
        gy += 2.0 * amps[o] * (-dVu * sn[o] + dVv * cs[o])
    return acc, gx, gy


@_njit(parallel=True)
def _fbm_nb(x, y, out, seeds, cs, sn, amps):
    for k in _prange(x.size):
        out[k] = fbm1(x[k], y[k], seeds, cs, sn, amps)


def _corner_np(seed, i, j):
    with np.errstate(over="ignore"):
        z = np.uint64(seed) ^ (i.astype(np.uint64) * np.uint64(_KI)) ^ (j.astype(np.uint64) * np.uint64(_KJ))
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * _INV53


def _fbm_np(x, y, seeds, cs, sn, amps):
    out = np.zeros(x.shape)
    for o in range(seeds.size):
        u = x * cs[o] - y * sn[o]
        v = x * sn[o] + y * cs[o]
        fu, fv = np.floor(u), np.floor(v)
        tu, tv = u - fu, v - fv
        tu = tu * tu * (3.0 - 2.0 * tu)
        tv = tv * tv * (3.0 - 2.0 * tv)
        i, j = fu.astype(np.int64), fv.astype(np.int64)
        s = seeds[o]
        a, b = _corner_np(s, i, j), _corner_np(s, i + 1, j)
        c, d = _corner_np(s, i, j + 1), _corner_np(s, i + 1, j + 1)
        ab = a + (b - a) * tu
        cd = c + (d - c) * tu
        out += amps[o] * (2.0 * (ab + (cd - ab) * tv) - 1.0)
    return out


def fbm_params(seed: int, scale_um: float, octaves: int = 3, gain: float = 0.5):
    """(seeds, cos / scale, sin / scale, amplitudes) of `fbm`'s octaves (for :func:`fbm1`)."""
    octaves = max(int(octaves), 1)
    amps = gain ** np.arange(octaves, dtype=np.float64)
    amps /= amps.sum()
    scales = float(scale_um) / LACUNARITY ** np.arange(octaves)
    turn = OCTAVE_TURN[np.arange(octaves) % OCTAVE_TURN.size]
    cs, sn = np.cos(turn) / scales, np.sin(turn) / scales
    seeds = np.array([((int(seed) + 7919 * o) * _KJ) & 0xFFFFFFFFFFFFFFFF for o in range(octaves)], np.uint64)
    return seeds, cs, sn, amps


def fbm(seed: int, x_um, y_um, scale_um: float, octaves: int = 3, gain: float = 0.5) -> np.ndarray:
    """Band-limited noise in about [-1, 1]: *octaves* of smooth value noise from *scale_um*
    down (each `LACUNARITY` finer and *gain* weaker)."""
    x = np.ascontiguousarray(x_um, np.float64)
    y = np.ascontiguousarray(y_um, np.float64)
    shape = np.broadcast(x, y).shape
    x, y = np.broadcast_to(x, shape).ravel(), np.broadcast_to(y, shape).ravel()
    seeds, cs, sn, amps = fbm_params(seed, scale_um, octaves, gain)
    if x.size == 0:
        return np.zeros(shape)
    if AVAILABLE:
        out = np.empty(x.size)
        _fbm_nb(np.ascontiguousarray(x), np.ascontiguousarray(y), out, seeds, cs, sn, amps)
    else:
        out = _fbm_np(x, y, seeds, cs, sn, amps)
    return out.reshape(shape)


@_njit()
def _site(seed, i, j, jitter):
    """(x, y, hash) of the jittered nucleation site of cell (i, j), in cell units."""
    z = seed ^ (np.uint64(i) * np.uint64(_KI)) ^ (np.uint64(j) * np.uint64(_KJ))
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
    z = z ^ (z >> np.uint64(31))
    fx = float((z >> np.uint64(11)) & np.uint64(0xFFFFF)) / 1048576.0
    fy = float((z >> np.uint64(31)) & np.uint64(0xFFFFF)) / 1048576.0
    return i + 0.5 + jitter * (fx - 0.5), j + 0.5 + jitter * (fy - 0.5), z


@_njit()
def nearest_site_hash(xv, yv, seed, jitter):
    """The hash of the nearest jittered site to (xv, yv) (cell units): `cells`' third output."""
    ci, cj = np.int64(math.floor(xv)), np.int64(math.floor(yv))
    best = 1e30
    bh = np.uint64(0)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            sx, sy, h = _site(seed, ci + di, cj + dj, jitter)
            d = (sx - xv) ** 2 + (sy - yv) ** 2
            if d < best:
                best, bh = d, h
    return bh


@_njit()
def cell1(xv, yv, seed, merge, jitter):
    """`cells` at one point: (distance to the nearest site, gap, the site's hash)."""
    ci, cj = np.int64(math.floor(xv)), np.int64(math.floor(yv))
    best, bx, by = 1e30, 0.0, 0.0
    bh = np.uint64(0)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            sx, sy, h = _site(seed, ci + di, cj + dj, jitter)
            d = (sx - xv) ** 2 + (sy - yv) ** 2
            if d < best:
                best, bx, by, bh = d, sx, sy, h
    # distance to the nearest bisector with a neighbour the island has not merged with
    gap = 1e30
    for di in range(-2, 3):
        for dj in range(-2, 3):
            sx, sy, h = _site(seed, ci + di, cj + dj, jitter)
            if h == bh:
                continue
            mx, my = sx - bx, sy - by
            L = math.sqrt(mx * mx + my * my)
            if L < 1e-9:
                continue
            pair = (bh ^ h) * np.uint64(0x9E3779B97F4A7C15)
            u = float(pair >> np.uint64(40)) / 16777216.0
            if u < merge:
                continue  # the two islands have coalesced
            t = ((xv - 0.5 * (sx + bx)) * mx + (yv - 0.5 * (sy + by)) * my) / L
            if -t < gap:
                gap = -t
    return math.sqrt(best), gap, bh


@_njit(parallel=True)
def _cells_nb(x, y, seed, merge, jitter, d1o, gapo, ho):
    for k in _prange(x.size):
        d1o[k], gapo[k], ho[k] = cell1(x[k], y[k], seed, merge[k], jitter)


def cells(seed: int, x, y, merge) -> tuple:
    """Jittered-grid cells at (x, y) in cell units: (distance to the nearest site, distance to
    the nearest cell boundary that is not merged away, the nearest site's hash). A boundary
    between two cells is merged away when the pair's hash falls below *merge* (per pixel)."""
    shape = np.shape(x)
    x = np.ascontiguousarray(x, np.float64).ravel()
    y = np.ascontiguousarray(y, np.float64).ravel()
    m = np.ascontiguousarray(np.broadcast_to(np.asarray(merge, np.float64), shape)).ravel()
    d1, gap = np.empty(x.size), np.empty(x.size)
    h = np.empty(x.size, np.uint64)
    sd = np.uint64((int(seed) * _KJ) & 0xFFFFFFFFFFFFFFFF)
    _cells_nb(x, y, sd, m, 0.8, d1, gap, h)  # without numba: the same loop, interpreted (slow)
    return d1.reshape(shape), gap.reshape(shape), h.reshape(shape)


@_njit(parallel=True)
def _nearest_nb(x, y, seed, jitter, sx, sy, d2o, ho):
    for k in _prange(x.size):
        xv, yv = x[k], y[k]
        ci, cj = np.int64(math.floor(xv)), np.int64(math.floor(yv))
        best, bx, by = 1e30, 0.0, 0.0
        bh = np.uint64(0)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                px, py, h = _site(seed, ci + di, cj + dj, jitter)
                d = (px - xv) ** 2 + (py - yv) ** 2
                if d < best:
                    best, bx, by, bh = d, px, py, h
        sx[k], sy[k], d2o[k], ho[k] = bx, by, best, bh


class FastLattice:
    """A jittered lattice (`cell_um` apart, sites within `jitter` of their cell centre) whose
    nearest-site query runs in numba: ``nearest(x, y)`` -> object with ``sx, sy`` (um),
    ``d2`` (um^2) and ``h`` (uint64 site hash), like :class:`..geometry.JitteredLattice`."""

    def __init__(self, cell_um: float, seed: int, salt: int = 0, jitter: float = 0.5):
        self.cell_um = float(cell_um)
        self.seed = np.uint64(((int(seed) * 0x9E3779B1 + int(salt) * 0x85EBCA77) * _KJ) & 0xFFFFFFFFFFFFFFFF)
        self.jitter = float(min(max(jitter, 0.0), 0.9))

    def nearest(self, x, y):
        from types import SimpleNamespace

        shape = np.shape(x)
        c = self.cell_um
        xs = np.ascontiguousarray(np.asarray(x, np.float64).ravel() / c)
        ys = np.ascontiguousarray(np.asarray(y, np.float64).ravel() / c)
        n = xs.size
        sx, sy, d2 = np.empty(n), np.empty(n), np.empty(n)
        h = np.empty(n, np.uint64)
        if n:
            _nearest_nb(xs, ys, self.seed, self.jitter, sx, sy, d2, h)
        return SimpleNamespace(sx=(sx * c).reshape(shape), sy=(sy * c).reshape(shape),
                               d2=(d2 * c * c).reshape(shape), h=h.reshape(shape))
