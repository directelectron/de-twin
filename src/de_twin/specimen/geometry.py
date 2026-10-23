"""Vectorised geometry and hash-driven primitives (port of VirtualSpecimen ``Geometry.cpp``).

Everything here works on numpy arrays of world coordinates (micrometres). The per-pixel hot
paths of the C++ (row caches, ``NearestCached``) become "hash every lattice cell the block
touches once, then gather": a raster block of P pixels whose lattice is resolved (>= 2 px per
cell) touches at most ~P/4 cells, so the hashing cost is a fraction of the pixel cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..hashing import SeedKind, hash_seed, mix_cell, splitmix64, uniform_from_hash

U64 = np.uint64
NM_PER_UM = 1000.0
LATTICE_JITTER_FRAC = 0.25
ANALYTIC_CELL_MIN_PX = 2.0  # kAnalyticCellMinPx
DUST_THRESHOLD_PX = 2.0  # FieldMap::kDustThresholdPx
ROUGH_PERIOD = 7
ROUGH_PEAK = 1.0 / 15.0 + 1.0 / 60.0  # peak radial ripple of MakeRoughEllipse


@dataclass(frozen=True)
class AABB:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @classmethod
    def from_center_half(cls, cx, cy, hx, hy) -> "AABB":
        return cls(cx - hx, cy - hy, cx + hx, cy + hy)

    @classmethod
    def from_center_radius(cls, cx, cy, r) -> "AABB":
        return cls(cx - r, cy - r, cx + r, cy + r)

    @property
    def empty(self) -> bool:
        return self.xmax < self.xmin or self.ymax < self.ymin

    def intersects(self, o: "AABB") -> bool:
        if self.empty or o.empty:
            return False
        return self.xmin <= o.xmax and self.xmax >= o.xmin and self.ymin <= o.ymax and self.ymax >= o.ymin

    def contains(self, x, y):
        return (x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax)

    def pad(self, m: float) -> "AABB":
        return AABB(self.xmin - m, self.ymin - m, self.xmax + m, self.ymax + m)

    def expand(self, o: "AABB") -> "AABB":
        return AABB(min(self.xmin, o.xmin), min(self.ymin, o.ymin), max(self.xmax, o.xmax), max(self.ymax, o.ymax))

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.xmin + self.xmax), 0.5 * (self.ymin + self.ymax))

    @property
    def half_extent(self) -> tuple[float, float]:
        return (0.5 * (self.xmax - self.xmin), 0.5 * (self.ymax - self.ymin))

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)


def rotated_rect_half_extent(hx, hy, rot):
    c, s = abs(math.cos(rot)), abs(math.sin(rot))
    return c * hx + s * hy, s * hx + c * hy


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z * 0.70710678118654752440)


def lognormal_from_hash(h, median, sigma_log, lo, hi):
    """SampleLogNormalFromHash: Box-Muller on two uniforms derived from one hash (vectorised)."""
    h = np.asarray(h, dtype=U64)
    u1 = uniform_from_hash(splitmix64(h))
    u2 = uniform_from_hash(splitmix64(h ^ U64(0xD6E8FEB86659FD93)))
    u1 = np.maximum(u1, 5e-324)
    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return np.clip(median * np.exp(sigma_log * z), lo, hi)


# -- rough ellipse (MakeRoughEllipse outline, analytic) --------------------------------------

def rough_rmod(theta, period=ROUGH_PERIOD, amplitude=1.0):
    """Radial modulation of MakeRoughEllipse in the ellipse's normalised frame."""
    return 1.0 + amplitude * (np.sin((theta + 0.43) * period) / 15.0
                              + np.sin((theta + 0.14) * period * 5) / 60.0)


def inside_rough_ellipse(dx, dy, rx, ry):
    """Exact inside test of the rough-ellipse outline (InsideRoughEllipse), world offsets."""
    u = dx / rx
    v = dy / ry
    rn2 = u * u + v * v
    out = rn2 <= (1.0 - ROUGH_PEAK) ** 2
    band = (~out) & (rn2 <= (1.0 + ROUGH_PEAK) ** 2)
    if np.any(band):
        th = np.arctan2(v[band], u[band])
        out[band] = np.sqrt(rn2[band]) <= rough_rmod(th)
    return out


def rough_ellipse_signed_dist(dx, dy, rx, ry):
    """RoughEllipseSignedDistUm: positive inside, micrometres."""
    u = dx / rx
    v = dy / ry
    rn = np.sqrt(u * u + v * v)
    plen = np.sqrt(dx * dx + dy * dy)
    safe = rn > 1e-12
    ray = np.where(safe, plen / np.where(safe, rn, 1.0), min(rx, ry))
    th = np.where(safe, np.arctan2(v, u), 0.0)
    return (rough_rmod(th) - rn) * ray


# -- jittered lattice -------------------------------------------------------------------------

@dataclass
class LatticeHit:
    i: np.ndarray  # int64 cell index
    j: np.ndarray
    sx: np.ndarray  # site, world um
    sy: np.ndarray
    d2: np.ndarray  # squared distance to the site, um^2
    h: np.ndarray  # uint64 cell hash
    site: Optional[np.ndarray] = None  # index into grid_h (block evaluation only)
    grid_h: Optional[np.ndarray] = None  # hashes of every site of the evaluated grid

    def per_site(self, func) -> np.ndarray:
        """``func(cell_hash)`` per pixel, evaluated once per site when a site grid is available."""
        if self.site is not None and self.grid_h is not None:
            return np.asarray(func(self.grid_h)).take(self.site)
        return np.asarray(func(self.h))

    def masked(self, m) -> "LatticeHit":
        return LatticeHit(self.i[m], self.j[m], self.sx[m], self.sy[m], self.d2[m], self.h[m],
                          None if self.site is None else self.site[m], self.grid_h)

    @property
    def dist(self) -> np.ndarray:
        return np.sqrt(self.d2)


class JitteredLattice:
    """A hashed, jittered grid of sites (Geometry.h JitteredLattice), vectorised."""

    _HY = U64(0xA24BAED4963EE407)

    def __init__(self, cell_um: float, seed: int, salt: int = 0, jitter_frac: float = LATTICE_JITTER_FRAC):
        self.cell_um = float(cell_um)
        self.seed = np.uint64(int(seed) & ((1 << 64) - 1))
        self.salt = int(salt)
        self.jitter_frac = float(jitter_frac)

    def cell_hash(self, i, j):
        return hash_seed(self.seed, SeedKind.GRAIN, mix_cell(np.asarray(i), np.asarray(j)), self.salt)

    def sites(self, i, j):
        i = np.asarray(i, np.int64)
        j = np.asarray(j, np.int64)
        hx = self.cell_hash(i, j)
        if self.jitter_frac != 0.0:
            hy = splitmix64(hx ^ self._HY)
            jx = self.jitter_frac * (2.0 * uniform_from_hash(hx) - 1.0)
            jy = self.jitter_frac * (2.0 * uniform_from_hash(hy) - 1.0)
        else:
            jx = jy = 0.0
        return (i + 0.5 + jx) * self.cell_um, (j + 0.5 + jy) * self.cell_um, hx

    def nearest(self, x, y) -> LatticeHit:
        """Nearest site to each point by an exhaustive 3x3 search (exact at jitter 0.25)."""
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        L = self.cell_um
        ci = np.floor(x / L).astype(np.int64)
        cj = np.floor(y / L).astype(np.int64)
        if x.size == 0:
            z = np.zeros(x.shape)
            return LatticeHit(ci, cj, z, z, z, np.zeros(x.shape, U64))
        imin, imax = int(ci.min()) - 1, int(ci.max()) + 1
        jmin, jmax = int(cj.min()) - 1, int(cj.max()) + 1
        W, H = imax - imin + 1, jmax - jmin + 1
        best = np.full(x.shape, np.inf)
        if W * H <= max(4 * x.size, 4096):
            gi = np.arange(imin, imax + 1, dtype=np.int64)
            gj = np.arange(jmin, jmax + 1, dtype=np.int64)
            SX, SY, HH = self.sites(gi[None, :], gj[:, None])
            SX, SY, HH = SX.ravel(), SY.ravel(), HH.ravel()
            base = (cj - jmin) * W + (ci - imin)
            bf = base.copy()
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    f = base + (dj * W + di)
                    ddx = SX.take(f) - x
                    ddy = SY.take(f) - y
                    d2 = ddx * ddx + ddy * ddy
                    m = d2 < best
                    np.copyto(best, d2, where=m)
                    np.copyto(bf, f, where=m)
            return LatticeHit(imin + bf % W, jmin + bf // W, SX.take(bf), SY.take(bf), best, HH.take(bf))
        # Unresolved lattice over a large block: hash per pixel (callers normally avoid this).
        bi, bj = ci.copy(), cj.copy()
        bsx = np.zeros(x.shape)
        bsy = np.zeros(x.shape)
        bh = np.zeros(x.shape, U64)
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                sx, sy, h = self.sites(ci + di, cj + dj)
                d2 = (sx - x) ** 2 + (sy - y) ** 2
                m = d2 < best
                np.copyto(best, d2, where=m)
                np.copyto(bi, ci + di, where=m)
                np.copyto(bj, cj + dj, where=m)
                np.copyto(bsx, sx, where=m)
                np.copyto(bsy, sy, where=m)
                np.copyto(bh, h, where=m)
        return LatticeHit(bi, bj, bsx, bsy, best, bh)


    # -- block evaluation (exact, coarse-to-fine) ------------------------------------------------
    def _grid(self, x_lo, x_hi, y_lo, y_hi):
        L = self.cell_um
        imin, imax = math.floor(x_lo / L) - 2, math.floor(x_hi / L) + 2
        jmin, jmax = math.floor(y_lo / L) - 2, math.floor(y_hi / L) + 2
        gi = np.arange(imin, imax + 1, dtype=np.int64)
        gj = np.arange(jmin, jmax + 1, dtype=np.int64)
        SX, SY, HH = self.sites(gi[None, :], gj[:, None])
        # site coordinates relative to the grid origin, in float32 (cell-scale precision is ample)
        x0, y0 = imin * L, jmin * L
        return ((SX - x0).astype(np.float32).ravel(), (SY - y0).astype(np.float32).ravel(), HH.ravel(),
                imin, jmin, imax - imin + 1, x0, y0)

    def _label(self, x, y, grid):
        """Flat grid index of the nearest site. Quadrant 2x2 search: with sites confined to the
        central half of their cells (jitter 0.25) the nearest site always lies in the 2x2 block
        of cells on the point's side of its cell (verified against the 3x3 search on 5x10^7
        random and adversarial points)."""
        SX, SY, _HH, imin, jmin, W, x0, y0 = grid
        L = self.cell_um
        u = (np.asarray(x, np.float64) - x0) * (1.0 / L)
        v = (np.asarray(y, np.float64) - y0) * (1.0 / L)
        ci = np.floor(u)
        cj = np.floor(v)
        if self.jitter_frac <= 0.25:
            ci -= (u - ci) < 0.5
            cj -= (v - cj) < 0.5
            offs = (0, 1, W, W + 1)
        else:
            ci -= 1
            cj -= 1
            offs = tuple(dj * W + di for dj in range(3) for di in range(3))
        base = (cj * W + ci).astype(np.int32)
        xf = (u * L).astype(np.float32)
        yf = (v * L).astype(np.float32)
        best = np.full(xf.shape, np.inf, np.float32)
        bf = base.copy()
        for o in offs:
            f = base + np.int32(o) if o else base
            ddx = SX.take(f) - xf
            ddy = SY.take(f) - yf
            d2 = ddx * ddx + ddy * ddy
            m = d2 < best
            np.copyto(best, d2, where=m)
            np.copyto(bf, f, where=m)
        return bf

    def block_labels(self, ctx, r0: int, r1: int, c0: int, c1: int, offset=(0.0, 0.0)):
        """Nearest-site label for every pixel centre of a raster window.

        Returns ``(labels (r1 - r0, c1 - c0) int64, grid)``; ``grid = (SX, SY, HH, imin, jmin, W)``
        holds the site table the labels index. Exact: sites are labelled on a regular coarse node
        grid; a block whose four corner nodes share a site lies entirely inside that site's
        (convex) Voronoi cell, because raster -> world is affine, so only blocks straddling a
        cell boundary are labelled per pixel. ``offset`` is subtracted from world coordinates.
        """
        ox, oy = float(offset[0]), float(offset[1])

        def world(r, c):
            X, Y = ctx.world(r, c)
            return X - ox, Y - oy

        H, Wd = r1 - r0, c1 - c0
        step = max(math.hypot(ctx.axc, ctx.ayc), math.hypot(ctx.axr, ctx.ayr))
        # node stride minimising (1/s^2 node work + ~4 s / C boundary work) for C px cells
        s = int(min(16, max(1, math.floor((0.5 * self.cell_um / step) ** (1.0 / 3.0) + 0.5))))
        nr_n = (H - 1) // s + 2 if s > 1 else 0
        nc_n = (Wd - 1) // s + 2 if s > 1 else 0
        cr = np.array([r0, r0, r0 + max(nr_n - 1, 0) * s + 1, r0 + max(nr_n - 1, 0) * s + 1], np.float64)
        cc = np.array([c0, c0 + max(nc_n - 1, 0) * s + 1, c0, c0 + max(nc_n - 1, 0) * s + 1], np.float64)
        cr = np.minimum(cr, r0 + max(H, (nr_n - 1) * s + 1))
        cx, cy = world(np.r_[cr, r1], np.r_[cc, c1])
        grid = self._grid(cx.min(), cx.max(), cy.min(), cy.max())
        from . import raster_nb as _rnb

        if _rnb.AVAILABLE:
            SX, SY, _HH, imin, jmin, gw, x0, y0 = grid
            out = np.empty((H, Wd), np.int32)
            _rnb.block_labels(np.array([ctx.ox, ctx.oy, ctx.axc, ctx.ayc, ctx.axr, ctx.ayr]), ox, oy, r0, r1, c0,
                              c1, s, nr_n, nc_n, SX, SY, float(x0), float(y0), float(self.cell_um), int(gw),
                              self.jitter_frac <= 0.25, out)
            return out, grid
        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        if s < 2 or H < 3 or Wd < 3:
            X, Y = world(rows[:, None], cols[None, :])
            X, Y = np.broadcast_to(X, (H, Wd)), np.broadcast_to(Y, (H, Wd))
            return self._label(X, Y, grid), grid
        nr = r0 + s * np.arange(nr_n)
        nc = c0 + s * np.arange(nc_n)
        Xn, Yn = world(nr[:, None], nc[None, :])
        Ln = self._label(np.broadcast_to(Xn, (nr_n, nc_n)), np.broadcast_to(Yn, (nr_n, nc_n)), grid)
        a = Ln[:-1, :-1]
        U = (a == Ln[1:, :-1]) & (a == Ln[:-1, 1:]) & (a == Ln[1:, 1:])
        lab = np.repeat(np.repeat(a, s, axis=0)[:H], s, axis=1)[:, :Wd]
        if not U.all():
            todo = ~np.repeat(np.repeat(U, s, axis=0)[:H], s, axis=1)[:, :Wd]
            tr, tc = np.nonzero(todo)
            X, Y = world(rows[tr], cols[tc])
            lab[tr, tc] = self._label(X, Y, grid)
        return lab, grid

    def nearest_block(self, ctx, r0: int, r1: int, c0: int, c1: int, offset=(0.0, 0.0)) -> "BlockHit":
        lab, grid = self.block_labels(ctx, r0, r1, c0, c1, offset)
        X, Y = ctx.world(np.arange(r0, r1)[:, None], np.arange(c0, c1)[None, :])
        shape = lab.shape
        return BlockHit(lab, grid, np.broadcast_to(X, shape) - offset[0], np.broadcast_to(Y, shape) - offset[1])


class BlockHit:
    """Lazy nearest-site result: per-pixel site labels into a small site table."""

    __slots__ = ("site", "grid", "X", "Y")

    def __init__(self, site, grid, X, Y):
        self.site, self.grid, self.X, self.Y = site, grid, X, Y

    def masked(self, m) -> "BlockHit":
        return BlockHit(self.site[m], self.grid, self.X[m], self.Y[m])

    @property
    def grid_h(self):
        return self.grid[2]

    @property
    def h(self):
        return self.grid[2].take(self.site)

    @property
    def sx(self):
        return self.grid[0].take(self.site) + self.grid[6]

    @property
    def sy(self):
        return self.grid[1].take(self.site) + self.grid[7]

    @property
    def d2(self):
        dx = (self.X - self.grid[6]) - self.grid[0].take(self.site)
        dy = (self.Y - self.grid[7]) - self.grid[1].take(self.site)
        return dx * dx + dy * dy

    @property
    def i(self):
        return self.grid[3] + self.site % self.grid[5]

    @property
    def j(self):
        return self.grid[4] + self.site // self.grid[5]

    def per_site(self, func) -> np.ndarray:
        """``func(cell_hash)`` per pixel, evaluated once per site of the table."""
        return np.asarray(func(self.grid[2])).take(self.site)


def value_noise(seed: int, x, y):
    """ValueNoise2D: bilinear value noise in [0, 1) on the integer lattice (x, y in cell units)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    fx = np.floor(x)
    fy = np.floor(y)
    tx = x - fx
    ty = y - fy
    ix = fx.astype(np.int64)
    iy = fy.astype(np.int64)
    s = np.uint64(int(seed) & ((1 << 64) - 1))
    if x.size == 0:
        return np.zeros(x.shape)
    imin, imax = int(ix.min()), int(ix.max()) + 1
    jmin, jmax = int(iy.min()), int(iy.max()) + 1
    W, H = imax - imin + 1, jmax - jmin + 1
    if W * H <= max(4 * x.size, 4096):
        gi = np.arange(imin, imax + 1, dtype=np.int64)
        gj = np.arange(jmin, jmax + 1, dtype=np.int64)
        G = uniform_from_hash(hash_seed(s, SeedKind.SUPPORT_FILM, gi[None, :], gj[:, None])).ravel()
        f = (iy - jmin) * W + (ix - imin)
        n00 = G.take(f)
        n10 = G.take(f + 1)
        n01 = G.take(f + W)
        n11 = G.take(f + W + 1)
    else:
        def hu(a, b):
            return uniform_from_hash(hash_seed(s, SeedKind.SUPPORT_FILM, a, b))
        n00, n10, n01, n11 = hu(ix, iy), hu(ix + 1, iy), hu(ix, iy + 1), hu(ix + 1, iy + 1)
    a = n00 + (n10 - n00) * tx
    b = n01 + (n11 - n01) * tx
    return a + (b - a) * ty


def _separable_noise_tables(seed: int, xs, ys):
    """``value_noise_separable``'s lattice table and per-row / per-column indices and
    fractions (G, ix, tx, iy, ty), or None when it would fall back to ``value_noise``."""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    fx, fy = np.floor(xs), np.floor(ys)
    tx, ty = xs - fx, ys - fy
    ix, iy = fx.astype(np.int64), fy.astype(np.int64)
    imin, imax = int(ix.min()), int(ix.max()) + 1
    jmin, jmax = int(iy.min()), int(iy.max()) + 1
    s = np.uint64(int(seed) & ((1 << 64) - 1))
    gi = np.arange(imin, imax + 1, dtype=np.int64)
    gj = np.arange(jmin, jmax + 1, dtype=np.int64)
    if gi.size * gj.size > 4 * xs.size * ys.size + 4096:
        return None
    G = uniform_from_hash(hash_seed(s, SeedKind.SUPPORT_FILM, gi[None, :], gj[:, None]))
    return G, ix - imin, tx, iy - jmin, ty


def value_noise_separable(seed: int, xs, ys):
    """``value_noise`` on the grid ``(ys[:, None], xs[None, :])`` (axis-aligned views).

    Interpolates along y on the hashed lattice rows first, then gathers along x: two gathers of
    the output size instead of four.
    """
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    fx, fy = np.floor(xs), np.floor(ys)
    tx, ty = xs - fx, ys - fy
    ix, iy = fx.astype(np.int64), fy.astype(np.int64)
    imin, imax = int(ix.min()), int(ix.max()) + 1
    jmin, jmax = int(iy.min()), int(iy.max()) + 1
    s = np.uint64(int(seed) & ((1 << 64) - 1))
    gi = np.arange(imin, imax + 1, dtype=np.int64)
    gj = np.arange(jmin, jmax + 1, dtype=np.int64)
    if gi.size * gj.size > 4 * xs.size * ys.size + 4096:
        X, Y = np.meshgrid(xs, ys)
        return value_noise(seed, X, Y)
    G = uniform_from_hash(hash_seed(s, SeedKind.SUPPORT_FILM, gi[None, :], gj[:, None]))
    r0 = G[iy - jmin]
    r1 = G[iy - jmin + 1]
    Ry = r0 + (r1 - r0) * ty[:, None]
    a = Ry[:, ix - imin]
    b = Ry[:, ix - imin + 1]
    return a + (b - a) * tx[None, :]


# -- polygons in raster space -----------------------------------------------------------------

def polygon_mask_rows(pcol, prow, r0: int, r1: int, nx: int) -> np.ndarray:
    """Pixel-centre inside mask of a polygon given in raster coordinates.

    ``pcol``/``prow`` are the polygon vertices in (column, row) pixel-index coordinates. Returns
    bool ``(r1 - r0, nx)``. Scanline parity: per row, the crossings of every edge are found once
    and turned into a running parity with a difference array, so the cost is O(edges x rows +
    pixels) independent of the polygon's size on screen. Works for any affine view.
    """
    pcol = np.asarray(pcol, np.float64)
    prow = np.asarray(prow, np.float64)
    R = r1 - r0
    out = np.zeros((R, nx), bool)
    if R <= 0 or len(pcol) < 3:
        return out
    x0, y0 = pcol, prow
    x1, y1 = np.roll(pcol, -1), np.roll(prow, -1)
    ylo = np.minimum(y0, y1)
    yhi = np.maximum(y0, y1)
    keep = (yhi >= r0 - 1) & (ylo <= r1)
    if not np.any(keep):
        return out
    x0, y0, x1, y1 = x0[keep], y0[keep], x1[keep], y1[keep]
    rows = np.arange(r0, r1, dtype=np.float64)[:, None]
    cond = (y0[None, :] > rows) != (y1[None, :] > rows)
    rr, ee = np.nonzero(cond)
    if rr.size == 0:
        return out
    yr = rows[rr, 0]
    xc = x0[ee] + (x1[ee] - x0[ee]) * (yr - y0[ee]) / ((y1[ee] - y0[ee]) + 1e-10)
    # pixel j counts this crossing when j < xc: toggles columns [0, ceil(xc)).
    k = np.clip(np.ceil(xc), 0, nx).astype(np.int64)
    diff = np.zeros((R, nx + 1), np.int32)
    np.add.at(diff, (rr, np.zeros_like(rr)), 1)
    np.add.at(diff, (rr, k), -1)
    return (np.cumsum(diff[:, :nx], axis=1) & 1).astype(bool)
