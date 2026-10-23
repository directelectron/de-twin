"""FieldMap rasterisation (port of VirtualSpecimen ``FieldMap.cpp``), vectorised.

Pipeline of :func:`rasterize_scene`, as in ``FieldMap::Rasterize``:

1. holder bulk - analytic per pixel (grid bars, rim, FIB body, chip membrane/heater), row blocks;
2. support film - only where the bulk pass found a placement area;
3. primitives in painter's order (layer), with the LOD rules: culled below ``lod_threshold_px``
   raster pixels across, "dust" (mass-conserving splat) below 2 px, otherwise drawn;
4. ``Structure.fill`` for every drawn primitive that owns a structure;
5. optional layers (descan, strain) are filled by the structures.

Drawn primitives are rasterised in vectorised batches: primitives are bucketed by the size of
their raster window (power-of-two ``K``) and each bucket is evaluated as an ``(n, K, K)`` stack
with analytic inside tests (ellipse, rough ellipse, faceted star polygon, rectangle, ring) at
pixel centres. Painter's-order compositing inside one batch uses "thickest contribution wins",
which is the per-pixel outcome of the C++ material rule for all but pathological overlaps.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

from ..hashing import SeedKind, hash_seed, uniform_from_hash
from . import patch_nb as _patch_nb
from . import raster_nb as _nb
from .fieldmap import LAYER_DESCAN, LAYER_STRAIN, FieldMap, GrainTable, ViewWindow
from .geometry import AABB, DUST_THRESHOLD_PX, ROUGH_PEAK, ROUGH_PERIOD, rough_rmod


class Shape(IntEnum):
    ELLIPSE = 0
    ROUGH_ELLIPSE = 1
    FACETED = 2  # star polygon, vertices j_k * (cos, sin)(2 pi k / n) in the normalised frame
    RECT = 3
    RING = 4


class Profile(IntEnum):
    FLAT = 0
    SPHERICAL = 1
    WEDGE = 2
    CURTAIN = 3


class Layer(IntEnum):
    HOLDER_BULK = 0
    SUPPORT_FILM = 1
    PREPARATION = 2
    STRUCTURE = 3


MAX_FACETS = 8

_FIELDS = {
    "cx": np.float64, "cy": np.float64, "rx": np.float64, "ry": np.float64, "rot": np.float64,
    "bound_r": np.float64, "xmin": np.float64, "ymin": np.float64, "xmax": np.float64,
    "ymax": np.float64, "thickness": np.float32, "profile": np.int8, "shape": np.int8,
    "layer": np.int8, "material": np.uint8, "grain": np.int32, "inner_frac": np.float32,
    "seed": np.uint64, "aggregate": np.bool_, "structure": np.int32, "facet_n": np.int8,
}


@dataclass
class Owner:
    """One primitive, as a structure's ``Fill`` sees its owner."""

    cx: float
    cy: float
    rx: float
    ry: float
    rot: float
    shape: int
    bounds: AABB
    material: int
    grain: int
    seed: int
    thickness: float = 0.0


class PrimitiveSet:
    """Struct-of-arrays store of rasterisable primitives (the C++ ``Primitive`` vector)."""

    def __init__(self, n: int = 0, structures: Optional[list] = None, **arrays):
        self.n = int(n)
        for k, dt in _FIELDS.items():
            if k in arrays:
                a = np.asarray(arrays.pop(k)).astype(dt, copy=False)
                if a.ndim == 0:
                    a = np.full(self.n, a, dt)
                setattr(self, k, a)
            else:
                default = {"structure": -1, "grain": -1, "layer": Layer.PREPARATION}.get(k, 0)
                setattr(self, k, np.full(self.n, default, dt))
        fj = arrays.pop("facet_j", None)
        self.facet_j = (np.ones((self.n, MAX_FACETS), np.float32) if fj is None
                        else np.asarray(fj, np.float32).reshape(self.n, MAX_FACETS))
        if arrays:
            raise TypeError(f"unknown primitive fields {sorted(arrays)}")
        self.structures = list(structures or [])

    @classmethod
    def make(cls, cx, cy, rx, ry, *, rot=0.0, shape=Shape.ELLIPSE, bounds=None, bound_r=None,
             structures=None, **extra_fields) -> "PrimitiveSet":
        cx = np.atleast_1d(np.asarray(cx, np.float64))
        n = cx.size
        cy = np.broadcast_to(np.asarray(cy, np.float64), (n,))
        rx = np.broadcast_to(np.asarray(rx, np.float64), (n,))
        ry = np.broadcast_to(np.asarray(ry, np.float64), (n,))
        rot = np.broadcast_to(np.asarray(rot, np.float64), (n,))
        shape_a = np.broadcast_to(np.asarray(shape, np.int8), (n,))
        if bound_r is None:
            bound_r = np.where(shape_a == Shape.RECT, np.hypot(rx, ry), np.maximum(rx, ry))
        bound_r = np.broadcast_to(np.asarray(bound_r, np.float64), (n,))
        if bounds is None:
            c, s = np.abs(np.cos(rot)), np.abs(np.sin(rot))
            hx = np.where(shape_a == Shape.RECT, c * rx + s * ry, bound_r)
            hy = np.where(shape_a == Shape.RECT, s * rx + c * ry, bound_r)
            bounds = (cx - hx, cy - hy, cx + hx, cy + hy)
        return cls(n, structures=structures, cx=cx, cy=cy, rx=rx, ry=ry, rot=rot, shape=shape_a,
                   bound_r=bound_r, xmin=bounds[0], ymin=bounds[1], xmax=bounds[2], ymax=bounds[3], **extra_fields)

    def __len__(self) -> int:
        return self.n

    def take(self, idx) -> "PrimitiveSet":
        idx = np.asarray(idx)
        out = PrimitiveSet.__new__(PrimitiveSet)
        for k in _FIELDS:
            setattr(out, k, getattr(self, k)[idx])
        out.facet_j = self.facet_j[idx]
        out.n = int(out.cx.shape[0])
        out.structures = self.structures
        return out

    @staticmethod
    def concat(sets: list["PrimitiveSet"]) -> "PrimitiveSet":
        sets = [s for s in sets if s is not None and s.n > 0]
        out = PrimitiveSet(0)
        if not sets:
            return out
        structures: list = []
        parts = {k: [] for k in _FIELDS}
        fjs = []
        for s in sets:
            for k in _FIELDS:
                a = getattr(s, k)
                if k == "structure":
                    a = np.where(a >= 0, a + len(structures), -1).astype(np.int32)
                parts[k].append(a)
            fjs.append(s.facet_j)
            structures.extend(s.structures)
        for k in _FIELDS:
            setattr(out, k, np.concatenate(parts[k]))
        out.facet_j = np.concatenate(fjs)
        out.n = int(out.cx.shape[0])
        out.structures = structures
        return out

    def intersects(self, box: AABB) -> np.ndarray:
        return (self.xmin <= box.xmax) & (self.xmax >= box.xmin) & (self.ymin <= box.ymax) & (self.ymax >= box.ymin)

    def owner(self, i: int) -> Owner:
        return Owner(float(self.cx[i]), float(self.cy[i]), float(self.rx[i]), float(self.ry[i]),
                     float(self.rot[i]), int(self.shape[i]),
                     AABB(float(self.xmin[i]), float(self.ymin[i]), float(self.xmax[i]), float(self.ymax[i])),
                     int(self.material[i]), int(self.grain[i]), int(self.seed[i]), float(self.thickness[i]))

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, k).nbytes for k in _FIELDS) + self.facet_j.nbytes


# ---------------------------------------------------------------------------------------------
# Raster context
# ---------------------------------------------------------------------------------------------

class RasterContext:
    """The raster being built: view affine, flat core layers and optional layers."""

    BLOCK_PIXELS = 1 << 17

    def __init__(self, view: ViewWindow, layers=frozenset(), grains: Optional[GrainTable] = None,
                 time_index: int = 0, lod_threshold_px: float = 0.5, span_px: Optional[int] = None):
        self.view = view
        self.ny, self.nx = int(view.shape[0]), int(view.shape[1])
        #: The size (pixels) of the view this raster is part of: what the scene's populate /
        #: aggregate choice is made on, so a piece of a view is drawn as the whole view is.
        self.span_px = int(span_px) if span_px else max(self.ny, self.nx)
        self.npix = self.ny * self.nx
        x00, y00 = view.pixel_to_world(0.0, 0.0)
        xr, yr = view.pixel_to_world(1.0, 0.0)
        xc, yc = view.pixel_to_world(0.0, 1.0)
        self.ox, self.oy = float(x00), float(y00)
        self.axc, self.ayc = float(xc - x00), float(yc - y00)  # world per +1 column
        self.axr, self.ayr = float(xr - x00), float(yr - y00)  # world per +1 row
        self.det = self.axc * self.ayr - self.axr * self.ayc
        # Rows are horizontal and columns vertical in world space (no rotation, any flip/tilt).
        self.axis_aligned = abs(self.axr) <= 1e-12 * abs(self.axc) and abs(self.ayc) <= 1e-12 * abs(self.ayr)
        self.pixel_um = float(view.pixel_um)  # LOD scale (nmPerRasterPx)
        self.step_um = math.hypot(self.axc, self.ayc)  # world step along a row
        self.aabb = AABB(*view.bounds_um())
        self.layers = frozenset(layers)
        self.grains = grains
        self.time_index = int(time_index)
        self.lod_threshold_px = float(lod_threshold_px)
        from .. import buffers  # recycled: their pages are mapped already (see de_twin.buffers)

        self.material = buffers.zeros(self.npix, np.uint8)
        self.thick = buffers.zeros(self.npix, np.float32)
        self.grain = buffers.full(self.npix, -1, np.int32)
        self.area = buffers.full(self.npix, -1, np.int32)
        self._mark = buffers.full(self.npix, -1, np.int32)
        self.descan = np.zeros((2, self.npix), np.float32) if LAYER_DESCAN in self.layers else None
        if LAYER_STRAIN in self.layers:
            self.strain = np.zeros((3, self.npix), np.float32)
            self.strain[0] = 1.0
            self.strain[2] = 1.0
        else:
            self.strain = None
        self.stats = {"considered": 0, "culled": 0, "dust": 0, "drawn": 0, "structures": 0}
        self.under_material: Optional[np.ndarray] = None
        self.under_thick: Optional[np.ndarray] = None

    # -- coordinates --------------------------------------------------------------------------
    def world(self, rows, cols):
        rows = np.asarray(rows, np.float64)
        cols = np.asarray(cols, np.float64)
        return (self.ox + self.axc * cols + self.axr * rows, self.oy + self.ayc * cols + self.ayr * rows)

    def linear(self, p: float, q: float, k: float, rows, cols):
        """``p * X + q * Y + k`` evaluated as an affine outer sum over (rows, cols)."""
        a0 = k + p * self.ox + q * self.oy
        ac = p * self.axc + q * self.ayc
        ar = p * self.axr + q * self.ayr
        return (a0 + ac * np.asarray(cols, np.float64)) + ar * np.asarray(rows, np.float64)

    def to_pixel(self, x, y):
        """World -> (row, col) float raster coordinates (pixel centres at integers)."""
        dx = np.asarray(x, np.float64) - self.ox
        dy = np.asarray(y, np.float64) - self.oy
        col = (self.ayr * dx - self.axr * dy) / self.det
        row = (-self.ayc * dx + self.axc * dy) / self.det
        return row, col

    def window(self, xmin, ymin, xmax, ymax, slack: int = 1):
        """Half-open (r0, r1, c0, c1) raster window covering a world box, or None."""
        xs = np.array([xmin, xmax, xmin, xmax], np.float64)
        ys = np.array([ymin, ymin, ymax, ymax], np.float64)
        r, c = self.to_pixel(xs, ys)
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(c))):
            return None
        r0 = max(0, int(math.floor(r.min())) - slack)
        r1 = min(self.ny, int(math.ceil(r.max())) + slack + 1)
        c0 = max(0, int(math.floor(c.min())) - slack)
        c1 = min(self.nx, int(math.ceil(c.max())) + slack + 1)
        if r1 <= r0 or c1 <= c0:
            return None
        return r0, r1, c0, c1

    def row_blocks(self, r0: int, r1: int, width: int):
        step = max(1, self.BLOCK_PIXELS // max(1, width))
        for a in range(r0, r1, step):
            yield a, min(r1, a + step)

    def diameter_px(self, radius_um):
        return 2.0 * np.asarray(radius_um) / self.pixel_um

    def iter_window(self, win):
        """Yield (flat_index (h, w), X, Y, (r0, r1, c0, c1)) row blocks of a raster window."""
        r0, r1, c0, c1 = win
        cols = np.arange(c0, c1)
        for a, b in self.row_blocks(r0, r1, c1 - c0):
            rows = np.arange(a, b)[:, None]
            X, Y = self.world(rows, cols[None, :])
            shape = (b - a, c1 - c0)
            yield (rows * self.nx + cols[None, :]), np.broadcast_to(X, shape), np.broadcast_to(Y, shape), (a, b, c0, c1)

    # -- compositing ----------------------------------------------------------------------------
    def add_thickness(self, flat, nm):
        """Unique-pixel thickness add with the C++ [0, 65535] saturation."""
        t = self.thick[flat].astype(np.float64) + nm
        self.thick[flat] = np.clip(t, 0.0, 65535.0)

    def apply(self, flat, t, mat, grain, overwrite=True, grain_with_claim=False, unique=False):
        """Composite a set of contributions (several may hit one pixel).

        Material rule of ``PaintPrimitive``: a vacuum pixel is always claimed; otherwise the
        (thickest) contribution claims it when ``t >= existing thickness``. Grain ids are written
        wherever a grain-bearing primitive covers the pixel (``PaintPrimitive``) or together with
        the material claim (``PaintDust``).
        """
        n = flat.size
        if n == 0:
            return
        mat = np.broadcast_to(np.asarray(mat, np.uint8), (n,))
        grain = np.broadcast_to(np.asarray(grain, np.int32), (n,))
        # Fast path: no pixel is hit twice (the common case). Detected without sorting by
        # scattering the contribution index and reading it back.
        if not unique:
            ar = np.arange(n, dtype=np.int32)
            self._mark[flat] = ar
            unique = np.array_equal(self._mark[flat], ar)
        if unique:
            existing = self.thick[flat].astype(np.float64)
            wasvac = self.material[flat] == 0
            self.thick[flat] = np.clip(existing + t, 0.0, 65535.0)
            ow = np.broadcast_to(np.asarray(overwrite, bool), (n,))
            claim = wasvac | (ow & (t >= existing))
            self.material[flat[claim]] = mat[claim]
            sel = claim if grain_with_claim else (grain >= 0)
            self.grain[flat[sel]] = grain[sel]
            return
        order = np.lexsort((t, flat))
        fs = flat[order]
        ts = t[order].astype(np.float64)
        new = np.empty(n, bool)
        new[0] = True
        np.not_equal(fs[1:], fs[:-1], out=new[1:])
        starts = np.flatnonzero(new)
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:] - 1
        ends[-1] = n - 1
        uniq = fs[starts]
        sums = np.add.reduceat(ts, starts)
        existing = self.thick[uniq].astype(np.float64)
        wasvac = self.material[uniq] == 0
        self.thick[uniq] = np.clip(existing + sums, 0.0, 65535.0)
        w = order[ends]
        tw = ts[ends]
        ow = np.broadcast_to(np.asarray(overwrite, bool), (n,))[w]
        claim = wasvac | (ow & (tw >= existing))
        self.material[uniq[claim]] = mat[w][claim]
        g = grain[w]
        sel = claim if grain_with_claim else (g >= 0)
        self.grain[uniq[sel]] = g[sel]

    def add_under(self, flat, material: int, nm) -> None:
        """Add `nm` of amorphous `material` to the under layer of pixels `flat` (a support
        film beneath what the pixel holds, stain around protein, ...)."""
        if self.under_thick is None:
            self.under_material = np.zeros(self.npix, np.uint8)
            self.under_thick = np.zeros(self.npix, np.float32)
        nm = np.broadcast_to(np.asarray(nm, np.float32), np.shape(self.under_thick[flat]))
        self.under_thick[flat] = np.maximum(self.under_thick[flat] + nm, 0.0)
        self.under_material[flat] = np.where(nm > 0, np.uint8(material), self.under_material[flat])

    def to_fieldmap(self, time_s: float = 0.0, generation: int = 0) -> FieldMap:
        shp = (self.ny, self.nx)
        fm = FieldMap(view=self.view, material_id=self.material.reshape(shp),
                      thickness_nm=self.thick.reshape(shp), grain_id=self.grain.reshape(shp),
                      generation=generation, time_s=time_s)
        if self.descan is not None:
            fm.descan = self.descan.reshape((2,) + shp)
        if self.strain is not None:
            fm.strain = self.strain.reshape((3,) + shp)
        if self.under_thick is not None:
            fm.under_material = self.under_material.reshape(shp)
            fm.under_thickness_nm = self.under_thick.reshape(shp)
        return fm


# ---------------------------------------------------------------------------------------------
# Batched windows
# ---------------------------------------------------------------------------------------------

def iter_patches(ctx: RasterContext, xmin, ymin, xmax, ymax, max_elems: int = 1 << 19):
    """Group world boxes into equal-size raster windows and yield vectorised stacks.

    Yields ``(sel, rows, cols, valid)`` where ``sel`` indexes the input boxes and ``rows``/
    ``cols``/``valid`` broadcast to ``(n, H, W)``.
    """
    xmin = np.asarray(xmin, np.float64)
    if xmin.size == 0:
        return
    xs = np.stack([xmin, np.asarray(xmax), xmin, np.asarray(xmax)])
    ys = np.stack([np.asarray(ymin), np.asarray(ymin), np.asarray(ymax), np.asarray(ymax)])
    r, c = ctx.to_pixel(xs, ys)
    r0 = np.clip(np.floor(r.min(0)) - 1, 0, ctx.ny).astype(np.int64)
    r1 = np.clip(np.ceil(r.max(0)) + 2, 0, ctx.ny).astype(np.int64)
    c0 = np.clip(np.floor(c.min(0)) - 1, 0, ctx.nx).astype(np.int64)
    c1 = np.clip(np.ceil(c.max(0)) + 2, 0, ctx.nx).astype(np.int64)
    h = r1 - r0
    w = c1 - c0
    ok = (h > 0) & (w > 0)
    size = np.maximum(h, w)
    # Square windows bucketed to multiples of 8 px up to 64, powers of two beyond.
    K = np.where(size <= 64, 8 * np.ceil(np.maximum(size, 1) / 8.0),
                 2 ** np.ceil(np.log2(np.maximum(size, 1)))).astype(np.int64)
    K = np.where(ok, K, 0)
    for k in np.unique(K[ok]):
        sel_all = np.flatnonzero(K == k)
        if k <= 64:
            per = max(1, max_elems // int(k * k))
            ar = np.arange(k)
            for a in range(0, sel_all.size, per):
                sel = sel_all[a:a + per]
                rows = r0[sel][:, None, None] + ar[None, :, None]
                cols = c0[sel][:, None, None] + ar[None, None, :]
                valid = (ar[None, :, None] < h[sel][:, None, None]) & (ar[None, None, :] < w[sel][:, None, None])
                yield sel, rows, cols, valid
        else:
            for i in sel_all:
                W = int(w[i])
                step = max(1, max_elems // W)
                for a in range(int(r0[i]), int(r1[i]), step):
                    b = min(int(r1[i]), a + step)
                    rows = np.arange(a, b)[None, :, None]
                    cols = np.arange(int(c0[i]), int(c1[i]))[None, None, :]
                    valid = np.ones((1, 1, 1), bool)
                    yield np.array([i]), rows, cols, valid


# ---------------------------------------------------------------------------------------------
# Primitive painting
# ---------------------------------------------------------------------------------------------

CURTAIN_OCTAVES = 20
CURTAIN_FREQUENCY = 7.5
CURTAIN_SAMPLES = 1024
_MEAN_ABS_SIN = 2.0 / math.pi


def curtain_table(seed: int) -> np.ndarray:
    """profile(u) = mean_o |sin(2 pi F u + phi_o)|, tabulated on u in [0, 1] (FillCurtainProfile)."""
    o = np.arange(CURTAIN_OCTAVES)
    phase = 2.0 * np.pi * uniform_from_hash(hash_seed(np.uint64(seed), SeedKind.LAMELLA, o))
    u = np.arange(CURTAIN_SAMPLES + 1) / CURTAIN_SAMPLES
    return np.abs(np.sin(2.0 * np.pi * CURTAIN_FREQUENCY * u[:, None] + phase[None, :])).mean(1)


def _local_frame(ctx: "RasterContext", P: PrimitiveSet, sel, rows, cols):
    """Normalised, un-rotated primitive-frame coordinates as affine outer sums."""
    ang = -P.rot[sel]
    c = np.cos(ang)
    s = np.sin(ang)
    rx = P.rx[sel]
    ry = P.ry[sel]
    irx = np.where(rx > 0, 1.0 / np.where(rx > 0, rx, 1.0), 0.0)
    iry = np.where(ry > 0, 1.0 / np.where(ry > 0, ry, 1.0), 0.0)
    dx0 = ctx.ox - P.cx[sel]
    dy0 = ctx.oy - P.cy[sel]
    e = (slice(None), None, None)
    lx = ((irx * (c * dx0 - s * dy0))[e] + (irx * (c * ctx.axc - s * ctx.ayc))[e] * cols)         + (irx * (c * ctx.axr - s * ctx.ayr))[e] * rows
    ly = ((iry * (s * dx0 + c * dy0))[e] + (iry * (s * ctx.axc + c * ctx.ayc))[e] * cols)         + (iry * (s * ctx.axr + c * ctx.ayr))[e] * rows
    return lx, ly


def _inside(P: PrimitiveSet, sel, shape: int, lx, ly, r2):
    if shape == Shape.RECT:
        return (np.abs(lx) <= 1.0) & (np.abs(ly) <= 1.0)
    if shape == Shape.ELLIPSE:
        return r2 <= 1.0
    if shape == Shape.RING:
        inner = P.inner_frac[sel].astype(np.float64)[:, None, None]
        return (r2 <= 1.0) & (r2 >= inner * inner)
    if shape == Shape.ROUGH_ELLIPSE:
        out = r2 <= (1.0 - ROUGH_PEAK) ** 2
        band = (~out) & (r2 <= (1.0 + ROUGH_PEAK) ** 2)
        if np.any(band):
            out[band] = np.sqrt(r2[band]) <= rough_rmod(np.arctan2(ly[band], lx[band]))
        return out
    if shape == Shape.FACETED:
        n = P.facet_n[sel].astype(np.int64)
        n = np.maximum(n, 3)
        nb = np.broadcast_to(n[:, None, None], lx.shape)
        phi = np.mod(np.arctan2(ly, lx), 2.0 * np.pi)
        k = np.minimum((phi * nb / (2.0 * np.pi)).astype(np.int64), nb - 1)
        k1 = (k + 1) % nb
        J = P.facet_j[sel].astype(np.float64).ravel()
        base = (np.arange(len(sel)) * MAX_FACETS)[:, None, None]
        jk = J[base + k]
        jk1 = J[base + k1]
        tk = 2.0 * np.pi * k / nb
        tk1 = 2.0 * np.pi * k1 / nb
        ax, ay = jk * np.cos(tk), jk * np.sin(tk)
        bx, by = jk1 * np.cos(tk1), jk1 * np.sin(tk1)
        return (bx - ax) * (ly - ay) - (by - ay) * (lx - ax) >= 0.0
    raise ValueError(f"unknown shape {shape}")


def _thickness(P: PrimitiveSet, sel, profile: int, lx, ly, r2, curtain_depth: float):
    t0 = P.thickness[sel].astype(np.float64)[:, None, None]
    if profile == Profile.FLAT:
        return np.broadcast_to(t0, lx.shape)
    if profile == Profile.SPHERICAL:
        return t0 * np.sqrt(np.maximum(0.0, 1.0 - r2))
    if profile == Profile.WEDGE:
        return t0 * np.clip(0.5 * (ly + 1.0), 0.0, 1.0)
    if profile == Profile.CURTAIN:
        tables = np.stack([curtain_table(int(s)) for s in P.seed[sel]])  # (n, S+1)
        s = np.clip(0.5 * (lx + 1.0), 0.0, 1.0) * CURTAIN_SAMPLES
        i0 = np.minimum(s.astype(np.int64), CURTAIN_SAMPLES)
        i1 = np.minimum(i0 + 1, CURTAIN_SAMPLES)
        fr = s - i0
        base = (np.arange(len(sel)) * (CURTAIN_SAMPLES + 1))[:, None, None]
        T = tables.ravel()
        prof = T[base + i0] + fr * (T[base + i1] - T[base + i0])
        t = t0 * (1.0 - curtain_depth * (prof - _MEAN_ABS_SIN) / _MEAN_ABS_SIN)
        return np.maximum(t, 0.0)
    raise ValueError(f"unknown profile {profile}")


LARGE_WINDOW_PX = 48


def _affine_coeffs(ctx: "RasterContext", P: PrimitiveSet, i: int):
    """(A0, Ac, Ar, B0, Bc, Br): lx = A0 + Ac*col + Ar*row, ly = B0 + Bc*col + Br*row."""
    ang = -float(P.rot[i])
    c, s = math.cos(ang), math.sin(ang)
    rx, ry = float(P.rx[i]), float(P.ry[i])
    irx = 1.0 / rx if rx > 0 else 0.0
    iry = 1.0 / ry if ry > 0 else 0.0
    dx0, dy0 = ctx.ox - float(P.cx[i]), ctx.oy - float(P.cy[i])
    return (irx * (c * dx0 - s * dy0), irx * (c * ctx.axc - s * ctx.ayc), irx * (c * ctx.axr - s * ctx.ayr),
            iry * (s * dx0 + c * dy0), iry * (s * ctx.axc + c * ctx.ayc), iry * (s * ctx.axr + c * ctx.ayr))


def row_spans(ctx: "RasterContext", coeffs, shape: int, r0: int, r1: int):
    """Exact per-row column spans of pixel centres inside a rect (|lx|,|ly| <= 1) or ellipse.

    Returns ``(rows, lo, hi)`` with inclusive integer bounds (``hi < lo`` = empty row).
    """
    A0, Ac, Ar, B0, Bc, Br = coeffs
    rows = np.arange(r0, r1, dtype=np.float64)
    a = A0 + Ar * rows
    b = B0 + Br * rows
    with np.errstate(divide="ignore", invalid="ignore"):
        if shape == Shape.RECT:
            lo = np.full(rows.shape, -np.inf)
            hi = np.full(rows.shape, np.inf)
            for k0, kc in ((a, Ac), (b, Bc)):
                if abs(kc) < 1e-300:
                    bad = np.abs(k0) > 1.0
                    lo[bad] = np.inf
                else:
                    c1 = (-1.0 - k0) / kc
                    c2 = (1.0 - k0) / kc
                    lo = np.maximum(lo, np.minimum(c1, c2))
                    hi = np.minimum(hi, np.maximum(c1, c2))
        else:
            qa = Ac * Ac + Bc * Bc
            qb = 2.0 * (a * Ac + b * Bc)
            qc = a * a + b * b - 1.0
            disc = qb * qb - 4.0 * qa * qc
            sq = np.sqrt(np.maximum(disc, 0.0))
            lo = np.where(disc >= 0, (-qb - sq) / (2.0 * qa), np.inf)
            hi = np.where(disc >= 0, (-qb + sq) / (2.0 * qa), -np.inf)
    lo = np.clip(np.ceil(lo - 1e-9), 0, ctx.nx).astype(np.int64)
    hi = np.clip(np.floor(hi + 1e-9), -1, ctx.nx - 1).astype(np.int64)
    return rows.astype(np.int64), lo, hi


def span_pixels(rows, lo, hi):
    """Flatten per-row spans into (row, col) pixel arrays."""
    cnt = np.maximum(hi - lo + 1, 0)
    n = int(cnt.sum())
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rr = np.repeat(rows, cnt)
    start = np.repeat(lo - (np.cumsum(cnt) - cnt), cnt)
    return rr, start + np.arange(n)


def paint_large(ctx: RasterContext, P: PrimitiveSet, i: int, curtain_depth: float, max_px: int = 1 << 19):
    """Scanline-exact painting of one large RECT / ELLIPSE / RING primitive."""
    shape = int(P.shape[i])
    win = ctx.window(P.xmin[i], P.ymin[i], P.xmax[i], P.ymax[i])
    if win is None:
        return
    co = _affine_coeffs(ctx, P, i)
    profile = int(P.profile[i])
    if _nb.AVAILABLE and profile != Profile.CURTAIN:
        _nb.paint_large(ctx.thick, ctx.material, ctx.grain, ctx.nx, win[0], win[1], np.array(co, np.float64),
                        shape == Shape.RECT, float(P.inner_frac[i]) ** 2 if shape == Shape.RING else -1.0,
                        profile, float(P.thickness[i]), np.uint8(P.material[i]), int(P.grain[i]),
                        bool(P.layer[i] != Layer.SUPPORT_FILM))
        return
    sel = np.array([i])
    step = max(1, max_px // max(1, win[3] - win[2]))
    for a in range(win[0], win[1], step):
        rows, lo, hi = row_spans(ctx, co, Shape.RECT if shape == Shape.RECT else Shape.ELLIPSE, a, min(win[1], a + step))
        rr, cc = span_pixels(rows, lo, hi)
        if rr.size == 0:
            continue
        lx = (co[0] + co[1] * cc + co[2] * rr)[None, None, :]
        ly = (co[3] + co[4] * cc + co[5] * rr)[None, None, :]
        r2 = lx * lx + ly * ly
        t = _thickness(P, sel, int(P.profile[i]), lx, ly, r2, curtain_depth).reshape(-1)
        m = t > 0.0
        if shape == Shape.RING:
            inner = float(P.inner_frac[i])
            m &= r2.reshape(-1) >= inner * inner
        if not np.any(m):
            continue
        ctx.apply((rr * ctx.nx + cc)[m], t[m], P.material[i], P.grain[i],
                  overwrite=P.layer[i] != Layer.SUPPORT_FILM, unique=True)


def paint_drawn(ctx: RasterContext, P: PrimitiveSet, idx: np.ndarray, curtain_depth: float = 0.35):
    """Rasterise the primitives ``idx`` of ``P`` (all of one painter layer)."""
    if idx.size == 0:
        return
    # Large rects / ellipses / rings: exact per-row spans instead of a padded window.
    size = np.maximum(P.xmax[idx] - P.xmin[idx], P.ymax[idx] - P.ymin[idx]) / ctx.pixel_um
    big = (size > LARGE_WINDOW_PX) & np.isin(P.shape[idx], (Shape.RECT, Shape.ELLIPSE, Shape.RING))
    for i in idx[big]:
        paint_large(ctx, P, int(i), curtain_depth)
    idx = idx[~big]
    if idx.size == 0:
        return
    shapes = P.shape[idx]
    profiles = P.profile[idx]
    for shape in np.unique(shapes):
        for profile in np.unique(profiles[shapes == shape]):
            grp = idx[(shapes == shape) & (profiles == profile)]
            fast = _nb.AVAILABLE and int(profile) != Profile.CURTAIN
            if not fast:
                ctx.stats["numpy_prims"] = ctx.stats.get("numpy_prims", 0) + 1
            for sub, rows, cols, valid in iter_patches(ctx, P.xmin[grp], P.ymin[grp], P.xmax[grp], P.ymax[grp]):
                sel = grp[sub]
                if fast:
                    _paint_patch_fast(ctx, P, sel, rows, cols, valid, int(shape), int(profile))
                    continue
                lx, ly = _local_frame(ctx, P, sel, rows, cols)
                r2 = lx * lx + ly * ly
                inside = _inside(P, sel, int(shape), lx, ly, r2) & valid
                if not np.any(inside):
                    continue
                t = _thickness(P, sel, int(profile), lx, ly, r2, curtain_depth)
                m = inside & (t > 0.0)
                if not np.any(m):
                    continue
                flat = np.broadcast_to(rows * ctx.nx + cols, m.shape)[m]
                owner = np.broadcast_to(np.arange(len(sel))[:, None, None], m.shape)[m]
                overwrite = P.layer[sel] != Layer.SUPPORT_FILM
                ctx.apply(flat, np.broadcast_to(t, m.shape)[m], P.material[sel][owner],
                          P.grain[sel][owner], overwrite=overwrite[owner])


def _paint_patch_fast(ctx, P, sel, rows, cols, valid, shape: int, profile: int):
    """One patch of `paint_drawn` through :func:`.raster_patch_nb.paint_patch`."""
    n = len(sel)
    ang = -P.rot[sel]
    c = np.cos(ang)
    s = np.sin(ang)
    rx = P.rx[sel]
    ry = P.ry[sel]
    irx = np.where(rx > 0, 1.0 / np.where(rx > 0, rx, 1.0), 0.0)
    iry = np.where(ry > 0, 1.0 / np.where(ry > 0, ry, 1.0), 0.0)
    dx0 = ctx.ox - P.cx[sel]
    dy0 = ctx.oy - P.cy[sel]
    co = np.stack([irx * (c * dx0 - s * dy0), irx * (c * ctx.axc - s * ctx.ayc), irx * (c * ctx.axr - s * ctx.ayr),
                   iry * (s * dx0 + c * dy0), iry * (s * ctx.axc + c * ctx.ayc), iry * (s * ctx.axr + c * ctx.ayr)], 1)
    vb = np.broadcast_to(valid, (n, rows.shape[1], cols.shape[2]))
    hs = vb[:, :, 0].sum(1).astype(np.int64)
    ws = vb[:, 0, :].sum(1).astype(np.int64)
    r0s = np.ascontiguousarray(rows[:, 0, 0], np.int64)
    c0s = np.ascontiguousarray(cols[:, 0, 0], np.int64)
    fj = np.ascontiguousarray(P.facet_j[sel], np.float64).reshape(n, -1)
    _patch_nb.paint_patch(ctx, np.ascontiguousarray(co), r0s, c0s, hs, ws, shape, profile,
                    P.thickness[sel].astype(np.float64), P.inner_frac[sel].astype(np.float64),
                    P.facet_n[sel].astype(np.int64), fj, P.material[sel].astype(np.uint8),
                    P.grain[sel].astype(np.int32), P.layer[sel] != Layer.SUPPORT_FILM, float(ROUGH_PERIOD),
                    (1.0 - ROUGH_PEAK) ** 2, (1.0 + ROUGH_PEAK) ** 2)


def paint_dust(ctx: RasterContext, P: PrimitiveSet, idx: np.ndarray):
    """PaintDust: splat each sub-2-px primitive's mass over the pixels it covers."""
    if idx.size == 0:
        return
    r, c = ctx.to_pixel(P.cx[idx], P.cy[idx])
    x = np.floor(c + 0.5).astype(np.int64)
    y = np.floor(r + 0.5).astype(np.int64)
    ok = (x >= 0) & (y >= 0) & (x < ctx.nx) & (y < ctx.ny)
    if not np.any(ok):
        return
    idx, r, c, x, y = idx[ok], r[ok], c[ok], x[ok], y[ok]
    d = ctx.diameter_px(P.bound_r[idx])
    mean = P.thickness[idx].astype(np.float64) * np.where(P.profile[idx] == Profile.SPHERICAL, 2.0 / 3.0, 1.0)
    # Mass from the primitive's true outline area (the C++ used the bounding circle, which carries
    # a 1.15 slack and the larger semi-axis and so over-weighted dust by ~25-45 %); the footprint
    # the mass is spread over still comes from the bounding diameter.
    px2 = ctx.pixel_um * ctx.pixel_um
    true_area = np.where(P.shape[idx] == Shape.RECT, 4.0, np.pi) * P.rx[idx] * P.ry[idx] / px2
    area = 0.25 * np.pi * d * d
    mass = true_area * mean
    keep = mass > 0
    idx, r, c, x, y, d, area, mass = (a[keep] for a in (idx, r, c, x, y, d, area, mass))
    if idx.size == 0:
        return
    mat = P.material[idx]
    grain = P.grain[idx]
    single = area <= 1.0
    flats = [(y[single] * ctx.nx + x[single])]
    ts = [mass[single]]
    mats = [mat[single]]
    grains = [grain[single]]
    multi = ~single
    if np.any(multi):
        rad = 0.5 * d[multi]
        off = np.arange(-1, 2)
        dy_, dx_ = np.meshgrid(off, off, indexing="ij")
        dy_ = dy_.ravel()[None, :]
        dx_ = dx_.ravel()[None, :]
        px = x[multi][:, None] + dx_
        py = y[multi][:, None] + dy_
        ex = px - c[multi][:, None]
        ey = py - r[multi][:, None]
        wgt = np.clip(0.5 + rad[:, None] - np.sqrt(ex * ex + ey * ey), 0.0, 1.0)
        wsum = wgt.sum(1)
        per = np.where(wsum > 0, mass[multi] / np.where(wsum > 0, wsum, 1.0), 0.0)
        inr = (px >= 0) & (py >= 0) & (px < ctx.nx) & (py < ctx.ny) & (wgt > 0)
        flats.append((py * ctx.nx + px)[inr])
        ts.append((per[:, None] * wgt)[inr])
        mats.append(np.broadcast_to(mat[multi][:, None], wgt.shape)[inr])
        grains.append(np.broadcast_to(grain[multi][:, None], wgt.shape)[inr])
    ctx.apply(np.concatenate(flats), np.concatenate(ts), np.concatenate(mats), np.concatenate(grains),
              overwrite=True, grain_with_claim=True)


def rasterize_scene(scene, ctx: RasterContext, curtain_depth: float = 0.35) -> RasterContext:
    """Holder bulk + film, then primitives with LOD, then structures (FieldMap::Rasterize)."""
    t0 = time.perf_counter()
    scene.holder.paint(ctx)
    t1 = time.perf_counter()
    P = scene.query(ctx)
    vis = P.intersects(ctx.aabb)
    dpx = ctx.diameter_px(P.bound_r)
    culled = dpx < ctx.lod_threshold_px
    dust = vis & ~culled & (dpx < DUST_THRESHOLD_PX)
    draw = vis & ~culled & (dpx >= DUST_THRESHOLD_PX)
    ctx.stats.update(considered=int(P.n), culled=int(np.count_nonzero(~vis | culled)),
                     dust=int(np.count_nonzero(dust)), drawn=int(np.count_nonzero(draw)))
    for layer in sorted(set(P.layer.tolist())):
        on = P.layer == layer
        paint_drawn(ctx, P, np.flatnonzero(on & draw), curtain_depth)
        paint_dust(ctx, P, np.flatnonzero(on & dust))
    t2 = time.perf_counter()
    owners = np.flatnonzero(draw & (P.structure >= 0))
    # Structures run in painter's order (the query is already layer-sorted).
    for i in owners:
        P.structures[P.structure[i]].fill(ctx, P.owner(int(i)))
        ctx.stats["structures"] += 1
    t3 = time.perf_counter()
    ctx.stats.update(holder_ms=1e3 * (t1 - t0), primitives_ms=1e3 * (t2 - t1),
                     structures_ms=1e3 * (t3 - t2), total_ms=1e3 * (t3 - t0))
    return ctx
