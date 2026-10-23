"""Calibration standards: Ted Pella's TEM / STEM calibration and test specimens.

Each product is a preset in :data:`CALIBRATION_STANDARDS` (``"Ted Pella 607-A - ..."``), with
its published numbers in :data:`PRODUCTS` so tests (and users) can check a calibration against
them. The physics they share lives here:

* **Shadowed replicas** (`ShadowedReplicaStructure`): a carbon replica's relief (a grating,
  latex spheres) plus a rough replica surface, shadowed with metal evaporated at an angle. The
  deposit per projected area follows the surface's slope towards the source,
  ``t = t0 (sin e - cos e (grad h . u)) / sin e`` (u: the source azimuth, e: its elevation),
  none where the relief shades the surface from the source. That is the embossed look of a
  shadowed replica: one side of every ridge dark (thick metal), the other bright.
* **Island films** (`island_film`): evaporated metal a few nm thick is not continuous but an
  island film near percolation: worm-like interconnected islands of 5-20 nm. It is a
  band-limited random field thresholded at the local coverage (which follows the local
  deposit, metal conserved), each island a crystal grain of the metal.

A pixel holds one material, so a metal island is that metal at the thickness that attenuates
like the island plus the carbon under it; where islands are smaller than two pixels the metal
is carbon-equivalent thickness, and the relief shading still shows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...hashing import SeedKind, hash_seed, normal_from_hash, splitmix64, uniform_from_hash
from .. import raster_nb as _rnb
from .. import replica_nb as _replica_nb
from ..fieldmap import GRAINS_PER_MATERIAL, grain_id_from_hash
from ..geometry import NM_PER_UM
from ..materials import MaterialId, absorption_lengths_nm
from ..noise import LACUNARITY, cells, fbm
from ..structures import _claim, _owner_pixels, _resolved
from .common import StandardStructure

U64 = np.uint64


# ------------------------------------------------------------------ shared physics
def island_film(
    seed: int,
    X_um,
    Y_um,
    deposit_nm,
    mean_deposit_nm: float,
    island_nm: float,
    coverage: float,
    metal: int,
    grain_nm: float = 2.2,
):
    """An evaporated island film on the pixels (X, Y) (um) with local *deposit_nm*: (inside
    island, island thickness nm, grain id of each inside pixel).

    The shapes are phase-separation stamps (:mod:`..stamps`): round droplets where the
    deposit is low, worms and a percolating labyrinth near half coverage, a film with holes
    above; features `island_nm` apart. The local coverage follows the deposit (``coverage``
    at the mean) and the metal is conserved (island thickness = deposit / local coverage).
    Each island is nanocrystalline (grains of `grain_nm`)."""
    from .. import stamps

    dep = np.asarray(deposit_nm, float)
    rel = dep / max(mean_deposit_nm, 1e-9)
    c = np.clip(coverage * rel**0.8, 0.0, 0.9)
    f = stamps.field(seed, X_um, Y_um, island_nm / NM_PER_UM, c)
    inside = (f > 0.0) & (dep > 0)
    t = np.where(inside, dep / np.maximum(c, 1e-3), 0.0)
    grain = np.zeros(0, np.int32)
    if inside.any():  # nanocrystalline: the nearest of jittered sites `grain_nm` apart
        G = grain_nm / NM_PER_UM
        _, _, h = cells(seed ^ 0x1F, X_um[inside] / G, Y_um[inside] / G, 0.0)
        grain = grain_id_from_hash(metal, h)
    return inside, t, grain


_LEVEL_LUT: list = []


def _lattice_index(ctx, r: int, c: int) -> tuple[int, int]:
    """(column, row) of raster pixel (r, c) on the view's world-fixed pixel lattice (the
    renderer's `_lattice`: pixel units along the view's own axes from the world origin)."""
    view = ctx.view
    x, y = ctx.world(float(r), float(c))
    x, y = float(x), float(y)
    rot = float(getattr(view, "rotation_rad", 0.0))
    if rot:
        cr, sr = math.cos(rot), math.sin(rot)
        x, y = cr * x + sr * y, -sr * x + cr * y
    sx = -1.0 if getattr(view, "flip_x", False) else 1.0
    sy = -1.0 if getattr(view, "flip_y", False) else 1.0
    lc = sx * x * max(view.cos_beta, 0.1) / view.pixel_um
    lr = sy * y * max(view.cos_alpha, 0.1) / view.pixel_um
    return math.floor(lc + 1e-6), math.floor(lr + 1e-6)


def _level_lut(cov, half, levels, table):
    """`stamps._level_offset` on a fine coverage grid (built once: 8192 x 3 floats)."""
    if not _LEVEL_LUT:
        _LEVEL_LUT.append(
            _replica_nb.level_offset_lut(
                cov,
                half,
                levels,
                table,
                __import__("de_twin.specimen.stamps", fromlist=["_OFFSETS"])._OFFSETS,
            )
        )
    return _LEVEL_LUT[0]


def carbon_equivalent(metal: int, ht_kv: float = 200.0) -> float:
    """Carbon nm that attenuates like 1 nm of *metal*."""
    lam = absorption_lengths_nm(ht_kv)
    return float(lam[int(MaterialId.AMORPHOUS_CARBON)] / lam[int(metal)])


# ------------------------------------------------------------------ reliefs
#: How much more slowly the line waviness varies across the lines than along them.
WAVY_ACROSS = 0.1


class WaffleGrating:
    """A replica cross grating: raised lines along x and y (the replica of a ruled master's
    grooves), `line_fraction` of the period wide with sloped flanks, rough edges, a long-range
    waviness that averages out (the mean period is exact) and, for the old style, defects:
    missing and doubled line segments."""

    def __init__(
        self,
        period_um: float,
        depth_nm: float = 30.0,
        line_fraction: float = 0.22,
        ramp_nm: float = 25.0,
        edge_nm: float = 5.0,
        edge_corr_nm: float = 60.0,
        wavy_nm: float = 15.0,
        wavy_um: float = 1.5,
        defects: float = 0.0,
        both: bool = True,
        trench: bool = False,
        back_ramp_nm: float | None = None,
    ):
        self.P, self.depth, self.w = float(period_um), float(depth_nm), float(line_fraction)
        self.ramp, self.edge, self.edge_corr = (
            ramp_nm / NM_PER_UM,
            edge_nm / NM_PER_UM,
            edge_corr_nm / NM_PER_UM,
        )
        # a ruled master's lines are sawtooth: a steep front wall (+x / +y side, `ramp_nm`
        # wide in projection) and a gentler back slope (`back_ramp_nm`)
        self.back = (back_ramp_nm if back_ramp_nm is not None else ramp_nm) / NM_PER_UM
        self.wavy, self.wavy_um, self.defects = wavy_nm / NM_PER_UM, float(wavy_um), float(defects)
        self.both, self.trench = bool(both), bool(trench)

    def mean_nm(self) -> float:
        f = self.w if not self.both else 1.0 - (1.0 - self.w) ** 2
        return self.depth * (1.0 - f if self.trench else f)

    def warp(self, seed, lx, ly):
        """Where (lx, ly) falls on the ideal grating: lines wiggle sideways (waviness that
        varies along a line and only slowly across lines, so the pitch stays exact) and their
        edges are rough (a short-range sideways jitter)."""
        u, v = lx, ly
        a = WAVY_ACROSS
        if self.wavy > 0:
            u = u + self.wavy * fbm(seed ^ 0x51, a * lx, ly, self.wavy_um, 2)
            v = v + self.wavy * fbm(seed ^ 0x52, lx, a * ly, self.wavy_um, 2)
        if self.edge > 0:
            u = u + self.edge * fbm(seed ^ 0x61, 0.3 * lx, ly, self.edge_corr, 2)
            v = v + self.edge * fbm(seed ^ 0x62, lx, 0.3 * ly, self.edge_corr, 2)
        return u, v

    def height_nm(self, seed, u, v, rough: bool = True) -> np.ndarray:
        """Relief height (nm) at warped owner-local (u, v) um."""
        P, half = self.P, 0.5 * self.w * self.P
        out = None
        axes = ((u, v, 0x61), (v, u, 0x62)) if self.both else ((u, v, 0x61),)
        for a, b, salt in axes:
            line = np.floor(a / P)
            sd = a - (line + 0.5) * P  # > 0 on the front (+) side of the line's centre
            d = np.abs(sd)
            if rough and self.edge > 0:
                near = np.abs(d - half) < 0.5 * self.ramp + 2.0 * self.edge
                if near.any():
                    d = d.copy()
                    d[near] += self.edge * fbm(
                        seed ^ salt, b[near], line[near] * 0.731 * P, self.edge_corr, 2
                    )
            ramp = np.where(sd > 0, self.ramp, self.back)
            r = np.clip((half - d) / np.maximum(ramp, 1e-9) + 0.5, 0.0, 1.0)
            if rough and self.defects > 0:  # missing segments of a line (old-style gratings)
                seg = np.floor(b / (2.0 * P))
                gone = (
                    uniform_from_hash(
                        hash_seed(
                            seed ^ salt,
                            SeedKind.STRUCTURE,
                            (line.astype(np.int64) * 100003 + seg.astype(np.int64)) & 0xFFFFFFFF,
                            0x33,
                        )
                    )
                    < self.defects
                )
                r = np.where(gone, 0.0, r)
            out = r if out is None else np.maximum(out, r)
        h = self.depth * out
        return self.depth - h if self.trench else h


class ReliefTile:
    """One period of a grating's shadowed relief, computed once: height, slope and the lit
    fraction (cast shadows with the source's penumbra) on an N x N grid over the period.
    The relief is periodic, so its shadows are too; a pixel only looks them up."""

    N = 384

    def __init__(self, relief: WaffleGrating, elev: float, to_source: tuple, penumbra: float):
        from scipy import ndimage

        n, P = self.N, relief.P
        self.P = P
        c = (np.arange(n) + 0.5) * P / n
        U, V = np.meshgrid(c, c)  # [row = v, col = u]
        h = relief.height_nm(0, U, V, rough=False).astype(np.float64)
        d = P / n * NM_PER_UM  # nm per tile pixel
        gx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * d)
        gy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) / (2 * d)
        ux, uy = to_source
        tan_lo = math.tan(max(elev - 0.5 * penumbra, 0.02))
        tan_hi = math.tan(elev + 0.5 * penumbra)
        reach = relief.depth / tan_lo  # nm
        worst = np.full(h.shape, -np.inf)
        rows, cols = np.mgrid[0:n, 0:n].astype(np.float64)
        for s_nm in np.arange(d, reach + 2 * d, 1.5 * d):
            hs = ndimage.map_coordinates(
                h, [rows + uy * s_nm / d, cols + ux * s_nm / d], order=1, mode="grid-wrap"
            )
            np.maximum(worst, (hs - h - 0.5) / s_nm, out=worst)
        lit = np.clip((tan_hi - worst) / (tan_hi - tan_lo), 0.0, 1.0)
        self.h, self.gx, self.gy, self.lit = (np.ascontiguousarray(a, np.float32) for a in (h, gx, gy, lit))

    def sample(self, u, v):
        """(h nm, dh/dx, dh/dy, lit) at relief coordinates (u, v) um, bilinear and periodic."""
        from ..stamps import sample_periodic

        if getattr(self, "_stack", None) is None:
            self._stack = np.stack([self.h, self.gx, self.gy, self.lit])
        return list(sample_periodic(self._stack, np.asarray(u) / self.P, np.asarray(v) / self.P))


@dataclass
class LatexSpheres:
    """Polystyrene latex spheres scattered on the replica (``density`` per um^2), e.g. the
    0.261 um spheres of the 603 / 673 grating standards: polymer thickness, and relief that
    shades the metal behind them."""

    diameter_nm: float
    density_per_um2: float
    sigma_frac: float = 0.02  # coefficient of variation of the diameter (normal)

    def cell_um(self) -> float:
        """A lattice finer than the mean spacing, sparsely occupied, so the spheres lie at
        random rather than on a grid (but never overlap)."""
        spacing = 1.0 / math.sqrt(max(self.density_per_um2, 1e-6))
        # sites stay within 0.2 cells of their cell centre (FastLattice jitter 0.4), so they are
        # >= 0.6 cells apart: a sphere up to 0.6 cells across never overlaps its neighbours
        return max(0.5 * spacing, 1.7 * self.diameter_nm * (1.0 + 3.0 * self.sigma_frac) / NM_PER_UM)

    def spheres(self, seed, lattice, hit):
        """(present, radius um, centre x, centre y) at the nearest lattice sites of *hit*."""
        h = hit.h
        present = (h % U64(1024)) < U64(int(min(1.0, self.density_per_um2 * self.cell_um() ** 2) * 1024))
        rj = 1.0 + self.sigma_frac * np.clip(normal_from_hash(splitmix64(h)), -3.0, 3.0)
        return present, 0.5 * self.diameter_nm / NM_PER_UM * rj, hit.sx, hit.sy


# ------------------------------------------------------------------ the replica structure
#: The replica texture's slope (which sets the angled metal deposit) from the noise's own
#: derivative, one evaluation per pixel; False: finite differences over 2 nm, three (the
#: NumPy fill's arithmetic).
ANALYTIC_SLOPE = True
#: In that mode the replica's aperiodic noise is evaluated on node grids this many nodes per
#: wavelength of each field's finest octave and interpolated per pixel.
NODES_PER_WAVELENGTH = 12

#: Angular size of the evaporation source seen from the specimen (radians): the penumbra.
SHADOW_PENUMBRA_RAD = math.radians(4.0)


class ShadowedReplicaStructure(StandardStructure):
    """A shadowed carbon replica (see the module docstring).

    The replica is a carbon film `base_nm` thick over the surface: `relief` (a grating, or
    None for a flat film), a rough "crumpled" surface (`rough_nm` over `rough_um`, fibrous at
    10kx), a long-wavelength undulation (`crumple_nm` over `crumple_um`) and optional latex
    spheres. The carbon is evaporated from straight above, so its projected thickness is the
    same everywhere, walls included. The contrast is the metal, evaporated at `elevation_deg` from
    `azimuth_deg`, `metal_nm` thick on a surface facing the source. It lands per projected
    area as ``metal_nm (sin e - cos e grad h . u)``, and not at all where the relief hides the
    surface: the embossed look at low magnification. At high magnification the deposit is an
    island film whose coverage follows it: dense worm-like islands where the surface faces
    the source, sparse droplets where it faces away.
    """

    required_layers = frozenset()

    def __init__(
        self,
        relief: WaffleGrating | None,
        base_nm: float = 20.0,
        rough_nm: float = 3.0,
        rough_um: float = 0.06,
        crumple_nm: float = 12.0,
        crumple_um: float = 0.3,
        metal: int = MaterialId.GOLD,
        metal_nm: float = 6.0,
        elevation_deg: float = 25.0,
        azimuth_deg: float = 45.0,
        island_nm: float = 9.0,
        coverage: float = 0.65,
        grain_nm: float = 2.2,
        shadow_leak: float = 0.12,
        spheres: LatexSpheres | None = None,
        sphere_material: int = MaterialId.PROTEIN,
        spheres_shadowed: bool = True,
    ):
        self.relief = relief
        self.base, self.rough_nm, self.rough_um = float(base_nm), float(rough_nm), float(rough_um)
        self.crumple_nm, self.crumple_um = float(crumple_nm), float(crumple_um)
        self.metal, self.metal_nm = int(metal), float(metal_nm)
        self.elev = math.radians(float(elevation_deg))
        az = math.radians(float(azimuth_deg))
        self.u = (math.cos(az), math.sin(az))
        self.island_nm, self.coverage, self.grain_nm = float(island_nm), float(coverage), float(grain_nm)
        self.shadow_leak = float(shadow_leak)  # metal that still reaches a shadow (source size, migration)
        # False: the latex went on after the shadowing (603, 673): no metal on it, no shadow
        self.spheres_shadowed = bool(spheres_shadowed)
        self.spheres, self.sphere_material = spheres, int(sphere_material)

    def base_nm(self) -> float:
        return self.mean_carbon_nm()

    def mean_nm(self) -> float:
        t = self.mean_carbon_nm()
        if self.spheres is not None:  # latex volume per area, as carbon
            r = 0.5 * self.spheres.diameter_nm
            t += (
                self.spheres.density_per_um2
                / NM_PER_UM**2
                * 4.0
                / 3.0
                * math.pi
                * r**3
                * (carbon_equivalent(self.sphere_material))
            )
        if self.metal_nm > 0:
            t += self.flat_deposit_nm() * carbon_equivalent(self.metal)
        return t

    def flat_deposit_nm(self) -> float:
        """Metal per projected area on a flat, level surface."""
        return self.metal_nm * math.sin(self.elev)

    def mean_carbon_nm(self) -> float:
        """Area-mean projected carbon: the replica film (evaporated from straight above, so
        the same per projected area everywhere)."""
        return self.base

    # -------------------------------------------------------------- the surface
    def _surface(self, seed, x, y, rough: bool, coarse: bool = False):
        """Surface height (nm) at owner-local (x, y) um, without the spheres."""
        if self.relief is not None:
            u, v = self.relief.warp(seed, x, y)
            h = self.relief.height_nm(seed, u, v, rough=not coarse)
        else:
            h = np.zeros(np.shape(x))
        if self.crumple_nm > 0:
            h = h + self.crumple_nm * fbm(seed ^ 0x93, x, y, self.crumple_um, 2)
        if rough and not coarse and self.rough_nm > 0:
            h = h + self.rough_nm * fbm(seed ^ 0x91, x, y, self.rough_um, 4, 0.6)
        return h

    def _sphere_set(self, seed, lx, ly):
        """Every latex sphere that can touch (or shade) the pixels at (lx, ly), as a binned
        :class:`..spheres.SphereSet` (um)."""
        from ..spheres import SphereSet, lattice_sites

        sp = self.spheres
        R0 = 0.5 * sp.diameter_nm / NM_PER_UM
        shade = (
            2.0 * R0 * (1.0 + 3 * sp.sigma_frac) / max(math.tan(self.elev), 1e-3)
            if self.spheres_shadowed
            else 0.0
        )
        reach = 2.0 * R0 + shade
        x0, x1 = float(np.min(lx)) - reach, float(np.max(lx)) + reach
        y0, y1 = float(np.min(ly)) - reach, float(np.max(ly)) + reach
        cell = sp.cell_um()
        xs, ys, hs = lattice_sites(seed ^ 0x2F, cell, x0, y0, x1, y1, 0.4)
        present = uniform_from_hash(hs ^ U64(0x71)) < min(1.0, sp.density_per_um2 * cell * cell)
        rj = 1.0 + sp.sigma_frac * np.clip(normal_from_hash(hs[present]), -3.0, 3.0)
        return SphereSet(xs[present], ys[present], R0 * rj)

    def tile(self) -> ReliefTile | None:
        if self.relief is None:
            return None
        if getattr(self, "_tile", None) is None:
            self._tile = ReliefTile(self.relief, self.elev, self.u, SHADOW_PENUMBRA_RAD)
        return self._tile

    def _texture(self, seed, x, y, rough: bool):
        """The replica's own surface (the crumple and the fine roughness): its slope (nm/nm)
        by finite differences of the noise."""
        e = 0.002  # um

        def h(xx, yy):
            out = np.zeros(np.shape(xx))
            if self.crumple_nm > 0:
                out = out + self.crumple_nm * fbm(seed ^ 0x93, xx, yy, self.crumple_um, 2)
            if rough and self.rough_nm > 0:
                out = out + self.rough_nm * fbm(seed ^ 0x91, xx, yy, self.rough_um, 4, 0.6)
            return out

        h0 = h(x, y)
        return h0, (h(x + e, y) - h0) / (e * NM_PER_UM), (h(x, y + e) - h0) / (e * NM_PER_UM)

    def numba_fill(self) -> bool:
        return _rnb.AVAILABLE and (self.crumple_nm > 0 or self.rough_nm > 0)

    def _fill_fast(self, ctx, owner, seed, px, mean_c, flat, rough, islands, tile):
        """`fill` through the numba kernel (:func:`..replica_nb.replica_block`): the same
        per-pixel arithmetic, in parallel rows."""
        from ..noise import fbm_params
        from ..stamps import _OFFSETS, LABYRINTH, _coverage_table, period_px_cached, phase_tiles

        b = owner.bounds
        win = ctx.window(b.xmin, b.ymin, b.xmax, b.ymax)
        if win is None:
            return
        r0, r1, c0, c1 = win
        relief = self.relief
        specs = [
            (seed ^ 0x51, relief.wavy_um if relief else 1.0, 2, 0.5),
            (seed ^ 0x52, relief.wavy_um if relief else 1.0, 2, 0.5),
            (seed ^ 0x61, relief.edge_corr if relief else 1.0, 2, 0.5),
            (seed ^ 0x62, relief.edge_corr if relief else 1.0, 2, 0.5),
            (seed ^ 0x93, self.crumple_um, 2, 0.5),
            (seed ^ 0x91, self.rough_um, 4, 0.6),
        ]
        FS = np.zeros((6, 4), np.uint64)
        FC, FSN, FA = (np.zeros((6, 4)) for _ in range(3))
        FN = np.zeros(6, np.int64)
        for k, (sd, sc, octs, gain) in enumerate(specs):
            a, c, sn, am = fbm_params(sd, sc, octs, gain)
            FS[k, :octs], FC[k, :octs], FSN[k, :octs], FA[k, :octs], FN[k] = a, c, sn, am, octs
        has_relief = tile is not None
        spheres = self.spheres is not None
        FL = np.array(
            [
                has_relief,
                self.crumple_nm > 0,
                rough and self.rough_nm > 0,
                spheres,
                self.spheres_shadowed,
                self.metal_nm > 0,
                islands,
                bool(relief and relief.wavy > 0),
                bool(relief and relief.edge > 0),
                ANALYTIC_SLOPE,
            ],
            np.bool_,
        )
        R = np.array(
            [
                relief.wavy if relief else 0.0,
                relief.edge if relief else 0.0,
                relief.P if relief else 1.0,
                WAVY_ACROSS,
            ]
        )
        stack = (
            np.stack([tile.h, tile.gx, tile.gy, tile.lit]) if has_relief else np.zeros((4, 1, 1), np.float32)
        )
        TX = np.array([self.crumple_nm, self.rough_nm, 0.002])
        ux, uy = self.u
        D = np.array(
            [
                self.metal_nm,
                math.sin(self.elev),
                math.cos(self.elev),
                ux,
                uy,
                flat,
                self.shadow_leak,
                self.base,
                mean_c,
                carbon_equivalent(self.sphere_material),
            ]
        )
        M = np.array(
            [self.metal, self.sphere_material, int(MaterialId.AMORPHOUS_CARBON), GRAINS_PER_MATERIAL],
            np.int64,
        )
        tiles, cov = phase_tiles()
        half = int(np.argmin(np.abs(cov - LABYRINTH)))
        levels, table = _coverage_table()
        feat = self.island_nm / NM_PER_UM
        I = np.array(
            [
                self.coverage,
                period_px_cached() / feat,
                1.0 / (12.0 * feat),
                6.0,
                self.grain_nm / NM_PER_UM,
                cov[1],
                cov[half],
            ]
        )
        fseed = np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)
        cseed = np.uint64(((int(seed) ^ 0x1F) * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF)
        covs = np.ascontiguousarray(cov[1 : half + 1], np.float64)
        lvx = np.arange(1, half + 1, dtype=np.float64)
        W = np.array([ctx.ox, ctx.oy, ctx.axc, ctx.ayc, ctx.axr, ctx.ayr])
        O = np.array([owner.cx, owner.cy, math.cos(-owner.rot), math.sin(-owner.rot), owner.rx, owner.ry])
        new_under = ctx.under_thick is None
        uthick = np.zeros(ctx.npix, np.float32) if new_under else ctx.under_thick
        umat = np.zeros(ctx.npix, np.uint8) if new_under else ctx.under_material
        used = np.zeros(1, np.uint8)
        bstep = max(1, ctx.BLOCK_PIXELS // max(1, c1 - c0))  # ctx.row_blocks
        nblk = (r1 - r0 + bstep - 1) // bstep if spheres else 1
        SPB = np.zeros((nblk, 11))
        SNB = np.ones((nblk, 2), np.int64)
        BSO = np.zeros((nblk, 2), np.int64)
        cxs, cys, rs, starts, members = [], [], [], [], []
        n_sph = n_mem = 0
        if spheres:  # the spheres of each row block's pixels, as the NumPy fill builds them per block
            ce, se = math.cos(self.elev), math.sin(self.elev)
            step = max(px, 1.0 / NM_PER_UM)
            for kb, (a, bb) in enumerate(ctx.row_blocks(r0, r1, c1 - c0)):
                rows = np.arange(a, bb)[:, None]
                cols = np.arange(c0, c1)[None, :]
                X, Y = ctx.world(rows, cols)
                dx, dy = X - owner.cx, Y - owner.cy
                lx = dx * O[2] - dy * O[3]
                ly = dx * O[3] + dy * O[2]
                m = (np.abs(lx) <= owner.rx) & (np.abs(ly) <= owner.ry)
                s0 = sum(len(x) for x in starts)
                if not m.any():
                    starts.append(np.zeros(2, np.int64) + n_mem)
                    BSO[kb] = (s0, s0 + 2)
                    continue
                ss = self._sphere_set(seed, lx[m], ly[m])
                rmax = float(ss.r.max()) if ss.r.size else 0.0
                reach = 2.0 * rmax / max(math.tan(self.elev), 1e-3) + rmax
                SPB[kb] = (
                    ss.x0,
                    ss.y0,
                    1.0 / ss.bin,
                    step,
                    step * NM_PER_UM,
                    ux * ce,
                    uy * ce,
                    se,
                    reach,
                    0.5 * ss.bin,
                    SHADOW_PENUMBRA_RAD,
                )
                SNB[kb] = (ss.nx, ss.ny)
                cxs.append(ss.cx)
                cys.append(ss.cy)
                rs.append(ss.r)
                starts.append(ss.start + n_mem)
                members.append(ss.members + n_sph)
                BSO[kb] = (s0, s0 + len(ss.start))
                n_sph += ss.cx.size
                n_mem += ss.members.size
        cat = lambda xs, dt: np.ascontiguousarray(np.concatenate(xs), dt) if xs else np.zeros(1, dt)
        sarr = (
            cat(cxs, np.float64),
            cat(cys, np.float64),
            cat(rs, np.float64),
            cat(starts, np.int64),
            cat(members, np.int64),
        )
        # node grids of the aperiodic noise (analytic mode): strides a fraction of each field's
        # finest octave, in raster pixels (1: every pixel, as at low magnification)
        finest = [
            ((relief.wavy_um if relief else 1.0) / LACUNARITY, bool(FL[7])),
            ((relief.edge_corr if relief else 1.0) / LACUNARITY, bool(FL[8])),
            (self.crumple_um / LACUNARITY, bool(FL[1])),
            (self.rough_um / LACUNARITY**3, bool(FL[2])),
        ]
        ST = np.array(
            [
                max(1, min(32, int(sc / (px * NODES_PER_WAVELENGTH)))) if (on and ANALYTIC_SLOPE) else 1
                for sc, on in finest
            ],
            np.int64,
        )
        # the node lattice is world-fixed: raster pixel (r0, c0) is lattice pixel (gr, gc) of the
        # view's pixel lattice, and nodes sit on its multiples of s, so the pieces of a raster
        # (strips, the part a stage move brings in) interpolate the same nodes as the whole
        gc, gr = _lattice_index(ctx, r0, c0)
        OFF = np.array([[gr - (gr // s) * s, gc - (gc // s) * s] for s in ST], np.int64)
        nodes = []
        for q, (s, (_, on)) in enumerate(zip(ST, finest)):
            shape = (
                ((r1 - 1 - r0 + OFF[q, 0]) // s + 2, (c1 - 1 - c0 + OFF[q, 1]) // s + 2, 2 if q < 2 else 3)
                if (on and ANALYTIC_SLOPE and s > 1)
                else (1, 1, 3)
            )  # stride 1: evaluated per pixel
            nodes.append(np.zeros(shape))
        if ANALYTIC_SLOPE:
            _replica_nb.replica_nodes(r0, r1, c0, c1, W, O, FS, FC, FSN, FA, FN, R, TX, ST, OFF, *nodes)
        _replica_nb.replica_block(
            ctx.thick,
            ctx.material,
            ctx.grain,
            uthick,
            umat,
            used,
            ctx.nx,
            r0,
            r1,
            c0,
            c1,
            W,
            O,
            FL,
            R,
            stack,
            FS,
            FC,
            FSN,
            FA,
            FN,
            TX,
            *sarr,
            SPB,
            SNB,
            BSO,
            bstep,
            D,
            M,
            I,
            tiles,
            fseed,
            cseed,
            covs,
            lvx,
            levels,
            table,
            _OFFSETS,
            ST,
            OFF,
            *nodes,
            *_level_lut(cov, half, levels, table),
        )
        if new_under and used[0]:
            ctx.under_material, ctx.under_thick = umat, uthick

    def fill(self, ctx, owner):
        seed = int(hash_seed(owner.seed, SeedKind.STRUCTURE, 0)) & 0xFFFFFFFF
        px = ctx.pixel_um
        scales = [
            self.crumple_um if self.crumple_nm > 0 else 1e9,
            self.relief.P if self.relief is not None else 1e9,
            self.spheres.diameter_nm / NM_PER_UM if self.spheres is not None else 1e9,
        ]
        if not _resolved(min(scales), ctx):
            return  # the owner carries the mean
        mean_c = self.mean_carbon_nm()
        flat = self.flat_deposit_nm()
        rough = (self.rough_nm > 0 or self.crumple_nm > 0) and _resolved(
            min(self.rough_um, self.crumple_um) / 2.0, ctx
        )
        islands = self.metal_nm > 0 and _resolved(self.island_nm / NM_PER_UM, ctx)
        tile = self.tile()
        ux, uy = self.u
        sin_e, cos_e = math.sin(self.elev), math.cos(self.elev)
        if _rnb.AVAILABLE and (rough or self.crumple_nm > 0):
            self._fill_fast(ctx, owner, seed, px, mean_c, flat, rough, islands, tile)
            ctx.stats["fast_fills"] = ctx.stats.get("fast_fills", 0) + 1
            return
        for B in _owner_pixels(ctx, owner):
            lx, ly = B.lx, B.ly
            # the relief (looked up) and the replica's own crumpled surface (noise)
            if tile is not None:
                u, v = self.relief.warp(seed, lx, ly)
                h, gx, gy, lit = tile.sample(u, v)
            else:
                h = gx = gy = np.zeros(lx.shape, np.float32)
                lit = np.ones(lx.shape, np.float32)
            if rough or self.crumple_nm > 0:
                hn, nx_, ny_ = self._texture(seed, lx, ly, rough)
                h, gx, gy = h + hn, gx + nx_, gy + ny_
            # the replica's carbon is evaporated straight down: the same thickness per
            # projected area on flats and walls alike (only the angled metal marks the walls)
            carbon = np.full(lx.shape, self.base)
            chord = np.zeros(lx.shape)
            ss = self._sphere_set(seed, lx, ly) if self.spheres is not None else None
            if ss is not None:
                top, chord = (a * NM_PER_UM for a in ss.top_chord(lx, ly))
                on = top > 0
                if on.any() and self.spheres_shadowed:
                    step = max(px, 1.0 / NM_PER_UM)
                    dn = step * NM_PER_UM
                    tx = ss.top_chord(lx + step, ly)[0] * NM_PER_UM
                    ty = ss.top_chord(lx, ly + step)[0] * NM_PER_UM
                    # the sphere's own surface, its slope capped where the rim is vertical
                    gx = np.where(on, np.clip((tx - top) / dn, -8.0, 8.0), gx)
                    gy = np.where(on, np.clip((ty - top) / dn, -8.0, 8.0), gy)
                    h = np.where(on, h + top, h)
            if self.metal_nm > 0:
                dep = self.metal_nm * np.maximum(sin_e - cos_e * (gx * ux + gy * uy), 0.0)
                dep = np.minimum(dep, 12.0 * flat)  # a near-vertical wall facing the source: a lot, projected
                if ss is not None and self.spheres_shadowed:
                    occ = ss.occlusion(lx, ly, h / NM_PER_UM, self.u, self.elev, SHADOW_PENUMBRA_RAD)
                    lit = np.minimum(lit, 1.0 - occ)
                dep = dep * lit + self.shadow_leak * flat * (1.0 - lit)
            else:
                dep = np.zeros(lx.shape)
            # layers: the replica film (carbon) with latex on it, and the metal on top. A pixel
            # with metal holds the metal, the film and latex go under it.
            latex = chord > 0
            ctx.add_thickness(B.flat, carbon - mean_c)
            if latex.any():  # latex on the film: latex on top, the film under it
                fl = B.sub(latex)
                ctx.add_thickness(fl, chord[latex] - carbon[latex])
                ctx.material[fl] = self.sphere_material
                ctx.add_under(fl, MaterialId.AMORPHOUS_CARBON, carbon[latex])
            if self.metal_nm <= 0:
                continue
            if islands:
                inside, t_m, grain = island_film(
                    seed, lx, ly, dep, flat, self.island_nm, self.coverage, self.metal, self.grain_nm
                )
            else:  # islands too fine to see: the deposit as a continuous metal layer
                inside, t_m, grain = dep > 0, dep, np.full(int((dep > 0).sum()), -1, np.int32)
            if inside.any():
                f = B.sub(inside)
                # the metal on top; whatever was there (film, or latex on film) goes under
                below = ctx.thick[f].astype(np.float64)
                on_latex = latex[inside]
                ctx.add_thickness(f, t_m[inside] - below)
                if (~on_latex).any():
                    ctx.add_under(f[~on_latex], MaterialId.AMORPHOUS_CARBON, below[~on_latex])
                if on_latex.any():  # the film under the latex counted as latex
                    tl = chord[inside][on_latex] + carbon[inside][on_latex] * carbon_equivalent(
                        self.sphere_material
                    )
                    ctx.add_under(f[on_latex], self.sphere_material, tl)
                _claim(ctx, f, grain, self.metal)
