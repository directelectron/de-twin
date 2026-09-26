"""Interior fills of placed items (port of ``Structure.cpp``, ``PreparationAnalytic.cpp`` and
``PreparationTime.cpp`` structures), vectorised over the owner's raster window.

Each structure is a small object with real constructor attributes (the C++ used pointer-keyed
side tables and smuggled primitive fields because its headers were frozen). ``fill(ctx, owner)``
writes into a :class:`~de_twin.specimen.raster.RasterContext`; ``owner.seed`` is the owning
primitive's sub-seed. Every structure honours the low-magnification floor: below
``ANALYTIC_CELL_MIN_PX`` (2) raster pixels per lattice cell it writes the area-averaged value.

Grain ids are always global (``grain_id_from_hash(material, cell_hash)``), so the renderer's
GrainTable lookup (orientation, nucleation order) is consistent everywhere.
"""

from __future__ import annotations

import functools
import math
from typing import Sequence

import numpy as np

from ..hashing import SeedKind, hash_seed, mix_cell, uniform_from_hash
from .fieldmap import LAYER_DESCAN, LAYER_STRAIN, grain_id_from_hash
from .geometry import (ANALYTIC_CELL_MIN_PX, NM_PER_UM, BlockHit, JitteredLattice, inside_rough_ellipse,
                       lognormal_from_hash, rough_ellipse_signed_dist, smoothstep, value_noise,
                       value_noise_separable, ROUGH_PEAK)
from .materials import MaterialId
from .raster import Owner, RasterContext, Shape

U64 = np.uint64
EMPTY_THICKNESS_NM = 0.05
GRAIN_STEP_MIX = U64(0x9E37)


def _resolved(cell_um: float, ctx: RasterContext) -> bool:
    return cell_um > 0 and cell_um / ctx.pixel_um >= ANALYTIC_CELL_MIN_PX


class _Block:
    """Pixels of one raster row block inside an owner.

    ``flat`` indexes the flat raster layers: a ``slice`` when the block covers whole raster rows
    entirely inside the owner (the common high-magnification case), else an index array.
    ``mask`` is ``None`` for a fully covered block, else the 2-D inside mask of the window.
    """

    __slots__ = ("flat", "X", "Y", "lx", "ly", "win", "mask")

    def __init__(self, flat, X, Y, lx, ly, win, mask):
        self.flat, self.X, self.Y, self.lx, self.ly, self.win, self.mask = flat, X, Y, lx, ly, win, mask

    def nearest(self, ctx: RasterContext, lattice: JitteredLattice, offset=(0.0, 0.0)):
        """Exact nearest lattice site for the block's pixels (coarse-to-fine, see geometry)."""
        hit = lattice.nearest_block(ctx, *self.win, offset=offset)
        if self.mask is None:
            return BlockHit(hit.site.ravel(), hit.grid, hit.X.ravel(), hit.Y.ravel())
        return hit.masked(self.mask)

    def noise(self, ctx: RasterContext, seed, scale: float):
        """``value_noise(seed, X * scale, Y * scale)`` for the block's pixels."""
        r0, r1, c0, c1 = self.win
        if ctx.axis_aligned:
            xs = (ctx.ox + ctx.axc * np.arange(c0, c1)) * scale
            ys = (ctx.oy + ctx.ayr * np.arange(r0, r1)) * scale
            n = value_noise_separable(seed, xs, ys)
            return n.ravel() if self.mask is None else n[self.mask]
        return value_noise(seed, self.X * scale, self.Y * scale)

    def sub(self, m):
        """Flat raster indices of the block pixels selected by boolean ``m``."""
        if isinstance(self.flat, slice):
            return np.flatnonzero(m) + self.flat.start
        return self.flat[m]


def _owner_pixels(ctx: RasterContext, owner: Owner, rect: bool = True, bounds_only: bool = False):
    """Yield a :class:`_Block` per raster row block of the owner's window.

    ``lx``/``ly`` are un-normalised offsets in the owner's un-rotated frame (micrometres).
    """
    b = owner.bounds
    win = ctx.window(b.xmin, b.ymin, b.xmax, b.ymax)
    if win is None:
        return
    c, s = math.cos(-owner.rot), math.sin(-owner.rot)
    for flat, X, Y, sub in ctx.iter_window(win):
        dx = X - owner.cx
        dy = Y - owner.cy
        lx = dx * c - dy * s
        ly = dx * s + dy * c
        if bounds_only:
            m = b.contains(X, Y)
        elif rect:
            m = (np.abs(lx) <= owner.rx) & (np.abs(ly) <= owner.ry)
        else:
            m = (lx / owner.rx) ** 2 + (ly / owner.ry) ** 2 <= 1.0
        if m.all():
            r0, r1, c0, c1 = sub
            f = slice(r0 * ctx.nx, r1 * ctx.nx) if (c0 == 0 and c1 == ctx.nx) else flat.ravel()
            yield _Block(f, X.ravel(), Y.ravel(), lx.ravel(), ly.ravel(), sub, None)
        elif m.any():
            yield _Block(flat[m], X[m], Y[m], lx[m], ly[m], sub, m)


def _claim(ctx: RasterContext, flat, grain, material):
    ctx.grain[flat] = grain
    ctx.material[flat] = material


def _matrix_claim(ctx: RasterContext, owner: Owner, flat):
    """The owner's own grain on the pixels that already carry the owner's material."""
    if owner.grain < 0:
        return
    m = ctx.material[flat] == owner.material
    if isinstance(flat, slice):
        ctx.grain[flat][m] = owner.grain
    else:
        ctx.grain[flat[m]] = owner.grain


class Structure:
    required_layers: frozenset = frozenset()
    time_dependent: bool = False

    def fill(self, ctx: RasterContext, owner: Owner) -> None:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------
# Proteins (design 5.1)
# ---------------------------------------------------------------------------------------------

PROTEIN_MIN_D_NM, PROTEIN_MAX_D_NM = 5.0, 30.0
PROTEIN_MAX_R_UM = 0.5 * PROTEIN_MAX_D_NM / NM_PER_UM
PROTEIN_PATCH_CELL_UM = 0.5
NEG_STAIN_INNER = 0.8
NEG_STAIN_GAIN = 1.5
STAIN_MIX = U64(0x5BD1E995)


def protein_mean_thickness_nm(density, median_d_nm, sigma_log, thickness_nm, fade=1.0):
    if density <= 0 or thickness_nm <= 0:
        return 0.0
    r = 0.5 * median_d_nm / NM_PER_UM
    return density * thickness_nm * (2.0 / 3.0) * math.pi * r * r * math.exp(2 * sigma_log ** 2) * fade


class ProteinFieldStructure(Structure):
    """Hashed-lattice protein blobs (projected spheres) embedded in the support film."""

    def __init__(self, median_diameter_nm, sigma_log, density_per_um2, thickness_nm,
                 material=MaterialId.PROTEIN, negative_stain_fraction=0.0, crystal_patch_fraction=0.0,
                 dose_sensitivity=0.0):
        self.median = float(median_diameter_nm)
        self.sigma = float(sigma_log)
        self.density = float(density_per_um2)
        self.thickness = float(thickness_nm)
        self.material = int(material)
        self.stain = float(negative_stain_fraction)
        self.patch = float(crystal_patch_fraction)
        self.dose = float(dose_sensitivity)
        self.time_dependent = self.dose > 0

    def fill(self, ctx, owner):
        if self.density <= 0 or self.thickness <= 0:
            return
        t = ctx.time_index
        fade = math.exp(-self.dose * t) if (self.dose > 0 and t > 0) else 1.0
        cell = 1.0 / math.sqrt(self.density)
        if not _resolved(cell, ctx):
            mean = protein_mean_thickness_nm(self.density, self.median, self.sigma, self.thickness, fade)
            if mean > 0:
                for B in _owner_pixels(ctx, owner):
                    ctx.add_thickness(B.flat, mean)  # uniform haze, no material claim
            return
        seed = hash_seed(owner.seed, SeedKind.STRUCTURE, 0)
        blobs = JitteredLattice(cell, seed, 0)
        crystal = JitteredLattice(cell, seed, 0, jitter_frac=0.0)
        patches = JitteredLattice(PROTEIN_PATCH_CELL_UM, seed, 1)
        thr = U64(int(min(max(self.patch, 0.0), 1.0) * 256.0))
        stain_nm = NEG_STAIN_GAIN * self.thickness * fade
        med, sig, stain = self.median, self.sigma, self.stain
        for B in _owner_pixels(ctx, owner):
            flat = B.flat
            hit = B.nearest(ctx, blobs)
            d2 = hit.d2
            diam = hit.per_site(lambda h: lognormal_from_hash(h, med, sig, PROTEIN_MIN_D_NM, PROTEIN_MAX_D_NM))
            stained = hit.per_site(lambda h: uniform_from_hash(h ^ STAIN_MIX) < stain) if stain > 0 else None
            grain = None
            if self.patch > 0:
                ph = B.nearest(ctx, patches)
                in_patch = ph.per_site(lambda h: (h % U64(256)) < thr)
                if np.any(in_patch):
                    ch = B.nearest(ctx, crystal)
                    d2 = np.where(in_patch, ch.d2, d2)
                    diam = np.where(in_patch, med, diam)
                    if stained is not None:
                        stained = np.where(in_patch, ch.per_site(lambda h: uniform_from_hash(h ^ STAIN_MIX) < stain),
                                           stained)
                    grain = np.where(in_patch, 0, -1)  # placeholder, material-dependent below
                    patch_h = ph.h
            r = 0.5 * diam / NM_PER_UM
            ok = d2 < np.minimum(r, PROTEIN_MAX_R_UM) ** 2
            if not np.any(ok):
                continue
            rn = np.sqrt(d2[ok]) / r[ok]
            add = self.thickness * np.sqrt(np.maximum(0.0, 1.0 - rn * rn)) * fade
            mat = np.full(add.shape, self.material, np.uint8)
            if stained is not None:
                st = (rn > NEG_STAIN_INNER) & stained[ok]
                add = np.where(st, add + stain_nm, add)
                mat[st] = MaterialId.PLATINUM
            fo = B.sub(ok)
            pos = add > 0
            fo, add, mat = fo[pos], add[pos], mat[pos]
            existing = ctx.thick[fo].astype(np.float64)
            wasvac = ctx.material[fo] == 0
            ctx.add_thickness(fo, add)
            claim = wasvac | (add >= existing)
            g = np.full(fo.shape, -1, np.int32)
            if grain is not None:
                inp = (grain[ok][pos] >= 0)
                if np.any(inp):
                    g[inp] = grain_id_from_hash(mat[inp], patch_h[ok][pos][inp])
            _claim(ctx, fo[claim], g[claim], mat[claim])


# ---------------------------------------------------------------------------------------------
# Deposited thin film (design 5.2)
# ---------------------------------------------------------------------------------------------

COARSE_CELL_MULT = 4.0
PINHOLE_CELL_MULT = 5.0
PINHOLE_RADIUS_FRAC = 0.4
PINHOLE_CELL_COVERAGE = 0.4953
GRAIN_THICKNESS_STEP = 0.12
EDGE_DEWET_GAIN = 4.0
CRACK_WIDTH_UM = 0.02
VN_MEAN_GRAD = 0.4268
VN_PDF_HALF = 1.8565
THIN_FILM_COARSE_FRACTION = 0.25


@functools.lru_cache(maxsize=64)
def mean_pinhole_probability(p: float) -> float:
    if p <= 0:
        return 0.0
    k = (2.0 * (np.arange(64) + 0.5) / 64) - 1.0
    d2 = k[None, :] ** 2 + k[:, None] ** 2
    q = np.minimum(1.0, p * (1.0 + EDGE_DEWET_GAIN * d2))
    return float(np.mean(np.floor(q * 4096.0) / 4096.0))


def crack_half_width(density: float) -> float:
    if density <= 0:
        return 0.0
    return min(0.5, 0.5 * CRACK_WIDTH_UM * VN_MEAN_GRAD * density)


def crack_area_fraction(density: float) -> float:
    w = crack_half_width(density)
    return 0.0 if w <= 0 else min(1.0, 2.0 * w * VN_PDF_HALF)


def thin_film_mean_thickness_nm(thickness, gradient_per_mm, pinhole, crack, center_x_um):
    grad = max(0.0, 1.0 + gradient_per_mm * (center_x_um / NM_PER_UM))
    ph = mean_pinhole_probability(pinhole) * PINHOLE_CELL_COVERAGE
    cr = crack_area_fraction(crack)
    return thickness * grad * (1.0 - min(max(ph, 0), 1)) * (1.0 - min(max(cr, 0), 1))


def _step_of(h):
    return 1.0 + GRAIN_THICKNESS_STEP * (uniform_from_hash(h ^ GRAIN_STEP_MIX) - 0.5)


class ThinFilmStructure(Structure):
    """Bimodal polycrystalline film: fine + 4x coarse jittered lattices, pinholes, cracks."""

    required_layers = frozenset()

    def __init__(self, median_grain_um, thickness_nm, gradient_per_mm, pinhole_fraction,
                 crack_density_per_um, material, coarse_fraction=THIN_FILM_COARSE_FRACTION):
        self.grain_um = float(median_grain_um)
        self.thickness = float(thickness_nm)
        self.gradient = float(gradient_per_mm)
        self.pinhole = float(pinhole_fraction)
        self.crack = float(crack_density_per_um)
        self.material = int(material)
        self.coarse_fraction = float(coarse_fraction)

    def fill(self, ctx, owner):
        if self.grain_um <= 0 or not _resolved(self.grain_um, ctx):
            return
        mean = thin_film_mean_thickness_nm(self.thickness, self.gradient, self.pinhole, self.crack, owner.cx)
        base = hash_seed(owner.seed, SeedKind.STRUCTURE, 0)
        fine = JitteredLattice(self.grain_um, base, 0)
        coarse = JitteredLattice(COARSE_CELL_MULT * self.grain_um, base, 1)
        pins = JitteredLattice(PINHOLE_CELL_MULT * self.grain_um, base, 2)
        pin_r = PINHOLE_RADIUS_FRAC * pins.cell_um
        crack_seed = hash_seed(base, SeedKind.STRUCTURE, 0, 3)
        crack_hw = crack_half_width(self.crack)
        cthr = U64(int(min(max(self.coarse_fraction, 0.0), 1.0) * 256.0))
        mat = self.material
        for B in _owner_pixels(ctx, owner):
            X, lx, ly = B.X, B.lx, B.ly
            removed = np.zeros(X.shape, bool)
            if self.pinhole > 0:
                ph = B.nearest(ctx, pins)
                cand = ph.d2 < pin_r * pin_r
                if np.any(cand):
                    d2 = (lx[cand] / owner.rx) ** 2 + (ly[cand] / owner.ry) ** 2
                    p = np.minimum(1.0, self.pinhole * (1.0 + EDGE_DEWET_GAIN * d2))
                    removed[cand] = (ph.h[cand] % U64(4096)) < (p * 4096.0).astype(np.uint64)
            if self.crack > 0:
                n = B.noise(ctx, crack_seed, self.crack)
                removed |= np.abs(n - 0.5) < crack_hw
            any_removed = bool(removed.any())
            if any_removed:
                fr = B.sub(removed)
                ctx.add_thickness(fr, -mean)
                ctx.grain[fr] = -1
                empty = ctx.thick[fr] <= EMPTY_THICKNESS_NM
                ctx.material[fr[empty]] = 0
            ch = B.nearest(ctx, coarse)
            fh = B.nearest(ctx, fine)
            # one merged site table: coarse sites first, fine sites after
            ncs = ch.grid_h.size
            use_c = ch.per_site(lambda h: (h % U64(256)) < cthr)
            site = np.where(use_c, ch.site, fh.site + ncs)
            H = np.concatenate([ch.grid_h, fh.grid_h])
            grain = grain_id_from_hash(mat, H).take(site)
            step = _step_of(H).take(site)
            local = np.maximum(0.0, self.thickness * (1.0 + self.gradient * (X / NM_PER_UM)) * step)
            if any_removed:
                keep = ~removed
                fk = B.sub(keep)
                ctx.add_thickness(fk, (local - mean)[keep])
                _claim(ctx, fk, grain[keep], mat)
            else:
                ctx.add_thickness(B.flat, local - mean)
                _claim(ctx, B.flat, grain, mat)


class VoronoiGrainsStructure(Structure):
    """Polycrystalline interior: the fine thin-film lattice alone, per-grain thickness jitter."""

    required_layers = frozenset()

    def __init__(self, grain_count, min_spacing_um, material, thickness_jitter_nm):
        self.count = max(1, int(grain_count))
        self.min_spacing = max(0.0, float(min_spacing_um))
        self.material = int(material)
        self.jitter = float(thickness_jitter_nm)

    def fill(self, ctx, owner):
        area = 4.0 * abs(owner.rx) * abs(owner.ry)
        cell = max(self.min_spacing, math.sqrt(area / self.count))
        if not _resolved(cell, ctx):
            return
        lat = JitteredLattice(cell, hash_seed(owner.seed, SeedKind.STRUCTURE, 0), 0)
        mat, jit = self.material, self.jitter
        for B in _owner_pixels(ctx, owner):
            hit = B.nearest(ctx, lat)
            if jit > 0:
                ctx.add_thickness(B.flat, hit.per_site(lambda h: jit * (uniform_from_hash(h ^ GRAIN_STEP_MIX) - 0.5)))
            _claim(ctx, B.flat, hit.per_site(lambda h: grain_id_from_hash(mat, h)), mat)


# ---------------------------------------------------------------------------------------------
# Bulk-sample structures (FIB lamellae)
# ---------------------------------------------------------------------------------------------

PRECIP_SIGMA_LOG = 0.5
PRECIP_R_MIN_MULT, PRECIP_R_MAX_MULT = 0.2, 4.0
PRECIP_RATIO_MIN, PRECIP_RATIO_MAX = 0.6, 1.4


class PrecipitatesStructure(Structure):
    """Second-phase rough ellipses of a different material inside the lamella matrix."""

    required_layers = frozenset()

    def __init__(self, count, median_radius_um, material, extra_thickness_nm):
        self.count = max(0, int(count))
        self.median_r = max(0.0, float(median_radius_um))
        self.material = int(material)
        self.extra = float(extra_thickness_nm)

    def fill(self, ctx, owner):
        if self.count <= 0 or self.median_r <= 0:
            return
        area = 4.0 * owner.rx * owner.ry if owner.shape != Shape.ELLIPSE else math.pi * owner.rx * owner.ry
        if area <= 0:
            return
        cell = math.sqrt(area / self.count)
        if not _resolved(cell, ctx):
            er2 = self.median_r ** 2 * math.exp(2 * PRECIP_SIGMA_LOG ** 2)
            lam = self.count * math.pi * er2 / area
            mean_extra = self.extra * (1.0 - math.exp(-min(max(lam, 0.0), 700.0)))
            for B in _owner_pixels(ctx, owner):
                _matrix_claim(ctx, owner, B.flat)
                if mean_extra > 0:
                    ctx.add_thickness(B.flat, mean_extra)
            return
        lat = JitteredLattice(cell, owner.seed, 0)
        med = self.median_r * NM_PER_UM
        mat = self.material

        def radius(h):
            return lognormal_from_hash(h, med, PRECIP_SIGMA_LOG, med * PRECIP_R_MIN_MULT,
                                       med * PRECIP_R_MAX_MULT) / NM_PER_UM

        def stretch(h):
            return np.sqrt(PRECIP_RATIO_MIN + (PRECIP_RATIO_MAX - PRECIP_RATIO_MIN)
                           * uniform_from_hash(hash_seed(h, SeedKind.STRUCTURE, 1)))

        for B in _owner_pixels(ctx, owner):
            flat = B.flat
            _matrix_claim(ctx, owner, flat)
            hit = B.nearest(ctx, lat)
            r = hit.per_site(radius)
            st = hit.per_site(stretch)
            ins = inside_rough_ellipse(B.X - hit.sx, B.Y - hit.sy, r * st, r / st)
            if not np.any(ins):
                continue
            fi = B.sub(ins)
            _claim(ctx, fi, hit.per_site(lambda h: grain_id_from_hash(mat, h))[ins], mat)
            if self.extra != 0:
                ctx.add_thickness(fi, self.extra)


class StrainFieldStructure(Structure):
    """Smooth strain field around a rough-ellipse inclusion (legacy Strain-Scan)."""

    required_layers = frozenset({LAYER_STRAIN})

    def __init__(self, sxx, sxy, syy, blur_sigma_um):
        self.sxx, self.sxy, self.syy = float(sxx), float(sxy), float(syy)
        self.blur = float(blur_sigma_um)

    def fill(self, ctx, owner):
        write = ctx.strain is not None
        if not write and owner.grain < 0:
            return
        rx = 0.5 * owner.rx
        ry = min(0.25 * owner.rx, 0.9 * owner.ry)
        if rx <= 0 or ry <= 0:
            return
        blur = self.blur if self.blur > 0 else 0.05 * rx
        outer = (1.0 + ROUGH_PEAK) * max(rx, ry) + blur
        for B in _owner_pixels(ctx, owner):
            flat, lx, ly = B.flat, B.lx, B.ly
            _matrix_claim(ctx, owner, flat)
            if not write:
                continue
            s = np.zeros(lx.shape)
            near = (np.abs(lx) <= outer) & (np.abs(ly) <= outer)
            if np.any(near):
                s[near] = smoothstep(rough_ellipse_signed_dist(lx[near], ly[near], rx, ry) / blur + 0.5)
            ctx.strain[0, flat] = 1.0 + (self.sxx - 1.0) * s
            ctx.strain[1, flat] = self.sxy * s
            ctx.strain[2, flat] = 1.0 + (self.syy - 1.0) * s


class JunctionStructure(Structure):
    """Descan step across a boundary (legacy PNJunction-Scan)."""

    required_layers = frozenset({LAYER_DESCAN})

    def __init__(self, position_frac, shift_x_px, shift_y_px, width_frac):
        self.pos = float(position_frac)
        self.sx, self.sy = float(shift_x_px), float(shift_y_px)
        self.width = float(width_frac) if width_frac > 0 else 0.6

    def fill(self, ctx, owner):
        write = ctx.descan is not None
        if not write and owner.grain < 0:
            return
        w = max(self.width, 1e-6)
        for B in _owner_pixels(ctx, owner):
            _matrix_claim(ctx, owner, B.flat)
            if not write:
                continue
            v = B.ly / (2.0 * owner.ry) + 0.5  # 0 at the top edge (+y down), 1 at the bottom
            s = smoothstep((v - self.pos) / w + 0.5)
            ctx.descan[0, B.flat] += self.sx * s
            ctx.descan[1, B.flat] += self.sy * s


class AmorphousStructure(Structure):
    """Explicitly amorphous interior: clears grain ids."""

    def fill(self, ctx, owner):
        for B in _owner_pixels(ctx, owner):
            ctx.grain[B.flat] = -1


def cross_grating_mean_nm(base_nm: float, depth_nm: float, line_fraction: float) -> float:
    """Area-mean thickness of a cross grating: ridges of `line_fraction` of the period in x
    and in y over a `base_nm` film."""
    w = min(max(float(line_fraction), 0.0), 1.0)
    return float(base_nm) + float(depth_nm) * (1.0 - (1.0 - w) ** 2)


class CrossGratingStructure(Structure):
    """A replica cross grating (the pixel-size standard, e.g. 2160 lines/mm): ridges along x
    and along y, `line_fraction` of the period wide, `depth_nm` over a `base_nm` film, at an
    exact `period_um` in the owner's own (unrotated) frame."""

    required_layers = frozenset()

    def __init__(self, period_um: float, base_nm: float, depth_nm: float, line_fraction: float = 0.5):
        self.period = float(period_um)
        self.base = float(base_nm)
        self.depth = float(depth_nm)
        self.w = min(max(float(line_fraction), 0.0), 1.0)

    def fill(self, ctx, owner):
        if self.period <= 0 or self.depth == 0 or not _resolved(self.period, ctx):
            return  # unresolved: the owner already carries the mean
        mean = cross_grating_mean_nm(self.base, self.depth, self.w)
        for B in _owner_pixels(ctx, owner):
            fx = np.mod(B.lx / self.period, 1.0)
            fy = np.mod(B.ly / self.period, 1.0)
            ridge = (fx < self.w) | (fy < self.w)
            ctx.add_thickness(B.flat, self.base + self.depth * ridge - mean)


class MultilayerStructure(Structure):
    """Stacked bands of different materials perpendicular to the owner's local y."""

    required_layers = frozenset()

    def __init__(self, band_um: float, materials: Sequence[int], deltas_nm: Sequence[float]):
        self.band = float(band_um) if band_um > 0 else 0.4
        self.materials = np.asarray(materials, np.uint8)
        self.deltas = np.asarray(list(deltas_nm) + [0.0] * (len(materials) - len(deltas_nm)), np.float64)

    def fill(self, ctx, owner):
        nb = self.materials.size
        if nb == 0:
            return
        if not _resolved(self.band, ctx):
            mean = float(self.deltas[:nb].mean())
            for B in _owner_pixels(ctx, owner):
                _matrix_claim(ctx, owner, B.flat)
                if mean != 0:
                    ctx.add_thickness(B.flat, mean)
            return
        for B in _owner_pixels(ctx, owner):
            band = np.floor(B.ly / self.band).astype(np.int64)
            col = np.floor(B.lx / self.band).astype(np.int64)
            b0, c0 = int(band.min()), int(col.min())
            gb = np.arange(b0, int(band.max()) + 1, dtype=np.int64)
            gc = np.arange(c0, int(col.max()) + 1, dtype=np.int64)
            gbi = np.mod(gb, nb)
            # grain per (absolute band, column) cell, evaluated once per cell
            G = grain_id_from_hash(self.materials[gbi][:, None],
                                   hash_seed(np.uint64(owner.seed), SeedKind.GRAIN, mix_cell(gb[:, None], gc[None, :])))
            bi = np.mod(band, nb)
            mats = self.materials[bi]
            _claim(ctx, B.flat, G[band - b0, col - c0], mats)
            d = self.deltas[bi]
            nz = d != 0
            if np.any(nz):
                ctx.add_thickness(B.sub(nz), d[nz])


# ---------------------------------------------------------------------------------------------
# Time evolution (nucleation + diffusion-limited growth + drift)
# ---------------------------------------------------------------------------------------------

GROWTH_WINDOW_STEPS = 8
TIME_EVOLVE_TMIN = 1
TIME_EVOLVE_WINDOW = GROWTH_WINDOW_STEPS  # tk in [1, 8]


class TimeEvolvingStructure(Structure):
    """Marks an in-situ scene as time dependent.

    Each deposited droplet keeps its own grain (a real orientation), and whether it is glassy or
    crystalline is decided per grain by the thermal-budget model (``Specimen.crystallinity``),
    so heating, not wall time, drives crystallisation and every droplet nucleates on its own.
    Specimen drift is applied to the whole field by ``Specimen.rasterize``.
    (The C++ model grew crystalline discs from sparse sites on wall time, which left most
    droplets glassy forever and ignored the temperature.)
    """

    required_layers = frozenset()
    time_dependent = True

    def __init__(self, cell_um: float, growth_um_per_step: float, drift_um_per_step: tuple[float, float]):
        self.cell_um = float(cell_um)
        self.growth = max(0.0, float(growth_um_per_step))
        self.drift = (float(drift_um_per_step[0]), float(drift_um_per_step[1]))

    def fill(self, ctx, owner):
        return
