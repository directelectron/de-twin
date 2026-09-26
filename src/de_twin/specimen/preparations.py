"""Preparations: what was put on the holder (port of ``Preparation*.cpp``).

A preparation populates one placement area at a time from that area's sub-seed
(``hash_seed(seed, HOLE_POPULATION, area.index)``), so areas can be populated in any order,
evicted and repopulated identically. ``aggregate_for`` returns the single area-averaged primitive
the scene uses at low magnification; it is built to carry the same mean projected thickness the
populated area would.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..hashing import SeedKind, hash_seed, rng_for
from .fieldmap import LAYER_DESCAN, LAYER_STRAIN, grain_id_from_hash
from .geometry import normal_cdf
from .holders import Holder, PlacementArea
from .materials import MaterialId, material
from .options import SpecimenOptions
from .raster import MAX_FACETS, Layer, PrimitiveSet, Profile, Shape
from .structures import (JunctionStructure, MultilayerStructure, PrecipitatesStructure,
                         ProteinFieldStructure, StrainFieldStructure, ThinFilmStructure,
                         TimeEvolvingStructure, VoronoiGrainsStructure, protein_mean_thickness_nm,
                         thin_film_mean_thickness_nm)

BOUNDING_SLACK = 1.15
FACET_JITTER = 0.12
CLUSTER_ATTEMPTS = 16
MAX_CLUSTERS = 65536
CLUSTER_SALT = 0xC1057E45
NM3_PER_UM2_TO_NM = 1.0e-6


@dataclass
class ClusterSeed:
    center: tuple[float, float]
    sigma_um: float
    count: int


@dataclass
class Batch:
    """Primitives generated for one placement area (PrimitiveBatch)."""

    area_index: int
    prims: PrimitiveSet
    clusters: list[ClusterSeed] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return self.prims.nbytes + 64 * len(self.clusters) + 256


def _rect_prim(area: PlacementArea, thickness, material_id, seed, *, rotation=0.0, grain=-1,
               structure: Optional[object] = None, aggregate=False, layer=Layer.PREPARATION) -> PrimitiveSet:
    hx, hy = abs(area.half[0]), abs(area.half[1])
    return PrimitiveSet.make(area.center[0], area.center[1], hx, hy, rot=rotation, shape=Shape.RECT,
                             thickness=thickness, profile=Profile.FLAT, layer=layer, material=material_id,
                             grain=grain, seed=np.uint64(int(seed) & ((1 << 64) - 1)), aggregate=aggregate,
                             structure=0 if structure is not None else -1,
                             structures=[structure] if structure is not None else None)


class Preparation:
    kind = "base"
    required_layers: frozenset = frozenset()
    time_dependent = False

    def populate(self, holder: Holder, area: PlacementArea, seed: int) -> Batch:
        raise NotImplementedError

    def aggregate_params(self, holder: Holder, area: PlacementArea, seed: int) -> tuple:
        """(thickness_nm, material, grain, rotation) of the area-averaged aggregate rectangle."""
        raise NotImplementedError

    def aggregate_for(self, holder: Holder, area: PlacementArea, seed: int) -> PrimitiveSet:
        t, m, g, rot = self.aggregate_params(holder, area, seed)
        return _rect_prim(area, t, m, seed, rotation=rot, grain=g, aggregate=True)

    def cluster_centers(self, holder: Holder, area: PlacementArea, seed: int) -> list[ClusterSeed]:
        return []

    def typical_feature_radius_um(self) -> float:
        return 0.0  # "never cull me"


# ---------------------------------------------------------------------------------------------
# Nanoparticles
# ---------------------------------------------------------------------------------------------

@dataclass
class NanoparticleParams:
    median_diameter_nm: float = 24.0
    sigma_log: float = 0.8
    min_diameter_nm: float = 2.0
    max_diameter_nm: float = 300.0
    density_per_um2: float = 24.0
    clustered_fraction: float = 0.35
    clusters_per_um2: float = 0.06
    cluster_count: int = 12
    cluster_sigma_um: float = 0.45
    max_particles_per_area: int = 400000
    faceted_fraction: float = 0.3
    facets_min: int = 5
    facets_max: int = 8
    aspect_jitter: float = 0.25
    material: int = MaterialId.GOLD
    open_hole_suppression: float = 0.85


def expected_diameter_cubed(median, sigma, lo, hi) -> float:
    """E[d^3] of the clamped log-normal SampleLogNormalNm draws."""
    if median <= 0:
        return 0.0
    lo = max(lo, 0.0)
    hi = max(hi, lo)
    if sigma <= 0:
        d = min(max(median, lo), hi)
        return d ** 3
    zmin = math.log(lo / median) / sigma if lo > 0 else -40.0
    zmax = math.log(hi / median) / sigma if hi > 0 else -40.0
    below = normal_cdf(zmin)
    above = 1.0 - normal_cdf(zmax)
    body = median ** 3 * math.exp(4.5 * sigma * sigma) * (normal_cdf(zmax - 3 * sigma) - normal_cdf(zmin - 3 * sigma))
    return lo ** 3 * below + hi ** 3 * above + body


class NanoparticlesPreparation(Preparation):
    """Log-normal crystalline particles, partly clustered, faceted or rough-ellipse outlines."""

    kind = "nanoparticles"

    def __init__(self, params: NanoparticleParams):
        self.p = params

    def _open_factor(self, area: PlacementArea) -> float:
        if area.film_covered and not area.film_broken:
            return 1.0
        return min(max(1.0 - self.p.open_hole_suppression, 0.0), 1.0)

    def expected_count(self, area: PlacementArea) -> float:
        fp = 4.0 * abs(area.half[0]) * abs(area.half[1])
        if fp <= 0:
            return 0.0
        mean = fp * self.p.density_per_um2 * self._open_factor(area)
        if self.p.max_particles_per_area > 0:
            mean = min(mean, float(self.p.max_particles_per_area))
        return max(mean, 0.0)

    def cluster_centers(self, holder, area, seed) -> list[ClusterSeed]:
        hx, hy = abs(area.half[0]), abs(area.half[1])
        if hx <= 0 or hy <= 0:
            return []
        fp = 4.0 * hx * hy
        if self.p.clusters_per_um2 > 0:
            n = int(min(max(math.floor(self.p.clusters_per_um2 * fp + 0.5), 1), MAX_CLUSTERS))
        else:
            n = max(1, min(self.p.cluster_count, MAX_CLUSTERS))
        per = self.expected_count(area) * min(max(self.p.clustered_fraction, 0.0), 1.0) / n
        rng = rng_for(seed, SeedKind.HOLE_POPULATION, 0, CLUSTER_SALT)
        cx = rng.uniform(area.center[0] - hx, area.center[0] + hx, n)
        cy = rng.uniform(area.center[1] - hy, area.center[1] + hy, n)
        counts = rng.poisson(per, n) if per > 0 else np.zeros(n, np.int64)
        return [ClusterSeed((float(a), float(b)), self.p.cluster_sigma_um, int(c)) for a, b, c in zip(cx, cy, counts)]

    def populate(self, holder, area, seed) -> Batch:
        p = self.p
        hx, hy = abs(area.half[0]), abs(area.half[1])
        empty = Batch(area.index, PrimitiveSet(0))
        if hx <= 0 or hy <= 0:
            return empty
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        mean = self.expected_count(area)
        count = int(rng.poisson(mean)) if mean > 0 else 0
        if count <= 0:
            return empty
        clusters = self.cluster_centers(holder, area, seed)
        ncl = len(clusters)
        if ncl <= 0:
            return empty
        ccx = np.array([c.center[0] for c in clusters])
        ccy = np.array([c.center[1] for c in clusters])
        acx, acy = area.center

        # -- positions: clustered (Gaussian around a centre, up to 16 attempts) or uniform --
        clustered = rng.random(count) < p.clustered_fraction
        ci = rng.integers(0, ncl, count)
        x = np.empty(count)
        y = np.empty(count)
        placed = np.zeros(count, bool)
        cidx = np.flatnonzero(clustered)
        pending = cidx
        for _attempt in range(CLUSTER_ATTEMPTS):  # retry only the draws that fell outside
            if pending.size == 0:
                break
            off = rng.normal(0.0, p.cluster_sigma_um, (pending.size, 2))
            px = ccx[ci[pending]] + off[:, 0]
            py = ccy[ci[pending]] + off[:, 1]
            ins = (np.abs(px - acx) <= hx) & (np.abs(py - acy) <= hy)
            got = pending[ins]
            x[got] = px[ins]
            y[got] = py[ins]
            placed[got] = True
            pending = pending[~ins]
        un = ~placed
        nu = int(un.sum())
        x[un] = rng.uniform(acx - hx, acx + hx, nu)
        y[un] = rng.uniform(acy - hy, acy + hy, nu)

        # -- sizes and shapes --
        aj = min(max(p.aspect_jitter, 0.0), 0.5)
        d = np.clip(p.median_diameter_nm * np.exp(p.sigma_log * rng.standard_normal(count)),
                    p.min_diameter_nm, p.max_diameter_nm)
        r = d / 2000.0
        rx = r * (1.0 + rng.uniform(-aj, aj, count))
        ry = r * (1.0 + rng.uniform(-aj, aj, count))
        rot = rng.uniform(0.0, 2.0 * math.pi, count)
        faceted = rng.random(count) < p.faceted_fraction
        lo, hi = min(p.facets_min, p.facets_max), max(p.facets_min, p.facets_max)
        hi = min(hi, MAX_FACETS)
        nf = rng.integers(lo, hi + 1, count).astype(np.int8)
        fj = np.ones((count, MAX_FACETS), np.float32)
        fidx = np.flatnonzero(faceted)
        fj[fidx] = 1.0 + rng.uniform(-FACET_JITTER, FACET_JITTER, (fidx.size, MAX_FACETS))
        pseed = hash_seed(np.uint64(int(seed)), SeedKind.PARTICLE, np.arange(count, dtype=np.int64))
        grain = grain_id_from_hash(p.material, pseed)
        bound = BOUNDING_SLACK * np.maximum(rx, ry)
        prims = PrimitiveSet.make(
            x, y, rx, ry, rot=rot, shape=np.where(faceted, Shape.FACETED, Shape.ROUGH_ELLIPSE).astype(np.int8),
            bound_r=bound, bounds=(x - bound, y - bound, x + bound, y + bound), thickness=d.astype(np.float32),
            profile=Profile.SPHERICAL, layer=Layer.PREPARATION, material=p.material, grain=grain, seed=pseed,
            facet_n=np.where(faceted, nf, 0).astype(np.int8), facet_j=fj)
        ok = placed & clustered
        counts = np.bincount(ci[ok], minlength=ncl)
        for c, k in zip(clusters, counts):
            c.count = int(k)
        return Batch(area.index, prims, clusters)

    def aggregate_params(self, holder, area, seed):
        p = self.p
        hx, hy = abs(area.half[0]), abs(area.half[1])
        d3 = expected_diameter_cubed(p.median_diameter_nm, p.sigma_log, p.min_diameter_nm, p.max_diameter_nm)
        fp = 4.0 * hx * hy
        density = self.expected_count(area) / fp if fp > 0 else 0.0
        mean = density * (math.pi / 6.0) * d3 * NM3_PER_UM2_TO_NM
        # A Rect, not the C++ inscribed Ellipse: the mean is per unit area of the whole rectangle,
        # so painting it over pi/4 of it lost 21 % of the mass at the aggregate/populate seam.
        return mean, p.material, -1, 0.0

    def typical_feature_radius_um(self) -> float:
        p = self.p
        d = min(max(p.median_diameter_nm * math.exp(2.0 * p.sigma_log), p.min_diameter_nm), p.max_diameter_nm)
        return BOUNDING_SLACK * (1.0 + min(max(p.aspect_jitter, 0.0), 0.5)) * d / 2000.0


# ---------------------------------------------------------------------------------------------
# Proteins / thin film / bulk / time-evolving
# ---------------------------------------------------------------------------------------------

class ProteinsPreparation(Preparation):
    kind = "proteins"

    def __init__(self, median_nm, sigma_log, density, thickness_nm, stain, patch, dose):
        self.median, self.sigma, self.density = median_nm, sigma_log, density
        self.thickness, self.stain, self.patch, self.dose = thickness_nm, stain, patch, dose
        self.time_dependent = dose > 0

    def populate(self, holder, area, seed) -> Batch:
        if abs(area.half[0]) <= 0 or abs(area.half[1]) <= 0:
            return Batch(area.index, PrimitiveSet(0))
        s = ProteinFieldStructure(self.median, self.sigma, self.density, self.thickness, MaterialId.PROTEIN,
                                  self.stain, self.patch, self.dose)
        # 0 nm owner: the blobs are discrete and add on top of the embedding film.
        return Batch(area.index, _rect_prim(area, 0.0, MaterialId.PROTEIN, seed, structure=s))

    def aggregate_params(self, holder, area, seed):
        mean = protein_mean_thickness_nm(self.density, self.median, self.sigma, self.thickness, 1.0)
        return mean, MaterialId.PROTEIN, -1, 0.0


class CrossGratingPreparation(Preparation):
    """A carbon replica cross grating: the calibration specimen for pixel size and image-shift
    calibrations (SerialEM's Find Pixel Size)."""

    kind = "cross_grating"

    def __init__(self, lines_per_mm: float, base_nm: float, depth_nm: float, line_fraction: float = 0.5):
        self.period_um = 1000.0 / max(float(lines_per_mm), 1e-6)
        self.base, self.depth, self.w = float(base_nm), float(depth_nm), float(line_fraction)

    def populate(self, holder, area, seed) -> Batch:
        from .structures import CrossGratingStructure, cross_grating_mean_nm

        if abs(area.half[0]) <= 0 or abs(area.half[1]) <= 0:
            return Batch(area.index, PrimitiveSet(0))
        s = CrossGratingStructure(self.period_um, self.base, self.depth, self.w)
        mean = cross_grating_mean_nm(self.base, self.depth, self.w)
        return Batch(area.index, _rect_prim(area, mean, MaterialId.AMORPHOUS_CARBON, seed, structure=s))

    def aggregate_params(self, holder, area, seed):
        from .structures import cross_grating_mean_nm

        return cross_grating_mean_nm(self.base, self.depth, self.w), MaterialId.AMORPHOUS_CARBON, -1, 0.0


class ThinFilmPreparation(Preparation):
    kind = "thin_film"

    def __init__(self, grain_um, thickness_nm, gradient_per_mm, pinhole, crack, material_id):
        self.grain_um, self.thickness, self.gradient = grain_um, thickness_nm, gradient_per_mm
        self.pinhole, self.crack, self.material = pinhole, crack, int(material_id)

    def _mean(self, area):
        return thin_film_mean_thickness_nm(self.thickness, self.gradient, self.pinhole, self.crack, area.center[0])

    def populate(self, holder, area, seed) -> Batch:
        if abs(area.half[0]) <= 0 or abs(area.half[1]) <= 0:
            return Batch(area.index, PrimitiveSet(0))
        s = ThinFilmStructure(self.grain_um, self.thickness, self.gradient, self.pinhole, self.crack, self.material)
        # Same builder as the aggregate: the LOD transition is seamless by construction.
        return Batch(area.index, _rect_prim(area, self._mean(area), self.material, seed, structure=s))

    def aggregate_params(self, holder, area, seed):
        return self._mean(area), self.material, -1, 0.0


PRECIP_DENSITY_PER_UM2 = 60.0
PRECIP_MEDIAN_R_UM = 0.06
PRECIP_EXTRA_NM = 20.0
PRECIP_COUNT_MAX = 200000
JUNCTION_POSITION = 0.5
STRAIN_BLUR_UM = 0.15
POLY_GRAIN_COUNT = 400
POLY_MIN_SPACING_UM = 0.02
POLY_JITTER_NM = 8.0
MULTILAYER_BAND_UM = 0.4
MULTILAYER_DELTAS = (0.0, 12.0, -4.0, 6.0)
BULK_KIND_MATERIAL = {0: MaterialId.IRON, 1: MaterialId.SILICON, 2: MaterialId.SILICON,
                      3: MaterialId.ALUMINUM, 4: MaterialId.SILICON}


class BulkSamplePreparation(Preparation):
    """The interior of one FIB lamella; the kind picks the structure."""

    kind = "bulk"

    def __init__(self, bulk_kind: int, material_id: int, opts: SpecimenOptions):
        self.bulk_kind = int(bulk_kind)
        self.material = int(material_id)
        self.opts = opts
        self.required_layers = {1: frozenset({LAYER_DESCAN}),
                                2: frozenset({LAYER_STRAIN})}.get(self.bulk_kind, frozenset())

    def _structure(self, area: PlacementArea):
        o = self.opts
        a = max(1e-9, 4.0 * abs(area.half[0]) * abs(area.half[1]))
        k = self.bulk_kind
        if k == 0:
            n = int(min(PRECIP_COUNT_MAX, max(1.0, PRECIP_DENSITY_PER_UM2 * a)))
            return PrecipitatesStructure(n, PRECIP_MEDIAN_R_UM, MaterialId.PLATINUM, PRECIP_EXTRA_NM)
        if k == 1:
            return JunctionStructure(JUNCTION_POSITION, o.junction_descan_x_px, o.junction_descan_y_px,
                                     max(0.01, o.junction_width_frac))
        if k == 2:
            m = o.strain_magnitude
            return StrainFieldStructure(1.0 + m, m, 1.0 - 0.5 * m, STRAIN_BLUR_UM)
        if k == 3:
            # The C++ hard-codes Aluminum here even when the post's matrix is gold; we use the
            # matrix material so the grains belong to the lamella that carries them.
            return VoronoiGrainsStructure(POLY_GRAIN_COUNT, POLY_MIN_SPACING_UM, self.material, POLY_JITTER_NM)
        return MultilayerStructure(MULTILAYER_BAND_UM, [MaterialId.SILICON, MaterialId.PLATINUM,
                                                        MaterialId.ALUMINUM, MaterialId.AMORPHOUS_CARBON],
                                   MULTILAYER_DELTAS)

    def _matrix_grain(self, area, seed) -> int:
        if self.material == MaterialId.VACUUM or not material(self.material).crystalline:
            return -1
        idx = max(0, area.post_index) * 16 + max(0, area.index)
        return int(grain_id_from_hash(self.material, hash_seed(seed, SeedKind.GRAIN, idx)))

    def populate(self, holder, area, seed) -> Batch:
        if abs(area.half[0]) <= 0 or abs(area.half[1]) <= 0:
            return Batch(area.index, PrimitiveSet(0))
        # 0 nm owner: the matrix is the holder's Curtain-profile lamella body.
        return Batch(area.index, _rect_prim(area, 0.0, self.material, seed, rotation=area.rotation,
                                            grain=self._matrix_grain(area, seed), structure=self._structure(area)))

    def aggregate_params(self, holder, area, seed):
        return 0.0, self.material, self._matrix_grain(area, seed), area.rotation


class TimeEvolvingPreparation(Preparation):
    """Wraps another preparation and adds nucleation + growth + drift driven by the time index."""

    kind = "time_evolving"
    time_dependent = True

    def __init__(self, inner: Preparation, nucleation_per_area: int, growth_um_per_step: float,
                 drift_um_per_step: tuple[float, float]):
        self.inner = inner
        self.nucleation = max(1, int(nucleation_per_area))
        self.growth = max(0.0, float(growth_um_per_step))
        self.drift = drift_um_per_step
        self.required_layers = inner.required_layers

    def populate(self, holder, area, seed) -> Batch:
        b = self.inner.populate(holder, area, seed)
        inner_mat = int(b.prims.material[0]) if b.prims.n else getattr(getattr(self.inner, "p", None), "material", 0)
        a = area.bounds.width * area.bounds.height
        cell = math.sqrt(a / self.nucleation) if a > 0 else 1.0
        s = TimeEvolvingStructure(cell, self.growth, self.drift)
        carrier = PrimitiveSet.make(area.center[0], area.center[1], area.half[0], area.half[1],
                                    shape=Shape.RECT, bound_r=math.hypot(*area.half),
                                    bounds=tuple(np.array([v]) for v in area.bounds.as_tuple()),
                                    thickness=0.0, profile=Profile.FLAT, layer=Layer.STRUCTURE,
                                    material=inner_mat, structure=0, structures=[s],
                                    seed=np.uint64(hash_seed(seed, SeedKind.TIME_EVOLUTION, area.index)))
        return Batch(area.index, PrimitiveSet.concat([b.prims, carrier]), b.clusters)

    def aggregate_params(self, holder, area, seed):
        return self.inner.aggregate_params(holder, area, seed)

    def cluster_centers(self, holder, area, seed):
        return self.inner.cluster_centers(holder, area, seed)


# ---------------------------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------------------------

def nanoparticle_params(o: SpecimenOptions) -> NanoparticleParams:
    p = NanoparticleParams()
    p.density_per_um2 = o.particle_density_per_um2
    p.median_diameter_nm = o.particle_median_diameter_nm
    p.sigma_log = o.particle_size_sigma_log
    p.clustered_fraction = o.clustered_fraction
    p.clusters_per_um2 = o.cluster_density_per_um2
    p.cluster_sigma_um = o.cluster_radius_um
    p.min_diameter_nm = min(2.0, 0.25 * p.median_diameter_nm)
    p.max_diameter_nm = max(p.median_diameter_nm * 12.5, p.min_diameter_nm + 1.0)
    return p


_MATERIAL_BY_NAME = {"gold": MaterialId.GOLD, "aluminum": MaterialId.ALUMINUM, "copper": MaterialId.COPPER,
                     "iron": MaterialId.IRON, "silicon": MaterialId.SILICON, "platinum": MaterialId.PLATINUM}


def make_preparation(kind: str, holder_kind: str, film: str, o: SpecimenOptions) -> Preparation:
    shipped = SpecimenOptions()
    if kind in ("nanoparticles", "time_evolving"):
        p = nanoparticle_params(o)
        if holder_kind == "insitu_heating_chip":
            # The chip specimen: denser, narrower 20-60 nm particles; the openings ARE the specimen.
            if o.particle_density_per_um2 == shipped.particle_density_per_um2:
                p.density_per_um2 = 40.0
            if o.particle_median_diameter_nm == shipped.particle_median_diameter_nm:
                p.median_diameter_nm = 22.0
            if o.particle_size_sigma_log == shipped.particle_size_sigma_log:
                p.sigma_log = 0.32
            p.min_diameter_nm = min(15.0, 0.6 * p.median_diameter_nm)
            p.max_diameter_nm = max(60.0, p.min_diameter_nm + 1.0)
            p.open_hole_suppression = 0.0
        inner = NanoparticlesPreparation(p)
        if kind == "time_evolving" or holder_kind == "insitu_heating_chip":
            return TimeEvolvingPreparation(inner, o.nucleation_sites_per_window, o.growth_rate_nm_per_step / 1000.0,
                                           (o.drift_per_step_x_nm / 1000.0, o.drift_per_step_y_nm / 1000.0))
        return inner
    if kind == "proteins":
        median = min(max(o.protein_diameter_nm, 2.0), 100.0)
        return ProteinsPreparation(median, 0.25, o.protein_density_per_um2, 0.7 * median,
                                   o.negative_stain_fraction, o.protein_crystal_patch_fraction,
                                   o.dose_fading_per_step)
    if kind == "thin_film":
        grain_um = min(max(o.grain_size_nm / 1000.0, 0.002), 5.0)
        thick = o.deposited_film_thickness_nm
        if o.thin_film_material != "auto":
            mat = _MATERIAL_BY_NAME[o.thin_film_material]
        else:
            mat = MaterialId.GOLD if holder_kind == "waffle_grid" else MaterialId.ALUMINUM
        if holder_kind == "waffle_grid":
            # A 40 nm Au sheet hides the waffle at navigation mags; the untouched generic defaults
            # are swapped for the calibration-film look (C++ ThinFilmParamsFromOptions).
            if abs(o.deposited_film_thickness_nm - 40.0) < 1e-6:
                thick = 10.0
            if abs(o.grain_size_nm - 150.0) < 1e-6:
                grain_um = 0.03
        return ThinFilmPreparation(grain_um, thick, o.thin_film_gradient_per_mm, o.pinhole_fraction,
                                   o.crack_density_per_um, mat)
    if kind == "cross_grating":
        return CrossGratingPreparation(o.grating_lines_per_mm, o.grating_base_nm, o.grating_depth_nm)
    if kind == "bulk":
        from .holders import _POST_SAMPLE_TO_KIND
        k = _POST_SAMPLE_TO_KIND.get(o.fib_post_sample, 0)
        return BulkSamplePreparation(k, BULK_KIND_MATERIAL[k], o)
    raise ValueError(f"unknown preparation {kind!r}")
