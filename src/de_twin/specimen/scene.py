"""The generated world: one holder, its preparation(s), and the lazily populated areas
(port of ``Scene.cpp``).

Level of detail at whole-area granularity (``Scene::Query``): a placement area whose diameter in
raster pixels is below ``max(96, 0.75 * max(ny, nx))``, or whose preparation's largest feature
would be culled anyway, is represented by one aggregate primitive carrying the area-averaged
projected thickness; otherwise it is populated (and kept in an LRU of at most 64 areas / 384 MB,
at most 4 new populations / 192 MB per query).
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from ..hashing import SeedKind, hash_seed
from .holders import BULK_SAMPLE_KINDS, Holder, PlacementArea
from .options import SpecimenConfig, SpecimenOptions
from .preparations import Batch, BulkSamplePreparation, Preparation, make_preparation
from .raster import Layer, PrimitiveSet, Profile, RasterContext, Shape

AGGREGATE_HOLE_DIAMETER_PX = 96.0
POPULATE_VIEW_FRACTION = 0.75
MAX_RESIDENT_AREAS = 64
MAX_RESIDENT_BYTES = 384 * 1024 * 1024
MAX_POPULATE_PER_QUERY = 4
MAX_POPULATE_BYTES_PER_QUERY = 192 * 1024 * 1024
MAX_FEATURE_TABLE = 64


@dataclass
class Feature:
    """A navigable feature (SceneFeature). ``center_um`` is in world coordinates."""

    center_um: tuple[float, float]
    radius_um: float = 0.0
    area_index: int = -1
    post_index: int = -1
    kind: str | None = None  # BulkSampleKind name for a lamella, else None
    weight: int = 1  # particle count of a cluster
    label: str = ""


class Scene:
    def __init__(self, config: SpecimenConfig, opts: SpecimenOptions):
        self.config = config
        self.opts = opts
        self.seed = int(config.seed) & 0xFFFFFFFF  # the C++ property is a uint32 in practice
        self.holder_kind = config.holder
        self.film = config.resolved_film()
        self.holder = Holder(config.holder, self.seed, self.film, opts)
        self.preparation: Preparation = make_preparation(config.preparation, config.holder, self.film, opts)
        self.per_post: list[Preparation] = []
        if config.holder == "fib_liftout" and config.preparation == "bulk":
            self.per_post = [BulkSamplePreparation(p.bulk_kind, p.matrix_material, opts) for p in self.holder.posts]
        self._aggregates: PrimitiveSet | None = None  # one row per placement area, built lazily
        self._resident: OrderedDict[int, Batch] = OrderedDict()
        self._resident_bytes = 0
        self._query_lock = threading.RLock()
        self.last_stats: dict = {}

    # ------------------------------------------------------------------------------------------
    def preparation_for(self, area: PlacementArea) -> Preparation:
        if 0 <= area.post_index < len(self.per_post):
            return self.per_post[area.post_index]
        return self.preparation

    def area_seed(self, index: int) -> int:
        return hash_seed(self.seed, SeedKind.HOLE_POPULATION, index)

    @property
    def required_layers(self) -> frozenset:
        out = set(self.preparation.required_layers)
        for p in self.per_post:
            out |= p.required_layers
        return frozenset(out)

    @property
    def time_dependent(self) -> bool:
        return self.preparation.time_dependent or any(p.time_dependent for p in self.per_post)

    def aggregate_table(self) -> PrimitiveSet:
        """Aggregate primitive of every placement area (row i = area i), built once."""
        if self._aggregates is None:
            h = self.holder
            rows = [self.preparation_for(a).aggregate_params(h, a, self.area_seed(a.index)) for a in h.areas]
            t, m, g, rot = (np.array(v) for v in zip(*rows))
            seeds = np.array([self.area_seed(a.index) for a in h.areas], np.uint64)
            self._aggregates = PrimitiveSet.make(
                h.area_cx, h.area_cy, np.abs(h.area_hx), np.abs(h.area_hy), rot=rot.astype(np.float64),
                shape=Shape.RECT, thickness=t.astype(np.float32), profile=Profile.FLAT, layer=Layer.PREPARATION,
                material=m.astype(np.uint8), grain=g.astype(np.int32), seed=seeds, aggregate=True)
        return self._aggregates

    def clear_population(self):
        self._resident.clear()
        self._resident_bytes = 0

    def _ensure(self, area: PlacementArea, protected: set) -> tuple[Batch, bool]:
        protected.add(area.index)
        if area.index in self._resident:
            self._resident.move_to_end(area.index)
            return self._resident[area.index], False
        batch = self.preparation_for(area).populate(self.holder, area, self.area_seed(area.index))
        self._resident[area.index] = batch
        self._resident_bytes += batch.nbytes
        while len(self._resident) > MAX_RESIDENT_AREAS or self._resident_bytes > MAX_RESIDENT_BYTES:
            victim = next((k for k in self._resident if k not in protected), None)
            if victim is None:
                break
            self._resident_bytes -= self._resident.pop(victim).nbytes
        return batch, True

    def query(self, ctx: RasterContext) -> PrimitiveSet:
        """Primitives visible in the view, in painter's order (Scene::Query). Serialised:
        it populates areas on demand, and strips of one raster are rasterised in threads."""
        with self._query_lock:
            return self._query(ctx)

    def _query(self, ctx: RasterContext) -> PrimitiveSet:
        t0 = time.perf_counter()
        stats = {"areas_visible": 0, "areas_populated": 0, "areas_aggregated": 0}
        box = ctx.aabb
        idx = self.holder.query_areas(box)
        stats["areas_visible"] = int(idx.size)
        span = getattr(ctx, "span_px", None) or max(ctx.ny, ctx.nx)
        threshold = max(AGGREGATE_HOLE_DIAMETER_PX, POPULATE_VIEW_FRACTION * span)
        parts: list[PrimitiveSet] = []
        protected: set = set()
        populated_bytes = 0
        aggregates: list[int] = []
        for i in idx:
            area = self.holder.areas[int(i)]
            prep = self.preparation_for(area)
            fr = prep.typical_feature_radius_um()
            all_culled = fr > 0 and float(ctx.diameter_px(fr)) < ctx.lod_threshold_px
            dpx = float(ctx.diameter_px(area.half_span))
            resident = area.index in self._resident
            budget = (not resident) and (stats["areas_populated"] >= MAX_POPULATE_PER_QUERY
                                         or populated_bytes >= MAX_POPULATE_BYTES_PER_QUERY)
            if dpx < threshold or all_culled or budget:
                aggregates.append(area.index)
                stats["areas_aggregated"] += 1
                continue
            batch, new = self._ensure(area, protected)
            if new:
                populated_bytes += batch.nbytes
                stats["areas_populated"] += 1
            P = batch.prims
            if P.n:
                vis = np.flatnonzero(P.intersects(box))
                if vis.size:
                    parts.append(P.take(vis) if vis.size < P.n else P)
        if aggregates:
            parts.append(self.aggregate_table().take(np.asarray(aggregates)))
        parts.append(self.holder.query_primitives(box))
        out = PrimitiveSet.concat(parts)
        if out.n > 1 and np.any(np.diff(out.layer.astype(np.int16)) < 0):
            out = out.take(np.argsort(out.layer, kind="stable"))  # painter's order, stable
        stats["primitives"] = int(out.n)
        stats["resident_bytes"] = int(self._resident_bytes)
        stats["query_ms"] = 1e3 * (time.perf_counter() - t0)
        self.last_stats = stats
        return out

    # -- features -------------------------------------------------------------------------------
    def _choose_area(self, x: float, y: float) -> PlacementArea | None:
        areas = self.holder.areas
        if not areas:
            return None
        b = self.holder.area_bounds
        inside = np.flatnonzero((b[:, 0] <= x) & (b[:, 2] >= x) & (b[:, 1] <= y) & (b[:, 3] >= y))
        if inside.size:
            return areas[int(inside[0])]
        d2 = (self.holder.area_cx - x) ** 2 + (self.holder.area_cy - y) ** 2
        return areas[int(np.argmin(d2))]

    def _clusters(self, area: PlacementArea, resident_ok: bool = True):
        if resident_ok and area.index in self._resident and self._resident[area.index].clusters:
            return self._resident[area.index].clusters
        return self.preparation_for(area).cluster_centers(self.holder, area, self.area_seed(area.index))

    def nearest_cluster(self, x: float, y: float):
        area = self._choose_area(x, y)
        if area is None:
            return None
        best, score = None, -1.0
        for c in self._clusters(area):
            if c.count <= 0:
                continue
            s = c.count / (1.0 + math.hypot(c.center[0] - x, c.center[1] - y))
            if s > score:
                best, score = c, s
        return (area, best) if best is not None else None

    def _lamella_feature(self, i: int) -> Feature:
        lam = self.holder.lamellae[i]
        ordinal = sum(1 for k in range(i) if self.holder.lamellae[k].post_index == lam.post_index)
        kind = BULK_SAMPLE_KINDS[lam.bulk_kind]
        return Feature(center_um=lam.center, radius_um=max(abs(lam.half[0]), abs(lam.half[1])),
                       area_index=lam.area_index, post_index=lam.post_index, kind=kind, weight=1,
                       label=f"post {lam.post_index} lamella {ordinal} {kind}")

    def _area_feature(self, area: PlacementArea) -> Feature:
        kind = BULK_SAMPLE_KINDS[area.feature_kind] if area.feature_kind is not None else None
        if self.holder_kind == "insitu_heating_chip":
            label = f"window {area.index}"
        elif area.post_index >= 0:
            label = f"post {area.post_index} lamella 0 {kind}"
        else:
            label = f"hole {area.index}"
        return Feature(center_um=area.center, radius_um=area.half_span, area_index=area.index,
                       post_index=area.post_index, kind=kind, weight=1, label=label)

    def nearest_feature(self, x: float, y: float) -> Feature | None:
        lam = self.holder.lamellae
        if self.holder_kind == "fib_liftout" and lam:
            d = [math.hypot(l.center[0] - x, l.center[1] - y) for l in lam]
            return self._lamella_feature(int(np.argmin(d)))
        if not self.holder.areas:
            return None
        nc = self.nearest_cluster(x, y)
        if nc is not None:
            area, c = nc
            return Feature(center_um=c.center, radius_um=0.0, area_index=area.index, post_index=area.post_index,
                           kind=None, weight=c.count, label=f"hole {area.index} cluster")
        area = self._choose_area(x, y)
        return self._area_feature(area) if area is not None else None

    def features(self) -> list[Feature]:
        if self.holder_kind == "fib_liftout" and self.holder.lamellae:
            return [self._lamella_feature(i) for i in range(len(self.holder.lamellae))]
        areas = self.holder.areas
        if not areas:
            return []
        d2 = self.holder.area_cx ** 2 + self.holder.area_cy ** 2
        order = np.lexsort((np.arange(len(areas)), d2))[:MAX_FEATURE_TABLE]
        out = []
        for i in order:
            area = areas[int(i)]
            f = self._area_feature(area)
            if self.holder_kind != "insitu_heating_chip" and area.post_index < 0:
                cl = [c for c in self._clusters(area, resident_ok=False) if c.count > 0]
                if cl:
                    best = max(cl, key=lambda c: c.count)
                    f = Feature(center_um=best.center, radius_um=best.sigma_um, area_index=area.index,
                                post_index=area.post_index, kind=None, weight=best.count,
                                label=f"hole {area.index} cluster")
            out.append(f)
        return out
