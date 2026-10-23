"""The :class:`Specimen` facade: scene + clock + in-situ thermal budget."""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Optional

import numpy as np

from ..state import HolderState
from .fieldmap import OPTIONAL_LAYERS, FieldMap, GrainTable, ViewWindow
from .materials import MaterialId
from .options import CLEARS_POPULATION, NON_REGENERATING, SpecimenConfig
from . import raster_nb as _raster_nb
from .raster import RasterContext, rasterize_scene
from .scene import Feature, Scene

#: A view is rasterised in parallel strips no thinner than this (rows).
MIN_STRIP_ROWS = 192


def _numba_threadsafe() -> bool:
    """Whether numba's parallel kernels may be called from several threads at once: its
    'omp' and 'tbb' threading layers are thread-safe, 'workqueue' is not. Unknown until
    a parallel kernel has run (then the first raster is serial)."""
    try:
        import numba
    except Exception:  # noqa: BLE001 - no numba: the NumPy paths
        return True
    try:
        return numba.threading_layer() in ("omp", "tbb")
    except ValueError:
        return False


# Thermal budget model (OpticsState.h / AdvanceThermalBudget).
REFERENCE_OVERHEAT_C = 100.0  # degrees above onset at which the budget advances at 1 s/s
MELT_RATE_PER_S = 40.0  # budget units shed per second at/above the melting point


def advance_thermal_budget(budget_s: float, temperature_c: float, onset_c: float, melt_c: float,
                           dt_s: float) -> float:
    """Integrate the crystallisation budget over ``dt_s`` at a constant temperature.

    Above the melting point the budget drains at 40 /s (re-amorphisation); below the onset it is
    frozen (a quench keeps whatever state it has); in between it grows linearly with overheat,
    1 s of budget per second at onset + 100 C.
    """
    if not dt_s > 0:
        return budget_s
    if temperature_c >= melt_c:
        return max(0.0, budget_s - MELT_RATE_PER_S * dt_s)
    if temperature_c < onset_c:
        return budget_s
    return budget_s + (temperature_c - onset_c) / REFERENCE_OVERHEAT_C * dt_s


def crystalline_fraction(nucleation_u, budget_s: float, start_s: float, end_s: float, growth_s: float):
    """Per-grain crystalline fraction (CrystalBinFor, continuous instead of 8 bins).

    Nucleation is log-uniform over the ``[start, end]`` budget window (more grains go early than
    late), followed by a linear growth ramp of ``growth_s``.
    """
    u = np.clip(np.asarray(nucleation_u, np.float64), 0.0, 1.0)
    tmin = max(0.1, start_s)
    tmax = max(tmin * 1.01, end_s)
    t_nuc = tmin * np.power(tmax / tmin, u)
    if not growth_s > 0:
        return (budget_s > t_nuc).astype(np.float32)
    return np.clip((budget_s - t_nuc) / growth_s, 0.0, 1.0).astype(np.float32)


@functools.lru_cache(maxsize=8)
def _grain_table(seed: int, textures: tuple = (), overrides: tuple = ()) -> GrainTable:
    return GrainTable.generate(seed, textures, overrides)


def grain_textures(scene: Scene) -> tuple[tuple, tuple]:
    """Default textures of a scene: ``<111>`` fibre for deposited thin films, a single crystal
    near a low-index zone ([110] for Si, [001] otherwise) for every FIB lamella matrix, and
    uniform random orientations for everything else (particles, polycrystals)."""
    from ..crystal.orientation import FIBRE_111, SI_110_LAMELLA, Texture
    from .preparations import BulkSamplePreparation, ThinFilmPreparation

    textures = []
    if isinstance(scene.preparation, ThinFilmPreparation):
        textures.append((int(scene.preparation.material), FIBRE_111))
    overrides = {}
    for area in scene.holder.areas:
        prep = scene.preparation_for(area)
        hook = getattr(prep, "grain_textures", None)
        if hook is not None:  # a preparation that fixes its own crystals' orientations
            t, o = hook()
            textures.extend(t)
            overrides.update(dict(o))
        if isinstance(prep, BulkSamplePreparation):
            gid = prep._matrix_grain(area, scene.area_seed(area.index))
            if gid >= 0:
                overrides[int(gid)] = (SI_110_LAMELLA if prep.material == MaterialId.SILICON
                                       else Texture("single", (0, 0, 1), 0.6))
    return tuple(textures), tuple(sorted(overrides.items()))


class Specimen:
    """A deterministic virtual specimen: holder + preparation + time evolution.

    >>> s = Specimen(SpecimenConfig(seed=1))
    >>> fm = s.rasterize(ViewWindow(center_um=(0.0, 0.0), pixel_um=0.002, shape=(256, 256)))
    """

    #: Every raster pixel depends only on where it is and the pixel size, not on the rest of
    #: the view, so a raster can be assembled from rasters of its parts (the renderer does
    #: that on a stage move; tests/test_render_caches.py checks it).
    raster_is_local = True

    def __init__(self, config: Optional[SpecimenConfig] = None):
        self.config = (config or SpecimenConfig()).copy()
        self.options = self.config.resolved_options()
        self._build()
        self._time_s = 0.0
        self._last_update_s: Optional[float] = None
        self._budget_s = 0.0
        self._temperature_c = self.options.specimen_temperature_c
        self._generation = 0

    def _build(self):
        self.scene = Scene(self.config, self.options)
        self.grains: GrainTable = _grain_table(int(self.config.seed) & 0xFFFFFFFF, *grain_textures(self.scene))

    # -- where to start looking ------------------------------------------------------------------
    def home_um(self) -> tuple[float, float]:
        """Stage position (µm) a twin starts at: somewhere there is specimen to see.

        The stage origin is often not: on a waffle grid it is a bar, on a heating chip the
        frame, on a sparse grid a bare patch of film. So: the placement area (grid square,
        well, lamella, chip window) nearest the origin whose film is intact where the holder
        has a film, and inside it the largest particle cluster when the preparation clusters,
        else the area's centre.
        """
        scene = self.scene
        areas = list(scene.holder.areas)
        if not areas:
            return (0.0, 0.0)
        filmed = self.config.holder in ("mesh_grid", "waffle_grid") and scene.film not in ("none",)

        def usable(a) -> bool:
            return (a.film_covered and not a.film_broken) if filmed else True

        ordered = sorted(areas, key=lambda a: math.hypot(a.center[0], a.center[1]))
        area = next((a for a in ordered if usable(a)), ordered[0])
        try:
            clusters = list(scene._clusters(area, resident_ok=False))
        except Exception:  # noqa: BLE001 - a preparation without clusters
            clusters = []
        if clusters:
            best = max(clusters, key=lambda c: (c.count, -math.hypot(c.center[0] - area.center[0],
                                                                     c.center[1] - area.center[1])))
            return (float(best.center[0]), float(best.center[1]))
        return (float(area.center[0]), float(area.center[1]))

    # -- options --------------------------------------------------------------------------------
    def set_options(self, **options) -> None:
        """Change options live. World-changing options regenerate the scene (and reset the
        thermal budget, as a new specimen is a new experiment); others apply in place."""
        new = self.options.replace(**options)
        changed = {k for k in options if getattr(new, k) != getattr(self.options, k)}
        self.config.options.update(options)
        self.options = new
        if changed - NON_REGENERATING:
            self._build()
            self._budget_s = 0.0
        elif changed & CLEARS_POPULATION:
            self._build()  # baked into populated areas: rebuild, keep clock and thermal budget
        else:
            self.scene.opts = new

    # -- clock ----------------------------------------------------------------------------------
    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def time_index(self) -> int:
        return int(math.floor(self._time_s / self.options.time_step_s + 1e-9)) if self._time_s > 0 else 0

    @property
    def thermal_budget_s(self) -> float:
        return self._budget_s

    @property
    def temperature_c(self) -> float:
        return self._temperature_c

    @property
    def crystallization_enabled(self) -> bool:
        v = self.options.in_situ_crystallization
        return (self.config.holder == "insitu_heating_chip") if v is None else bool(v)

    @property
    def time_dependent(self) -> bool:
        return self.scene.time_dependent

    def update(self, time_s: float, holder: Optional[HolderState] = None) -> None:
        """Advance the scene clock to ``time_s`` and integrate the thermal budget.

        The temperature is ``holder.temperature_c`` when a holder state is given (and is not a
        ``kind == "none"`` placeholder), else ``options.specimen_temperature_c``. It is taken as
        constant over the interval since the previous update. The first call only sets the
        reference time. Time going backwards moves the clock but never un-crystallises.
        """
        time_s = float(time_s)
        if holder is not None and getattr(holder, "kind", "none") != "none":
            temp = float(holder.temperature_c)
        else:
            temp = float(self.options.specimen_temperature_c)
        if self._last_update_s is not None:
            dt = time_s - self._last_update_s
            self._budget_s = advance_thermal_budget(self._budget_s, temp, self.options.crystallization_onset_c,
                                                    self.options.melting_temperature_c, dt)
        self._last_update_s = time_s
        self._time_s = time_s
        self._temperature_c = temp

    def reset_clock(self) -> None:
        self._time_s = 0.0
        self._last_update_s = None
        self._budget_s = 0.0

    def crystallinity(self, grain_id) -> np.ndarray:
        """Crystalline fraction in [0, 1] per grain id.

        Grain -1 on crystalline material means "not yet nucleated" (glassy, 0) in a
        time-evolving in-situ scene, and "unresolved fine grains" (powder, 1) elsewhere.
        """
        g = np.asarray(grain_id)
        out = np.ones(g.shape, np.float32)
        if self.time_dependent:
            out[g < 0] = 0.0
        if not self.crystallization_enabled:
            return out
        ok = (g >= 0) & (g < self.grains.nucleation_u.shape[0])
        o = self.options
        out[ok] = crystalline_fraction(self.grains.nucleation_u[g[ok]], self._budget_s, o.nucleation_start_s,
                                       o.nucleation_end_s, o.crystal_growth_s)
        return out

    # -- geometry -------------------------------------------------------------------------------
    def bounds_um(self) -> tuple[float, float, float, float]:
        return self.scene.holder.bounds.as_tuple()

    def required_layers(self) -> frozenset:
        """Optional FieldMap layers this specimen can fill (descan/strain for FIB lamellae)."""
        return frozenset(self.scene.required_layers)

    def rasterize(self, view: ViewWindow, layers: frozenset = frozenset(), *, span_px: Optional[int] = None,
                  threads: int = 1) -> FieldMap:
        """The specimen on ``view``. ``span_px``: the size of the whole view when ``view`` is
        a piece of one (the scene draws a piece as it draws the whole). ``threads`` > 1
        rasterises row strips of a large view in parallel (the pixels are local, so the
        strips are the rows of the whole), when numba's threading layer allows it."""
        layers = frozenset(layers) & OPTIONAL_LAYERS
        drifted = view
        dx, dy = self.drift_um()
        if dx or dy:  # the specimen moved by (dx, dy): look at world - drift
            drifted = dataclasses.replace(view, center_um=(view.center_um[0] - dx, view.center_um[1] - dy))
        span = int(span_px) if span_px else max(view.shape)
        n = max(1, min(int(threads), max(view.shape) // MIN_STRIP_ROWS))
        if n > 1 and _numba_threadsafe() and self._numpy_work(drifted, span):
            fm = self._rasterize_strips(drifted, layers, span, n)
        else:
            ctx = self._rasterize_ctx(drifted, layers, span)
            self.last_stats = dict(ctx.stats, **self.scene.last_stats)
            fm = ctx.to_fieldmap(time_s=self._time_s)
        self._generation += 1
        fm.generation = self._generation
        fm.view = view
        return fm

    def _rasterize_ctx(self, view: ViewWindow, layers: frozenset, span: int) -> RasterContext:
        ctx = RasterContext(view, layers, grains=self.grains, time_index=self.time_index,
                            lod_threshold_px=self.options.lod_threshold_px, span_px=span)
        rasterize_scene(self.scene, ctx, curtain_depth=min(max(self.options.curtain_depth, 0.0), 1.0))
        return ctx

    def _numpy_work(self, view: ViewWindow, span: int) -> bool:
        """Whether rasterising ``view`` runs NumPy code (a structure or primitive profile without
        a numba kernel): only then are parallel strips worth their overhead, the numba kernels
        being parallel already. A function of the view alone (the same scene query the raster
        makes), so a view is always rasterised the same way."""
        from types import SimpleNamespace

        from .geometry import AABB, DUST_THRESHOLD_PX
        from .raster import Profile

        if not _raster_nb.AVAILABLE or self.scene.holder.kind not in ("mesh_grid", "waffle_grid", "none"):
            return True
        ny, nx = view.shape
        px = float(view.pixel_um)
        probe = SimpleNamespace(aabb=AABB(*view.bounds_um()), ny=ny, nx=nx, span_px=span, pixel_um=px,
                                lod_threshold_px=self.options.lod_threshold_px,
                                diameter_px=lambda r: 2.0 * np.asarray(r) / px)
        P = self.scene.query(probe)
        dpx = probe.diameter_px(P.bound_r)
        draw = P.intersects(probe.aabb) & (dpx >= probe.lod_threshold_px) & (dpx >= DUST_THRESHOLD_PX)
        if np.any(P.profile[draw] == Profile.CURTAIN):
            return True
        for i in np.flatnonzero(draw & (P.structure >= 0)):
            if not P.structures[P.structure[i]].numba_fill():
                return True
        return False

    def _rasterize_strips(self, view: ViewWindow, layers: frozenset, span: int, n: int) -> FieldMap:
        """``n`` strips across the longer side of the view (rows of a tall view, columns of a
        wide one: the strip a stage move brings in is wide and short), in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        ny, nx = view.shape
        by_rows = ny >= nx
        edges = np.linspace(0, ny if by_rows else nx, n + 1).astype(int)

        def strip(i):
            import numba

            numba.set_num_threads(max(1, numba.config.NUMBA_NUM_THREADS // n))  # thread-local
            a0, a1 = int(edges[i]), int(edges[i + 1])
            r0, r1, c0, c1 = (a0, a1, 0, nx) if by_rows else (0, ny, a0, a1)
            x, y = view.pixel_to_world(np.array([(r0 + r1 - 1) / 2.0]), np.array([(c0 + c1 - 1) / 2.0]))
            sub = dataclasses.replace(view, center_um=(float(x[0]), float(y[0])), shape=(r1 - r0, c1 - c0))
            return (r0, r1, c0, c1), self._rasterize_ctx(sub, layers, span).to_fieldmap(time_s=self._time_s)
        with ThreadPoolExecutor(n, thread_name_prefix="de-twin-raster") as ex:
            parts = list(ex.map(strip, range(n)))
        fm = FieldMap.empty(view, layers)
        fm.time_s = self._time_s
        if any(p.under_thickness_nm is not None for _, p in parts):
            fm.under_material = np.zeros((ny, nx), np.uint8)
            fm.under_thickness_nm = np.zeros((ny, nx), np.float32)
        for (r0, r1, c0, c1), p in parts:
            for name in ("material_id", "thickness_nm", "grain_id", "under_material", "under_thickness_nm"):
                a = getattr(p, name)
                if a is not None:
                    getattr(fm, name)[r0:r1, c0:c1] = a
            for name in ("descan", "strain"):
                a = getattr(p, name)
                if a is not None and getattr(fm, name) is not None:
                    getattr(fm, name)[:, r0:r1, c0:c1] = a
        self.last_stats = dict(self.scene.last_stats, strips=n)
        return fm

    def drift_um(self) -> tuple[float, float]:
        """Accumulated specimen drift (um) at the current time index."""
        t = self.time_index if self.time_dependent else 0
        o = self.options
        return (o.drift_per_step_x_nm * t / 1000.0, o.drift_per_step_y_nm * t / 1000.0)

    def features(self) -> list[Feature]:
        return self.scene.features()

    def nearest_feature(self, x_um: float, y_um: float) -> Optional[Feature]:
        return self.scene.nearest_feature(float(x_um), float(y_um))

    def overview(self, shape=(1024, 1024)) -> np.ndarray:
        """Whole-holder projected-thickness image (DebugOverview), float32 nm, 5 % padding."""
        xmin, ymin, xmax, ymax = self.bounds_um()
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        ny, nx = int(shape[0]), int(shape[1])
        px = max(1.05 * (xmax - xmin) / nx, 1.05 * (ymax - ymin) / ny)
        fm = self.rasterize(ViewWindow(center_um=(cx, cy), pixel_um=px, shape=(ny, nx)))
        return fm.thickness_nm
