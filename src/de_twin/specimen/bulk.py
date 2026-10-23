"""Bulk polished specimens for EBSD: a polycrystal surface with a grain map and known orientations.

A :class:`BulkSurface` is a flat, infinite, polished surface (sample frame: x right, y down,
z into the sample, the de-twin specimen frame). Its microstructure is a Voronoi tessellation
of a jittered lattice of grain sites (:class:`~de_twin.specimen.geometry.JitteredLattice`,
the same hashed lattices the TEM specimens use), so it is deterministic, infinite and
evaluated point by point. Every grain has

* a phase, drawn per grain from the phase fractions (``BulkPhase``), and
* an orientation drawn from that phase's :class:`~de_twin.crystal.Texture` with a generator
  seeded by the grain's hash, unique per grain (not the 512-per-material TEM grain table).

Near a boundary the EBSD signal mixes the two grains (see :meth:`BulkSurface.sample`), and the
pattern quality falls with surface damage (a uniform deformed layer plus polishing scratches).

The specimen also rasterises like a TEM specimen (opaque bulk: 100 µm thick, grain ids from
the shared grain table) so it can sit in any twin; EBSD is what it is for.

Presets (:data:`BULK_PRESETS`): FCC nickel (random, and cube-textured aluminium), BCC ferrite
with a gamma fibre, a ferrite + austenite duplex steel and a Si (001) wafer for pattern-centre
calibration.
"""

from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..crystal.orientation import RANDOM, Texture
from ..hashing import SeedKind, hash_seed, rng_for, splitmix64, uniform_from_hash
from .fieldmap import FieldMap, GrainTable, ViewWindow, _direct_basis
from .geometry import JitteredLattice
from .materials import GRAINS_PER_MATERIAL, MaterialId, material

#: Thickness a bulk specimen reports to TEM renderers (nm): opaque.
BULK_THICKNESS_NM = 1.0e5
_PHASE_SALT = np.uint64(0x5A17C0DE)
_SCRATCH_SALT = 0x5C2A7C


@dataclass(frozen=True)
class BulkPhase:
    material: int
    fraction: float = 1.0
    texture: Texture = RANDOM


@dataclass
class BulkSurfaceConfig:
    """What a bulk EBSD specimen is made of (all lengths µm)."""

    name: str = ""
    seed: int = 42
    phases: tuple = (BulkPhase(MaterialId.NICKEL),)
    grain_size_um: float = 8.0  # lattice cell = mean grain spacing
    grain_jitter: float = 0.45  # 0: square grains; 0.45: an irregular, equiaxed tessellation
    single_crystal: bool = False  # one grain everywhere (phase 0)
    surface_damage: float = 0.05  # 0..1: pattern quality lost everywhere (deformed layer)
    scratch_density_per_mm: float = 0.0  # polishing scratches crossing a millimetre
    scratch_width_um: float = 1.0
    scratch_depth: float = 0.7  # quality lost at a scratch's centre
    scratch_angle_deg: float = 30.0  # mean scratch direction from +x
    scratch_angle_spread_deg: float = 25.0
    #: Grain-boundary zone: polishing / etching grooves, relief and the distorted lattice next
    #: to a boundary lower the pattern quality within ~0.1-0.3 µm of it (what makes boundaries
    #: continuous dark lines in real band-contrast maps at 0.1-0.5 µm steps; the EBSD footprint
    #: alone, ~30 x 80 nm for Ni at 20 kV, only mixes points within it). Gaussian sigma, µm, and
    #: the quality lost on the boundary.
    boundary_zone_um: float = 0.2
    boundary_zone_depth: float = 0.5
    #: Indents (:class:`~de_twin.specimen.indent.Indent`): residual strain, lattice rotation,
    #: plastic zone and cracks around each (the EBSD renderer distorts the patterns with them).
    indents: tuple = ()

    def copy(self) -> "BulkSurfaceConfig":
        return copy.deepcopy(self)


class BulkSurface:
    """A polished bulk polycrystal (see the module docstring)."""

    raster_is_local = False
    time_dependent = False
    generation = 0
    #: EBSD renderers call :meth:`sample` instead of rasterising.
    ebsd_surface = True

    def __init__(self, config: Optional[BulkSurfaceConfig] = None):
        self.config = (config or BulkSurfaceConfig()).copy()
        c = self.config
        fr = np.array([max(float(p.fraction), 0.0) for p in c.phases], np.float64)
        if not len(fr) or fr.sum() <= 0:
            raise ValueError("a bulk specimen needs at least one phase with a positive fraction")
        for p in c.phases:
            if material(int(p.material)).crystal is None:
                raise ValueError(f"phase material {p.material} is not crystalline")
        self._cum = np.cumsum(fr / fr.sum())
        self._mats = np.array([int(p.material) for p in c.phases], np.uint8)
        self._lattice = JitteredLattice(
            max(float(c.grain_size_um), 1e-3),
            hash_seed(c.seed, SeedKind.STRUCTURE, 0),
            0,
            jitter_frac=min(max(float(c.grain_jitter), 0.0), 0.49),
        )
        self._orient: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()
        self._scratches = self._make_scratches()
        from .indent import IndentField

        self._indents = [IndentField(i) for i in self.config.indents]
        self._time_s = 0.0

    # ------------------------------------------------------------------ specimen duck type
    @property
    def name(self) -> str:
        return self.config.name

    @property
    def grains(self) -> GrainTable:
        """The shared TEM grain table (only for TEM rendering of the raster's grain ids)."""
        from .model import _grain_table

        tex = tuple((int(p.material), p.texture) for p in self.config.phases)
        return _grain_table(int(self.config.seed) & 0xFFFFFFFF, tex, ())

    def update(self, time_s: float, holder=None) -> None:
        self._time_s = float(time_s)

    def bounds_um(self) -> tuple[float, float, float, float]:
        return (-5000.0, -5000.0, 5000.0, 5000.0)  # a 10 mm polished coupon

    def nearest_feature(self, x_um: float, y_um: float):
        return None

    def features(self) -> list:
        return []

    def required_layers(self) -> frozenset:
        return frozenset()

    def rasterize(self, view: ViewWindow, layers: frozenset = frozenset(), **_) -> FieldMap:
        x, y = view.world_grid()
        s = self.sample(x.ravel(), y.ravel(), blend=False)
        ny, nx = view.shape
        fm = FieldMap.empty(view, layers)
        fm.material_id[:] = s["material"].reshape(ny, nx)
        fm.thickness_nm[:] = BULK_THICKNESS_NM
        k = (s["hash"] % np.uint64(GRAINS_PER_MATERIAL)).astype(np.int64)
        fm.grain_id[:] = (s["material"].astype(np.int64) * GRAINS_PER_MATERIAL + k).reshape(ny, nx)
        return fm

    # ------------------------------------------------------------------ microstructure
    def _phase_of(self, h: np.ndarray) -> np.ndarray:
        if len(self._mats) == 1:
            return np.zeros(h.shape, np.int64)
        u = uniform_from_hash(splitmix64(h ^ _PHASE_SALT))
        return np.minimum(np.searchsorted(self._cum, u, side="right"), len(self._mats) - 1)

    def sample(self, x_um, y_um, *, blend: bool = True) -> dict:
        """The microstructure at surface points (arrays, sample frame, µm).

        Returns ``hash`` (uint64, the grain's key) and ``grain_id`` (int32 >= 0, unique per
        grain up to hash collisions), ``phase`` (index into ``config.phases``) and ``material``
        of the grain each point is in; with ``blend``, also the nearest other grain
        (``hash2``, ``grain_id2``, ``phase2``, ``material2``), the distance to the boundary
        between them (``boundary_um``) and its unit normal (``normal``, (n, 2), pointing into
        the other grain), and the surface ``quality`` in (0, 1].
        """
        x = np.asarray(x_um, np.float64).ravel()
        y = np.asarray(y_um, np.float64).ravel()
        n = x.size
        c = self.config
        if c.single_crystal:
            h1 = np.full(n, np.uint64(hash_seed(c.seed, SeedKind.GRAIN, 0)))
            out = {"hash": h1, "phase": np.zeros(n, np.int64)}
            if blend:
                out.update(
                    hash2=h1.copy(),
                    phase2=out["phase"].copy(),
                    boundary_um=np.full(n, np.inf),
                    normal=np.zeros((n, 2)),
                )
        else:
            out = self._voronoi(x, y, blend)
        for suffix in ("", "2") if blend else ("",):
            if "hash" + suffix in out:
                out["material" + suffix] = self._mats[out["phase" + suffix]]
                out["grain_id" + suffix] = (out["hash" + suffix] >> np.uint64(33)).astype(np.int32)
        if blend:
            q = self.quality(x, y)
            c = self.config
            if not c.single_crystal and c.boundary_zone_um > 0 and c.boundary_zone_depth > 0:
                d = np.asarray(out["boundary_um"], np.float64) / float(c.boundary_zone_um)
                q = q * (1.0 - min(float(c.boundary_zone_depth), 0.95) * np.exp(-0.5 * d * d)).astype(
                    np.float32
                )
            if self._indents:
                F = np.tile(np.eye(3), (x.size, 1, 1))
                gnd = np.zeros(x.size)
                for f in self._indents:
                    F += f.deformation_gradient(x, y) - np.eye(3)  # superposed (small) displacement gradients
                    gnd += f.gnd(x, y)
                    q = q * f.quality(x, y).astype(np.float32)
                out["F"], out["gnd"] = F, gnd
            out["quality"] = q
        return out

    def _voronoi(self, x, y, blend: bool) -> dict:
        lat = self._lattice
        L = lat.cell_um
        ci = np.floor(x / L).astype(np.int64)
        cj = np.floor(y / L).astype(np.int64)
        n = x.size
        d1 = np.full(n, np.inf)
        d2 = np.full(n, np.inf)
        s1 = np.zeros((n, 2))
        s2 = np.zeros((n, 2))
        h1 = np.zeros(n, np.uint64)
        h2 = np.zeros(n, np.uint64)
        for dj in range(-2, 3):
            for di in range(-2, 3):
                sx, sy, hh = lat.sites(ci + di, cj + dj)
                d = (sx - x) ** 2 + (sy - y) ** 2
                first = d < d1
                second = ~first & (d < d2)
                # demote the old nearest to second where a new nearest arrives
                d2 = np.where(first, d1, np.where(second, d, d2))
                h2 = np.where(first, h1, np.where(second, hh, h2))
                s2[first] = s1[first]
                s2[second, 0], s2[second, 1] = sx[second], sy[second]
                d1 = np.where(first, d, d1)
                h1 = np.where(first, hh, h1)
                s1[first, 0], s1[first, 1] = sx[first], sy[first]
        out = {"hash": h1, "phase": self._phase_of(h1)}
        if blend:
            sep = s2 - s1
            dist = np.linalg.norm(sep, axis=1)
            nrm = sep / np.maximum(dist, 1e-12)[:, None]
            out.update(
                hash2=h2,
                phase2=self._phase_of(h2),
                normal=nrm,
                boundary_um=np.maximum((d2 - d1) / (2.0 * np.maximum(dist, 1e-12)), 0.0),
            )
        return out

    def orientations(self, hashes, phases) -> np.ndarray:
        """Crystal->sample matrices (n, 3, 3) of the grains with these hashes / phase indices."""
        h = np.asarray(hashes, np.uint64).ravel()
        p = np.asarray(phases, np.int64).ravel()
        out = np.empty((h.size, 3, 3))
        with self._lock:
            for i, (hi, pi) in enumerate(zip(h.tolist(), p.tolist())):
                m = self._orient.get(hi)
                if m is None:
                    ph = self.config.phases[pi]
                    rng = rng_for(self.config.seed, SeedKind.GRAIN, hi, 0xEB5D)
                    m = ph.texture.draw(rng, _direct_basis(int(ph.material)))
                    self._orient[hi] = m
                    if len(self._orient) > 200_000:
                        self._orient.popitem(last=False)
                out[i] = m
        return out

    # ------------------------------------------------------------------ surface quality
    def _make_scratches(self):
        c = self.config
        k = int(round(max(c.scratch_density_per_mm, 0.0) * 10.0))  # over the 10 mm coupon
        if k <= 0:
            return None
        rng = rng_for(c.seed, SeedKind.STRUCTURE, _SCRATCH_SALT)
        ang = np.radians(c.scratch_angle_deg + c.scratch_angle_spread_deg * rng.standard_normal(k))
        normal = np.stack([-np.sin(ang), np.cos(ang)], 1)
        offset = rng.uniform(-5000.0, 5000.0, k)
        width = c.scratch_width_um * np.exp(0.4 * rng.standard_normal(k))
        depth = c.scratch_depth * rng.uniform(0.4, 1.0, k)
        return normal, offset, width, depth

    def quality(self, x_um, y_um) -> np.ndarray:
        """Pattern quality factor in (0, 1] of the surface at points (1: perfect)."""
        x = np.asarray(x_um, np.float64).ravel()
        y = np.asarray(y_um, np.float64).ravel()
        q = np.full(x.size, 1.0 - min(max(self.config.surface_damage, 0.0), 0.95))
        if self._scratches is not None:
            nrm, off, wid, dep = self._scratches
            loss = np.zeros(x.size)
            for k in range(len(off)):
                d = x * nrm[k, 0] + y * nrm[k, 1] - off[k]
                loss = np.maximum(loss, dep[k] * np.exp(-0.5 * (d / wid[k]) ** 2))
            q *= 1.0 - loss
        return np.clip(q, 0.02, 1.0).astype(np.float32)


# ---------------------------------------------------------------------- presets
def _p(mid, frac=1.0, tex=RANDOM):
    return BulkPhase(int(mid), float(frac), tex)


#: Cube texture {001}<100>, 10 deg spread (recrystallised aluminium / nickel sheet).
CUBE = Texture("oriented", (0, 0, 1), 10.0, (1, 0, 0), 0.0)
#: Gamma fibre <111> || ND, 10 deg spread (rolled and annealed ferritic steel).
GAMMA_FIBRE = Texture("fibre", (1, 1, 1), 10.0)

_BULK = {
    "EBSD Ni polycrystal": BulkSurfaceConfig(
        name="EBSD Ni polycrystal", phases=(_p(MaterialId.NICKEL),), grain_size_um=8.0, surface_damage=0.05
    ),
    "EBSD Al cube texture": BulkSurfaceConfig(
        name="EBSD Al cube texture",
        seed=7,
        phases=(_p(MaterialId.ALUMINUM, 1.0, CUBE),),
        grain_size_um=20.0,
        surface_damage=0.1,
        scratch_density_per_mm=3.0,
    ),
    "EBSD ferrite gamma fibre": BulkSurfaceConfig(
        name="EBSD ferrite gamma fibre",
        seed=11,
        phases=(_p(MaterialId.IRON, 1.0, GAMMA_FIBRE),),
        grain_size_um=10.0,
        surface_damage=0.1,
    ),
    "EBSD duplex steel": BulkSurfaceConfig(
        name="EBSD duplex steel",
        seed=23,
        phases=(_p(MaterialId.IRON, 0.55), _p(MaterialId.AUSTENITE, 0.45)),
        grain_size_um=6.0,
        surface_damage=0.15,
        scratch_density_per_mm=2.0,
        scratch_width_um=0.6,
    ),
    "EBSD Si (001) wafer": BulkSurfaceConfig(
        name="EBSD Si (001) wafer",
        seed=0,
        phases=(_p(MaterialId.SILICON, 1.0, Texture("oriented", (0, 0, 1), 0.0, (1, 1, 0), 0.0)),),
        single_crystal=True,
        surface_damage=0.0,
    ),
}
_INDENT_SI = dict(
    phases=(_p(MaterialId.SILICON, 1.0, Texture("oriented", (0, 0, 1), 0.0, (1, 1, 0), 0.0)),),
    single_crystal=True,
    surface_damage=0.0,
    seed=0,
)


def _indent(kind, **kw):
    from .indent import Indent

    return Indent(kind=kind, **kw)


_BULK["EBSD Si (001) Vickers indent"] = BulkSurfaceConfig(
    name="EBSD Si (001) Vickers indent", indents=(_indent("vickers", size_um=5.0),), **_INDENT_SI
)
_BULK["EBSD Si (001) Berkovich indent"] = BulkSurfaceConfig(
    name="EBSD Si (001) Berkovich indent",
    indents=(_indent("berkovich", size_um=6.0, rotation_deg=-90.0),),
    **_INDENT_SI,
)

#: Bulk EBSD specimens by name (``DigitalTwin(specimen=<name>)`` builds a :class:`BulkSurface`).
BULK_PRESETS: dict[str, BulkSurfaceConfig] = _BULK


def bulk_from_name(name: str, seed: Optional[int] = None) -> Optional[BulkSurfaceConfig]:
    """The bulk preset called ``name`` (case-insensitive), or None."""
    key = str(name).strip().lower()
    for k, cfg in BULK_PRESETS.items():
        if k.lower() == key:
            out = cfg.copy()
            if seed is not None:
                out.seed = int(seed)
            return out
    return None
