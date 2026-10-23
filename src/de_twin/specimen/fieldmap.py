"""Contract between the specimen model and the renderers.

The specimen rasterises the part of the world under a :class:`ViewWindow`
into a :class:`FieldMap` (struct-of-arrays layers). Renderers only ever see a
FieldMap plus the :class:`~de_twin.optics.state.OpticsState`; they never touch
holder/preparation geometry directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..hashing import SeedKind, hash_seed, rng_for
from .materials import GRAINS_PER_MATERIAL, material, n_materials

# Optional layer names a renderer may request.
LAYER_DESCAN = "descan"  # (2, ny, nx) float32 detector pixels (x, y)
LAYER_STRAIN = "strain"  # (3, ny, nx) float32 (xx, xy, yy); identity = (1, 0, 1)
LAYER_THICKNESS_FINE = "thickness_fine"  # float32 thickness (core layer is always float32 here)
OPTIONAL_LAYERS = frozenset({LAYER_DESCAN, LAYER_STRAIN})


@dataclass(frozen=True)
class ViewWindow:
    """A rectangular raster laid over world (specimen) coordinates.

    Pixel (row i, col j) centre maps to world coordinates::

        u = (j + 0.5 - nx/2) * pixel_um * flip_x
        v = (i + 0.5 - ny/2) * pixel_um * flip_y
        (u, v) /= (cos_beta, cos_alpha)          # tilt foreshortening
        world = center + R(rotation) @ (u, v)

    ``cos_alpha``/``cos_beta`` < 1 means the specimen is tilted, so one raster
    pixel covers more specimen along that axis (alpha rotates about x, beta about y;
    the same stage rotation turns every grain, see :mod:`de_twin.crystal.orientation`).
    """

    center_um: tuple[float, float]
    pixel_um: float
    shape: tuple[int, int]  # (ny, nx)
    rotation_rad: float = 0.0
    flip_x: bool = False
    flip_y: bool = False
    cos_alpha: float = 1.0
    cos_beta: float = 1.0

    @property
    def ny(self) -> int:
        return self.shape[0]

    @property
    def nx(self) -> int:
        return self.shape[1]

    def _uv(self, rows, cols):
        u = (np.asarray(cols, dtype=np.float64) + 0.5 - self.nx / 2.0) * self.pixel_um
        v = (np.asarray(rows, dtype=np.float64) + 0.5 - self.ny / 2.0) * self.pixel_um
        if self.flip_x:
            u = -u
        if self.flip_y:
            v = -v
        return u / max(self.cos_beta, 0.1), v / max(self.cos_alpha, 0.1)

    def pixel_to_world(self, rows, cols):
        u, v = self._uv(rows, cols)
        c, s = np.cos(self.rotation_rad), np.sin(self.rotation_rad)
        return self.center_um[0] + c * u - s * v, self.center_um[1] + s * u + c * v

    def world_to_pixel(self, x, y):
        dx = np.asarray(x, dtype=np.float64) - self.center_um[0]
        dy = np.asarray(y, dtype=np.float64) - self.center_um[1]
        c, s = np.cos(self.rotation_rad), np.sin(self.rotation_rad)
        u = c * dx + s * dy
        v = -s * dx + c * dy
        u *= max(self.cos_beta, 0.1)
        v *= max(self.cos_alpha, 0.1)
        if self.flip_x:
            u = -u
        if self.flip_y:
            v = -v
        cols = u / self.pixel_um + self.nx / 2.0 - 0.5
        rows = v / self.pixel_um + self.ny / 2.0 - 0.5
        return rows, cols

    def world_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """(X, Y) world coordinates of every pixel centre, each (ny, nx) float64."""
        rows = np.arange(self.ny)[:, None]
        cols = np.arange(self.nx)[None, :]
        x, y = self.pixel_to_world(rows, cols)
        return np.broadcast_to(x, self.shape), np.broadcast_to(y, self.shape)

    def row_world(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """World coordinates of one raster row (for row-wise samplers)."""
        return self.pixel_to_world(np.full(self.nx, row), np.arange(self.nx))

    def bounds_um(self) -> tuple[float, float, float, float]:
        """Axis-aligned world bounding box (xmin, ymin, xmax, ymax)."""
        rows = np.array([-0.5, -0.5, self.ny - 0.5, self.ny - 0.5])
        cols = np.array([-0.5, self.nx - 0.5, -0.5, self.nx - 0.5])
        x, y = self.pixel_to_world(rows, cols)
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    @property
    def lod_pixel_um(self) -> float:
        """Smallest world distance covered by one pixel (used for LOD decisions)."""
        return self.pixel_um


@dataclass
class FieldMap:
    """Rasterised specimen under a view.

    Core layers are always present; optional layers are ``None`` unless the
    renderer asked for them and the scene produced them.
    """

    view: ViewWindow
    material_id: np.ndarray  # (ny, nx) uint8, MaterialId
    thickness_nm: np.ndarray  # (ny, nx) float32 projected thickness (not tilt-corrected)
    grain_id: np.ndarray  # (ny, nx) int32, -1 = no grain / amorphous
    descan: np.ndarray | None = None  # (2, ny, nx) float32
    strain: np.ndarray | None = None  # (3, ny, nx) float32
    # An amorphous layer in the same column as the primary material (a support film under
    # particles, stain around protein, ...): its absorption, phase and diffuse scattering add
    # to the primary's. None when the scene has none.
    under_material: np.ndarray | None = None  # (ny, nx) uint8
    under_thickness_nm: np.ndarray | None = None  # (ny, nx) float32
    generation: int = 0  # bumps whenever content changes (cache key)
    time_s: float = 0.0  # scene time the raster represents

    @classmethod
    def empty(cls, view: ViewWindow, layers: frozenset[str] | set[str] = frozenset()) -> FieldMap:
        ny, nx = view.shape
        field_map = cls(
            view=view,
            material_id=np.zeros((ny, nx), np.uint8),
            thickness_nm=np.zeros((ny, nx), np.float32),
            grain_id=np.full((ny, nx), -1, np.int32),
        )
        if LAYER_DESCAN in layers:
            field_map.descan = np.zeros((2, ny, nx), np.float32)
        if LAYER_STRAIN in layers:
            field_map.strain = np.zeros((3, ny, nx), np.float32)
            field_map.strain[0] = 1.0
            field_map.strain[2] = 1.0
        return field_map


@dataclass
class GrainTable:
    """Fixed per-grain crystallography, indexed by grain id.

    Grain ids are ``material * GRAINS_PER_MATERIAL + k``; the table is generated once per
    seed (and texture set) so any grain id maps to the same orientation everywhere in the
    world. Orientations are orix/diffsims rotations stored as unit quaternions (see
    :mod:`de_twin.crystal.orientation` for the convention); the stage tilt is applied on
    top at render time, so tilting changes every pattern and the diffraction contrast.
    """

    quaternions: np.ndarray  # (N, 4) float64, orix Rotation data
    nucleation_u: np.ndarray  # (N,) float32 uniform in [0, 1) (in-situ crystallisation)
    _matrices: np.ndarray | None = field(default=None, repr=False, compare=False)

    @classmethod
    def generate(cls, seed: int, textures: tuple = (), overrides: tuple = ()) -> GrainTable:
        """Draw every grain from ``rng_for(seed, GRAIN, id)``. ``textures`` is a tuple of
        ``(material_id, Texture)`` (default: uniform random), ``overrides`` a tuple of
        ``(grain_id, Texture)`` for individual grains (e.g. a single-crystal lamella matrix)."""
        from ..crystal.orientation import RANDOM, rotation_from_crystal_to_lab

        tex = dict(textures)
        over = dict(overrides)
        n = n_materials() * GRAINS_PER_MATERIAL
        mats = np.empty((n, 3, 3))
        nuc = np.zeros(n, np.float32)
        for g in range(n):
            mid = g // GRAINS_PER_MATERIAL
            rng = rng_for(seed, SeedKind.GRAIN, g)
            nuc[g] = rng.uniform()
            mats[g] = over.get(g, tex.get(mid, RANDOM)).draw(rng, _direct_basis(mid))
        return cls(rotation_from_crystal_to_lab(mats), nuc)

    @property
    def matrices(self) -> np.ndarray:
        """(N, 3, 3) crystal -> specimen (untilted lab) matrices."""
        if self._matrices is None:
            from ..crystal.orientation import crystal_to_lab

            self._matrices = crystal_to_lab(self.quaternions)
        return self._matrices

    def __len__(self) -> int:
        return len(self.nucleation_u)

    def material_of(self, grain_id):
        return np.asarray(grain_id) // GRAINS_PER_MATERIAL


def _direct_basis(mid: int) -> np.ndarray:
    """[uvw] -> Cartesian for texture axes (identity for cubic lattices)."""
    c = material(mid).crystal
    if c is None or not c.lattice or (len(set(c.lattice[:3])) == 1 and set(c.lattice[3:]) == {90.0}):
        return np.eye(3)
    from ..crystal.phases import phase_for

    return np.asarray(phase_for(mid).structure.lattice.base).T


def grain_id_from_hash(material_id, h):
    """Grain id for a feature of ``material_id`` from a 64-bit hash (C++ GrainIdFromHash)."""
    k = hash_seed(h, SeedKind.GRAIN, 0)
    if isinstance(k, np.ndarray):
        return (
            np.asarray(material_id, np.int64) * GRAINS_PER_MATERIAL
            + (k % np.uint64(GRAINS_PER_MATERIAL)).astype(np.int64)
        ).astype(np.int32)
    return int(material_id) * GRAINS_PER_MATERIAL + int(k % GRAINS_PER_MATERIAL)
