"""What the calibration standards share: the structure interface, objects laid out cell by
cell, and grain orientations tied to an in-plane angle."""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from ...hashing import SeedKind, hash_seed, normal_from_hash, splitmix64, uniform_from_hash
from ..fieldmap import GRAINS_PER_MATERIAL
from ..materials import MaterialId
from ..structures import Structure

U64 = np.uint64


def structure_seed(owner) -> int:
    """The 32-bit seed a standard's structure draws its layout from."""
    return int(hash_seed(owner.seed, SeedKind.STRUCTURE, 0)) & 0xFFFFFFFF


class StandardStructure(Structure):
    """A standard's structure. Its owner is a rectangle of `base_material`, `base_nm` thick,
    that the structure edits; `mean_nm` is the carbon-equivalent area mean (the aggregate at
    low magnification) and `grain_textures` fixes crystal orientations it relies on."""

    required_layers = frozenset()
    base_material: int = MaterialId.AMORPHOUS_CARBON

    def base_nm(self) -> float:
        raise NotImplementedError

    def mean_nm(self) -> float:
        return self.base_nm()

    def grain_textures(self) -> tuple[tuple, tuple]:
        return (), ()


# ------------------------------------------------------------------ cells of objects
def cells_near(lx, ly, cell_um: float, reach_um: float) -> Iterator[tuple[int, int]]:
    """Lattice cells (i, j) whose objects (reaching `reach_um` from their cell) can touch the
    pixels at owner-local (lx, ly)."""
    x0, x1 = float(np.min(lx)) - reach_um, float(np.max(lx)) + reach_um
    y0, y1 = float(np.min(ly)) - reach_um, float(np.max(ly)) + reach_um
    for j in range(math.floor(y0 / cell_um), math.floor(y1 / cell_um) + 1):
        for i in range(math.floor(x0 / cell_um), math.floor(x1 / cell_um) + 1):
            yield i, j


def cell_rng(seed: int, salt: int, i: int, j: int) -> "CellRng":
    return CellRng(int(hash_seed(seed, SeedKind.STRUCTURE, (i * 73856093) ^ (j * 19349663), salt)))


class CellRng:
    """A tiny deterministic stream of numbers from a cell's hash."""

    def __init__(self, h: int):
        self.h = int(h) & 0xFFFFFFFFFFFFFFFF

    def _next(self) -> int:
        self.h = int(splitmix64(self.h))
        return self.h

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        return lo + (hi - lo) * float(uniform_from_hash(self._next()))

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return mu + sigma * float(normal_from_hash(self._next()))

    def lognormal(self, median: float, sigma: float) -> float:
        return median * math.exp(sigma * float(normal_from_hash(self._next())))

    def poisson(self, lam: float) -> int:
        """Knuth's method (small lam)."""
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self.uniform()
            if p <= L:
                return k
            k += 1

    def hash(self) -> int:
        return self._next()


# ------------------------------------------------------------------ orientation buckets
#: In-plane orientations are quantised to this many buckets over 180 degrees (0.35 deg).
ANGLE_BUCKETS = GRAINS_PER_MATERIAL


def angle_bucket(angle_rad) -> np.ndarray:
    """The bucket of an in-plane angle (radians, modulo pi)."""
    return np.mod(np.rint(np.asarray(angle_rad) / math.pi * ANGLE_BUCKETS).astype(np.int64), ANGLE_BUCKETS)


def angle_grain(material: int, angle_rad) -> np.ndarray:
    """Grain id of `material` whose orientation is the bucket of `angle_rad` (see
    `angle_overrides`)."""
    return (int(material) * GRAINS_PER_MATERIAL + angle_bucket(angle_rad)).astype(np.int32)


def angle_overrides(material: int, axis, inplane, spread_deg: float = 0.0) -> tuple:
    """Grain overrides giving grain k of `material` zone axis `axis` along the beam and crystal
    direction `inplane` at 180 k / ANGLE_BUCKETS degrees: crystals whose shape fixes their
    orientation pick their grain with `angle_grain`."""
    from ...crystal.orientation import Texture

    base = int(material) * GRAINS_PER_MATERIAL
    return tuple((base + k, Texture("oriented", tuple(axis), spread_deg, tuple(inplane), 180.0 * k / ANGLE_BUCKETS))
                 for k in range(ANGLE_BUCKETS))
