"""Crystal standards: molybdenum trioxide laths (625, image / diffraction rotation)."""

from __future__ import annotations

import math

import numpy as np

from ..materials import MaterialId
from ..structures import _claim, _owner_pixels, _resolved
from .common import StandardStructure, angle_grain, angle_overrides, cell_rng, cells_near, structure_seed
from .replica import carbon_equivalent


class MoO3LathsStructure(StandardStructure):
    """MoO3 crystals on a carbon film: thin laths lying on their (010) face (Pbnm; (100) in the
    Pnma setting used here), elongated along [001] (Pbnm c = 0.3697 nm; Pnma b). The long
    edge of a lath is a known crystal direction, so comparing it in the image with the
    spots of its diffraction pattern measures the image / diffraction rotation."""

    def __init__(
        self,
        film_nm: float = 15.0,
        cell_um: float = 4.0,
        per_cell: float = 1.3,
        length_um: float = 2.5,
        length_sigma: float = 0.45,
        aspect=(3.0, 7.0),
        thickness_nm: float = 60.0,
        thickness_sigma: float = 0.35,
    ):
        self.film, self.cell, self.per_cell = float(film_nm), float(cell_um), float(per_cell)
        self.length, self.length_sigma, self.aspect = float(length_um), float(length_sigma), tuple(aspect)
        self.thick, self.thick_sigma = float(thickness_nm), float(thickness_sigma)
        self.max_half = 4.0  # um

    def base_nm(self) -> float:
        return self.film

    def mean_nm(self) -> float:
        L = self.length * math.exp(0.5 * self.length_sigma**2)
        w = L / (0.5 * sum(self.aspect))
        t = self.thick * math.exp(0.5 * self.thick_sigma**2)
        eq = carbon_equivalent(MaterialId.MOLYBDENUM_TRIOXIDE)
        return self.film + self.per_cell / self.cell**2 * L * w * t * eq

    def grain_textures(self):
        # zone [100] (Pnma) along the beam, the long axis [010] (Pnma) at the grain's angle
        return (), angle_overrides(MaterialId.MOLYBDENUM_TRIOXIDE, (1, 0, 0), (0, 1, 0), 0.3)

    def laths(self, seed, i, j):
        r = cell_rng(seed, 0x625, i, j)
        for _ in range(r.poisson(self.per_cell)):
            cx, cy = (i + r.uniform()) * self.cell, (j + r.uniform()) * self.cell
            ang = r.uniform(0.0, math.pi)
            L = min(r.lognormal(self.length, self.length_sigma), 2.0 * self.max_half)
            W = max(L / r.uniform(*self.aspect), 0.06)
            t = min(max(r.lognormal(self.thick, self.thick_sigma), 12.0), 150.0)
            steps = (r.uniform(-0.3, 0.3), r.uniform(0.0, 1.0), r.uniform(0.1, 0.25))  # a terrace
            # the ends are cut by crystal faces at an angle (a facetted, sword-like tip)
            yield (
                cx,
                cy,
                ang,
                L,
                W,
                t,
                steps,
                (r.uniform(0.3, 1.2), r.uniform(-0.6, 0.6), r.uniform(0.3, 1.2), r.uniform(-0.6, 0.6)),
            )

    def fill(self, context, owner):
        if not _resolved(0.05, context):
            return
        seed = structure_seed(owner)
        mean = self.mean_nm()
        eq = carbon_equivalent(MaterialId.MOLYBDENUM_TRIOXIDE)
        for B in _owner_pixels(context, owner):
            context.add_thickness(B.flat, self.film - mean)
            lx, ly = B.lx, B.ly
            for i, j in cells_near(lx, ly, self.cell, self.max_half):
                for cx, cy, ang, L, W, t, (sx, sp, sh), (k1, o1, k2, o2) in self.laths(seed, i, j):
                    c, s = math.cos(ang), math.sin(ang)
                    u = (lx - cx) * c + (ly - cy) * s
                    v = -(lx - cx) * s + (ly - cy) * c
                    # straight sides; each end cut by two faces: u < L/2 - k |v - o W/2|
                    vv = v / (0.5 * W)
                    inside = np.abs(v) < 0.5 * W
                    inside &= u < 0.5 * L - k1 * 0.5 * W * np.abs(vv - o1)
                    inside &= -u < 0.5 * L - k2 * 0.5 * W * np.abs(vv - o2)
                    if not inside.any():
                        continue
                    # one terrace: a step across the lath at `sx` of its length
                    tt = t * (1.0 + np.where(u[inside] / L > sx, sh if sp > 0.5 else -sh, 0.0))
                    f = B.sub(inside)
                    context.add_thickness(f, tt - self.film + self.film / eq)
                    g = angle_grain(MaterialId.MOLYBDENUM_TRIOXIDE, ang + owner.rot)
                    _claim(context, f, np.full(f.shape, g, np.int32), MaterialId.MOLYBDENUM_TRIOXIDE)
