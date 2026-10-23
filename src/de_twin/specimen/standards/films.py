"""Film standards: evaporated polycrystalline aluminium (619, camera length)."""

from __future__ import annotations

import math

import numpy as np

from .. import raster_nb as _rnb
from ..fieldmap import GRAINS_PER_MATERIAL, grain_id_from_hash
from ..geometry import NM_PER_UM
from ..materials import MaterialId
from ..noise import cells, fbm
from ..structures import _claim, _owner_pixels, _resolved
from .common import StandardStructure, structure_seed
from .replica import carbon_equivalent


class PolycrystalFilmStructure(StandardStructure):
    """An evaporated polycrystalline film (619: aluminium, ~31 nm): equiaxed grains `grain_nm`
    across, randomly oriented (ring patterns), shallow grooves at the grain boundaries."""

    def __init__(
        self,
        material: int = MaterialId.ALUMINUM,
        thickness_nm: float = 31.0,
        grain_nm: float = 15.0,
        groove_nm: float = 3.0,
        groove_width_nm: float = 1.0,
    ):
        self.base_material = int(material)
        self.t, self.grain_nm = float(thickness_nm), float(grain_nm)
        self.groove, self.groove_w = float(groove_nm), float(groove_width_nm)

    def base_nm(self) -> float:
        return self.t

    def mean_nm(self) -> float:
        frac = min(1.0, 3.4 * self.groove_w / self.grain_nm)  # boundary area fraction (~3.4 w / d)
        return (self.t - 0.5 * self.groove * frac) * carbon_equivalent(self.base_material)

    def numba_fill(self) -> bool:
        return _rnb.AVAILABLE

    def fill(self, ctx, owner):
        if not _resolved(self.grain_nm / NM_PER_UM, ctx):
            return
        seed = structure_seed(owner)
        L = self.grain_nm / NM_PER_UM
        if _rnb.AVAILABLE:
            b = owner.bounds
            win = ctx.window(b.xmin, b.ymin, b.xmax, b.ymax)
            if win is None:
                return
            from ..noise import fbm_params

            p1, p2 = fbm_params(seed ^ 0xC1, L, 1), fbm_params(seed ^ 0xC2, L, 1)
            W = np.array([ctx.ox, ctx.oy, ctx.axc, ctx.ayc, ctx.axr, ctx.ayr])
            O = np.array([owner.cx, owner.cy, math.cos(-owner.rot), math.sin(-owner.rot), owner.rx, owner.ry])
            cseed = np.uint64((int(seed) * 0xC2B2AE3D27D4EB4F) & 0xFFFFFFFFFFFFFFFF)
            _rnb.polycrystal_film(
                ctx.thick,
                ctx.material,
                ctx.grain,
                ctx.nx,
                win[0],
                win[1],
                win[2],
                win[3],
                W,
                O,
                *p1,
                *p2,
                L,
                0.2 * L,
                cseed,
                self.groove,
                self.grain_nm,
                self.groove_w,
                self.base_material,
                GRAINS_PER_MATERIAL,
            )
            ctx.stats["fast_fills"] = ctx.stats.get("fast_fills", 0) + 1
            return
        for B in _owner_pixels(ctx, owner):
            wx = B.lx + 0.2 * L * fbm(seed ^ 0xC1, B.lx, B.ly, L, 1)
            wy = B.ly + 0.2 * L * fbm(seed ^ 0xC2, B.lx, B.ly, L, 1)
            _, gap, h = cells(seed, wx / L, wy / L, 0.0)
            groove = self.groove * np.exp(-0.5 * (gap * self.grain_nm / self.groove_w) ** 2)
            ctx.add_thickness(B.flat, -groove)
            _claim(ctx, B.flat, grain_id_from_hash(self.base_material, h), self.base_material)
