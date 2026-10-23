import numpy as np
import pytest

from de_twin.specimen import MaterialId, Specimen, SpecimenConfig, ViewWindow
from de_twin.specimen.raster import Shape


def _mesh(lod=0.5, **options):
    return Specimen(
        SpecimenConfig(
            seed=5,
            holder="mesh_grid",
            preparation="nanoparticles",
            film="continuous",
            options=dict(lod_threshold_px=lod, **options),
        )
    )


def _central_area(s):
    h = s.scene.holder
    d = h.area_cx**2 + h.area_cy**2
    return h.areas[int(np.argmin(np.where(h.area_film_ok, d, np.inf)))]


def test_population_mass_matches_aggregate():
    s = _mesh()
    area = _central_area(s)
    prep = s.scene.preparation
    seed = s.scene.area_seed(area.index)
    batch = prep.populate(s.scene.holder, area, seed)
    d = batch.prims.thickness.astype(np.float64)
    footprint = 4.0 * area.half[0] * area.half[1]
    populated_nm = np.sum(np.pi / 6.0 * d**3) * 1e-6 / footprint  # nm^3 per um^2 -> nm
    agg = prep.aggregate_for(s.scene.holder, area, seed)
    assert agg.shape[0] == Shape.RECT and bool(agg.aggregate[0])
    assert populated_nm == pytest.approx(float(agg.thickness[0]), rel=0.15)


def test_low_mag_aggregate_conserves_mean_thickness():
    # lod 0: nothing culled, so the populated (dust) raster conserves the particle mass exactly
    s = _mesh(lod=0.0)
    area = _central_area(s)
    cx, cy = area.center
    fine = s.rasterize(ViewWindow(center_um=(cx, cy), pixel_um=0.09, shape=(1024, 1024)))
    assert s.last_stats["areas_populated"] + 0 >= 0 and s.last_stats["dust"] > 10000
    coarse = s.rasterize(ViewWindow(center_um=(cx, cy), pixel_um=0.5, shape=(256, 256)))
    assert s.last_stats["areas_aggregated"] >= 1

    def central_mean(fm, px, n):
        c = (np.arange(n) - n / 2 + 0.5) * px
        X, Y = np.meshgrid(c, c)
        m = (np.abs(X) < 30) & (np.abs(Y) < 30) & (fm.material_id != MaterialId.COPPER)
        return float(fm.thickness_nm[m].mean())

    a = central_mean(fine, 0.09, 1024)
    b = central_mean(coarse, 0.5, 256)
    film = 12.0
    assert (a - film) == pytest.approx(b - film, rel=0.15)
    assert b - film > 1.0  # the aggregate really carries particle mass


def test_whole_grid_uses_aggregation():
    s = _mesh()
    s.rasterize(ViewWindow(center_um=(0.0, 0.0), pixel_um=3.2, shape=(1024, 1024)))
    st = s.last_stats
    assert st["areas_populated"] == 0 and st["areas_aggregated"] > 300
    assert st["primitives"] < 1000  # one aggregate per hole, no per-particle work


def test_culling_and_dust_rules():
    s = _mesh()
    area = _central_area(s)
    s.rasterize(ViewWindow(center_um=area.center, pixel_um=0.09, shape=(1024, 1024)))
    st = s.last_stats
    assert st["culled"] > 0 and st["dust"] > 0  # sub-0.5 px culled, 0.5-2 px dust
    s.rasterize(ViewWindow(center_um=area.center, pixel_um=0.001, shape=(512, 512)))
    assert s.last_stats["drawn"] > 0 and s.last_stats["dust"] == 0


def test_resident_cache_is_bounded():
    s = _mesh(particle_density_per_um2=1.0)
    for area in s.scene.holder.areas[:72]:
        s.rasterize(ViewWindow(center_um=area.center, pixel_um=0.4, shape=(256, 256)))
        assert s.last_stats["areas_populated"] <= 4
    assert len(s.scene._resident) <= 64
