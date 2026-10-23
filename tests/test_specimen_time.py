import numpy as np
import pytest

from de_twin.specimen import (
    MaterialId,
    Specimen,
    SpecimenConfig,
    ViewWindow,
    advance_thermal_budget,
    from_name,
)
from de_twin.specimen.materials import GRAINS_PER_MATERIAL
from de_twin.state import HolderState

GOLD_GRAINS = np.arange(MaterialId.GOLD * GRAINS_PER_MATERIAL, (MaterialId.GOLD + 1) * GRAINS_PER_MATERIAL)


def heat(t):
    return HolderState(kind="heating", temperature_c=t)


def test_thermal_budget_rules():
    assert advance_thermal_budget(10.0, 300.0, 400.0, 800.0, 5.0) == 10.0  # below onset: frozen
    assert advance_thermal_budget(10.0, 500.0, 400.0, 800.0, 5.0) == pytest.approx(15.0)  # 1 s/s at +100 C
    assert advance_thermal_budget(10.0, 600.0, 400.0, 800.0, 5.0) == pytest.approx(20.0)
    assert advance_thermal_budget(10.0, 900.0, 400.0, 800.0, 0.1) == pytest.approx(6.0)  # melting drains
    assert advance_thermal_budget(1.0, 900.0, 400.0, 800.0, 1.0) == 0.0


def test_crystallinity_monotonic_in_budget():
    s = Specimen(from_name("Droplet crystallization"))
    assert s.crystallization_enabled
    s.update(0.0, heat(25))
    prev = s.crystallinity(GOLD_GRAINS)
    assert np.all(prev == 0.0)  # as deposited: glassy
    means = []
    for k in range(1, 20):
        s.update(10.0 * k, heat(460))  # 0.6 s of budget per second
        c = s.crystallinity(GOLD_GRAINS)
        assert np.all(c >= prev) and np.all((c >= 0) & (c <= 1))
        prev = c
        means.append(c.mean())
    assert means[0] < 0.05 and means[-1] == pytest.approx(1.0)
    assert 0.0 < means[6] < 1.0  # a partially crystalline field in between
    # log-uniform nucleation: more grains nucleate early in the window than late
    b = s.thermal_budget_s
    assert b > 75


def test_melting_reamorphises_and_quench_holds():
    s = Specimen(from_name("Droplet crystallization"))
    s.update(0.0, heat(25))
    s.update(50.0, heat(600))  # budget 100 s: past the whole nucleation window + growth
    assert s.crystallinity(GOLD_GRAINS).mean() == pytest.approx(1.0)
    s.update(51.5, heat(900))  # 1.5 s molten drains 60 s of budget
    c_melt = s.crystallinity(GOLD_GRAINS).mean()
    assert c_melt < 1.0
    s.update(500.0, heat(25))  # quench: frozen
    assert s.crystallinity(GOLD_GRAINS).mean() == pytest.approx(c_melt)


def test_crystallinity_deterministic_and_off_elsewhere():
    hist = [(0.0, 25), (30.0, 480), (60.0, 520), (61.0, 850), (90.0, 450)]
    out = []
    for _ in range(2):
        s = Specimen(from_name("Droplet crystallization"))
        for t, T in hist:
            s.update(t, heat(T))
        out.append(s.crystallinity(GOLD_GRAINS))
    assert np.array_equal(out[0], out[1])
    grid = Specimen(from_name("Dense Au on holey C"))
    grid.update(0.0, heat(25))
    grid.update(100.0, heat(600))
    assert np.all(grid.crystallinity(GOLD_GRAINS) == 1.0)  # model off off the chip
    assert np.all(grid.crystallinity(np.array([-1])) == 1.0)
    forced = Specimen(SpecimenConfig(options={"in_situ_crystallization": True}))
    assert forced.crystallization_enabled and forced.crystallinity(GOLD_GRAINS).max() == 0.0


def test_each_droplet_crystallises_on_heating():
    """Every droplet is its own grain; heating (not wall time) takes it from glassy to crystalline."""
    s = Specimen(from_name("Droplet crystallization"))
    area = s.scene.holder.areas[4]
    v = ViewWindow(center_um=area.center, pixel_um=0.01, shape=(512, 512))  # populated, not aggregated
    s.update(0.0, heat(25))
    fm = s.rasterize(v)
    au = fm.material_id == MaterialId.GOLD
    assert au.sum() > 100
    assert np.all(fm.grain_id[au] >= 0)  # droplets carry grains (orientations) from the start
    gids = np.unique(fm.grain_id[au])
    assert len(gids) > 1
    assert np.all(s.crystallinity(gids) == 0.0)  # as deposited: glassy
    s.update(40.0, heat(25))  # time alone does nothing
    assert np.all(s.crystallinity(gids) == 0.0)
    s.update(90.0, heat(650))  # 50 s at 650 C: budget 125 s, past the nucleation window
    assert np.all(s.crystallinity(gids) == 1.0)
    assert s.crystallinity(np.array([-1]))[0] == 0.0  # grain-less gold here is glassy, not powder


def test_drift_moves_the_whole_field():
    base = from_name("Droplet crystallization")
    a = Specimen(base)
    b = Specimen(base.with_options(drift_per_step_x_nm=200.0))
    area = a.scene.holder.areas[4]
    v = ViewWindow(center_um=area.center, pixel_um=0.01, shape=(512, 512))
    for s in (a, b):
        s.update(30.0)
    assert b.drift_um() == pytest.approx((6.0, 0.0))
    ta, tb = a.rasterize(v).thickness_nm, b.rasterize(v).thickness_nm
    assert not np.array_equal(ta, tb)
    # b's field is a's field seen 6 um to the left
    shifted = ViewWindow(center_um=(area.center[0] - 6.0, area.center[1]), pixel_um=0.01, shape=(512, 512))
    assert np.array_equal(a.rasterize(shifted).thickness_nm, tb)


def test_time_step_option_scales_the_index():
    s = Specimen(from_name("Droplet crystallization").with_options(time_step_s=0.5))
    s.update(3.0)
    assert s.time_index == 6
