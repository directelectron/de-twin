"""Performance smoke tests (generous limits; the targets are ~0.25 s on a desktop)."""

import time

from perf_budget import budget

from de_twin.specimen import Specimen, ViewWindow, from_name


def _time(fn, repeat=2):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def test_populated_grid_square_1024_is_fast():
    s = Specimen(from_name("Dense Au on holey C"))
    f = s.features()[0]
    v = ViewWindow(center_um=f.center_um, pixel_um=0.001, shape=(1024, 1024))
    t0 = time.perf_counter()
    s.rasterize(v)  # includes populating the hole (~180k particles)
    first = time.perf_counter() - t0
    steady = _time(lambda: s.rasterize(v))
    print(f"populate+raster {first:.3f}s, steady {steady:.3f}s")
    assert first < budget(1.5)
    assert steady < budget(0.75)


def test_low_mag_whole_grid_is_fast():
    s = Specimen(from_name("Dense Au on holey C"))
    v = ViewWindow(center_um=(0.0, 0.0), pixel_um=3.2, shape=(1024, 1024))
    t = _time(lambda: s.rasterize(v))
    print(f"whole grid {t:.3f}s")
    assert s.last_stats["areas_populated"] == 0
    assert t < budget(1.0)


def test_specimen_construction_is_cheap():
    t = _time(lambda: Specimen(from_name("Dense Au on holey C")))
    assert t < budget(0.5)
