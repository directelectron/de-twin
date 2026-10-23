import numpy as np
from perf_budget import budget

from de_twin.clock import Clock, ManualClock
from de_twin.hashing import SeedKind, hash_seed, mix_cell, rng_for, splitmix64, uniform_from_hash
from de_twin.specimen.fieldmap import GrainTable, ViewWindow
from de_twin.specimen.materials import MATERIALS, MaterialId
from de_twin.state import AcquisitionRequest, ExposureMode


def test_splitmix64_reference_value():
    # First output of the reference splitmix64.c generator seeded with 0.
    assert splitmix64(0) == 0xE220A8397B1DCDAF


def test_vectorised_hash_matches_scalar():
    idx = np.arange(-50, 50, dtype=np.int64)
    vec = hash_seed(np.uint64(42), SeedKind.GRAIN, idx, 7)
    for i, v in zip(idx, vec):
        assert int(v) == hash_seed(42, SeedKind.GRAIN, int(i) & ((1 << 64) - 1), 7)
    cells = mix_cell(idx, idx[::-1])
    for i, j, v in zip(idx, idx[::-1], cells):
        assert int(v) == mix_cell(int(i), int(j))


def test_uniform_range_and_rng_determinism():
    u = uniform_from_hash(hash_seed(np.uint64(1), 1, np.arange(10000), 0))
    assert u.min() >= 0 and u.max() < 1 and abs(u.mean() - 0.5) < 0.02
    assert rng_for(3, 4, 5).random() == rng_for(3, 4, 5).random()
    assert rng_for(3, 4, 5).random() != rng_for(3, 4, 6).random()


def test_view_window_roundtrip_with_tilt_and_rotation():
    v = ViewWindow(center_um=(10.0, -5.0), pixel_um=0.01, shape=(64, 128), rotation_rad=0.3,
                   flip_x=True, cos_alpha=0.8)
    rows, cols = np.mgrid[0:64, 0:128]
    x, y = v.pixel_to_world(rows, cols)
    r2, c2 = v.world_to_pixel(x, y)
    assert np.allclose(r2, rows) and np.allclose(c2, cols)
    # centre of the raster maps to the view centre
    cx, cy = v.pixel_to_world(31.5, 63.5)
    assert np.isclose(cx, 10.0) and np.isclose(cy, -5.0)


def test_grain_table_deterministic():
    a, b = GrainTable.generate(5), GrainTable.generate(5)
    assert np.array_equal(a.quaternions, b.quaternions)
    assert np.array_equal(a.nucleation_u, b.nucleation_u)
    c = GrainTable.generate(6)
    assert not np.array_equal(a.quaternions, c.quaternions)
    # unit quaternions, and proper rotation matrices
    assert np.allclose(np.linalg.norm(a.quaternions, axis=1), 1.0)
    assert np.allclose(np.linalg.det(a.matrices), 1.0)


def test_materials_absorption_increases_with_voltage():
    gold = MATERIALS[MaterialId.GOLD]
    assert gold.absorption_length_nm(300) > gold.absorption_length_nm(200) > gold.absorption_length_nm(80)
    assert np.isclose(gold.absorption_length_nm(200), 25.0)
    assert gold.crystalline and not gold.amorphous
    assert MATERIALS[MaterialId.AMORPHOUS_CARBON].amorphous
    assert not MATERIALS[MaterialId.VACUUM].amorphous and not MATERIALS[MaterialId.VACUUM].crystalline


def test_exposure_mode_blocks_beam():
    assert not AcquisitionRequest(exposure_mode=ExposureMode.DARK).beam_reaches_detector
    assert not AcquisitionRequest(beam_blank_cmd=0).beam_reaches_detector
    assert AcquisitionRequest().beam_reaches_detector


def test_manual_clock_sleep_advances():
    c = ManualClock()
    c.sleep(2.5)
    assert c.now() == 2.5
    c.sleep_until(10)
    assert c.now() == 10


def test_scaled_clock_runs_fast():
    import time

    c = Clock(time_scale=1000)
    t0, w0 = c.now(), time.perf_counter()
    c.sleep(1.0)  # 1 ms of wall time
    assert c.now() - t0 >= 1.0  # at least the requested simulated time passed
    assert time.perf_counter() - w0 < budget(0.5)  # ...in a fraction of the wall time
