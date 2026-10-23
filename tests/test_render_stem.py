"""STEM: parked / 4D CBED, scan addressing, descan, blend, virtual images, perf."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from perf_budget import budget

from de_twin.optics import OpticsConfig, derive_optics
from de_twin.render import RenderConfig, Renderer
from de_twin.render.testing import Particle, StubCamera, SyntheticSpecimen
from de_twin.specimen.materials import MaterialId
from de_twin.state import AcquisitionRequest, MicroscopeState, RenderMode, Roi, TemStem

CAM = StubCamera()  # DE16, 256^2 ROI at the sensor centre
NSCAN = 32
STEP_UM = 0.002


def _particle_at(o, ix, iy, radius_nm=4.0, **kw):
    wx, wy = o.view.pixel_to_world(iy, ix)
    return Particle(float(wx), float(wy), radius_nm, **kw)


def _spec_with_particle(o, ix=24, iy=9, **kw):
    return SyntheticSpecimen(film_material=MaterialId.VACUUM, particles=[_particle_at(o, ix, iy, **kw)])


def test_geometry(make_stem_optics):
    o = make_stem_optics()
    assert o.render_mode == RenderMode.STEM_4D
    assert o.view.shape == (NSCAN, NSCAN) and o.output_shape == (256, 256)
    assert o.disk_radius_px == pytest.approx(10.0 / (6.5 / 13.0), rel=1e-6)


def test_vacuum_cbed_is_the_probe_current(make_stem_optics):
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM)
    o = make_stem_optics()
    r = Renderer(spec)
    img = r.render(o, scan_point=(3, 4))
    assert img.shape == (256, 256) and img.dtype == np.float32 and img.min() >= 0
    ratio = img.sum() / o.pattern_e_per_s
    assert 1 / 1.1 - 1e-3 <= ratio <= 1 / 0.9 + 1e-3  # intensity jitter 0.9 .. 1.1
    r2 = Renderer(spec, RenderConfig(intensity_jitter=False))
    assert r2.render(o, scan_point=(3, 4)).sum() == pytest.approx(o.pattern_e_per_s, rel=1e-4)
    # a centred disk of the convergence angle
    y, x = np.indices(img.shape)
    rr = np.hypot(x - 128, y - 128)
    assert img[rr > o.disk_radius_px + 12].sum() < 1e-5 * img.sum()


def test_scan_point_addressing_and_frame_index(make_stem_optics):
    o = make_stem_optics()
    spec = _spec_with_particle(o, 24, 9, radius_nm=4.0)
    r = Renderer(spec, RenderConfig(probe_footprint_blend=False, intensity_jitter=False))
    on = r.render(o, scan_point=(24, 9))
    off = r.render(o, scan_point=(9, 24))
    y, x = np.indices(on.shape)
    outside = np.hypot(x - 128, y - 128) > o.disk_radius_px + 12
    assert on[outside].sum() > 0.05 * on.sum()  # Bragg / halo / diffuse scattering
    assert off[outside].sum() < 1e-5 * off.sum()
    assert on.sum() < off.sum()  # mass-thickness attenuation
    # frame_index runs the raster row by row
    same = r.render(o, frame_index=9 * NSCAN + 24)
    assert np.array_equal(same, on)


def test_parked_probe_uses_park_position(make_stem_optics):
    o = make_stem_optics(scan=False)
    assert o.render_mode == RenderMode.STEM_PARKED
    spec = _spec_with_particle(o, NSCAN // 2, NSCAN // 2, radius_nm=4.0)
    r = Renderer(spec, RenderConfig(probe_footprint_blend=False))
    centre = r.render(o)
    y, x = np.indices(centre.shape)
    outside = np.hypot(x - 128, y - 128) > o.disk_radius_px + 12
    assert centre[outside].sum() > 0.05 * centre.sum()


def test_descan_ramp_and_layer_shift_the_pattern(make_stem_optics):
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM)
    o = make_stem_optics()
    r0 = Renderer(spec, RenderConfig(intensity_jitter=False))
    r1 = Renderer(spec, RenderConfig(intensity_jitter=False, add_descan=True))
    y, x = np.indices((256, 256))
    for ix, iy in ((0, 0), (31, 31), (16, 8)):
        a = r0.render(o, scan_point=(ix, iy))
        b = r1.render(o, scan_point=(ix, iy))
        nx = 2.0 * (ix - NSCAN / 2) / NSCAN
        ny = 2.0 * (iy - NSCAN / 2) / NSCAN
        ex = nx * 6 + 6 + iy * 4 / NSCAN
        ey = ny * 4 + 4
        dx = (b * x).sum() / b.sum() - (a * x).sum() / a.sum()
        dy = (b * y).sum() / b.sum() - (a * y).sum() / a.sum()
        assert (dx, dy) == pytest.approx((ex, ey), abs=0.05)


class _LayeredSpecimen(SyntheticSpecimen):
    def rasterize(self, view, layers=frozenset()):
        fm = super().rasterize(view, layers)
        if fm.descan is not None:
            fm.descan[0] = 3.0
            fm.descan[1] = -2.0
        if fm.strain is not None:
            fm.strain[0, :, :16] = 1.05  # xx strain on the left half
        return fm


def test_descan_and_strain_layers(make_stem_optics):
    spec = _LayeredSpecimen(film_material=MaterialId.VACUUM)
    o = make_stem_optics()
    r = Renderer(spec, RenderConfig(intensity_jitter=False))
    y, x = np.indices((256, 256))
    right = r.render(o, scan_point=(24, 5))  # descan only
    cx = (right * x).sum() / right.sum()
    cy = (right * y).sum() / right.sum()
    assert (cx, cy) == pytest.approx((128 + 3.0, 128 - 2.0), abs=0.05)
    left = r.render(o, scan_point=(4, 5))  # descan + 5 % xx strain: the disk widens in x
    wx = lambda a, c: math.sqrt((a * (x - c) ** 2).sum() / a.sum())
    assert wx(left, (left * x).sum() / left.sum()) == pytest.approx(1.05 * wx(right, cx), rel=0.02)


def test_probe_blend_mixes_neighbours(make_stem_optics):
    # a wide probe (d0 = 8 nm on a 2 nm step -> +-2 step arms) sees the particle from next door
    ocfg = OpticsConfig(probe_d0_nm=8.0)
    o = make_stem_optics(config=ocfg)
    assert o.extras["probe_blend_offset_px"] == pytest.approx(2.0)
    spec = _spec_with_particle(o, 16, 16, radius_nm=1.5)
    y, x = np.indices((256, 256))
    outside = np.hypot(x - 128, y - 128) > o.disk_radius_px + 12
    blended = Renderer(spec, RenderConfig(intensity_jitter=False)).render(o, scan_point=(18, 16))
    single = Renderer(spec, RenderConfig(intensity_jitter=False, probe_footprint_blend=False)
                      ).render(o, scan_point=(18, 16))
    assert blended[outside].sum() > 10 * max(single[outside].sum(), 1e-30)


def test_thickness_and_tilt_attenuation(make_stem_optics):
    o = make_stem_optics()
    thin = SyntheticSpecimen(film_material=MaterialId.VACUUM, particles=[_particle_at(o, 10, 10, 3.0)])
    thick = SyntheticSpecimen(film_material=MaterialId.VACUUM, particles=[_particle_at(o, 10, 10, 12.0)])
    config = RenderConfig(intensity_jitter=False, probe_footprint_blend=False, diffuse_scattering=False)
    a = Renderer(thin, config).render(o, scan_point=(10, 10)).sum()
    b = Renderer(thick, config).render(o, scan_point=(10, 10)).sum()
    assert b < a < o.pattern_e_per_s
    s = MicroscopeState()
    s.tem_stem = TemStem.STEM
    s.stage.alpha_deg = 60.0
    s.camera_length_mm = 13.0
    rq = AcquisitionRequest(hw_roi=Roi(1920, 1920, 256, 256))
    rq.scan.enabled = True
    rq.scan.size = (NSCAN, NSCAN)
    rq.scan.step_um = STEP_UM
    ot = derive_optics(s, rq, CAM)
    film = SyntheticSpecimen(film_thickness_nm=30.0)
    flat = Renderer(film, config).render(o, scan_point=(5, 5)).sum()
    tilt = Renderer(film, config).render(ot, scan_point=(5, 5)).sum()
    assert tilt / flat == pytest.approx(math.exp(-30.0 / 150.0), rel=1e-3)  # 2x path, e^-t/L once more


def test_virtual_bright_field_and_haadf(make_stem_optics):
    o = make_stem_optics()
    spec = _spec_with_particle(o, 24, 9, radius_nm=5.0)
    r = Renderer(spec, RenderConfig(intensity_jitter=False))
    bf = r.virtual_image(o, 0.0, o.convergence_mrad)
    adf = r.virtual_image(o, 40.0, 200.0)
    assert bf.shape == (NSCAN, NSCAN) and adf.shape == (NSCAN, NSCAN)
    assert bf[9, 24] < 0.8 * bf[24, 9]
    assert bf[24, 9] == pytest.approx(o.pattern_e_per_s, rel=1e-3)  # vacuum: the whole probe
    assert adf[9, 24] > 20 * max(adf[24, 9], 1e-9 * o.pattern_e_per_s)
    assert np.unravel_index(np.argmin(bf), bf.shape) == (9, 24)


def test_virtual_image_agrees_with_cbed(make_stem_optics):
    o = make_stem_optics()
    spec = _spec_with_particle(o, 16, 16, radius_nm=6.0)
    r = Renderer(spec)
    edge = o.convergence_mrad + 1.5  # include the blurred disk edge
    vbf = r.virtual_image(o, 0.0, edge)
    y, x = np.indices((256, 256))
    rr = np.hypot(x - 128, y - 128) * o.extras["mrad_per_px"]
    for ix, iy in ((16, 16), (14, 16), (2, 2)):
        cbed = r.render(o, scan_point=(ix, iy))
        assert cbed[rr < edge].sum() == pytest.approx(vbf[iy, ix], rel=0.05)


def test_blanked_stem(make_stem_optics):
    o = make_stem_optics(beam_blanked=True)
    r = Renderer(SyntheticSpecimen())
    assert not r.render(o, scan_point=(1, 1)).any()
    assert not r.virtual_image(o, 0, 10).any()


def test_cache_is_used_and_warm_up(make_stem_optics):
    o = make_stem_optics()
    rng = np.random.default_rng(1)
    parts = [_particle_at(o, int(i), int(j), 3.0, grain=int(g))
             for i, j, g in zip(rng.integers(0, NSCAN, 20), rng.integers(0, NSCAN, 20), range(20))]
    spec = SyntheticSpecimen(particles=parts)
    r = Renderer(spec)
    n = r.warm_up(o, budget_s=5.0)
    assert n > 5
    misses = r.cache.stats.misses
    for iy in range(0, NSCAN, 4):
        for ix in range(0, NSCAN, 4):
            r.render(o, scan_point=(ix, iy))
    assert r.cache.stats.misses - misses < 10
    assert spec.rasterize_calls == 1


@pytest.mark.slow
def test_perf_cbed_from_cache(make_stem_optics):
    o = make_stem_optics()
    spec = SyntheticSpecimen(particles=[_particle_at(o, 16, 16, 5.0)])
    r = Renderer(spec)
    r.render(o, scan_point=(16, 16))
    r.render(o, scan_point=(17, 16))
    t0 = time.perf_counter()
    n = 0
    for ix in range(NSCAN):
        r.render(o, scan_point=(ix, 16))
        n += 1
    per = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    r.virtual_image(o, 40, 200)
    vi = time.perf_counter() - t0
    print(f"CBED 256^2 from cache: {per * 1000:.2f} ms/pattern; virtual image {vi * 1000:.1f} ms")
    assert per < budget(0.01)
