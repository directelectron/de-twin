"""TEM imaging: flux, sign conventions, CTF physics (Thon rings, astigmatism, beam tilt), caching."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from de_twin.render import RenderConfig, Renderer
from de_twin.render.testing import (Particle, StubCamera, SyntheticSpecimen, fit_defocus, on_zone_grains,
                                    phase_correlation_shift)
from de_twin.specimen.materials import MaterialId
from de_twin.state import Vec2
from perf_budget import budget

CAM1K = StubCamera(sensor_shape=(1024, 1024))
CAM512 = StubCamera(sensor_shape=(512, 512))


def _centroid_dark(img):
    w = np.clip(np.median(img) - img, 0, None)
    ys, xs = np.mgrid[:img.shape[0], :img.shape[1]]
    return float((w * xs).sum() / w.sum()), float((w * ys).sum() / w.sum())


# ---------------------------------------------------------------- basics
def test_vacuum_is_uniform_dose(make_optics):
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM)
    o = make_optics(magnification=20000, defocus_um=-1.0)
    img = Renderer(spec).render(o)
    assert img.shape == o.output_shape and img.dtype == np.float32
    assert np.allclose(img, o.dose_e_per_px_s, rtol=1e-4)


def test_blanked_gives_zeros(make_optics):
    o = make_optics(beam_blanked=True)
    img = Renderer(SyntheticSpecimen()).render(o)
    assert img.shape == o.output_shape and not img.any()


def test_intensity_non_negative_and_contrast(make_optics):
    spec = SyntheticSpecimen(particles=[Particle(0.0, 0.0, 20.0)], holes=[(0.15, 0.0, 0.05)])
    o = make_optics(magnification=20000, defocus_um=-2.0)
    img = Renderer(spec).render(o)
    assert img.min() >= 0.0
    cx, cy = img.shape[1] // 2, img.shape[0] // 2
    assert img[cy, cx] < 0.5 * o.dose_e_per_px_s  # 40 nm of gold is dark


def test_stage_moves_image_content_with_stage(make_optics):
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM, particles=[Particle(0.0, 0.0, 15.0)])
    r = Renderer(spec)
    o0 = make_optics(magnification=20000)
    x0, y0 = _centroid_dark(r.render(o0))
    o1 = make_optics(magnification=20000, stage_x=0.02)
    x1, y1 = _centroid_dark(r.render(o1))
    expected = 20.0 / o0.specimen_pixel_nm
    assert x1 - x0 == pytest.approx(expected, abs=1.0)
    assert y1 == pytest.approx(y0, abs=0.5)
    o2 = make_optics(magnification=20000, image_shift_um=Vec2(0.02, 0.0))
    x2, _ = _centroid_dark(r.render(o2))
    assert x2 - x0 == pytest.approx(-expected, abs=1.0)


def test_magnification_scales_features(make_optics):
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM, particles=[Particle(0.0, 0.0, 30.0)])
    r = Renderer(spec)
    a = r.render(make_optics(magnification=20000))
    b = r.render(make_optics(magnification=40000))
    area_a = (a < 0.5 * a.max()).sum()
    area_b = (b < 0.5 * b.max()).sum()
    assert area_b / area_a == pytest.approx(4.0, rel=0.1)


def test_dose_scales_with_spot_and_mag(make_optics):
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM)
    r = Renderer(spec)
    a = r.render(make_optics(magnification=20000, spot_size=3)).mean()
    b = r.render(make_optics(magnification=20000, spot_size=1)).mean()
    c = r.render(make_optics(magnification=40000, spot_size=3)).mean()
    assert b / a == pytest.approx(9.0, rel=1e-3)
    assert a / c == pytest.approx(4.0, rel=1e-3)


def test_illuminated_disk_edge_is_visible_when_focused(make_optics):
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM)
    o = make_optics(camera=CAM512, magnification=2000, intensity=0.0)  # D = 0.1 um, FOV = 1.66 um
    img = Renderer(spec).render(o)
    assert img[256, 256] == pytest.approx(o.dose_e_per_px_s, rel=1e-3)
    assert img[0, 0] < 1e-3 * o.dose_e_per_px_s
    radius_px = o.illuminated_diameter_um * 1000 / 2 / o.specimen_pixel_nm
    assert ((img > 0.5 * img.max()).sum()) == pytest.approx(math.pi * radius_px ** 2, rel=0.05)


def test_zero_thickness_film_is_vacuum(make_optics):
    spec = SyntheticSpecimen(film_thickness_nm=0.0)
    o = make_optics(magnification=50000, defocus_um=-1.0)
    img = Renderer(spec).render(o)
    assert np.allclose(img, o.dose_e_per_px_s, rtol=1e-4)


# ------------------------------------------------------------ CTF physics
@pytest.mark.parametrize("df_um", [-0.5, -1.0, -2.0])
def test_thon_rings_match_ctf_zeros(df_um, make_optics):
    o = make_optics(camera=CAM1K, magnification=50000, defocus_um=df_um)
    img = Renderer(SyntheticSpecimen()).render(o)
    grid = np.linspace(df_um * 1000 * 0.7, df_um * 1000 * 1.3, 301)
    df_fit, score = fit_defocus(img, o.specimen_pixel_nm, o.wavelength_nm, grid, cs_mm=o.cs_mm)
    assert score > 0.8
    assert df_fit == pytest.approx(df_um * 1000, rel=0.02)


def test_thon_ring_contrast_is_realistic(make_optics):
    o = make_optics(camera=CAM1K, magnification=50000, defocus_um=-1.0)
    img = Renderer(SyntheticSpecimen()).render(o)
    assert 0.03 < img.std() / img.mean() < 0.4


def test_astigmatism_gives_elliptical_rings(make_optics):
    a1 = 150.0
    o = make_optics(camera=CAM1K, magnification=50000, defocus_um=-1.0,
                objective_stig=Vec2(a1 / 1000.0, 0.0))
    assert o.astigmatism_nm == pytest.approx((a1, 0.0))
    img = Renderer(SyntheticSpecimen()).render(o)
    grid = np.linspace(-1500, -500, 501)
    dfx, _ = fit_defocus(img, o.specimen_pixel_nm, o.wavelength_nm, grid, sector_rad=0.0)
    dfy, _ = fit_defocus(img, o.specimen_pixel_nm, o.wavelength_nm, grid, sector_rad=math.pi / 2)
    assert dfx == pytest.approx(-1000 + a1, abs=20)
    assert dfy == pytest.approx(-1000 - a1, abs=20)
    # the 45-degree component rotates the ellipse
    o = make_optics(camera=CAM1K, magnification=50000, defocus_um=-1.0,
                objective_stig=Vec2(0.0, a1 / 1000.0))
    img = Renderer(SyntheticSpecimen()).render(o)
    d45, _ = fit_defocus(img, o.specimen_pixel_nm, o.wavelength_nm, grid, sector_rad=math.pi / 4)
    d135, _ = fit_defocus(img, o.specimen_pixel_nm, o.wavelength_nm, grid, sector_rad=3 * math.pi / 4)
    assert d45 == pytest.approx(-1000 + a1, abs=20)
    assert d135 == pytest.approx(-1000 - a1, abs=20)


@pytest.mark.parametrize("df_um", [-2.0, 2.0])
def test_beam_tilt_shift_is_defocus_times_tilt(df_um, make_optics):
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM,
                             particles=[Particle(0.0, 0.0, 15.0), Particle(0.03, -0.02, 10.0, grain=5)])
    r = Renderer(spec)
    a = r.render(make_optics(magnification=40000, defocus_um=df_um))
    o = make_optics(magnification=40000, defocus_um=df_um, beam_tilt_mrad=Vec2(2.0, 1.0))
    b = r.render(o)
    dx, dy = phase_correlation_shift(b, a)
    p = o.specimen_pixel_nm
    assert dx == pytest.approx(df_um * 1000 * 2e-3 / p, abs=1.0)
    assert dy == pytest.approx(df_um * 1000 * 1e-3 / p, abs=1.0)


def test_no_tilt_shift_at_focus(make_optics):
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM, particles=[Particle(0.0, 0.0, 15.0)])
    r = Renderer(spec)
    a = r.render(make_optics(magnification=40000, defocus_um=0.0))
    b = r.render(make_optics(magnification=40000, defocus_um=0.0, beam_tilt_mrad=Vec2(2.0, 0.0)))
    dx, dy = phase_correlation_shift(b, a)
    assert abs(dx) <= 1 and abs(dy) <= 1


def test_fresnel_fringe_at_hole_edge_flips_with_defocus(make_optics):
    """A hole in carbon: underfocus puts a bright Fresnel fringe just inside the hole
    edge, overfocus a dark one (the C++ fresnel_sign convention, here from the CTF)."""
    spec = SyntheticSpecimen(holes=[(0.0, 0.0, 0.03)], film_thickness_nm=30.0)
    config = RenderConfig(phase_texture_scale=0.0)
    rows = {}
    for df in (-3.0, 3.0):
        o = make_optics(magnification=30000, defocus_um=df)
        img = Renderer(spec, config).render(o) / o.dose_e_per_px_s
        c = img.shape[0] // 2
        r_px = int(30.0 / o.specimen_pixel_nm)
        rows[df] = img[c, c + r_px - 16:c + r_px]  # inside the hole, next to the edge
    assert rows[-3.0].max() > 1.1
    assert rows[3.0].min() < 0.9
    assert rows[-3.0].mean() > rows[3.0].mean()





def test_lattice_fringes_resolved_at_high_mag(make_optics):
    grains = on_zone_grains(MaterialId.GOLD * 512 + 1, (0, 1, 1))  # exactly on [011]
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM, particles=[Particle(0.0, 0.0, 6.0, grain=1)],
                             grains=grains)
    o = make_optics(camera=CAM512, magnification=400000, defocus_um=-0.05)
    img = Renderer(spec).render(o)
    d = 0.4078 / math.sqrt(3)  # {111}, the strongest [011] zone reflections
    c = img.shape[0] // 2
    crop = img[c - 64:c + 64, c - 64:c + 64]
    f = np.abs(np.fft.fftshift(np.fft.fft2(crop - crop.mean())))
    ky, kx = np.mgrid[-64:64, -64:64] / (128 * o.specimen_pixel_nm)
    k = np.hypot(kx, ky)
    near = (np.abs(k - 1 / d) < 0.4)
    far = (k > 1 / d + 1.0) & (k < 1 / d + 2.0)
    assert f[near].max() > 5 * np.median(f[far])


# ---------------------------------------------------------------- caching
def test_raster_and_frame_caches(make_optics):
    spec = SyntheticSpecimen()
    r = Renderer(spec)
    o = make_optics(magnification=50000, defocus_um=-1.0)
    a = r.render(o)
    b = r.render(o)
    assert spec.rasterize_calls == 1 and r.frames_from_cache == 1
    assert np.array_equal(a, b)
    r.render(make_optics(magnification=50000, defocus_um=-1.5))  # focus change: no re-raster
    assert spec.rasterize_calls == 1
    r.render(make_optics(magnification=50000, defocus_um=-1.5, stage_x=0.005))  # a nudge: a crop
    assert spec.rasterize_calls == 1
    r.render(make_optics(magnification=50000, stage_x=50.0))  # a jump beyond it: re-raster
    assert spec.rasterize_calls == 2
    spec.generation += 1  # specimen content changed
    r.render(make_optics(magnification=50000, stage_x=50.0))
    assert spec.rasterize_calls == 3
    r.invalidate()
    r.render(make_optics(magnification=50000, stage_x=50.0))
    assert spec.rasterize_calls == 4


def test_texture_is_world_locked(make_optics):
    """The amorphous texture moves with the specimen (integer-pixel stage steps)."""
    spec = SyntheticSpecimen()
    r = Renderer(spec)
    o0 = make_optics(magnification=50000, defocus_um=-1.0)
    p = o0.specimen_pixel_nm
    a = r.render(o0)
    b = r.render(make_optics(magnification=50000, defocus_um=-1.0, stage_x=10 * p / 1000.0))
    dx, dy = phase_correlation_shift(b, a)
    assert (dx, dy) == (10, 0)


@pytest.mark.slow
def test_perf_tem_4k(make_optics):
    spec = SyntheticSpecimen(particles=[Particle(0.0, 0.0, 12.0)])
    r = Renderer(spec)
    o = make_optics(camera=StubCamera(), magnification=30000, defocus_um=-1.0)
    r.field_map(o)  # rasterisation belongs to the specimen
    r.render(make_optics(camera=StubCamera(sensor_shape=(256, 256))))  # warm thread pool / imports
    t0 = time.perf_counter()
    img = r.render(o)
    first = time.perf_counter() - t0
    assert img.shape == (4096, 4096)
    t0 = time.perf_counter()
    r.render(make_optics(camera=StubCamera(), magnification=30000, defocus_um=-1.2))
    refocus = time.perf_counter() - t0
    t0 = time.perf_counter()
    r.render(make_optics(camera=StubCamera(), magnification=30000, defocus_um=-1.2))
    cached = time.perf_counter() - t0
    print(f"TEM 4096^2: first {first:.3f}s, refocus {refocus:.3f}s, cached {cached:.3f}s")
    assert first < budget(1.5) and refocus < budget(0.8) and cached < budget(0.2)
