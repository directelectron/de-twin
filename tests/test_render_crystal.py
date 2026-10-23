"""Rendering on the crystal library: disk sizes follow the column's convergence in every
mode, CBED geometry, tilt-dependent bright-field contrast, determinism and speed."""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
import pytest
from perf_budget import budget

from de_twin.clock import ManualClock
from de_twin.column import Column
from de_twin.optics import Calibration, OpticsConfig, derive_optics
from de_twin.optics.physics import electron_wavelength_nm
from de_twin.render import RenderConfig, Renderer
from de_twin.render.diffraction import blur_sigma_px
from de_twin.render.testing import Particle, StubCamera, SyntheticSpecimen, on_zone_grains
from de_twin.specimen import Specimen, from_name
from de_twin.specimen.materials import GRAINS_PER_MATERIAL, MaterialId
from de_twin.state import AcquisitionRequest, MicroscopeState, Projection, Roi, TemStem

CAM = StubCamera(sensor_shape=(1024, 1024))
AU = MaterialId.GOLD
LAM = electron_wavelength_nm(200.0)


def _column_state(probe_mode: str, alpha: int = 3, stem: bool = False) -> MicroscopeState:
    column = Column(clock=ManualClock())
    if stem:
        column.set("TemStemMode", 1)
    column.set("ProbeMode", probe_mode)
    column.set("AlphaSelector", alpha)
    return column.state()


def _disk_radius(image, cx, cy, window):
    """Radius of a uniform disk from the second moment of the rendered (disk (x) PSF) spot."""
    y, x = np.indices(image.shape)
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    m = r2 < window**2
    mom = (image[m] * r2[m]).sum() / image[m].sum()
    return mom


@pytest.mark.parametrize("probe,cl_mm", [("TEM", 250.0), ("NBD", 250.0), ("CBD", 60.0)])
def test_diffraction_disk_radius_follows_the_convergence(probe, cl_mm):
    s = _column_state(probe)
    s.projection = Projection.DIFFRACTION
    s.camera_length_mm = cl_mm
    o = derive_optics(s, AcquisitionRequest(), CAM, Calibration.default(), OpticsConfig(sa_aperture_um=0.5))
    alpha = s.convergence_semi_angle_mrad
    lo, hi = {"TEM": (0.01, 0.1), "NBD": (0.5, 3.0), "CBD": (3.0, 15.0)}[probe]
    assert lo <= alpha <= hi and o.convergence_mrad == pytest.approx(alpha)
    expect = alpha / 1000.0 / (LAM * o.recip_pixel_inv_nm)
    assert o.disk_radius_px == pytest.approx(max(expect, 0.75), abs=1e-6)
    image = Renderer(SyntheticSpecimen(film_material=MaterialId.VACUUM)).render(o)
    cx, cy = o.diffraction_center_px
    r = max(o.disk_radius_px, 0.75)
    sigma = blur_sigma_px(r)
    mom = _disk_radius(image, cx, cy, r + 6 * sigma + 5)
    measured = math.sqrt(max(2.0 * (mom - 2.0 * sigma * sigma), 0.0))  # <rho^2> = r^2/2 + 2 sigma^2
    assert measured == pytest.approx(expect, abs=1.0)
    if probe == "TEM":  # parallel illumination: a small spot, still at least PSF sized
        assert image.max() < 0.5 * image.sum() and expect < 5.0


def test_stem_convergence_is_a_stem_convergence():
    s = _column_state("TEM", alpha=3, stem=True)
    assert 5.0 <= s.convergence_semi_angle_mrad <= 30.0
    r = AcquisitionRequest(hw_roi=Roi(1920, 1920, 256, 256))
    r.scan.enabled = True
    r.scan.size = (8, 8)
    o = derive_optics(s, r, StubCamera(), Calibration.default(), OpticsConfig())
    assert o.convergence_mrad == pytest.approx(s.convergence_semi_angle_mrad)
    bare = MicroscopeState()
    bare.tem_stem = TemStem.STEM
    assert 5.0 <= derive_optics(bare, r, StubCamera()).convergence_mrad <= 30.0  # fallback


def _cbed(alpha_mrad, cl_mm=100.0):
    gid = AU * GRAINS_PER_MATERIAL + 2
    spec = SyntheticSpecimen(
        film_material=MaterialId.VACUUM,
        particles=[Particle(0.0, 0.0, 30.0, grain=2)],
        grains=on_zone_grains(gid, (0, 1, 1)),
    )
    s = MicroscopeState()
    s.tem_stem = TemStem.STEM
    s.camera_length_mm = cl_mm
    s.convergence_semi_angle_mrad = alpha_mrad
    r = AcquisitionRequest(hw_roi=Roi(1792, 1792, 512, 512))
    r.scan.enabled = True
    r.scan.size = (9, 9)
    r.scan.step_um = 0.002
    o = derive_optics(s, r, StubCamera(), Calibration.default(), OpticsConfig())
    image = Renderer(
        spec,
        RenderConfig(
            intensity_jitter=False, probe_footprint_blend=False, diffuse_scattering=False, film_halo=False
        ),
    ).render(o, scan_point=(4, 4))
    return o, image


def test_cbed_disks_separate_when_alpha_is_below_half_the_bragg_angle():
    theta_111 = 1000.0 * LAM * math.sqrt(3) / 0.4078  # 10.6 mrad
    for alpha, overlap in ((4.0, False), (10.0, True)):
        o, image = _cbed(alpha)
        mpp = o.extras["mrad_per_px"]
        assert o.disk_radius_px == pytest.approx(alpha / mpp, rel=1e-6)
        cx, cy = o.diffraction_center_px
        # find the brightest {111} disk centre on the ring of radius theta_111
        y, x = np.indices(image.shape)
        ring = np.abs(np.hypot(x - cx, y - cy) * mpp - theta_111) < 0.3
        iy, ix = np.unravel_index(np.argmax(np.where(ring, image, 0)), image.shape)
        mid = image[round((iy + cy) / 2), round((ix + cx) / 2)]  # half-way to the direct disk
        if overlap:
            assert mid > 0.5 * image[int(cy), int(cx)]  # the disks overlap (2 alpha > theta_111)
        else:
            assert mid < 1e-3 * image[int(cy), int(cx)]  # dark gap between separate disks
            assert image[iy, ix] > 1e-3 * image[int(cy), int(cx)]  # the {111} disk is there


def test_tem_bright_field_contrast_changes_with_tilt():
    gid = AU * GRAINS_PER_MATERIAL + 4
    spec = SyntheticSpecimen(
        film_material=MaterialId.VACUUM,
        particles=[Particle(0.0, 0.0, 12.0, grain=4)],
        grains=on_zone_grains(gid, (0, 1, 1), in_plane_rad=0.4),
    )
    r = Renderer(spec, RenderConfig(phase_texture_scale=0.0, lattice_fringes=False))
    config = OpticsConfig(objective_aperture_mrad=5.0)
    means = []
    for a in np.arange(0.0, 6.01, 0.5):
        s = MicroscopeState()
        s.magnification = 50000
        s.stage.alpha_deg = float(a)
        o = derive_optics(
            s, AcquisitionRequest(), StubCamera(sensor_shape=(256, 256)), Calibration.default(), config
        )
        image = r.render(o) / o.dose_e_per_px_s
        field_map, _ = r.field_map(o)
        inside = field_map.grain_id == gid
        t = field_map.thickness_nm[inside] * o.thickness_tilt_factor
        means.append(float((image[inside] / np.exp(-t / 25.0)).mean()))  # remove the mass thickness
    means = np.array(means)
    assert np.ptp(means) > 0.1  # the grain goes through / off Bragg conditions
    assert means[0] < means.max()  # on zone it is dark in bright field


def test_determinism():
    a = Renderer(Specimen(from_name("Dense Au on holey C")))
    b = Renderer(Specimen(from_name("Dense Au on holey C")))
    s = MicroscopeState()
    s.projection = Projection.DIFFRACTION
    s.camera_length_mm = 150.0
    o = derive_optics(
        s,
        AcquisitionRequest(),
        StubCamera(sensor_shape=(256, 256)),
        Calibration.default(),
        OpticsConfig(sa_aperture_um=2.0),
    )
    assert np.array_equal(a.render(o), b.render(o))


# ---------------------------------------------------------------- perf
def _many_grains(n=200, seed=5):
    rng = np.random.default_rng(seed)
    return SyntheticSpecimen(
        film_thickness_nm=10.0,
        particles=[
            Particle(float(x), float(y), 8.0, AU, int(g))
            for x, y, g in zip(rng.uniform(-0.2, 0.2, n), rng.uniform(-0.2, 0.2, n), range(n))
        ],
    )


def test_perf_saed_and_tilt_step():
    spec = _many_grains()
    s = MicroscopeState()
    s.projection = Projection.DIFFRACTION
    s.camera_length_mm = 120.0
    o = derive_optics(s, AcquisitionRequest(), CAM, Calibration.default(), OpticsConfig(sa_aperture_um=0.5))
    r = Renderer(spec)
    r.render(o)  # libraries, raster
    times = []
    for a in (0.5, 1.0, 1.5):
        t0 = time.perf_counter()
        r.render(dataclasses.replace(o, alpha_rad=math.radians(a)))
        times.append(time.perf_counter() - t0)
    print(f"SAED 1024^2, 200 grains, tilt step: {min(times) * 1000:.0f} ms")
    assert min(times) < budget(0.5)
    # TEM bright field (512^2) tilt step
    t = MicroscopeState()
    t.magnification = 8000  # 416 nm field: all 200 grains in view
    tem = derive_optics(
        t, AcquisitionRequest(), StubCamera(sensor_shape=(512, 512)), Calibration.default(), OpticsConfig()
    )
    r.render(tem)
    assert len(np.unique(r.field_map(tem)[0].grain_id)) > 150
    times = []
    for a in (0.5, 1.0):
        t0 = time.perf_counter()
        r.render(dataclasses.replace(tem, alpha_rad=math.radians(a)))
        times.append(time.perf_counter() - t0)
    print(f"TEM 512^2, 200 grains, tilt step: {min(times) * 1000:.0f} ms")
    assert min(times) < budget(0.5)


def test_perf_virtual_image():
    spec = _many_grains(400)
    s = MicroscopeState()
    s.tem_stem = TemStem.STEM
    s.convergence_semi_angle_mrad = 10.0
    s.camera_length_mm = 100.0
    r = AcquisitionRequest(hw_roi=Roi(1920, 1920, 256, 256))
    r.scan.enabled = True
    r.scan.size = (128, 128)
    r.scan.step_um = 0.003
    o = derive_optics(s, r, StubCamera(), Calibration.default(), OpticsConfig())
    rend = Renderer(spec)
    rend.field_map(o, frozenset({"descan", "strain"}))
    t0 = time.perf_counter()
    haadf = rend.virtual_image(o, 40.0, 200.0)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    rend.virtual_image(o, 0.0, 10.0)
    again = time.perf_counter() - t0
    print(f"virtual image 128^2 (400 grains): first {first * 1000:.0f} ms, again {again * 1000:.0f} ms")
    assert haadf.shape == (128, 128) and haadf.max() > 0
    assert first < budget(1.0) and again < budget(0.3)
