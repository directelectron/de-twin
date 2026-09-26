"""Optics: calibration, beam/dose model and derive_optics (port of OpticsState::FromParams)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from de_twin.optics import (Calibration, OpticsConfig, beam_current_pa, derive_optics,
                            electron_wavelength_nm, illuminated_diameter_um, interaction_constant,
                            track_live_view)
from de_twin.render.testing import StubCamera
from de_twin.state import (AcquisitionRequest, ExposureMode, MicroscopeState, ProbeMode, Projection,
                           RenderMode, Roi, TemStem, Vec2)

DE16 = StubCamera()
MAG_YAML = Path(r"C:\direct_electron\virtual-specimen\configurations")


def _state(**kw) -> MicroscopeState:
    s = MicroscopeState()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _derive(state=None, request=None, camera=DE16, cfg=None, cal=None):
    return derive_optics(state or MicroscopeState(), request or AcquisitionRequest(), camera,
                         cal or Calibration.default(), cfg or OpticsConfig())


# ------------------------------------------------------------ physics
def test_wavelength_and_sigma():
    assert electron_wavelength_nm(200) == pytest.approx(0.0025079, rel=1e-4)
    assert electron_wavelength_nm(300) == pytest.approx(0.0019687, rel=1e-4)
    assert interaction_constant(200) == pytest.approx(0.00729, rel=5e-3)


# -------------------------------------------------------- calibration
def test_default_calibration_is_geometric():
    cal = Calibration.default()
    s = _state(magnification=20000)
    assert cal.specimen_pixel_nm(s, DE16) == pytest.approx(6.5e3 / 20000)
    assert cal.recip_pixel_inv_nm(s, DE16) == pytest.approx((6.5 / 250.0) / (1000 * electron_wavelength_nm(200)))
    assert cal.mag_ladder("MAG1")[0] == 2000 and cal.mag_ladder("MAG1")[-1] == 800000
    assert cal.mag_ladder("LowMAG")[0] == 50 and cal.mag_ladder("LowMAG")[-1] == 1500
    assert 250 in cal.cl_ladder()
    assert cal.snap_mag("MAG1", 23000) == 25000
    assert cal.snap_cl(230) == 250
    # a magnification that is not on the ladder is not calibrated (-> fallback)
    assert cal.specimen_pixel_nm(_state(magnification=12345), DE16) == 0.0


@pytest.mark.skipif(not (MAG_YAML / "mag.yaml").exists(), reason="DE-Server configurations not present")
@pytest.mark.parametrize("name", ["mag.yaml", "mag_JEOL.yaml", "mag_TFS.yaml"])
def test_mag_yaml_loads(name):
    cal = Calibration.from_mag_yaml(MAG_YAML / name)
    assert cal.mag_modes()
    assert cal.cl_ladder()


@pytest.mark.skipif(not (MAG_YAML / "mag.yaml").exists(), reason="DE-Server configurations not present")
def test_mag_yaml_values():
    cal = Calibration.from_mag_yaml(MAG_YAML / "mag.yaml")
    s = _state(magnification=20000, mag_mode="MAG1")
    assert cal.specimen_pixel_nm(s, DE16) == pytest.approx(0.2045)
    s.camera_length_mm = 250.0
    assert cal.recip_pixel_inv_nm(s, DE16) == pytest.approx(0.0176)
    s.mag_mode = "LowMAG"
    s.magnification = 150
    assert cal.specimen_pixel_nm(s, DE16) == pytest.approx(4.4537)
    s.tem_stem = TemStem.STEM
    s.magnification = 1_000_000
    assert cal.stem_step_nm(s, (256, 256)) == pytest.approx(80.0 / 256)


def test_mag_yaml_with_tabs_and_ht_table(tmp_path):
    (tmp_path / "mag.yaml").write_text(
        "- Project: MAG1\n  Mode: 1\n  SubMode: 0\n  TemStemMode: 0\n  Mags:\n    1000:\t2.0\n"
        "- Project: Diff Mag\n  Mode: 2\n  SubMode: 4\n  TemStemMode: 0\n  CamLength(cm):\n    25:\t0.02\n")
    (tmp_path / "ht.yaml").write_text(
        "HT:\n  300000:\n    TEM_Scale_Factor: 1.1\n    TEM_Diff_Scale_Factor: 0.8\n")
    cal = Calibration.from_mag_yaml(tmp_path / "mag.yaml", ht_table=tmp_path / "ht.yaml")
    s = _state(magnification=1000, ht_kv=300.0, camera_length_mm=250.0)
    assert cal.specimen_pixel_nm(s, DE16) == pytest.approx(2.2)
    assert cal.recip_pixel_inv_nm(s, DE16) == pytest.approx(0.016)


# ---------------------------------------------------------- dispatch
def test_render_mode_selection_and_override():
    assert _derive().render_mode == RenderMode.TEM_IMAGING
    assert _derive(_state(projection=Projection.DIFFRACTION)).render_mode == RenderMode.TEM_DIFFRACTION
    assert _derive(_state(tem_stem=TemStem.STEM, projection=Projection.DIFFRACTION)).render_mode \
        == RenderMode.STEM_PARKED  # STEM is tested before diffraction
    r = AcquisitionRequest()
    r.scan.enabled = True
    assert _derive(request=r).render_mode == RenderMode.STEM_4D
    cfg = OpticsConfig(render_mode_override=RenderMode.TEM_DIFFRACTION)
    assert _derive(cfg=cfg).render_mode == RenderMode.TEM_DIFFRACTION


# ------------------------------------------------------------- scale
def test_raster_downsampling_and_scale():
    o = _derive(_state(magnification=20000))
    assert o.output_shape == (4096, 4096)
    assert o.raster_downsample == 4 and o.view.shape == (1024, 1024)  # MAX_RASTER_PIXELS
    assert o.specimen_pixel_nm == pytest.approx(0.325)
    assert o.view.pixel_um == pytest.approx(0.325e-3 * 4)
    small = StubCamera(sensor_shape=(1024, 1024))
    o = _derive(_state(magnification=20000), camera=small)
    assert o.raster_downsample == 1 and o.view.shape == (1024, 1024)


def test_magnification_scaling():
    a = _derive(_state(magnification=20000))
    b = _derive(_state(magnification=40000))
    assert a.specimen_pixel_nm / b.specimen_pixel_nm == pytest.approx(2.0)


def test_fallback_and_override_pixel():
    o = _derive(_state(magnification=12345), cfg=OpticsConfig(fallback_pixel_nm=0.7))
    assert o.specimen_pixel_nm == pytest.approx(0.7)
    o = _derive(cfg=OpticsConfig(pixel_size_override_nm=0.2))
    assert o.specimen_pixel_nm == pytest.approx(0.2)


def test_saed_raster_covers_selected_area():
    s = _state(projection=Projection.DIFFRACTION, intensity=0.5)
    o = _derive(s, cfg=OpticsConfig(sa_aperture_um=1.0))
    assert o.view.shape == (256, 256)
    assert o.view.pixel_um * 256 == pytest.approx(1.0)
    o = _derive(s, cfg=OpticsConfig(sa_aperture_um=100.0))  # the illuminated area limits
    assert o.view.pixel_um * 256 == pytest.approx(illuminated_diameter_um(s))


def test_camera_length_scale():
    a = _derive(_state(projection=Projection.DIFFRACTION, camera_length_mm=200.0))
    b = _derive(_state(projection=Projection.DIFFRACTION, camera_length_mm=400.0))
    assert a.recip_pixel_inv_nm / b.recip_pixel_inv_nm == pytest.approx(2.0)
    lam = electron_wavelength_nm(200)
    assert a.recip_pixel_inv_nm == pytest.approx(6.5 / 200.0 / (1000 * lam))


# ------------------------------------------------------ sign conventions
def test_view_centre_sign_convention():
    s = _state(magnification=20000)
    s.stage.x_um = 1.0
    s.stage.y_um = -2.0
    o = _derive(s)
    assert o.view.center_um == pytest.approx((-1.0, 2.0))
    s = _state(image_shift_um=Vec2(0.5, 0.25), beam_shift_um=Vec2(0.1, 0.0))
    o = _derive(s)
    assert o.view.center_um == pytest.approx((0.5, 0.25)), "beam shift moves the beam, not the image"
    assert o.beam_offset_px == pytest.approx((0.1 / o.view.pixel_um, 0.0), rel=1e-6)
    s = _state()
    s.stage.x_um = 1.0
    o = _derive(s, cfg=OpticsConfig(flip_x=True, stage_offset_um=(0.5, 0.0, 0.0)))
    assert o.view.center_um == pytest.approx((1.5, 0.0))


def test_stage_moves_content_with_it():
    """README 2.2: +1 um of stage x moves the content by +1000/nm_per_px pixels."""
    s = _state(magnification=20000)
    o0 = _derive(s)
    s.stage.x_um = 0.1
    o1 = _derive(s)
    r0, c0 = o0.view.world_to_pixel(0.0, 0.0)
    r1, c1 = o1.view.world_to_pixel(0.0, 0.0)
    d = o0.raster_downsample
    assert (c1 - c0) * d == pytest.approx(100.0 / o0.specimen_pixel_nm)
    assert r1 == pytest.approx(r0)
    s.stage.x_um = 0.0
    s.image_shift_um = Vec2(0.1, 0.0)
    o2 = _derive(s)
    _, c2 = o2.view.world_to_pixel(0.0, 0.0)
    assert (c2 - c0) * d == pytest.approx(-100.0 / o0.specimen_pixel_nm)


def test_tilt_foreshortening():
    s = _state()
    s.stage.alpha_deg = 60.0
    o = _derive(s)
    assert o.view.cos_alpha == pytest.approx(0.5)
    assert o.thickness_tilt_factor == pytest.approx(2.0)
    s.stage.alpha_deg = 89.9
    assert _derive(s).view.cos_alpha == pytest.approx(0.1)


def test_defocus_includes_z_offsets_and_eucentric():
    s = _state(defocus_um=-1.0)
    s.stage.z_um = 3.0
    o = _derive(s, cfg=OpticsConfig(defocus_offset_um=0.5, eucentric_height_um=1.0,
                                    stage_offset_um=(0, 0, 0.25)))
    assert o.defocus_um == pytest.approx(-1.0 + 0.5 + (3.0 + 0.25 - 1.0))
    assert o.fresnel_sign == -1.0  # overfocus
    o = _derive(_state(defocus_um=-10.0))
    assert o.fresnel_sign == 1.0 and o.fresnel_gain == pytest.approx(0.6)


def test_blur_matches_cpp_formula():
    cfg = OpticsConfig(illumination_semi_angle_mrad=0.1)
    small = StubCamera(sensor_shape=(1024, 1024))
    o = _derive(_state(defocus_um=5.0), camera=small, cfg=OpticsConfig(
        illumination_semi_angle_mrad=0.1, pixel_size_override_nm=0.5))
    assert o.blur_sigma_px == pytest.approx(1.0)  # 5000 nm * 1e-4 / 0.5
    o = _derive(_state(defocus_um=0.0), camera=small, cfg=cfg)
    assert o.blur_sigma_px == pytest.approx(0.5)


def test_probe_and_disk():
    s = _state(tem_stem=TemStem.STEM, probe_mode=ProbeMode.NANOPROBE, spot_size=3,
               convergence_semi_angle_mrad=20.0, defocus_um=0.1, camera_length_mm=100.0)
    r = AcquisitionRequest()
    r.scan.enabled = True
    o = _derive(s, r)
    assert o.convergence_mrad == pytest.approx(20.0)
    assert o.probe_diameter_nm == pytest.approx(1.3 + 2 * 100.0 * 0.02)
    lam = electron_wavelength_nm(200)
    assert o.disk_radius_px == pytest.approx(0.02 / (lam * o.recip_pixel_inv_nm))
    assert o.extras["mrad_per_px"] == pytest.approx(6.5 / 100.0)


def test_stem_geometry():
    s = _state(tem_stem=TemStem.STEM, magnification=100000)
    r = AcquisitionRequest()
    r.scan.enabled = True
    r.scan.size = (64, 32)
    r.scan.rotation_deg = 30.0
    r.hw_roi = Roi(1920, 1920, 256, 256)
    o = _derive(s, r)
    assert o.render_mode == RenderMode.STEM_4D
    assert o.view.shape == (32, 64) and o.scan_shape == (32, 64)
    assert o.scan_rotation_rad == pytest.approx(math.radians(30.0))
    assert o.output_shape == (256, 256)
    assert o.diffraction_center_px == pytest.approx((128.0, 128.0))
    r.scan.step_um = 0.01
    o = _derive(s, r)
    assert o.scan_step_um == pytest.approx(0.01) and o.view.pixel_um == pytest.approx(0.01)
    r.scan.park = True
    r.scan.park_position = (10, 5)
    o = _derive(s, r)
    wx, wy = o.view.pixel_to_world(5, 10)
    assert o.park_um == pytest.approx((float(wx), float(wy)))


def test_diffraction_shift_moves_centre():
    s = _state(projection=Projection.DIFFRACTION, diffraction_shift_mrad=Vec2(1.0, -0.5))
    o = _derive(s)
    mpp = o.extras["mrad_per_px"]
    assert o.diffraction_center_px == pytest.approx((2048 + 1.0 / mpp, 2048 - 0.5 / mpp))


# --------------------------------------------------------------- dose
def test_lowmag_spreads_the_beam():
    a = illuminated_diameter_um(_state(mag_mode="MAG1"))
    b = illuminated_diameter_um(_state(mag_mode="LowMAG"))
    assert b / a == pytest.approx(30.0)


def test_beam_current_model():
    assert beam_current_pa(_state(spot_size=3)) == pytest.approx(600.0)
    assert beam_current_pa(_state(spot_size=1)) / beam_current_pa(_state(spot_size=5)) == pytest.approx(25.0)
    nano = beam_current_pa(_state(spot_size=3, probe_mode=ProbeMode.NANOPROBE))
    assert nano == pytest.approx(30.0)
    assert beam_current_pa(_state(ht_on=False)) == 0.0
    small_cla = beam_current_pa(_state(condenser_aperture_index=2))
    assert small_cla == pytest.approx(600.0 * (70 / 150) ** 2)


def test_realistic_tem_dose_on_de16():
    for mag, lo, hi in ((20000, 20, 80), (30000, 10, 50), (50000, 3, 20)):
        o = _derive(_state(magnification=mag, spot_size=3, intensity=0.5))
        assert lo < o.dose_e_per_px_s < hi, (mag, o.dose_e_per_px_s)


def test_dose_scaling():
    base = _derive(_state(magnification=20000))
    assert _derive(_state(magnification=40000)).dose_e_per_px_s == pytest.approx(base.dose_e_per_px_s / 4)
    assert _derive(_state(magnification=20000, spot_size=1)).dose_e_per_px_s == pytest.approx(
        base.dose_e_per_px_s * 9)
    # spreading the beam (higher intensity -> larger area) lowers the dose ~ 1/D^2
    s1, s2 = _state(intensity=0.4), _state(intensity=0.6)
    r = _derive(s1).dose_e_per_px_s / _derive(s2).dose_e_per_px_s
    assert r == pytest.approx((illuminated_diameter_um(s2) / illuminated_diameter_um(s1)) ** 2)
    # physical: J * p^2 with J = I / (e * area)
    d_nm = illuminated_diameter_um(MicroscopeState()) * 1000
    j = 600e-12 / 1.602176634e-19 / (math.pi * d_nm ** 2 / 4)
    assert base.dose_e_per_px_s == pytest.approx(j * 0.325 ** 2)


def test_stem_dose_is_probe_current():
    s = _state(tem_stem=TemStem.STEM, spot_size=3)
    r = AcquisitionRequest()
    r.scan.enabled = True
    o = _derive(s, r)
    assert o.pattern_e_per_s == pytest.approx(beam_current_pa(s) * 1e-12 / 1.602176634e-19)


def test_saed_dose_limited_by_aperture():
    s = _state(projection=Projection.DIFFRACTION, intensity=0.5)
    d = illuminated_diameter_um(s)
    full = _derive(s, cfg=OpticsConfig(sa_aperture_um=0.0)).pattern_e_per_s
    part = _derive(s, cfg=OpticsConfig(sa_aperture_um=d / 2)).pattern_e_per_s
    assert part / full == pytest.approx(0.25)


def test_blanking_flags():
    assert _derive(_state(beam_blanked=True)).beam_blanked
    assert _derive(_state(column_valves_open=False)).beam_blanked
    r = AcquisitionRequest(exposure_mode=ExposureMode.DARK)
    assert _derive(request=r).beam_blanked
    assert not _derive().beam_blanked


def test_astigmatism_and_aberrations():
    o = _derive(_state(objective_stig=Vec2(0.05, -0.02)))
    assert o.astigmatism_nm == pytest.approx((50.0, -20.0))
    assert o.cs_mm == 1.2 and o.focal_spread_nm > 1.0


def test_roi_offsets_view_in_tem():
    r = AcquisitionRequest(hw_roi=Roi(0, 0, 1024, 1024))
    o = _derive(_state(magnification=20000), r)
    assert o.output_shape == (1024, 1024)
    # the ROI centre is 1536 px left/up of the sensor centre
    assert o.view.center_um == pytest.approx((-1536 * 0.325e-3, -1536 * 0.325e-3))


def test_track_live_view():
    s = _state(tem_stem=TemStem.STEM)
    r = AcquisitionRequest()
    r.scan.enabled = True
    o = _derive(s, r)
    assert track_live_view(o, s) is o
    s.stage.x_um = 0.5
    o2 = track_live_view(o, s)
    assert o2.view.center_um == pytest.approx((-0.5, 0.0))
    assert o2.view.shape == o.view.shape


def test_max_raster_pixels_zero_renders_at_the_frame_sampling():
    capped = _derive(_state(magnification=20000))
    assert capped.raster_downsample == 4 and "upsampled 4x" in capped.resolution_warning
    native = _derive(_state(magnification=20000), cfg=OpticsConfig(max_raster_pixels=0))
    assert native.raster_downsample == 1 and native.view.shape == (4096, 4096)
    assert native.resolution_warning == ""
    assert native.view.pixel_um == pytest.approx(0.325e-3)
    mid = _derive(_state(magnification=20000), cfg=OpticsConfig(max_raster_pixels=2048 * 2048))
    assert mid.raster_downsample == 2


def test_intensity_zoom_keeps_the_dose_per_pixel_as_the_magnification_changes():
    from de_twin.state import Projection

    doses = {}
    for zoom in (False, True):
        cfg = OpticsConfig(intensity_zoom=zoom)
        doses[zoom] = [_derive(_state(magnification=m), cfg=cfg).dose_e_per_px_s
                       for m in (5000, 20000, 80000)]
    assert doses[True][0] == pytest.approx(doses[True][1], rel=1e-6)
    assert doses[True][2] == pytest.approx(doses[True][1], rel=1e-6)
    assert doses[False][0] > 10 * doses[False][2], "without it, dose/px goes as 1/mag^2"
    # at the reference magnification the two agree
    ref = OpticsConfig().intensity_zoom_reference_mag
    s = _state(magnification=ref)
    assert _derive(s, cfg=OpticsConfig(intensity_zoom=True)).dose_e_per_px_s == pytest.approx(
        _derive(s).dose_e_per_px_s, rel=1e-6)
    # diffraction is left alone
    d = _state(magnification=5000, projection=Projection.DIFFRACTION)
    assert _derive(d, cfg=OpticsConfig(intensity_zoom=True)).illuminated_diameter_um == pytest.approx(
        _derive(d).illuminated_diameter_um)


def test_beam_shift_moves_a_scan_even_on_a_column_that_reports_tem():
    """A 4D-STEM request on a column in TEM (DE-Server's Scan - Enable) is a scan: beam shift
    moves the probe, i.e. the scanned area."""
    r = AcquisitionRequest()
    r.scan.enabled = True
    a = _derive(_state(), request=r)
    b = _derive(_state(beam_shift_um=Vec2(0.5, 0.0)), request=r)
    assert a.render_mode == RenderMode.STEM_4D
    assert b.view.center_um[0] - a.view.center_um[0] == pytest.approx(0.5)
