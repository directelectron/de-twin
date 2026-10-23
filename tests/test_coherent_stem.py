"""Coherent 4D-STEM: BF-disk interference, consistency with the kinematic model, frame-by-frame
rendering, detector integration, model selection and ground truth."""

from __future__ import annotations

import math

import numpy as np
import pytest

from coherent_helpers import stem_optics
from de_twin.clock import ManualClock
from de_twin.render import RenderConfig, Renderer
from de_twin.render.testing import Particle, SyntheticSpecimen
from de_twin.specimen.materials import MaterialId
from de_twin.state import Roi, ScanRequest, TemStem
from de_twin.twin import DigitalTwin

COH = RenderConfig(stem_model="coherent")


def _particle(o, ix, iy, r_nm, **kw):
    x, y = o.view.pixel_to_world(iy, ix)
    return Particle(float(x), float(y), r_nm, **kw)


def test_vacuum_bf_disk_is_uniform_and_sized():
    o = stem_optics(defocus_nm=-200.0, probe_ab={"C3": 0.0}, det_k_over_alpha=1.6)
    r = Renderer(SyntheticSpecimen(film_material=MaterialId.VACUUM), COH)
    p = r.render(o, scan_point=(5, 7))
    cx, cy = o.diffraction_center_px
    y, x = np.indices(p.shape)
    rr = np.hypot(x - cx, y - cy)
    inside = p[rr < 0.85 * o.disk_radius_px]
    assert inside.std() / inside.mean() < 0.03  # no specimen, no interference
    assert p[rr > o.disk_radius_px + 2].sum() < 1e-3 * p.sum()
    assert p.sum() == pytest.approx(o.pattern_e_per_s, rel=2e-3)


def test_defocused_bf_disk_shows_a_moving_shadow_image():
    """With defocus the BF disk is a shadow image (Ronchigram) of the specimen: a particle at r0
    appears at detector k = (r0 - r_p) / (lambda C1), so it moves when the probe moves."""
    df = -300.0
    o = stem_optics(scan=(16, 16), step_nm=0.5, defocus_nm=df, probe_ab={"C3": 0.0}, det_k_over_alpha=1.3,
                    roi=128)
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM, particles=[_particle(o, 8, 8, 0.8)])
    cfg = RenderConfig(stem_model="coherent", coherent_focal_spread=False, coherent_source_size=False)
    r = Renderer(spec, cfg)
    vac = Renderer(SyntheticSpecimen(film_material=MaterialId.VACUUM), cfg).render(o, scan_point=(0, 0))
    cx, cy = o.diffraction_center_px
    y, x = np.indices(vac.shape)
    disk = np.hypot(x - cx, y - cy) < 0.8 * o.disk_radius_px

    from scipy.ndimage import gaussian_filter

    def shadow_x(ix):
        d = gaussian_filter(np.abs(r.render(o, scan_point=(ix, 8)) - vac) * disk, 2.0)
        return float(np.unravel_index(int(np.argmax(d)), d.shape)[1])

    x_left, x_right = shadow_x(6), shadow_x(10)  # probe moves +2 nm in x
    expected = (4 * 0.5) / (o.wavelength_nm * abs(df)) / o.recip_pixel_inv_nm  # detector px, underfocus: +
    assert x_right - x_left == pytest.approx(expected, rel=0.25)
    # the shadow is magnified by 1 / (lambda |C1|): a 1.6 nm particle spans ~2 / (lambda |C1| dk) px
    d = np.abs(r.render(o, scan_point=(8, 8)) - vac) * disk
    assert d.max() > 0.1 * vac[disk].mean()


def test_coherent_and_kinematic_agree_on_totals_and_adf():
    from de_twin.optics import Calibration, OpticsConfig, derive_optics
    from de_twin.render.testing import StubCamera
    from de_twin.state import AcquisitionRequest, MicroscopeState

    s = MicroscopeState()
    s.tem_stem = TemStem.STEM
    s.camera_length_mm = 13.0
    s.convergence_semi_angle_mrad = 10.0
    rq = AcquisitionRequest(hw_roi=Roi(1920, 1920, 256, 256), hw_binning=(2, 2))
    rq.scan.enabled = True
    rq.scan.size = (16, 16)
    rq.scan.step_um = 0.0008
    o = derive_optics(s, rq, StubCamera(), Calibration.default(), OpticsConfig())
    parts = [_particle(o, 4, 4, 2.0), _particle(o, 11, 7, 3.0)]
    spec = SyntheticSpecimen(film_material=MaterialId.AMORPHOUS_CARBON, film_thickness_nm=10, particles=parts)
    rk = Renderer(spec, RenderConfig(stem_model="kinematic", intensity_jitter=False, probe_footprint_blend=False))
    rc = Renderer(spec, COH)
    tc = rc.datacube(o).sum(axis=(2, 3))
    tk = rk.datacube(o).sum(axis=(2, 3))
    assert tc.mean() == pytest.approx(tk.mean(), rel=0.02)
    assert np.abs(tc / tk - 1).max() < 0.03
    for lo, hi in ((40.0, 200.0), (0.0, 10.0)):
        vc, vk = rc.virtual_image(o, lo, hi), rk.virtual_image(o, lo, hi)
        assert np.corrcoef(vc.ravel(), vk.ravel())[0, 1] > 0.95
        assert vc.max() == pytest.approx(vk.max(), rel=0.3)
    adf_c, adf_k = rc.virtual_image(o, 40, 200), rk.virtual_image(o, 40, 200)
    film = (0, 15)  # corner: carbon only
    contrast_c = adf_c.max() / adf_c[film]
    contrast_k = adf_k.max() / adf_k[film]
    assert contrast_c == pytest.approx(contrast_k, rel=0.5)


def test_frame_by_frame_matches_datacube_and_uses_blocks():
    o = stem_optics(scan=(12, 10), defocus_nm=-20.0, probe_ab={"C3": 0.0})
    spec = SyntheticSpecimen(film_material=MaterialId.AMORPHOUS_CARBON, film_thickness_nm=4,
                             particles=[_particle(o, 5, 5, 1.0)])
    r = Renderer(spec, COH)
    cube = r.datacube(o)
    assert cube.shape == (10, 12, 64, 64)
    before = r._coherent.blocks_computed
    for fi in (0, 17, 63, 119):
        img = r.render(o, frame_index=fi)
        assert np.array_equal(img, cube[fi // 12, fi % 12])
    assert r._coherent.blocks_computed == before  # served from the block cache


def test_live_scan_reads_ahead_and_matches_the_datacube():
    """Walking the scan frame by frame computes the next blocks in the background; every frame is
    the datacube's pattern."""
    o = stem_optics(scan=(12, 10), defocus_nm=-20.0, probe_ab={"C3": 0.0})
    spec = SyntheticSpecimen(film_material=MaterialId.AMORPHOUS_CARBON, film_thickness_nm=4,
                             particles=[_particle(o, 5, 5, 1.0)])
    cfg = RenderConfig(stem_model="coherent", coherent_max_grid=128)
    cube = Renderer(spec, cfg).datacube(o)
    r = Renderer(spec, cfg)
    for fi in range(120):
        assert np.array_equal(r.render(o, frame_index=fi), cube[fi // 12, fi % 12])


def test_probe_is_kept_across_a_stage_move_and_a_new_scan_step():
    import dataclasses

    o = stem_optics(scan=(6, 6), defocus_nm=-20.0, probe_ab={"C3": 0.0})
    r = Renderer(SyntheticSpecimen(film_material=MaterialId.AMORPHOUS_CARBON, film_thickness_nm=4), COH)
    r.render(o, frame_index=0)
    p = r._coherent.probe(o)
    moved = dataclasses.replace(o, view=dataclasses.replace(o.view, center_um=(o.view.center_um[0] + 0.002,
                                                                               o.view.center_um[1])))
    r.render(moved, frame_index=0)
    r.render(stem_optics(scan=(6, 6), step_nm=0.3, defocus_nm=-20.0, probe_ab={"C3": 0.0}), frame_index=0)
    assert r._coherent.probe(moved) is p and len(r._coherent._probes) == 1


def test_a_sparse_scan_is_split_into_small_tiles():
    """A scan field larger than coherent_field_max_px is never one tile spanning all the points
    of a block (a coarse scan used to build a tile of the whole row: GBs); the patterns are the
    per-point ones."""
    o = stem_optics(scan=(4, 4), step_nm=40.0, defocus_nm=-20.0, probe_ab={"C3": 0.0})
    spec = SyntheticSpecimen(film_material=MaterialId.AMORPHOUS_CARBON, film_thickness_nm=4,
                             particles=[_particle(o, 1, 2, 1.0)])
    limit = 400_000
    r = Renderer(spec, RenderConfig(stem_model="coherent", coherent_field_max_px=limit))
    c = r._coherent
    s = c.sampling(o)
    pts = c._full_scan_points(o)
    parts = c._field_views(o, s, pts)
    assert len(parts) > 1 and all(v.shape[0] * v.shape[1] <= limit for _, v in parts)
    cube = r.datacube(o)
    one = Renderer(spec, RenderConfig(stem_model="coherent", coherent_field_max_px=limit))
    for ix, iy in ((1, 2), (3, 0)):
        assert np.array_equal(one.render(o, scan_point=(ix, iy)), cube[iy, ix])


def test_coherent_prefetch_computes_the_predicted_view():
    o = stem_optics(scan=(8, 8), defocus_nm=-20.0, probe_ab={"C3": 0.0})
    o2 = stem_optics(scan=(8, 8), defocus_nm=-25.0, probe_ab={"C3": 0.0})
    spec = SyntheticSpecimen(film_material=MaterialId.AMORPHOUS_CARBON, film_thickness_nm=4,
                             particles=[_particle(o, 4, 4, 1.0)])
    r = Renderer(spec, COH)
    r.render(o, frame_index=0)
    assert r.prefetch(o2)
    before = r._coherent.blocks_computed
    img = r.render(o2, frame_index=0)
    assert r._coherent.blocks_computed == before  # the prefetched block
    assert np.array_equal(img, Renderer(spec, COH).render(o2, frame_index=0))
    assert not r.prefetch(o2)  # nothing left to do


def test_auto_model_selection():
    spec = SyntheticSpecimen(film_material=MaterialId.VACUUM)
    r = Renderer(spec)
    assert r.config.stem_model == "auto"
    assert r.stem_model(stem_optics()) == "coherent"  # 64^2 ptychography-style scan
    # huge patterns and a coarse scan -> kinematic
    big = stem_optics(roi=1024, step_nm=50.0, probe_ab={"C3": 0.0}, defocus_nm=0.0)
    assert r.stem_model(big) == "kinematic"
    assert Renderer(spec, RenderConfig(stem_model="kinematic")).stem_model(stem_optics()) == "kinematic"
    assert Renderer(spec, COH).stem_model(big) == "coherent"


def test_ground_truth_ptychography_contents():
    o = stem_optics(scan=(8, 6), defocus_nm=-15.0, probe_ab={"C3": 500.0, "B2": 5 + 2j},
                    condenser_stig=(0.002, 0.0), rotation_deg=20.0)
    spec = SyntheticSpecimen(film_material=MaterialId.AMORPHOUS_CARBON, film_thickness_nm=3)
    gt = Renderer(spec, COH).ground_truth_ptychography(o)
    s = gt["sampling"]
    assert gt["object"].dtype == np.complex64 and gt["object"].ndim == 2
    assert gt["positions_px"].shape == (48, 2)
    assert gt["C1_nm"] == pytest.approx(-15.0) and gt["A1_nm"] == pytest.approx(2.0)
    assert gt["C3_nm"] == pytest.approx(500.0) and gt["B2_nm"] == pytest.approx(5 + 2j)
    assert gt["dx_nm"] == pytest.approx(s.dx_nm) and gt["probe"].shape == (s.n, s.n)
    assert gt["mode_weights"].sum() == pytest.approx(1.0)
    # positions follow the (rotated) scan: step 0.25 nm along the fast axis, rotated by 20 deg
    d = (gt["positions_nm"][1] - gt["positions_nm"][0])
    assert math.hypot(*d) == pytest.approx(0.25, abs=gt["dx_nm"])
    assert math.degrees(math.atan2(d[0], d[1])) == pytest.approx(20.0, abs=6.0)
    # the object sits inside the tile around every probe window
    n2 = s.n // 2
    assert gt["positions_px"].min() >= n2 and (gt["positions_px"] + n2 <= np.array(gt["object"].shape)).all()


def test_twin_frames_on_a_binned_integrating_camera_reproduce_the_patterns():
    spec = SyntheticSpecimen(film_material=MaterialId.AMORPHOUS_CARBON, film_thickness_nm=5,
                             particles=[Particle(0.0, 0.0, 1.5)])
    tw = DigitalTwin(spec, camera="Celeritas", clock=ManualClock())
    c = tw.column
    c.set_tem_stem(TemStem.STEM)
    c.set_convergence_mrad(10.0)
    c.set_defocus_um(-0.02)
    c.set_camera_length_mm(40.0)
    c.set_spot_size(8)
    dwell = 1e-4
    req = tw.request(hw_roi=Roi(384, 384, 256, 256), hw_binning=(2, 2), frame_time_s=dwell, total_frames=36,
                     scan=ScanRequest(enabled=True, size=(6, 6), step_um=0.0005, dwell_s=dwell))
    assert tw.stem_model(req) == "coherent"
    cube = tw.datacube(req)
    assert cube.shape == (6, 6, 128, 128)
    refs = tw.processor.references(req)
    ape = tw.detector.model.adu_per_electron_at(c.state().ht_kv)
    frames, points = [], []
    for raw, meta in tw.frames(req):
        assert raw.shape == (128, 128)
        frames.append(tw.processor.correct(raw, refs) / ape * 4.0)  # hardware binning averages 2x2
        points.append(meta.scan_point)
    x = np.array(frames)
    mu = np.array([cube[iy, ix] for ix, iy in points]) * dwell
    assert x.sum() / mu.sum() == pytest.approx(1.0, rel=0.03)
    bright = mu > 5
    z = (x[bright] - mu[bright]) / np.sqrt(mu[bright])
    cv = tw.detector.model.adu_cv if hasattr(tw.detector.model, "adu_cv") else 0.7
    assert abs(z.mean()) < 0.1
    assert 0.6 < z.std() < 1.5 * math.sqrt(1 + cv * cv)  # shot noise (+ deposit statistics), no bias
