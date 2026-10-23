"""End-to-end behaviour of the assembled twin (column + specimen + optics + detector)."""

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.render.testing import phase_correlation_shift
from de_twin.state import ExposureMode, Projection, ScanRequest, TemStem
from de_twin.twin import DigitalTwin


@pytest.fixture(scope="module")
def twin():
    tw = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(), seed=1)
    tw.column.set("Magnification", 20000)
    return tw


def test_raw_frames_carry_detector_signature(twin):
    raw = twin.raw_frame(0.025)
    dark = twin.raw_frame(0.025, exposure_mode=ExposureMode.DARK)
    assert raw.dtype == np.uint16 and raw.shape == (1024, 1024)
    assert 300 < dark.mean() < 450  # offset + fixed pattern, no beam
    assert raw.mean() > dark.mean() + 100  # the beam adds signal


def test_corrected_image_matches_flux_in_electrons(twin):
    request = twin.request(frame_time_s=0.025, total_frames=8)
    flux = twin.flux(request)
    image = twin.processor.acquire(request, units="electrons")
    expected = flux.mean() * 0.025 * 8
    assert image.mean() == pytest.approx(expected, rel=0.08)


def test_blanked_beam_gives_no_signal(twin):
    twin.column.set_beam_blank(True)
    try:
        image = twin.snap(0.2)
        assert abs(image.mean()) < 0.05
    finally:
        twin.column.set_beam_blank(False)


def test_stage_move_translates_the_image(twin):
    px_nm = twin.optics(twin.request()).specimen_pixel_nm
    ref = twin.flux(twin.request())
    x0, y0 = twin.column.state().stage.x_um, twin.column.state().stage.y_um
    dx_px = 40
    twin.column.move_stage(x=x0 + dx_px * px_nm / 1000.0)
    try:
        moved = twin.flux(twin.request())
        sx, sy = phase_correlation_shift(moved, ref)
        assert abs(abs(sx) - dx_px) <= 2 and abs(sy) <= 2
    finally:
        twin.column.move_stage(x=x0, y=y0)


def test_defocus_changes_the_image(twin):
    ref = twin.flux(twin.request())
    twin.column.set_defocus_um(-3.0)
    try:
        df = twin.flux(twin.request())
    finally:
        twin.column.set_defocus_um(0.0)
    assert not np.allclose(ref, df, rtol=0.02)


def test_diffraction_mode_centres_the_direct_beam(twin):
    twin.column.set_projection(Projection.DIFFRACTION)
    try:
        dp = twin.flux(twin.request())
        cy, cx = np.unravel_index(np.argmax(dp), dp.shape)
        assert abs(cy - 512) < 20 and abs(cx - 512) < 20
    finally:
        twin.column.set_projection(Projection.IMAGING)


def test_stem_frames_follow_the_scan(twin):
    twin.column.set_tem_stem(TemStem.STEM)
    try:
        request = twin.request(
            frame_time_s=0.001, total_frames=4, scan=ScanRequest(enabled=True, size=(2, 2))
        )
        points = [meta.scan_point for _, meta in twin.frames(request)]
        assert points == [(0, 0), (1, 0), (0, 1), (1, 1)]
        vi = twin.virtual_image(twin.request(scan=ScanRequest(enabled=True, size=(32, 32))), 40, 150)
        assert vi.shape == (32, 32) and np.isfinite(vi).all()
    finally:
        twin.column.set_tem_stem(TemStem.TEM)  # like the JEOL Dummy this lands in LowMAG
        twin.column.set("Magnification", 20000)


def test_gain_reference_is_measured_not_ground_truth(twin):
    from scipy.ndimage import gaussian_filter

    request = twin.request(frame_time_s=0.025)
    refs = twin.processor.take_gain(request, n_frames=5)
    truth = twin.detector.expected_gain(request)
    truth = truth / truth.mean()
    good = ~twin.detector.bad_pixel_mask
    # Per pixel, a 5-frame reference is dominated by shot noise (a real, under-dosed
    # gain reference) ...
    r_pix = np.corrcoef(refs.gain[good], truth[good])[0, 1]
    assert r_pix < 0.5
    # ... but the low-frequency gain structure (radial fall-off, stitch blocks) is measured.
    g = gaussian_filter(np.where(good, refs.gain, 1.0), 16)
    t = gaussian_filter(np.where(good, truth, 1.0), 16)
    assert np.corrcoef(g.ravel(), t.ravel())[0, 1] > 0.8


def test_in_situ_heating_crystallises_the_specimen():
    clock = ManualClock()
    tw = DigitalTwin("In Situ Heating (Nanoparticles)", camera="DESim", clock=clock, holder="sim-heating")
    tw.specimen.update(0.0, tw.holder_state(0.0))
    gid = np.arange(5 * 512, 6 * 512)  # gold grains
    before = tw.specimen.crystallinity(gid).mean()
    tw.holder.set(650.0)
    for t in np.arange(1.0, 400.0, 1.0):
        clock.set(t)
        tw.specimen.update(t, tw.holder_state(t))
    after = tw.specimen.crystallinity(gid).mean()
    assert tw.holder_state().temperature_c == pytest.approx(650, abs=5)
    assert after > before + 0.3


def test_repeated_acquisitions_do_not_repeat_the_noise():
    """Live view is a stream of one-frame acquisitions; each must get fresh noise, and a fresh
    twin must still reproduce the same sequence."""

    def two_frames():
        tw = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(), seed=1)
        request = tw.request(frame_time_s=0.025, total_frames=1)
        return [next(tw.frames(request))[0].astype(np.int64) for _ in range(2)]

    a, b = two_frames()
    assert np.abs(a - b).mean() > 1.0  # different shot and read noise
    a2, b2 = two_frames()
    assert np.array_equal(a, a2) and np.array_equal(b, b2)  # still deterministic
