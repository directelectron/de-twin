"""SerialEM's geometric calibrations, run against a realistic twin and checked against its truth.

Each test does what SerialEM does (images, cross-correlation, an FFT of a grating) and compares
the answer with `DigitalTwin.calibration_truth`. The measuring code is test-only on purpose:
de-twin makes the data and the truth, the calibration methods live elsewhere.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.optics import OpticsConfig
from de_twin.twin import DigitalTwin


def _twin(name="Dense Au on holey C", realistic=True, seed=7, mag=20000.0):
    cfg = OpticsConfig.realistic(seed=seed) if realistic else OpticsConfig()
    tw = DigitalTwin(name, camera="DESim", clock=ManualClock(), seed=1, optics_config=cfg)
    tw.column.set("Magnification", mag)
    return tw


def _shift_px(a: np.ndarray, b: np.ndarray, up: int = 20) -> np.ndarray:
    """(dx, dy) in pixels that b is displaced by relative to a: phase correlation, refined by
    a local upsampled DFT (Guizar-Sicairos)."""
    A = np.fft.fft2(a - a.mean())
    B = np.fft.fft2(b - b.mean())
    R = B * np.conj(A)
    R /= np.maximum(np.abs(R), 1e-12)
    c = np.fft.ifft2(R).real
    ny, nx = c.shape
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    y0 = iy - ny if iy > ny // 2 else iy
    x0 = ix - nx if ix > nx // 2 else ix
    # upsampled DFT around the coarse peak
    span = 1.5
    ys = y0 + np.linspace(-span, span, int(2 * span * up) + 1)
    xs = x0 + np.linspace(-span, span, int(2 * span * up) + 1)
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    Ey = np.exp(2j * np.pi * ys[:, None] * ky.T)  # (len ys, ny)
    Ex = np.exp(2j * np.pi * kx.T * xs[None, :])  # (nx, len xs)
    fine = (Ey @ R @ Ex).real
    j, i = np.unravel_index(np.argmax(fine), fine.shape)
    return np.array([xs[i], ys[j]])


def _img(tw, req=None):
    return tw.flux(req or tw.request()).astype(np.float64)


def _predicted_px(truth: dict, world_um) -> np.ndarray:
    """Pixel displacement of the image when the view centre moves by *world_um* (x, y):
    features move the other way, in the camera's rotated frame, over the true pixel."""
    rot = math.radians(truth["image_rotation_deg"])
    c, s = math.cos(rot), math.sin(rot)
    wx, wy = world_um
    u = c * wx + s * wy
    v = -s * wx + c * wy
    return -np.array([u, v]) * 1000.0 / truth["true_pixel_nm"]


def test_the_ideal_twin_reports_an_ideal_column():
    t = _twin(realistic=False).calibration_truth()
    assert t["image_rotation_deg"] == 0.0 and t["true_pixel_nm"] == t["nominal_pixel_nm"]
    assert t["is_matrix_um_per_unit"] == [[1.0, 0.0], [0.0, 1.0]] and t["mag_offset_um"] == (0.0, 0.0)
    assert t["stage_error_um"] == (0.0, 0.0)


def test_find_pixel_size_on_a_cross_grating():
    tw = _twin("Cross grating 2160 l/mm", mag=2000.0)
    truth = tw.calibration_truth()
    img = _img(tw)
    n = 4096  # zero-padded FFT: sub-bin period
    F = np.abs(np.fft.fft2(img - img.mean(), s=(n, n)))
    # SerialEM looks for the grating's first order near where the nominal pixel puts it
    fy = np.fft.fftfreq(n)[:, None] * n
    fx = np.fft.fftfreq(n)[None, :] * n
    r = np.hypot(fx, fy)
    expect = n * truth["nominal_pixel_nm"] / (1e6 / 2160.0)
    F[(r < 0.8 * expect) | (r > 1.2 * expect)] = 0
    ky, kx = np.unravel_index(np.argmax(F), F.shape)
    win = np.s_[ky - 3:ky + 4, kx - 3:kx + 4]  # centroid of the first-order peak
    w = F[win] ** 2
    k = float((r[win] * w).sum() / w.sum())
    period_px = n / k
    measured_nm = 1e6 / 2160.0 / period_px
    assert measured_nm == pytest.approx(truth["true_pixel_nm"], rel=0.01)
    assert abs(measured_nm - truth["true_pixel_nm"]) < 0.5 * abs(truth["nominal_pixel_nm"] - truth["true_pixel_nm"])
    assert abs(truth["true_pixel_nm"] / truth["nominal_pixel_nm"] - 1) > 0.002, "the calibration had something to find"


def test_image_shift_calibration_recovers_the_matrix_and_rotation():
    tw = _twin()
    truth = tw.calibration_truth()
    a = _img(tw)
    measured = []
    for ex, ey in ((0.2, 0.0), (0.0, 0.2)):
        tw.column.set("ImageShift", (ex, ey))
        measured.append(_shift_px(a, _img(tw)))
        tw.column.set("ImageShift", (0.0, 0.0))
    M = np.asarray(truth["is_matrix_um_per_unit"])
    for (ex, ey), got in zip(((0.2, 0.0), (0.0, 0.2)), measured):
        want = _predicted_px(truth, M @ (ex, ey))
        assert got == pytest.approx(want, abs=0.25)


def test_stage_calibration_sees_the_image_rotation():
    tw = _twin()
    truth = tw.calibration_truth()
    a = _img(tw)
    s = tw.column.state().stage
    tw.column.move_stage(x=s.x_um + 0.3)
    got = _shift_px(a, _img(tw))
    want = _predicted_px(truth, (-0.3, 0.0))  # the view centre moves against the stage
    assert got == pytest.approx(want, abs=0.25)
    angle = math.degrees(math.atan2(got[1], got[0]))
    assert abs(truth["image_rotation_deg"]) > 1.0, "a rotation to calibrate"
    assert angle == pytest.approx(math.degrees(math.atan2(want[1], want[0])), abs=1.0)


def test_each_magnification_has_its_own_rotation_scale_and_offset():
    tw = _twin()
    seen = []
    for mag in (15000.0, 20000.0, 25000.0):
        tw.column.set("Magnification", mag)
        t = tw.calibration_truth()
        seen.append((round(t["image_rotation_deg"], 3), round(t["true_pixel_nm"] / t["nominal_pixel_nm"], 5),
                     t["mag_offset_um"]))
    assert len({s[0] for s in seen}) == 3 and len({s[1] for s in seen}) == 3
    # and the offset is where the view centre goes
    from de_twin.optics.derive import view_center_um

    st = tw.column.state()
    ideal = view_center_um(st, OpticsConfig())
    real = view_center_um(st, tw.optics_config)
    M = np.asarray(tw.calibration_truth()["is_matrix_um_per_unit"])
    assert np.subtract(real, ideal) == pytest.approx(tw.calibration_truth()["mag_offset_um"], abs=1e-9)
    assert M.shape == (2, 2)


def test_backlash_depends_on_the_approach_direction():
    tw = _twin()
    s = tw.column.state().stage
    x = s.x_um
    tw.column.move_stage(x=x - 1.0)
    tw.column.move_stage(x=x)  # approached from below
    a = _img(tw)
    ea = tw.calibration_truth()["stage_error_um"]
    tw.column.move_stage(x=x + 1.0)
    tw.column.move_stage(x=x)  # approached from above
    b = _img(tw)
    eb = tw.calibration_truth()["stage_error_um"]
    backlash = tw.optics_config.realism.backlash_um
    assert eb[0] - ea[0] == pytest.approx(backlash)
    got = _shift_px(a, _img(tw))
    want = _predicted_px(tw.calibration_truth(), (-(eb[0] - ea[0]), 0.0))
    assert got == pytest.approx(want, abs=0.3)
    assert tw.column.state().stage.x_um == x, "the reported position is the commanded one"


def test_a_tilted_specimen_off_eucentric_height_moves():
    tw = _twin(realistic=False)
    tw.column.move_stage(z=0.1)  # 0.1 um above eucentric
    tw.column.move_stage(alpha=-20.0)
    a = _img(tw)
    tw.column.move_stage(alpha=20.0)
    got = _shift_px(a, _img(tw))
    # the image moves by dz sin(tilt) (not dz sin cos: 6 % less at 20 deg)
    dy_um = 0.1 * (math.sin(math.radians(20.0)) - math.sin(math.radians(-20.0)))
    px = tw.calibration_truth()["true_pixel_nm"]
    assert abs(got[1]) == pytest.approx(dy_um * 1000.0 / px, rel=0.02)
    assert abs(got[0]) < 0.5, "across the tilt axis only"
    tw.column.move_stage(z=0.0)
    tw.column.move_stage(alpha=-2.0)
    c = _img(tw)
    tw.column.move_stage(alpha=2.0)
    assert np.abs(_shift_px(c, _img(tw))).max() < 0.5, "at eucentric height tilting does not move it"


# ------------------------------------------------------------------ illumination (phase 2)
def _disc(img: np.ndarray):
    """Centroid (x, y) and equivalent diameter (px) of the illuminated disc: pixels brighter
    than half the brightest."""
    m = img > 0.5 * np.percentile(img, 99.5)
    y, x = np.nonzero(m)
    return np.array([x.mean(), y.mean()]), 2.0 * math.sqrt(m.sum() / math.pi)


def test_beam_crossover_is_where_the_beam_is_smallest():
    tw = _twin("Cross grating 2160 l/mm", mag=2000.0)
    x0 = tw.calibration_truth()["crossover_intensity"]
    xs = np.round(np.linspace(x0 - 0.03, x0 + 0.03, 13), 4)
    peak = []
    for x in xs:
        tw.column.set("Intensity", float(x))
        peak.append(float(np.percentile(_img(tw), 99.9)))
    best = xs[int(np.argmax(peak))]
    assert best == pytest.approx(x0, abs=0.006), "SerialEM's Beam Crossover finds it"
    # and the beam spreads on both sides of it
    assert peak[0] < max(peak) and peak[-1] < max(peak)


def test_beam_shift_calibration_follows_the_disc():
    tw = _twin("Cross grating 2160 l/mm", mag=2000.0)
    truth = tw.calibration_truth()
    x0 = truth["crossover_intensity"]
    tw.column.set("Intensity", x0 + 0.012)  # a ~1.5 um disc inside a 5 um field
    a = _img(tw)
    ca, da = _disc(a)
    assert 50 < da < 900, "the disc is inside the field"
    B = np.asarray(truth["bs_matrix_um_per_unit"])
    for e in ((0.5, 0.0), (0.0, 0.5)):
        tw.column.set("BeamShift", e)
        cb, _ = _disc(_img(tw))
        tw.column.set("BeamShift", (0.0, 0.0))
        want = -_predicted_px(truth, B @ e)  # the disc moves WITH the beam
        assert cb - ca == pytest.approx(want, abs=2.0)


def test_beam_shift_moves_the_beam_not_the_image():
    tw = _twin(realistic=False, mag=20000.0)
    a = _img(tw)
    tw.column.set("BeamShift", (0.01, 0.0))
    b = _img(tw)
    # the specimen does not move (only the illumination edge, far outside this field)
    assert np.abs(_shift_px(a, b)).max() < 0.1


def test_texture_moves_with_the_specimen_when_the_view_is_rotated():
    """A texture-dominated specimen (no particles to lean on): an image-shift or stage move
    that re-renders must move the carbon texture the way the specimen moves."""
    tw = _twin("Negative stain on carbon", mag=100000.0)
    tw.column.set("Intensity", 0.95)
    truth = tw.calibration_truth()
    assert abs(truth["image_rotation_deg"]) > 1.0
    a = _img(tw)
    s = tw.column.state().stage
    tw.column.move_stage(x=s.x_um + 0.02)  # beyond the pan margin: a full re-render
    got = _shift_px(a, _img(tw))
    want = _predicted_px(truth, (-0.02, 0.0))
    assert got == pytest.approx(want, abs=1.0)
