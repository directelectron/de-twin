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


def _grating(tw, lines_per_mm: float = 2160.0) -> tuple[float, float]:
    """(pixel size nm, first-order angle deg) from a cross-grating image, as Find Pixel Size
    does: a spread beam, a Hann window against edge streaks, the first order searched near
    where the nominal pixel puts it, and its centroid in a 4x zero-padded FFT."""
    tw.column.set("Intensity", 0.95)
    img = _img(tw)
    ny, nx = img.shape
    win = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    n = 4 * max(ny, nx)
    F = np.abs(np.fft.fft2((img - img.mean()) * win, s=(n, n)))
    fy = np.broadcast_to(np.fft.fftfreq(n)[:, None] * n, (n, n))
    fx = np.broadcast_to(np.fft.fftfreq(n)[None, :] * n, (n, n))
    r = np.hypot(fx, fy)
    expect = n * tw.calibration_truth()["nominal_pixel_nm"] / (1e6 / lines_per_mm)
    G = np.where((r > 0.85 * expect) & (r < 1.15 * expect) & (fy >= 0), F, 0.0)
    ky, kx = np.unravel_index(np.argmax(G), G.shape)
    rows = (ky + np.arange(-3, 4)) % n
    cols = (kx + np.arange(-3, 4)) % n
    w = F[np.ix_(rows, cols)] ** 2
    gy = np.fft.fftfreq(n)[rows] * n
    gx = np.fft.fftfreq(n)[cols] * n
    cy = float((gy[:, None] * w).sum() / w.sum())
    cx = float((gx[None, :] * w).sum() / w.sum())
    return (1e6 / lines_per_mm) / (n / math.hypot(cx, cy)), math.degrees(math.atan2(cy, cx))


def _xcorr_shift(a: np.ndarray, b: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """(dx, dy) px of b against a from a plain (unwhitened) cross-correlation of low-passed
    images, parabola-refined: follows the large features when the fine contrast differs."""
    from scipy.ndimage import gaussian_filter

    a = gaussian_filter(a, sigma)
    b = gaussian_filter(b, sigma)
    c = np.fft.ifft2(np.fft.fft2(b - b.mean()) * np.conj(np.fft.fft2(a - a.mean()))).real
    ny, nx = c.shape
    iy, ix = np.unravel_index(np.argmax(c), c.shape)

    def par(m, z, p):
        return 0.5 * (m - p) / (m - 2 * z + p)

    dy = par(c[iy - 1, ix], c[iy, ix], c[(iy + 1) % ny, ix])
    dx = par(c[iy, ix - 1], c[iy, ix], c[iy, (ix + 1) % nx])
    return np.array([(ix - nx if ix > nx // 2 else ix) + dx, (iy - ny if iy > ny // 2 else iy) + dy])


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
    measured_nm, _ = _grating(tw)
    assert measured_nm == pytest.approx(truth["true_pixel_nm"], rel=0.005)
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


# ------------------------------------------------------------------ focus and coma (phase 3)
def _camera_vec(truth: dict, world_nm) -> np.ndarray:
    """A specimen-plane displacement (nm) as a camera-pixel vector, in the camera's rotated
    frame (sign as the image moves with the specimen feature)."""
    return -_predicted_px(truth, (world_nm[0] / 1000.0, world_nm[1] / 1000.0))


def _tilt_pair_shift(tw, tau_mrad: float) -> np.ndarray:
    tw.column.set_beam_tilt_mrad(tau_mrad, 0.0)
    a = _img(tw)
    tw.column.set_beam_tilt_mrad(-tau_mrad, 0.0)
    b = _img(tw)
    tw.column.set_beam_tilt_mrad(0.0, 0.0)
    return _shift_px(a, b)


def test_autofocus_calibration_tilt_pairs_measure_the_defocus():
    """SerialEM's autofocus: the image displacement between +tau and -tau beam tilt is
    2 * defocus * tau, so a calibration of it reads the defocus back."""
    tw = _twin()
    px_nm = tw.calibration_truth()["true_pixel_nm"]
    tau = 3.0  # mrad
    shifts = {}
    for df in (-0.5, -1.0, -2.0):
        tw.column.set_defocus_um(df)
        shifts[df] = _tilt_pair_shift(tw, tau)
    for df, s in shifts.items():
        want_px = 2.0 * abs(df) * 1000.0 * tau * 1e-3 / px_nm
        assert np.hypot(*s) == pytest.approx(want_px, rel=0.05, abs=0.3)
    # linear in defocus, and along one line (the tilt direction in the camera frame)
    ratio = np.hypot(*shifts[-2.0]) / np.hypot(*shifts[-1.0])
    assert ratio == pytest.approx(2.0, rel=0.05)
    u, v = shifts[-1.0] / np.hypot(*shifts[-1.0]), shifts[-2.0] / np.hypot(*shifts[-2.0])
    assert abs(float(u @ v)) > 0.99


def test_image_shift_brings_coma_which_the_coma_vs_is_calibration_measures():
    """Off the coma-free axis a defocus change moves the image by (defocus change) * (the
    effective tilt of the image shift): SerialEM's ComaVsIS measures that tilt."""
    tw = _twin()
    truth = tw.calibration_truth()
    K = np.asarray(truth["is_coma_mrad_per_is_um"]).T  # mrad per specimen um of image shift
    M = np.asarray(truth["is_matrix_um_per_unit"])

    tw.column.set("Intensity", 0.95)  # a spread beam: no disc edge in the field

    def focus_step(is_units):
        # a small focus step: the fine contrast changes, so follow the large features
        tw.column.set("ImageShift", is_units)
        tw.column.set_defocus_um(-0.5)
        a = _img(tw)
        tw.column.set_defocus_um(-1.5)
        b = _img(tw)
        tw.column.set_defocus_um(0.0)
        tw.column.set("ImageShift", (0.0, 0.0))
        return _xcorr_shift(a, b)

    assert np.abs(focus_step((0.0, 0.0))).max() < 0.1, "on axis, focus does not move the image"
    is_units = (4.0, 0.0)
    tau = K @ (M @ is_units)  # mrad
    s = focus_step(is_units)
    want_px = 1000.0 * np.hypot(*tau) * 1e-3 / truth["true_pixel_nm"]  # 1 um of defocus change
    assert np.hypot(*s) == pytest.approx(want_px, rel=0.15)
    d = _camera_vec(truth, tau)  # along the induced tilt, in the camera frame
    assert abs(float(s @ d)) / (np.hypot(*s) * np.hypot(*d)) > 0.95


def test_high_defocus_changes_magnification_and_rotation():
    tw = _twin("Cross grating 2160 l/mm", mag=2000.0)
    p0, a0 = _grating(tw)
    t0 = tw.calibration_truth()
    tw.column.set_defocus_um(-200.0)
    p1, a1 = _grating(tw)
    t1 = tw.calibration_truth()
    assert t1["true_pixel_nm"] / t0["true_pixel_nm"] == pytest.approx(1.04, rel=1e-6)
    # the defocus envelope falls steeply across the first order at -200 um and skews its
    # centroid a little: ~1 %, as SerialEM's own high-defocus calibration
    assert p1 / p0 == pytest.approx(t1["true_pixel_nm"] / t0["true_pixel_nm"], rel=0.01)
    assert p1 / p0 > 1.025, "the 4 % change is measured"
    turn = (a1 - a0 + 45.0) % 90.0 - 45.0  # a square grating: its orders repeat every 90 deg
    want = (t1["image_rotation_deg"] - t0["image_rotation_deg"] + 45.0) % 90.0 - 45.0
    # image features turn by MINUS the reported rotation (world -> camera, rows down)
    assert abs(want) > 1.0 and turn == pytest.approx(-want, abs=0.3)


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
