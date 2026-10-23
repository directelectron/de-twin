"""Detector model: catalogue facts, noise statistics, references, geometry, counting."""

from __future__ import annotations

import time

import numpy as np
import pytest

from de_twin.detector import CAMERAS, Detector, camera
from de_twin.state import AcquisitionRequest, Roi
from perf_budget import budget


def req(**kw) -> AcquisitionRequest:
    return AcquisitionRequest(**kw)


# ------------------------------------------------------------------ catalogue
def test_catalogue_facts():
    assert camera("DESim").sensor_shape == (1024, 1024)
    for n in ("DE16", "Vision16", "LV16", "DirectView2", "SEMCam", "Meridian"):
        assert camera(n).sensor_shape == (4096, 4096)
        assert camera(n).bit_depth == 12 and camera(n).bytes_per_pixel == 2
    for n in ("DE64", "Vision64"):
        assert camera(n).sensor_shape == (8192, 8192) and camera(n).num_parts == 2
    for n in ("Celeritas", "CeleritasXS", "Zenith"):
        m = camera(n)
        assert m.sensor_shape == (1024, 1024) and m.topology == "topbottom" and m.num_parts == 2
    for n in ("Apollo", "ApolloXS", "Artemis", "Centuri", "CenturiPlus"):
        m = camera(n)
        assert m.hardware_counting and m.bytes_per_pixel == 1 and m.dtype == np.uint8
        assert m.sensor_shape == (4096, 4096)
    assert camera("DE16").saturation_adu == 2192
    assert camera("DE64").saturation_adu == 2600
    assert camera("Zenith").saturation_adu == 4061
    assert camera("DE16").max_fps == pytest.approx(1 / 10.821e-3)


def test_lookup_aliases():
    assert camera("SIMULATOR") is CAMERAS["DESim"]
    assert camera("de-16") is CAMERAS["DE16"]
    assert camera("DIRECTVIEW2") is CAMERAS["DirectView2"]
    assert camera("CENTURIPLUS") is CAMERAS["CenturiPlus"]
    with pytest.raises(KeyError):
        camera("NoSuchCam")


def test_adu_per_electron_ht_table():
    m = camera("DE16")
    assert m.adu_per_electron_at(300) == pytest.approx(206)
    assert m.adu_per_electron_at(200) == pytest.approx(316)
    assert 206 < m.adu_per_electron_at(250) < 316
    assert m.adu_per_electron_at(60) > 1379  # extrapolated, more deposit at low HT
    assert m.charge_spread_at(120) > m.charge_spread_at(300)


def test_low_gain_mode_fills_the_adc_with_a_hundred_electrons():
    high = camera("DESim")
    low = high.with_(gain_mode="low")
    assert high.gain_mode == "high"
    span = high.max_value - high.dark_offset_adu
    for kv in (20.0, 300.0):  # the full well in electrons, whatever the energy
        assert low.adu_per_electron_at(kv) == pytest.approx(span / 100.0)
    assert high.adu_per_electron_at(20.0) > 20 * low.adu_per_electron_at(20.0)
    with pytest.raises(ValueError):
        high.with_(gain_mode="medium").adu_per_electron_at(20.0)

    # a pixel saturates near 100 electrons a frame; the per-electron spread is the high gain's
    assert low.adu_cv == high.adu_cv and low.read_noise_adu == high.read_noise_adu
    d = Detector("DESim", seed=3, ht_kv=20.0)
    d.set_gain_mode("low")
    t = 1e-3
    shape = d.output_shape(req())
    for electrons, expected in ((50.0, 0.5), (120.0, 1.0)):
        frame, _ = d.expose(np.full(shape, electrons / t, np.float32), t, req(), 0)
        level = (np.median(frame) - high.dark_offset_adu) / span
        assert level == pytest.approx(expected, abs=0.08)
    one, _ = d.expose(np.full(shape, 1.0 / t, np.float32), t, req(), 1)  # ~1 electron a pixel
    deposit = one.astype(np.float64) - d.expected_dark(t)
    assert deposit.mean() == pytest.approx(span / 100.0, rel=0.05)
    assert deposit.std() > 2 * high.read_noise_adu  # each electron seen, not lost in the read noise
    d.set_gain_mode("high")
    assert d.model.gain_mode == "high"


def test_overrides():
    d = Detector("DESim", read_noise_adu=3.0)
    assert d.model.read_noise_adu == 3.0
    with pytest.raises(TypeError):
        Detector("DESim", nonsense=1)


# ------------------------------------------------------------------ geometry
def test_roi_and_output_shape():
    d = Detector("DE16")
    assert d.output_shape(req()) == (4096, 4096)
    r = req(hw_roi=Roi(100, 200, 256, 128))
    assert d.roi(r) == Roi(100, 200, 256, 128)
    assert d.output_shape(r) == (128, 256)
    # clamped to the sensor
    assert d.roi(req(hw_roi=Roi(4000, -5, 1000, 50))) == Roi(4000, 0, 96, 50)
    # zero size means full
    assert d.roi(req(hw_roi=Roi(0, 0, 0, 0))) == Roi(0, 0, 4096, 4096)
    # binning after ROI (integer division)
    assert d.output_shape(req(hw_roi=Roi(0, 0, 1025, 512), hw_binning=(2, 2))) == (256, 512)


def test_roi_frame_matches_expected_dark_slice():
    d = Detector("DE16", seed=3)
    r = req(hw_roi=Roi(512, 1024, 256, 128))
    frames = np.stack([d.expose(None, 0.01, r, i)[0] for i in range(16)]).astype(float)
    exp = d.expected_dark(0.01, r)
    good = ~d.bad_pixel_mask[1024:1152, 512:768]
    assert frames.shape[1:] == (128, 256)
    assert np.abs(frames.mean(0) - exp)[good].mean() < 3.0


# ------------------------------------------------------------------ dark
def test_dark_frame_level_and_read_noise():
    d = Detector("DESim", seed=1)
    f0, info = d.expose(None, 0.01, req(), 0)
    f1, _ = d.expose(None, 0.01, req(), 1)
    assert f0.dtype == np.uint16 and f0.shape == (1024, 1024)
    assert info["blanked"] and info["dose_e_per_px"] == 0.0
    good = ~d.bad_pixel_mask
    exp = d.expected_dark(0.01)
    # level: offset 360 + fixed pattern
    assert abs(f0[good].mean() - exp[good].mean()) < 0.5
    assert 360 < f0[good].mean() < 400
    # temporal noise: difference of two dark frames
    diff = f0.astype(float) - f1.astype(float)
    assert diff[good].std() / np.sqrt(2) == pytest.approx(d.model.read_noise_adu, rel=0.05)
    # fixed pattern reproduces GrabberSim's (x%8)(y%8) structure
    tile = exp[:64, 64:128]
    assert tile[7, 7] - tile[0, 0] == pytest.approx(49, abs=15)


def test_hot_pixels_visible_in_dark_and_scale_with_exposure():
    d = Detector("DE16", seed=2, hot_pixel_fraction=1e-4)
    ys, xs, _ = d.hot_pixels
    y, x = ys[0], xs[0]
    r = req(hw_roi=Roi(int(x) - 8 if x >= 8 else 0, int(y) - 8 if y >= 8 else 0, 16, 16))
    short = d.expose(None, 0.001, r, 0)[0].astype(float)
    long = d.expose(None, 0.05, r, 0)[0].astype(float)
    hy, hx = int(y) - d.roi(r).y, int(x) - d.roi(r).x
    assert long[hy, hx] - short[hy, hx] > 500
    assert d.bad_pixel_mask[y, x]


# ------------------------------------------------------------------ signal
def test_flat_flux_mean_signal():
    d = Detector("DESim", seed=4)
    dose = 3.0
    t = 0.01
    f, info = d.expose(np.full((1024, 1024), dose / t, np.float32), t, req(), 0)
    assert info["dose_e_per_px"] == pytest.approx(dose, rel=1e-6)
    assert not info["blanked"]
    good = ~d.bad_pixel_mask
    signal = f.astype(float) - d.expected_dark(t)
    expected = dose * d.model.adu_per_electron * d.gain_map
    assert signal[good].mean() == pytest.approx(expected[good].mean(), rel=0.01)


def test_ht_dependence():
    t = 0.01
    lo = Detector("DESim", seed=5, ht_kv=200)
    hi = Detector("DESim", seed=5, ht_kv=300)
    flux = np.full((1024, 1024), 2 / t, np.float32)
    s200 = lo.expose(flux, t)[0].astype(float).mean() - lo.expected_dark(t).mean()
    s300 = hi.expose(flux, t)[0].astype(float).mean() - hi.expected_dark(t).mean()
    assert s200 / s300 == pytest.approx(316 / 206, rel=0.03)


def test_single_electron_events_are_discrete_and_skewed():
    """At low dose the frame shows isolated electron clusters (counting on DE16 needs them)."""
    d = Detector("DESim", seed=6, read_noise_adu=0.0, dsnu_adu=0.0, fpn_column_adu=0.0,
                 fpn_pattern=False, gain_pixel_sigma=0, gain_column_sigma=0, gain_radial=0,
                 gain_stitch=0, hot_pixel_fraction=0, dead_pixel_fraction=0,
                 bad_column_fraction=0, dark_current_adu_per_s=0)
    t = 0.01
    f = d.expose(np.full((1024, 1024), 0.01 / t, np.float32), t)[0].astype(float) - 360
    lit = f > 30
    assert 0.005 < lit.mean() < 0.1  # ~1% of pixels (plus neighbours) carry charge
    vals = f[f > 60]
    skew = ((vals - vals.mean()) ** 3).mean() / vals.std() ** 3
    assert skew > 0.3  # Landau-like tail


def test_references_flatten_the_image():
    """Dark-subtract and gain-normalise with twin-acquired references => flat."""
    d = Detector("DESim", seed=7)
    t = 0.01
    flux = np.full((1024, 1024), 12 / t, np.float32)
    dark = np.mean([d.expose(None, t, req(), i)[0] for i in range(10)], axis=0)
    flat = np.mean([d.expose(flux, t, req(), 100 + i)[0] for i in range(24)], axis=0)
    img = np.mean([d.expose(flux, t, req(), 200 + i)[0] for i in range(24)], axis=0)
    good = ~d.bad_pixel_mask & (d.gain_map > 0.5)
    gain = flat - dark
    corrected = np.where(good, (img - dark) / np.where(good, gain, 1) * gain[good].mean(), 0)
    raw = img - dark

    def blocks(a):  # 32x32 block means: shot noise averages away, gain structure remains
        m = np.where(good, a, np.nan).reshape(32, 32, 32, 32)
        return np.nanmean(m, axis=(1, 3))

    raw_var = np.std(blocks(raw)) / np.mean(blocks(raw))
    cor_var = np.std(blocks(corrected)) / np.mean(blocks(corrected))
    assert raw_var > 0.01  # the raw image shows the gain structure
    assert cor_var < raw_var / 2  # references remove it
    # the gain reference recovers the true gain structure
    truth = blocks(d.gain_map)
    assert np.corrcoef(blocks(gain).ravel(), truth.ravel())[0, 1] > 0.95
    # and the dark reference matches the truth
    assert np.abs(dark - d.expected_dark(t))[good].mean() < 3.0


# ------------------------------------------------------------------ binning
def test_binning_shape_and_values():
    d = Detector("DESim", seed=8)
    t = 0.01
    r1 = req()
    r2 = req(hw_binning=(2, 2))
    flux = np.full((1024, 1024), 2 / t, np.float32)
    f1 = d.expose(flux, t, r1, 0)[0].astype(float)
    f2 = d.expose(flux, t, r2, 0)[0].astype(float)
    assert f2.shape == (512, 512)
    assert f2.mean() == pytest.approx(f1.mean(), rel=0.01)
    # binned dark: average of 4 pixels -> read noise / 2
    a = d.expose(None, t, r2, 0)[0].astype(float)
    b = d.expose(None, t, r2, 1)[0].astype(float)
    good = ~d.bad_pixel_mask.reshape(512, 2, 512, 2).any(axis=(1, 3))
    assert (a - b)[good].std() / np.sqrt(2) == pytest.approx(d.model.read_noise_adu / 2, rel=0.08)
    assert np.abs(a - d.expected_dark(t, r2))[good].mean() < 5


def test_flux_at_binned_shape_uses_direct_path():
    d = Detector("DESim", seed=9)
    t = 0.01
    r2 = req(hw_binning=(2, 2))
    full = d.expose(np.full((1024, 1024), 2 / t, np.float32), t, r2, 0)[0].astype(float)
    direct, info = d.expose(np.full((512, 512), 2 / t, np.float32), t, r2, 0)
    assert direct.shape == (512, 512)
    assert info["dose_e_per_px"] == pytest.approx(2.0)
    assert direct.astype(float).mean() == pytest.approx(full.mean(), rel=0.01)
    with pytest.raises(ValueError):
        d.expose(np.ones((100, 100), np.float32), t, r2, 0)


# ------------------------------------------------------------------ saturation
def test_saturation_clipping_and_bit_depth():
    d = Detector("DESim", seed=10)
    t = 0.01
    f, info = d.expose(np.full((1024, 1024), 1e5, np.float32), t, req(), 0)
    good = d.gain_map > 0.5
    assert info["saturated"] and info["saturated_pixels"] > 0.99 * good.sum()
    assert f.max() == 4095
    assert (f[good] == 4095).mean() > 0.999
    f10, _ = d.expose(np.full((1024, 1024), 1e5, np.float32), t, req(bit_depth=10), 0)
    assert f10.max() == 1023
    dark10, _ = d.expose(None, t, req(bit_depth=10), 0)
    assert dark10[~d.bad_pixel_mask].mean() == pytest.approx(d.expected_dark(t).mean() / 4, rel=0.02)


# ------------------------------------------------------------------ counting
def test_counting_camera_events_and_coincidence_loss():
    d = Detector("Apollo", seed=11)
    r = req(hw_roi=Roi(0, 1024, 4096, 512))
    t = 40 * d.model.full_frame_time_s  # 40 cycles per delivered frame
    low_dose = 0.5
    f, info = d.expose(np.full((512, 4096), low_dose / t, np.float32), t, r, 0)
    assert f.dtype == np.uint8 and f.shape == (512, 4096)
    assert info["counting"]
    eff_low = f.mean() / low_dose
    assert eff_low == pytest.approx(1.0, abs=0.05)
    high_dose = 20.0
    fh, _ = d.expose(np.full((512, 4096), high_dose / t, np.float32), t, r, 1)
    eff_high = fh.mean() / high_dose
    assert eff_high < 0.9 * eff_low  # coincidence loss
    assert fh.max() <= 40  # at most one count per pixel per cycle
    dark, dinfo = d.expose(None, t, r, 2)
    assert dinfo["blanked"] and dark.mean() < 0.01


# ------------------------------------------------------------------ determinism
def test_determinism():
    a = Detector("DESim", seed=12)
    b = Detector("DESim", seed=12)
    flux = np.full((1024, 1024), 300.0, np.float32)
    fa, _ = a.expose(flux, 0.01, req(), 5)
    fb, _ = b.expose(flux, 0.01, req(), 5)
    assert np.array_equal(fa, fb)
    fc, _ = a.expose(flux, 0.01, req(), 6)
    assert not np.array_equal(fa, fc)
    c = Detector("DESim", seed=13)
    assert not np.array_equal(a.dark_map, c.dark_map)
    assert np.array_equal(a.dark_map, b.dark_map)


def test_determinism_independent_of_thread_count(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import de_twin.detector.detector as D

    flux = np.full((1024, 1024), 300.0, np.float32)
    ref, _ = Detector("DESim", seed=14).expose(flux, 0.01, req(), 3)
    monkeypatch.setattr(D, "_POOL", ThreadPoolExecutor(1))
    one, _ = Detector("DESim", seed=14).expose(flux, 0.01, req(), 3)
    assert np.array_equal(ref, one)


# ------------------------------------------------------------------ performance
def test_performance_4k():
    d = Detector("DE16", seed=15)
    flux = np.full((4096, 4096), 200.0, np.float32)
    d.expose(flux, 0.01, req(), 0)  # warm up (maps, buffers)
    ts = []
    for i in range(3):
        t0 = time.perf_counter()
        f, _ = d.expose(flux, 0.01, req(), i + 1)
        ts.append(time.perf_counter() - t0)
    best = min(ts)
    print(f"DE16 4096^2 expose: best {best * 1e3:.0f} ms")
    assert f.shape == (4096, 4096)
    assert best < budget(0.6)  # target ~0.2 s on a multi-core desktop
