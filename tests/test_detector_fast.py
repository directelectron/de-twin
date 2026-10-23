"""The fused numba detector: the numpy path's physics, a fraction of its time."""

import numpy as np
import pytest

from de_twin.detector import Detector, camera, fast
from de_twin.state import AcquisitionRequest

pytestmark = pytest.mark.skipif(not fast.AVAILABLE, reason="numba is not installed")


@pytest.fixture
def numpy_path(monkeypatch):
    monkeypatch.setenv("DE_TWIN_NUMPY_DETECTOR", "1")


def _frames(name, flux, n=12, request=None, seed=4):
    d = Detector(camera(name), seed=seed)
    req = request or AcquisitionRequest(camera_model=name)
    return np.stack([d.expose(flux, 0.025, req, i)[0] for i in range(n)]).astype(np.float64), d


def _flux(shape, level=120.0):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (level * (1.0 + 0.5 * np.sin(xx / 17.0) * np.cos(yy / 23.0))).astype(np.float32)


@pytest.mark.parametrize("name", ["DESim", "Celeritas"])
def test_the_statistics_match_the_numpy_path(name, monkeypatch):
    shape = camera(name).sensor_shape
    flux = _flux(shape)
    fast_frames, _ = _frames(name, flux)
    monkeypatch.setenv("DE_TWIN_NUMPY_DETECTOR", "1")
    ref, _ = _frames(name, flux)
    assert fast_frames.mean() == pytest.approx(ref.mean(), rel=0.01)
    # temporal variance per pixel (shot + gain + read noise) and the image itself
    assert fast_frames.var(axis=0).mean() == pytest.approx(ref.var(axis=0).mean(), rel=0.03)
    np.testing.assert_allclose(
        fast_frames.mean(axis=0).mean(axis=0), ref.mean(axis=0).mean(axis=0), rtol=0.03
    )


def test_a_seed_reproduces_and_frames_differ():
    flux = _flux((1024, 1024))
    a, _ = _frames("DESim", flux, n=2)
    b, _ = _frames("DESim", flux, n=2)
    assert np.array_equal(a, b), "same seed, same frames — whatever the thread schedule"
    assert np.abs(a[0] - a[1]).mean() > 1.0, "a new frame, new noise"


def test_binned_and_dark_frames():
    name = "DESim"
    flux = _flux(camera(name).sensor_shape)
    req = AcquisitionRequest(camera_model=name, hw_binning=(2, 2))
    d = Detector(camera(name), seed=1)
    image, info = d.expose(flux, 0.025, req, 0)
    assert image.shape == (512, 512) and info["dose_e_per_px"] == pytest.approx(flux.mean() * 0.025, rel=0.01)
    dark, info = d.expose(None, 0.025, AcquisitionRequest(camera_model=name), 0)
    m = camera(name)
    assert abs(float(np.median(dark)) - m.dark_offset_adu) < 20 and info["dose_e_per_px"] == 0.0


def test_saturation_is_counted():
    name = "DESim"
    d = Detector(camera(name), seed=1)
    image, info = d.expose(
        np.full(camera(name).sensor_shape, 1e5, np.float32), 0.025, AcquisitionRequest(camera_model=name), 0
    )
    assert info["saturated"] and info["saturated_pixels"] > 0.9 * image.size
    assert int(image.max()) == camera(name).max_value
