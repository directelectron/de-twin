"""EBSD camera frames made by the fused CPU kernel (de_twin.ebsd.frames, ebsd.framesource):
the general path's patterns and noise, and frames published through the shared-memory face.

The general path (``render_ebsd_frame`` + ``Detector.expose``) is the reference;
``DE_TWIN_FUSED_EBSD=0`` selects it."""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

from de_twin import DigitalTwin
from de_twin.clock import ManualClock
from de_twin.state import ScanRequest

KV = 20.0
DWELL = 0.125e-3


def _twin(specimen="EBSD duplex steel", seed=1, **geom):
    tw = DigitalTwin(specimen, camera="DESim", seed=seed, clock=ManualClock(),
                     detector_overrides={"gain_mode": "low"})
    tw.configure_ebsd(beam_kv=KV, probe_current_na=100.0, **geom)
    return tw


def _scan(tw, nx=24, ny=16, step=0.5, **kw):
    return tw.request(hw_binning=(2, 2), frame_time_s=DWELL, total_frames=nx * ny,
                      scan=ScanRequest(enabled=True, size=(nx, ny), step_um=step, dwell_s=DWELL), **kw)


def _general(tw, req, monkeypatch):
    monkeypatch.setenv("DE_TWIN_FUSED_EBSD", "0")
    try:
        return [raw.copy() for raw, _ in tw.frames(req)]
    finally:
        monkeypatch.delenv("DE_TWIN_FUSED_EBSD")


def _points(tw, req):
    """Corners, the middle, and points mixing two grains."""
    optics = tw.optics(req)
    ny, nx = optics.view.shape
    mix = tw.renderer.ebsd.tables(tw.specimen, optics).mix
    mixed = [(int(x), int(y)) for y, x in zip(*np.nonzero(mix > 0.05))][:3]
    return [(0, 0), (nx - 1, ny - 1), (nx // 2, ny // 3), *mixed]


@pytest.mark.parametrize("specimen", ["EBSD duplex steel", "EBSD Si (001) Vickers indent"])
def test_noise_free_patterns_match_the_general_path(specimen):
    """Two phases with grain-boundary mixing; a strained lattice (a deformation gradient per
    point)."""
    tw = _twin(specimen)
    req = _scan(tw)
    optics = tw.optics(req)
    source = tw.ebsd_frame_source(req)
    assert source is not None
    points = _points(tw, req)
    nx = optics.view.shape[1]
    lam = source.means([y * nx + x for x, y in points])
    for b, point in enumerate(points):
        _, (ref, _, dose, _, _) = tw.renderer.render_ebsd_frame(optics, 0, point, 0.0, DWELL * 4)
        assert np.abs(lam[b] - ref).max() <= 1e-5 * ref.max(), point
        assert lam[b].sum(dtype=np.float64) == pytest.approx(dose, rel=1e-6)


def test_supersampled_binning_takes_the_general_path():
    tw = _twin("EBSD Si (001) wafer", supersample=2)
    assert tw.ebsd_frame_source(_scan(tw)) is None


def test_frames_match_the_general_path(monkeypatch):
    """The same noise streams: frames differ only where float rounding tips a draw."""
    tw = _twin()
    req = _scan(tw, 6, 4, seed=7)
    fused = [raw for raw, _ in tw.frames(req)]
    general = _general(tw, req, monkeypatch)
    adu = tw.detector.model.adu_per_electron_at(KV)
    assert len(fused) == len(general) == 24
    for f, g in zip(fused, general):
        assert f.shape == (512, 512) and f.dtype == np.uint16
        assert (f != g).mean() < 1e-3
        assert np.abs(f.astype(int) - g.astype(int)).max() <= 4 * adu
        assert f.mean() == pytest.approx(g.mean(), rel=1e-5)
    assert len({id(f) for f in fused}) == 24, "a frame the caller keeps is never written over"
    assert not np.array_equal(fused[0], fused[1])


def test_frame_metadata_matches_the_general_path(monkeypatch):
    tw = _twin()
    req = _scan(tw, 4, 2, seed=3)
    fused = [meta for _, meta in tw.frames(req)]
    monkeypatch.setenv("DE_TWIN_FUSED_EBSD", "0")
    general = [meta for _, meta in tw.frames(req)]
    for f, g in zip(fused, general):
        assert f.scan_point == g.scan_point and f.frame_index == g.frame_index and not f.blanked
        assert f.electrons_per_pixel == pytest.approx(g.electrons_per_pixel, rel=1e-6)


def test_noise_statistics_match_the_general_path(monkeypatch):
    """Independent frames at one scan point: per-block mean and per-pixel variance agree with
    the general path's (other seeds, so other noise) within their sampling error, and
    successive frames share no noise."""
    tw = _twin()
    n = 160

    def blocks(a):  # 16 x 16 pixel blocks
        return a.reshape(32, 16, 32, 16).mean(axis=(1, 3))

    def stack(seed):
        req = tw.request(hw_binning=(2, 2), frame_time_s=DWELL, total_frames=n, seed=seed,
                         scan=ScanRequest(enabled=True, size=(8, 8), step_um=0.5, dwell_s=DWELL, park=True,
                                          park_position=(5, 3)))
        total, squares, block_means, samples = np.zeros((512, 512)), np.zeros((512, 512)), [], []
        for raw, _ in tw.frames(req):
            x = raw.astype(np.float64)
            total += x
            squares += x * x
            block_means.append(blocks(x))
            samples.append(x.ravel()[::7])
        mean = total / n
        noise = np.array(samples) - np.mean(samples, axis=0)
        correlations = [np.corrcoef(a, b)[0, 1] for a, b in zip(noise[:-1], noise[1:])]
        return mean, squares / n - mean ** 2, np.array(block_means), correlations

    fused_mean, fused_var, fused_blocks, correlations = stack(11)
    monkeypatch.setenv("DE_TWIN_FUSED_EBSD", "0")
    general_mean, general_var, general_blocks, _ = stack(12)
    monkeypatch.delenv("DE_TWIN_FUSED_EBSD")
    error = np.sqrt((fused_blocks.var(axis=0) + general_blocks.var(axis=0)) / n)
    z = (fused_blocks.mean(axis=0) - general_blocks.mean(axis=0)) / error
    assert np.abs(z).mean() < 1.0 and np.abs(z).max() < 5.0
    ratio = fused_var.mean() / general_var.mean()
    assert ratio == pytest.approx(1.0, abs=0.02)
    lit = general_mean > np.percentile(general_mean, 50)
    assert (fused_var[lit] / general_var[lit]).mean() == pytest.approx(1.0, abs=0.03)
    assert abs(np.mean(correlations)) < 0.01 and max(abs(c) for c in correlations) < 0.03


def test_low_gain_saturates_at_about_a_hundred_electrons(monkeypatch):
    """Low gain: ~100 electrons fill a pixel. A beam that delivers more clips there, on both
    paths alike."""
    tw = _twin()
    m = tw.detector.model
    adu = m.adu_per_electron_at(KV)
    assert (m.max_value - m.dark_offset_adu) / adu == pytest.approx(100, rel=0.15)
    tw.configure_ebsd(probe_current_na=400.0)
    req = _scan(tw, 2, 2, seed=5)
    fused = [raw for raw, _ in tw.frames(req)]
    general = _general(tw, req, monkeypatch)
    for f, g in zip(fused, general):
        assert f.max() == m.max_value and (f == m.max_value).mean() > 0.05
        assert (f == m.max_value).mean() == pytest.approx((g == m.max_value).mean(), rel=0.01)


def test_a_steered_scan_publishes_where_the_voltages_put_the_beam():
    """The shared-memory face on a private mapping: DE-Server's coil voltages place the beam,
    each frame is the pattern of the point they give."""
    from de_twin.faces.shm_face import ShmFace
    from de_twin.optics.scan_coils import point_volts
    from de_twin.transport import scan_output as S
    from de_twin.transport.shm import FrameConsumer

    tag = f"{os.getpid()}_{uuid.uuid4().hex[:6]}"
    scan_name, frame_name = f"DE_ScanOutputFused_{tag}", f"DE_ExternalFramesFused_{tag}"
    tw = _twin()
    geometry = S.ScanGeometry(size=(16, 16), voltage_h=(-1.0, 1.0), voltage_v=(-1.0, 1.0), rotation_deg=0.0,
                              flip=(False, False), points_per_frame=1.0, total_points=100_000)
    scan = ScanRequest(enabled=True, size=geometry.size)
    targets = [(2, 3), (13, 9), (7, 14)]
    writer = S.ScanOutputWriter(scan_name)
    consumer = FrameConsumer(frame_name, max_frame_bytes=512 * 512 * 2)
    face = ShmFace(tw, name=frame_name, scan_output=scan_name, pace=False, warm_up=False).start()
    try:
        writer.begin(geometry)
        for x, y in targets:
            writer.write(np.tile(point_volts(scan, x, y), (40, 1)))
        consumer.begin(frame_shape=(512, 512), binning=(2, 2), sensor_shape=(1024, 1024),
                       frame_time_s=DWELL, total_frames=120, scan_size=geometry.size)
        frames = [consumer.read(timeout=300)[0].astype(np.float64) for _ in range(120)]
    finally:
        face.stop()
        consumer.close()
        writer.close()
    req = tw.request(hw_binning=(2, 2), frame_time_s=DWELL, scan=ScanRequest(enabled=True, size=(16, 16)))
    source = tw.ebsd_frame_source(req)
    lam = source.means([y * 16 + x for x, y in targets]).reshape(3, -1)
    lam -= lam.mean(axis=1, keepdims=True)
    for i, frame in enumerate(frames):
        f = frame.ravel() - frame.mean()
        best = int(np.argmax(lam @ f))
        assert best == i // 40, (i, best)


def test_twin_frames_through_a_steered_point_list_are_made_one_at_a_time():
    """Points a running acquisition can still change are not read ahead."""

    class Wanted:
        point = (1, 1)

        def __len__(self):
            return 1 << 30

        def __getitem__(self, i):
            return self.point

    tw = _twin()
    wanted = Wanted()
    req = tw.request(hw_binning=(2, 2), frame_time_s=DWELL, total_frames=0,
                     scan=ScanRequest(enabled=True, size=(8, 8), step_um=0.5, dwell_s=DWELL, points=wanted))
    it = tw.frames(req)
    _, meta = next(it)
    assert meta.scan_point == (1, 1)
    wanted.point = (6, 2)
    _, meta = next(it)
    assert meta.scan_point == (6, 2)
    it.close()
