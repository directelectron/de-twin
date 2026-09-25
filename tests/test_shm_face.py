"""The shared-memory face serving a Python FrameConsumer (standing in for DE-Server)."""

import os
import uuid

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.faces.shm_face import ShmFace
from de_twin.state import ExposureMode
from de_twin.transport import shm_layout as L
from de_twin.transport.shm import FrameConsumer
from de_twin.twin import DigitalTwin


@pytest.fixture
def served():
    name = f"DE_ExternalFramesFace_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock())
    consumer = FrameConsumer(name, max_frame_bytes=1024 * 1024 * 2)  # DE-Server creates the mapping
    face = ShmFace(twin, name=name, pace=False).start()
    yield twin, face, consumer
    face.stop()
    consumer.close()


def test_frames_arrive_in_order(served):
    twin, face, consumer = served
    consumer.begin(frame_shape=(1024, 1024), frame_time_s=0.01, total_frames=10)
    frames = [consumer.read(timeout=10) for _ in range(10)]
    assert [h["frame_index"] for _, h in frames] == list(range(10))
    assert all(f.shape == (1024, 1024) and f.dtype == np.uint16 for f, _ in frames)
    assert frames[0][0].mean() > 450  # beam on
    assert not frames[0][1]["flags"] & L.FLAG_BLANKED


def test_dark_request_produces_blanked_frames(served):
    twin, face, consumer = served
    consumer.begin(frame_shape=(1024, 1024), total_frames=3, exposure_mode=ExposureMode.DARK)
    for _ in range(3):
        frame, info = consumer.read(timeout=10)
        assert info["flags"] & L.FLAG_BLANKED
        assert 300 < frame.mean() < 450


def test_new_request_preempts_live_stream(served):
    twin, face, consumer = served
    consumer.begin(frame_shape=(1024, 1024), total_frames=0)  # live: until told otherwise
    for _ in range(3):
        consumer.read(timeout=10)
    rid = consumer.begin(frame_shape=(512, 512), total_frames=2, binning=(2, 2), sensor_shape=(1024, 1024))
    frame, info = consumer.read(timeout=10)
    assert info["request_id"] == rid and info["frame_index"] == 0
    assert frame.shape == (512, 512)


def test_the_face_warms_the_twin_up_before_serving():
    """DE-Server gives up on an acquisition about a second after its first frame is due, far
    less than a fresh twin's first render: the face renders one frame before it attaches."""
    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock())
    rendered = []
    frames = twin.frames
    twin.frames = lambda *a, **k: (rendered.append(1), frames(*a, **k))[1]
    name = f"DE_ExternalFramesWarm_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    face = ShmFace(twin, name=name, pace=False).start()  # no mapping yet: nothing to serve
    try:
        assert face.ready.wait(60)
        assert rendered == [1] and face.producer is None
    finally:
        face.stop()


def test_warm_up_can_be_skipped():
    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock())
    twin.frames = lambda *a, **k: pytest.fail("rendered without a request")
    face = ShmFace(twin, name=f"DE_ExternalFramesCold_{uuid.uuid4().hex[:6]}", pace=False, warm_up=False).start()
    try:
        assert face.ready.wait(5)
    finally:
        face.stop()


def test_a_slow_render_does_not_starve_the_consumer(served, monkeypatch):
    """DE-Server stops an acquisition when a frame is ~1 s late; while the twin is busy
    with a slow render the face republishes the last frame instead."""
    import time

    from de_twin.faces import shm_face

    twin, face, consumer = served
    monkeypatch.setattr(shm_face, "KEEPALIVE_S", 0.1)
    real = twin.frames

    def slow_frames(request, **kw):
        for k, item in enumerate(real(request, **kw)):
            if k == 2:
                time.sleep(1.0)  # a slow render
            yield item

    monkeypatch.setattr(twin, "frames", slow_frames)
    consumer.begin(frame_shape=(1024, 1024), frame_time_s=0.01, total_frames=0)
    gaps, t = [], time.monotonic()
    for _ in range(12):
        consumer.read(timeout=10)
        now = time.monotonic()
        gaps.append(now - t)
        t = now
    assert max(gaps[1:]) < 0.5, gaps
    assert face.keepalives >= 3


def _served_with(**kw):
    name = f"DE_ExternalFramesPool_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock())
    consumer = FrameConsumer(name, max_frame_bytes=1024 * 1024 * 2)
    face = ShmFace(twin, name=name, pace=False, **kw).start()
    return twin, face, consumer


def test_recycling_publishes_faster_than_the_twin_renders(monkeypatch):
    import time

    twin, face, consumer = _served_with(reuse=4, pool_size=4)
    real = twin.frames

    def slow_frames(request, **kw):
        for item in real(request, **kw):
            time.sleep(0.05)  # the physics: 20 frames a second
            yield item

    monkeypatch.setattr(twin, "frames", slow_frames)
    try:
        consumer.begin(frame_shape=(1024, 1024), frame_time_s=0.001, total_frames=0)
        buf = np.empty((1024, 1024), np.uint16)
        consumer.read(timeout=10, out=buf)
        t0, n = time.monotonic(), 0
        while time.monotonic() - t0 < 1.0:
            consumer.read(timeout=10, out=buf)
            n += 1
        assert n > 40, n  # more than twice the rendered rate
        assert face.frames_reused > 0
        rendered = face.frames_published - face.frames_reused
        assert face.frames_reused <= 3 * rendered + 4 * face.pool_size  # at most `reuse` uses each
    finally:
        face.stop()
        consumer.close()


def test_a_view_change_drops_recycled_frames():
    """A recycled frame never shows a view the column has left."""
    from de_twin.faces.shm_face import _FramePool

    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock())
    pool = _FramePool(twin, twin.request(total_frames=0, frame_time_s=0.01), size=4, reuse=100,
                      pace=False)
    try:
        (item, fresh) = pool.take(10)
        x0 = item[1].microscope.stage.x_um
        twin.column.move_stage(x=x0 + 1.0)
        twin.column.state()  # the stage settles on the manual clock
        import time

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            (item, fresh) = pool.take(10)
            if item[1].microscope.stage.x_um != x0:
                break
        # from here on, only the new view comes back, fresh or recycled
        for _ in range(20):
            (item, _) = pool.take(10)
            assert item[1].microscope.stage.x_um != x0
    finally:
        pool.close()


def test_live_view_as_short_acquisitions_is_kept_fed(monkeypatch):
    """DE-MC's live view is a string of short acquisitions, one request each. The pool
    (and so the keep-alive) carries over between them, so a slow render at the start of
    one does not starve DE-Server."""
    import time

    from de_twin.faces import shm_face

    twin, face, consumer = _served_with()
    monkeypatch.setattr(shm_face, "KEEPALIVE_S", 0.1)
    real = twin.frames
    calls = {"n": 0}

    def slow_frames(request, **kw):
        calls["n"] += 1
        for k, item in enumerate(real(request, **kw)):
            if k == 7:
                time.sleep(1.0)  # a slow render, in the second acquisition
            yield item

    monkeypatch.setattr(twin, "frames", slow_frames)
    try:
        gaps, t = [], None
        for _ in range(4):  # four 4-frame acquisitions
            consumer.begin(frame_shape=(1024, 1024), frame_time_s=0.001, total_frames=4)
            for _ in range(4):
                consumer.read(timeout=10)
                now = time.monotonic()
                if t is not None:
                    gaps.append(now - t)
                t = now
        assert max(gaps) < 0.5, gaps
        assert calls["n"] == 1, "one render stream for the whole live view"
        assert face.keepalives >= 3
    finally:
        face.stop()
        consumer.close()
