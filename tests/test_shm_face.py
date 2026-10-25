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
    assert face.ready.wait(60)  # warmed up: a test that wraps twin.frames sees only its own calls
    yield twin, face, consumer
    face.stop()
    consumer.close()


def test_frames_arrive_in_order(served):
    _twin, _face, consumer = served
    consumer.begin(frame_shape=(1024, 1024), frame_time_s=0.01, total_frames=10)
    frames = [consumer.read(timeout=10) for _ in range(10)]
    assert [h["frame_index"] for _, h in frames] == list(range(10))
    assert all(f.shape == (1024, 1024) and f.dtype == np.uint16 for f, _ in frames)
    assert frames[0][0].mean() > 450  # beam on
    assert not frames[0][1]["flags"] & L.FLAG_BLANKED


def test_dark_request_produces_blanked_frames(served):
    _twin, _face, consumer = served
    consumer.begin(frame_shape=(1024, 1024), total_frames=3, exposure_mode=ExposureMode.DARK)
    for _ in range(3):
        frame, info = consumer.read(timeout=10)
        assert info["flags"] & L.FLAG_BLANKED
        assert 300 < frame.mean() < 450


def test_new_request_preempts_live_stream(served):
    _twin, _face, consumer = served
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
    face = ShmFace(
        twin, name=f"DE_ExternalFramesCold_{uuid.uuid4().hex[:6]}", pace=False, warm_up=False
    ).start()
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
    assert face.ready.wait(60)  # warmed up: a test that wraps twin.frames sees only its own calls
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
        buffer = np.empty((1024, 1024), np.uint16)
        consumer.read(timeout=60, out=buffer)  # the first frame may wait for the kernels to compile
        pub0, reused0 = face.frames_published, face.frames_reused
        t0, n = time.monotonic(), 0
        while time.monotonic() - t0 < 1.0:
            consumer.read(timeout=10, out=buffer)
            n += 1
        # relative, not wall-clock: a slow runner renders fewer frames, but recycling still
        # publishes well over one frame per render
        published = face.frames_published - pub0
        rendered = published - (face.frames_reused - reused0)
        assert n >= 10, n
        assert published > 1.5 * max(rendered, 1), (published, rendered)
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
    pool = _FramePool(twin, twin.request(total_frames=0, frame_time_s=0.01), size=4, reuse=100, pace=False)
    try:
        (item, _fresh) = pool.take(10)
        x0 = item[1].microscope.stage.x_um
        twin.column.move_stage(x=x0 + 1.0)
        twin.column.state()  # the stage settles on the manual clock
        import time

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            (item, _fresh) = pool.take(10)
            if item[1].microscope.stage.x_um != x0:
                break
        # from here on, only the new view comes back, fresh or recycled
        for _ in range(20):
            (item, _) = pool.take(10)
            assert item[1].microscope.stage.x_um != x0
    finally:
        pool.close()


def test_the_first_frame_after_a_move_shows_the_new_view():
    """Once a move of the stage has returned, the very next frame is of the new view, even
    from a pool full of recycled frames of the old one: an acquisition straight after a
    move must not register the stage as having stayed put."""
    from de_twin.faces.shm_face import _FramePool

    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock())
    pool = _FramePool(twin, twin.request(total_frames=0, frame_time_s=0.01), size=8, reuse=10**9, pace=False)
    try:
        x0 = pool.take(60)[0][1].microscope.stage.x_um
        for step in (1.0, 2.0, 3.0):
            assert len(pool.take_many(50, 10)) == 50  # the old view, recycled
            twin.column.move_stage(x=x0 + step)
            after = [item[1].microscope.stage.x_um for item, _ in pool.take_many(5, 60)]
            assert after == pytest.approx([x0 + step] * len(after))
        twin.column.set_defocus_um(twin.column.state().defocus_um + 0.5)  # not only the stage
        item, _ = pool.take(60)
        assert item[1].microscope == twin.column.state()
    finally:
        pool.close()


def test_a_full_pool_is_a_library_until_the_view_changes(monkeypatch):
    """With frames to recycle, the renderer stops once the pool is full: publishing picks
    from it instead, and the interpreter is left to the thread that publishes. A move of
    the column renders a new set, even in diffraction, whose optics are frozen for the
    whole of a ``frames`` call."""
    import time

    from de_twin.faces.shm_face import _FramePool
    from de_twin.state import TemStem

    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock())
    twin.column.set_tem_stem(TemStem.STEM)
    rendered = {"n": 0}
    real = twin.frames

    def counted(request, **kw):
        for item in real(request, **kw):
            rendered["n"] += 1
            yield item

    monkeypatch.setattr(twin, "frames", counted)
    pool = _FramePool(twin, twin.request(total_frames=0, frame_time_s=0.001), size=4, reuse=10**9, pace=False)
    try:
        x0 = pool.take(10)[0][1].microscope.stage.x_um
        assert len(pool.take_many(100, 10)) == 100  # far more than were rendered
        time.sleep(0.3)
        assert rendered["n"] <= 5  # the pool, and the one frame waiting to join it
        twin.column.move_stage(x=x0 + 1.0)
        twin.column.state()  # the stage settles on the manual clock
        deadline = time.monotonic() + 20
        while pool.take(10)[0][1].microscope.stage.x_um == x0:
            assert time.monotonic() < deadline
        assert all(item[1].microscope.stage.x_um != x0 for item, _ in pool.take_many(20, 10))
    finally:
        pool.close()


def test_frames_are_taken_in_the_order_take_hands_them_out():
    """take_many is take, many times over, under one lock: fresh frames first, then the
    least-used, never more than *reuse* times each."""
    from de_twin.faces.shm_face import _FramePool

    pool = _FramePool.__new__(_FramePool)
    import threading

    pool.reuse, pool._cond, pool._end, pool._last, pool._watch = 2, threading.Condition(), None, None, False
    pool._ring = [[("a", None), 1], [("b", None), 0], [("c", None), 0]]
    got = pool.take_many(10, 0.0)
    assert [(item[0], fresh) for item, fresh in got] == [
        ("b", True),
        ("c", True),
        ("a", False),
        ("b", False),
        ("c", False),
    ]
    assert pool.take_many(1, 0.0) == [(("c", None), False)]  # spent: the last frame, as a keep-alive


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


def test_the_face_renders_with_a_few_threads_unless_told():
    """A frame is milliseconds of work: every core of a workstation renders it no faster,
    and on Windows numba's idle OpenMP threads spin, so the face takes a few."""
    from de_twin.faces import shm_face

    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock())
    assert ShmFace(twin, warm_up=False).threads == shm_face.RENDER_THREADS
    assert ShmFace(twin, warm_up=False, threads=2).threads == 2
