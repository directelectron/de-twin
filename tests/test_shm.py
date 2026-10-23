"""Protocol v2 in Python: FrameConsumer (stands in for DE-Server) and FrameProducer (the twin)."""

from __future__ import annotations

import os
import threading
import uuid

import numpy as np
import pytest

from de_twin.state import ExposureMode
from de_twin.transport import shm_layout as L
from de_twin.transport.shm import FrameConsumer, FrameProducer, mapping_exists


def uname():
    return f"DE_ExternalFramesTest_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def test_producer_needs_deserver_to_create_the_mapping():
    name = uname()
    assert not mapping_exists(name)
    with pytest.raises(FileNotFoundError):
        FrameProducer(name)


def test_request_roundtrip_and_frames_in_order():
    name = uname()
    with FrameConsumer(name, max_frame_bytes=64 * 48 * 2) as cons:
        prod = FrameProducer(name)
        rid = cons.begin(
            frame_shape=(48, 64),
            exposure_mode=ExposureMode.GAIN,
            frame_time_s=0.02,
            total_frames=5,
            binning=(2, 2),
            sensor_shape=(96, 128),
        )
        got = prod.poll_request(timeout=1)
        assert got is not None and got[0] == rid
        request = got[1]
        assert request.exposure_mode == ExposureMode.GAIN and request.total_frames == 5
        assert request.hw_binning == (2, 2) and request.frame_time_s == pytest.approx(0.02)
        for i in range(5):
            assert prod.wait_slot_free(1)
            prod.publish(np.full((48, 64), i, np.uint16), request_id=rid, frame_index=i)
        for i in range(5):
            frame, info = cons.read(1)
            assert info["frame_index"] == i and frame.shape == (48, 64) and np.all(frame == i)
        prod.close()


def test_back_pressure_limits_the_producer_to_the_ring():
    name = uname()
    with FrameConsumer(name, max_frame_bytes=16 * 16 * 2) as cons:
        prod = FrameProducer(name)
        rid = cons.begin(frame_shape=(16, 16))
        prod.poll_request(timeout=1)
        for i in range(L.SLOT_COUNT):
            assert prod.wait_slot_free(0.1)
            prod.publish(np.zeros((16, 16), np.uint16), request_id=rid, frame_index=i)
        assert not prod.wait_slot_free(0.05)  # ring full until DE-Server reads
        cons.read(1)
        assert prod.wait_slot_free(0.05)
        prod.close()


def test_new_request_skips_frames_made_for_the_old_one():
    name = uname()
    with FrameConsumer(name, max_frame_bytes=16 * 16 * 2) as cons:
        prod = FrameProducer(name)
        old = cons.begin(frame_shape=(16, 16))
        prod.poll_request(timeout=1)
        prod.publish(np.full((16, 16), 7, np.uint16), request_id=old, frame_index=0)
        new = cons.begin(frame_shape=(16, 16))
        # the producer was mid-acquisition and published one more stale frame
        prod.publish(np.full((16, 16), 8, np.uint16), request_id=old, frame_index=1)
        assert prod.poll_request(timeout=1)[0] == new
        prod.publish(np.full((16, 16), 9, np.uint16), request_id=new, frame_index=0)
        frame, info = cons.read(1)
        assert info["request_id"] == new and np.all(frame == 9)
        prod.close()


def test_threaded_stream():
    name = uname()
    n = 200
    with FrameConsumer(name, max_frame_bytes=32 * 32 * 2) as cons:
        prod = FrameProducer(name)
        rid = cons.begin(frame_shape=(32, 32), total_frames=n)

        def run():
            _, request = prod.poll_request(timeout=2)
            for i in range(request.total_frames):
                while not prod.wait_slot_free(0.05):
                    pass
                prod.publish(np.full((32, 32), i % 4096, np.uint16), request_id=rid, frame_index=i)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        values = [int(cons.read(5)[0][0, 0]) for _ in range(n)]
        t.join(5)
        assert values == [i % 4096 for i in range(n)]
        prod.close()
