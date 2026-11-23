"""Recorded datasets replayed as the twin's camera (`de_twin.recording`).

Synthetic recordings whose every frame says where and when it was taken (``1 + scan * 100 +
row * nx + column``, 0 where nothing was), so each test reads off which recorded frame it was
served: the three formats and where they say their geometry, reading in runs, the scan
geometry (subsampling, regions, offsets, the middle without a scan, ROI and binning), the
recorded clock on a fake wall clock, and the twin's faces serving them.
"""

from __future__ import annotations

import json
import os
import struct
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.optics.scan_coils import points_volts
from de_twin.recording import (
    FrameFiles,
    FramePart,
    RecordedClock,
    Replay,
    SideNeeded,
    Unreadable,
    open_recording,
)
from de_twin.state import AcquisitionRequest, ExposureMode, Roi, ScanRequest

SIDE, SCANS, PX = 8, 3, 4


def code(scan, row, column, nx=SIDE):
    return 1 + scan * 100 + row * nx + column


def coded_frames(nx=SIDE, ny=SIDE, scans=SCANS, px=PX, dtype=np.uint16):
    """Raster scans, every frame filled with its own code."""
    n = np.arange(scans * nx * ny)
    value = code(n // (nx * ny), (n % (nx * ny)) // nx, n % nx, nx)
    return np.broadcast_to(value[:, None, None], (len(n), px, px)).astype(dtype)


def write_mrc(path, frames):
    n, h, w = frames.shape
    head = bytearray(1024)
    mode = {np.dtype(np.uint16): 6, np.dtype(np.uint8): 0, np.dtype(np.float32): 2}[frames.dtype]
    struct.pack_into("<4i", head, 0, w, h, n, mode)
    head[212] = 0x44
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(np.ascontiguousarray(frames).tobytes())


def write_info(path, **values):
    lines = [f"{name:<64}= {value}" for name, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def info_for(nx=SIDE, ny=SIDE, frame_time_ns=1000, controller="Internal"):
    return {
        "Frame Time (nanoseconds)": frame_time_ns,
        "Scan - Enable": "Enable",
        "Scan - Controller": controller,
        "Scan - Size X": nx,
        "Scan - Size Y": ny,
    }


def write_up(path, frames, *, version=1, map_size=None):
    _, h, w = frames.shape
    if version >= 3:
        columns, rows = map_size or (0, 0)
        head = struct.pack("<4IB2IB2d", version, w, h, 42, 0, columns, rows, 0, 1.0, 1.0)
    else:
        head = struct.pack("<4I", version, w, h, 16)
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(np.ascontiguousarray(frames).tobytes())


@pytest.fixture
def movie(tmp_path):
    path = tmp_path / "20240101_00001_movie.mrc"
    write_mrc(path, coded_frames())
    write_info(tmp_path / "20240101_00001_info.txt", **info_for())
    return path


def served(recording_or_path, request, count, first=0, **kw):
    """Frames *first*.. of *request* from a replay of the recording, and their codes."""
    replay = (
        recording_or_path
        if isinstance(recording_or_path, Replay)
        else Replay(open_recording(recording_or_path))
    )
    frames, points = replay.chunk(request, first, count, **kw)
    return frames, frames[:, 0, 0].astype(int).tolist() if frames.shape[1] else [], points


def scan_request(nx=SIDE, ny=SIDE, **scan):
    return AcquisitionRequest(
        frame_time_s=1e-6, total_frames=nx * ny, scan=ScanRequest(enabled=True, size=(nx, ny), **scan)
    )


# ── formats ──────────────────────────────────────────────────────────────────


def test_a_movie_reads_its_scan_and_frame_time_from_the_info_beside_it(movie):
    recording = open_recording(movie)
    assert recording.field == (SIDE, SIDE) and recording.passes == SCANS
    assert recording.frame_shape == (PX, PX) and recording.dtype == np.uint16
    assert recording.frame_time_s == pytest.approx(1e-6)
    assert recording.describe()["scan_period_s"] == pytest.approx(SIDE * SIDE * 1e-6)


def test_an_externally_scanned_movie_asks_for_its_side_and_takes_it(tmp_path):
    path = tmp_path / "x_movie.mrc"
    write_mrc(path, coded_frames())
    # an external controller drove the scan: DE-Server's raster says nothing of the beam
    write_info(tmp_path / "x_info.txt", **info_for(nx=16, ny=12, controller="External Controller"))
    with pytest.raises(SideNeeded) as asked:
        open_recording(path)
    assert asked.value.frames == SCANS * SIDE * SIDE and asked.value.frame_shape == (PX, PX)
    recording = open_recording(path, side=SIDE)
    assert recording.field == (SIDE, SIDE) and recording.passes == SCANS
    with pytest.raises(Unreadable, match="not whole scans"):
        open_recording(path, side=5)


def test_a_movie_without_its_info_is_refused(tmp_path):
    path = tmp_path / "lonely_movie.mrc"
    write_mrc(path, coded_frames())
    with pytest.raises(Unreadable, match="_info.txt"):
        open_recording(path)


def test_a_version_3_up2_says_its_own_scan_size(tmp_path):
    path = tmp_path / "map_0_movie.up2"
    write_up(path, coded_frames(nx=6, ny=5, scans=1), version=3, map_size=(6, 5))
    recording = open_recording(path)
    assert recording.field == (6, 5) and recording.passes == 1
    _, codes, _ = served(path, scan_request(6, 5), 30)
    assert codes == [code(0, r, c, 6) for r in range(5) for c in range(6)]


def test_a_up1_takes_its_scan_size_from_the_info_then_the_ang_else_asks(tmp_path):
    frames = coded_frames(scans=1, dtype=np.uint8)
    path = tmp_path / "20260101_1_stem4d_0_movie.up1"
    write_up(path, frames)
    with pytest.raises(SideNeeded):
        open_recording(path)
    (tmp_path / "20260101_1_stem4d_0_movie.ang").write_text(
        "# NCOLS_ODD: 8\n# NCOLS_EVEN: 8\n# NROWS: 8\n0 0 0\n", encoding="utf-8"
    )
    assert open_recording(path).field == (SIDE, SIDE)
    write_info(tmp_path / "20260101_1_stem4d_info.txt", **info_for(nx=4, ny=16, frame_time_ns=5000))
    recording = open_recording(path)
    assert recording.field == (4, 16) and recording.frame_time_s == pytest.approx(5e-6)


def test_a_run_folder_places_each_frame_at_its_recorded_point_and_pass(tmp_path):
    """A 3 x 2 region of an 8 x 8 field, scanned twice, saved in two files: each frame at the
    point and pass the record's index says, and nothing recorded elsewhere."""
    region = [(r, c) for r in range(2, 4) for c in range(1, 4)]  # (row, column)
    np.save(tmp_path / "positions.npy", np.array(region, np.int64))
    rows = [(point, 100 + k, k // len(region)) for k, point in enumerate(list(range(6)) * 2)]
    index = np.array(rows, dtype=[("point", "<i8"), ("frame", "<i8"), ("pass", "<i4")])
    frames = np.stack([np.full((PX, PX), code(p, *region[point]), np.uint16) for point, _, p in rows])
    write_mrc(tmp_path / "run_movie_0.mrc", frames[:7])
    write_mrc(tmp_path / "run_movie_1.mrc", frames[7:])
    np.save(tmp_path / "index_0.npy", index[:7])
    np.save(tmp_path / "index_1.npy", index[7:])
    files = [
        {"path": "run_movie_0.mrc", "first": 0, "complete": True, "index": "index_0.npy"},
        {"path": "run_movie_1.mrc", "first": 7, "complete": True, "index": "index_1.npy"},
    ]
    record = {
        "format": "de_autopilot.dataset",
        "version": 1,
        "scan": {"type": "XY File", "field": [SIDE, SIDE], "positions": "positions.npy"},
        "detectors": {"camera": {"files": files}},
    }
    (tmp_path / "dataset.json").write_text(json.dumps(record), encoding="utf-8")
    write_info(tmp_path / "run_info.txt", **info_for())
    recording = open_recording(tmp_path)
    assert recording.field == (SIDE, SIDE) and recording.passes == 2 and recording.kind == "Autopilot run"
    rows_, columns = np.array([2, 3, 3, 0]), np.array([1, 3, 3, 0])
    assert recording.locate(np.array([0, 0, 1, 1]), rows_, columns).tolist() == [0, 5, 11, -1]
    replay = Replay(recording)
    replay.clock = RecordedClock(1.0, 6, 2)  # unstarted: scan 0
    _, codes, _ = served(replay, scan_request(), 64)
    expected = [code(0, r, c) if (r, c) in region else 0 for r in range(SIDE) for c in range(SIDE)]
    assert codes == expected


# ── reading ──────────────────────────────────────────────────────────────────


def test_frames_are_read_in_runs_into_place_with_zeros_where_none(tmp_path):
    frames = coded_frames(scans=1)
    write_mrc(tmp_path / "a.mrc", frames[:40])
    write_mrc(tmp_path / "b.mrc", frames[40:])
    files = FrameFiles(
        [FramePart(str(tmp_path / "a.mrc"), 1024, 40, 0), FramePart(str(tmp_path / "b.mrc"), 1024, 24, 40)],
        (PX, PX),
        np.uint16,
    )
    wanted = np.array([5, 6, 7, 3, -1, 41, 39, 7, 63, 64])
    out = files.read(wanted, np.empty((len(wanted), PX, PX), np.uint16))
    expected = [frames[n, 0, 0] if 0 <= n < 64 else 0 for n in wanted]
    assert out[:, 0, 0].tolist() == expected
    # 3..7 is one read (the gap is merged), 39 another, 41 another (the next file), 63 one more
    assert files.reads == 4
    # a raster in order lands straight where it goes, one read
    files.reads = 0
    out = files.read(np.arange(8, 16), np.empty((8, PX, PX), np.uint16))
    assert files.reads == 1 and out[:, 0, 0].tolist() == frames[8:16, 0, 0].tolist()
    files.close()


# ── geometry ─────────────────────────────────────────────────────────────────


def test_a_scan_of_the_recorded_size_replays_it_point_for_point(movie):
    _, codes, points = served(movie, scan_request(), SIDE * SIDE)
    assert codes == [code(0, r, c) for r in range(SIDE) for c in range(SIDE)]
    assert points[9].tolist() == [1, 1]


def test_a_smaller_scan_subsamples_the_same_field(movie):
    _, codes, _ = served(movie, scan_request(4, 4), 16)
    # each scan point is two recorded points wide: the one under its middle
    assert codes == [code(0, 2 * r + 1, 2 * c + 1) for r in range(4) for c in range(4)]


def test_a_region_and_an_offset_read_the_points_under_them_and_zeros_off_the_field(movie):
    region = [(6, 1), (7, 1), (6, 2)]  # (x, y), as an XY array or a scan ROI addresses them
    request = scan_request(points=region)
    _, codes, _ = served(movie, request, 3)
    assert codes == [code(0, 1, 6), code(0, 1, 7), code(0, 2, 6)]
    # a pixel offset of one point right moves every one of them; (7, 1) goes off the field
    _, codes, _ = served(movie, scan_request(points=region, pixel_offset=(1.0, 0.0)), 3)
    assert codes == [code(0, 1, 7), 0, code(0, 2, 7)]


def test_de_servers_own_voltages_place_the_beam(movie):
    scan = ScanRequest(enabled=True, size=(SIDE, SIDE))
    volts = points_volts(scan, np.array([[3, 4], [0, 0], [-2, 0]]))
    _, codes, _ = served(movie, scan_request(), 3, volts=volts)
    assert codes == [code(0, 4, 3), code(0, 0, 0), 0]


def test_a_roi_is_cut_from_the_middle_and_binning_sums(tmp_path):
    ramp = np.arange(64, dtype=np.uint16).reshape(8, 8)
    path = tmp_path / "r_movie.mrc"
    write_mrc(path, np.broadcast_to(ramp, (4, 8, 8)).copy())
    write_info(tmp_path / "r_info.txt", **info_for(nx=2, ny=2))
    replay = Replay(open_recording(path))
    request = AcquisitionRequest(frame_time_s=1e-6, total_frames=1, hw_roi=Roi(0, 0, 4, 2))
    frames, _ = replay.chunk(request, 0, 1)
    assert frames.shape == (1, 2, 4) and frames[0].tolist() == ramp[3:5, 2:6].tolist()
    request = AcquisitionRequest(frame_time_s=1e-6, total_frames=1, hw_binning=(2, 2))
    frames, _ = replay.chunk(request, 0, 1)
    assert frames.shape == (1, 4, 4) and frames[0, 0, 0] == ramp[:2, :2].sum()
    assert replay.output_shape(request) == (4, 4)


def test_a_frame_without_a_scan_is_the_middle_point(movie):
    request = AcquisitionRequest(frame_time_s=1e-6, total_frames=2)
    _, codes, points = served(movie, request, 2)
    assert codes == [code(0, SIDE // 2, SIDE // 2)] * 2 and points is None
    _, codes, _ = served(movie, request, 2, blanked=True)
    assert codes == [0, 0]  # the beam off the camera: the recording's dark was taken off


# ── the recorded clock ───────────────────────────────────────────────────────


class Wall:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def clock(frame_time=0.01, pass_frames=100, passes=5):
    wall = Wall()
    return RecordedClock(frame_time, pass_frames, passes, wall=wall), wall  # a scan is 1 s


def test_the_clock_waits_for_the_first_acquisition_that_is_not_a_search():
    recorded, wall = clock()
    wall.t += 50
    assert recorded.take(3).tolist() == [0, 0, 0] and recorded.now() == 0
    recorded.set_searching(True)
    kind = recorded.begin()
    wall.t += 30
    assert recorded.take(400).tolist() == [0] * 400  # a search looks without starting it
    recorded.end(kind)
    wall.t += 30
    assert recorded.now() == 0
    recorded.set_searching(False)
    kind = recorded.begin()
    assert recorded.take(150).tolist() == [0] * 100 + [1] * 50
    recorded.end(kind)


def test_acquiring_continuously_plays_the_scans_back_to_back():
    recorded, wall = clock()
    kind = recorded.begin()
    passes = np.concatenate([recorded.take(70) for _ in range(5)])
    wall.t += 1000  # however long it took on the wall: frames drive it
    passes = np.concatenate([passes, recorded.take(150)])
    assert passes.tolist() == [k for k in range(5) for _ in range(100)]
    recorded.end(kind)


def test_between_acquisitions_the_clock_runs_on_the_wall():
    recorded, wall = clock()
    kind = recorded.begin()
    recorded.take(50)  # 0.5 s
    recorded.end(kind)
    wall.t += 1.25  # 1.75 s
    assert recorded.now() == pytest.approx(1.75) and recorded.scan_now() == 1
    recorded.set_searching(True)  # a search, once started, sees the experiment go on
    kind = recorded.begin()
    wall.t += 0.5
    assert recorded.take(2).tolist() == [2, 2]
    recorded.end(kind)
    recorded.set_searching(False)
    kind = recorded.begin()
    assert recorded.now() == pytest.approx(2.25)
    assert recorded.take(100).tolist() == [2] * 75 + [3] * 25
    recorded.end(kind)


def test_a_series_start_goes_back_to_the_beginning():
    recorded, wall = clock()
    kind = recorded.begin()
    recorded.take(330)
    recorded.end(kind)
    wall.t += 10
    recorded.series_started()
    wall.t += 10  # the wait before the series' first acquisition is not the experiment's
    assert recorded.now() == 0 and recorded.state()["mode"] == "unstarted"
    kind = recorded.begin()
    assert recorded.take(1).tolist() == [0]
    recorded.take(250)
    recorded.series_started()  # mid-acquisition: on from 0 there and then
    assert recorded.take(101).tolist() == [0] * 100 + [1]
    recorded.end(kind)


def test_past_the_end_the_last_scan_repeats_and_one_scan_always_repeats():
    recorded, wall = clock()
    kind = recorded.begin()
    assert set(recorded.take(2000)[500:].tolist()) == {4}
    recorded.end(kind)
    wall.t += 1e6
    assert recorded.scan_now() == 4
    single, wall = clock(passes=1)
    kind = single.begin()
    assert set(single.take(1000).tolist()) == {0}
    single.end(kind)


# ── the twin, through its faces ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def twin():
    from de_twin.twin import DigitalTwin

    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock())
    yield twin
    twin.unload_recording()


def test_loading_locks_the_sensor_to_the_recording_and_unloading_restores_it(twin, movie):
    rendered = tuple(twin.detector.model.sensor_shape)
    replay = twin.load_recording(movie)
    try:
        assert tuple(twin.detector.model.sensor_shape) == (PX, PX)
        assert twin.recording is replay
        frames = list(twin.frames(scan_request()))
        assert [int(f[0, 0]) for f, _ in frames] == [code(0, r, c) for r in range(SIDE) for c in range(SIDE)]
        assert frames[10][1].scan_point == (2, 1)
    finally:
        twin.unload_recording()
    assert tuple(twin.detector.model.sensor_shape) == rendered and twin.recording is None


def test_the_twins_acquisitions_drive_the_recorded_clock(twin, movie):
    replay = twin.load_recording(movie)
    try:
        twin.searching(True)
        with twin.acquiring():
            assert {int(f[0, 0]) // 100 for f, _ in twin.frames(scan_request())} == {0}
        twin.searching(False)
        seen = []
        with twin.acquiring():
            for _ in range(SCANS + 1):
                seen.append({int(f[0, 0]) // 100 for f, _ in twin.frames(scan_request())})
        assert seen == [{0}, {1}, {2}, {2}]
        twin.series_started()
        assert replay.clock.scan_now() == 0
    finally:
        twin.unload_recording()


def test_a_region_scan_over_deapi_forms_the_recordings_virtual_image(twin, movie):
    deapi = pytest.importorskip("deapi")
    from de_twin.faces.deapi_server import TwinDeapiServer

    twin.load_recording(movie)
    server = TwinDeapiServer(twin, port=0, pace=False).start()
    client = deapi.Client()
    client.usingMmf = False
    try:
        client.connect(port=server.port)
        assert client["Sensor Size X (pixels)"] == PX and client["Image Size Y (pixels)"] == PX
        client["Scan - Enable"] = "On"
        client["Scan - Size X"] = SIDE
        client["Scan - Size Y"] = SIDE
        client.update_scan_size()
        client["Scan - ROI Enable"] = "On"
        client["Scan - ROI Offset X"], client["Scan - ROI Offset Y"] = 2, 3
        client["Scan - ROI Size X"], client["Scan - ROI Size Y"] = 4, 4  # DE-Server's smallest
        client.start_acquisition(1)
        deadline = time.time() + 20
        while client.acquiring and time.time() < deadline:
            time.sleep(0.01)
        virtual, *_ = client.get_result("virtual_image0")
        expected = np.zeros((SIDE, SIDE))
        for y in range(3, 7):
            for x in range(2, 6):
                expected[y, x] = code(0, y, x) * PX * PX  # mask 0 is all ones
        assert virtual.shape == (SIDE, SIDE)
        np.testing.assert_allclose(virtual, expected)
    finally:
        client.disconnect()
        server.stop()
        twin.unload_recording()


def test_the_feed_serves_the_recording_where_de_servers_voltages_put_the_beam(twin, movie):
    from de_twin.faces.shm_face import ShmFace
    from de_twin.transport import scan_output as S
    from de_twin.transport.shm import FrameConsumer

    tag = f"{os.getpid()}_{uuid.uuid4().hex[:6]}"
    scan_name, frame_name = f"DE_ScanOutput_{tag}", f"DE_ExternalFrames_{tag}"
    geometry = S.ScanGeometry(
        size=(SIDE, SIDE),
        voltage_h=(-1.0, 1.0),
        voltage_v=(-1.0, 1.0),
        rotation_deg=0.0,
        flip=(False, False),
        points_per_frame=1.0,
        total_points=SIDE * SIDE,
    )
    scan = ScanRequest(enabled=True, size=(SIDE, SIDE))
    twin.load_recording(movie)
    writer = S.ScanOutputWriter(scan_name)
    consumer = FrameConsumer(frame_name, max_frame_bytes=8 * 8 * 2)
    face = ShmFace(twin, name=frame_name, scan_output=scan_name, pace=False, warm_up=False).start()
    try:
        writer.begin(geometry)
        writer.write(points_volts(scan, np.array([[1, 2], [5, 5], [9, 0]])))
        # DE-Server asks for 8 x 8 frames: the 4 x 4 recorded frame sits in their middle
        consumer.begin(frame_shape=(8, 8), frame_time_s=1e-4, total_frames=3, scan_size=(SIDE, SIDE))
        got = [consumer.read(timeout=20)[0] for _ in range(3)]
        assert [int(f[4, 4]) for f in got] == [code(0, 2, 1), code(0, 5, 5), 0]
        assert int(got[0][0, 0]) == 0
    finally:
        face.stop()
        consumer.close()
        writer.close()
        twin.unload_recording()


def test_the_chunked_virtual_detectors_are_the_per_frame_sums():
    """The deapi face's virtual detectors, a stack at a time: plus and minus masks, Sum and
    Difference, exactly as frame-by-frame sums give them."""
    from de_twin.faces.deapi_results import ResultsMixin

    rng = np.random.default_rng(1)
    shape = (6, 5)
    masks = [rng.integers(0, 3, shape).astype(np.int8) for _ in range(5)]
    calcs = ["sum", "difference", "sum", "difference", "sum"]
    face = SimpleNamespace(virtual_masks=masks, _mask_cache={})
    face.__getitem__ = None
    face._masks_for = lambda s: [
        (m == 1, (m == 2) if (m == 2).any() else None, c) for m, c in zip(masks, calcs, strict=True)
    ]
    weights = ResultsMixin._mask_weights(face, shape)
    frames = rng.integers(0, 4000, (7, *shape)).astype(np.uint16)
    stacked = frames.reshape(7, -1).astype(np.float32) @ weights.T
    for k, (m, c) in enumerate(zip(masks, calcs, strict=True)):
        plus = frames[:, m == 1].sum(axis=1, dtype=np.int64)
        minus = frames[:, m == 2].sum(axis=1, dtype=np.int64)
        np.testing.assert_allclose(stacked[:, k], plus - minus if c == "difference" else plus + minus)


def test_a_dark_acquisition_of_a_recording_is_zeros(twin, movie):
    twin.load_recording(movie)
    try:
        request = AcquisitionRequest(frame_time_s=1e-6, total_frames=2, exposure_mode=ExposureMode.DARK)
        assert [int(f.max()) for f, _ in twin.frames(request)] == [0, 0]
    finally:
        twin.unload_recording()
