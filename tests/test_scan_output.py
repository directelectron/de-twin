"""DE-Server's scan coil voltages steering the twin's beam (de_twin.transport.scan_output)."""

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from de_twin.faces.shm_face import ShmFace
from de_twin.optics.scan_coils import point_from_volts, point_volts
from de_twin.state import FrameMeta, RenderMode, ScanRequest
from de_twin.transport import scan_output as S
from de_twin.transport.shm import FrameConsumer

GEOMETRY = S.ScanGeometry(
    size=(8, 8),
    voltage_h=(-1.0, 1.0),
    voltage_v=(-1.0, 1.0),
    rotation_deg=0.0,
    flip=(False, False),
    points_per_frame=1.0,
    total_points=10_000,
)


#: DE-Server's scan output for a parked beam: one point (`ScanController::_ParkSimulatedOutput`).
PARK = S.ScanGeometry(
    size=(8, 8),
    voltage_h=(-1.0, 1.0),
    voltage_v=(-1.0, 1.0),
    rotation_deg=0.0,
    flip=(False, False),
    points_per_frame=1.0,
    total_points=1,
)
#: Where `PointTwin` says the beam was in a frame no coil voltages placed.
UNSTEERED = (99, 99)


def _names():
    tag = f"{os.getpid()}_{uuid.uuid4().hex[:6]}"
    return f"DE_ScanOutput_{tag}", f"DE_ExternalFrames_{tag}"


def _volts(scan, x, y, n):
    return np.tile(point_volts(scan, x, y), (n, 1))


class TestRing:
    def test_points_read_back_once_written_and_until_overwritten(self):
        name, _ = _names()
        writer = S.ScanOutputWriter(name, capacity=16)
        try:
            writer.begin(GEOMETRY)
            reader = S.ScanOutputReader(name)
            assert reader.volts(0) is None and reader.latest() is None
            writer.write(np.arange(40, dtype=float).reshape(20, 2))
            assert reader.written == 20
            assert reader.volts(3) is None  # overwritten: the ring holds the last 16
            assert reader.volts(4) == (8.0, 9.0)
            assert reader.volts(19) == (38.0, 39.0)
            assert reader.volts(20) is None  # not written yet
            assert reader.latest() == (19, (38.0, 39.0))
            assert reader.geometry() == GEOMETRY
            reader.close()
        finally:
            writer.close()

    def test_a_new_scan_is_seen_by_its_id(self):
        name, _ = _names()
        writer = S.ScanOutputWriter(name)
        try:
            first = writer.begin(GEOMETRY)
            reader = S.ScanOutputReader(name)
            assert reader.scan_id == first
            assert writer.begin(GEOMETRY) == first + 1 == reader.scan_id
            reader.close()
        finally:
            writer.close()

    def test_no_scan_output_is_not_an_error_for_the_face(self):
        name, _ = _names()
        with pytest.raises(FileNotFoundError):
            S.ScanOutputReader(name)


CPP = Path(__file__).resolve().parents[1] / "cpp"
EXE = CPP / "build" / "test_consumer.exe"


class TestLayout:
    """Python agrees with DE-Server's header (cpp/ScanOutput.h)."""

    def test_constants_match_header(self):
        text = (CPP / "ScanOutput.h").read_text()
        assert f'kMappingName = "{S.DEFAULT_NAME}"' in text
        assert "kMagic       = 0x4F534544" in text and S.MAGIC == 0x4F534544
        assert f"kVersion     = {S.VERSION}" in text
        assert f"kHeaderBytes = {S.HEADER_BYTES}" in text
        assert "kCapacity    = 1ull << 20" in text and S.CAPACITY == 1 << 20

    @pytest.mark.skipif(
        sys.platform != "win32" or not EXE.exists(), reason="build cpp/test_consumer.exe with cpp\\build.bat"
    )
    def test_offsets_match_compiled_cpp(self):
        out = subprocess.run([str(EXE), "layout"], capture_output=True, text=True, check=True).stdout
        cpp = {}
        for line in out.splitlines():
            name, offset = line.split()
            struct_name, field = name.split(".")
            if struct_name == "ScanOutputHeader":
                cpp[field] = int(offset)
        assert cpp.pop("sizeof") == S.HEADER_STRUCT_BYTES
        snake = {field: "".join("_" + c.lower() if c.isupper() else c for c in field) for field in cpp}
        assert {snake[f]: o for f, o in cpp.items()} == {n: f.offset for n, f in S.HEADER.items()}


class TestInverse:
    @pytest.mark.parametrize(
        "rotation, flip",
        [(0.0, (False, False)), (30.0, (False, False)), (-75.0, (True, False)), (90.0, (True, True))],
    )
    def test_voltages_give_back_the_point(self, rotation, flip):
        scan = ScanRequest(
            enabled=True,
            size=(64, 48),
            voltage_h=(-2.0, 1.0),
            voltage_v=(-1.0, 1.5),
            rotation_deg=rotation,
            flip=flip,
            pixel_offset=(3.25, -1.5),
        )
        for x, y in [(0, 0), (63, 47), (10.5, 20.25), (-4.0, 50.0)]:
            assert point_from_volts(scan, *point_volts(scan, x, y)) == pytest.approx((x, y))


class _Column:
    def __init__(self):
        self.stage_x = 0.0

    def state(self):
        return self.stage_x


class PointTwin:
    """Frames that say where the beam was: pixel (0, 0) is the scan point's x, (0, 1) its y."""

    #: What an acquisition without a scan renders as.
    unscanned = RenderMode.STEM_PARKED

    def __init__(self):
        self.column = _Column()
        self.rendered = 0

    def request(self, **kw):
        raise NotImplementedError

    def optics(self, request):
        return SimpleNamespace(render_mode=RenderMode.STEM_4D if request.scan.enabled else self.unscanned)

    def frames(self, request, pace=False, stop=None):
        i = 0
        while request.total_frames <= 0 or i < request.total_frames:
            if stop is not None and stop.is_set():
                return
            points = request.scan.points
            x, y = points[i // max(1, request.scan.frames_per_point)] if points else UNSTEERED
            frame = np.zeros((4, 4), np.uint16)
            frame[0, 0], frame[0, 1] = x, y
            self.rendered += 1
            yield (
                frame,
                FrameMeta(
                    frame_index=i,
                    time_s=0.0,
                    exposure_s=request.frame_time_s,
                    electrons_per_pixel=1.0,
                    render_mode=RenderMode.STEM_4D,
                    blanked=False,
                    scan_point=(x, y),
                    microscope=self.column.state(),
                ),
            )
            i += 1


def _scan_request():
    from de_twin.state import AcquisitionRequest

    return AcquisitionRequest(
        frame_time_s=1e-4, total_frames=0, scan=ScanRequest(enabled=True, size=GEOMETRY.size)
    )


def _serve(twin, n_frames, write, **face_kw):
    scan_name, frame_name = _names()
    writer = S.ScanOutputWriter(scan_name)
    consumer = FrameConsumer(frame_name, max_frame_bytes=4 * 4 * 2)
    face = ShmFace(twin, name=frame_name, scan_output=scan_name, pace=False, warm_up=False, **face_kw).start()
    try:
        writer.begin(GEOMETRY)
        write(writer)
        consumer.begin(frame_shape=(4, 4), frame_time_s=1e-4, total_frames=n_frames, scan_size=GEOMETRY.size)
        points = []
        for _ in range(n_frames):
            px, _ = consumer.read(timeout=20)
            points.append((int(px[0, 0]), int(px[0, 1])))
        return points, face
    finally:
        face.stop()
        consumer.close()
        writer.close()


class TestSteeredFace:
    def test_each_frame_is_where_its_voltages_put_the_beam(self):
        scan = ScanRequest(enabled=True, size=GEOMETRY.size)

        def write(writer):
            writer.write(_volts(scan, 2, 3, 5))
            writer.write(_volts(scan, 6, 1, 5))

        points, _ = _serve(PointTwin(), 10, write)
        assert points == [(2, 3)] * 5 + [(6, 1)] * 5

    def test_the_coils_hold_the_last_voltages_written(self):
        scan = ScanRequest(enabled=True, size=GEOMETRY.size)
        points, _ = _serve(PointTwin(), 8, lambda writer: writer.write(_volts(scan, 5, 4, 3)))
        assert points == [(5, 4)] * 8

    def test_an_offset_beyond_the_grid_keeps_the_beam_on_its_edge(self):
        scan = ScanRequest(enabled=True, size=GEOMETRY.size)
        points, _ = _serve(PointTwin(), 2, lambda writer: writer.write(_volts(scan, -3, 20, 2)))
        assert points == [(0, 7)] * 2

    def test_a_held_probe_is_rendered_into_a_library_then_picked_from(self):
        scan = ScanRequest(enabled=True, size=GEOMETRY.size)
        twin = PointTwin()
        points, face = _serve(
            twin, 300, lambda writer: writer.write(_volts(scan, 1, 1, 300)), reuse=10**9, pool_size=16
        )
        assert points == [(1, 1)] * 300
        assert twin.rendered == 16
        assert face.frames_reused == 300 - 16

    def test_held_voltages_give_way_to_the_ones_written_after(self):
        """Frames ahead of the voltages hold the last ones, and only until the next are
        written: nothing worked out for them is kept."""
        from de_twin.faces.shm_face import _PointLibrary

        name, _ = _names()
        scan = ScanRequest(enabled=True, size=GEOMETRY.size)
        writer = S.ScanOutputWriter(name)
        writer.begin(GEOMETRY)
        reader = S.ScanOutputReader(name)
        library = _PointLibrary(PointTwin(), _scan_request(), reader, size=4)
        try:
            writer.write(_volts(scan, 2, 2, 3))
            assert [meta.scan_point for (_, meta), _ in library.frames(0, 2)] == [(2, 2)] * 2
            assert [meta.scan_point for (_, meta), _ in library.frames(2, 3)] == [(2, 2)] * 3  # 3, 4 held
            writer.write(_volts(scan, 7, 5, 5))
            assert [meta.scan_point for (_, meta), _ in library.frames(3, 2)] == [(7, 5)] * 2
        finally:
            library.close()
            reader.close()
            writer.close()

    def test_without_scan_output_the_twin_rasters_by_frame_index(self):
        _, frame_name = _names()
        consumer = FrameConsumer(frame_name, max_frame_bytes=4 * 4 * 2)
        face = ShmFace(
            PointTwinRaster(), name=frame_name, scan_output=_names()[0], pace=False, warm_up=False
        ).start()
        try:
            consumer.begin(frame_shape=(4, 4), frame_time_s=1e-4, total_frames=3, scan_size=(8, 8))
            points = [tuple(int(v) for v in consumer.read(timeout=20)[0][0, :2]) for _ in range(3)]
        finally:
            face.stop()
            consumer.close()
        assert points == [(0, 0), (1, 0), (2, 0)]


def _park(writer, x, y):
    """What DE-Server does each time it parks the beam: a new scan output of one point."""
    writer.begin(PARK)
    writer.write(_volts(ScanRequest(enabled=True, size=PARK.size), x, y, 1))


class TestParkedFace:
    """A parked beam: an acquisition without a scan, the coils held at the park point."""

    def _face(self, twin):
        scan_name, frame_name = _names()
        writer = S.ScanOutputWriter(scan_name)
        consumer = FrameConsumer(frame_name, max_frame_bytes=4 * 4 * 2)
        face = ShmFace(twin, name=frame_name, scan_output=scan_name, pace=False, warm_up=False).start()
        return writer, consumer, face

    @staticmethod
    def _read(consumer, n):
        return [tuple(int(v) for v in consumer.read(timeout=20)[0][0, :2]) for _ in range(n)]

    def test_the_frames_are_from_the_park_point(self):
        writer, consumer, face = self._face(PointTwin())
        try:
            _park(writer, 5, 2)
            consumer.begin(frame_shape=(4, 4), frame_time_s=1e-4, total_frames=6)
            assert self._read(consumer, 6) == [(5, 2)] * 6
        finally:
            face.stop()
            consumer.close()
            writer.close()

    def test_moving_the_park_during_the_acquisition_moves_the_beam(self):
        writer, consumer, face = self._face(PointTwin())
        try:
            _park(writer, 1, 1)
            consumer.begin(frame_shape=(4, 4), frame_time_s=1e-4, total_frames=0)
            assert set(self._read(consumer, 20)) == {(1, 1)}
            _park(writer, 6, 3)
            deadline = time.monotonic() + 20
            while self._read(consumer, 1) != [(6, 3)]:  # the frames already in the ring first
                assert time.monotonic() < deadline, "the beam did not follow the park"
            assert set(self._read(consumer, 20)) == {(6, 3)}
        finally:
            face.stop()
            consumer.close()
            writer.close()

    def test_a_tem_acquisition_is_not_steered_by_a_park(self):
        twin = PointTwin()
        twin.unscanned = RenderMode.TEM_IMAGING
        writer, consumer, face = self._face(twin)
        try:
            _park(writer, 5, 2)
            consumer.begin(frame_shape=(4, 4), frame_time_s=1e-4, total_frames=3)
            assert self._read(consumer, 3) == [UNSTEERED] * 3
        finally:
            face.stop()
            consumer.close()
            writer.close()

    def test_a_scan_output_of_many_points_is_not_a_park(self):
        writer, consumer, face = self._face(PointTwin())
        try:
            writer.begin(GEOMETRY)
            writer.write(_volts(ScanRequest(enabled=True, size=GEOMETRY.size), 5, 2, 4))
            consumer.begin(frame_shape=(4, 4), frame_time_s=1e-4, total_frames=3)
            assert self._read(consumer, 3) == [UNSTEERED] * 3
        finally:
            face.stop()
            consumer.close()
            writer.close()


class PointTwinRaster(PointTwin):
    """A twin rastering the grid itself, as the real one does without a point list."""

    def frames(self, request, pace=False, stop=None):
        nx, _ = request.scan.size
        for i in range(request.total_frames):
            frame = np.zeros((4, 4), np.uint16)
            frame[0, 0], frame[0, 1] = i % nx, i // nx
            yield (
                frame,
                FrameMeta(
                    frame_index=i,
                    time_s=0.0,
                    exposure_s=request.frame_time_s,
                    electrons_per_pixel=1.0,
                    render_mode=RenderMode.STEM_4D,
                    blanked=False,
                    scan_point=(i % nx, i // nx),
                    microscope=0.0,
                ),
            )
