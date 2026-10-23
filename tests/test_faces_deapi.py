"""deapi face: a real deapi.Client against TwinFakeServer (stub twin, and the real twin if ready)."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

deapi = pytest.importorskip("deapi")
pytestmark = pytest.mark.deapi

from de_twin.clock import Clock
from de_twin.column import Column
from de_twin.faces.deapi_protocol import TEMPERATURE_PROP
from de_twin.faces.deapi_request import request_from_properties
from de_twin.faces.deapi_server import TwinDeapiServer
from de_twin.holder import NoHolder
from de_twin.state import ExposureMode, FrameMeta, RenderMode, Roi


class StubDetector:
    def __init__(self, shape=(64, 96)):
        self.model = SimpleNamespace(name="StubCam", sensor_shape=shape, full_frame_time_s=0.001)

    def roi(self, request):
        h, w = self.model.sensor_shape
        return request.hw_roi or Roi(0, 0, w, h)

    def output_shape(self, request):
        r = self.roi(request)
        bx, by = request.hw_binning
        return r.h // by, r.w // bx


class StubTwin:
    """Raw frames: dark offset 100 ADU + (beam) a bright column at stage x."""

    def __init__(self):
        self.lock = threading.RLock()
        self.clock = Clock()
        self.column = Column(clock=self.clock, stage_speed_um_s=1e6)
        self.detector = StubDetector()
        self.holder = NoHolder(clock=self.clock)
        self.requests = []
        self.specimen_name = None

    def set_specimen(self, name):
        self.specimen_name = name

    def frames(self, request, pace=False, stop=None):
        self.requests.append(request)
        h, w = self.detector.output_shape(request)
        rng = np.random.default_rng(request.acquisition_index)
        for i in range(request.total_frames):
            if stop is not None and stop.is_set():
                return
            state = self.column.state()
            blanked = not request.beam_reaches_detector or state.beam_blanked
            frame = 100.0 + rng.normal(0, 2, (h, w))
            if not blanked:
                if request.exposure_mode == ExposureMode.GAIN:
                    frame += 400.0
                else:
                    column = int(state.stage.x_um) % w
                    frame[:, column] += 3000.0
                    frame += 50.0
            meta = FrameMeta(
                frame_index=i,
                time_s=self.clock.now(),
                exposure_s=request.frame_time_s,
                electrons_per_pixel=0.0 if blanked else 5.0,
                render_mode=RenderMode.TEM_IMAGING,
                blanked=blanked,
                scan_point=_scan_point(request, i),
                microscope=state,
            )
            yield np.clip(frame, 0, 65535).astype(np.uint16), meta

    def virtual_image(self, request, inner, outer):
        nx, ny = request.scan.size
        return np.arange(nx * ny, dtype=np.float32).reshape(ny, nx)


def _scan_point(request, i):
    if not request.scan.enabled:
        return None
    nx, ny = request.scan.size
    p = (i // request.scan.frames_per_point) % (nx * ny)
    return p % nx, p // nx


@pytest.fixture
def twin():
    return StubTwin()


@pytest.fixture
def server(twin):
    srv = TwinDeapiServer(twin, port=0, pace=False).start()
    yield srv
    srv.stop()


@pytest.fixture
def client(server):
    c = deapi.Client()
    c.usingMmf = False
    c.connect(port=server.port)
    yield c
    c.disconnect()


def _acquire(client, frame_type="singleframe_integrated", n=1):
    client.start_acquisition(n)
    deadline = time.time() + 10
    while client.acquiring and time.time() < deadline:
        time.sleep(0.01)
    image, _pixel_format, attrs, _hist = client.get_result(frame_type)
    return image, attrs


def test_connect_and_properties(client, twin):
    assert client["Camera Name"] == "StubCam"
    assert client["Sensor Size X (pixels)"] == 96 and client["Sensor Size Y (pixels)"] == 64
    assert client["Image Size X (pixels)"] == 96
    assert client["Instrument Project Magnification"] == 20000.0
    assert client["Instrument Project Name"] == "MAG1"
    assert "PROJ_Magnification=20000" in client["Instrument Metadata"]
    twin.column.move_stage(x=42.0)  # waits: the stage travels at 100 um/s
    assert client["Instrument Stage Position X (micrometers)"] == 42.0
    client["Frames Per Second"] = 100
    client["Exposure Time (seconds)"] = 0.05
    assert client["Frame Count"] == 5
    client["Frame Count"] = 3
    assert client["Exposure Time (seconds)"] == pytest.approx(0.03)
    assert TEMPERATURE_PROP in client.list_properties()
    client["Simulator Virtual Specimen"] = "Gold grating"
    assert twin.specimen_name == "Gold grating"
    spec = client.get_property_specifications("Exposure Mode")
    # deapi releases name these fields differently (options/valueType vs values/prop_allowable_type)
    options = getattr(spec, "options", None) or getattr(spec, "values", None)
    allowable = getattr(spec, "valueType", None) or getattr(spec, "prop_allowable_type", None)
    assert "Dark" in options and "Set" in getattr(allowable, "name", str(allowable))


def test_request_mapping(client, server, twin):
    client["Frames Per Second"] = 50
    client["Frame Count"] = 4
    client["Exposure Mode"] = "Dark"
    client["Hardware Binning X"] = 2
    client.set_hw_roi(0, 0, 64, 32)
    req = server.fake.request()
    assert req.exposure_mode == ExposureMode.DARK
    assert req.frame_time_s == pytest.approx(0.02) and req.total_frames == 4
    assert req.hw_binning == (2, 2)
    assert req.hw_roi == Roi(0, 0, 64, 32)
    assert req.camera_model == "StubCam"
    props = {
        "Frames Per Second": 10,
        "Frame Count": 2,
        "Exposure Mode": "Normal",
        "Hardware ROI Offset X": 0,
        "Hardware ROI Offset Y": 0,
        "Hardware ROI Size X": 96,
        "Hardware ROI Size Y": 64,
        "Hardware Binning X": 1,
        "Hardware Binning Y": 1,
        "Scan - Enable": "On",
        "Scan - Size X": 4,
        "Scan - Size Y": 3,
        "Scan - Camera Frames Per Point": 2,
        "Scan - Repeats": 1,
        "Scan - ROI Enable": "Off",
        "Grabbing - Frames Per Buffer": 1,
        "Scan - Type": "Raster",
    }
    r = request_from_properties(props.__getitem__, "X", (64, 96))
    assert r.hw_roi is None and r.scan.enabled and r.scan.size == (4, 3)
    assert r.total_frames == 4 * 3 * 2


def test_singleframe_dark_vs_normal_and_sum(client, twin):
    client["Frames Per Second"] = 100
    client["Frame Count"] = 3
    normal, attrs = _acquire(client, "singleframe_integrated")
    assert normal.shape == (64, 96) and normal.dtype == np.uint16
    assert attrs.frameCount == 3
    client["Exposure Mode"] = "Dark"
    dark, _ = _acquire(client, "singleframe_integrated")
    assert dark.mean() == pytest.approx(100, abs=2)
    assert normal.mean() > dark.mean() + 40
    client["Exposure Mode"] = "Normal"
    total, attrs = _acquire(client, "sumtotal")
    assert total.dtype == np.float32
    assert total.mean() == pytest.approx(3 * normal.mean(), rel=0.05)
    assert twin.requests[1].exposure_mode == ExposureMode.DARK
    assert not twin.requests[1].beam_reaches_detector


def test_gain_mode_is_flood(client):
    client["Exposure Mode"] = "Gain"
    flood, _ = _acquire(client)
    assert flood.mean() == pytest.approx(500, abs=5)
    assert flood.std() < 10  # no specimen, no bright column


def test_stage_moves_change_the_image(client, twin):
    a, _ = _acquire(client)
    twin.column.set_stage(x=30.0)
    twin.column.wait_idle()
    b, _ = _acquire(client)
    assert int(np.argmax(a.mean(axis=0))) == 0
    assert int(np.argmax(b.mean(axis=0))) == 30
    assert client["Instrument Stage Position X (micrometers)"] == 30.0


def test_hw_roi_and_binning_shape(client):
    client.set_hw_roi(16, 8, 64, 32)
    client["Hardware Binning X"] = 2
    client.update_image_size()
    image, attrs = _acquire(client)
    assert image.shape == (16, 32)
    assert (attrs.frameWidth, attrs.frameHeight) == (32, 16)


def test_scan_virtual_images(client, twin):
    client["Scan - Enable"] = "On"
    client["Scan - Size X"] = 6
    client["Scan - Size Y"] = 4
    client.update_scan_size()
    client.start_acquisition(1)
    while client.acquiring:
        time.sleep(0.01)
    vi, *_ = client.get_result("virtual_image0")
    assert vi.shape == (4, 6)
    assert np.all(vi > 0)
    assert twin.requests[-1].scan.enabled and twin.requests[-1].total_frames == 24
    haadf, *_ = client.get_result("external_image1")
    assert haadf.shape == (4, 6)
    assert haadf[3, 5] == pytest.approx(23)


def test_movie_buffer_has_real_frames(client):
    client["Frames Per Second"] = 100  # keeps Exposure Time, so set it before Frame Count
    client["Frame Count"] = 4
    client.start_acquisition(1, request_movie_buffer=True)
    info = client.get_movie_buffer_info()
    frames = []
    while True:
        status, _total, n, buffer = client.get_movie_buffer(
            None, info.headerBytes + info.imageBufferBytes, info.framesInBuffer
        )
        if status.name != "OK":
            break
        assert (info.imageW, info.imageH) == (96, 64)
        data = np.frombuffer(buffer, np.uint16, offset=info.headerBytes).reshape(n, 64, 96)
        frames.extend(data)
    assert len(frames) == 4
    assert np.mean(frames) > 120


def test_temperature_property_drives_holder(client, twin):
    assert client[TEMPERATURE_PROP] == pytest.approx(25.0)
    client[TEMPERATURE_PROP] = 400.0  # a NoHolder is replaced by a heating holder
    time.sleep(0.1)
    assert twin.holder.state().target_c == 400.0
    assert client[TEMPERATURE_PROP] == pytest.approx(400.0, abs=1.0)


def test_state_is_shared_across_connections(server):
    c1 = deapi.Client()
    c1.usingMmf = False
    c1.connect(port=server.port)
    c1["Frame Count"] = 7
    c1.disconnect()
    c2 = deapi.Client()
    c2.usingMmf = False
    c2.connect(port=server.port)
    assert c2["Frame Count"] == 7
    c2.disconnect()


def test_stop_acquisition(client):
    client["Frame Count"] = 100000
    client.start_acquisition(1)
    time.sleep(0.05)
    assert client.stop_acquisition()
    deadline = time.time() + 5
    while client.acquiring and time.time() < deadline:
        time.sleep(0.01)
    assert not client.acquiring


def test_zero_acquisitions_is_live_until_stopped(client):
    """start_acquisition(0) is live view: it repeats until stopped (Ground Crew relies on it)."""
    client["Frame Count"] = 1
    client.start_acquisition(0)
    time.sleep(0.3)
    assert client.acquiring  # a single one-frame acquisition would have finished by now
    time.sleep(0.3)
    assert client.acquiring
    assert client.stop_acquisition()
    deadline = time.time() + 5
    while client.acquiring and time.time() < deadline:
        time.sleep(0.01)
    assert not client.acquiring


def test_a_live_scan_follows_the_offset_and_zoom_as_they_change(client, twin):
    """The coils are what a client steers while scanning (probe, drift, pan, zoom)."""
    client["Scan - Enable"] = "On"
    client["Scan - Size X"] = 4  # the smallest scan DE-Server accepts
    client["Scan - Size Y"] = 4
    client["Scan - Voltage H Max"] = 0.5
    client.start_acquisition(0)
    try:
        deadline = time.time() + 5
        while not twin.requests and time.time() < deadline:
            time.sleep(0.01)
        assert twin.requests[-1].scan.voltage_h == (-1.0, 0.5)
        client["Scan - Pixel Offset X"] = 2.5
        client["Scan - Voltage V Min"] = -0.25
        while twin.requests[-1].scan.pixel_offset != (2.5, 0.0) and time.time() < deadline:
            time.sleep(0.01)
        assert twin.requests[-1].scan.pixel_offset == (2.5, 0.0)
        assert twin.requests[-1].scan.voltage_v == (-0.25, 1.0)
        assert twin.requests[-1].total_frames == 16  # the grid itself does not change mid-scan
    finally:
        client.stop_acquisition()
        while client.acquiring:
            time.sleep(0.01)


def _offset_per_repeat(server, twin, *, output_buffer_ms, change_at_frame=20):
    """Scan 8x8 ten times, change the pixel offset at ``change_at_frame``, and return the
    offset each repeat scanned with."""
    face = server.fake
    face["Scan - Enable"] = "On"
    face["Scan - Size X"] = face["Scan - Size Y"] = 8
    face["Scan - Repeats"] = 10
    face["Frames Per Second"] = 40
    face["Scan - Output Buffer (milliseconds)"] = output_buffer_ms
    frames = twin.frames

    def changing(request, pace=False, stop=None):
        for raw, meta in frames(request, pace=pace, stop=stop):
            if sum(r.total_frames for r in twin.requests[:-1]) + meta.frame_index == change_at_frame:
                face["Scan - Pixel Offset X"] = 3.0
            yield raw, meta

    twin.frames = changing
    face._start(_start_command(1))
    face._acq.thread.join(timeout=10)
    return [r.scan.pixel_offset[0] for r in twin.requests]


def _start_command(n):
    from deapi.buffer_protocols import pb

    command = pb.DEPacket()
    c = command.command.add()
    c.command_id = 0
    p = c.parameter.add()
    p.type, p.p_int = pb.AnyParameter.P_INT, n
    return command


def test_an_offset_lands_one_output_buffer_later_on_a_repeat_boundary(server, twin):
    """DE-Server queues the output buffer ahead of the beam (6.4 s at 40 fps is 256 points):
    an offset written at point 20 is in the points written at 20 + 256, i.e. from repeat 5."""
    assert _offset_per_repeat(server, twin, output_buffer_ms=6400.0) == [0.0] * 5 + [3.0] * 5


def test_by_default_an_offset_waits_out_half_the_fifo(server, twin):
    """Without an output buffer set, DE-Server queues 524,288 points: a 640-point scan never
    sees an offset written while it runs."""
    assert _offset_per_repeat(server, twin, output_buffer_ms=0.0) == [0.0] * 10


def test_real_twin_if_available():
    try:
        from de_twin.twin import DigitalTwin

        twin = DigitalTwin(camera="DE16")
        twin.detector.output_shape(twin.request())
    except Exception as exc:
        pytest.skip(f"DigitalTwin not ready: {exc}")
    with TwinDeapiServer(twin, port=0, pace=False) as srv:
        c = deapi.Client()
        c.usingMmf = False
        c.connect(port=srv.port)
        c["Simulator Auto References"] = "Off"
        c.set_hw_roi(0, 0, 512, 512)
        c.update_image_size()
        c["Frame Count"] = 1
        raw, _ = _acquire(c)  # no references yet: uncorrected, like a fresh DE-Server
        c["Exposure Mode"] = "Dark"
        c["Frame Count"] = 10
        dark, _ = _acquire(c)  # stored as the dark reference
        c["Exposure Mode"] = "Normal"
        c["Frame Count"] = 1
        corrected, _ = _acquire(c)
        c.disconnect()
    assert raw.shape == dark.shape == corrected.shape
    assert raw.mean() > dark.mean() + 50
    # dark-subtracted now: the offset is gone, the beam signal remains
    assert corrected.mean() == pytest.approx(raw.mean() - dark.mean(), rel=0.1)


def test_deapi_own_loop_with_patched_factory(twin, monkeypatch):
    """deapi's initialize_server loop, patched like Ground Crew's launcher patches inp_file."""
    import socket

    from deapi.simulated_server import initialize_server

    from de_twin.faces.deapi_server import make_factory

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(initialize_server, "FakeServer", make_factory(twin, pace=False))
    monkeypatch.setattr("sys.argv", ["x"])
    threading.Thread(target=initialize_server.main, args=(port,), daemon=True).start()
    c = deapi.Client()
    c.usingMmf = False
    for _ in range(100):
        try:
            c.connect(port=port)
            break
        except OSError:
            time.sleep(0.05)
    assert c["Camera Name"] == "StubCam"
    image, _ = _acquire(c)
    assert image.shape == (64, 96) and image.mean() > 120
    c.disconnect()


def test_specimen_pixel_size_includes_binning():
    """DE-Server reports the pixel size of a binned pixel (ParseEmMetaData multiplies by the
    hardware and software binning); clients converting pixels to stage moves rely on it."""
    from de_twin.twin import DigitalTwin

    twin = DigitalTwin(camera="DESim")
    with TwinDeapiServer(twin, port=0, pace=False) as srv:
        c = deapi.Client()
        c.usingMmf = False
        c.connect(port=srv.port)
        unbinned = c["Specimen Pixel Size X (nanometers)"]
        c["Hardware Binning X"] = 2
        binned = c["Specimen Pixel Size X (nanometers)"]
        c.disconnect()
    assert unbinned > 0
    assert binned == pytest.approx(2 * unbinned)


def test_a_fresh_face_exposes_for_half_a_second():
    """One frame per image reads as noise on a DE camera: a frame is sparse by design."""
    from de_twin.faces import deapi_protocol as face_mod
    from de_twin.twin import DigitalTwin

    with TwinDeapiServer(DigitalTwin(camera="DESim"), port=0, pace=False) as srv:
        c = deapi.Client()
        c.usingMmf = False
        c.connect(port=srv.port)
        exposure = float(c["Exposure Time (seconds)"])
        c.disconnect()
    assert exposure == pytest.approx(face_mod.DEFAULT_EXPOSURE_S, rel=0.05)


def test_port_is_known_after_stop(twin):
    srv = TwinDeapiServer(twin, port=0, pace=False).start()
    port = srv.port
    assert port > 0
    srv.stop()
    assert srv.port == port  # no getsockname on a closed socket


def test_simulator_source_marks_the_frames(client):
    from de_twin import __version__

    assert client["Simulator Source"] == f"de-twin {__version__}"


def test_apollo_super_resolution_over_deapi():
    from de_twin.clock import ManualClock
    from de_twin.twin import DigitalTwin

    twin = DigitalTwin("Dense Au on holey C", camera="Apollo", clock=ManualClock())
    with TwinDeapiServer(twin, port=0, pace=False) as srv:
        c = deapi.Client()
        c.usingMmf = False
        c.connect(port=srv.port)
        c["Simulator Auto References"] = "Off"
        c.set_hw_roi(0, 0, 256, 256)
        c.update_image_size()
        c["Frame Count"] = 1
        std, _ = _acquire(c, "singleframe_integrated")
        c["Centroiding Mode"] = "Super-resolution"
        c.update_image_size()
        assert int(c["Image Size X (pixels)"]) == 512
        sr, _ = _acquire(c, "singleframe_integrated")
        c.disconnect()
    assert std.shape == (256, 256) and sr.shape == (512, 512)
    assert srv.fake.request().super_resolution is True


def test_specimen_pixel_size_stays_nominal_behind_a_post_column_magnification():
    """The column does not know the camera sits behind a post-column magnification, so the
    served pixel size is the calibration's even though the rendered pixel is smaller."""
    from de_twin.optics import OpticsConfig
    from de_twin.twin import DigitalTwin

    served = []
    for post_magnification in (1.0, 1.4):
        twin = DigitalTwin(camera="DESim", optics_config=OpticsConfig(post_magnification=post_magnification))
        with TwinDeapiServer(twin, port=0, pace=False) as srv:
            c = deapi.Client()
            c.usingMmf = False
            c.connect(port=srv.port)
            served.append(c["Specimen Pixel Size X (nanometers)"])
            c.disconnect()
    assert served[0] > 0
    assert served[1] == served[0]
