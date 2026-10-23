"""deapi face: a DE-Server impersonation (deapi fake server) that renders through the twin.

:class:`TwinFakeServer` subclasses ``deapi.simulated_server.fake_server.FakeServer``
(so every property from deapi's ``prop_dump.json`` and every command it already
answers keeps working) and replaces the synthetic data with the twin:

* ``start_acquisition`` maps the deapi properties to an
  :class:`~de_twin.state.AcquisitionRequest` (:func:`request_from_properties`)
  and renders ``twin.frames(request)`` on a background thread, honouring
  ``Exposure Mode`` (Dark -> blanked frames, Gain/Vacuum -> flood, both done
  by the twin), ROI/binning, frames per second, frame count and STEM scans.
* References work like DE-Server: an acquisition in ``Exposure Mode`` Dark or
  Gain stores a dark/gain reference (keyed by ROI, binning and frame time), and
  every other acquisition is dark/gain corrected with the matching references.
  ``Simulator Auto References`` (default On) measures missing references on
  first use. Turn it Off to exercise reference workflows.
* ``get_result``: single-frame types (2..8) return the latest (corrected) frame,
  ``SUMINTERMEDIATE``/``SUMTOTAL``/``CUMULATIVE`` the sum of the frames so far,
  ``VIRTUAL_MASK*`` the masks, ``VIRTUAL_IMAGE*`` virtual detector images
  integrated from the real frames through the masks (``Sum`` / ``Difference``),
  ``EXTERNAL_IMAGE*`` the twin's noiseless annular (HAADF) image.
* ``get_movie_buffer`` streams the real frames (when the acquisition asked for a
  movie buffer).
* The twin is shared by every connection: one :class:`TwinFakeServer` instance
  per twin, whose socket is thread-local, so properties persist across
  reconnects like on a real DE-Server.

Extra properties merged into the property table (like Ground Crew's launcher):
``Simulator Virtual Specimen Temperature (C)`` (drives the twin's holder),
``Simulator Virtual Specimen`` (preset name), ``Scan - Pixel Offset X/Y``,
twin clock/holder read-outs and DE-Server's read-only ``Instrument ...``
properties mirroring the column (``Instrument Stage Position X (micrometers)``,
``Instrument Project Magnification``, ``Instrument Metadata``, ...).

Extra commands answered (beyond deapi's fake server): ``SET_HW_ROI`` (16),
``SET_SCAN_SIZE`` (27), ``SET_SCAN_ROI`` (28), ``SET_SCAN_XY_ARRAY`` (32),
``GET_PROPERTY_SPECIFICATIONS`` (35) and ``GET_GPU_FRAMES`` (44): the twin's frames in
shared GPU memory, as DE-Server shares its movie buffers (:mod:`de_twin.faces.gpu_frames`).

A scan acquisition runs a repeat at a time, and each repeat scans with the coil settings
(pixel offset, voltages, rotation, flips) DE-Server would have written for it:
``Scan - Output Buffer (milliseconds)`` of scan output is queued ahead of the beam, so a
change reaches the beam that much later, on a repeat boundary (:class:`_CoilTimeline`).

Launcher (deapi-compatible CLI, "started .... " banner)::

    python -m de_twin.faces.deapi_server 13240 --specimen "Dense Au on holey C" --camera DE16 --soap-port 5002
"""

from __future__ import annotations

import argparse
import bisect
import itertools
import logging
import socket as _socketmod
import struct
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import replace
from typing import Any, Callable, Optional

import numpy as np

from ..state import AcquisitionRequest, ExposureMode, Roi, ScanRequest

try:  # deapi is an optional dependency of the twin
    from deapi.buffer_protocols import pb
    from deapi.simulated_server.fake_server import FakeServer, Property, add_parameter
    from deapi.version import commandVersion
    try:
        from deapi.version import fake_server_software_version as _SERVER_VERSION
    except ImportError:  # older deapi
        _SERVER_VERSION = "2.8.0.11901"
except ImportError:  # pragma: no cover - exercised only without deapi
    pb = None
    FakeServer = object  # type: ignore[assignment,misc]
    Property = object  # type: ignore[assignment,misc]
    add_parameter = None
    commandVersion = 0
    _SERVER_VERSION = "2.8.0.11901"

log = logging.getLogger(__name__)


def _sends_scan_pattern_index(server_version: str) -> bool:
    """DE-Server 2.8.0 build 12073+ appends ``current_scan_pattern_idx`` to GET_RESULT's
    attributes (command version 16). The face reports the installed deapi's fake-server
    version, so it answers the way that DE-Server build would."""
    try:
        major, minor, patch, build = (int(p) for p in str(server_version).split(".")[:4])
    except ValueError:
        return False
    return (major, minor, patch) >= (2, 8, 0) and build >= 12073 and commandVersion >= 16

EXPOSURE_MODES = {
    "dark": ExposureMode.DARK, "trial": ExposureMode.TRIAL, "gain": ExposureMode.GAIN,
    "normal": ExposureMode.NORMAL, "raw": ExposureMode.RAW, "vacuum": ExposureMode.VACUUM,
}
PIXEL_DTYPES = {1: np.uint8, 5: np.uint16, 13: np.float32}

# command ids (FakeServer has some of these; the client's full list in client.py)
SET_HW_ROI = 16
SET_SCAN_SIZE = 27
SET_SCAN_ROI = 28
SET_SCAN_XY_ARRAY = 32
GET_PROPERTY_SPECIFICATIONS = 35
GET_GPU_FRAMES = 44

#: Frames per shared GPU buffer, and the longest a frame waits for its buffer to fill
#: before it is handed over anyway (a live view should not wait for 64 frames).
GPU_BUFFER_FRAMES = 64
GPU_PUBLISH_S = 0.02

TEMPERATURE_PROP = "Simulator Virtual Specimen Temperature (C)"

# DE-Server "Instrument Project Name" per (function mode, TEM/STEM) (mag.yaml projects)
_PROJECT_NAMES = {(0, 0): "MAG1", (1, 0): "MAG2", (2, 0): "Low Mag", (3, 0): "SMAG",
                  (4, 0): "Diff Mag", (2, 1): "HR_STEM", (4, 1): "NBD_STEM"}


def _norm(name: str) -> str:
    return name.replace(" ", "_").lower().replace("(", "").replace(")", "")


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("on", "true", "1", "yes")


#: The exposure a fresh face starts with, in seconds (see `_frame_count`).
DEFAULT_EXPOSURE_S = 0.5

class DynamicProperty(Property):  # type: ignore[misc]
    """A property whose value is computed (and optionally applied) by callables."""

    def __init__(self, name: str, getter: Callable[[], Any], setter: Optional[Callable[[Any], None]] = None,
                 *, data_type: str = "String", category: str = "Twin", options: str = "",
                 value_type: Optional[str] = None, default_value: str = "", server=None):
        super().__init__(name=name, value="", data_type=data_type, category=category,
                         value_type=value_type or ("ReadOnly" if setter is None else "AllowAll"),
                         options=options, default_value=default_value, server=server)
        self._getter = getter
        self._setter = setter

    @property  # type: ignore[override]
    def value(self):
        v = self._getter()
        if self.data_type == "Integer":
            return str(int(round(float(v))))
        if self.data_type == "Float":
            return str(float(v))
        return str(v)

    @value.setter
    def value(self, value):
        if self._setter is None:
            log.warning("Property %s is read only", self.name)
            return
        self._setter(value)


#: ``Simulator EBSD <label>`` deapi properties -> EbsdGeometry fields.
EBSD_PROPERTIES = {
    "enabled": "Enable",
    "sample_tilt_deg": "Sample Tilt (degrees)",
    "detector_tilt_deg": "Detector Tilt (degrees)",
    "working_distance_mm": "Working Distance (mm)",
    "pcx": "Pattern Center X",
    "pcy": "Pattern Center Y",
    "pcz": "Pattern Center Z",
    "probe_current_na": "Probe Current (nA)",
    "step_um": "Default Step (micrometers)",
    "band_contrast": "Band Contrast",
    "background": "Background",
    "boundary_blend": "Boundary Blend",
}


#: Read-only property naming the simulator behind the frames (``"de-twin <version>"``), so
#: a client can stamp provenance on what it saves. A real DE-Server has no such property.
SIMULATOR_SOURCE_PROPERTY = "Simulator Source"


def model_counting(twin) -> bool:
    return bool(getattr(twin.detector.model, "hardware_counting", False))


#: DE-Server's scan coil properties, with DE-Server's defaults.
SCAN_COIL_PROPERTIES = {
    "Scan - Voltage H Min": -1.0, "Scan - Voltage H Max": 1.0,
    "Scan - Voltage V Min": -1.0, "Scan - Voltage V Max": 1.0,
    "Scan - Pixel Offset X": 0.0, "Scan - Pixel Offset Y": 0.0,
    "Scan - Rotation": 0.0,
}
SCAN_FLIP_PROPERTIES = ("Scan - Flip Horizontally", "Scan - Flip Vertically")
#: How much scan output DE-Server queues ahead of the beam, in milliseconds; 0 is its
#: default, two quarter-FIFO blocks (DEFAULT_QUEUED_POINTS).
OUTPUT_BUFFER_PROPERTY = "Scan - Output Buffer (milliseconds)"
DEFAULT_QUEUED_POINTS = 2 * (1048576 // 4)


def scan_coil_settings(get: Callable[[str], Any]) -> dict:
    """The ScanRequest fields the scan coils take from properties (de_twin.optics.scan_coils).

    These are what a client changes WHILE scanning (pan, zoom, drift correction, a moving
    probe), so a running acquisition re-reads them between repetitions."""
    def number(name):
        try:
            return float(get(name))
        except (KeyError, TypeError, ValueError):
            return SCAN_COIL_PROPERTIES[name]

    def flag(name):
        try:
            return _to_bool(get(name))
        except KeyError:
            return False

    return {
        "voltage_h": (number("Scan - Voltage H Min"), number("Scan - Voltage H Max")),
        "voltage_v": (number("Scan - Voltage V Min"), number("Scan - Voltage V Max")),
        "pixel_offset": (number("Scan - Pixel Offset X"), number("Scan - Pixel Offset Y")),
        "rotation_deg": number("Scan - Rotation"),
        "flip": tuple(flag(name) for name in SCAN_FLIP_PROPERTIES),
    }


def queued_scan_points(get: Callable[[str], Any], fps: float) -> int:
    """How many scan points DE-Server has written ahead of the beam, so how many points
    later a coil setting written now reaches it (ScanController's analog output: four
    blocks of at least 2 ms when ``Scan - Output Buffer (milliseconds)`` is set)."""
    try:
        ms = float(get(OUTPUT_BUFFER_PROPERTY))
    except (KeyError, TypeError, ValueError):
        ms = 0.0
    if ms <= 0:
        return DEFAULT_QUEUED_POINTS
    block = max(1, int(0.002 * fps), int(ms * 1e-3 * fps) // 4)
    return min(4 * block, DEFAULT_QUEUED_POINTS)


class _CoilTimeline:
    """The coil settings a client wrote during one acquisition, by the scan point the beam
    was on when it wrote them.

    DE-Server writes scan output ``queued`` points ahead of the beam and adds the offset
    (and the rest of the coil settings) as it writes, so repeat ``r`` of an N-point
    pattern scans with the settings as of point ``r * N - queued``: a change made now
    lands ``queued`` points later, on a repeat boundary.
    """

    def __init__(self, settings: dict):
        self._points = [0]
        self._settings = [settings]

    def record(self, point: int, settings: dict) -> None:
        if settings != self._settings[-1]:
            self._points.append(point)
            self._settings.append(settings)

    def at(self, point: int) -> dict:
        return self._settings[max(0, bisect.bisect_right(self._points, point) - 1)]


def request_from_properties(get: Callable[[str], Any], camera_model: str,
                            sensor_shape: tuple[int, int], *,
                            scan_points: Optional[list] = None,
                            acquisition_index: int = 0) -> AcquisitionRequest:
    """Map deapi / DE-Server property values to an AcquisitionRequest.

    ``get(name)`` returns a property value (string or number).
    """
    h, w = sensor_shape
    fps = max(float(get("Frames Per Second")), 1e-6)
    frame_count = max(1, int(float(get("Frame Count"))))
    mode = EXPOSURE_MODES.get(str(get("Exposure Mode")).strip().lower(), ExposureMode.NORMAL)
    ox, oy = int(float(get("Hardware ROI Offset X"))), int(float(get("Hardware ROI Offset Y")))
    sx, sy = int(float(get("Hardware ROI Size X"))), int(float(get("Hardware ROI Size Y")))
    roi = None if (ox, oy, sx, sy) == (0, 0, w, h) else Roi(ox, oy, sx, sy)
    bx = int(float(get("Hardware Binning X") or 1))
    by = int(float(get("Hardware Binning Y") or 1))
    scan = ScanRequest(enabled=False)
    if _to_bool(get("Scan - Enable")):
        nx, ny = int(float(get("Scan - Size X"))), int(float(get("Scan - Size Y")))
        fpp = max(1, int(float(get("Scan - Camera Frames Per Point") or 1)))
        repeats = max(1, int(float(get("Scan - Repeats") or 1)))
        sroi = None
        if _to_bool(get("Scan - ROI Enable")):
            sroi = Roi(int(float(get("Scan - ROI Offset X"))), int(float(get("Scan - ROI Offset Y"))),
                       int(float(get("Scan - ROI Size X"))), int(float(get("Scan - ROI Size Y"))))
        points = None
        if scan_points and str(get("Scan - Type")) in ("XY File", "XY Array"):
            points = [tuple(map(int, p)) for p in scan_points]
        elif sroi is not None:
            points = [(sroi.x + i, sroi.y + j) for j in range(sroi.h) for i in range(sroi.w)]
        scan = ScanRequest(enabled=True, size=(nx, ny), dwell_s=fpp / fps, frames_per_point=fpp,
                           roi=sroi, repeats=repeats, points=points, **scan_coil_settings(get))
        n_points = len(points) if points else nx * ny
        frame_count = n_points * fpp * repeats
    return AcquisitionRequest(
        camera_model=camera_model, exposure_mode=mode, frame_time_s=1.0 / fps,
        total_frames=frame_count,
        frames_per_buffer=max(1, int(float(get("Grabbing - Frames Per Buffer") or 1))),
        hw_roi=roi, hw_binning=(bx, by), scan=scan, acquisition_index=acquisition_index,
        super_resolution=_super_resolution(get))


def _super_resolution(get: Callable[[str], Any]) -> bool:
    """DE-Server's "Centroiding Mode" asks for super-resolution (counting cameras only)."""
    try:
        return "super" in str(get("Centroiding Mode") or "").lower()
    except Exception:  # noqa: BLE001 - an integrating camera has no such property
        return False


def _resample(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize to (h, w), preserving dtype."""
    if img.shape == shape:
        return img
    ys = (np.arange(shape[0]) * img.shape[0] / shape[0]).astype(int)
    xs = (np.arange(shape[1]) * img.shape[1] / shape[1]).astype(int)
    return img[ys][:, xs]


def _gpu_image_dtype():
    from .gpu_frames import IMAGE

    return IMAGE


def _convert(img: np.ndarray, pixel_format: int) -> np.ndarray:
    dtype = PIXEL_DTYPES[pixel_format]
    if dtype is np.float32:
        return np.ascontiguousarray(img, dtype=np.float32)
    info = np.iinfo(dtype)
    return np.ascontiguousarray(np.clip(np.rint(img) if img.dtype.kind == "f" else img,
                                        info.min, info.max).astype(dtype))


class _Acquisition:
    """One start_acquisition: frames rendered by a background thread."""

    def __init__(self, server: "TwinFakeServer", request: AcquisitionRequest, n_acq: int,
                 movie: bool):
        self.request = request
        # 0 acquisitions = repeat until stopped (live view), as DE-Server does
        self.n_acq = max(0, int(n_acq))
        self.continuous = self.n_acq == 0
        self.movie = movie
        self.stop = threading.Event()
        self.cond = threading.Condition()
        self.last: Optional[np.ndarray] = None
        self.last_meta = None
        self.sum: Optional[np.ndarray] = None
        self.count = 0
        self.index = 0  # acquisition index within n_acq
        self.movie_frames: deque = deque()
        self.movie_next = 0
        self.error: Optional[str] = None
        self.refs = None
        self.done = False
        self.started = time.time()
        self.gpu_images: list = []
        self.gpu_info: list = []
        self.gpu_batch_started = 0.0
        scan = request.scan
        self.virtual = None
        if scan.enabled:
            nx, ny = scan.size
            self.virtual = np.zeros((5, ny, nx), np.float64)
        self._server = server
        self.thread = threading.Thread(target=self._run, name="TwinFakeServer-acq", daemon=True)

    def start(self) -> "_Acquisition":
        self.thread.start()
        return self

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def _prepare_references(self) -> None:
        """Pick up (or measure) dark/gain references like DE-Server's RefFinder."""
        self.refs = None
        processor = getattr(self._server.twin, "processor", None)
        if processor is None or self.request.exposure_mode in (ExposureMode.DARK, ExposureMode.GAIN):
            return
        self.refs = processor.lookup(self.request)
        if getattr(self._server, "auto_references", False) and (self.refs is None or self.refs.gain is None):
            self.refs = processor.references(self.request)

    def _store_references(self) -> None:
        processor = getattr(self._server.twin, "processor", None)
        if processor is None or not self.count or self.stop.is_set():
            return
        mean = (self.sum / self.count).astype(np.float32)
        mode = self.request.exposure_mode
        if mode == ExposureMode.DARK:
            processor.store_dark(self.request, mean, self.count)
        elif mode == ExposureMode.GAIN:
            try:
                processor.store_gain(self.request, mean, self.count)
            except RuntimeError as exc:
                log.warning("gain reference not stored: %s", exc)

    def _run(self) -> None:
        srv = self._server
        try:
            self._prepare_references()
            for a in (itertools.count() if self.continuous else range(self.n_acq)):
                self.index = a
                if self.continuous and a:
                    with self.cond:
                        self.sum = None  # the sum is per repetition, like one live buffer
                    # a client steers the probe while it scans: take the coils' settings now
                    if self.request.scan.enabled:
                        self.request = replace(self.request, scan=replace(
                            self.request.scan, **scan_coil_settings(srv.__getitem__)))
                if self.request.scan.enabled:
                    self._scan()
                else:
                    for raw, meta in srv.twin.frames(self.request, pace=srv.pace, stop=self.stop):
                        self._add(raw, meta)
                        if self.stop.is_set():
                            break
                if self.stop.is_set():
                    break
            self._share_gpu_frames()
            self._store_references()
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            log.error("twin acquisition failed:\n%s", traceback.format_exc())
        finally:
            with self.cond:
                self.done = True
                self.cond.notify_all()

    def _scan(self) -> None:
        """One scan acquisition, a repeat at a time, each repeat with the coil settings
        DE-Server would have written for it (see :class:`_CoilTimeline`)."""
        srv, request = self._server, self.request
        scan = request.scan
        repeats = max(1, scan.repeats)
        frames_per_repeat = request.total_frames // repeats
        points_per_repeat = max(1, frames_per_repeat // max(1, scan.frames_per_point))
        queued = queued_scan_points(srv.__getitem__, 1.0 / max(request.frame_time_s, 1e-9))
        timeline = _CoilTimeline({name: getattr(scan, name) for name in scan_coil_settings(srv.__getitem__)})
        frames = 0
        for repeat in range(repeats):
            settings = timeline.at(repeat * points_per_repeat - queued)
            this_repeat = replace(request, total_frames=frames_per_repeat,
                                  scan=replace(scan, repeats=1, **settings))
            for raw, meta in srv.twin.frames(this_repeat, pace=srv.pace, stop=self.stop):
                self._add(raw, meta)
                frames += 1
                timeline.record(frames // max(1, scan.frames_per_point), scan_coil_settings(srv.__getitem__))
                if self.stop.is_set():
                    return

    def _share_gpu_frames(self) -> None:
        """Hand the frames gathered so far to a GPU frames client, if one is attached."""
        shared = self._server._shared_frames
        if not self.gpu_images or shared is None:
            self.gpu_images, self.gpu_info = [], []
            return
        images = np.stack(self.gpu_images).astype(np.float32, copy=False)
        info = np.zeros(len(images), _gpu_image_dtype())
        info["frame"], info["scanX"], info["scanY"] = np.array(self.gpu_info).T
        info["valid"] = 1
        self.gpu_images, self.gpu_info = [], []
        shared.allocate(GPU_BUFFER_FRAMES, *images.shape[1:])
        shared.publish(images, info, stop=self.stop)

    def _gather_gpu_frame(self, frame: np.ndarray, meta) -> None:
        shared = self._server._shared_frames
        if shared is None or not shared.attached:
            return
        if not self.gpu_images:
            self.gpu_batch_started = time.monotonic()
        point = getattr(meta, "scan_point", None)
        self.gpu_images.append(frame)
        self.gpu_info.append((self.count - 1, *(point if point is not None else (-1, -1))))
        if len(self.gpu_images) >= GPU_BUFFER_FRAMES or time.monotonic() - self.gpu_batch_started >= GPU_PUBLISH_S:
            self._share_gpu_frames()

    def _add(self, raw: np.ndarray, meta) -> None:
        if self.refs is not None:
            raw = self._server.twin.processor.correct(raw, self.refs)
        masks = self._server._masks_for(raw.shape) if self.virtual is not None else None
        vi_vals = None
        if masks is not None and getattr(meta, "scan_point", None) is not None:
            f = raw.astype(np.float64, copy=False)
            vi_vals = []
            for k, (pos, neg, calc) in enumerate(masks):
                s = float(f[pos].sum())
                if calc == "difference":
                    s -= float(f[neg].sum())
                elif neg is not None:
                    s += float(f[neg].sum())
                vi_vals.append(s)
        with self.cond:
            self.last = raw
            self.last_meta = meta
            self.sum = raw.astype(np.float64) if self.sum is None else self.sum + raw
            self.count += 1
            if self.movie:
                self.movie_frames.append(raw)
            if vi_vals is not None:
                ix, iy = meta.scan_point
                ny, nx = self.virtual.shape[1:]
                if 0 <= iy < ny and 0 <= ix < nx:
                    self.virtual[:, iy, ix] += vi_vals
            self.cond.notify_all()
        self._gather_gpu_frame(raw, meta)

    def wait_first(self, timeout: float) -> bool:
        with self.cond:
            return self.cond.wait_for(lambda: self.count > 0 or self.done, timeout)


class TwinFakeServer(FakeServer):  # type: ignore[misc,valid-type]
    """deapi's FakeServer, rendering through a de_twin twin.

    ``twin`` needs: ``lock``, ``column``, ``detector`` (``.model`` with ``name``,
    ``sensor_shape``; optionally ``output_shape(request)``), ``holder``,
    ``frames(request, pace=, stop=)``; optionally ``virtual_image``,
    ``set_specimen``, ``calibration``, ``clock``.
    """

    #: class-level default twin, for code that instantiates the class with
    #: deapi's own signature (e.g. a patched ``initialize_server.FakeServer``)
    twin = None
    pace = True

    def __init__(self, dataset: str = "grains", socket=None, *, twin=None, pace: Optional[bool] = None,
                 default_fps: float = 40.0):
        if pb is None:
            raise ImportError("TwinFakeServer needs the deapi package")
        self._tls = threading.local()
        self.twin = twin if twin is not None else type(self).twin
        if self.twin is None:
            raise ValueError("TwinFakeServer needs a twin (pass twin= or set TwinFakeServer.twin)")
        if pace is not None:
            self.pace = pace
        self._acq: Optional[_Acquisition] = None
        self._acq_count = 0
        self._shared_frames = None  # de_twin.faces.gpu_frames.SharedFrames, made on first attach
        self._scan_points: Optional[list] = None
        self._specimen_name = ""
        self._mask_cache: dict = {}
        self._cmd_lock = threading.RLock()
        super().__init__(dataset=dataset, socket=socket)
        self._install_properties(default_fps)

    # ------------------------------------------------------------ socket
    @property
    def socket(self):
        return getattr(self._tls, "socket", None)

    @socket.setter
    def socket(self, value):
        self._tls.socket = value

    # ------------------------------------------------------------ properties
    def _put(self, prop) -> None:
        prop.server = self
        self._values[_norm(prop.name)] = prop

    def _plain(self, name, value, data_type="String", value_type="AllowAll", options="", category="Twin"):
        self._put(Property(name=name, value=str(value), data_type=data_type, category=category,
                           value_type=value_type, options=options, default_value=str(value), server=self))

    def _install_properties(self, default_fps: float) -> None:
        model = self.twin.detector.model
        h, w = (int(v) for v in model.sensor_shape)
        max_fps = float(getattr(model, "max_fps", 0) or 0) or (
            1.0 / float(getattr(model, "full_frame_time_s", 0.01) or 0.01))
        v = self._values
        v["camera_name"]._value = str(model.name)
        for axis, n in (("x", w), ("y", h)):
            v[f"sensor_size_{axis}_pixels"]._value = str(n)
            v[f"sensor_size_{axis}_pixels"].default_value = str(n)
            v[f"hardware_roi_size_{axis}"]._value = str(n)
            v[f"hardware_roi_size_{axis}"].options = f"1.0, {float(n)}, '1 - {n}'"
            v[f"hardware_roi_offset_{axis}"].options = f"0.0, {float(n - 1)}, '0 - {n - 1}'"
        v["exposure_mode"].options = "'Dark', 'Gain', 'Normal*', 'Trial', 'Vacuum'"
        # hardware binning: X also sets Y (deapi's set_also passes a sympy float that its
        # own Set check then rejects, so the coupling is done here)
        self._hw_bin = [1, 1]

        def set_bin(axis):
            def setter(value):
                b = int(float(value))
                if b not in (1, 2, 4):
                    log.warning("hardware binning %s not supported", value)
                    return
                if axis == 0:
                    self._hw_bin[:] = [b, b]
                else:
                    self._hw_bin[1] = b
            return setter

        for i, axis in enumerate("XY"):
            self._put(DynamicProperty(f"Hardware Binning {axis}", (lambda i=i: self._hw_bin[i]),
                                      set_bin(i), data_type="Integer", value_type="Set",
                                      options="'1*', '2'", default_value="1", category="Basic"))
        # exposure: frames per second / frame count / exposure time stay consistent
        self._fps = min(float(default_fps), max_fps * 10)
        # A 0.5 s exposure to begin with, not one frame: a DE frame is sparse by
        # design (a pixel holds a few electrons), and an image is the sum of frames
        # over the exposure. One frame per image reads as pure noise.
        self._frame_count = max(1, int(round(DEFAULT_EXPOSURE_S * self._fps)))

        def set_fps(value):
            exposure = self._frame_count / self._fps
            self._fps = min(max(float(value), 0.06), 1e6)
            self._frame_count = max(1, int(round(exposure * self._fps)))

        def set_count(value):
            self._frame_count = max(1, int(float(value)))

        def set_exposure(value):
            self._frame_count = max(1, int(round(float(value) * self._fps)))

        self._put(DynamicProperty("Frames Per Second", lambda: self._fps, set_fps, data_type="Float",
                                  value_type="Range", options="0.06, 1000000.0, '0.06 - 1000000.0'",
                                  category="Basic"))
        self._put(DynamicProperty("Frame Count", lambda: self._frame_count, set_count,
                                  data_type="Integer", value_type="Range",
                                  options="1.0, 100000000.0, '1 - 100000000'", category="Basic"))
        self._put(DynamicProperty("Exposure Time (seconds)", lambda: self._frame_count / self._fps,
                                  set_exposure, data_type="Float", value_type="Range",
                                  options="0.0, 1000000.0, '0.0 - 1000000.0'", category="Basic"))
        # twin extras
        self._put(DynamicProperty(TEMPERATURE_PROP, self._get_temperature, self._set_temperature,
                                  data_type="Float", category="Simulator"))
        self._put(DynamicProperty("Simulator Virtual Specimen", lambda: self._specimen_name,
                                  self._set_specimen, category="Simulator"))
        for name, value in SCAN_COIL_PROPERTIES.items():
            self._plain(name, value, "Float", category="Scan")
        self._plain(OUTPUT_BUFFER_PROPERTY, 0.0, "Float", category="Scan")
        for name in SCAN_FLIP_PROPERTIES:
            self._plain(name, "Off", category="Scan")
        clock = getattr(self.twin, "clock", None)
        self._put(DynamicProperty("Twin Clock (seconds)",
                                  lambda: clock.now() if hasattr(clock, "now") else time.time(),
                                  data_type="Float"))
        self._put(DynamicProperty("Twin Holder", lambda: self._holder_state().kind))
        self.auto_references = True
        self._put(DynamicProperty(
            "Simulator Auto References", lambda: "On" if self.auto_references else "Off",
            lambda v: setattr(self, "auto_references", _to_bool(v)),
            category="Simulator", options="On, Off"))
        self._install_ebsd_properties()
        self._install_instrument_properties()
        self.virtual_masks = [np.ones((self._image_shape()), np.int8) for _ in range(5)]

    def _install_ebsd_properties(self) -> None:
        """``Simulator EBSD ...``: the twin's EBSD geometry (``DigitalTwin.configure_ebsd``), so a
        deapi client can switch the twin to an SEM-EBSD system and set its pattern centre. A
        real DE-Server has none of these: its metadata carries no EBSD geometry."""
        twin = self.twin
        if not callable(getattr(twin, "configure_ebsd", None)):
            return
        from ..ebsd import EbsdGeometry

        def current():
            return twin.ebsd or EbsdGeometry(enabled=False)

        def setter(field):
            def set_(value):
                if field == "enabled" or isinstance(getattr(EbsdGeometry, field, None), bool):
                    value = _to_bool(value)
                g = current()
                twin.configure_ebsd(g, **{field: value})
            return set_

        for field, label in EBSD_PROPERTIES.items():
            default = EbsdGeometry.describe()[field]["default"]
            if isinstance(default, bool):
                self._put(DynamicProperty(f"Simulator EBSD {label}",
                                          (lambda f=field: "On" if getattr(current(), f) else "Off"),
                                          setter(field), category="Simulator", options="On, Off"))
            else:
                self._put(DynamicProperty(f"Simulator EBSD {label}", (lambda f=field: getattr(current(), f)),
                                          setter(field), data_type="Float", category="Simulator"))

        def set_kv(value):
            twin.configure_ebsd(current(), beam_kv=float(value))

        # user master pattern: a file on the twin's machine (deapi properties are strings: no upload)
        self._ebsd_master_phase = "Nickel"
        self._ebsd_master_status = "kinematic"

        def set_master_file(value):
            path = str(value).strip()
            try:
                if path:
                    mp = twin.set_master_pattern(self._ebsd_master_phase, path)
                    self._ebsd_master_status = (f"{self._ebsd_master_phase}: {path} at {mp.kv:g} kV, "
                                                f"{mp.nbytes / 1e6:.1f} MB")
                else:
                    twin.clear_master_patterns(self._ebsd_master_phase)
                    self._ebsd_master_status = "kinematic"
            except Exception as exc:  # noqa: BLE001 - reported through the status property
                self._ebsd_master_status = f"error: {exc}"
                log.warning("EBSD master pattern %s: %s", path, exc)

        self._put(DynamicProperty("Simulator EBSD Master Pattern Phase", lambda: self._ebsd_master_phase,
                                  lambda v: setattr(self, "_ebsd_master_phase", str(v)), category="Simulator"))
        self._put(DynamicProperty("Simulator EBSD Master Pattern File", lambda: "", set_master_file,
                                  category="Simulator"))
        self._put(DynamicProperty("Simulator EBSD Master Pattern Status", lambda: self._ebsd_master_status,
                                  category="Simulator"))
        self._put(DynamicProperty("Simulator EBSD Beam Energy (kV)", lambda: twin.column.state().ht_kv,
                                  set_kv,
                                  data_type="Float", category="Simulator"))

    def _install_instrument_properties(self) -> None:
        col = self.twin.column

        def st():
            return col.state()

        def g(name):
            return lambda: col.get(name)

        from ..column.soap import DESERVER_BATCH
        from .temchannel_soap import TemChannelServer

        batcher = TemChannelServer(col, port=0)  # formatter only, never started

        def metadata():
            fields = []
            for key in DESERVER_BATCH:
                val = batcher.batch_field(key)
                if not val:
                    continue
                fields.append(f"{key}={val.replace(';', ',')}")
            return ";".join(fields)

        def project_name():
            s = st()
            return _PROJECT_NAMES.get((col.function_mode, int(s.tem_stem)), "Unknown")

        def binning(axis):
            # DE-Server reports pixel sizes per *binned* pixel: hardware x software binning
            b = 1.0
            for name in (f"Hardware Binning {axis}", f"Binning {axis}"):
                try:
                    b *= float(self[name] or 1)
                except Exception:  # noqa: BLE001 - property not present
                    pass
            return b

        def pixel_nm(axis="X"):
            try:
                px = float(self.twin.calibration.specimen_pixel_nm(st(), self.twin.detector.model))
            except Exception:  # noqa: BLE001
                return -1.0
            return px * binning(axis)

        def recip_px():
            try:
                return float(self.twin.calibration.recip_pixel_inv_nm(st(), self.twin.detector.model))
            except Exception:  # noqa: BLE001
                return -1.0

        ro = [
            ("Instrument Metadata", metadata, "String"),
            ("Instrument Project Name", project_name, "String"),
            ("Instrument Project Mode", g("ProjectionMode"), "Integer"),
            ("Instrument Project Sub Mode", g("SubMode"), "Integer"),
            ("Instrument Project Magnification", g("Magnification"), "Float"),
            ("Instrument Project Camera Length (centimeters)", g("CameraLength"), "Float"),
            ("Instrument Project TEMorSTEM Mode", g("TemStemMode"), "Integer"),
            ("Instrument Accelerating Voltage (V)", g("HighTension"), "Integer"),
            ("Instrument Screen Position", g("ScreenPosition"), "Integer"),
            ("Instrument Beam Blanking", g("BeamBlank"), "Integer"),
            ("Instrument Stage Position X (micrometers)", g("StageX"), "Float"),
            ("Instrument Stage Position Y (micrometers)", g("StageY"), "Float"),
            ("Instrument Stage Position Z (micrometers)", g("StageZ"), "Float"),
            ("Instrument Stage Tilt Alpha (degrees)", g("StageA"), "Float"),
            ("Instrument Stage Tilt Beta (degrees)", g("StageB"), "Float"),
            ("Instrument Beam Shift X (micrometers)", lambda: col.get("BeamShift")[0], "Float"),
            ("Instrument Beam Shift Y (micrometers)", lambda: col.get("BeamShift")[1], "Float"),
            ("Instrument Image Shift X (micrometers)", lambda: col.get("ImageShift")[0], "Float"),
            ("Instrument Image Shift Y (micrometers)", lambda: col.get("ImageShift")[1], "Float"),
            ("Instrument Defocus (micrometers)", g("Defocus"), "Float"),
            ("Instrument Spot Size", g("SpotSize"), "Integer"),
            ("Instrument Probe Mode", lambda: st().probe_mode.name, "String"),
            ("Instrument Convergence Semi-Angle (mrad)", g("ConvergenceAngle"), "Float"),
            ("Instrument Alpha Selector", g("AlphaSelector"), "Integer"),
            ("Instrument Condenser Aperture Index", g("CondenserApertureIndex"), "Integer"),
            ("Instrument Type", g("InstrumentType"), "String"),
            ("Specimen Pixel Size X (nanometers)", lambda: pixel_nm("X"), "Float"),
            ("Specimen Pixel Size Y (nanometers)", lambda: pixel_nm("Y"), "Float"),
            ("Diffraction Pixel Size X", recip_px, "Float"),
            ("Diffraction Pixel Size Y", recip_px, "Float"),
        ]

        # [twin] aberration corrector (read only): which side(s), and the residual of the
        # probe side (image side when only an image corrector is fitted). Ground truth,
        # from MicroscopeState, so a MirrorColumn shows the real corrector's last reading.
        def corr_ab():
            s = st()
            side = "image" if s.corrector == "image" else "probe"
            return s, side, (s.image_aberrations if side == "image" else s.probe_aberrations)

        def corr_mag(name, scale=1.0):
            def get():
                _, _, ab = corr_ab()
                return abs(complex(ab.get(name, 0j))) * scale
            return get

        def corr_pi4():
            s, _, ab = corr_ab()
            if not ab:
                return 0.0
            from ..column.corrector import pi4_angle_mrad
            from ..optics.aberrations import Aberrations
            return pi4_angle_mrad(Aberrations(ab), s.ht_kv)

        ro += [
            ("Instrument Corrector", lambda: st().corrector, "String"),
            ("Instrument Corrector Residual C1 (nm)", corr_mag("C1"), "Float"),
            ("Instrument Corrector Residual A1 (nm)", corr_mag("A1"), "Float"),
            ("Instrument Corrector Residual B2 (nm)", corr_mag("B2"), "Float"),
            ("Instrument Corrector Residual A2 (nm)", corr_mag("A2"), "Float"),
            ("Instrument Corrector Residual C3 (um)", corr_mag("C3", 1e-3), "Float"),
            ("Instrument Corrector Residual S3 (um)", corr_mag("S3", 1e-3), "Float"),
            ("Instrument Corrector Residual A3 (um)", corr_mag("A3", 1e-3), "Float"),
            ("Instrument Corrector Residual C5 (mm)", corr_mag("C5", 1e-6), "Float"),
            ("Instrument Corrector Pi/4 Angle (mrad)", corr_pi4, "Float"),
        ]
        for name, getter, dt in ro:
            self._put(DynamicProperty(name, getter, data_type=dt, category="Instrument"))
        # Counting cameras: DE-Server's "Centroiding Mode" (Standard | Super-resolution).
        # Super-resolution reads out 2x the ROI each way, and the image size says so.
        if model_counting(self.twin):
            self._centroiding = "Standard"

            def set_centroiding(value):
                v = str(value).strip()
                if v.lower() not in ("standard", "super-resolution"):
                    log.warning("centroiding mode %s not supported", value)
                    return
                self._centroiding = "Super-resolution" if "super" in v.lower() else "Standard"

            self._put(DynamicProperty("Centroiding Mode", lambda: self._centroiding, set_centroiding,
                                      data_type="String", value_type="Set",
                                      options="'Standard*', 'Super-resolution'",
                                      default_value="Standard", category="Basic"))
            for axis in ("x", "y"):
                orig = self._values.get(f"image_size_{axis}_pixels")
                if orig is None:
                    continue

                def size(orig=orig):
                    n = int(float(orig.value))
                    return 2 * n if (self._centroiding == "Super-resolution"
                                     and tuple(self._hw_bin) == (1, 1)) else n

                self._put(DynamicProperty(orig.name, size, data_type="Integer", category=orig.category))

        # Provenance: frames from here are simulated, and by what
        from .. import __version__

        self._put(DynamicProperty(SIMULATOR_SOURCE_PROPERTY, lambda: f"de-twin {__version__}",
                                  data_type="String", category="Twin"))

    def _holder_state(self):
        holder = self.twin.holder
        fn = getattr(holder, "state", None)
        if callable(fn):
            return fn()
        from ..state import HolderState

        return HolderState()

    def _get_temperature(self) -> float:
        return float(self._holder_state().temperature_c)

    def _set_temperature(self, value) -> None:
        """Written by autopilot's ServerFurnace: the twin's holder follows it.
        A holder without a temperature channel is swapped for a simulated
        heating holder on the twin's clock (the twin now has a furnace)."""
        holder = self.twin.holder
        if "temperature" not in tuple(getattr(holder, "channels", ())):
            from ..holder import SimHeatingHolder

            holder = SimHeatingHolder(reason="created by a deapi temperature write",
                                      clock=getattr(self.twin, "clock", None))
            with self.twin.lock:
                self.twin.holder = holder
        holder.set(float(value))

    def _set_specimen(self, value) -> None:
        name = str(value)
        setter = getattr(self.twin, "set_specimen", None)
        if callable(setter):
            setter(name)
        self._specimen_name = name

    # ------------------------------------------------------------ geometry
    def _image_shape(self) -> tuple[int, int]:
        return int(float(self["Image Size Y (pixels)"])), int(float(self["Image Size X (pixels)"]))

    def request(self) -> AcquisitionRequest:
        """The AcquisitionRequest the current properties describe."""
        model = self.twin.detector.model
        return request_from_properties(self.__getitem__, model.name, tuple(model.sensor_shape),
                                       scan_points=self._scan_points,
                                       acquisition_index=self._acq_count)

    def _masks_for(self, shape):
        key = (shape, tuple(id(m) for m in self.virtual_masks))
        if key not in self._mask_cache:
            out = []
            for k, m in enumerate(self.virtual_masks[:5]):
                mm = _resample(np.asarray(m), shape)
                try:
                    calc = str(self[f"Scan - Virtual Detector {k} Calculation"]).lower()
                except KeyError:
                    calc = "sum"
                pos = mm == 1
                neg = mm == 2
                out.append((pos, neg if neg.any() else None, calc))
            self._mask_cache = {key: out}
        return self._mask_cache[key]

    # ------------------------------------------------------------ status
    @property
    def acquisition_status(self):
        return "Acquiring" if (self._acq is not None and self._acq.running) else "Idle"

    @property
    def remaining_number_of_acquisitions(self):
        a = self._acq
        if a is None or not a.running:
            return 0
        if a.continuous:
            return 1  # until stopped
        return max(0, a.n_acq - a.index)

    def stop(self):
        if self._acq is not None:
            self._acq.stop.set()

    # ------------------------------------------------------------ commands
    def _ack(self, command, *values, error: bool = False):
        ack = pb.DEPacket()
        ack.type = pb.DEPacket.P_ACKNOWLEDGE
        a1 = ack.acknowledge.add()
        a1.command_id = command.command[0].command_id
        if error:
            a1.error = True
        for v in values:
            add_parameter(a1, v)
        return ack

    def _respond_to_command(self, command=None):
        if command is None:
            return False
        cid = command.command[0].command_id - commandVersion * 100
        params = command.command[0].parameter
        with self._cmd_lock:
            if cid == SET_HW_ROI:
                ox, oy, sx, sy = (p.p_int for p in params[:4])
                self["Hardware ROI Offset X"], self["Hardware ROI Offset Y"] = ox, oy
                self["Hardware ROI Size X"], self["Hardware ROI Size Y"] = sx, sy
                return (self._ack(command),)
            if cid == SET_SCAN_SIZE:
                self["Scan - Size X"], self["Scan - Size Y"] = params[0].p_int, params[1].p_int
                return (self._ack(command),)
            if cid == SET_SCAN_ROI:
                vals = [p.p_bool if p.type == pb.AnyParameter.P_BOOL else p.p_int for p in params[:5]]
                self["Scan - ROI Enable"] = "On" if vals[0] else "Off"
                (self["Scan - ROI Offset X"], self["Scan - ROI Offset Y"],
                 self["Scan - ROI Size X"], self["Scan - ROI Size Y"]) = vals[1:5]
                return (self._ack(command),)
            if cid == SET_SCAN_XY_ARRAY:
                return self._set_scan_xy_array(command)
            if cid == GET_PROPERTY_SPECIFICATIONS:
                return self._property_specifications(command)
            if cid == self.START_ACQUISITION:
                return self._start(command)
            if cid == self.GET_RESULT:
                return self._get_result(command)
            if cid == self.GET_MOVIE_BUFFER_INFO:
                return self._movie_buffer_info(command)
            if cid == self.GET_MOVIE_BUFFER:
                return self._movie_buffer(command)
            if cid == self.SET_VIRTUAL_MASK:
                out = self._set_virtual_mask(command)
                self._mask_cache = {}
                return out
            if cid == self.GET_VIRTUAL_IMAGE_INFO:
                return self._virtual_image_info(command)
            if cid == self.GET_VIRTUAL_IMAGE:
                return self._virtual_image(command)
            if cid == GET_GPU_FRAMES:
                return self._gpu_frames(command)
            if cid == self.LIST_CAMERAS:
                return (self._ack(command, str(self.twin.detector.model.name)),)
            return super()._respond_to_command(command)

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.socket.recv(n - len(buf))
            if not chunk:
                raise ConnectionResetError("connection closed while receiving data")
            buf += chunk
        return buf

    def _set_virtual_mask(self, command):
        mask_id = command.command[0].parameter[0].p_int
        w = command.command[0].parameter[1].p_int
        h = command.command[0].parameter[2].p_int
        data = self._recv_exact(w * h)
        mask = np.frombuffer(data, dtype=np.int8).reshape((h, w)).copy()
        if 0 <= mask_id < len(self.virtual_masks):
            self.virtual_masks[mask_id] = mask
        return (self._ack(command),)

    def _set_scan_xy_array(self, command):
        vals = [p.p_int for p in command.command[0].parameter]
        width, height, counts = vals[0], vals[1], vals[2:]
        sock_reply = self._ack(command)
        # the client sends the positions right after reading our acknowledge
        packet = struct.pack("I", sock_reply.ByteSize()) + sock_reply.SerializeToString()
        self.socket.sendall(packet)
        points = []
        for n in counts:
            xs = np.frombuffer(self._recv_exact(4 * n), np.int32)
            ys = np.frombuffer(self._recv_exact(4 * n), np.int32)
            points.extend(zip(xs.tolist(), ys.tolist()))
        self._scan_points = points
        self["Scan - Size X"], self["Scan - Size Y"] = int(width), int(height)
        self._values["scan_-_type"]._value = "XY File"
        return ()

    def _property_specifications(self, command):
        name = command.command[0].parameter[0].p_string
        prop = self._values.get(_norm(name))
        if prop is None:
            return (self._ack(command, f"Property not in Server: '{name}'", error=True),)
        vt = str(prop.value_type)
        opts: list = []
        raw = prop.options if isinstance(prop.options, str) else ",".join(map(str, prop.options or []))
        if vt == "Range" and raw:
            parts = raw.split(",")[:2]
            try:
                opts = [float(parts[0]), float(parts[1])]
            except (ValueError, IndexError):
                opts = []
        elif vt == "Set" and raw:
            opts = [o.replace("*", "").replace("'", "").replace("`", "").strip() for o in raw.split(",")]
        read_only = vt in ("ReadOnly", "READ_ONLY", "Read Only", "Server")
        values = [str(prop.data_type), vt, *opts, str(prop.category), str(prop.default_value or ""),
                  str(prop.value), bool(read_only)]
        return (self._ack(command, *values),)

    def _start(self, command):
        params = command.command[0].parameter
        n_acq = params[0].p_int if len(params) > 0 else 1
        movie = bool(params[1].p_bool) if len(params) > 1 else False
        if self._acq is not None and self._acq.running:
            self._acq.stop.set()
            self._acq.thread.join(timeout=5)
        req = self.request()
        if self._shared_frames is not None:
            self._shared_frames.new_acquisition()
        self._acq_count += 1
        self.number_of_frames_requested = req.total_frames * max(1, n_acq)
        self._acq = _Acquisition(self, req, n_acq, movie).start()
        return (self._ack(command),)

    def _gpu_frames(self, command):
        """GET_GPU_FRAMES (client pid, attach): attach answers the client's handle to the
        control block, as DE-Server does. Without GPU memory sharing on this computer it
        answers "simulated", as deapi's fake server does."""
        from . import gpu_frames

        params = command.command[0].parameter
        pid = int(params[0].p_string)
        attach = bool(params[1].p_bool) if len(params) > 1 else True
        if not attach:
            if self._shared_frames is not None:
                self._shared_frames.detach()
            return (self._ack(command),)
        if self._shared_frames is None:
            if not gpu_frames.sharing_available():
                log.warning("this GPU cannot share memory with another process: GPU frames are simulated")
                return (self._ack(command, "simulated"),)
            self._shared_frames = gpu_frames.SharedFrames()
        h, w = self._frame_shape()
        try:
            handle = self._shared_frames.attach(pid, (GPU_BUFFER_FRAMES, h, w))
        except OSError as exc:
            log.warning("GPU frames: %s", exc)
            return (self._ack(command, error=True),)
        return (self._ack(command, str(handle)),)

    def _frame_shape(self) -> tuple[int, int]:
        """The shape of the frames the next acquisition will render."""
        detector = self.twin.detector
        request = self.request()
        if hasattr(detector, "output_shape"):
            return tuple(int(v) for v in detector.output_shape(request))
        return self._image_shape()

    # ------------------------------------------------------------ results
    def _result_image(self, frame_type: int, timeout_s: float):
        """(image array, frame count, eppix) for a GET_RESULT frame type."""
        acq = self._acq
        if 12 <= frame_type <= 16:  # the masks themselves
            return np.asarray(self.virtual_masks[frame_type - 12]), 1, 0.0
        if acq is None:
            shape = self._image_shape()
            return np.zeros(shape, np.uint16), 0, 0.0
        acq.wait_first(timeout_s)
        with acq.cond:
            meta = acq.last_meta
            eppix = float(getattr(meta, "electrons_per_pixel", 0.0) or 0.0)
            if 2 <= frame_type <= 8:
                img = acq.last if acq.last is not None else np.zeros(self._image_shape(), np.uint16)
                return img, acq.count, eppix
            if frame_type in (9, 10, 11):
                img = acq.sum if acq.sum is not None else np.zeros(self._image_shape())
                return img, acq.count, eppix * acq.count
            if 17 <= frame_type <= 21:
                if acq.virtual is None:
                    return np.zeros((1, 1)), acq.count, eppix
                return acq.virtual[frame_type - 17].copy(), acq.count, eppix
        if 22 <= frame_type <= 25:
            vi = getattr(self.twin, "virtual_image", None)
            if callable(vi) and acq.request.scan.enabled:
                try:
                    return np.asarray(vi(acq.request, 50.0, 200.0), np.float32), acq.count, eppix
                except Exception as exc:  # noqa: BLE001
                    log.info("twin.virtual_image failed (%s); using mask 0", exc)
            if acq.virtual is not None:
                return acq.virtual[0].copy(), acq.count, eppix
            return np.zeros((1, 1)), acq.count, eppix
        raise ValueError(f"Frame type {frame_type} not supported by the twin fake server")

    def _get_result(self, command):
        p = command.command[0].parameter
        frame_type = p[0].p_int
        pixel_format = p[1].p_int
        window_w, window_h = p[5].p_int, p[6].p_int
        timeout_ms = p[13].p_int if len(p) > 13 else 5000
        histo_min = p[14].p_float if len(p) > 14 else 0.0
        histo_max = p[15].p_float if len(p) > 15 else 0.0
        histo_bins = p[16].p_int if len(p) > 16 else 256
        img, count, eppix = self._result_image(frame_type, max(timeout_ms, 1000) / 1000.0)
        if pixel_format not in PIXEL_DTYPES:  # AUTO
            if img.dtype in (np.uint8, np.int8) and frame_type in (12, 13, 14, 15, 16):
                pixel_format = 1
            elif img.dtype == np.uint16 and frame_type <= 8:
                pixel_format = 5
            else:
                pixel_format = 13
        if window_w <= 0:
            window_w = img.shape[1]
        if window_h <= 0:
            window_h = img.shape[0]
        img = _resample(np.asarray(img), (window_h, window_w))
        out = _convert(img, pixel_format)
        data = out.tobytes()
        acq = self._acq
        fps = float(self._fps)
        mean = float(out.mean()) if out.size else 0.0
        hmin, hmax = (float(out.min()), float(out.max())) if out.size else (0.0, 0.0)
        mapping = [
            int(pixel_format), int(window_w), int(window_h), "de_twin",
            int(acq.request.acquisition_index if acq else 0),
            bool(acq is not None and acq.running), int(max(count - 1, 0)), int(count),
            hmin, hmax, mean, float(out.std()) if out.size else 0.0,
            float(eppix), float(eppix * out.size * fps), float(eppix * fps), 0.0, float(eppix),
            0.0, 0.0, 0.0, 0.0, 0.0,  # incident
            0.0, 0.0, 0.0,  # red/orange warning, saturation
            f"{time.time():.6f}", hmin, hmax, 1.0,
        ]
        if _sends_scan_pattern_index(_SERVER_VERSION):
            mapping.append(0)  # current_scan_pattern_idx
        if histo_min == 0 and histo_max == 0:
            histo_min, histo_max = hmin, hmax
        if histo_max <= histo_min:
            histo_max = histo_min + 1
        hist, _ = np.histogram(out.ravel(), bins=max(1, histo_bins), range=(histo_min, histo_max))
        mapping += [float(histo_min), float(histo_max), float(histo_max)]
        mapping += [int(v) for v in hist]
        ack = self._ack(command, *mapping)
        header = pb.DEPacket()
        header.type = pb.DEPacket.P_DATA_HEADER
        header.data_header.bytesize = len(data)
        return (ack, header, data)

    def _movie_buffer_info(self, command):
        h, w = self._image_shape()
        fpb = int(float(self["Grabbing - Frames Per Buffer"]))
        # client order: headerBytes, imageBufferBytes, frameIndexStartPos, imageStartPos, W, H, frames, type
        return (self._ack(command, 1024, h * w * fpb * 2, 0, 1024, w, h, fpb, 5),)

    def _movie_buffer(self, command):
        p = command.command[0].parameter
        timeout_s = (p[0].p_int if len(p) else 5000) / 1000.0
        fpb = max(1, int(float(self["Grabbing - Frames Per Buffer"])))
        acq = self._acq
        if acq is None:
            return (self._ack(command, 4, 0, 0),)
        with acq.cond:
            acq.cond.wait_for(lambda: len(acq.movie_frames) >= fpb or acq.done, timeout_s)
            n = min(fpb, len(acq.movie_frames))
            if n == 0:
                return (self._ack(command, 4 if acq.done else 3, 0, 0),)
            frames = [acq.movie_frames.popleft() for _ in range(n)]
            first = acq.movie_next
            acq.movie_next += n
        data = np.stack(frames).astype(np.uint16).tobytes()
        numbers = np.zeros(128, np.int64)
        numbers[: min(n, 128)] = np.arange(first, first + min(n, 128))
        return (self._ack(command, 5, len(data) + 1024, n), numbers.tobytes(), data)

    def _virtual_image_info(self, command):
        nx, ny = int(float(self["Scan - Size X"])), int(float(self["Scan - Size Y"]))
        return (self._ack(command, nx * ny * 2, nx, ny, 5),)

    def _virtual_image(self, command):
        vid = command.command[0].parameter[0].p_int
        acq = self._acq
        nx, ny = int(float(self["Scan - Size X"])), int(float(self["Scan - Size Y"]))
        img = np.zeros((ny, nx))
        if acq is not None and acq.virtual is not None and 0 <= vid < 5:
            with acq.cond:
                img = acq.virtual[vid].copy()
        peak = float(img.max()) if img.size else 0.0
        if peak > 65535:
            img = img * (65535.0 / peak)
        raw = _convert(img, 5).tobytes()
        return (self._ack(command, 5, len(raw), int(acq.count if acq else 0)), raw)


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

def _recv_exact(conn, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionResetError(f"connection closed after {len(buf)} of {n} bytes")
        buf += chunk
    return buf


def make_factory(twin, *, pace: bool = True):
    """A ``FakeServer``-signature factory returning ONE shared TwinFakeServer.

    For patching deapi's loop: ``initialize_server.FakeServer = make_factory(twin)``
    (Ground Crew patches ``fake_server.inp_file`` the same way).
    """
    shared: dict = {}

    def factory(dataset="grains", socket=None):
        srv = shared.get("srv")
        if srv is None:
            srv = shared["srv"] = TwinFakeServer(socket=socket, twin=twin, pace=pace)
        srv.socket = socket
        return srv

    return factory


class TwinDeapiServer:
    """The deapi protocol on a TCP port (plus the UDP stop listener), for one twin.

    Unlike deapi's ``initialize_server.main`` it serves several connections at
    once (one thread each, commands serialised), binds any host and can be
    stopped, which is what tests and ``de-twin serve`` need.
    """

    def __init__(self, twin, port: int = 13240, host: str = "127.0.0.1", *, pace: bool = True):
        self.twin = twin
        self.host = host
        self._requested_port = int(port)
        self._port = int(port)
        self.fake = TwinFakeServer(twin=twin, pace=pace)
        self._stop = threading.Event()
        self._tcp: Optional[_socketmod.socket] = None
        self._threads: list[threading.Thread] = []

    @property
    def port(self) -> int:
        """The bound TCP port (cached at `start`: the socket may be closed by now)."""
        return self._port

    def start(self) -> "TwinDeapiServer":
        tcp = _socketmod.socket(_socketmod.AF_INET, _socketmod.SOCK_STREAM)
        tcp.setsockopt(_socketmod.SOL_SOCKET, _socketmod.SO_REUSEADDR, 1)
        tcp.bind((self.host, self._requested_port))
        tcp.listen()
        tcp.settimeout(0.25)
        self._port = int(tcp.getsockname()[1])
        self._tcp = tcp
        for target in (self._accept_loop, self._udp_loop):
            t = threading.Thread(target=target, daemon=True, name=f"TwinDeapi-{target.__name__}")
            t.start()
            self._threads.append(t)
        return self

    def serve_forever(self) -> None:
        if self._tcp is None:
            self.start()
        try:
            while not self._stop.wait(0.5):
                pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        self.fake.stop()
        shared = getattr(self.fake, "_shared_frames", None)
        if shared is not None:
            if self.fake._acq is not None:
                self.fake._acq.thread.join(timeout=5)
            shared.close()
            self.fake._shared_frames = None
        if self._tcp is not None:
            try:
                self._tcp.close()
            except OSError:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _udp_loop(self) -> None:
        with _socketmod.socket(_socketmod.AF_INET, _socketmod.SOCK_DGRAM) as udp:
            try:
                udp.bind((self.host, self.port))
            except OSError as e:
                log.warning("stop_acquisition will not be answered (UDP %d: %s)", self.port, e)
                return
            udp.settimeout(0.25)
            while not self._stop.is_set():
                try:
                    message, addr = udp.recvfrom(64)
                except (_socketmod.timeout, OSError):
                    continue
                if message.startswith(b"PyClientStopAcq"):
                    self.fake.stop()
                udp.sendto(b"Stopped", addr)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._tcp.accept()
            except (_socketmod.timeout, OSError):
                continue
            conn.settimeout(None)
            t = threading.Thread(target=self._serve_conn, args=(conn,), daemon=True,
                                 name="TwinDeapi-conn")
            t.start()

    def _serve_conn(self, conn) -> None:
        fake = self.fake
        fake.socket = conn
        with conn:
            while not self._stop.is_set():
                try:
                    size = struct.unpack("I", _recv_exact(conn, 4))[0]
                    packet = pb.DEPacket()
                    packet.ParseFromString(_recv_exact(conn, size))
                    responses = fake._respond_to_command(packet)
                    for r in responses:
                        if isinstance(r, pb.DEPacket):
                            conn.sendall(struct.pack("I", r.ByteSize()) + r.SerializeToString())
                        else:
                            conn.sendall(r)
                except (ConnectionResetError, ConnectionAbortedError, OSError):
                    return
                except Exception:  # noqa: BLE001
                    traceback.print_exc(file=sys.stderr)
                    return


def serve(twin, port: int = 13240, host: str = "127.0.0.1", *, pace: bool = True) -> None:
    """Serve the deapi protocol for an existing twin. Blocks (run it in a thread)."""
    TwinDeapiServer(twin, port=port, host=host, pace=pace).serve_forever()


def main(argv=None) -> int:
    """deapi-compatible launcher: ``python -m de_twin.faces.deapi_server [port]``."""
    p = argparse.ArgumentParser(prog="de_twin.faces.deapi_server", description=main.__doc__)
    p.add_argument("port_pos", nargs="?", type=int, help="port (deapi's positional form)")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--specimen", default="Dense Au on holey C")
    p.add_argument("--camera", default="DE16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holder", default="none", help="none | sim-heating | sim-biasing | impulse")
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--soap-port", type=int, default=None,
                   help="also serve the DE-TEM-Channel SOAP face for the same twin")
    p.add_argument("--mirror-temchannel", metavar="HOST[:PORT]", default=None,
                   help="follow a real/Dummy DE-TEM-Channel instead of simulating the column")
    p.add_argument("--no-pace", action="store_true", help="render frames as fast as possible")
    p.add_argument("--deapi-loop", action="store_true",
                   help="use deapi's own initialize_server loop (single connection) patched to the twin")
    args = p.parse_args(argv)
    port = args.port or args.port_pos or 13240

    from ..clock import Clock
    from ..twin import DigitalTwin

    clock = Clock(time_scale=args.time_scale)
    column = None
    if args.mirror_temchannel:
        from ..column import MirrorColumn

        h, _, pt = args.mirror_temchannel.partition(":")
        column = MirrorColumn(host=h, port=int(pt or 5002), clock=clock)
    twin = DigitalTwin(specimen=args.specimen, camera=args.camera, column=column, holder=args.holder,
                       seed=args.seed, clock=clock)
    if args.soap_port:
        from .temchannel_soap import TemChannelServer

        TemChannelServer(twin.column, host="0.0.0.0", port=args.soap_port).start()
        sys.stderr.write(f"DE-TEM-Channel SOAP face on port {args.soap_port}\n")
    if args.deapi_loop:
        from deapi.simulated_server import initialize_server

        initialize_server.FakeServer = make_factory(twin, pace=not args.no_pace)
        initialize_server.main(port)  # prints deapi's own "started .... " banner
        return 0
    server = TwinDeapiServer(twin, port=port, host=args.host, pace=not args.no_pace).start()
    sys.stdout.write("started .... \n\n")
    sys.stdout.flush()
    sys.stderr.write(f"Waiting for a Connection to: \n    Host: {args.host}\n    Port: {server.port} \n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
