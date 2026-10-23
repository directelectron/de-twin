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
import logging
import socket as _socketmod
import struct
import sys
import threading
import time
import traceback
from typing import Any, Callable, Optional

import numpy as np

from ..state import AcquisitionRequest
from .deapi_acquisition import (_Acquisition, _convert, _resample)
from .deapi_request import (OUTPUT_BUFFER_PROPERTY, SCAN_COIL_PROPERTIES, SCAN_FLIP_PROPERTIES, SIMULATOR_SOURCE_PROPERTY, model_counting, request_from_properties)
from .deapi_protocol import (DEFAULT_EXPOSURE_S, GET_GPU_FRAMES, GET_PROPERTY_SPECIFICATIONS, GPU_BUFFER_FRAMES, PIXEL_DTYPES, SET_HW_ROI, SET_SCAN_ROI, SET_SCAN_SIZE, SET_SCAN_XY_ARRAY, TEMPERATURE_PROP, _PROJECT_NAMES, _property_key, _to_bool)

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


#: Read-only property naming the simulator behind the frames (``"de-twin <version>"``), so
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
        self._values[_property_key(prop.name)] = prop

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
        self._install_instrument_properties()
        self.virtual_masks = [np.ones((self._image_shape()), np.int8) for _ in range(5)]

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
        prop = self._values.get(_property_key(name))
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
