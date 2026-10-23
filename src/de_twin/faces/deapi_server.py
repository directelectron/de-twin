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
import traceback

import numpy as np

from ..state import AcquisitionRequest
from .deapi_acquisition import _Acquisition
from .deapi_compat import FakeServer, Property, add_parameter, commandVersion, pb
from .deapi_properties import PropertiesMixin
from .deapi_protocol import (
    GET_GPU_FRAMES,
    GET_PROPERTY_SPECIFICATIONS,
    SET_HW_ROI,
    SET_SCAN_ROI,
    SET_SCAN_SIZE,
    SET_SCAN_XY_ARRAY,
    _property_key,
    _recv_exact,
)
from .deapi_request import (
    request_from_properties,
)
from .deapi_results import ResultsMixin

log = logging.getLogger(__name__)


#: Read-only property naming the simulator behind the frames (``"de-twin <version>"``), so
class TwinFakeServer(PropertiesMixin, ResultsMixin, FakeServer):  # type: ignore[misc,valid-type]
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

    def __init__(
        self,
        dataset: str = "grains",
        socket=None,
        *,
        twin=None,
        pace: bool | None = None,
        default_fps: float = 40.0,
    ):
        if pb is None:
            raise ImportError("TwinFakeServer needs the deapi package")
        self._tls = threading.local()
        self.twin = twin if twin is not None else type(self).twin
        if self.twin is None:
            raise ValueError("TwinFakeServer needs a twin (pass twin= or set TwinFakeServer.twin)")
        if pace is not None:
            self.pace = pace
        self._acq: _Acquisition | None = None
        self._acq_count = 0
        self._shared_frames = None  # de_twin.faces.gpu_frames.SharedFrames, made on first attach
        self._scan_points: list | None = None
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
        self._put(
            Property(
                name=name,
                value=str(value),
                data_type=data_type,
                category=category,
                value_type=value_type,
                options=options,
                default_value=str(value),
                server=self,
            )
        )

    def request(self) -> AcquisitionRequest:
        """The AcquisitionRequest the current properties describe."""
        model = self.twin.detector.model
        return request_from_properties(
            self.__getitem__,
            model.name,
            tuple(model.sensor_shape),
            scan_points=self._scan_points,
            acquisition_index=self._acq_count,
        )

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
                (
                    self["Scan - ROI Offset X"],
                    self["Scan - ROI Offset Y"],
                    self["Scan - ROI Size X"],
                    self["Scan - ROI Size Y"],
                ) = vals[1:5]
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

    def _set_scan_xy_array(self, command):
        vals = [p.p_int for p in command.command[0].parameter]
        width, height, counts = vals[0], vals[1], vals[2:]
        sock_reply = self._ack(command)
        # the client sends the positions right after reading our acknowledge
        packet = struct.pack("I", sock_reply.ByteSize()) + sock_reply.SerializeToString()
        self.socket.sendall(packet)
        points = []
        for n in counts:
            xs = np.frombuffer(_recv_exact(self.socket, 4 * n), np.int32)
            ys = np.frombuffer(_recv_exact(self.socket, 4 * n), np.int32)
            points.extend(zip(xs.tolist(), ys.tolist()))
        self._scan_points = points
        self["Scan - Size X"], self["Scan - Size Y"] = int(width), int(height)
        self._values["scan_-_type"]._value = "XY File"
        return ()

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


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


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
        self._tcp: _socketmod.socket | None = None
        self._threads: list[threading.Thread] = []

    @property
    def port(self) -> int:
        """The bound TCP port (cached at `start`: the socket may be closed by now)."""
        return self._port

    def start(self) -> TwinDeapiServer:
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
                except (TimeoutError, OSError):
                    continue
                if message.startswith(b"PyClientStopAcq"):
                    self.fake.stop()
                udp.sendto(b"Stopped", addr)

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._tcp.accept()
            except (TimeoutError, OSError):
                continue
            conn.settimeout(None)
            t = threading.Thread(target=self._serve_conn, args=(conn,), daemon=True, name="TwinDeapi-conn")
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
                except Exception:
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
    p.add_argument(
        "--soap-port",
        type=int,
        default=None,
        help="also serve the DE-TEM-Channel SOAP face for the same twin",
    )
    p.add_argument(
        "--mirror-temchannel",
        metavar="HOST[:PORT]",
        default=None,
        help="follow a real/Dummy DE-TEM-Channel instead of simulating the column",
    )
    p.add_argument("--no-pace", action="store_true", help="render frames as fast as possible")
    p.add_argument(
        "--deapi-loop",
        action="store_true",
        help="use deapi's own initialize_server loop (single connection) patched to the twin",
    )
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
    twin = DigitalTwin(
        specimen=args.specimen,
        camera=args.camera,
        column=column,
        holder=args.holder,
        seed=args.seed,
        clock=clock,
    )
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
