"""TEM-Channel SOAP face: impersonate DE-TEM-Channel on port 5002.

:class:`TemChannelServer` answers the DE-TEM-Channel SOAP API
(``service/EMinstrumentWebservice.cpp`` + ``.wsdl``) from a
:class:`~de_twin.column.Column`, so that

* DE-Server's gSOAP client (``Instrument.cpp``, RPC/encoded, polling
  ``executeBatchCommands`` every cycle),
* ``de_microscope.Microscope`` (regex tag extraction), autopilot, Ground Crew,
* :class:`~de_twin.column.SoapTemChannelClient` / :class:`MirrorColumn`

all drive and read the twin's column unchanged.

Wire formats are the channel's own:

* responses are named after their gSOAP *type* (``getfloatResponse``,
  ``gettwofloatResponse``, ``getfivefloatResponse``, ``getshortResponse``,
  ``getintResponse``, ``setResult``, ``getPropertyResponse``, ``<op>Response``
  with ``<r>``), matching ``docs/soap-samples``;
* ``executeBatchCommands`` fields use C++ ``ostream`` default formatting
  (``%g``, 6 significant digits), ``STAGE_PositionFull`` is ``x;y;z;a;b`` (um,
  deg), pairs are ``x;y``, ``ILLUM_ConvergenceAngle`` is ``%.3f``, unknown keys
  (and ``STAGE_Position``, which the channel never implemented) answer empty,
  ``PROJ_ImageShift``/``PROJ_ImageBeamShift``/``STAGE_Status`` answer ``0``;
* units: ``GUN_HT`` volts, ``PROJ_CameraLength`` cm, ``PROJ_Defocus`` um,
  shifts um, ``PROJ_SubMode`` the raw JEOL EOS function mode (0 MAG1,
  1 MAG2, 2 LowMAG, 3 SAMAG, 4 DIFF) or the TFS submode when the column
  reports ``FEI``;
* ``getProperty`` answers ``%d`` / ``%.6g`` strings and a per-property
  operation status (0 idle, 1 running, 2 done, 3 failed, 4 aborted);
* setters answer ``setResult/success`` 1 or 0; BeamTilt/ObjStig/CondenserStig
  outside [-1, 1] are refused as the service does.
* the aberration corrector (F1/F2: ``getCorrector*``, ``getAberration(s)``,
  ``corrector*`` commands, ``get/setCorrectorBeamTilt``) is answered from
  ``column.corrector`` (:mod:`de_twin.column.corrector`) with CorrectorService's
  semantics: commands are *posted* (``success`` = accepted), ``getCorrectorStatus``
  is 0 idle / 1 running / 2 done / 3 failed / 6 absent, results are read back with
  ``getCorrectorResult`` / ``getAberrations`` (``"C1=x,y;A1=x,y;..."``, metres,
  ``%.9g``) / ``getCorrectorError``; ``waitMs`` (capped at 5 s) only for corrections
  and the corrector beam tilt (WD, radians). Only one corrector is served (the probe
  one unless ``corrector_side="image"``), as the channel keeps one endpoint;
  ``getCorrectorEndpoints`` lists both. No corrector: present 0, status 6, commands
  refused (``correctorReconnect`` accepted), as on a column without one.

DE-Server pings the channel host (ICMP) before connecting; loopback answers.
"""

from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

from ..column.column import Column
from ..column.errors import ColumnRefused
from ..column.soap import PROP_OP, SoapError, envelope, fault_envelope, parse_body

log = logging.getLogger(__name__)

#: Where the real WSDL lives on a DE workstation (served for ``GET /?wsdl`` when present).
WSDL_CANDIDATES = (
    Path(r"C:\direct_electron\DE-TEM-Channel\service\EMinstrumentWebservice.wsdl"),
)

OP_IDLE, OP_RUNNING, OP_DONE, OP_FAILED, OP_ABORTED = 0, 1, 2, 3, 4
_STAGE_OPS = {PROP_OP["STAGE_X"], PROP_OP["STAGE_Y"], PROP_OP["STAGE_Z"],
              PROP_OP["STAGE_A"], PROP_OP["STAGE_B"], PROP_OP["STAGEPOSITION"]}


def _g(v: float) -> str:
    """C++ ``ostream << float`` default formatting (``%g``, precision 6)."""
    return format(float(v), ".6g")


def _as_float(fields: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(fields.get(key, default) or default)
    except ValueError:
        return default


def _as_int(fields: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(fields.get(key, default) or default))
    except ValueError:
        return default


class TemChannelServer:
    """A threaded HTTP server speaking the DE-TEM-Channel SOAP API over a Column.

    >>> srv = TemChannelServer(column, host="127.0.0.1", port=5002).start()
    >>> ...
    >>> srv.stop()

    ``port=0`` picks a free port (read it back from ``srv.port``).
    ``handle(xml)`` answers one request without any socket (used to replay the
    recorded samples).
    """

    def __init__(self, column: Column, host: str = "0.0.0.0", port: int = 5002, *,
                 enforce_service_ranges: bool = True, log_requests: bool = False,
                 corrector_side: Optional[str] = None):
        self.column = column
        self.corrector_side = corrector_side
        self.host = host
        self._requested_port = int(port)
        self.enforce_service_ranges = enforce_service_ranges
        self.log_requests = log_requests
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._ops_lock = threading.Lock()
        self._ops: dict[int, int] = {}
        self._global_op = OP_IDLE
        self._stage_mask = 0
        self.requests = 0
        self._handlers: dict[str, Callable[[dict], tuple[str, list]]] = self._build_handlers()

    # ------------------------------------------------------------ lifecycle
    @property
    def port(self) -> int:
        if self._httpd is not None:
            return int(self._httpd.server_address[1])
        return self._requested_port

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}/"

    def start(self) -> "TemChannelServer":
        if self._httpd is not None:
            return self
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # noqa: N802
                if server.log_requests:
                    log.info("%s - %s", self.address_string(), fmt % args)

            def _read_body(self) -> bytes:
                if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                    chunks = []
                    while True:
                        line = self.rfile.readline().strip()
                        size = int(line.split(b";")[0] or b"0", 16)
                        if size == 0:
                            self.rfile.readline()
                            break
                        chunks.append(self.rfile.read(size))
                        self.rfile.readline()
                    return b"".join(chunks)
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length > 0 else b""

            def _send(self, code: int, body: str, ctype: str = "text/xml; charset=utf-8"):
                data = body.encode("utf-8")
                self.send_response(code)
                self.send_header("Server", "gSOAP/2.8")
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True

            def do_POST(self):  # noqa: N802
                body = self._read_body()
                code, text = server.handle(body)
                self._send(code, text)

            def do_GET(self):  # noqa: N802
                if "wsdl" in self.path.lower():
                    for p in WSDL_CANDIDATES:
                        if p.exists():
                            return self._send(200, p.read_text(encoding="utf-8", errors="replace"))
                self._send(200 if self.path in ("/", "") else 404,
                           "de_twin DE-TEM-Channel SOAP face (POST SOAP requests here)",
                           "text/plain; charset=utf-8")

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self._httpd = Server((self.host, self._requested_port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="TemChannelServer",
                                        daemon=True)
        self._thread.start()
        log.info("DE-TEM-Channel SOAP face on %s:%d", self.host, self.port)
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> "TemChannelServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------ dispatch
    def handle(self, xml: bytes | str) -> tuple[int, str]:
        """Answer one SOAP request. Returns (HTTP status, response XML)."""
        self.requests += 1
        try:
            op, fields = parse_body(xml)
        except SoapError as exc:
            return 500, fault_envelope(str(exc))
        handler = self._handlers.get(op)
        if handler is None:
            return 500, fault_envelope(f"Method '{op}' not implemented: method name or namespace "
                                       f"not recognized", "SOAP-ENV:Client")
        try:
            element, out = handler(fields)
        except Exception as exc:  # noqa: BLE001
            log.exception("SOAP %s failed", op)
            return 500, fault_envelope(f"{op}: {exc}", "SOAP-ENV:Server")
        return 200, envelope(element, out)

    def operations(self) -> list[str]:
        return sorted(self._handlers)

    # ------------------------------------------------------------ op state
    def _begin(self, op_id: int, ok: bool, *, stage_mask: int = 0) -> None:
        with self._ops_lock:
            state = OP_DONE if ok else OP_FAILED
            if ok and op_id in _STAGE_OPS:
                self._stage_mask |= stage_mask or 0x1F
            self._ops[op_id] = state
            if op_id == PROP_OP["STAGEPOSITION"] and ok:
                for i, oid in enumerate(("STAGE_X", "STAGE_Y", "STAGE_Z", "STAGE_A", "STAGE_B")):
                    if stage_mask & (1 << i):
                        self._ops[PROP_OP[oid]] = OP_DONE
            self._global_op = state
            self._last_op = op_id

    def op_status(self, op_id: int) -> int:
        """GetPropertyOperationStatus: stage ops are Running while the column moves."""
        with self._ops_lock:
            moving = self.column.op_state() == OP_RUNNING
            if op_id == PROP_OP["GLOBAL"]:
                if moving and getattr(self, "_last_op", None) in _STAGE_OPS:
                    return OP_RUNNING
                return self._global_op
            if op_id in _STAGE_OPS:
                axis = {111: 0, 112: 1, 113: 2, 114: 3, 115: 4}.get(op_id)
                commanded = self._stage_mask if axis is None else self._stage_mask & (1 << axis)
                if moving and commanded:
                    return OP_RUNNING
            return self._ops.get(op_id, OP_IDLE)

    def _set(self, op_key: Optional[str], fn: Callable[[], Any], *, stage_mask: int = 0) -> tuple[str, list]:
        try:
            fn()
            ok = True
        except (ColumnRefused, ValueError, KeyError, TypeError) as exc:
            log.info("SOAP set refused: %s", exc)
            ok = False
        if op_key is not None:
            self._begin(PROP_OP[op_key], ok, stage_mask=stage_mask)
        return "setResult", [("success", 1 if ok else 0)]

    # ------------------------------------------------------------ handlers
    def _build_handlers(self) -> dict[str, Callable[[dict], tuple[str, list]]]:
        c = self.column
        g = c.get

        def fl(key):
            return lambda f: ("getfloatResponse", [("floatResult", float(key()))])

        def sh(key):
            return lambda f: ("getshortResponse", [("shortResult", int(key()))])

        def it(key):
            return lambda f: ("getintResponse", [("intResult", int(key()))])

        def pair(name):
            def h(f):
                x, y = g(name)
                return "gettwofloatResponse", [("floatone", float(x)), ("floattwo", float(y))]
            return h

        def string(element, key):
            return lambda f: (element, [("r", str(key()))])

        def set_value(prop, op_key, arg, conv=float):
            return lambda f: self._set(op_key, lambda: c.set(prop, conv(f.get(arg, "0") or 0)))

        def set_int(prop, op_key, arg):
            return set_value(prop, op_key, arg, lambda v: int(float(v)))

        def set_pair(prop, op_key, check_range=False):
            def h(f):
                x, y = _as_float(f, "x"), _as_float(f, "y")
                if check_range and self.enforce_service_ranges and (abs(x) > 1.0 or abs(y) > 1.0):
                    self._begin(PROP_OP[op_key], False)
                    return "setResult", [("success", 0)]
                return self._set(op_key, lambda: c.set(prop, (x, y)))
            return h

        def set_stage(f):
            mask = _as_int(f, "axes-mask")
            axes = {}
            for i, k in enumerate("xyzab"):
                if mask & (1 << i):
                    axes[k] = _as_float(f, k)
            if not axes:
                # mask 0: nothing to move, like the Dummy (accepted, no-op)
                return self._set("STAGEPOSITION", lambda: None, stage_mask=0)
            return self._set("STAGEPOSITION", lambda: c.set("StagePosition", axes), stage_mask=mask)

        def stop_op(f):
            c.stop()
            with self._ops_lock:
                self._global_op = OP_ABORTED
                for k in list(self._ops):
                    if self._ops[k] == OP_RUNNING:
                        self._ops[k] = OP_ABORTED
                self._stage_mask = 0
            return "setResult", [("success", 1)]

        def stage(f):
            x, y, z, a, b = c.stage_position()
            return "getfivefloatResponse", [("one", x), ("two", y), ("three", z), ("four", a),
                                            ("five", b)]

        def op_status(f):
            return "getintResponse", [("intResult", self.op_status(_as_int(f, "val")))]

        def aperture_size(f):
            kind, size = _as_int(f, "kind"), _as_int(f, "size")
            return self._set("APERTURESIZE", lambda: c.set("ApertureSize", (kind, size)))

        def set_mag_index(f):
            return self._set("MAGINDEX", lambda: c.set("MagnificationIndex", _as_int(f, "idx")))

        h: dict[str, Callable[[dict], tuple[str, list]]] = {
            # --- gun / vacuum
            "getAccelVoltage": fl(lambda: g("HighTension")),
            "getMaxHTValue": fl(lambda: g("MaxHighTension")),
            "getHTState": sh(lambda: g("HTState")),
            "getFilamentValue": fl(lambda: g("Filament")),
            "getBeamBlankState": it(lambda: g("BeamBlank")),
            "getVacuumStatus": fl(lambda: g("VacuumStatus")),
            "getVacuumGauges": fl(lambda: g("VacuumGauges")),
            "getColumnValveStatus": sh(lambda: g("ColumnValvesOpen")),
            "getPVPStatus": sh(lambda: 0),
            # --- projection
            "getProjectMode": sh(lambda: g("ProjectionMode")),
            "getProjectSubMode": sh(lambda: g("SubMode")),
            "getMagValue": sh(lambda: g("MagnificationIndex")),
            "getMagMode": sh(lambda: c.function_mode),
            "getMagStr": string("getMagStrResponse", lambda: g("MagString")),
            "getFocusValue": fl(lambda: g("Focus")),
            "getDeFocusValue": fl(lambda: g("Defocus")),
            "getScreenPosition": sh(lambda: g("ScreenPosition")),
            "getExposureTime": fl(lambda: g("ExposureTime")),
            "getImageShift": pair("ImageShift"),
            "getDiffractionShift": pair("DiffractionShift"),
            "getBeamShift": pair("BeamShift"),
            "getBeamTilt": pair("BeamTilt"),
            "getObjectiveStig": pair("ObjectiveStig"),
            "getCondenserStig": pair("CondenserStig"),
            "getTemStemMode": sh(lambda: g("TemStemMode")),
            "getSpotSize": sh(lambda: g("SpotSize")),
            "getIntensity": fl(lambda: g("Intensity")),
            # --- stage
            "getStagePosition": stage,
            "getStageHolder": sh(lambda: g("StageHolder")),
            "identifyInstrument": it(lambda: 0),
            # --- generic
            "getProperty": self._get_property,
            "executeBatchCommands": self._batch,
            "getOperationStatus": op_status,
            "stopOperation": stop_op,
            # --- setters
            "setBeamBlank": set_int("BeamBlank", "BEAMBLANK", "blank"),
            "setSpotSize": set_int("SpotSize", "SPOTSIZE", "spot"),
            "setMagnificationIndex": set_mag_index,
            "setMagnification": set_value("Magnification", "MAGINDEX", "mag"),
            "setIntensity": set_value("Intensity", "INTENSITY", "value"),
            "setCameraLength": set_value("CameraLength", "CAMERALENGTH", "cl-cm"),
            "setDiffractionFocus": set_value("DiffractionFocus", "DIFFFOCUS", "value"),
            "setScanRotation": set_value("ScanRotation", "SCANROTATION", "deg"),
            "setGunShift": set_pair("GunShift", "GUNSHIFT"),
            "setGunTilt": set_pair("GunTilt", "GUNTILT"),
            "setDefocus": set_value("Defocus", "DEFOCUS", "defocus-m"),
            "setImageShift": set_pair("ImageShift", "IMAGESHIFT"),
            "setDiffractionShift": set_pair("DiffractionShift", "DIFFRACTIONSHIFT"),
            "setBeamShift": set_pair("BeamShift", "BEAMSHIFT"),
            "setBeamTilt": set_pair("BeamTilt", "BEAMTILT", check_range=True),
            "setObjectiveStig": set_pair("ObjectiveStig", "OBJSTIG", check_range=True),
            "setCondenserStig": set_pair("CondenserStig", "CONDENSERSTIG", check_range=True),
            "setScreenPosition": set_int("ScreenPosition", "SCREENPOSITION", "pos"),
            "setColumnValvesOpen": set_int("ColumnValvesOpen", "COLUMNVALVES", "open"),
            "setStagePosition": set_stage,
            "setProjectionMode": set_int("ProjectionMode", "PROJECTIONMODE", "mode"),
            "setTemStemMode": set_int("TemStemMode", "TEMSTEMMODE", "mode"),
            "setFunctionMode": set_int("SubMode", "FUNCTIONMODE", "mode"),
            "setProbeMode": set_int("ProbeMode", "PROBEMODE", "mode"),
            "setAlphaSelector": set_int("AlphaSelector", "ALPHASELECTOR", "alpha"),
            "setStageMode": set_int("StageMode", "STAGEMODE", "mode"),
            "selectApertureKind": set_int("ApertureKind", "APERTUREKIND", "kind"),
            "setApertureSize": aperture_size,
        }
        h.update(self._corrector_handlers())
        return h

    # ------------------------------------------------------------ corrector
    #: CorrectorService.cpp kMaxCorrectorWaitMs
    MAX_CORRECTOR_WAIT_MS = 5000

    def corrector_unit(self):
        """The served :class:`~de_twin.column.corrector.CorrectorUnit`, or None."""
        corr = getattr(self.column, "corrector", None)
        if corr is None:
            return None
        side = self.corrector_side or corr.served_side
        return corr.units.get(side, corr.served)

    def _corrector_handlers(self) -> dict[str, Callable[[dict], tuple[str, list]]]:
        from ..column.ceos import ABSENT, DONE

        unit = self.corrector_unit

        def sleeper():
            corr = getattr(self.column, "corrector", None)
            return getattr(corr, "_sleeper", None)

        def ok(flag: bool) -> tuple[str, list]:
            return "setResult", [("success", 1 if flag else 0)]

        def string(element, fn, absent=""):
            def h(f):
                u = unit()
                return element, [("r", fn(u) if u is not None else absent)]
            return h

        def present(f):
            return "getintResponse", [("intResult", 1 if unit() is not None else 0)]

        def status(f):
            u = unit()
            return "getintResponse", [("intResult", u.status() if u is not None else ABSENT)]

        def endpoints(u):
            corr = self.column.corrector
            first = [u] + [v for v in corr.units.values() if v is not u]
            return "\n".join(v.endpoint() for v in first)

        def result(u):
            u.status()
            return u.last_result

        def error(u):
            u.status()
            return u.last_error

        def aberration(f):
            u = unit()
            name = f.get("name", "") or ""
            x, y, st = u.aberration(name) if u is not None else (0.0, 0.0, 0)
            return "getAberrationResponse", [("x", float(x)), ("y", float(y)), ("status", int(st))]

        def posted(fn):
            def h(f):
                u = unit()
                if u is None:
                    return ok(False)  # "no corrector is present"
                accepted, why = fn(u, f)
                if not accepted:
                    log.info("corrector command refused: %s", why)
                return ok(accepted)
            return h

        def wait_ms(f) -> int:
            w = _as_int(f, "waitMs")
            return 0 if w <= 0 else min(w, self.MAX_CORRECTOR_WAIT_MS)

        def correct(f):
            u = unit()
            if u is None:
                return ok(False)
            name = f.get("name", "") or ""
            scale = 1.0 if name == "WD" else 1e9  # WD radians, the rest metres
            value = (complex(_as_float(f, "valueX"), _as_float(f, "valueY")) * scale
                     if _as_int(f, "useValue") else None)
            target = (complex(_as_float(f, "targetX"), _as_float(f, "targetY")) * scale
                      if _as_int(f, "useTarget") else None)
            accepted, why = u.post_correct(name, value, target, f.get("select", "") or "")
            w = wait_ms(f)
            if not accepted or w == 0:
                if not accepted:
                    log.info("corrector correction refused: %s", why)
                return ok(accepted)
            return ok(u.wait(w / 1000.0, sleeper()) == DONE)

        def set_tilt(f):
            u = unit()
            if u is None:
                return ok(False)
            w = wait_ms(f)
            completed, still_running, why = u.set_beam_tilt(
                _as_float(f, "x"), _as_float(f, "y"), _as_int(f, "relative") != 0,
                wait_s=w / 1000.0, sleeper=sleeper())
            return ok(still_running if w == 0 else completed)

        def get_tilt(f):
            u = unit()
            if u is None:
                return "getAberrationResponse", [("x", 0.0), ("y", 0.0), ("status", 0)]
            with u.lock:
                t, known = u.beam_tilt_commanded, u.beam_tilt_known
            return "getAberrationResponse", [("x", float(t.real)), ("y", float(t.imag)),
                                             ("status", 1 if known else 0)]

        def reconnect(f):
            u = unit()
            if u is None:
                return ok(True)  # accepted; detection finds nothing and status stays 6
            return ok(u.post_reconnect()[0])

        def tableau(u, f):
            return u.post_tableau(f.get("tabType", "") or "", _as_float(f, "angle"),
                                  f.get("maxFit", "") or "")

        return {
            "getCorrectorPresent": present,
            "getCorrectorStatus": status,
            "getCorrectorInfo": string("getCorrectorInfoResponse", lambda u: u.info()),
            "getCorrectorEndpoints": string("getCorrectorEndpointsResponse", endpoints),
            "getCorrectorError": string("getCorrectorErrorResponse", error, "no corrector responded"),
            "getCorrectorResult": string("getCorrectorResultResponse", result),
            "getAberrations": string("getAberrationsResponse", lambda u: u.aberrations_string()),
            "getAberration": aberration,
            "getAlignmentFile": string("getAlignmentFileResponse", lambda u: u.alignment_file),
            "correctorMeasureC1A1": posted(lambda u, f: u.post_measure_c1a1()),
            "correctorAcquireTableau": posted(tableau),
            "correctorCorrectAberration": correct,
            "correctorFetchAlignmentFile": posted(lambda u, f: u.post_fetch_alignment()),
            "correctorPutAlignmentFile": posted(lambda u, f: u.post_put_alignment(f.get("data", "") or "")),
            "correctorGetConfigOption": posted(lambda u, f: u.post_get_config(f.get("name", "") or "")),
            "correctorSetConfigOption": posted(
                lambda u, f: u.post_set_config(f.get("name", "") or "", f.get("value", "") or "")),
            "correctorReadBack": posted(lambda u, f: u.post_readback()),
            "correctorReconnect": reconnect,
            "setCorrectorBeamTilt": set_tilt,
            "getCorrectorBeamTilt": get_tilt,
        }

    # ------------------------------------------------------------ getProperty
    _PROPERTY_OPS = {
        "BeamBlank": "BEAMBLANK", "SpotSize": "SPOTSIZE", "Defocus": "DEFOCUS",
        "ScreenPosition": "SCREENPOSITION", "ColumnValvesOpen": "COLUMNVALVES",
        "ProjectionMode": "PROJECTIONMODE", "TemStemMode": "TEMSTEMMODE", "StageMode": "STAGEMODE",
        "Magnification": "MAGINDEX", "Intensity": "INTENSITY", "CameraLength": "CAMERALENGTH",
        "DiffractionFocus": "DIFFFOCUS", "ScanRotation": "SCANROTATION", "GunShiftX": "GUNSHIFT",
        "GunShiftY": "GUNSHIFT", "GunTiltX": "GUNTILT", "GunTiltY": "GUNTILT",
        "ImageShiftX": "IMAGESHIFT", "ImageShiftY": "IMAGESHIFT",
        "DiffractionShiftX": "DIFFRACTIONSHIFT", "DiffractionShiftY": "DIFFRACTIONSHIFT",
        "BeamShiftX": "BEAMSHIFT", "BeamShiftY": "BEAMSHIFT", "StageX": "STAGE_X",
        "StageY": "STAGE_Y", "StageZ": "STAGE_Z", "StageA": "STAGE_A", "StageB": "STAGE_B",
        # twin extensions (not in the channel; harmless for its clients)
        "SubMode": "FUNCTIONMODE", "FunctionMode": "FUNCTIONMODE", "ProbeMode": "GLOBAL",
    }

    def property_value(self, key: str) -> str:
        """The getProperty string for ``key`` ("" for unknown keys, like the channel)."""
        g = self.column.get
        ints = {"BeamBlank", "SpotSize", "ScreenPosition", "ColumnValvesOpen", "ProjectionMode",
                "TemStemMode", "StageMode", "ProbeMode", "AlphaSelector", "CondenserApertureIndex",
                "SubMode", "FunctionMode"}
        floats = {"Defocus", "Magnification", "ScreenCurrent", "Intensity", "CameraLength",
                  "DiffractionFocus", "ScanRotation", "StageX", "StageY", "StageZ", "StageA",
                  "StageB"}
        if key in ints:
            return "%d" % int(g(key))
        if key in floats:
            return "%.6g" % float(g(key))
        if key == "ConvergenceAngle":
            return "%.3f" % float(g("ConvergenceAngle"))
        if key in ("VacuumGauges", "CryoInfo", "AutoloaderInfo"):
            return ""  # JEOL-shaped: no gauges/dewar/autoloader strings
        if key == "InstrumentType":
            return str(g("InstrumentType"))
        for base in ("GunShift", "GunTilt", "ImageShift", "DiffractionShift", "BeamShift"):
            if key in (base + "X", base + "Y"):
                v = g(base)
                return "%.6g" % float(v[0] if key.endswith("X") else v[1])
        return ""

    def _get_property(self, fields: dict) -> tuple[str, list]:
        key = fields.get("property", "") or ""
        value = self.property_value(key)
        op = self._PROPERTY_OPS.get(key, "GLOBAL")
        return "getPropertyResponse", [("value", value), ("status", self.op_status(PROP_OP[op]))]

    # ------------------------------------------------------------ batch
    def batch_field(self, key: str) -> str:
        """One executeBatchCommands field, formatted like the channel's stringstream."""
        g = self.column.get
        if key == "GUN_HT":
            return _g(g("HighTension"))
        if key == "GUN_HTMax":
            return _g(g("MaxHighTension"))
        if key == "GUN_HTState":
            return str(int(g("HTState")))
        if key == "PROJ_BeamBlankState":
            return str(int(g("BeamBlank")))
        if key == "ILLUM_Filament":
            return _g(g("Filament"))
        if key == "PROJ_Mode":
            return str(int(g("ProjectionMode")))
        if key == "PROJ_Magnification":
            return _g(g("Magnification"))
        if key in ("PROJ_ImageShift", "PROJ_ImageBeamShift", "STAGE_Status"):
            return "0"
        if key == "PROJ_Focus":
            return str(int(g("Focus")))
        if key == "PROJ_Defocus":
            return _g(g("Defocus"))
        if key == "PROJ_ScreenPosition":
            return str(int(g("ScreenPosition")))
        if key == "CAMERA_MeasuredExposureTime":
            return _g(g("ExposureTime"))
        if key == "STAGE_Holder":
            return str(int(g("StageHolder")))
        if key == "VACUUM_Status":
            return str(int(g("VacuumStatus")))
        if key == "VACUUM_ColumnValvesOpen":
            return str(int(g("ColumnValvesOpen")))
        if key == "VACUUM_PVPRunning":
            return _g(int(g("VacuumStatus")))
        if key == "VACUUM_Gauges":
            return _g(g("VacuumGauges"))
        if key == "PROJ_CameraLength":
            return _g(g("CameraLength"))
        if key == "INSTRUMENT_Type":
            return str(g("InstrumentType"))
        if key == "PROJ_SubMode":
            return str(int(g("SubMode")))
        if key == "PROJ_TemStemMode":
            return str(int(g("TemStemMode")))
        if key == "STAGE_PositionFull":
            return ";".join(_g(v) for v in self.column.stage_position())
        pairs = {"ILLUM_ImageShift": "ImageShift", "PROJ_DiffractionShift": "DiffractionShift",
                 "ILLUM_BeamShift": "BeamShift", "ILLUM_BeamTilt": "BeamTilt",
                 "PROJ_ObjStig": "ObjectiveStig", "ILLUM_CondenserStig": "CondenserStig"}
        if key in pairs:
            x, y = g(pairs[key])
            return f"{_g(x)};{_g(y)}"
        if key == "OP_Status":
            return str(self.op_status(PROP_OP["GLOBAL"]))
        if key == "STAGE_Mode":
            return str(int(g("StageMode")))
        if key == "ILLUM_SpotSize":
            return str(int(g("SpotSize")))
        if key == "ILLUM_ProbeMode":
            return str(int(g("ProbeMode")))
        if key == "ILLUM_ConvergenceAngle":
            return "%.3f" % float(g("ConvergenceAngle"))
        if key == "ILLUM_AlphaSelector":
            return str(int(g("AlphaSelector")))
        if key == "ILLUM_CondenserApertureIndex":
            return str(int(g("CondenserApertureIndex")))
        return ""  # unknown keys (and STAGE_Position) answer empty, as the channel does

    def _batch(self, fields: dict) -> tuple[str, list]:
        commands = fields.get("commands", "") or ""
        keys = commands.split(",") if commands else []
        total = _as_int(fields, "totalCommands", len(keys))
        total = min(total, len(keys))  # never trust totalCommands beyond what parsed
        with self.column.lock:
            out = ",".join(self.batch_field(k.strip()) for k in keys[:total])
        return "executeBatchCommandsResponse", [("r", out)]


def serve(column: Column, host: str = "0.0.0.0", port: int = 5002) -> TemChannelServer:
    """Start a server and return it (non-blocking)."""
    return TemChannelServer(column, host=host, port=port).start()


def main(argv=None) -> int:
    """``python -m de_twin.faces.temchannel_soap [--port 5002] [--vendor JEOL]``: a
    stand-alone simulated column (a Dummy DE-TEM-Channel without Windows)."""
    import argparse
    import time

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5002)
    p.add_argument("--vendor", default="JEOL")
    p.add_argument("--corrector", default="none", choices=("none", "probe", "image", "both"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    column = Column(instrument_type=args.vendor, corrector=args.corrector, seed=args.seed)
    srv = TemChannelServer(column, host=args.host, port=args.port).start()
    print(f"started .... DE-TEM-Channel SOAP face on {args.host}:{srv.port}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
