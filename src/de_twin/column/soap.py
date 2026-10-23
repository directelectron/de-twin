"""DE-TEM-Channel SOAP protocol: envelopes, the batch key list, and a stdlib client.

DE-TEM-Channel is a gSOAP RPC/encoded service on port 5002, namespace
``http://tempuri.org/ns.xsd`` (``service/EMinstrumentWebservice.wsdl``). A call is
``<ns:method><param>..</param></ns:method>`` in the SOAP body; the response is a
struct named after its *type*, not the operation (``getfloatResponse``,
``gettwofloatResponse``, ``getfivefloatResponse``, ``getshortResponse``,
``getintResponse``, ``setResult``, ``getPropertyResponse``, or
``<op>Response`` with an ``<r>`` string). See ``docs/soap-samples``.

``executeBatchCommands(commands="K1,K2,...", totalCommands=n)`` answers one
comma-separated string, one field per key, with multi-valued fields ``;``
separated and unknown keys answered empty. DE-Server's client
(``Instrument.cpp``) only accepts the answer when it splits into exactly
``totalCommands`` fields.
"""

from __future__ import annotations

import math
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from typing import Any
from xml.sax.saxutils import escape

NAMESPACE = "http://tempuri.org/ns.xsd"
DEFAULT_PORT = 5002

#: DE-TEM-Channel ``core/confige.cpp`` command_list, in wire order. DE-Server's
#: Instrument.cpp mirrors this list (its enum must stay aligned).
BATCH_KEYS = (
    "GUN_HT", "GUN_HTMax", "GUN_HTState", "PROJ_BeamBlankState", "ILLUM_Filament",
    "PROJ_Mode", "PROJ_Magnification", "PROJ_ImageShift", "PROJ_Focus", "PROJ_Defocus",
    "PROJ_ImageBeamShift", "PROJ_ScreenPosition", "CAMERA_MeasuredExposureTime",
    "STAGE_Status", "STAGE_Holder", "STAGE_Position", "VACUUM_Status",
    "VACUUM_ColumnValvesOpen", "VACUUM_PVPRunning", "VACUUM_Gauges", "PROJ_CameraLength",
    "INSTRUMENT_Type", "PROJ_SubMode", "PROJ_TemStemMode", "STAGE_PositionFull",
    "ILLUM_ImageShift", "ILLUM_BeamShift", "ILLUM_BeamTilt", "PROJ_ObjStig",
    "ILLUM_CondenserStig", "OP_Status", "STAGE_Mode", "ILLUM_SpotSize", "ILLUM_ProbeMode",
    "ILLUM_ConvergenceAngle", "ILLUM_AlphaSelector", "ILLUM_CondenserApertureIndex",
    "PROJ_DiffractionShift",
)

#: The batch DE-Server polls every cycle (Instrument.cpp get_Properties order).
DESERVER_BATCH = (
    "PROJ_Mode", "PROJ_SubMode", "PROJ_Magnification", "PROJ_CameraLength", "PROJ_Defocus",
    "GUN_HT", "PROJ_ScreenPosition", "PROJ_BeamBlankState", "PROJ_TemStemMode",
    "INSTRUMENT_Type", "STAGE_PositionFull", "ILLUM_BeamShift", "ILLUM_ImageShift",
    "ILLUM_SpotSize", "ILLUM_ProbeMode", "ILLUM_ConvergenceAngle", "ILLUM_AlphaSelector",
    "ILLUM_CondenserApertureIndex",
)

#: getProperty status ids (DE-TEM-Channel core/global.h PropertyOperationId).
PROP_OP = {
    "GLOBAL": 0, "BEAMBLANK": 100, "SPOTSIZE": 101, "MAGINDEX": 102, "DEFOCUS": 103,
    "IMAGESHIFT": 104, "BEAMSHIFT": 105, "BEAMTILT": 106, "OBJSTIG": 107,
    "CONDENSERSTIG": 108, "SCREENPOSITION": 109, "COLUMNVALVES": 110, "STAGE_X": 111,
    "STAGE_Y": 112, "STAGE_Z": 113, "STAGE_A": 114, "STAGE_B": 115, "STAGEPOSITION": 116,
    "PROJECTIONMODE": 117, "TEMSTEMMODE": 118, "APERTUREKIND": 119, "APERTURESIZE": 120,
    "STAGEMODE": 121, "INTENSITY": 122, "CAMERALENGTH": 123, "DIFFFOCUS": 124,
    "SCANROTATION": 125, "GUNSHIFT": 126, "GUNTILT": 127, "FUNCTIONMODE": 128,
    "PROBEMODE": 129, "ALPHASELECTOR": 130, "DIFFRACTIONSHIFT": 131,
}

#: OperationState (core/global.h).
OP_IDLE, OP_RUNNING, OP_DONE, OP_FAILED, OP_ABORTED, OP_TIMEOUT = range(6)

_ENVELOPE_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    "<SOAP-ENV:Envelope\n"
    ' xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/"\n'
    ' xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/"\n'
    ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
    ' xmlns:xsd="http://www.w3.org/2001/XMLSchema"\n'
    ' xmlns:ns="http://tempuri.org/ns.xsd">\n'
    ' <SOAP-ENV:Body SOAP-ENV:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">\n'
)
_ENVELOPE_TAIL = " </SOAP-ENV:Body>\n</SOAP-ENV:Envelope>\n"


class SoapError(RuntimeError):
    """A SOAP call failed (transport, fault or unparseable response)."""


def fmt_value(value: Any) -> str:
    """Serialise one SOAP value the way gSOAP does (floats ``%.9G``)."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value in (float("inf"), float("-inf")):
            return "INF" if value > 0 else "-INF"
        return format(value, ".9G")
    return escape("" if value is None else str(value))


def envelope(element: str, fields: Iterable[tuple[str, Any]]) -> str:
    """A gSOAP-shaped envelope with ``<ns:element>`` holding ``fields`` in order."""
    body = "".join(f"   <{k}>{fmt_value(v)}</{k}>\n" for k, v in fields)
    return f"{_ENVELOPE_HEAD}  <ns:{element}>\n{body}  </ns:{element}>\n{_ENVELOPE_TAIL}"


def fault_envelope(message: str, code: str = "SOAP-ENV:Client") -> str:
    return (
        f"{_ENVELOPE_HEAD}  <SOAP-ENV:Fault>\n   <faultcode>{code}</faultcode>\n"
        f"   <faultstring>{escape(message)}</faultstring>\n  </SOAP-ENV:Fault>\n{_ENVELOPE_TAIL}"
    )


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def parse_body(xml: str | bytes) -> tuple[str, dict[str, str]]:
    """``(element local name, {child local name: text})`` of the first Body child.

    Namespace/prefix agnostic, tolerant of ``xsi:type`` and whitespace, so it
    reads gSOAP, de_microscope and our own envelopes alike. Raises
    :class:`SoapError` for a SOAP Fault.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise SoapError(f"unparseable SOAP message: {exc}") from None
    body = None
    for el in root.iter():
        if _local(el.tag) == "Body":
            body = el
            break
    if body is None:
        raise SoapError("no SOAP Body")
    children = list(body)
    if not children:
        raise SoapError("empty SOAP Body")
    op = children[0]
    fields: dict[str, str] = {}
    for child in op:
        fields[_local(child.tag)] = (child.text or "").strip()
    if _local(op.tag) == "Fault":
        raise SoapError(fields.get("faultstring") or "SOAP Fault")
    return _local(op.tag), fields


def split_batch(response: str, total: int) -> list[str] | None:
    """Split an executeBatchCommands answer exactly as DE-Server does.

    DE-Server's ``split`` only produces fields when a comma is present and then
    accepts the answer only if the field count equals ``totalCommands``. For a
    single key (no comma) the whole answer is the field.
    """
    if total <= 1:
        return [response] if total == 1 else []
    parts = response.split(",")
    return parts if len(parts) == total else None


def _num(text: str):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return text
    if v.is_integer() and re.fullmatch(r"[-+]?\d+", text.strip() or "x"):
        return int(v)
    return v


# ---------------------------------------------------------------------------
# Property registry (the same names as de_microscope / Column)
# ---------------------------------------------------------------------------
#   get: ("batch", key) | ("value", method, tag) | ("pair", method) | ("prop", key)
#        | ("props", kx, ky) | ("stage", index) | ("stage_all",) | ("string", method)
#   set: ("value", method, arg) | ("pair", method) | ("stage", axis) | ("stage_all",)
#        | ("aperture", kind)
PROPERTIES: dict[str, tuple[tuple | None, tuple | None, str | None]] = {
    "Magnification": (("batch", "PROJ_Magnification"), ("value", "setMagnification", "mag"), "Magnification"),
    "MagnificationIndex": (None, ("value", "setMagnificationIndex", "idx"), "Magnification"),
    "CameraLength": (("batch", "PROJ_CameraLength"), ("value", "setCameraLength", "cl-cm"), "CameraLength"),
    "Defocus": (("value", "getDeFocusValue", "floatResult"), ("value", "setDefocus", "defocus-m"), "Defocus"),
    "SpotSize": (("value", "getSpotSize", "shortResult"), ("value", "setSpotSize", "spot"), "SpotSize"),
    "ProbeMode": (("prop", "ProbeMode"), ("value", "setProbeMode", "mode"), None),
    "ConvergenceAngle": (("prop", "ConvergenceAngle"), None, None),
    "AlphaSelector": (("prop", "AlphaSelector"), ("value", "setAlphaSelector", "alpha"), None),
    "CondenserApertureIndex": (("prop", "CondenserApertureIndex"), ("aperture", 1), None),
    "Intensity": (("prop", "Intensity"), ("value", "setIntensity", "value"), "Intensity"),
    "ImageShift": (("pair", "getImageShift"), ("pair", "setImageShift"), "ImageShiftX"),
    "BeamShift": (("pair", "getBeamShift"), ("pair", "setBeamShift"), "BeamShiftX"),
    "BeamTilt": (("pair", "getBeamTilt"), ("pair", "setBeamTilt"), None),
    "ObjectiveStig": (("pair", "getObjectiveStig"), ("pair", "setObjectiveStig"), None),
    "CondenserStig": (("pair", "getCondenserStig"), ("pair", "setCondenserStig"), None),
    "DiffractionShift": (("pair", "getDiffractionShift"), ("pair", "setDiffractionShift"), "DiffractionShiftX"),
    "StagePosition": (("stage_all",), ("stage_all",), None),
    "StageX": (("stage", 0), ("stage", "x"), "StageX"),
    "StageY": (("stage", 1), ("stage", "y"), "StageY"),
    "StageZ": (("stage", 2), ("stage", "z"), "StageZ"),
    "StageA": (("stage", 3), ("stage", "a"), "StageA"),
    "StageB": (("stage", 4), ("stage", "b"), "StageB"),
    "TemStemMode": (("value", "getTemStemMode", "shortResult"), ("value", "setTemStemMode", "mode"), "TemStemMode"),
    "ProjectionMode": (("value", "getProjectMode", "shortResult"), ("value", "setProjectionMode", "mode"), "ProjectionMode"),
    "SubMode": (("value", "getProjectSubMode", "shortResult"), ("value", "setFunctionMode", "mode"), "SubMode"),
    "BeamBlank": (("value", "getBeamBlankState", "intResult"), ("value", "setBeamBlank", "blank"), "BeamBlank"),
    "ScreenPosition": (("value", "getScreenPosition", "shortResult"), ("value", "setScreenPosition", "pos"), "ScreenPosition"),
    "ColumnValvesOpen": (("value", "getColumnValveStatus", "shortResult"), ("value", "setColumnValvesOpen", "open"), "ColumnValvesOpen"),
    "HighTension": (("value", "getAccelVoltage", "floatResult"), None, None),
    "MaxHighTension": (("value", "getMaxHTValue", "floatResult"), None, None),
    "HTState": (("value", "getHTState", "shortResult"), None, None),
    "Filament": (("value", "getFilamentValue", "floatResult"), None, None),
    "Focus": (("value", "getFocusValue", "floatResult"), None, None),
    "InstrumentType": (("batch", "INSTRUMENT_Type"), None, None),
    "OperationStatus": (("value", "getOperationStatus", "intResult"), None, None),
    "StageMode": (("prop", "StageMode"), ("value", "setStageMode", "mode"), "StageMode"),
    "StageHolder": (("value", "getStageHolder", "shortResult"), None, None),
    "ScanRotation": (("prop", "ScanRotation"), ("value", "setScanRotation", "deg"), "ScanRotation"),
    "DiffractionFocus": (("prop", "DiffractionFocus"), ("value", "setDiffractionFocus", "value"), "DiffractionFocus"),
    "GunShift": (("props", "GunShiftX", "GunShiftY"), ("pair", "setGunShift"), "GunShiftX"),
    "GunTilt": (("props", "GunTiltX", "GunTiltY"), ("pair", "setGunTilt"), "GunTiltX"),
    "ScreenCurrent": (("prop", "ScreenCurrent"), None, None),
    "VacuumStatus": (("value", "getVacuumStatus", "floatResult"), None, None),
    "ExposureTime": (("value", "getExposureTime", "floatResult"), None, None),
    "MagString": (("string", "getMagStr"), None, None),
}


class SoapTemChannelClient:
    """A small, stdlib-only client for DE-TEM-Channel (or the twin's SOAP face).

    >>> c = SoapTemChannelClient("127.0.0.1", 5002)
    >>> c.execute_batch(["GUN_HT", "PROJ_Magnification"])     # ['200000', '20000']
    >>> c.get("StagePosition"); c.set("Magnification", 50000)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 10.0):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.url = f"http://{host}:{self.port}/"

    def __repr__(self) -> str:
        return f"SoapTemChannelClient({self.host!r}, {self.port})"

    # ------------------------------------------------------------ transport
    def call(self, method: str, **params: Any) -> dict[str, str]:
        """Invoke ``method``; keyword ``cl_cm`` style names map to ``cl-cm``.
        Returns the response element's children as strings."""
        fields = [(k.replace("_", "-"), v) for k, v in params.items()]
        payload = envelope(method, fields).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "text/xml; charset=utf-8", "SOAPAction": '""'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                text = resp.read()
        except urllib.error.HTTPError as exc:  # gSOAP answers faults with HTTP 500
            text = exc.read()
            parse_body(text)  # a SOAP fault body raises its own error
            raise SoapError(f"{method}: HTTP {exc.code}") from None
        except OSError as exc:
            raise SoapError(f"{method} failed: {exc}") from None
        _, out = parse_body(text)
        return out

    # ---------------------------------------------------------------- batch
    def execute_batch(self, keys: Iterable[str]) -> list[str]:
        keys = list(keys)
        out = self.call("executeBatchCommands", commands=",".join(keys), totalCommands=len(keys))
        fields = split_batch(out.get("r", ""), len(keys))
        if fields is None:
            raise SoapError(f"batch answered {out.get('r', '')!r} for {len(keys)} keys")
        return fields

    def batch_dict(self, keys: Iterable[str] = BATCH_KEYS) -> dict[str, str]:
        keys = list(keys)
        return dict(zip(keys, self.execute_batch(keys)))

    def get_property(self, key: str) -> tuple[str, int]:
        out = self.call("getProperty", property=key)
        return out.get("value", ""), int(out.get("status", "0") or 0)

    def operation_status(self, op_id: int = 0) -> int:
        return int(self.call("getOperationStatus", val=int(op_id)).get("intResult", "0"))

    def stop(self) -> bool:
        return self.call("stopOperation", val=1).get("success") == "1"

    # ------------------------------------------------------ typed getters
    def get_stage_position(self) -> tuple[float, float, float, float, float]:
        out = self.call("getStagePosition", val=1)
        return tuple(float(out[k]) for k in ("one", "two", "three", "four", "five"))  # type: ignore

    def get_pair(self, method: str) -> tuple[float, float]:
        out = self.call(method, val=1)
        return float(out["floatone"]), float(out["floattwo"])

    def set_stage_position(self, x=None, y=None, z=None, a=None, b=None) -> bool:
        axes = {"x": x, "y": y, "z": z, "a": a, "b": b}
        mask = sum(1 << i for i, v in enumerate(axes.values()) if v is not None)
        params = {k: float(v or 0.0) for k, v in axes.items()}
        out = self.call("setStagePosition", **params, **{"axes-mask": mask})
        return out.get("success") == "1"

    # ------------------------------------------------------- name registry
    def get(self, name: str) -> Any:
        try:
            spec = PROPERTIES[name][0]
        except KeyError:
            raise KeyError(f"unknown property {name!r}") from None
        if spec is None:
            raise KeyError(f"{name!r} is write-only")
        kind = spec[0]
        if kind == "batch":
            return _num(self.execute_batch([spec[1]])[0])
        if kind == "value":
            return _num(self.call(spec[1], val=1).get(spec[2], ""))
        if kind == "string":
            return self.call(spec[1], val=1).get("r", "")
        if kind == "pair":
            return self.get_pair(spec[1])
        if kind == "prop":
            return _num(self.get_property(spec[1])[0])
        if kind == "props":
            return (_num(self.get_property(spec[1])[0]), _num(self.get_property(spec[2])[0]))
        if kind == "stage":
            return self.get_stage_position()[spec[1]]
        if kind == "stage_all":
            return dict(zip("xyzab", self.get_stage_position()))
        raise SoapError(f"bad get spec {spec!r}")

    def set(self, name: str, value: Any) -> bool:
        """Write ``name``. Returns the channel's ``success`` flag."""
        try:
            spec = PROPERTIES[name][1]
        except KeyError:
            raise KeyError(f"unknown property {name!r}") from None
        if spec is None:
            raise KeyError(f"{name!r} is read-only")
        kind = spec[0]
        if kind == "value":
            out = self.call(spec[1], **{spec[2]: value})
        elif kind == "pair":
            x, y = value
            out = self.call(spec[1], x=float(x), y=float(y))
        elif kind == "stage":
            return self.set_stage_position(**{spec[1]: float(value)})
        elif kind == "stage_all":
            v = dict(value)
            v = {{"alpha": "a", "beta": "b"}.get(k, k): val for k, val in v.items()}
            return self.set_stage_position(**v)
        elif kind == "aperture":
            out = self.call("setApertureSize", kind=int(spec[1]), size=int(value))
        else:
            raise SoapError(f"bad set spec {spec!r}")
        return out.get("success") == "1"

    def status(self, name: str) -> int:
        key = PROPERTIES.get(name, (None, None, None))[2]
        if key:
            return self.get_property(key)[1]
        return self.operation_status(0)

    # ------------------------------------------------------ corrector (F1/F2)
    # Commands are posted: True = accepted. Poll corrector_status() (0 idle,
    # 1 running, 2 done, 3 failed, 6 absent) or corrector_wait(), then read
    # corrector_result() / aberrations() / corrector_error(). CEOS units: metres
    # (radians for WD).
    def _ok(self, method: str, **params: Any) -> bool:
        return self.call(method, **params).get("success") == "1"

    def corrector_present(self) -> bool:
        return self.call("getCorrectorPresent", val=0).get("intResult") == "1"

    def corrector_status(self) -> int:
        return int(self.call("getCorrectorStatus", val=0).get("intResult", "6") or 6)

    def corrector_info(self) -> dict[str, str]:
        """``getCorrectorInfo`` parsed (``correctorType``, ``mode``, ``currentLabel``, ...)."""
        from .ceos import parse_info

        return parse_info(self.call("getCorrectorInfo", val=0).get("r", ""))

    def corrector_endpoints(self) -> list[str]:
        r = self.call("getCorrectorEndpoints", val=0).get("r", "")
        return [line for line in r.splitlines() if line.strip()]

    def corrector_error(self) -> str:
        return self.call("getCorrectorError", val=0).get("r", "")

    def corrector_result(self) -> str:
        return self.call("getCorrectorResult", val=0).get("r", "")

    def aberrations(self) -> list[tuple[str, tuple[float, float]]]:
        """Last measurement, ``[(name, (x, y))]`` in the corrector's order and units."""
        from .ceos import parse_aberration_set

        return parse_aberration_set(self.call("getAberrations", val=0).get("r", ""))

    def aberration(self, name: str) -> tuple[float, float, bool]:
        out = self.call("getAberration", name=name)
        return float(out.get("x", 0) or 0), float(out.get("y", 0) or 0), out.get("status") == "1"

    def alignment_file(self) -> str:
        return self.call("getAlignmentFile", val=0).get("r", "")

    def corrector_measure_c1a1(self) -> bool:
        return self._ok("correctorMeasureC1A1", val=0)

    def corrector_acquire_tableau(self, tab_type: str = "", angle_mrad: float = 0.0,
                                  max_fit: str = "") -> bool:
        return self._ok("correctorAcquireTableau", tabType=tab_type, angle=float(angle_mrad),
                        maxFit=max_fit)

    def corrector_correct(self, name: str, value: tuple[float, float] | None = None,
                          target: tuple[float, float] | None = None, select: str = "",
                          wait_ms: int = 0) -> bool:
        """``correctorCorrectAberration``. ``value`` None = the corrector's own estimate,
        ``target`` None = zero. With ``wait_ms`` > 0, True means *completed*."""
        vx, vy = value if value is not None else (0.0, 0.0)
        tx, ty = target if target is not None else (0.0, 0.0)
        return self._ok("correctorCorrectAberration", name=name, useValue=int(value is not None),
                        valueX=float(vx), valueY=float(vy), useTarget=int(target is not None),
                        targetX=float(tx), targetY=float(ty), select=select, waitMs=int(wait_ms))

    def corrector_fetch_alignment_file(self) -> bool:
        return self._ok("correctorFetchAlignmentFile", val=0)

    def corrector_put_alignment_file(self, data: str) -> bool:
        return self._ok("correctorPutAlignmentFile", data=data)

    def corrector_get_config_option(self, name: str) -> bool:
        return self._ok("correctorGetConfigOption", name=name)

    def corrector_set_config_option(self, name: str, value: Any) -> bool:
        if isinstance(value, bool):
            value = "true" if value else "false"
        return self._ok("correctorSetConfigOption", name=name, value=str(value))

    def corrector_read_back(self) -> bool:
        return self._ok("correctorReadBack", val=0)

    def corrector_reconnect(self) -> bool:
        return self._ok("correctorReconnect", val=0)

    def set_corrector_beam_tilt(self, x_rad: float, y_rad: float, relative: bool = True,
                                wait_ms: int = 0) -> bool:
        return self._ok("setCorrectorBeamTilt", x=float(x_rad), y=float(y_rad),
                        relative=int(bool(relative)), waitMs=int(wait_ms))

    def get_corrector_beam_tilt(self) -> tuple[float, float, bool]:
        out = self.call("getCorrectorBeamTilt", val=0)
        return float(out.get("x", 0) or 0), float(out.get("y", 0) or 0), out.get("status") == "1"

    def corrector_wait(self, timeout: float = 120.0, poll_s: float = 0.05) -> int:
        """Poll ``getCorrectorStatus`` (wall clock) until it leaves 1 (running)."""
        import time as _time

        deadline = _time.monotonic() + timeout
        while True:
            st = self.corrector_status()
            if st != 1 or _time.monotonic() > deadline:
                return st
            _time.sleep(poll_s)
