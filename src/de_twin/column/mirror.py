"""MirrorColumn: a Column that follows a real (or Dummy) DE-TEM-Channel.

It polls ``executeBatchCommands`` at a fixed period and turns the answer into a
:class:`MicroscopeState` with the same keys, units and vendor normalisation as
DE-Server's ``Camera::ParseEmMetaData`` (Camera.cpp): ``GUN_HT`` volts -> kV,
``PROJ_CameraLength`` cm -> mm, ``PROJ_Defocus`` um, ``STAGE_PositionFull``
"x;y;z;a;b" um/deg, ``ILLUM_ProbeMode`` normalised by ``INSTRUMENT_Type``
(JEOL 0 TEM/1 EDS/2 NBD/3 CBD, otherwise 0 nanoprobe/1 microprobe),
``ILLUM_ConvergenceAngle`` < 0 = "not reported". Missing or empty fields keep
their previous value, as DE-Server keeps its defaults for older channels.

Writes (``set`` and every typed setter) are forwarded to the real channel, and
the mirror re-polls immediately so a read after a write sees the channel's
answer.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from ..state import MicroscopeState, ProbeMode, Projection, StagePosition, TemStem, Vec2
from . import ladders as L
from .column import (
    BEAM_TILT_MRAD_PER_UNIT,
    DIFF_SHIFT_MRAD_PER_UNIT,
    Column,
    ColumnRefused,
    is_tfs,
    probe_mode_from_raw,
    screen_from_raw,
)
from .soap import BATCH_KEYS, SoapError, SoapTemChannelClient

log = logging.getLogger(__name__)


def _f(values: dict, key: str) -> Optional[float]:
    v = values.get(key, "")
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _i(values: dict, key: str) -> Optional[int]:
    v = _f(values, key)
    return None if v is None else int(v)


def _parse_semicolon_floats(values: dict, key: str, n: int) -> Optional[list[float]]:
    raw = values.get(key, "")
    if not raw:
        return None
    parts = str(raw).replace(",", ";").split(";")
    try:
        out = [float(p) for p in parts[:n]]
    except ValueError:
        return None
    return out if len(out) == n else None


def state_from_batch(
    values: dict[str, str],
    prev: Optional[MicroscopeState] = None,
    *,
    camera_length_unit: str = "cm",
) -> tuple[MicroscopeState, dict[str, Any]]:
    """Parse an executeBatchCommands answer (``{key: field}``) into a state.

    Returns ``(state, info)`` where ``info`` carries the raw fields that have no
    MicroscopeState slot (``op_status``, ``convergence_explicit``, ``stage_mode``,
    ``max_ht_v``, ``filament``, ``raw_probe_mode``, ``raw_sub_mode``).
    """
    s = (prev or MicroscopeState()).copy()
    info: dict[str, Any] = {}

    vendor = values.get("INSTRUMENT_Type")
    if vendor:
        s.instrument_type = vendor.strip()
    vendor = s.instrument_type

    ht = _f(values, "GUN_HT")
    if ht is not None and ht > 0:
        s.ht_kv = ht / 1000.0 if ht > 1000 else ht  # channel reports volts
    if (v := _f(values, "GUN_HTMax")) is not None:
        info["max_ht_v"] = v
    if (v := _i(values, "GUN_HTState")) is not None:
        s.ht_on = bool(v)
    if (v := _i(values, "PROJ_BeamBlankState")) is not None:
        s.beam_blanked = bool(v)
    if (v := _f(values, "ILLUM_Filament")) is not None:
        info["filament"] = v
    if (v := _i(values, "VACUUM_ColumnValvesOpen")) is not None:
        s.column_valves_open = bool(v)
    if (v := _i(values, "PROJ_ScreenPosition")) is not None:
        try:
            s.screen_position = screen_from_raw(vendor, v)
        except ColumnRefused:
            pass

    tem_stem = _i(values, "PROJ_TemStemMode")
    if tem_stem in (0, 1):
        s.tem_stem = TemStem(tem_stem)
    mode = _i(values, "PROJ_Mode")
    sub = _i(values, "PROJ_SubMode")
    if sub is not None:
        info["raw_sub_mode"] = sub
    fm: Optional[int] = None
    if mode == 2:
        fm = L.FM_DIFF
    elif sub is not None:
        if is_tfs(vendor):
            fm = L.FM_FROM_FEI_SUBMODE.get(sub)
        elif 0 <= sub <= 4:
            fm = sub
    if fm is not None:
        s.mag_mode = L.FUNCTION_MODE_NAMES[fm]
        s.projection = Projection.DIFFRACTION if fm == L.FM_DIFF else Projection.IMAGING
    elif mode in (1, 2):
        s.projection = Projection(mode)

    if (v := _f(values, "PROJ_Magnification")) is not None and v > 0:
        s.magnification = v
    if (v := _f(values, "PROJ_CameraLength")) is not None and v > 0:
        s.camera_length_mm = v * (10.0 if camera_length_unit == "cm" else 1.0)
    if (v := _f(values, "PROJ_Defocus")) is not None:
        s.defocus_um = v  # DE-Server reads PROJ_Defocus as micrometres

    if (pos := _parse_semicolon_floats(values, "STAGE_PositionFull", 5)) is not None:
        s.stage = StagePosition(*pos)
    if (p := _parse_semicolon_floats(values, "ILLUM_BeamShift", 2)) is not None:
        s.beam_shift_um = Vec2(*p)
    if (p := _parse_semicolon_floats(values, "ILLUM_ImageShift", 2)) is not None:
        s.image_shift_um = Vec2(*p)
    if (p := _parse_semicolon_floats(values, "ILLUM_BeamTilt", 2)) is not None:
        s.beam_tilt_mrad = Vec2(p[0] * BEAM_TILT_MRAD_PER_UNIT, p[1] * BEAM_TILT_MRAD_PER_UNIT)
    if (p := _parse_semicolon_floats(values, "PROJ_ObjStig", 2)) is not None:
        s.objective_stig = Vec2(*p)
    if (p := _parse_semicolon_floats(values, "ILLUM_CondenserStig", 2)) is not None:
        s.condenser_stig = Vec2(*p)
    if (p := _parse_semicolon_floats(values, "PROJ_DiffractionShift", 2)) is not None:
        s.diffraction_shift_mrad = Vec2(p[0] * DIFF_SHIFT_MRAD_PER_UNIT, p[1] * DIFF_SHIFT_MRAD_PER_UNIT)

    if (v := _i(values, "ILLUM_SpotSize")) is not None and v > 0:
        s.spot_size = v
    if (v := _i(values, "ILLUM_ProbeMode")) is not None:
        info["raw_probe_mode"] = v
        pm = probe_mode_from_raw(vendor, v)
        if pm != ProbeMode.UNKNOWN:
            s.probe_mode = pm
    conv = _f(values, "ILLUM_ConvergenceAngle")
    info["convergence_explicit"] = conv if conv is not None and conv > 0 else None
    if conv is not None and conv > 0:
        s.convergence_semi_angle_mrad = conv
    if (v := _i(values, "ILLUM_AlphaSelector")) is not None and v >= 0:
        s.alpha_selector = v
    if (v := _i(values, "ILLUM_CondenserApertureIndex")) is not None and v >= 0:
        s.condenser_aperture_index = v
    if (v := _i(values, "OP_Status")) is not None:
        info["op_status"] = v
        s.op_status = 1 if v == 1 else 0
    if (v := _i(values, "STAGE_Mode")) is not None:
        info["stage_mode"] = v
    return s, info


def _apply_corrector(state: MicroscopeState, side: str, items: list) -> None:
    """Fill ``state``'s corrector fields from a real channel's last measurement."""
    from .corrector import ceos_to_nm, operator_c1a1

    if side == "none":
        state.corrector = "none"
        state.probe_aberrations = {}
        state.image_aberrations = {}
        return
    state.corrector = side
    if not items:
        return
    ab = ceos_to_nm(items)
    c1, a1 = operator_c1a1(state, side)
    if "C1" in ab:
        ab["C1"] -= c1
    if "A1" in ab:
        ab["A1"] -= a1
    setattr(state, f"{side}_aberrations", {k: v for k, v in ab.items() if v != 0})


class MirrorColumn(Column):
    """A :class:`Column` that follows a DE-TEM-Channel over SOAP.

    Parameters
    ----------
    host, port
        The channel (a real one, the Dummy instrument, or another twin's
        :class:`~de_twin.faces.temchannel_soap.TemChannelServer`).
    period_s
        Poll period. The first poll happens in the constructor (errors are
        logged, not raised, unless ``strict=True``).
    client
        An existing :class:`SoapTemChannelClient` (overrides host/port).
    poll_intensity
        Also read ``getProperty("Intensity")`` each cycle (not in the batch).
    corrector_every
        Read the channel's aberration corrector every N polls (0 = never):
        ``getCorrectorPresent``, ``getCorrectorInfo`` (mode STEM -> probe side,
        TEM -> image side) and ``getAberrations`` (the corrector's *last
        measurement*, metres). The measured C1 / A1 include the operator's focus and
        stigmator, so those are subtracted (the optics adds them back) before the
        set lands in ``state.probe_aberrations`` / ``image_aberrations``; until a
        measurement exists the side's set is left as it was. A channel without the
        corrector operations is asked once.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5002,
        *,
        clock=None,
        period_s: float = 0.5,
        client: Optional[SoapTemChannelClient] = None,
        timeout: float = 5.0,
        start: bool = True,
        strict: bool = False,
        poll_intensity: bool = True,
        camera_length_unit: str = "cm",
        corrector_every: int = 4,
    ):
        self.corrector_every = int(corrector_every)
        self.corrector_info: dict = {}
        self._corrector_supported: Optional[bool] = None
        super().__init__(clock=clock, stage_speed_um_s=float("inf"), tilt_speed_deg_s=float("inf"),
                         stage_limits=(1e9,) * 5)
        self.client = client or SoapTemChannelClient(host, port, timeout=timeout)
        self.host, self.port = self.client.host, self.client.port
        self.period_s = float(period_s)
        self.poll_intensity = poll_intensity
        self.camera_length_unit = camera_length_unit
        self.connected = False
        self.last_error: Optional[str] = None
        self.polls = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._supported: Optional[list[str]] = None
        try:
            self.poll()
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            log.warning("MirrorColumn: first poll of %s:%s failed: %s", self.host, self.port, exc)
        if start:
            self.start()

    @property
    def real(self) -> bool:
        return True

    # ------------------------------------------------------------ polling
    def poll(self) -> MicroscopeState:
        """One synchronous refresh from the channel. Returns the new state."""
        keys = list(BATCH_KEYS)
        try:
            fields = self.client.execute_batch(keys)
        except SoapError:
            # Older channels: DE-Server falls back to its first ten keys.
            keys = keys[:24]
            fields = self.client.execute_batch(keys)
        values = dict(zip(keys, fields))
        if self.poll_intensity:
            try:
                val, _ = self.client.get_property("Intensity")
                values["_Intensity"] = val
            except SoapError:
                pass
        corr = None
        if (self.corrector_every > 0 and self._corrector_supported is not False
                and self.polls % self.corrector_every == 0):
            corr = self.poll_corrector()
        with self.lock:
            prev = self._s.copy()
            prev.stage = StagePosition(*self._stage_now)
            state, info = state_from_batch(values, prev, camera_length_unit=self.camera_length_unit)
            if values.get("_Intensity"):
                try:
                    state.intensity = float(values["_Intensity"])
                except ValueError:
                    pass
            if corr is not None:
                _apply_corrector(state, *corr)
            extra = {}
            if "stage_mode" in info:
                extra["StageMode"] = info["stage_mode"]
            if "max_ht_v" in info:
                extra["MaxHighTension"] = info["max_ht_v"]
            if "filament" in info:
                extra["Filament"] = info["filament"]
            op = info.get("op_status")
            changed = self._apply_state(state, op_state=op, extra=extra,
                                        convergence_explicit=info.get("convergence_explicit"))
            self._remote_op = op
        self.connected = True
        self.last_error = None
        self.polls += 1
        for name in changed:
            try:
                self._notify(name, self.get(name))
            except KeyError:
                pass
        return self.state()

    def poll_corrector(self) -> Optional[tuple[str, list]]:
        """Read the channel's corrector: ``(side, [(name, (x_m, y_m)), ...])``,
        ``("none", [])`` when it has none, None when the channel lacks the operations."""
        try:
            present = self.client.corrector_present()
        except SoapError:
            if self._corrector_supported is None:
                self._corrector_supported = False
                log.info("MirrorColumn: channel has no corrector operations")
            return None
        self._corrector_supported = True
        if not present:
            self.corrector_info = {}
            return "none", []
        try:
            info = self.client.corrector_info()
            items = self.client.aberrations()
        except SoapError as exc:
            log.debug("corrector read failed: %s", exc)
            return None
        self.corrector_info = info
        mode = info.get("mode", "").upper()
        ctype = info.get("correctorType", "").upper()
        side = "image" if mode == "TEM" or (not mode and ctype == "CETCOR") else "probe"
        return side, items

    def state(self) -> MicroscopeState:
        s = super().state()
        op = getattr(self, "_remote_op", None)
        s.op_status = 1 if op == 1 else 0
        return s

    def op_state(self) -> int:
        op = getattr(self, "_remote_op", None)
        return int(op) if op is not None else super().op_state()

    def _loop(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                self.poll()
            except Exception as exc:  # noqa: BLE001
                self.connected = False
                if str(exc) != self.last_error:
                    log.warning("MirrorColumn poll failed: %s", exc)
                self.last_error = str(exc)

    def start(self) -> "MirrorColumn":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="MirrorColumn-poll", daemon=True)
            self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.period_s + 1)
            self._thread = None

    # ------------------------------------------------------------ writes
    def set(self, name: str, value: Any) -> None:
        """Forward a write to the channel, then refresh. Raises ColumnRefused on ``success=0``."""
        self._guard(name)
        if name in ("InstrumentType", "HT", "HTState", "Filament", "HighTension", "MagString",
                    "FunctionMode", "ApertureKind", "ApertureSize", "ConvergenceAngle"):
            if name == "FunctionMode":
                name = "SubMode"
            else:
                raise ColumnRefused(f"{name} cannot be written through DE-TEM-Channel")
        if name in ("BeamTilt", "DiffractionShift", "ImageShift", "BeamShift", "ObjectiveStig",
                    "CondenserStig", "GunShift", "GunTilt"):
            if isinstance(value, dict):
                value = (value["x"], value["y"])
            value = tuple(float(v) for v in value)
        if name == "ProbeMode" and not isinstance(value, (int, float)) and not str(value).lstrip("-").isdigit():
            from .column import probe_mode_to_raw
            pm = value if isinstance(value, ProbeMode) else ProbeMode[str(value).upper()]
            value = probe_mode_to_raw(self.instrument_type, pm)
        if name == "StagePosition":
            if not isinstance(value, dict):
                value = dict(zip("xyzab", value))
            value = {{"alpha": "a", "beta": "b"}.get(k, k): v for k, v in value.items() if v is not None}
        try:
            ok = self.client.set(name, value)
        except SoapError as exc:
            raise ColumnRefused(f"{name}: {exc}") from None
        if not ok:
            raise ColumnRefused(f"DE-TEM-Channel refused {name}={value!r}")
        try:
            self.poll()
        except Exception as exc:  # noqa: BLE001
            log.debug("post-set poll failed: %s", exc)
        try:
            readback = self.get(name)
        except KeyError:
            readback = value
        self._notify(name, readback)

    def stop(self) -> None:
        try:
            self.client.stop()
        finally:
            try:
                self.poll()
            except Exception:  # noqa: BLE001
                pass

    def wait_idle(self, timeout: Optional[float] = 30.0) -> bool:
        """Poll until the channel's OP_Status is no longer Running (wall-clock timeout)."""
        import time as _time

        deadline = None if timeout is None else _time.monotonic() + timeout
        while True:
            try:
                self.poll()
            except Exception:  # noqa: BLE001
                pass
            if getattr(self, "_remote_op", None) != 1:
                return True
            if deadline is not None and _time.monotonic() > deadline:
                return False
            _time.sleep(min(self.period_s, 0.1))
