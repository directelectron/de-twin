"""ImpulseFollower: follow a DENSsolutions holder (or DENS's own simulator) via impulsePy.

Wraps ``impulsePy`` exactly like de_autopilot's ``ImpulseHeater``: reads use
``heat.data.getLastData()`` (never the blocking ``getNewData``),
``waitForControl`` is only called from :meth:`wait_for_control` on a helper
thread, ``set``/``startRamp``/``stopRamp`` drive the ``heat`` (and, when present,
``bias``) module. ``impulsePy`` is imported lazily, so the twin runs without it.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from ..state import HolderState
from .base import DEFAULT_RAMP_RATE, ChannelUnsupported, Reading, clock_fn

log = logging.getLogger(__name__)


def _field(data: Any, *names: str) -> float | None:
    """First of ``names`` the sample carries (attribute or key), as a float."""
    for name in names:
        value = getattr(data, name, None)
        if value is None and isinstance(data, dict):
            value = data.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


#: How long `import_impulse` waits. impulsePy connects to Impulse when it is imported and
#: blocks for good when Impulse is not running.
IMPORT_TIMEOUT_S = 5.0
_IMPORT: dict = {}
_IMPORT_LOCK = threading.Lock()


def import_impulse(timeout_s: float | None = None):
    """``impulsePy``, imported on a thread of its own: a TimeoutError rather than a hang
    when Impulse is not running. An import still stuck from an earlier call fails at once
    (a second import would only queue behind it on the import lock)."""
    import importlib
    import sys

    if "impulsePy" in sys.modules:  # imported (or blocked: None raises ImportError)
        return importlib.import_module("impulsePy")
    with _IMPORT_LOCK:
        pending = _IMPORT.get("pending")
        if pending is None:
            done, result = threading.Event(), {}

            def run():
                try:
                    result["module"] = importlib.import_module("impulsePy")
                except BaseException as e:
                    result["error"] = e
                finally:
                    done.set()
                    with _IMPORT_LOCK:
                        _IMPORT.pop("pending", None)

            pending = _IMPORT["pending"] = (done, result)
            threading.Thread(target=run, name="impulsePy-import", daemon=True).start()
            wait = IMPORT_TIMEOUT_S if timeout_s is None else float(timeout_s)
        else:
            wait = 0.0
    done, result = pending
    if not done.wait(wait):
        raise TimeoutError(f"importing impulsePy did not finish in {wait:g} s (is Impulse running?)")
    if "error" in result:
        raise result["error"]
    return result["module"]


class ImpulseFollower:
    """A real DENS holder through ``impulsePy`` (or any object with its API)."""

    real = True
    reason = ""

    def __init__(self, impulse: Any = None, *, clock=None):
        if impulse is None:
            impulse = import_impulse()
        self._impulse = impulse
        self._now = clock_fn(clock)
        self._lock = threading.RLock()
        self._target = float("nan")
        self._bias_target: float | None = None
        self.channels = ("temperature",) + (("bias",) if getattr(impulse, "bias", None) is not None else ())

    def _channel(self, name: str):
        if name == "temperature":
            return self._impulse.heat
        if name == "bias":
            bias = getattr(self._impulse, "bias", None)
            if bias is None:
                raise ChannelUnsupported("This holder has no bias channel (impulsePy exposes no `bias`).")
            return bias
        raise ChannelUnsupported(f"Unknown stimulus {name!r}")

    @property
    def busy(self) -> bool:
        try:
            return bool(self._impulse.heat.busy)
        except Exception as e:
            log.debug("heat.busy failed: %s", e)
            return False

    def read(self) -> Reading:
        with self._lock:
            data = self._impulse.heat.data.getLastData()
        measured = _field(data, "temperature", "measuredTemperature", "temp")
        target = _field(data, "targetTemperature", "setpoint", "target")
        voltage = _field(data, "voltage", "measuredVoltage")
        current = _field(data, "current", "measuredCurrent")
        source = "heater"
        if "bias" in self.channels:
            try:
                with self._lock:
                    bias_data = self._impulse.bias.data.getLastData()
            except Exception as e:
                log.debug("bias.getLastData failed: %s", e)
            else:
                bv = _field(bias_data, "voltage", "measuredVoltage", "bias")
                ba = _field(bias_data, "current", "measuredCurrent")
                if bv is not None or ba is not None:
                    voltage, current, source = bv, ba, "bias"
        return Reading(
            measured=float(measured if measured is not None else float("nan")),
            target=float(target if target is not None else self._target),
            t=time.time(),
            power=_field(data, "power", "heatingPower"),
            resistance=_field(data, "resistance"),
            voltage=voltage,
            current=current,
            source=source,
            bias_target=self._bias_target,
        )

    def set(self, value: float, *, channel: str = "temperature") -> None:
        with self._lock:
            self._channel(channel).set(float(value))
            if channel == "bias":
                self._bias_target = float(value)
            else:
                self._target = float(value)

    def ramp(
        self,
        value: float,
        *,
        rate_c_per_s: float | None = None,
        seconds: float | None = None,
        channel: str = "temperature",
    ) -> None:
        with self._lock:
            module = self._channel(channel)
            if seconds and seconds > 0 and not rate_c_per_s:
                module.startRamp(float(value), "rampTime", float(seconds))
            else:
                module.startRamp(float(value), "rampRate", float(rate_c_per_s or DEFAULT_RAMP_RATE))
            if channel == "bias":
                self._bias_target = float(value)
            else:
                self._target = float(value)

    def stop(self) -> None:
        with self._lock:
            try:
                self._impulse.heat.stopRamp()
            except Exception as e:
                log.warning("stopRamp failed: %s", e)

    def flag(self, text: str) -> None:
        try:
            self._impulse.heat.data.setFlag(str(text))
        except Exception as e:
            log.debug("setFlag(%r) failed: %s", text, e)

    def describe(self) -> str:
        return "DENSsolutions holder (Impulse)"

    def wait_for_control(self, timeout_s: float = 30.0) -> bool:
        done = threading.Event()

        def _wait() -> None:
            try:
                self._impulse.waitForControl()
            except Exception as e:
                log.warning("waitForControl failed: %s", e)
            finally:
                done.set()

        threading.Thread(target=_wait, daemon=True, name="impulse-control").start()
        return done.wait(timeout_s)

    def state(self, t: float | None = None) -> HolderState:
        """The holder's latest sample as a HolderState (``t`` only stamps it)."""
        r = self.read()
        if t is None:
            t = self._now() if self._now else r.t
        biasing = "bias" in self.channels
        return HolderState(
            kind="heating_biasing" if biasing else "heating",
            temperature_c=r.measured,
            target_c=r.target,
            power_w=r.power or 0.0,
            resistance_ohm=r.resistance or 0.0,
            bias_v=(r.voltage or 0.0) if r.source == "bias" else (self._bias_target or 0.0),
            current_a=r.current or 0.0,
            t_s=float(t),
        )

    def close(self) -> None:
        try:
            self._impulse.disconnect()
        except Exception as e:
            log.debug("impulse disconnect failed: %s", e)
