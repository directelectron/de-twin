"""Shared holder types: autopilot-compatible ``Reading`` and ``NoHolder``.

The duck type every holder follows is de_autopilot's ``de_automate.impulse``
heater (``SimHeater`` / ``ImpulseHeater``)::

    real: bool            reason: str            channels: tuple[str, ...]
    busy (property)       read() -> Reading      set(value, *, channel="temperature")
    ramp(value, *, rate_c_per_s=None, seconds=None, channel="temperature")
    stop()                flag(text)             describe() -> str
    wait_for_control(timeout_s) -> bool          close()

plus, for the twin, ``state(t=None) -> de_twin.state.HolderState``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..state import HolderState

log = logging.getLogger(__name__)

#: How close counts as arrived, degC (autopilot DEFAULT_TOLERANCE_C).
DEFAULT_TOLERANCE_C = 2.0
#: Ramp rate when a caller names a target and no rate, degC/s (autopilot).
DEFAULT_RAMP_RATE = 2.0
AMBIENT_C = 25.0


@dataclass
class Reading:
    """One sample from the holder (field-for-field de_automate.impulse.Reading).

    Everything SI: degC, W, ohm, V, A. ``t`` is seconds (the holder's clock;
    epoch seconds when no clock was injected, like autopilot).
    """

    measured: float
    target: float
    t: float
    power: float | None = None
    resistance: float | None = None
    voltage: float | None = None
    current: float | None = None
    source: str = "heater"
    bias_target: float | None = None

    @property
    def settled(self) -> bool:
        return abs(self.measured - self.target) <= DEFAULT_TOLERANCE_C

    def to_json(self) -> dict:
        return {
            "measured": round(self.measured, 1),
            "target": round(self.target, 1),
            "power": None if self.power is None else round(self.power, 3),
            "voltage": None if self.voltage is None else round(self.voltage, 4),
            "current": None if self.current is None else round(self.current, 9),
            "source": self.source,
            "settled": self.settled,
        }


class HeaterUnavailable(RuntimeError):
    """Asked to heat, with no holder to heat with."""


class ChannelUnsupported(RuntimeError):
    """Asked for a stimulus this holder does not have (e.g. biasing a heating-only chip)."""


def clock_fn(clock) -> Callable[[], float] | None:
    if clock is None:
        return None
    now = getattr(clock, "now", None)
    if callable(now):
        return now
    if callable(clock):
        return clock
    raise TypeError("clock must be callable or have .now()")


class NoHolder:
    """No in-situ holder: a standard grid holder at room temperature."""

    real = False
    channels: tuple[str, ...] = ()
    busy = False

    def __init__(self, reason: str = "No in-situ holder fitted.", clock=None):
        self.reason = reason
        self._now = clock_fn(clock)

    def _t(self) -> float:
        return self._now() if self._now else time.time()

    def read(self) -> Reading:
        return Reading(measured=AMBIENT_C, target=AMBIENT_C, t=self._t())

    def set(self, value: float, *, channel: str = "temperature") -> None:
        raise HeaterUnavailable("No in-situ holder: nothing to heat or bias.")

    def ramp(self, value: float, *, rate_c_per_s=None, seconds=None, channel: str = "temperature") -> None:
        raise HeaterUnavailable("No in-situ holder: nothing to heat or bias.")

    def stop(self) -> None:
        pass

    def flag(self, text: str) -> None:
        log.debug("holder flag (no holder): %s", text)

    def describe(self) -> str:
        return "No in-situ holder"

    def wait_for_control(self, timeout_s: float = 30.0) -> bool:
        return True

    def state(self, t: float | None = None) -> HolderState:
        return HolderState(kind="none", t_s=self._t() if t is None else float(t))

    def close(self) -> None:
        pass
