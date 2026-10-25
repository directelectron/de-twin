"""ColumnAdapter: the twin's column behind de_microscope/de_autopilot's column interface.

de_microscope ``column.SimColumn``/``RealColumn`` and de_autopilot
``column.SimColumn``/``RealColumn`` answer the same handful of methods
(``real``, ``host``, ``reason``, ``lock``, ``stage_xy``, ``move_stage``, ``tilt``,
``tilt_to``, ``get``, ``set``, ``values``, ``stop``, ``close``, ``on_move``,
``on_set``). This adapter gives any :class:`~de_twin.column.Column` (or
:class:`MirrorColumn`) that shape, so autopilot / Ground Crew can drive the
twin's column in-process and every frame the twin renders follows.

Differences from a bare Column, all taken from those modules:
* pairs are returned as lists, like SimColumn
* HighTension / HTState / Filament are refused (``FORBIDDEN``)
* ``move_stage`` / ``tilt_to`` refuse (not clamp) out-of-travel targets and
  block until the stage has arrived (on the column's clock)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .column import Column
from .errors import ColumnRefused
from .vendor import is_tfs

log = logging.getLogger(__name__)

FORBIDDEN = frozenset({"HighTension", "HTState", "Filament", "HT"})

#: What ``values()`` reports (de_autopilot REAL_PROPS + a few).
VALUE_PROPS = (
    "HighTension",
    "SpotSize",
    "Intensity",
    "BeamBlank",
    "ColumnValvesOpen",
    "Magnification",
    "CameraLength",
    "Defocus",
    "DiffractionFocus",
    "ProjectionMode",
    "SubMode",
    "ScreenPosition",
    "ScreenCurrent",
    "ProbeMode",
    "TemStemMode",
    "ImageShift",
    "BeamShift",
    "DiffractionShift",
    "BeamTilt",
    "ObjectiveStig",
    "CondenserStig",
    "GunShift",
    "GunTilt",
    "ScanRotation",
    "ExposureTime",
    "StageX",
    "StageY",
    "StageZ",
    "StageA",
    "StageB",
    "InstrumentType",
    "WorkingDistance",
    "SampleTilt",
    "ConvergenceAngle",
    "AlphaSelector",
)

TILT_LIMIT_DEG = 70.0

SCREEN_TFS = {"unknown": 1, "up": 2, "down": 3}
SCREEN_DEFAULT = {"up": 0, "down": 1}


def screen_positions(instrument_type: object, observed: object = None) -> dict[str, int]:
    """de_microscope/autopilot ``screen_positions``: the vendor's ScreenPosition table."""
    table = dict(SCREEN_TFS) if is_tfs(str(instrument_type or "")) else dict(SCREEN_DEFAULT)
    try:
        value = int(observed)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return table
    if value in (2, 3):
        return dict(SCREEN_TFS)
    if value == 0:
        return dict(SCREEN_DEFAULT)
    return table


def _listify(v: Any) -> Any:
    return list(v) if isinstance(v, tuple) else v


class ColumnAdapter:
    """A de_microscope/autopilot ``SimColumn``-shaped view of a twin Column."""

    def __init__(self, column: Column, *, reason: str = "", host: str = ""):
        self.column = column
        self.lock = column.lock
        self.real = bool(getattr(column, "real", False))
        self.host = host or (f"{column.host}:{column.port}" if hasattr(column, "host") else "")
        self.reason = reason or ("" if self.real else "de_twin digital-twin column")
        self._move_observers: list[Callable[[float, float], None]] = []
        self._set_observers: list[Callable[[str, Any], None]] = []

    # ---------------------------------------------------------- observers
    def on_move(self, callback: Callable[[float, float], None]) -> None:
        self._move_observers.append(callback)

    def on_set(self, callback: Callable[[str, Any], None]) -> None:
        self._set_observers.append(callback)

    def _notify_moved(self, x: float, y: float) -> None:
        for cb in list(self._move_observers):
            try:
                cb(float(x), float(y))
            except Exception:
                log.exception("stage-move observer failed")

    def _notify_set(self, prop: str, value: Any) -> None:
        for cb in list(self._set_observers):
            try:
                cb(str(prop), value)
            except Exception:
                log.exception("column-set observer failed")

    # --------------------------------------------------------------- stage
    def stage_xy(self) -> tuple[float, float]:
        x, y, *_ = self.column.stage_position()
        return float(x), float(y)

    def move_stage(self, x: float, y: float) -> None:
        x, y = float(x), float(y)
        lx, ly = self.column.stage_limits[:2]
        if abs(x) > lx or abs(y) > ly:
            raise ColumnRefused(f"({x:.0f}, {y:.0f}) is outside the ±{lx:.0f} µm travel")
        try:
            self.column.set_stage(x=x, y=y)
        except ColumnRefused:
            raise
        except Exception as e:
            raise ColumnRefused(str(e)) from None
        self.column.wait_idle()
        self._notify_moved(x, y)

    def tilt(self) -> float:
        return float(self.column.stage_position()[3])

    def tilt_to(self, alpha_deg: float) -> None:
        alpha = float(alpha_deg)
        if abs(alpha) > TILT_LIMIT_DEG:
            raise ColumnRefused(f"tilt {alpha:.1f}° is outside ±{TILT_LIMIT_DEG:.0f}°")
        self.column.set_stage(alpha=alpha)
        self.column.wait_idle()

    # ----------------------------------------------------------- properties
    def get(self, prop: str) -> Any:
        try:
            return _listify(self.column.get(prop))
        except KeyError:
            return None

    def set(self, prop: str, value: Any) -> None:
        if prop in FORBIDDEN:
            raise ColumnRefused(f"{prop} is not controllable from the UI")
        if prop in ("StageX", "StageY"):
            x, y = self.stage_xy()
            self.move_stage(value if prop == "StageX" else x, value if prop == "StageY" else y)
            readback = float(value)
        elif prop == "StageA":
            self.tilt_to(value)
            readback = float(value)
        else:
            try:
                self.column.set(prop, tuple(value) if isinstance(value, list) else value)
            except KeyError as e:
                raise ColumnRefused(str(e)) from None
            readback = self.get(prop)
        self._notify_set(prop, readback)

    def values(self) -> dict[str, Any]:
        with self.lock:
            out = {}
            for name in VALUE_PROPS:
                try:
                    out[name] = _listify(self.column.get(name))
                except KeyError:
                    pass
            return out

    def screen_positions(self, observed: int | None = None) -> dict[str, int]:
        if observed is None:
            observed = self.get("ScreenPosition")
        return screen_positions(self.column.instrument_type, observed)

    @property
    def vendor(self) -> str:
        return str(self.column.instrument_type).upper()

    def stop(self) -> None:
        self.column.stop()

    def close(self) -> None:
        """The adapter does not own the column; nothing to release."""
