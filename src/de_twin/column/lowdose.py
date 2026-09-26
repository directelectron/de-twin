"""Low dose: the column's Search / View / Focus / Record areas.

Each area stores its own magnification, spot size, intensity, defocus offset (from the
Record focus), image shift and beam shift. Switching area stores the live settings into the
area left, as a real column's low-dose mode does (an operator adjusts an area while in it),
and applies the one entered. Defocus is kept relative to Record's, so refocusing in Record
carries View's large offset along.

On a realistic column (`OpticsConfig.realistic`) the areas do not line up by themselves:
each magnification has its own image rotation, true pixel size, image-shift matrix and image
offset, and View's high defocus changes its scale and rotation again. That is what SerialEM's
low-dose calibration (aligning View to Record, setting the area offsets) measures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

AREAS = ("Search", "View", "Focus", "Record")

#: Defaults when low dose is switched on: View this much below Record's magnification and
#: this far underfocus; Search further out; Focus image-shifted along x (the tilt axis) by
#: this many image-shift units.
VIEW_MAG_FACTOR = 8.0
VIEW_DEFOCUS_UM = -200.0
SEARCH_MAG_FACTOR = 50.0
SEARCH_DEFOCUS_UM = -200.0
FOCUS_OFFSET_UNITS = 1.5


@dataclass
class LowDoseArea:
    magnification: float
    spot_size: int
    intensity: float
    defocus_offset_um: float = 0.0
    image_shift: tuple[float, float] = (0.0, 0.0)
    beam_shift: tuple[float, float] = (0.0, 0.0)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["image_shift"] = list(self.image_shift)
        d["beam_shift"] = list(self.beam_shift)
        return d


@dataclass
class LowDose:
    """The column's low-dose state; `column.Column` owns one and drives it."""

    enabled: bool = False
    area: str = "Record"
    areas: dict = field(default_factory=dict)
    base_focus_um: float = 0.0  # Record's focus

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _rung(ladder, mag: float) -> float:
        return float(min(ladder, key=lambda m: abs(m - mag)))

    def seed_from(self, col, ladder) -> None:
        """Areas from the column's current state (Record = now)."""
        s = col._s
        rec = area_from_state(s)
        self.base_focus_um = float(s.defocus_um)
        focus = area_from_state(s)
        focus.image_shift = (s.image_shift_um.x + FOCUS_OFFSET_UNITS, s.image_shift_um.y)
        view = area_from_state(s)
        view.magnification = self._rung(ladder, s.magnification / VIEW_MAG_FACTOR)
        view.defocus_offset_um = VIEW_DEFOCUS_UM
        search = area_from_state(s)
        search.magnification = self._rung(ladder, s.magnification / SEARCH_MAG_FACTOR)
        search.defocus_offset_um = SEARCH_DEFOCUS_UM
        self.areas = {"Search": search, "View": view, "Focus": focus, "Record": rec}
        self.area = "Record"

    def store(self, col) -> None:
        """The live settings into the current area (in diffraction the area keeps its
        imaging magnification)."""
        from . import ladders as L

        s = col._s
        a = self.areas[self.area]
        if col._fm != L.FM_DIFF:
            a.magnification = float(s.magnification)
        a.spot_size = int(s.spot_size)
        a.intensity = float(s.intensity)
        a.image_shift = (float(s.image_shift_um.x), float(s.image_shift_um.y))
        a.beam_shift = (float(s.beam_shift_um.x), float(s.beam_shift_um.y))
        if self.area == "Record":
            self.base_focus_um = float(s.defocus_um)
            a.defocus_offset_um = 0.0
        else:
            a.defocus_offset_um = float(s.defocus_um) - self.base_focus_um

    def apply(self, col, name: str, setters) -> None:
        """Enter area *name* (the live settings are NOT stored: call `store` first). All or
        nothing: if a setting is refused the column is left as it was."""
        from . import ladders as L

        a = self.areas[name]
        saved = (col._s.copy(), col._fm, getattr(col, "_last_imaging_fm", None))
        try:
            if col._fm != L.FM_DIFF:  # diffraction has no magnification to set
                setters["Magnification"](col, a.magnification)
            setters["SpotSize"](col, a.spot_size)
            setters["Intensity"](col, a.intensity)
            setters["Defocus"](col, self.base_focus_um + (0.0 if name == "Record" else a.defocus_offset_um))
            setters["ImageShift"](col, a.image_shift)
            setters["BeamShift"](col, a.beam_shift)
        except Exception:
            col._s, col._fm = saved[0], saved[1]
            if saved[2] is not None:
                col._last_imaging_fm = saved[2]
            raise
        self.area = name


def area_from_state(s) -> LowDoseArea:
    """An area holding the column's current settings (no defocus offset)."""
    return LowDoseArea(float(s.magnification), int(s.spot_size), float(s.intensity), 0.0,
                       (float(s.image_shift_um.x), float(s.image_shift_um.y)),
                       (float(s.beam_shift_um.x), float(s.beam_shift_um.y)))


def normalise_area(name) -> Optional[str]:
    n = str(name).strip().lower()
    for a in AREAS:
        if a.lower() == n or (a == "Record" and n in ("exposure", "record")):
            return a
    return None
