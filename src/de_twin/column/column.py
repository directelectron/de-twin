"""The simulated TEM column.

:class:`Column` owns a :class:`~de_twin.state.MicroscopeState` and behaves like
DE-TEM-Channel's in-memory ``DummyEMInstrument`` (``instruments/tem/dummy``):

* JEOL function modes MAG1/MAG2/LowMAG/SAMAG/DIFF crossed with TEM/STEM, and
  only the seven (Mode, SubMode, TemStemMode) triples DE-Server's ``mag.yaml``
  has projects for (entering STEM from MAG1 lands on LowMAG/HR_STEM, MAG1 is
  refused in STEM, ...).
* Magnification and camera length snap to the detent ladders
  (:mod:`de_twin.column.ladders`); a TEM magnification outside the active
  imaging ladder crosses between LowMAG and MAG1.
* The stage slews to its target at 100 um/s per linear axis and 10 deg/s per
  tilt, interpolated from the injected clock on every read, so a virtual clock
  can fast-forward a move. ``op_status`` is 1 while any axis is moving.
* The convergence semi-angle is derived from (probe mode, alpha selector) in TEM and
  from the STEM table (alpha selector: 5/10/15/22/30 mrad) in STEM, unless it has been
  set explicitly. It is what ``ILLUM_ConvergenceAngle`` reports on every face.
* ``INSTRUMENT_Type`` is ``"JEOL"`` by default (configurable ``"FEI"``). The
  vendor only changes how raw values are *reported* (``PROJ_SubMode``,
  ``ILLUM_ProbeMode``, ``ScreenPosition`` enumerations); the state is always
  vendor-normalised.

Safety as in de_microscope: high tension and filament writes are refused unless
the column was built with ``allow_ht=True``.

Property names for :meth:`Column.get` / :meth:`Column.set` are the
DE-TEM-Channel / de_microscope names and units:

======================  ====================================================
Magnification           x (snaps to the active ladder)
MagnificationIndex      write only, index into the active ladder
CameraLength            **cm** (DE-TEM-Channel unit; state keeps mm)
Defocus                 um
SpotSize                1..5
ProbeMode               raw vendor value (JEOL 0 TEM 1 EDS 2 NBD 3 CBD;
                        FEI 0 nanoprobe 1 microprobe); a ProbeMode or its
                        name is accepted on set
ConvergenceAngle        mrad; set <= 0 to go back to the derived value
AlphaSelector           0..7
CondenserApertureIndex  0..4
Intensity               C2, echoed unclamped (like the Dummy)
ImageShift, BeamShift   (x, y) um
BeamTilt                (x, y) normalised units, x BEAM_TILT_MRAD_PER_UNIT mrad
DiffractionShift        (x, y) normalised units, x DIFF_SHIFT_MRAD_PER_UNIT mrad
Precession              [twin] bool: drive the beam-tilt coils round a cone
PrecessionAngle         [twin] cone half-angle, mrad
PrecessionFrequency     [twin] Hz
PrecessionDescan        [twin] bool: descan brings the pattern back (default on)
ObjectiveStig,
CondenserStig           (x, y) normalised units
StagePosition           dict x, y, z (um), a, b (deg); set any subset
StageX/Y/Z, StageA/B    one axis
TemStemMode             0 TEM, 1 STEM
ProjectionMode          1 imaging, 2 diffraction
SubMode / FunctionMode  raw vendor function mode (JEOL EOS 0..4, TFS 1..6)
BeamBlank               0/1
ScreenPosition          JEOL 0 up / 1 down; FEI 2 up / 3 down
ColumnValvesOpen        0/1
HighTension (HT)        volts (write refused unless allow_ht)
InstrumentType          "JEOL" | "FEI" | ...
======================  ====================================================

Aberration corrector: ``Column(corrector="probe" | "image" | "both", seed=...)`` fits
a CEOS-like corrector (:mod:`de_twin.column.corrector`, ``column.corrector``). Its
residual aberrations fill ``state().probe_aberrations`` / ``image_aberrations`` and
``state().corrector``; an HT change detunes it, a probe-mode / function-mode /
TEM-STEM change kicks C1/A1/B2/A2. The operator's defocus and stigmators stay
separate fields (the optics adds them as C1 / A1).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable, Iterable, Optional

from ..state import MicroscopeState, ProbeMode, Projection, StagePosition, TemStem, Vec2
from . import ladders as L

log = logging.getLogger(__name__)

#: Beam tilt: DE-TEM-Channel's range is the normalised [-1, 1]; one unit is this many mrad.
BEAM_TILT_MRAD_PER_UNIT = 10.0
#: Diffraction shift: normalised +/-1 per axis; one unit is this many mrad.
DIFF_SHIFT_MRAD_PER_UNIT = 10.0

#: Never accepted unless the column is built with ``allow_ht=True`` (de_microscope policy).
FORBIDDEN = frozenset({"HighTension", "HT", "HTState", "Filament"})

STAGE_AXES = ("x", "y", "z", "a", "b")


class ColumnRefused(ValueError):
    """The column understood the request and refused it (the Dummy's ``false``)."""


# ---------------------------------------------------------------------------
# Vendor enumerations
# ---------------------------------------------------------------------------

def is_jeol(instrument_type: str) -> bool:
    return "JEOL" in str(instrument_type or "").upper()


def is_tfs(instrument_type: str) -> bool:
    v = str(instrument_type or "").upper()
    return any(t in v for t in ("FEI", "TFS", "THERMO"))


_JEOL_RAW = {ProbeMode.TEM: 0, ProbeMode.EDS: 1, ProbeMode.NBD: 2, ProbeMode.CBD: 3,
             ProbeMode.MICROPROBE: 0, ProbeMode.NANOPROBE: 2, ProbeMode.UNKNOWN: 0}
_FEI_RAW = {ProbeMode.NANOPROBE: 0, ProbeMode.MICROPROBE: 1, ProbeMode.TEM: 1,
            ProbeMode.EDS: 0, ProbeMode.NBD: 0, ProbeMode.CBD: 0, ProbeMode.UNKNOWN: 1}


def probe_mode_to_raw(instrument_type: str, mode: ProbeMode) -> int:
    """Vendor-normalised probe mode -> the raw value a vendor's channel reports."""
    table = _JEOL_RAW if is_jeol(instrument_type) or not is_tfs(instrument_type) else _FEI_RAW
    return table[ProbeMode(mode)]


def probe_mode_from_raw(instrument_type: str, raw: int) -> ProbeMode:
    """DE-Server ``NormalizeProbeMode`` (Camera.cpp)."""
    raw = int(raw)
    if is_jeol(instrument_type):
        return {0: ProbeMode.TEM, 1: ProbeMode.EDS, 2: ProbeMode.NBD, 3: ProbeMode.CBD}.get(
            raw, ProbeMode.UNKNOWN)
    return {0: ProbeMode.NANOPROBE, 1: ProbeMode.MICROPROBE}.get(raw, ProbeMode.UNKNOWN)


def jeol_row(instrument_type: str, mode: ProbeMode) -> int:
    """Row of the alpha/convergence table for a normalised probe mode."""
    return _JEOL_RAW[ProbeMode(mode)]


def screen_to_raw(instrument_type: str, position: int) -> int:
    """0 up / 1 down -> the vendor's enumeration (TFS: 2 up, 3 down)."""
    if is_tfs(instrument_type):
        return 3 if position else 2
    return 1 if position else 0


def screen_from_raw(instrument_type: str, raw: int) -> int:
    raw = int(raw)
    if is_tfs(instrument_type) or raw in (2, 3):
        # TFS: 1 = unknown/moving (treated as not-up: the beam may be intercepted)
        return 0 if raw in (0, 2) else 1
    if raw not in (0, 1):
        raise ColumnRefused(f"invalid screen position {raw} (0 = up, 1 = down)")
    return raw


def submode_to_raw(instrument_type: str, function_mode: int) -> int:
    if is_tfs(instrument_type):
        return L.FEI_SUBMODE_FROM_FM[function_mode]
    return int(function_mode)


def submode_from_raw(instrument_type: str, raw: int) -> int:
    raw = int(raw)
    if is_tfs(instrument_type):
        if raw not in L.FM_FROM_FEI_SUBMODE:
            raise ColumnRefused(f"unknown TFS projection submode {raw}")
        return L.FM_FROM_FEI_SUBMODE[raw]
    if not 0 <= raw <= 4:
        raise ColumnRefused(f"unknown JEOL function mode {raw} (0..4)")
    return raw


def function_mode_from_name(name: str) -> int:
    key = str(name or "").strip().upper()
    for i, n in enumerate(L.FUNCTION_MODE_NAMES):
        if n.upper() == key:
            return i
    aliases = {"LOWMAG": L.FM_LOWMAG, "LM": L.FM_LOWMAG, "SMAG": L.FM_SAMAG, "SA": L.FM_SAMAG,
               "M": L.FM_MAG1, "MH": L.FM_MAG2, "D": L.FM_DIFF, "LAD": L.FM_DIFF,
               "DIFFRACTION": L.FM_DIFF, "SADIFF": L.FM_DIFF}
    return aliases.get(key, L.FM_MAG1)


def _pair(value: Any, name: str) -> tuple[float, float]:
    try:
        if isinstance(value, dict):
            x, y = value["x"], value["y"]
        elif isinstance(value, Vec2):
            x, y = value.x, value.y
        else:
            x, y = value
        x, y = float(x), float(y)
    except (TypeError, ValueError, KeyError):
        raise ColumnRefused(f"{name} needs exactly two numeric coordinates") from None
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ColumnRefused(f"{name} coordinates must be finite")
    return x, y


def _clock_fn(clock) -> Callable[[], float]:
    if clock is None:
        t0 = time.perf_counter()
        return lambda: time.perf_counter() - t0
    now = getattr(clock, "now", None)
    if callable(now):
        return now
    if callable(clock):
        return clock
    raise TypeError("clock must be a callable returning seconds or have .now()")


# ---------------------------------------------------------------------------
# Column
# ---------------------------------------------------------------------------

class Column:
    """A simulated column. Thread-safe (``self.lock`` is an ``RLock``).

    Parameters
    ----------
    state
        Initial state (defaults: :class:`MicroscopeState` defaults, i.e. 200 kV,
        MAG1 20 kx, screen up so the camera sees the beam).
    clock
        Callable returning seconds, or an object with ``now()`` (``de_twin.clock.Clock``).
        Stage motion is interpolated on this clock; a virtual clock fast-forwards it.
    instrument_type
        Vendor string reported as ``INSTRUMENT_Type`` ("JEOL" default, or "FEI").
    allow_ht
        Permit high-tension / filament writes (refused by default).
    stage_speed_um_s, tilt_speed_deg_s, stage_limits
        Stage dynamics; ``stage_speed_um_s=float('inf')`` makes moves instantaneous.
    corrector
        ``None`` / ``"none"`` (default; or ``state.corrector``), ``"probe"``, ``"image"``
        or ``"both"``: fit a CEOS-like aberration corrector (``self.corrector``).
    seed
        Seeds the corrector's native aberrations, tune, drift and noise.
    corrector_options
        Extra keyword arguments for :class:`~de_twin.column.corrector.Corrector`
        (``tuned=False`` for a cold start, ``drift_rates={...}``, ``served="image"``).
    """

    def __init__(
        self,
        state: Optional[MicroscopeState] = None,
        *,
        clock=None,
        instrument_type: Optional[str] = None,
        allow_ht: bool = False,
        stage_speed_um_s: float = L.STAGE_SPEED_UM_PER_S,
        tilt_speed_deg_s: float = L.TILT_SPEED_DEG_PER_S,
        stage_limits: Iterable[float] = L.STAGE_LIMITS,
        corrector: Optional[str] = None,
        seed: int = 0,
        corrector_options: Optional[dict] = None,
    ):
        self.lock = threading.RLock()
        self.clock = clock
        self._now = _clock_fn(clock)
        self.allow_ht = bool(allow_ht)
        self.stage_speed_um_s = float(stage_speed_um_s)
        self.tilt_speed_deg_s = float(tilt_speed_deg_s)
        self.stage_limits = tuple(float(v) for v in stage_limits)
        self._callbacks: list[Callable[[str, Any], None]] = []

        s = (state or MicroscopeState()).copy()
        if instrument_type is not None:
            s.instrument_type = str(instrument_type)
        self._s = s
        fm = L.FM_DIFF if s.projection == Projection.DIFFRACTION else function_mode_from_name(s.mag_mode)
        if s.tem_stem == TemStem.STEM and fm not in (L.FM_LOWMAG, L.FM_DIFF):
            fm = L.FM_LOWMAG
        self._fm = fm
        self._last_imaging_fm = fm if fm != L.FM_DIFF else L.FM_MAG1
        self._convergence_explicit: Optional[float] = (
            float(s.convergence_semi_angle_mrad) if s.convergence_semi_angle_mrad > 0 else None)
        pos = [s.stage.x_um, s.stage.y_um, s.stage.z_um, s.stage.alpha_deg, s.stage.beta_deg]
        self._stage_start = list(pos)
        self._stage_target = list(pos)
        self._stage_now = list(pos)
        self._move_t0 = self._now()
        # [twin] stage backlash (um) and each of x/y's last approach direction (+1 / -1 / 0)
        self.backlash_um = 0.0
        self._approach = [0.0, 0.0]
        self._last_op_state = 0  # DE-TEM-Channel OperationState of the last set
        sizes = [2] * L.APERTURE_KIND_COUNT
        sizes[L.APERTURE_KIND_CLA] = int(s.condenser_aperture_index)
        self._extra: dict[str, Any] = {
            "MaxHighTension": max(200000.0, s.ht_kv * 1000.0),
            "Filament": 1.0,
            "StageMode": 0,
            "StageHolder": 1,
            "ScanRotation": 0.0,
            "DiffractionFocus": 0.0,
            "GunShift": (0.0, 0.0),
            "GunTilt": (0.0, 0.0),
            "ApertureKind": L.APERTURE_KIND_CLA,
            "ApertureSizes": sizes,
            "VacuumStatus": 1.0,
            "VacuumGauges": 0.0,
            "ScreenCurrent": 100.0,
            "ExposureTime": 0.0,
        }
        self._resnap()
        self.seed = int(seed)
        self.corrector = None
        kind = corrector if corrector is not None else s.corrector
        from .corrector import normalize_kind

        kind = normalize_kind(kind)
        s.corrector = kind
        if kind != "none":
            from .corrector import Corrector

            opts = dict(corrector_options or {})
            opts.setdefault("sleeper", getattr(clock, "sleep", None))
            self.corrector = Corrector(
                kind, clock=self._now, seed=self.seed, lock=self.lock, ht_kv=s.ht_kv,
                operator=self._operator_c1a1, beam_check=self._corrector_beam_check,
                label=self._corrector_label, **opts)

    # ------------------------------------------------------------ corrector
    def _operator_c1a1(self, side: str) -> tuple[float, complex]:
        """The operator's focus (C1) and stigmator (A1), nm, as derive_optics adds them:
        C1 = (defocus + stage z) * 1000, A1 = objective (image) / condenser (probe) stig
        x 1000 nm per unit (OpticsConfig defaults)."""
        s = self._s
        self._update_stage()
        c1 = (float(s.defocus_um) + float(self._stage_now[2])) * 1000.0
        stig = s.condenser_stig if side == "probe" else s.objective_stig
        return c1, complex(stig.x * 1000.0, stig.y * 1000.0)

    def _corrector_beam_check(self, side: str) -> Optional[str]:
        """Why a corrector measurement cannot acquire now (None = it can)."""
        s = self._s
        if not s.ht_on or not s.column_valves_open or s.beam_blanked:
            return "no beam (HT off, column valves closed or beam blanked)"
        if side == "probe" and s.tem_stem != TemStem.STEM:
            return "the probe corrector measures in STEM mode (Ronchigram); switch to STEM"
        if side == "image" and (s.tem_stem != TemStem.TEM or self._fm == L.FM_DIFF):
            return "the image corrector measures in TEM imaging mode; switch to TEM imaging"
        return None

    def _corrector_label(self) -> str:
        s = self._s
        return (f"{s.ht_kv:g}kV {'STEM' if s.tem_stem == TemStem.STEM else 'TEM'} "
                f"{L.FUNCTION_MODE_NAMES[self._fm]} {s.probe_mode.name}")

    def _tune_key(self) -> tuple:
        s = self._s
        return (float(s.ht_kv), int(s.probe_mode), int(self._fm), int(s.tem_stem))

    # ------------------------------------------------------------ callbacks
    def on_change(self, callback: Callable[[str, Any], None]) -> None:
        """Call ``callback(name, value)`` after every accepted write (outside the lock)."""
        self._callbacks.append(callback)

    def _notify(self, name: str, value: Any) -> None:
        for cb in list(self._callbacks):
            try:
                cb(name, value)
            except Exception:  # noqa: BLE001
                log.exception("column on_change callback failed")

    # ---------------------------------------------------------------- time
    def now(self) -> float:
        return self._now()

    # ---------------------------------------------------------------- state
    @property
    def instrument_type(self) -> str:
        return self._s.instrument_type

    @property
    def function_mode(self) -> int:
        """Raw JEOL function mode 0..4 (MAG1, MAG2, LowMAG, SAMAG, DIFF)."""
        return self._fm

    def mag_ladder(self):
        """Magnifications a ``Magnification`` write can reach from here (``None`` in TEM
        diffraction). In TEM imaging that is LowMAG's and MAG1's together: a write
        beyond the active mode's ladder crosses into the other (`imaging_mode_for`),
        as a real column's mag knob does, so a client stepping this ladder is not
        stuck at the top of LowMAG."""
        with self.lock:
            if self._s.tem_stem == TemStem.TEM and self._fm != L.FM_DIFF:
                return L.LOWMAG_MAGS + L.MAG1_MAGS
            return L.active_ladders(self._fm, int(self._s.tem_stem))[0]

    def cl_ladder(self):
        """Camera lengths (cm) of the active optics mode (``None`` in TEM imaging)."""
        with self.lock:
            return L.active_ladders(self._fm, int(self._s.tem_stem))[1]

    def _update_stage(self) -> bool:
        """Advance the interpolated stage to now. Returns True while moving."""
        dt = max(0.0, self._now() - self._move_t0)
        moving = False
        for i in range(5):
            speed = self.stage_speed_um_s if i < 3 else self.tilt_speed_deg_s
            start, target = self._stage_start[i], self._stage_target[i]
            span = target - start
            travelled = speed * dt
            if travelled >= abs(span):
                self._stage_now[i] = target
            else:
                self._stage_now[i] = start + (travelled if span >= 0 else -travelled)
                moving = True
        return moving

    def time_to_idle(self) -> float:
        """Clock seconds until the stage has arrived (0 when idle)."""
        with self.lock:
            self._update_stage()
            rem = 0.0
            for i in range(5):
                speed = self.stage_speed_um_s if i < 3 else self.tilt_speed_deg_s
                d = abs(self._stage_target[i] - self._stage_now[i])
                if d > 0:
                    rem = max(rem, d / speed if speed > 0 and math.isfinite(speed) else 0.0)
            return rem

    @property
    def busy(self) -> bool:
        with self.lock:
            return self._update_stage()

    def convergence_mrad(self) -> float:
        with self.lock:
            return self._convergence()

    def _convergence(self) -> float:
        if self._convergence_explicit is not None:
            return self._convergence_explicit
        if is_tfs(self._s.instrument_type):
            row = 2 if self._s.probe_mode in (ProbeMode.NANOPROBE, ProbeMode.NBD, ProbeMode.CBD,
                                              ProbeMode.EDS) else 0
        else:
            row = jeol_row(self._s.instrument_type, self._s.probe_mode)
        return L.convergence_mrad(row, self._s.alpha_selector, stem=self._s.tem_stem == TemStem.STEM)

    def state(self) -> MicroscopeState:
        """Snapshot, with the stage interpolated to now."""
        with self.lock:
            moving = self._update_stage()
            s = self._s.copy()
            x, y, z, a, b = self._stage_now
            s.stage = StagePosition(x, y, z, a, b)
            # backlash: the stage stops short of the reported position, against its approach
            h = 0.5 * float(self.backlash_um)
            s.stage_error_um = Vec2(-h * self._approach[0], -h * self._approach[1])
            s.op_status = 1 if moving else 0
            s.mag_mode = L.FUNCTION_MODE_NAMES[self._fm]
            s.projection = Projection.DIFFRACTION if self._fm == L.FM_DIFF else Projection.IMAGING
            s.convergence_semi_angle_mrad = self._convergence()
            s.condenser_aperture_index = int(self._extra["ApertureSizes"][L.APERTURE_KIND_CLA])
            if self.corrector is not None:
                s.corrector = self.corrector.kind
                s.probe_aberrations = self.corrector.residual_dict("probe")
                s.image_aberrations = self.corrector.residual_dict("image")
            return s

    def op_state(self) -> int:
        """DE-TEM-Channel OperationState: 1 running while the stage moves, else the last set's."""
        with self.lock:
            if self._update_stage():
                return 1
            return self._last_op_state

    def stage_position(self) -> tuple[float, float, float, float, float]:
        with self.lock:
            self._update_stage()
            return tuple(self._stage_now)  # type: ignore[return-value]

    def stage_target(self) -> tuple[float, float, float, float, float]:
        with self.lock:
            return tuple(self._stage_target)  # type: ignore[return-value]

    # ----------------------------------------------------------- get / set
    def get(self, name: str) -> Any:
        """Read a property by its DE-TEM-Channel name (see the module docstring)."""
        getter = _GETTERS.get(name)
        if getter is None:
            raise KeyError(f"unknown column property {name!r}")
        with self.lock:
            return getter(self)

    def values(self) -> dict[str, Any]:
        """Every readable property."""
        with self.lock:
            return {name: g(self) for name, g in _GETTERS.items()}

    @staticmethod
    def property_names() -> list[str]:
        return sorted(set(_GETTERS) | set(_SETTERS))

    def set(self, name: str, value: Any) -> None:
        """Write a property by its DE-TEM-Channel name. Raises :class:`ColumnRefused`."""
        self._guard(name)
        with self.lock:
            self._set_local(name, value)
            readback = _GETTERS[name](self) if name in _GETTERS else value
        self._notify(name, readback)

    def _guard(self, name: str) -> None:
        if name not in _SETTERS:
            raise KeyError(f"unknown or read-only column property {name!r}")
        if name in FORBIDDEN and not self.allow_ht:
            raise ColumnRefused(f"{name} is not controllable (build the Column with allow_ht=True)")

    def _set_local(self, name: str, value: Any) -> None:
        with self.lock:
            before = self._tune_key() if self.corrector is not None else None
            try:
                _SETTERS[name](self, value)
            except ColumnRefused:
                self._last_op_state = 3
                raise
            self._last_op_state = 2
            if before is not None:
                after = self._tune_key()
                if after[0] != before[0]:
                    self.corrector.on_ht_change(after[0])
                elif after != before:
                    self.corrector.perturb("mode")

    # --------------------------------------------------------- typed setters
    # These route through set() so a MirrorColumn forwards them unchanged.
    def set_magnification(self, mag: float) -> None:
        self.set("Magnification", mag)

    def set_magnification_index(self, index: int) -> None:
        self.set("MagnificationIndex", index)

    def set_camera_length_mm(self, mm: float) -> None:
        self.set("CameraLength", float(mm) / 10.0)

    def set_camera_length_cm(self, cm: float) -> None:
        self.set("CameraLength", cm)

    def set_defocus_um(self, um: float) -> None:
        self.set("Defocus", um)

    def set_spot_size(self, spot: int) -> None:
        self.set("SpotSize", spot)

    def set_probe_mode(self, mode) -> None:
        self.set("ProbeMode", mode)

    def set_alpha_selector(self, alpha: int) -> None:
        self.set("AlphaSelector", alpha)

    def set_convergence_mrad(self, mrad: float) -> None:
        self.set("ConvergenceAngle", mrad)

    def set_condenser_aperture(self, index: int) -> None:
        self.set("CondenserApertureIndex", index)

    def set_intensity(self, value: float) -> None:
        self.set("Intensity", value)

    def set_image_shift_um(self, x: float, y: float) -> None:
        self.set("ImageShift", (x, y))

    def set_beam_shift_um(self, x: float, y: float) -> None:
        self.set("BeamShift", (x, y))

    def set_beam_tilt_mrad(self, x: float, y: float) -> None:
        self.set("BeamTilt", (x / BEAM_TILT_MRAD_PER_UNIT, y / BEAM_TILT_MRAD_PER_UNIT))

    def set_diffraction_shift_mrad(self, x: float, y: float) -> None:
        self.set("DiffractionShift", (x / DIFF_SHIFT_MRAD_PER_UNIT, y / DIFF_SHIFT_MRAD_PER_UNIT))

    def set_objective_stig(self, x: float, y: float) -> None:
        self.set("ObjectiveStig", (x, y))

    def set_condenser_stig(self, x: float, y: float) -> None:
        self.set("CondenserStig", (x, y))

    def set_stage(self, x=None, y=None, z=None, alpha=None, beta=None) -> None:
        """Start a stage move (non-blocking; see :meth:`wait_idle`). um / degrees."""
        axes = {k: v for k, v in zip(STAGE_AXES, (x, y, z, alpha, beta)) if v is not None}
        if axes:
            self.set("StagePosition", axes)

    def move_stage(self, x=None, y=None, z=None, alpha=None, beta=None, *, wait: bool = True,
                   timeout: Optional[float] = None) -> bool:
        self.set_stage(x=x, y=y, z=z, alpha=alpha, beta=beta)
        return self.wait_idle(timeout) if wait else True

    def set_tem_stem(self, mode) -> None:
        self.set("TemStemMode", int(mode))

    def set_projection(self, mode) -> None:
        self.set("ProjectionMode", int(mode))

    def set_function_mode(self, mode) -> None:
        """Raw vendor function mode, or a JEOL name ("MAG1", "LowMAG", "DIFF", ...)."""
        if isinstance(mode, str):
            mode = submode_to_raw(self.instrument_type, function_mode_from_name(mode))
        self.set("SubMode", int(mode))

    def set_beam_blank(self, blanked: bool) -> None:
        self.set("BeamBlank", int(bool(blanked)))

    def set_screen(self, down: bool) -> None:
        self.set("ScreenPosition", screen_to_raw(self.instrument_type, int(bool(down))))

    def set_column_valves(self, open_: bool) -> None:
        self.set("ColumnValvesOpen", int(bool(open_)))

    def set_ht_kv(self, kv: float) -> None:
        self.set("HighTension", float(kv) * 1000.0)

    def set_instrument_type(self, vendor: str) -> None:
        self.set("InstrumentType", vendor)

    # ------------------------------------------------------------- control
    def stop(self) -> None:
        """Freeze the stage wherever its trajectory has reached (stopStage)."""
        with self.lock:
            self._update_stage()
            self._stage_start = list(self._stage_now)
            self._stage_target = list(self._stage_now)
            self._move_t0 = self._now()
            self._last_op_state = 4

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until the stage has arrived. Uses ``clock.sleep`` when the clock
        has one, so a ``ManualClock`` is simply advanced. ``timeout`` in clock seconds."""
        start = self._now()
        sleeper = getattr(self.clock, "sleep", None)
        while True:
            rem = self.time_to_idle()
            if rem <= 0:
                return True
            if timeout is not None and self._now() - start >= timeout:
                return False
            step = rem if timeout is None else min(rem, max(0.0, timeout - (self._now() - start)))
            if callable(sleeper):
                sleeper(step + 1e-9)
            else:
                time.sleep(min(step, 0.05) + 1e-4)

    def close(self) -> None:
        """Nothing to release for a simulated column."""

    # -------------------------------------------------------------- internals
    def _ladders(self):
        return L.active_ladders(self._fm, int(self._s.tem_stem))

    def _resnap(self) -> None:
        mags, cams = self._ladders()
        if mags is not None:
            self._s.magnification = L.snap(self._s.magnification, mags)
        if cams is not None:
            self._s.camera_length_mm = L.snap(self._s.camera_length_mm / 10.0, cams) * 10.0

    def _mag_index(self) -> int:
        mags, cams = self._ladders()
        if mags is not None:
            return int(min(range(len(mags)), key=lambda i: abs(mags[i] - self._s.magnification)))
        if cams is not None:
            cl = self._s.camera_length_mm / 10.0
            return int(min(range(len(cams)), key=lambda i: abs(cams[i] - cl)))
        return 0

    # ---- setters (under the lock) ----
    def _s_mag(self, value) -> None:
        mag = float(value)
        if mag <= 0:
            raise ColumnRefused("magnification must be > 0")
        s = self._s
        if s.tem_stem == TemStem.TEM and self._fm != L.FM_DIFF:
            wanted = L.imaging_mode_for(self._fm, mag)
            if wanted != self._fm:
                self._fm = wanted
                self._last_imaging_fm = wanted
        mags, _ = self._ladders()
        if mags is None:
            raise ColumnRefused("no magnification in TEM diffraction")
        s.magnification = L.snap(mag, mags)

    def _s_mag_index(self, value) -> None:
        idx = int(value)
        mags, cams = self._ladders()
        s = self._s
        if s.tem_stem == TemStem.TEM and self._fm != L.FM_DIFF:
            if idx < 0 and self._fm != L.FM_LOWMAG:
                return self._s_mag(L.LOWMAG_MAGS[-1])
            if mags is not None and idx >= len(mags) and self._fm == L.FM_LOWMAG:
                return self._s_mag(L.MAG1_MAGS[0])
        if mags is not None:
            s.magnification = float(mags[min(max(idx, 0), len(mags) - 1)])
            return None
        if cams is not None:
            s.camera_length_mm = float(cams[min(max(idx, 0), len(cams) - 1)]) * 10.0
            return None
        raise ColumnRefused("no ladder in this mode")

    def _s_cl_cm(self, value) -> None:
        cl = float(value)
        if cl <= 0:
            raise ColumnRefused("camera length must be > 0")
        _, cams = self._ladders()
        if cams is None:
            raise ColumnRefused("no camera length in TEM imaging")
        self._s.camera_length_mm = L.snap(cl, cams) * 10.0

    def _s_function_mode(self, fm: int) -> None:
        if not 0 <= fm <= 4:
            raise ColumnRefused(f"function mode {fm} out of range")
        if self._s.tem_stem == TemStem.STEM and fm not in (L.FM_LOWMAG, L.FM_DIFF):
            raise ColumnRefused("MAG1/MAG2/SAMAG have no STEM project; use LowMAG or DIFF")
        self._fm = fm
        if fm != L.FM_DIFF:
            self._last_imaging_fm = fm
        self._resnap()

    def _s_submode(self, value) -> None:
        self._s_function_mode(submode_from_raw(self._s.instrument_type, int(value)))

    def _s_projection(self, value) -> None:
        mode = int(value)
        if mode == 2:
            return self._s_function_mode(L.FM_DIFF)
        if mode != 1:
            raise ColumnRefused("projection mode must be 1 (imaging) or 2 (diffraction)")
        imaging = L.FM_LOWMAG if self._s.tem_stem == TemStem.STEM else self._last_imaging_fm
        return self._s_function_mode(imaging)

    def _s_tem_stem(self, value) -> None:
        mode = int(value)
        if mode not in (0, 1):
            raise ColumnRefused("TemStemMode must be 0 (TEM) or 1 (STEM)")
        self._s.tem_stem = TemStem(mode)
        if mode == 1 and self._fm not in (L.FM_LOWMAG, L.FM_DIFF):
            self._fm = L.FM_LOWMAG
            self._last_imaging_fm = L.FM_LOWMAG
        self._resnap()

    def _s_probe(self, value) -> None:
        vendor = self._s.instrument_type
        if isinstance(value, ProbeMode):
            mode = value
        elif isinstance(value, str) and not value.strip().lstrip("-").isdigit():
            try:
                mode = ProbeMode[value.strip().upper()]
            except KeyError:
                raise ColumnRefused(f"unknown probe mode {value!r}") from None
        else:
            raw = int(value)
            if is_tfs(vendor):
                raw = min(max(raw, 0), 1)
            else:
                raw = min(max(raw, 0), 3)  # Dummy: ClampShort(mode, 0, 3)
            mode = probe_mode_from_raw(vendor if (is_jeol(vendor) or is_tfs(vendor)) else "JEOL", raw)
        self._s.probe_mode = ProbeMode(mode)

    def _s_convergence(self, value) -> None:
        v = float(value)
        self._convergence_explicit = v if v > 0 else None

    def _s_stage(self, value) -> None:
        if isinstance(value, StagePosition):
            value = {"x": value.x_um, "y": value.y_um, "z": value.z_um,
                     "a": value.alpha_deg, "b": value.beta_deg}
        elif not isinstance(value, dict):
            vals = list(value)
            value = dict(zip(STAGE_AXES, vals))
        aliases = {"alpha": "a", "beta": "b", "x_um": "x", "y_um": "y", "z_um": "z",
                   "alpha_deg": "a", "beta_deg": "b"}
        axes = {aliases.get(k, k): v for k, v in value.items() if v is not None}
        unknown = set(axes) - set(STAGE_AXES)
        if unknown:
            raise ColumnRefused(f"unknown stage axes {sorted(unknown)} (use x, y, z, a, b)")
        self._update_stage()
        self._stage_start = list(self._stage_now)
        for i, axis in enumerate(STAGE_AXES):
            if axis in axes:
                v = float(axes[axis])
                if not math.isfinite(v):
                    raise ColumnRefused(f"stage {axis} must be finite")
                lim = self.stage_limits[i]
                self._stage_target[i] = min(max(v, -lim), lim)
                if i < 2 and self._stage_target[i] != self._stage_now[i]:
                    self._approach[i] = 1.0 if self._stage_target[i] > self._stage_now[i] else -1.0
        self._move_t0 = self._now()

    def _s_screen(self, value) -> None:
        raw = int(value)
        vendor = self._s.instrument_type
        if is_tfs(vendor):
            if raw not in (2, 3):
                raise ColumnRefused(f"TFS screen positions are 2 (up) and 3 (down), not {raw}")
        elif raw not in (0, 1):
            raise ColumnRefused(f"invalid screen position {raw} (0 = up, 1 = down)")
        self._s.screen_position = screen_from_raw(vendor, raw)

    def _s_aperture_size(self, value) -> None:
        kind, size = value
        kind, size = int(kind), int(size)
        if not 0 <= kind < L.APERTURE_KIND_COUNT:
            raise ColumnRefused(f"aperture kind {kind} out of range")
        self._extra["ApertureKind"] = kind
        self._extra["ApertureSizes"][kind] = min(max(size, 0), L.APERTURE_HOLE_MAX)

    def _s_aperture_kind(self, value) -> None:
        kind = int(value)
        if not 0 <= kind < L.APERTURE_KIND_COUNT:
            raise ColumnRefused(f"aperture kind {kind} out of range")
        self._extra["ApertureKind"] = kind

    def _s_ht(self, value) -> None:
        volts = float(value)
        if volts <= 0:
            raise ColumnRefused("high tension must be > 0")
        self._s.ht_kv = volts / 1000.0
        self._extra["MaxHighTension"] = max(self._extra["MaxHighTension"], volts)

    def _stage_dict(self) -> dict[str, float]:
        self._update_stage()
        return dict(zip(STAGE_AXES, self._stage_now))

    # ---- extras for MirrorColumn ----
    def _apply_state(self, s: MicroscopeState, *, op_state: Optional[int] = None,
                     extra: Optional[dict] = None, convergence_explicit: Optional[float] = None) -> list[str]:
        """Replace the whole state (no dynamics, no refusals). Returns changed names."""
        with self.lock:
            before = self.values()
            self._s = s.copy()
            fm = L.FM_DIFF if s.projection == Projection.DIFFRACTION else function_mode_from_name(s.mag_mode)
            self._fm = fm
            if fm != L.FM_DIFF:
                self._last_imaging_fm = fm
            pos = [s.stage.x_um, s.stage.y_um, s.stage.z_um, s.stage.alpha_deg, s.stage.beta_deg]
            self._stage_start = list(pos)
            self._stage_target = list(pos)
            self._stage_now = list(pos)
            self._move_t0 = self._now()
            self._convergence_explicit = convergence_explicit
            self._extra["ApertureSizes"][L.APERTURE_KIND_CLA] = int(s.condenser_aperture_index)
            if extra:
                self._extra.update(extra)
            if op_state is not None:
                self._last_op_state = int(op_state)
            after = self.values()
        return [k for k in after if after[k] != before.get(k)]


# ---------------------------------------------------------------------------
# Property tables
# ---------------------------------------------------------------------------

def _v2(v: Vec2, scale: float = 1.0) -> tuple[float, float]:
    return (v.x / scale, v.y / scale)


def _set_vec(attr: str, scale: float = 1.0):
    def setter(col: Column, value) -> None:
        x, y = _pair(value, attr)
        setattr(col._s, attr, Vec2(x * scale, y * scale))
    return setter


def _set_flag(attr: str):
    def setter(col: Column, value) -> None:
        v = value.strip().lower() in ("1", "true", "on", "yes") if isinstance(value, str) else bool(value)
        setattr(col._s, attr, v)
    return setter


def _set_float(attr: str, lo: float, hi: float):
    def setter(col: Column, value) -> None:
        v = float(value)
        if not lo <= v <= hi:
            raise ColumnRefused(f"{attr} must be in [{lo:g}, {hi:g}], got {v:g}")
        setattr(col._s, attr, v)
    return setter


def _set_extra_pair(key: str):
    def setter(col: Column, value) -> None:
        col._extra[key] = _pair(value, key)
    return setter


def _set_extra_float(key: str):
    def setter(col: Column, value) -> None:
        col._extra[key] = float(value)
    return setter


def _set_axis(i: int):
    def setter(col: Column, value) -> None:
        col._s_stage({STAGE_AXES[i]: float(value)})
    return setter


def _get_axis(i: int):
    def getter(col: Column) -> float:
        col._update_stage()
        return float(col._stage_now[i])
    return getter


_GETTERS: dict[str, Callable[[Column], Any]] = {
    "Magnification": lambda c: float(c._s.magnification),
    "MagString": lambda c: format(float(c._s.magnification), ".6g"),
    "MagnificationIndex": lambda c: c._mag_index(),
    "CameraLength": lambda c: float(c._s.camera_length_mm) / 10.0,
    "Defocus": lambda c: float(c._s.defocus_um),
    "Focus": lambda c: int(c._s.defocus_um),
    "SpotSize": lambda c: int(c._s.spot_size),
    "ProbeMode": lambda c: probe_mode_to_raw(c._s.instrument_type, c._s.probe_mode),
    "ConvergenceAngle": lambda c: float(c._convergence()),
    "AlphaSelector": lambda c: int(c._s.alpha_selector),
    "CondenserApertureIndex": lambda c: int(c._extra["ApertureSizes"][L.APERTURE_KIND_CLA]),
    "Intensity": lambda c: float(c._s.intensity),
    "ImageShift": lambda c: _v2(c._s.image_shift_um),
    "BeamShift": lambda c: _v2(c._s.beam_shift_um),
    "BeamTilt": lambda c: _v2(c._s.beam_tilt_mrad, BEAM_TILT_MRAD_PER_UNIT),
    "DiffractionShift": lambda c: _v2(c._s.diffraction_shift_mrad, DIFF_SHIFT_MRAD_PER_UNIT),
    "Precession": lambda c: bool(c._s.precession_on),
    "PrecessionAngle": lambda c: float(c._s.precession_mrad),
    "PrecessionFrequency": lambda c: float(c._s.precession_hz),
    "PrecessionDescan": lambda c: bool(c._s.precession_descan),
    "ObjectiveStig": lambda c: _v2(c._s.objective_stig),
    "CondenserStig": lambda c: _v2(c._s.condenser_stig),
    "StagePosition": lambda c: c._stage_dict(),
    "StageX": _get_axis(0),
    "StageY": _get_axis(1),
    "StageZ": _get_axis(2),
    "StageA": _get_axis(3),
    "StageB": _get_axis(4),
    "TemStemMode": lambda c: int(c._s.tem_stem),
    "ProjectionMode": lambda c: 2 if c._fm == L.FM_DIFF else 1,
    "SubMode": lambda c: submode_to_raw(c._s.instrument_type, c._fm),
    "FunctionMode": lambda c: submode_to_raw(c._s.instrument_type, c._fm),
    "MagMode": lambda c: L.FUNCTION_MODE_NAMES[c._fm],
    "BeamBlank": lambda c: int(bool(c._s.beam_blanked)),
    "ScreenPosition": lambda c: screen_to_raw(c._s.instrument_type, c._s.screen_position),
    "ColumnValvesOpen": lambda c: int(bool(c._s.column_valves_open)),
    "HighTension": lambda c: float(c._s.ht_kv) * 1000.0,
    "HT": lambda c: float(c._s.ht_kv) * 1000.0,
    "MaxHighTension": lambda c: float(c._extra["MaxHighTension"]),
    "HTState": lambda c: int(bool(c._s.ht_on)),
    "Filament": lambda c: float(c._extra["Filament"]),
    "InstrumentType": lambda c: str(c._s.instrument_type),
    "OperationStatus": lambda c: c.op_state(),
    "StageMode": lambda c: int(c._extra["StageMode"]),
    "StageHolder": lambda c: int(c._extra["StageHolder"]),
    "ScanRotation": lambda c: float(c._extra["ScanRotation"]),
    "DiffractionFocus": lambda c: float(c._extra["DiffractionFocus"]),
    "GunShift": lambda c: tuple(c._extra["GunShift"]),
    "GunTilt": lambda c: tuple(c._extra["GunTilt"]),
    "ApertureKind": lambda c: int(c._extra["ApertureKind"]),
    "VacuumStatus": lambda c: float(c._extra["VacuumStatus"]),
    "VacuumGauges": lambda c: float(c._extra["VacuumGauges"]),
    "ScreenCurrent": lambda c: float(c._extra["ScreenCurrent"]),
    "ExposureTime": lambda c: float(c._extra["ExposureTime"]),
}


def _set_spot(c: Column, v) -> None:
    c._s.spot_size = min(max(int(v), 1), 5)


def _set_alpha(c: Column, v) -> None:
    c._s.alpha_selector = min(max(int(v), 0), 7)


def _set_blank(c: Column, v) -> None:
    c._s.beam_blanked = bool(int(v))


def _set_valves(c: Column, v) -> None:
    c._s.column_valves_open = bool(int(v))


def _set_ht_state(c: Column, v) -> None:
    c._s.ht_on = bool(int(v))


def _set_stage_mode(c: Column, v) -> None:
    c._extra["StageMode"] = min(max(int(v), 0), 1)


def _set_cla(c: Column, v) -> None:
    c._s_aperture_size((L.APERTURE_KIND_CLA, int(v)))


def _set_defocus(c: Column, v) -> None:
    c._s.defocus_um = float(v)


def _set_intensity(c: Column, v) -> None:
    c._s.intensity = float(v)


def _set_vendor(c: Column, v) -> None:
    c._s.instrument_type = str(v)


_SETTERS: dict[str, Callable[[Column, Any], None]] = {
    "Magnification": Column._s_mag,
    "MagnificationIndex": Column._s_mag_index,
    "CameraLength": Column._s_cl_cm,
    "Defocus": _set_defocus,
    "SpotSize": _set_spot,
    "ProbeMode": Column._s_probe,
    "ConvergenceAngle": Column._s_convergence,
    "AlphaSelector": _set_alpha,
    "CondenserApertureIndex": _set_cla,
    "Intensity": _set_intensity,
    "ImageShift": _set_vec("image_shift_um"),
    "BeamShift": _set_vec("beam_shift_um"),
    "BeamTilt": _set_vec("beam_tilt_mrad", BEAM_TILT_MRAD_PER_UNIT),
    "DiffractionShift": _set_vec("diffraction_shift_mrad", DIFF_SHIFT_MRAD_PER_UNIT),
    "Precession": _set_flag("precession_on"),
    "PrecessionAngle": _set_float("precession_mrad", 0.0, 100.0),
    "PrecessionFrequency": _set_float("precession_hz", 0.0, 1.0e5),
    "PrecessionDescan": _set_flag("precession_descan"),
    "ObjectiveStig": _set_vec("objective_stig"),
    "CondenserStig": _set_vec("condenser_stig"),
    "StagePosition": Column._s_stage,
    "StageX": _set_axis(0),
    "StageY": _set_axis(1),
    "StageZ": _set_axis(2),
    "StageA": _set_axis(3),
    "StageB": _set_axis(4),
    "TemStemMode": Column._s_tem_stem,
    "ProjectionMode": Column._s_projection,
    "SubMode": Column._s_submode,
    "FunctionMode": Column._s_submode,
    "BeamBlank": _set_blank,
    "ScreenPosition": Column._s_screen,
    "ColumnValvesOpen": _set_valves,
    "HighTension": Column._s_ht,
    "HT": Column._s_ht,
    "HTState": _set_ht_state,
    "Filament": _set_extra_float("Filament"),
    "InstrumentType": _set_vendor,
    "StageMode": _set_stage_mode,
    "ScanRotation": _set_extra_float("ScanRotation"),
    "DiffractionFocus": _set_extra_float("DiffractionFocus"),
    "GunShift": _set_extra_pair("GunShift"),
    "GunTilt": _set_extra_pair("GunTilt"),
    "ApertureKind": Column._s_aperture_kind,
    "ApertureSize": Column._s_aperture_size,
}
