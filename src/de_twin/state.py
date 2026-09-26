"""Shared state types: the contracts every part of the twin codes against.

Unit conventions (the same as DE-Server's VirtualSpecimen):

* specimen/world coordinates are micrometres, +x right, +y down
* stage x/y/z, beam shift and image shift are micrometres
* tilts are degrees at the API boundary (converted to radians internally)
* thickness is nanometres
* convergence and diffraction angles are milliradians
* camera length is millimetres (DE-Server metadata uses cm; faces convert)
* high tension is kilovolts
* time is seconds

Field names carry their unit as a suffix wherever there is any ambiguity.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class ProbeMode(IntEnum):
    """Vendor-normalised illumination mode (values match DE-Server ``ProbeMode``)."""

    UNKNOWN = -1
    NANOPROBE = 0
    MICROPROBE = 1
    TEM = 2
    EDS = 3
    NBD = 4
    CBD = 5


class TemStem(IntEnum):
    TEM = 0
    STEM = 1


class Projection(IntEnum):
    """Projector mode (values match DE-Server ``projectMode``)."""

    IMAGING = 1
    DIFFRACTION = 2


class ExposureMode(IntEnum):
    """DE-Server ``ExposureMode`` (Constants/Params.h)."""

    DARK = 0
    TRIAL = 1
    GAIN = 2
    NORMAL = 3
    RAW = 4
    VACUUM = 5
    INTERVALS = 6
    CONTINUOUS = 7


class RenderMode(IntEnum):
    TEM_IMAGING = 0
    TEM_DIFFRACTION = 1
    STEM_PARKED = 2
    STEM_4D = 3


@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class StagePosition:
    x_um: float = 0.0
    y_um: float = 0.0
    z_um: float = 0.0
    alpha_deg: float = 0.0
    beta_deg: float = 0.0


@dataclass
class MicroscopeState:
    """Everything the column can report or be told.

    This is a superset of DE-Server's ``MicroscopeMetadata`` and of what
    DE-TEM-Channel exposes, so that every face (SOAP server, deapi fake server,
    shared-memory producer, in-process API) serves the same values.
    """

    # Gun / vacuum
    instrument_type: str = "JEOL"  # vendor string DE-Server keys semantics on
    ht_kv: float = 200.0
    ht_on: bool = True
    column_valves_open: bool = True
    beam_blanked: bool = False
    screen_position: int = 0  # 0 = up (camera exposed), 1 = down

    # Mode
    tem_stem: TemStem = TemStem.TEM
    projection: Projection = Projection.IMAGING
    mag_mode: str = "MAG1"  # JEOL-style function mode name (MAG1, LowMAG, SAMAG, DIFF, ...)

    # Illumination
    spot_size: int = 3
    probe_mode: ProbeMode = ProbeMode.TEM
    convergence_semi_angle_mrad: float = 0.0  # 0 = derive from probe mode / alpha
    alpha_selector: int = 3
    condenser_aperture_index: int = 1
    intensity: float = 0.5  # C2/brightness, 0..1; sets illuminated area in TEM
    beam_shift_um: Vec2 = field(default_factory=Vec2)
    beam_tilt_mrad: Vec2 = field(default_factory=Vec2)
    condenser_stig: Vec2 = field(default_factory=Vec2)
    # Precession [twin]: the beam-tilt coils driven round a cone of half-angle
    # `precession_mrad` at `precession_hz`; with descan the pattern is brought back.
    # [twin] true stage position minus reported (backlash), x/y um
    stage_error_um: Vec2 = field(default_factory=Vec2)
    precession_on: bool = False
    precession_mrad: float = 10.0
    precession_hz: float = 100.0
    precession_descan: bool = True

    # Projection
    magnification: float = 20000.0
    camera_length_mm: float = 250.0
    defocus_um: float = 0.0
    image_shift_um: Vec2 = field(default_factory=Vec2)
    diffraction_shift_mrad: Vec2 = field(default_factory=Vec2)
    objective_stig: Vec2 = field(default_factory=Vec2)

    # Stage
    stage: StagePosition = field(default_factory=StagePosition)

    # Residual axial aberrations (CEOS names -> complex nm; see de_twin.optics.aberrations).
    # These EXCLUDE the operator's focus and stigmators (defocus_um, objective_stig,
    # condenser_stig), which derive_optics adds as C1 / A1. Empty = uncorrected column
    # (C3 from OpticsConfig/cs). Filled by the column's native aberrations + corrector.
    probe_aberrations: dict = field(default_factory=dict)  # STEM probe-forming side
    image_aberrations: dict = field(default_factory=dict)  # TEM imaging side
    corrector: str = "none"  # none | probe | image | both

    # Housekeeping
    op_status: int = 0  # 0 idle, 1 running (DE-TEM-Channel OP_Status)

    def copy(self) -> "MicroscopeState":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class HolderState:
    """In-situ holder state. SI units, like autopilot's ``Reading``."""

    kind: str = "none"  # none | heating | biasing | heating_biasing | liquid | gas
    temperature_c: float = 25.0
    target_c: float = 25.0
    power_w: float = 0.0
    resistance_ohm: float = 0.0
    bias_v: float = 0.0
    current_a: float = 0.0
    pressure_mbar: float = 0.0
    flow_ul_min: float = 0.0
    t_s: float = 0.0  # holder clock (time of the reading)

    def copy(self) -> "HolderState":
        return copy.deepcopy(self)


@dataclass
class Roi:
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0


@dataclass
class ScanRequest:
    enabled: bool = False
    size: tuple[int, int] = (256, 256)  # (nx, ny) scan points
    dwell_s: float = 1e-3
    frames_per_point: int = 1
    rotation_deg: float = 0.0
    step_um: float = 0.0  # 0 = derive from magnification calibration
    roi: Optional[Roi] = None
    park: bool = False
    park_position: tuple[int, int] = (0, 0)
    repeats: int = 1
    points: Optional[list[tuple[int, int]]] = None  # explicit XY list overrides raster


@dataclass
class AcquisitionRequest:
    """What the camera has been asked to produce.

    Filled from DE-Server (shared-memory control block), from the deapi fake
    server's properties, or directly by an in-process caller.
    """

    camera_model: str = "DE16"
    exposure_mode: ExposureMode = ExposureMode.NORMAL
    frame_time_s: float = 0.01
    total_frames: int = 1
    frames_per_buffer: int = 1
    hw_roi: Optional[Roi] = None  # None = full sensor
    hw_binning: tuple[int, int] = (1, 1)
    bit_depth: Optional[int] = None  # None = model default
    counting: bool = False
    beam_blank_cmd: int = 2  # DE-Server BeamBlankCmd: 0 close, 1 open, 2 auto
    shutter_cmd: int = 2  # ExposureShutterCmd: 0 close, 1 open, 2 auto
    camera_inserted: bool = True
    scan: ScanRequest = field(default_factory=ScanRequest)
    acquisition_index: int = 0
    #: Counting cameras: read out at 2x the ROI each way (Apollo "Super-resolution").
    super_resolution: bool = False
    #: Pins this acquisition's detector noise: frame k is seeded by (seed, k) whatever the
    #: twin exposed before. None: the twin's own seed and running frame count.
    seed: Optional[int] = None

    @property
    def beam_reaches_detector(self) -> bool:
        """False when the acquisition itself blocks the beam (dark reference etc.)."""
        if self.exposure_mode == ExposureMode.DARK:
            return False
        if self.beam_blank_cmd == 0 or self.shutter_cmd == 0:
            return False
        return self.camera_inserted


@dataclass
class FrameMeta:
    """Metadata published alongside every simulated frame."""

    frame_index: int
    time_s: float  # twin clock at exposure start
    exposure_s: float
    electrons_per_pixel: float  # mean dose actually delivered to the frame
    render_mode: RenderMode
    blanked: bool
    scan_point: Optional[tuple[int, int]] = None
    microscope: Optional[MicroscopeState] = None
    holder: Optional[HolderState] = None
