"""Camera model catalogue.

Geometry, timing and bit depth come from DE-Server's camera headers
(``code/source/CameraServer/Camera/*.h``), ``Constants/Params.cpp``
(``FrameParams::UpdateHWFrameSize``) and ``configurations/server_*.xml``:

=============  ==========  ======  =====  ==========  =====  ===========  =====
model          sensor      pixel   bits   topology    parts  frame time   sat.
=============  ==========  ======  =====  ==========  =====  ===========  =====
DESim          1024x1024   10 um   12     single      1      0.1 ms       2100
DE16 family    4096x4096   6.5 um  12     single      1      10.821 ms    2192
DE64/Vision64  8192x8192   6.5 um  12     left/right  2      21.58 ms     2600
Celeritas*     1024x1024   15 um   12     top/bottom  2      0.51 ms      4022
Apollo family  4096x4096   8 um    8 (hw counting)  l/r  2   0.418 ms/cyc  240
=============  ==========  ======  =====  ==========  =====  ===========  =====

("sat." is DE-Server's ``Saturation - Value (ADU)`` property, the per-pixel
level at which DE-Server flags a pixel as saturated. The raw ADC clips at
``2**bit_depth - 1``.)

``adu_table`` is DE-Server's HT table (``configurations/ht_table_*.yaml``):
the *total* ADU a single primary electron deposits, summed over its charge
cloud (DE-Server's ``m_aduPerElectronBin1``). Because the charge spread
conserves the sum, the mean per-pixel signal of a flat beam is
``dose_e_per_px * adu_per_electron * gain``.

The remaining sensor-physics parameters are not in DE-Server; they are
documented, plausible defaults chosen so that dark/gain references matter:

* ``adu_cv``: relative spread of the per-electron deposit (Landau-like,
  sampled as a gamma distribution with shape ``1/adu_cv**2``).
* ``charge_spread``: neighbour weight ``a`` of the separable 3-tap charge
  sharing kernel ``[a, 1-2a, a]`` at 300 kV (MTF at Nyquist = ``(1-4a)`` per
  axis). It grows slowly at lower HT.
* ``dark_offset_adu``: 360 ADU baseline, like GrabberSim's ``GenImage`` dark.
* fixed pattern: GrabberSim's ``(x%8)*(y%8)`` term plus its segment offsets
  (+10 on even segments, +20 on segment 0, -20 on segment 1), plus random
  per-column (``fpn_column_adu``) and per-pixel (``dsnu_adu``) offsets.
* gain non-uniformity: radial fall-off, a gradient across each stitch block
  (``segment_shape[0]`` rows, like ``GenImage``), per-column and per-pixel
  PRNU, normalised to mean 1.
* defects: hot pixels (large dark current), dead pixels (zero gain),
  bad columns (low gain + offset step).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

# DE-Server HT tables (kV, ADU per primary electron).
ADU_TABLE_DE_SERIES = ((80.0, 1379.0), (120.0, 791.0), (200.0, 316.0), (300.0, 206.0))
ADU_TABLE_DE16ER = ((40.0, 628.0), (60.0, 374.0), (80.0, 269.0), (100.0, 205.0),
                    (120.0, 175.0), (200.0, 129.0), (300.0, 86.0))
ADU_TABLE_CELERITAS = ((80.0, 1148.0), (200.0, 301.0), (300.0, 208.0))


@dataclass(frozen=True)
class CameraModel:
    # --- identity / geometry (DE-Server facts)
    name: str
    de_model: str  # DE-Server CameraModel enum name (Stringify), e.g. "DE16", "SIMULATOR"
    sensor_shape: tuple[int, int]  # (h, w) pixels
    pixel_um: float
    bit_depth: int
    bytes_per_pixel: int
    topology: str = "single"  # single | leftright | topbottom
    num_parts: int = 1
    sensor_family: str = "ADELE"  # ADELE | DESIREE | CASPAR2
    hardware_counting: bool = False
    full_frame_time_s: float = 0.010821  # fastest full-sensor frame (counting: one cycle)
    hw_quanta: int = 1  # counting: sensor cycles summed per delivered frame (default)
    saturation_adu: int = 2192  # DE-Server "Saturation - Value (ADU)"
    segment_shape: tuple[int, int] = (2048, 256)  # (m_segmentHeight, m_segmentWidth)
    supports_hw_roi_x: bool = False
    # --- sensor physics (defaults, see module docstring)
    adu_table: tuple[tuple[float, float], ...] = ADU_TABLE_DE_SERIES
    adu_cv: float = 0.7
    charge_spread: float = 0.20
    read_noise_adu: float = 9.0
    dark_offset_adu: float = 360.0
    dark_current_adu_per_s: float = 5.0
    dsnu_adu: float = 3.0
    fpn_column_adu: float = 4.0
    fpn_pattern: bool = True  # GrabberSim-style (x%8)(y%8) + segment offsets
    gain_pixel_sigma: float = 0.015
    gain_column_sigma: float = 0.005
    gain_radial: float = 0.04  # relative fall-off at the corners
    gain_stitch: float = 0.02  # gradient across a stitch block
    hot_pixel_fraction: float = 2e-5
    hot_dark_current_adu_per_s: tuple[float, float] = (2e4, 2e5)
    dead_pixel_fraction: float = 1e-5
    bad_column_fraction: float = 5e-4
    # counting cameras
    cluster_area_px: float = 2.5  # coincidence area of one event
    false_event_rate: float = 2e-6  # per pixel per sensor cycle
    reference_ht_kv: float = 300.0
    #: Below this beam energy an electron stops inside the sensitive layer and deposits all of
    #: its energy, so the ADU per electron falls in proportion to the energy (the HT table's
    #: log-log trend only holds while electrons pass through). SEM energies (EBSD, 5-30 kV).
    full_stop_kv: float = 40.0
    #: "high" (the HT table) or "low": a low-gain readout trades sensitivity for range, so that
    #: ``low_gain_full_well_e`` electrons in one pixel take it from the dark offset to the top of
    #: the ADC at any beam energy. The per-electron spread (``adu_cv``), read noise and the maps
    #: are those of the high-gain mode. For bright beams: an SEM's EBSD current puts several
    #: electrons in a pixel per frame, which pins a high-gain pixel at 20 kV (~1800 ADU each).
    gain_mode: str = "high"
    low_gain_full_well_e: float = 100.0
    aliases: tuple[str, ...] = field(default=(), compare=False)

    # ------------------------------------------------------------- derived
    @property
    def sensor_w(self) -> int:
        return self.sensor_shape[1]

    @property
    def sensor_h(self) -> int:
        return self.sensor_shape[0]

    @property
    def max_value(self) -> int:
        return (1 << self.bit_depth) - 1

    @property
    def dtype(self):
        return np.uint8 if self.bytes_per_pixel == 1 else np.uint16

    @property
    def max_fps(self) -> float:
        """Fastest full-sensor frame rate (counting cameras: delivered frames at hw_quanta)."""
        return 1.0 / (self.full_frame_time_s * max(1, self.hw_quanta))

    @property
    def adu_per_electron(self) -> float:
        return self.adu_per_electron_at(self.reference_ht_kv)

    def adu_per_electron_at(self, ht_kv: float) -> float:
        """Total ADU per primary electron at ``ht_kv`` (log-log interpolation of the HT table; in
        the low-gain mode the ADC range over ``low_gain_full_well_e``)."""
        if self.hardware_counting:
            return 1.0
        if self.gain_mode == "low":
            return (self.max_value - self.dark_offset_adu) / max(float(self.low_gain_full_well_e), 1e-9)
        if self.gain_mode != "high":
            raise ValueError(f"unknown gain mode {self.gain_mode!r} (high | low)")
        kv = np.array([p[0] for p in self.adu_table])
        adu = np.array([p[1] for p in self.adu_table])
        ht = float(np.clip(ht_kv, 1.0, 1000.0))
        stop = float(self.full_stop_kv)
        if 0.0 < stop and ht < stop and ht < kv[0]:  # fully absorbed: deposit ~ energy
            return self.adu_per_electron_at(stop) * ht / stop
        if kv[0] <= ht <= kv[-1]:
            return float(np.exp(np.interp(np.log(ht), np.log(kv), np.log(adu))))
        return _loglog_extrapolate(kv, adu, ht)

    def charge_spread_at(self, ht_kv: float) -> float:
        """Neighbour weight of the 3-tap charge kernel; mildly larger at low HT (heuristic)."""
        a = self.charge_spread * (self.reference_ht_kv / max(float(ht_kv), 20.0)) ** 0.3
        return float(min(a, 0.25))

    def with_(self, **changes) -> "CameraModel":
        return dataclasses.replace(self, **changes)


def _loglog_extrapolate(kv, adu, ht):
    i = (0, 1) if ht < kv[0] else (-2, -1)
    lx, ly = np.log(kv[list(i)]), np.log(adu[list(i)])
    slope = (ly[1] - ly[0]) / (lx[1] - lx[0])
    return float(np.exp(ly[0] + slope * (np.log(ht) - lx[0])))


def _adele(name, de_model, *, sat=2192, aliases=(), **kw) -> CameraModel:
    base = dict(sensor_shape=(4096, 4096), pixel_um=6.5, bit_depth=12, bytes_per_pixel=2,
                sensor_family="ADELE", full_frame_time_s=10.821e-3, saturation_adu=sat,
                supports_hw_roi_x=True)
    base.update(kw)
    return CameraModel(name=name, de_model=de_model, aliases=aliases, **base)


def _desiree(name, de_model, *, sat, aliases=()) -> CameraModel:
    return CameraModel(
        name=name, de_model=de_model, sensor_shape=(1024, 1024), pixel_um=15.0,
        bit_depth=12, bytes_per_pixel=2, topology="topbottom", num_parts=2,
        sensor_family="DESIREE", full_frame_time_s=0.51e-3, saturation_adu=sat,
        segment_shape=(512, 128), adu_table=ADU_TABLE_CELERITAS,
        charge_spread=0.12,  # 15 um pixels: less charge sharing per pixel
        read_noise_adu=12.0, aliases=aliases)


def _caspar(name, de_model, aliases=()) -> CameraModel:
    return CameraModel(
        name=name, de_model=de_model, sensor_shape=(4096, 4096), pixel_um=8.0,
        bit_depth=8, bytes_per_pixel=1, topology="leftright", num_parts=2,
        sensor_family="CASPAR2", hardware_counting=True, full_frame_time_s=0.418e-3,
        hw_quanta=40, saturation_adu=240, segment_shape=(2048, 1024),
        read_noise_adu=0.0, dark_offset_adu=0.0, dark_current_adu_per_s=0.0,
        dsnu_adu=0.0, fpn_column_adu=0.0, fpn_pattern=False,
        gain_pixel_sigma=0.01, gain_column_sigma=0.003, gain_radial=0.02, gain_stitch=0.01,
        hot_dark_current_adu_per_s=(50.0, 500.0),  # hot pixels: spurious counts per second
        aliases=aliases)


_MODELS = [
    _adele("DESim", "SIMULATOR", sat=2100, sensor_shape=(1024, 1024), pixel_um=10.0,
           full_frame_time_s=0.1e-3, segment_shape=(256, 32), supports_hw_roi_x=False,
           aliases=("SIMULATOR", "Simulator", "DE-Sim")),
    _adele("DE16", "DE16", sat=2192, aliases=("DE-16",)),
    _adele("DE16ER", "DE16", sat=2100, adu_table=ADU_TABLE_DE16ER, aliases=("DE-16ER",)),
    _adele("LV16", "LV16", sat=2100, aliases=("DE16LV", "LV-16")),
    _adele("Vision16", "VISION16", sat=2192, aliases=("VISION16",)),
    _adele("DirectView2", "DIRECTVIEW2", sat=2100, aliases=("DE-DirectView2", "DIRECTVIEW2")),
    _adele("SEMCam", "SEMCAM", sat=2192, aliases=("DESEMCam", "SEMCAM")),
    _adele("Meridian", "MERIDIAN", sat=2192, aliases=("MERIDIAN",)),
    _adele("DE64", "DE64", sat=2600, sensor_shape=(8192, 8192), topology="leftright",
           num_parts=2, full_frame_time_s=21.58e-3, aliases=("DE-64",)),
    _adele("Vision64", "VISION64", sat=2600, sensor_shape=(8192, 8192), topology="leftright",
           num_parts=2, full_frame_time_s=21.58e-3, aliases=("VISION64",)),
    _desiree("Celeritas", "CELERITAS", sat=4022, aliases=("CELERITAS",)),
    _desiree("CeleritasXS", "CELERITASXS", sat=4022, aliases=("CELERITASXS",)),
    _desiree("Zenith", "ZENITH", sat=4061, aliases=("ZENITH",)),
    _caspar("Apollo", "APOLLO", aliases=("APOLLO",)),
    _caspar("ApolloXS", "APOLLOXS", aliases=("APOLLOXS",)),
    _caspar("Artemis", "ARTEMIS", aliases=("ARTEMIS",)),
    _caspar("Centuri", "CENTURI", aliases=("CENTURI",)),
    _caspar("CenturiPlus", "CENTURIPLUS", aliases=("CENTURIPLUS", "Centuri Plus")),
]

CAMERAS: dict[str, CameraModel] = {m.name: m for m in _MODELS}

_LOOKUP: dict[str, CameraModel] = {}
for _m in _MODELS:
    for _key in (_m.name, *_m.aliases):
        _LOOKUP.setdefault(_key.lower().replace("-", "").replace(" ", ""), _m)


def camera(name: "str | CameraModel") -> CameraModel:
    """Look up a camera model by name, alias or DE-Server enum name (case-insensitive)."""
    if isinstance(name, CameraModel):
        return name
    key = str(name).strip().lower().replace("-", "").replace(" ", "")
    try:
        return _LOOKUP[key]
    except KeyError:
        raise KeyError(f"unknown camera model {name!r}; known: {sorted(CAMERAS)}") from None
