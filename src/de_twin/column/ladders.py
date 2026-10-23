"""Detent ladders and fixed tables of the simulated column.

Copied from DE-TEM-Channel's ``instruments/tem/dummy/DummyDetents.h`` (itself
generated from DE-Server ``configurations/mag.yaml``, sha256 87b690e6..., 2026-09-04)
and ``DummyEMInstrument.cpp``, so the twin's column reports exactly the
magnifications / camera lengths DE-Server's mag table has keys for.

JEOL function modes (the raw EOS value DE-TEM-Channel reports as ``PROJ_SubMode``):

    0 MAG1, 1 MAG2, 2 LowMAG, 3 SAMAG, 4 DIFF

crossed with TEM (0) / STEM (1). Only the seven (Mode, SubMode, TemStemMode)
triples that exist in mag.yaml can be reached; see :func:`active_ladders`.
"""

from __future__ import annotations

import math
from typing import Optional

FM_MAG1 = 0
FM_MAG2 = 1
FM_LOWMAG = 2
FM_SAMAG = 3
FM_DIFF = 4

#: JEOL function-mode names, indexed by the raw EOS value.
FUNCTION_MODE_NAMES = ("MAG1", "MAG2", "LowMAG", "SAMAG", "DIFF")

#: TFS/FEI ``PROJ_SubMode`` values (DE-Server ComputeMicroscopeMode):
#: 1 LM, 2 M, 3 SA, 4 Mh, 5 LAD, 6 D.  Mapping onto the JEOL-shaped state:
FEI_SUBMODE_FROM_FM = {FM_LOWMAG: 1, FM_MAG1: 2, FM_SAMAG: 3, FM_MAG2: 4, FM_DIFF: 6}
FM_FROM_FEI_SUBMODE = {1: FM_LOWMAG, 2: FM_MAG1, 3: FM_SAMAG, 4: FM_MAG2, 5: FM_DIFF, 6: FM_DIFF}

TEM = 0
STEM = 1


class Ladder(tuple):
    """An ascending tuple of the values an instrument knob can take (magnifications, or camera
    lengths in mm), with the two ways to pick the rung nearest a request.

    The column's detents step to the nearest rung by absolute difference; a calibration table
    picks the nearest by ratio, which is what an operator means by "about 2300x".
    """

    def __new__(cls, values=()):
        return super().__new__(cls, (float(value) for value in values))

    def __add__(self, other):
        return Ladder(tuple(self) + tuple(other))

    def snap_linear(self, value: float) -> float:
        """The rung nearest by absolute difference; ties take the lower rung (``SnapToNearest``
        in the Dummy)."""
        best = self[0]
        best_distance = abs(float(value) - best)
        for rung in self[1:]:
            distance = abs(float(value) - rung)
            if distance < best_distance:
                best, best_distance = rung, distance
        return float(best)

    def snap_logarithmic(self, value: float) -> float:
        """The rung nearest by ratio; ``value`` itself when the ladder is empty, the lowest rung
        for a value that is not positive."""
        if not self:
            return value
        if value is None or value <= 0:
            return self[0]
        return min(self, key=lambda rung: abs(math.log(rung / value)))

# --- DummyDetents.h (its camera lengths are cm; converted to mm, the twin's unit) ---
MAG1_MAGS = Ladder((
    2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 6000.0, 8000.0, 10000.0, 12000.0, 15000.0,
    20000.0, 25000.0, 30000.0, 40000.0, 50000.0, 60000.0, 80000.0, 100000.0, 120000.0,
    150000.0, 200000.0, 250000.0, 300000.0, 400000.0, 500000.0, 600000.0, 800000.0,
))
LOWMAG_MAGS = Ladder((120.0, 150.0, 200.0))
DIFF_CAMERA_LENGTHS_MM = Ladder((
    15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 130.0, 140.0, 150.0,
    200.0, 250.0, 300.0, 400.0, 800.0, 1000.0, 2000.0, 3000.0, 4000.0,
))
STEM_MAGS = Ladder((
    2500.0, 3000.0, 4000.0, 5000.0, 6000.0, 8000.0, 10000.0, 12000.0, 15000.0, 20000.0,
    25000.0, 30000.0, 40000.0, 50000.0, 60000.0, 80000.0, 120000.0, 150000.0, 200000.0,
    250000.0, 300000.0, 400000.0, 500000.0, 600000.0, 800000.0, 1000000.0, 1200000.0,
    1500000.0, 2000000.0, 2500000.0, 3000000.0, 4000000.0, 5000000.0, 6000000.0,
    8000000.0, 10000000.0, 12000000.0, 15000000.0, 20000000.0,
))
NBD_STEM_MAGS = STEM_MAGS  # kNbdStemMags == kHrStemMags
HR_STEM_MAGS = STEM_MAGS
STEM_CAMERA_LENGTHS_MM = Ladder((
    15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 130.0, 140.0, 150.0,
    200.0, 250.0, 300.0, 400.0, 800.0, 1000.0,
))

# --- The calibration's built-in tables (a JEOL-like set, used when no mag.yaml is loaded) ---
DEFAULT_LOWMAG_MAGS = Ladder((50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1500))
DEFAULT_SAMAG_MAGS = Ladder((5000, 6000, 8000, 10000, 12000, 15000, 20000, 25000, 30000, 40000, 50000,
                             60000, 80000, 100000, 120000, 150000))
DEFAULT_STEM_MAGS = Ladder((2500, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000, 20000, 25000, 30000,
                            40000, 50000, 60000, 80000, 100000, 120000, 150000, 200000, 250000, 300000,
                            400000, 500000, 600000, 800000, 1000000, 1200000, 1500000, 2000000, 2500000,
                            3000000, 4000000, 5000000, 6000000, 8000000, 10000000, 12000000, 15000000,
                            20000000))
DEFAULT_CAMERA_LENGTHS_MM = Ladder((20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600,
                                    800, 1000, 1200, 1500, 2000))

# --- DummyEMInstrument.cpp ------------------------------------------------
#: Stage slew rates (per linear axis / per tilt axis).
STAGE_SPEED_UM_PER_S = 100.0
TILT_SPEED_DEG_PER_S = 10.0
#: Stage travel, +/- (X, Y, Z um; alpha, beta deg).
STAGE_LIMITS = (2000.0, 2000.0, 500.0, 70.0, 30.0)
#: Hole indices of an aperture (0..4).
APERTURE_HOLE_MAX = 4
APERTURE_KIND_COUNT = 5  # 0 none, 1 CLA, 2 OLA, 3 HCA, 4 SAA
APERTURE_KIND_CLA = 1

#: Convergence semi-angle (mrad) [raw JEOL probe mode][alpha selector], column 4
#: covering every alpha >= 4. Rows: 0 TEM (parallel), 1 EDS, 2 NBD, 3 CBD. Realistic
#: values: parallel TEM/SAED ~0.01-0.1 mrad, NBD ~0.5-3 mrad, CBD ~3-15 mrad.
ALPHA_TABLE_MRAD = (
    (0.02, 0.03, 0.05, 0.08, 0.10),
    (1.00, 1.50, 2.50, 4.00, 5.00),
    (0.50, 1.00, 1.50, 2.00, 3.00),
    (3.00, 5.00, 8.00, 12.00, 15.00),
)
#: STEM probe convergence semi-angle (mrad) per alpha selector (clamped to the last entry).
STEM_ALPHA_TABLE_MRAD = (5.0, 10.0, 15.0, 22.0, 30.0)


def active_ladders(function_mode: int, tem_stem: int) -> tuple[Optional[Ladder], Optional[Ladder]]:
    """(magnifications, camera lengths in mm) live in one optics mode.

    ``None`` means the mode has no such ladder: no magnification in TEM
    diffraction, no camera length in TEM imaging (the Dummy refuses those sets).
    """
    if tem_stem == STEM:
        if function_mode == FM_DIFF:
            return NBD_STEM_MAGS, STEM_CAMERA_LENGTHS_MM
        return HR_STEM_MAGS, STEM_CAMERA_LENGTHS_MM
    if function_mode == FM_DIFF:
        return None, DIFF_CAMERA_LENGTHS_MM
    if function_mode == FM_LOWMAG:
        return LOWMAG_MAGS, None
    return MAG1_MAGS, None


def imaging_mode_for(function_mode: int, target_mag: float) -> int:
    """TEM imaging mode that covers ``target_mag`` (Dummy ``ImagingModeFor``).

    LowMAG's ceiling and MAG1's floor do not meet, so a request outside the
    active ladder crosses to the other TEM imaging mode instead of clamping.
    """
    if function_mode == FM_LOWMAG:
        return FM_MAG1 if target_mag > LOWMAG_MAGS[-1] else FM_LOWMAG
    return FM_LOWMAG if target_mag < MAG1_MAGS[0] else function_mode


def convergence_mrad(raw_jeol_probe: int, alpha: int, stem: bool = False) -> float:
    a = min(max(int(alpha), 0), 4)
    if stem:
        return STEM_ALPHA_TABLE_MRAD[a]
    return ALPHA_TABLE_MRAD[min(max(int(raw_jeol_probe), 0), 3)][a]


