"""Reference 4D-STEM reconstructions that check twin data actually reconstructs.

Test helpers only, not part of the de_twin package: the twin produces data, and
reconstruction belongs to the tools people use (numpy only).

* :func:`tcbf` - tilt-corrected bright field / parallax: per-pixel virtual BF images, their
  cross-correlation shifts fitted to the probe aberrations (C1, A1, scan rotation, optionally
  B2 / A2), shift-and-sum image.
* :func:`epie` - single-slice ePIE ptychography with a known (or refined) probe;
  :func:`probe_from_aberrations`, :func:`object_on_grid` and :func:`phase_correlation` help set
  it up from ``DigitalTwin.ground_truth_ptychography`` and score it.
"""

from .ptycho import (
    PtychoResult,
    align_phase,
    epie,
    object_on_grid,
    phase_correlation,
    probe_from_aberrations,
    probe_modes_on_grid,
)
from .tcbf import TcbfResult, bf_disk, tcbf

__all__ = [
    "PtychoResult",
    "TcbfResult",
    "align_phase",
    "bf_disk",
    "epie",
    "object_on_grid",
    "phase_correlation",
    "probe_from_aberrations",
    "probe_modes_on_grid",
    "tcbf",
]
