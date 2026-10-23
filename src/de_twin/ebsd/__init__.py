"""Electron backscatter diffraction (EBSD) in an SEM, with a Direct Electron camera as the detector.

* :mod:`.geometry`: :class:`EbsdGeometry` (sample / detector tilt, working distance, pattern
  centre in the Bruker convention, probe current, ...), the sample and detector frames, and
  how the pattern centre follows the beam.
* :mod:`.master`: kinematic Kikuchi master patterns per phase and energy, cached.
* :mod:`.kernels`: numba kernels (master sum, gnomonic projection).
* :mod:`.render`: :class:`EbsdRenderer`, the per-scan-point patterns and ground truth.
* :mod:`.io`: user master patterns (EMsoft / kikuchipy HDF5, NumPy) and the EMsoft writer.

Put a twin in EBSD mode with ``DigitalTwin.configure_ebsd`` (or ``OpticsConfig.ebsd``) and a
bulk specimen (:data:`de_twin.specimen.bulk.BULK_PRESETS`); patterns then come out of
``frames`` / ``flux`` / ``datacube`` and every face, like 4D-STEM.
"""

from .geometry import PC_CONVENTION, EbsdGeometry, convert_pc, detector_position, pc_pixels
from .io import load_master_pattern, write_emsoft
from .master import MASTERS, MasterPattern, build_master, master_for
from .render import EbsdRenderer

__all__ = [
    "EbsdGeometry",
    "EbsdRenderer",
    "MASTERS",
    "MasterPattern",
    "PC_CONVENTION",
    "build_master",
    "convert_pc",
    "detector_position",
    "load_master_pattern",
    "master_for",
    "pc_pixels",
    "write_emsoft",
]
