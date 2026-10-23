"""The virtual specimen: holder + preparation + time evolution, rasterised into FieldMaps.

Port of DE-Server's VirtualSpecimen scene model (Geometry, Holder, Preparation, Structure,
Scene, FieldMap rasterisation, options/presets/patterns). See ``docs/architecture.md``.
"""

from .fieldmap import FieldMap, GrainTable, ViewWindow
from .materials import MaterialId
from .model import Specimen, advance_thermal_budget, crystalline_fraction
from .options import (
    CALIBRATION_STANDARDS,
    LEGACY_ALIASES,
    PATTERNS,
    PRESETS,
    SpecimenConfig,
    SpecimenOptions,
    from_name,
)
from .scene import Feature

__all__ = [
    "CALIBRATION_STANDARDS",
    "LEGACY_ALIASES",
    "PATTERNS",
    "PRESETS",
    "Feature",
    "FieldMap",
    "GrainTable",
    "MaterialId",
    "Specimen",
    "SpecimenConfig",
    "SpecimenOptions",
    "ViewWindow",
    "advance_thermal_budget",
    "crystalline_fraction",
    "from_name",
]
