"""Calibration standards: Ted Pella's grating replicas and diffraction standards.

Each product is a preset in :data:`~de_twin.specimen.CALIBRATION_STANDARDS`
(``"Ted Pella 607-A - ..."``), with what the catalogue publishes about it in :data:`PRODUCTS`
(``numbers``: the values a calibration is checked against) and what the twin had to assume in
``model``. Families:

* **Grating replicas** (607, 607-A, 606, 603, 603-A, 677, 673) and **gold-shadowed latex**
  (628-B): shadowed carbon replicas (:mod:`.replica`). The gratings' relief, line profile and
  metal coat are fitted to Ted Pella's images (unit-cell averages and their FFT harmonics).
* **Diffraction**: evaporated aluminium (619, camera length) and molybdenum trioxide laths
  (625, image / diffraction rotation).

Ted Pella's other test specimens (latex spheres, ferritin, catalase, holey carbon, Pt/Ir,
gold islands, carbon black, gold foil, MAG*I*CAL, ...) are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..materials import MaterialId
from .common import StandardStructure
from .crystals import MoO3LathsStructure
from .films import PolycrystalFilmStructure
from .replica import (
    LatexSpheres,
    ShadowedReplicaStructure,
    WaffleGrating,
    carbon_equivalent,
    fbm,
    island_film,
)

__all__ = [
    "FAMILIES",
    "PRODUCTS",
    "LatexSpheres",
    "Product",
    "ShadowedReplicaStructure",
    "StandardStructure",
    "WaffleGrating",
    "build_structure",
    "carbon_equivalent",
    "fbm",
    "island_film",
    "mean_thickness_nm",
    "preset_name",
]


@dataclass(frozen=True)
class Product:
    """A Ted Pella calibration standard: what the catalogue says it is (``numbers``: published
    values) and what the twin assumes where it says nothing (``model``)."""

    number: str
    name: str
    family: str
    description: str
    use: str
    numbers: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)


PRODUCTS: dict[str, Product] = {}

#: Families, in the order a menu shows them.
FAMILIES = ("Grating replicas", "Diffraction")


def _p(number, name, family, description, use, numbers=None, model=None):
    PRODUCTS[number] = Product(number, name, family, description, use, dict(numbers or {}), dict(model or {}))


G, D = FAMILIES
_REPLICA = "Carbon replica with Au/Pd shadowing"
_p(
    "607",
    "2160 l/mm grating replica (waffle)",
    G,
    f"{_REPLICA} of a cross-line (waffle) diffraction grating, 2,160 lines/mm in both directions; "
    "G400 copper Gilder grid.",
    "Magnification calibration up to ~80-100,000x.",
    {"lines_per_mm": 2160.0, "period_nm": 462.9},
)
_p(
    "607-A",
    "2160 l/mm grating replica, special applications (waffle)",
    G,
    f"{_REPLICA}, 2,160 lines/mm waffle; the older style, with more defects and a less clear pattern "
    "than the current 607; G400 copper grid.",
    "Aberration corrector alignment and SerialEM.",
    {"lines_per_mm": 2160.0, "period_nm": 462.9},
)
_p(
    "606",
    "2160 l/mm grating replica (parallel lines)",
    G,
    f"{_REPLICA} of a parallel-line grating, 462.9 nm line spacing (2,160 lines/mm); G400 copper grid.",
    "Magnification calibration up to ~40-50,000x.",
    {"lines_per_mm": 2160.0, "period_nm": 462.9},
)
_p(
    "603",
    "2160 l/mm grating replica with 0.261 um latex spheres",
    G,
    f"{_REPLICA}, 2,160 lines/mm, with 0.261 um latex spheres; G400 copper Gilder grid.",
    "Two-in-one magnification calibration (the spheres a double check).",
    {"lines_per_mm": 2160.0, "period_nm": 462.9, "sphere_nm": 261.0},
)
_p(
    "603-A",
    "2160 l/mm grating replica with 0.261 um latex spheres, special applications",
    G,
    f"{_REPLICA}, the older-style 2,160 lines/mm grating with 0.261 um latex spheres; G400 copper grid.",
    "Aberration corrector alignment and SerialEM; magnification double check.",
    {"lines_per_mm": 2160.0, "period_nm": 462.9, "sphere_nm": 261.0},
)
_p(
    "677",
    "500 nm cross-line grating replica",
    G,
    f"{_REPLICA} of a cross-line grating with trench-type grooves, 500 nm pitch (2,000 lines/mm); "
    "G400 copper grid.",
    "Magnification calibration up to 100,000x.",
    {"lines_per_mm": 2000.0, "period_nm": 500.0},
)
_p(
    "673",
    "500 nm cross-line grating replica with 261 nm latex spheres",
    G,
    f"{_REPLICA}, 500 nm pitch (2,000 lines/mm), with 261 nm latex spheres; G400 copper grid.",
    "Two-in-one magnification calibration up to 150,000x.",
    {"lines_per_mm": 2000.0, "period_nm": 500.0, "sphere_nm": 261.0},
)
_p(
    "628-B",
    "Gold-shadowed latex",
    G,
    "Latex particles, 0.204 um diameter, shadowed with gold; 3.05 mm grid.",
    "A test object for STEM.",
    {"sphere_nm": 204.0},
)
for _n in ("607", "603", "606"):
    PRODUCTS[_n].model.update(
        {
            "relief_nm": 90.0,
            "line_fraction": 0.24,
            "walls_nm": (8.0, 75.0),
            "metal": "gold",
            "metal_nm": 5.0,
            "shadow_elevation_deg": 20.0,
            "fitted": "Ted Pella 607 image",
        }
    )
for _n in ("677", "673"):
    PRODUCTS[_n].model.update(
        {
            "relief_nm": 60.0,
            "line_fraction": 0.12,
            "walls_nm": (70.0, 10.0),
            "metal": "gold",
            "metal_nm": 12.0,
            "shadow_elevation_deg": 20.0,
            "fitted": "Ted Pella 673 image",
        }
    )
for _n in ("603", "603-A", "673"):
    PRODUCTS[_n].model["sphere_nm"] = 262.0  # the tech notes' value; applied after the shadowing

_p(
    "619",
    "Diffraction standard, evaporated aluminum",
    D,
    "Evaporated aluminum, ~31 nm thick; very small crystallites give ring patterns; G400 copper Gilder grid.",
    "Camera length calibration.",
    {
        "thickness_nm": 31.0,
        "a_nm": 0.40494,
        "d_nm": {"111": 0.2338, "200": 0.2024, "220": 0.1431, "311": 0.1221, "222": 0.1169, "400": 0.10124},
    },
    {"grain_nm": 15.0, "texture": "random"},
)
_p(
    "625",
    "Image rotation standard, molybdenum trioxide",
    D,
    "Molybdenum trioxide crystals; G400 copper Gilder grid.",
    "Rotation between the selected-area image and its diffraction pattern.",
    {"lath_axis": "[001] (Pbnm) = [010] (Pnma), 0.3697 nm"},
    {"film_nm": 15.0, "lath_um": 2.5, "lath_thickness_nm": 60.0, "face": "(010) Pbnm"},
)
del G, D


def build_structure(number: str) -> StandardStructure:
    """The structure of product *number* (see :data:`PRODUCTS`)."""
    P2160 = 1.0 / 2.160
    au = {"metal": MaterialId.GOLD}  # Au/Pd: gold stands in for the alloy
    old, new = (
        {"edge_nm": 10.0, "edge_corr_nm": 60.0, "wavy_nm": 40.0, "wavy_um": 1.2},
        {"edge_nm": 3.0, "edge_corr_nm": 80.0, "wavy_nm": 6.0, "wavy_um": 3.0},
    )
    # The current gratings (607, 603, 606, 677, 673): square wells between narrow raised lines,
    # shadowed along the diagonal at a low angle, so each well floor has a bare L-shaped shadow
    # and the rest a grainy coat. The lines are sawtooth (a steep wall and a gentle slope, as
    # on a ruled master). Relief, line profile and coat fitted to Ted Pella's images (unit-cell
    # averages and their harmonics). The older style (607-A, 603-A): a softer, rougher,
    # wavier embossed relief.
    if number in ("607", "603"):
        g = WaffleGrating(P2160, depth_nm=90.0, line_fraction=0.24, ramp_nm=8.0, back_ramp_nm=75.0, **new)
        sph = LatexSpheres(262.0, 0.15) if number == "603" else None
        return ShadowedReplicaStructure(
            g,
            rough_nm=2.5,
            rough_um=0.1,
            crumple_nm=6.0,
            metal_nm=5.0,
            elevation_deg=20.0,
            spheres=sph,
            spheres_shadowed=False,
            **au,
        )
    if number in ("607-A", "603-A"):
        g = WaffleGrating(P2160, depth_nm=25.0, line_fraction=0.18, ramp_nm=45.0, **old)
        sph = LatexSpheres(262.0, 0.15) if number == "603-A" else None
        return ShadowedReplicaStructure(
            g, rough_nm=7.0, rough_um=0.1, elevation_deg=20.0, spheres=sph, spheres_shadowed=False, **au
        )
    if number == "606":
        return ShadowedReplicaStructure(
            WaffleGrating(
                P2160, depth_nm=90.0, line_fraction=0.24, ramp_nm=8.0, back_ramp_nm=75.0, both=False, **new
            ),
            rough_nm=2.5,
            rough_um=0.1,
            crumple_nm=6.0,
            metal_nm=5.0,
            elevation_deg=20.0,
            **au,
        )
    if number in ("677", "673"):
        g = WaffleGrating(0.5, depth_nm=60.0, line_fraction=0.12, ramp_nm=70.0, back_ramp_nm=10.0, **new)
        sph = LatexSpheres(262.0, 0.15) if number == "673" else None
        return ShadowedReplicaStructure(
            g,
            rough_nm=2.0,
            rough_um=0.1,
            crumple_nm=6.0,
            metal_nm=12.0,
            elevation_deg=20.0,
            spheres=sph,
            spheres_shadowed=False,
            **au,
        )
    if number == "628-B":
        # a heavy, low-angle gold shadow: shadows ~3.5 diameters long
        return ShadowedReplicaStructure(
            None,
            base_nm=15.0,
            rough_nm=1.5,
            crumple_nm=4.0,
            metal_nm=15.0,
            elevation_deg=16.0,
            spheres=LatexSpheres(204.0, 0.8, 0.02),
            island_nm=10.0,
            coverage=0.7,
            **au,
        )
    if number == "619":
        return PolycrystalFilmStructure(MaterialId.ALUMINUM, 31.0, 15.0)
    if number == "625":
        return MoO3LathsStructure()
    raise KeyError(f"no structure for product {number!r}")


def mean_thickness_nm(number: str) -> float:
    """Area-mean carbon-equivalent thickness of product *number* (its low-magnification look)."""
    return build_structure(number).mean_nm()


def preset_name(number: str) -> str:
    return f"Ted Pella {number} - {PRODUCTS[number].name}"
