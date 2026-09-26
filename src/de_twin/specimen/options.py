"""Specimen options, presets and test patterns (port of ``VirtualSpecimenProps`` + pattern table).

Every *specimen-related* ``Simulator Virtual Specimen ...`` property of DE-Server's
VirtualSpecimen module appears here as a snake_case field of :class:`SpecimenOptions`, with the
same default and range. Optics/render/detector knobs (pixel-size fallbacks, flips, SA aperture,
beam stop, probe, diffraction cache, render threads, ...) belong to other subpackages and are
not repeated here.

Dropped options (dead or superseded in the C++ as well):

* ``Chip Window Grid`` - the authored in-situ SVG has exactly ten openings; the N x N window grid
  of the old parametric chip is never read.
* ``Support Film`` / ``Seed`` - these are :attr:`SpecimenConfig.film` / :attr:`SpecimenConfig.seed`.
* ``Preset`` - a preset is a whole :class:`SpecimenConfig` (see :data:`PRESETS`); the C++ used
  ``options.preset`` only to smuggle the thin-film material, which is now the real option
  :attr:`SpecimenOptions.thin_film_material`.
* ``Temperature (C)`` is kept as a fallback only; the live value comes from
  ``HolderState.temperature_c`` passed to :meth:`Specimen.update`.
* ``Nearest Cluster`` / ``Nearest Feature`` / ``Feature Table`` (read-only publishers) are the
  methods :meth:`Specimen.nearest_feature` / :meth:`Specimen.features`.
* ``Legacy Scan Patterns`` (gate) - legacy names are always accepted by :func:`from_name`.

The C++ global option snapshot (``CurrentVirtualSpecimenOptions``), the Structure constructor
side tables and the primitive-field smuggling of ``TimeEvolvingStructure`` are gone: every
consumer receives a resolved :class:`SpecimenOptions` (or real constructor arguments).
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Optional

HOLDERS = ("mesh_grid", "waffle_grid", "fib_liftout", "insitu_heating_chip")
PREPARATIONS = ("nanoparticles", "proteins", "thin_film", "bulk", "time_evolving", "cross_grating")
FILMS = ("holey", "lacey", "continuous", "none", "vitreous_ice", "auto")
FIB_POST_SAMPLES = ("auto", "precipitate_matrix", "pn_junction", "strained_inclusion",
                    "polycrystal", "multilayer")
THIN_FILM_MATERIALS = ("auto", "gold", "aluminum", "copper", "iron", "silicon", "platinum")


def _opt(default, lo=None, hi=None, doc="", choices=None):
    return field(default=default, metadata={"lo": lo, "hi": hi, "doc": doc, "choices": choices})


@dataclass
class SpecimenOptions:
    """Every specimen knob of the VirtualSpecimen module (defaults and ranges as in the C++)."""

    # -- level of detail -------------------------------------------------------------------
    lod_threshold_px: float = _opt(0.5, 0.0, 16.0, "Simulator LOD Threshold (pixels): primitives "
                                   "smaller than this many raster pixels across are culled.")

    # -- support film ------------------------------------------------------------------------
    film_thickness_nm: float = _opt(0.0, 0.0, 1000.0, "Support-film thickness; 0 keeps the film "
                                    "type's default (12 continuous, 20 holey, 15 lacey, ice = "
                                    "ice_thickness_nm).")

    # -- nanoparticles --------------------------------------------------------------------------
    particle_density_per_um2: float = _opt(24.0, 0.0, 5000.0, "Particles per um^2 of placement area.")
    particle_median_diameter_nm: float = _opt(24.0, 0.5, 1000.0, "Median particle diameter (log-normal).")
    particle_size_sigma_log: float = _opt(0.8, 0.0, 2.0, "Std-dev of ln(diameter).")
    clustered_fraction: float = _opt(0.35, 0.0, 1.0, "Fraction of particles in agglomerates.")
    cluster_density_per_um2: float = _opt(0.06, 0.0, 100.0, "Agglomerate centres per um^2.")
    cluster_radius_um: float = _opt(0.45, 0.01, 50.0, "Gaussian sigma of an agglomerate.")

    # -- proteins -----------------------------------------------------------------------------
    protein_diameter_nm: float = _opt(12.0, 2.0, 100.0, "Median protein blob diameter.")
    protein_density_per_um2: float = _opt(2000.0, 0.0, 50000.0, "Protein blobs per um^2.")
    ice_thickness_nm: float = _opt(60.0, 0.0, 1000.0, "Vitreous-ice layer thickness.")
    negative_stain_fraction: float = _opt(0.0, 0.0, 1.0, "Fraction of blobs with a Pt stain halo.")
    dose_fading_per_step: float = _opt(0.0, 0.0, 1.0, "Protein contrast lost per time step "
                                       "(fade = exp(-k * time_index)).")
    protein_crystal_patch_fraction: float = _opt(0.0, 0.0, 1.0, "Fraction of the area in 2D-crystal "
                                                 "patches (C++ ProteinParams field without a "
                                                 "property; exposed here).")

    # -- deposited thin film -----------------------------------------------------------------
    grain_size_nm: float = _opt(150.0, 2.0, 5000.0, "Median grain size (fine lattice) of a deposited film.")
    deposited_film_thickness_nm: float = _opt(40.0, 1.0, 1000.0, "Nominal deposited-film thickness.")
    grating_lines_per_mm: float = _opt(2160.0, 10.0, 10000.0, "Cross grating: lines per mm (period 1/that).")
    grating_base_nm: float = _opt(20.0, 0.0, 500.0, "Cross grating: carbon film under the ridges, nm.")
    grating_depth_nm: float = _opt(40.0, 0.0, 500.0, "Cross grating: ridge height, nm.")
    pinhole_fraction: float = _opt(0.03, 0.0, 1.0, "Fraction of the film area that is a pinhole.")
    crack_density_per_um: float = _opt(0.02, 0.0, 5.0, "Film cracks per micrometre.")
    thin_film_material: str = _opt("auto", doc="Deposited-film material; 'auto' = gold on a waffle "
                                   "grid, aluminium otherwise (replaces the C++ preset check).",
                                   choices=THIN_FILM_MATERIALS)
    thin_film_gradient_per_mm: float = _opt(0.4, -10.0, 10.0, "Fractional thickness change per mm "
                                            "across the grid (C++ ThinFilmParams constant).")

    # -- FIB liftout ---------------------------------------------------------------------------
    fib_post_count: int = _opt(4, 3, 5, "Number of FIB posts (authored SVG flags used).")
    fib_post_sample: str = _opt("auto", doc="Force every post's BulkSampleKind; 'auto' keeps the "
                                "seeded distinct permutation.", choices=FIB_POST_SAMPLES)
    lamella_thickness_nm: float = _opt(100.0, 20.0, 1000.0, "Nominal lamella thickness (+/-20% per lamella).")
    curtain_depth: float = _opt(0.35, 0.0, 1.0, "FIB curtaining depth (fraction of mean thickness).")
    junction_descan_x_px: float = _opt(2.0, -512.0, 512.0, "PN-junction descan step, x (detector px).")
    junction_descan_y_px: float = _opt(4.0, -512.0, 512.0, "PN-junction descan step, y (detector px).")
    junction_width_frac: float = _opt(0.60, 0.01, 2.0, "Junction smoothing width (fraction of lamella height).")
    strain_magnitude: float = _opt(0.02, 0.0, 0.5, "Strained-inclusion magnitude m -> (1+m, m, 1-m/2).")

    # -- time evolution (nucleation + growth, TimeEvolvingPreparation) ---------------------------
    nucleation_sites_per_window: int = _opt(8, 0, 1000, "Nucleation sites per placement area.")
    growth_rate_nm_per_step: float = _opt(100.0, 0.0, 100000.0, "Growth radius r = rate * sqrt(t - t_k).")
    drift_per_step_x_nm: float = _opt(0.0, -100000.0, 100000.0, "Field drift per time step, x.")
    drift_per_step_y_nm: float = _opt(0.0, -100000.0, 100000.0, "Field drift per time step, y.")
    time_step_s: float = _opt(1.0, 1e-6, 1e6, "Seconds of scene time per time index step (the C++ "
                              "used whole seconds of acquisition time, or the scan repeat).")

    # -- in-situ crystallisation (thermal budget) ------------------------------------------------
    in_situ_crystallization: Optional[bool] = _opt(None, doc="Per-grain glassy->crystalline model; "
                                                   "None = on for the in-situ heating chip only.")
    nucleation_start_s: float = _opt(25.0, 0.0, 3600.0, "Thermal budget at which the first grain nucleates.")
    nucleation_end_s: float = _opt(75.0, 0.0, 3600.0, "Thermal budget at which the last grain nucleates.")
    crystal_growth_s: float = _opt(6.0, 0.1, 600.0, "Budget from nucleation to fully crystalline.")
    specimen_temperature_c: float = _opt(25.0, -273.0, 1500.0, "Fallback temperature when update() "
                                         "gets no holder.")
    crystallization_onset_c: float = _opt(400.0, -273.0, 1500.0, "Budget advances at and above this.")
    melting_temperature_c: float = _opt(800.0, -273.0, 3000.0, "At and above this the specimen re-amorphises.")

    # -- hints for other subpackages (carried by presets/aliases, not used by the specimen) -------
    dose_rate_counts_per_s: float = _opt(4000.0, 1.0, 1e6, "Preset dose-rate hint for the renderer.")
    start_dose_e_per_px_s: float = _opt(0.0, 0.0, 1e6, "Dose rate a twin starts the beam at, in "
                                        "e-/detector px/s (it solves the Intensity for it). 0 is the "
                                        "detector's own safe rate: well inside what a pixel holds per "
                                        "frame at 40 fps, so frames never saturate.")
    start_defocus_um: float = _opt(0.0, -20.0, 20.0, "Defocus a twin starts at: a thin phase object "
                                   "(proteins in ice) shows no contrast in focus.")
    legacy_force_descan: bool = _opt(False, doc="PNJunction-Scan quirk: renderer should force descan on.")

    # ------------------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, values: Optional[dict] = None) -> "SpecimenOptions":
        """Build from a (possibly partial) dict; unknown keys raise, numbers are clamped to range."""
        opts = cls()
        for k, v in (values or {}).items():
            opts.set(k, v)
        return opts

    def set(self, name: str, value: Any) -> None:
        meta = {f.name: f for f in fields(self)}
        if name not in meta:
            raise KeyError(f"unknown specimen option {name!r}")
        f = meta[name]
        md = f.metadata
        if md.get("choices") is not None:
            value = str(value).lower().replace(" ", "_")
            if value not in md["choices"]:
                raise ValueError(f"{name} must be one of {md['choices']}, got {value!r}")
        elif isinstance(f.default, bool) or f.default is None:
            value = None if value is None else bool(value)
        elif isinstance(f.default, int):
            value = int(round(float(value)))
        else:
            value = float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            lo, hi = md.get("lo"), md.get("hi")
            if lo is not None:
                value = max(lo, value)
            if hi is not None:
                value = min(hi, value)
        setattr(self, name, value)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def replace(self, **kw) -> "SpecimenOptions":
        out = copy.deepcopy(self)
        for k, v in kw.items():
            out.set(k, v)
        return out

    @staticmethod
    def describe() -> dict[str, dict]:
        """Name -> {default, lo, hi, doc, choices} for every option."""
        return {f.name: {"default": f.default, **f.metadata} for f in fields(SpecimenOptions)}


# Options whose change does NOT require regenerating the world (C++ RequiresSceneRegeneration
# is false for these); all others rebuild the scene.
NON_REGENERATING = frozenset({
    "lod_threshold_px", "junction_descan_x_px", "junction_descan_y_px", "junction_width_frac",
    "strain_magnitude", "drift_per_step_x_nm", "drift_per_step_y_nm", "time_step_s",
    "in_situ_crystallization", "nucleation_start_s", "nucleation_end_s", "crystal_growth_s",
    "specimen_temperature_c", "crystallization_onset_c", "melting_temperature_c",
    "dose_rate_counts_per_s", "legacy_force_descan", "start_dose_e_per_px_s", "start_defocus_um",
})
# Of those, the ones baked into populated areas (C++ clears the population cache for them).
CLEARS_POPULATION = frozenset({
    "junction_descan_x_px", "junction_descan_y_px", "junction_width_frac", "strain_magnitude",
    "drift_per_step_x_nm", "drift_per_step_y_nm",
})


@dataclass
class SpecimenConfig:
    """What to build: seed, holder, preparation, support film and option overrides."""

    seed: int = 42
    holder: str = "mesh_grid"  # mesh_grid | waffle_grid | fib_liftout | insitu_heating_chip
    preparation: str = "nanoparticles"  # nanoparticles | proteins | thin_film | bulk | time_evolving
    film: str = "auto"  # holey | lacey | continuous | none | vitreous_ice | auto
    options: dict = field(default_factory=dict)  # SpecimenOptions overrides (snake_case)
    name: str = ""  # preset / pattern name it came from (informational)

    def __post_init__(self):
        self.holder = _norm(self.holder)
        self.preparation = _norm(self.preparation)
        self.film = _norm(self.film)
        if self.holder not in HOLDERS:
            raise ValueError(f"holder must be one of {HOLDERS}, got {self.holder!r}")
        if self.preparation not in PREPARATIONS:
            raise ValueError(f"preparation must be one of {PREPARATIONS}, got {self.preparation!r}")
        if self.film not in FILMS:
            raise ValueError(f"film must be one of {FILMS}, got {self.film!r}")

    def resolved_options(self) -> SpecimenOptions:
        return SpecimenOptions.from_dict(self.options)

    def resolved_film(self) -> str:
        """The film after resolving 'auto' to the pattern's default film (C++ defaultFilm)."""
        if self.film != "auto":
            return self.film
        return DEFAULT_FILM.get((self.holder, self.preparation), "holey")

    def with_options(self, **kw) -> "SpecimenConfig":
        out = copy.deepcopy(self)
        out.options.update(kw)
        return out

    def copy(self) -> "SpecimenConfig":
        return copy.deepcopy(self)


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


# Pattern table (VirtualSpecimen.cpp PatternTable): holder, preparation, default film.
DEFAULT_FILM = {
    ("mesh_grid", "nanoparticles"): "holey",
    ("mesh_grid", "proteins"): "vitreous_ice",
    ("mesh_grid", "thin_film"): "continuous",
    ("waffle_grid", "thin_film"): "continuous",
    ("fib_liftout", "bulk"): "none",
    ("insitu_heating_chip", "nanoparticles"): "continuous",
    ("insitu_heating_chip", "time_evolving"): "continuous",
}

_PATTERN_TABLE = (
    ("Virtual Specimen - TEM Grid (Nanoparticles)", "mesh_grid", "nanoparticles"),
    ("Virtual Specimen - TEM Grid (Proteins)", "mesh_grid", "proteins"),
    ("Virtual Specimen - TEM Grid (Thin Film)", "mesh_grid", "thin_film"),
    ("Virtual Specimen - Waffle Grid (Thin Film)", "waffle_grid", "thin_film"),
    ("Virtual Specimen - FIB Liftout", "fib_liftout", "bulk"),
    ("Virtual Specimen - In Situ Heating (Nanoparticles)", "insitu_heating_chip", "nanoparticles"),
)

# ApplyPreset (VirtualSpecimenProps.cpp): option overrides + support film per preset, applied on
# top of the defaults. The (holder, preparation) pairing is the pattern each preset was designed
# for (PHASE5_DESIGN 7.1 "intended preset").
_PRESET_TABLE: dict[str, tuple[str, str, str, dict]] = {
    "Sparse Au on holey C": ("mesh_grid", "nanoparticles", "holey", dict(
        film_thickness_nm=0.0, particle_density_per_um2=6.0, particle_median_diameter_nm=22.0,
        particle_size_sigma_log=0.75, clustered_fraction=0.35, cluster_density_per_um2=0.02,
        cluster_radius_um=0.80, dose_rate_counts_per_s=4000.0)),
    "Dense Au on holey C": ("mesh_grid", "nanoparticles", "holey", dict(
        film_thickness_nm=0.0, particle_density_per_um2=24.0, particle_median_diameter_nm=24.0,
        particle_size_sigma_log=0.80, clustered_fraction=0.35, cluster_density_per_um2=0.06,
        cluster_radius_um=0.45, dose_rate_counts_per_s=4000.0)),
    "Au clusters on lacey C": ("mesh_grid", "nanoparticles", "lacey", dict(
        film_thickness_nm=0.0, particle_density_per_um2=14.0, particle_median_diameter_nm=16.0,
        particle_size_sigma_log=0.70, clustered_fraction=0.85, cluster_density_per_um2=0.05,
        cluster_radius_um=0.25, dose_rate_counts_per_s=6000.0)),
    "Custom": ("mesh_grid", "nanoparticles", "auto", {}),
    "Apoferritin in ice": ("mesh_grid", "proteins", "vitreous_ice", dict(
        film_thickness_nm=0.0, ice_thickness_nm=60.0, protein_diameter_nm=12.0,
        protein_density_per_um2=2000.0, negative_stain_fraction=0.0, dose_fading_per_step=0.0,
        dose_rate_counts_per_s=6000.0, start_defocus_um=-1.5)),
    "Negative stain on carbon": ("mesh_grid", "proteins", "continuous", dict(
        film_thickness_nm=12.0, ice_thickness_nm=60.0, protein_diameter_nm=18.0,
        protein_density_per_um2=900.0, negative_stain_fraction=0.9, dose_fading_per_step=0.0,
        dose_rate_counts_per_s=4000.0)),
    "Cross grating 2160 l/mm": ("mesh_grid", "cross_grating", "none", dict()),
    "Au thin film 20 nm": ("mesh_grid", "thin_film", "continuous", dict(
        deposited_film_thickness_nm=20.0, grain_size_nm=30.0, pinhole_fraction=0.02,
        crack_density_per_um=0.0, thin_film_material="gold")),
    "Al thin film 100 nm": ("waffle_grid", "thin_film", "continuous", dict(
        deposited_film_thickness_nm=100.0, grain_size_nm=250.0, pinhole_fraction=0.05,
        crack_density_per_um=0.05, thin_film_material="aluminum")),
    "Steel lamella with carbides": ("fib_liftout", "bulk", "none", dict(
        fib_post_sample="precipitate_matrix", lamella_thickness_nm=100.0)),
    "PN junction lamella": ("fib_liftout", "bulk", "none", dict(
        fib_post_sample="pn_junction", lamella_thickness_nm=100.0, junction_descan_x_px=2.0,
        junction_descan_y_px=4.0, junction_width_frac=0.60)),
    "Strained inclusion lamella": ("fib_liftout", "bulk", "none", dict(
        fib_post_sample="strained_inclusion", lamella_thickness_nm=100.0, strain_magnitude=0.02)),
    "Droplet crystallization": ("insitu_heating_chip", "nanoparticles", "auto", dict(
        particle_density_per_um2=60.0, nucleation_sites_per_window=8, growth_rate_nm_per_step=100.0,
        drift_per_step_x_nm=0.0, drift_per_step_y_nm=0.0)),
}

# LegacyAlias.cpp: legacy name -> (pattern, preset, forced post sample, force descan).
_LEGACY_ALIASES = (
    ("NanoCrystals-Scan", "Virtual Specimen - TEM Grid (Nanoparticles)", "Dense Au on holey C",
     None, False),
    ("Strain-Scan", "Virtual Specimen - FIB Liftout", "Strained inclusion lamella",
     "strained_inclusion", False),
    ("PNJunction-Scan", "Virtual Specimen - FIB Liftout", "PN junction lamella", "pn_junction", True),
    ("DropletCrystallization-Scan", "Virtual Specimen - In Situ Heating (Nanoparticles)",
     "Droplet crystallization", None, False),
)


def _build_presets() -> dict[str, SpecimenConfig]:
    out = {}
    for name, (holder, prep, film, opts) in _PRESET_TABLE.items():
        SpecimenOptions.from_dict(opts)  # validate
        out[name] = SpecimenConfig(seed=42, holder=holder, preparation=prep, film=film,
                                   options=dict(opts), name=name)
    return out


def _build_patterns() -> dict[str, SpecimenConfig]:
    out = {}
    for name, holder, prep in _PATTERN_TABLE:
        out[name] = SpecimenConfig(seed=42, holder=holder, preparation=prep, film="auto",
                                   options={}, name=name)
    presets = _build_presets()
    for legacy, pattern, preset, post, descan in _LEGACY_ALIASES:
        base = out[pattern]
        opts = dict(presets[preset].options)
        if post is not None:
            opts["fib_post_sample"] = post
        if descan:
            opts["legacy_force_descan"] = True
        film = presets[preset].film if presets[preset].film != "auto" else "auto"
        out[legacy] = SpecimenConfig(seed=42, holder=base.holder, preparation=base.preparation,
                                     film=film, options=opts, name=legacy)
    return out


#: The 12 VirtualSpecimen presets (ApplyPreset), each paired with the pattern it was designed for.
PRESETS: dict[str, SpecimenConfig] = _build_presets()
#: The 6 "Virtual Specimen - ..." test patterns plus the 4 legacy ``*-Scan`` aliases.
PATTERNS: dict[str, SpecimenConfig] = _build_patterns()
LEGACY_ALIASES: tuple[str, ...] = tuple(a[0] for a in _LEGACY_ALIASES)


def from_name(name: str, seed: Optional[int] = None) -> SpecimenConfig:
    """Look up a preset, test pattern or legacy alias by name (case-insensitive).

    Pattern names may omit the ``"Virtual Specimen - "`` prefix. Returns a fresh copy.
    """
    key = str(name).strip().lower()
    for table in (PRESETS, PATTERNS):
        for k, cfg in table.items():
            kl = k.lower()
            if key == kl or (kl.startswith("virtual specimen - ") and key == kl[len("virtual specimen - "):]):
                out = cfg.copy()
                if seed is not None:
                    out.seed = int(seed)
                return out
    raise KeyError(f"unknown specimen preset/pattern {name!r}; known: "
                   f"{sorted(PRESETS) + sorted(PATTERNS)}")
