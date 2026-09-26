import pytest

from de_twin.specimen import LEGACY_ALIASES, PATTERNS, PRESETS, SpecimenConfig, SpecimenOptions, from_name


def test_thirteen_presets_and_six_patterns():
    assert len(PRESETS) == 13
    assert {"Sparse Au on holey C", "Dense Au on holey C", "Au clusters on lacey C", "Custom",
            "Apoferritin in ice", "Negative stain on carbon", "Au thin film 20 nm", "Al thin film 100 nm",
            "Steel lamella with carbides", "PN junction lamella", "Strained inclusion lamella",
            "Droplet crystallization", "Cross grating 2160 l/mm"} == set(PRESETS)
    virtual = [n for n in PATTERNS if n.startswith("Virtual Specimen - ")]
    assert len(virtual) == 6
    assert set(LEGACY_ALIASES) <= set(PATTERNS) and len(LEGACY_ALIASES) == 4


def test_preset_values_match_apply_preset():
    o = PRESETS["Sparse Au on holey C"].resolved_options()
    assert (o.particle_density_per_um2, o.particle_median_diameter_nm, o.cluster_radius_um) == (6.0, 22.0, 0.8)
    assert PRESETS["Au clusters on lacey C"].film == "lacey"
    assert PRESETS["Negative stain on carbon"].resolved_options().negative_stain_fraction == pytest.approx(0.9)
    assert PRESETS["PN junction lamella"].resolved_options().fib_post_sample == "pn_junction"
    assert PRESETS["Droplet crystallization"].holder == "insitu_heating_chip"


def test_from_name_lookup():
    a = from_name("dense au on holey c")
    assert a.holder == "mesh_grid" and a.preparation == "nanoparticles"
    b = from_name("FIB Liftout", seed=7)
    assert b.holder == "fib_liftout" and b.seed == 7
    pn = from_name("PNJunction-Scan")
    assert pn.options["fib_post_sample"] == "pn_junction" and pn.options["legacy_force_descan"]
    # returns copies
    a.options["particle_density_per_um2"] = 1.0
    assert "particle_density_per_um2" not in from_name("Custom").options
    with pytest.raises(KeyError):
        from_name("no such specimen")


def test_options_validation_and_clamping():
    o = SpecimenOptions.from_dict({"particle_density_per_um2": 1e9, "fib_post_count": 9, "curtain_depth": -1})
    assert o.particle_density_per_um2 == 5000.0 and o.fib_post_count == 5 and o.curtain_depth == 0.0
    with pytest.raises(KeyError):
        SpecimenOptions.from_dict({"chip_window_grid": 3})  # dropped dead option
    with pytest.raises(ValueError):
        SpecimenOptions.from_dict({"fib_post_sample": "banana"})
    assert "lod_threshold_px" in SpecimenOptions.describe()


def test_auto_film_resolution():
    assert SpecimenConfig(holder="mesh_grid", preparation="proteins").resolved_film() == "vitreous_ice"
    assert SpecimenConfig(holder="mesh_grid", preparation="thin_film").resolved_film() == "continuous"
    assert SpecimenConfig(holder="fib_liftout", preparation="bulk").resolved_film() == "none"
    assert SpecimenConfig(holder="mesh_grid", film="Lacey").resolved_film() == "lacey"
    with pytest.raises(ValueError):
        SpecimenConfig(holder="teapot")


def test_set_options_regenerates_only_when_needed():
    from de_twin.specimen import Specimen
    s = Specimen(from_name("Dense Au on holey C"))
    scene = s.scene
    s.set_options(lod_threshold_px=2.0, crystallization_onset_c=300.0)
    assert s.scene is scene and s.options.lod_threshold_px == 2.0
    s.set_options(particle_density_per_um2=1.0)
    assert s.scene is not scene and s.config.options["particle_density_per_um2"] == 1.0
