"""The cross grating replica: exact period, metal shadowing, granular metal, rough edges."""

from __future__ import annotations

import math

import numpy as np

from de_twin.clock import ManualClock
from de_twin.specimen.materials import MaterialId
from de_twin.specimen.structures import CrossGratingStructure
from de_twin.twin import DigitalTwin

P = 1.0 / 2.160  # um


def _structure(**kw):
    return CrossGratingStructure(P, 20.0, 40.0, 0.5, shadow_azimuth_deg=0.0, rough=False, dirt=False, **kw)


def test_shadowing_coats_the_flanks_facing_the_source_and_shades_the_ground_behind():
    """Source along +x: a ridge spans x in [P/4, 3P/4] (line fraction 0.5). The ground just
    before its near edge (x < P/4) is in its shadow; its +x flank faces the source."""
    g = _structure()
    x = np.linspace(0.0, P, 400)
    u, v = x, np.full_like(x, 0.001)  # v in a well of the y-ridges: only the x-ridges matter
    h = g._height(u, v)
    metal = g._metal_nm(u, v, h, pixel_um=0.001)
    flat = 2.0 * math.sin(math.radians(30.0))
    ramp = 0.030
    # ground in front of the ridge, within its shadow (40 nm / tan 30 deg = 69 nm from the
    # full-height top, which starts half a ramp past the edge)
    before = (x > P / 4 - 0.045) & (x < P / 4 - ramp / 2 - 0.005)
    far_flank = (x > 3 * P / 4 - ramp / 2 + 0.003) & (x < 3 * P / 4 + ramp / 2 - 0.003)
    top = (x > P / 4 + 0.05) & (x < 3 * P / 4 - 0.05)
    assert np.all(metal[before] == 0.0), "shaded by the ridge"
    assert np.allclose(metal[far_flank], 2.0 * flat), "the flank facing the source catches more"
    assert np.allclose(metal[top], flat)


def test_the_metal_is_granular_crystalline_platinum_at_high_magnification():
    twin = DigitalTwin("Cross grating 2160 l/mm", camera="DESim", clock=ManualClock(), seed=1)
    twin.column.set("Magnification", 100000)
    gt = twin.ground_truth(as_arrays=True)
    pt = gt["material_id"] == int(MaterialId.PLATINUM)
    assert 0.05 < pt.mean() < 0.5
    grains = np.unique(gt["grain_id"][pt])
    assert grains.size > 50 and np.all(grains >= 0), "many islands, each its own grain"


def test_gold_shadowing_and_no_metal_are_options():
    from de_twin.specimen import Specimen, from_name

    for metal, mid in (("gold", MaterialId.GOLD), ("none", None)):
        config = from_name("Cross grating 2160 l/mm", seed=1)
        config.options["grating_metal"] = metal
        twin = DigitalTwin(Specimen(config), camera="DESim", clock=ManualClock(), seed=1)
        twin.column.set("Magnification", 100000)
        m = twin.ground_truth(as_arrays=True)["material_id"]
        if mid is None:
            assert not np.isin(m, [int(MaterialId.GOLD), int(MaterialId.PLATINUM)]).any()
        else:
            assert (m == int(mid)).any()


def test_the_islands_give_platinum_lattice_fringes():
    """At atomic resolution the shadowing's islands add a ring at a Pt lattice spacing to the
    image's power spectrum that the bare carbon replica does not have."""
    from de_twin.specimen import Specimen, from_name

    def spectrum(metal):
        config = from_name("Cross grating 2160 l/mm", seed=1)
        config.options["grating_metal"] = metal
        twin = DigitalTwin(Specimen(config), camera="DESim", clock=ManualClock(), seed=1)
        twin.column.set("Magnification", 400000)
        twin.column.set_defocus_um(-0.03)
        request = twin.request()
        f = twin.flux(request).astype(float)
        p_nm = twin.optics(request).specimen_pixel_nm
        n = f.shape[0]
        w = np.hanning(n)
        F = np.abs(np.fft.fftshift(np.fft.fft2((f - f.mean()) * w[:, None] * w[None, :]))) ** 2
        y, x = np.indices(F.shape)
        r = np.hypot(x - n // 2, y - n // 2) / (n * p_nm)
        return lambda lo, hi: float(F[(r >= lo) & (r < hi)].mean())

    pt, bare = spectrum("platinum"), spectrum("none")
    g220 = math.sqrt(8.0) / 0.3924  # 1/nm
    assert pt(g220 - 0.1, g220 + 0.1) > 3.0 * bare(g220 - 0.1, g220 + 0.1)
