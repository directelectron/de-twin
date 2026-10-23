"""A twin starts where an operator would: over the specimen, with a beam the detector takes.

The stage origin is often a grid bar, a chip frame or bare film, and the default spread beam
used to leave some presets black; every preset has to show structure in an ordinary exposure.
"""

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.column import Column
from de_twin.specimen import options
from de_twin.twin import DigitalTwin

#: Presets whose contrast is subtle by nature at the start magnification (a 100 nm
#: silicon lamella's junction or strain field, a phase object in ice): they must still show
#: repeatable structure, just less of it.
SUBTLE = {"Apoferritin in ice", "PN junction lamella", "Strained inclusion lamella", "Al thin film 100 nm"}


def _repeat_corr(twin, seconds=0.5):
    request = twin.request(frame_time_s=0.025, total_frames=round(seconds / 0.025))
    a = twin.processor.acquire(request, units="electrons")
    b = twin.processor.acquire(request, units="electrons")
    a0, b0 = a - a.mean(), b - b.mean()
    return float((a0 * b0).mean() / (a0.std() * b0.std() + 1e-12))


@pytest.mark.parametrize("name", sorted(options.PRESETS))
def test_every_preset_starts_over_specimen_with_structure(name):
    twin = DigitalTwin(name, camera="DESim", clock=ManualClock(), seed=3)
    flux = twin.flux(twin.request())
    assert flux.mean() > 1.0, "the beam reaches the camera through the specimen"
    # Two independent exposures agree where there is structure; noise does not.
    assert _repeat_corr(twin) > (0.15 if name in SUBTLE else 0.45)


@pytest.mark.parametrize("camera", ["DESim", "DE16", "Celeritas"])
def test_the_start_beam_does_not_saturate_a_frame(camera):
    twin = DigitalTwin("Dense Au on holey C", camera=camera, clock=ManualClock(), seed=3)
    raw = twin.raw_frame(0.025)
    good = ~twin.detector.bad_pixel_mask
    # Hot pixels saturate on their own; the beam must not.
    assert np.percentile(raw[good], 99.5) < twin.detector.model.saturation_adu


def test_cryo_starts_underfocus():
    twin = DigitalTwin("Apoferritin in ice", camera="DESim", clock=ManualClock(), seed=3)
    assert twin.column.state().defocus_um == pytest.approx(-1.5)


def test_a_new_specimen_is_started_on_too():
    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(), seed=3)
    twin.set_specimen("Al thin film 100 nm")
    assert twin.flux(twin.request()).mean() > 1.0, "the waffle grid's origin is a bar"


def test_a_supplied_column_is_left_where_it_is():
    """A column the twin did not build may mirror real hardware: its stage is not ours."""
    column = Column(clock=ManualClock())
    before = column.state().stage
    DigitalTwin("Al thin film 100 nm", camera="DESim", column=column, clock=ManualClock())
    after = column.state().stage
    assert (after.x_um, after.y_um) == (before.x_um, before.y_um)


def test_the_start_can_be_skipped():
    twin = DigitalTwin("Al thin film 100 nm", camera="DESim", clock=ManualClock(), start_on_specimen=False)
    st = twin.column.state()
    assert (st.stage.x_um, st.stage.y_um) == (0.0, 0.0)


@pytest.mark.parametrize("name", sorted(options.PATTERNS))
def test_every_pattern_starts_over_specimen_too(name):
    """The generic patterns (the picker lists them beside the presets) start the same way;
    unstained proteins underfocus."""
    twin = DigitalTwin(name, camera="DESim", clock=ManualClock(), seed=3)
    assert twin.flux(twin.request()).mean() > 1.0
    subtle = any(k in name for k in ("Proteins", "FIB", "Waffle", "PNJunction", "Strain"))
    assert _repeat_corr(twin) > (0.15 if subtle else 0.45)


def test_a_lamella_of_an_amorphous_material_renders_in_both_projections():
    """The FIB-liftout pattern's "auto" post can be amorphous: no crystal library, so no
    Bragg beams — it used to raise instead."""
    from de_twin.state import Projection

    twin = DigitalTwin("Virtual Specimen - FIB Liftout", camera="DESim", clock=ManualClock(), seed=3)
    assert np.isfinite(twin.flux(twin.request())).all()
    twin.column.set_projection(Projection.DIFFRACTION)
    assert np.isfinite(twin.flux(twin.request())).all()
