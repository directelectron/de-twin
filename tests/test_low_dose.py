"""Low dose: Search / View / Focus / Record areas, and aligning View to Record on a realistic column."""

from __future__ import annotations

import math

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.column import Column, ColumnRefused
from de_twin.optics import OpticsConfig
from de_twin.twin import DigitalTwin


def _col():
    c = Column(clock=ManualClock())
    c.set("Magnification", 25000)
    c.set("Defocus", -1.5)
    return c


def test_switching_areas_applies_and_remembers_their_settings():
    c = _col()
    with pytest.raises(ColumnRefused):
        c.set("LowDoseArea", "View")
    c.set("LowDose", True)
    assert c.get("LowDoseArea") == "Record"
    c.set("LowDoseArea", "View")
    s = c.state()
    assert s.magnification < 25000 and s.defocus_um == pytest.approx(-1.5 - 200.0)
    c.set("SpotSize", 5)  # adjusted while in View: stays with View (default 3)
    c.set("LowDoseArea", "Focus")
    s = c.state()
    assert s.magnification == 25000 and s.image_shift_um.x == pytest.approx(1.5) and s.spot_size != 5
    c.set("LowDoseArea", "Record")
    c.set("Defocus", -2.0)  # refocus in Record: View's offset follows
    c.set("LowDoseArea", "View")
    s = c.state()
    assert s.spot_size == 5 and s.defocus_um == pytest.approx(-2.0 - 200.0)
    areas = c.get("LowDoseAreas")
    assert set(areas) == {"Search", "View", "Focus", "Record"} and areas["View"]["spot_size"] == 5
    c.set("LowDose", False)
    s = c.state()
    assert c.get("LowDoseArea") == "" and s.magnification == 25000 and s.defocus_um == pytest.approx(-2.0)


def test_area_settings_can_be_written():
    c = _col()
    c.set("LowDose", True)
    c.set("LowDoseAreas", {"Focus": {"image_shift": (0.0, -2.0), "defocus_offset_um": -3.0}})
    c.set("LowDoseArea", "Focus")
    s = c.state()
    assert (s.image_shift_um.x, s.image_shift_um.y) == (0.0, -2.0)
    assert s.defocus_um == pytest.approx(-1.5 - 3.0)
    with pytest.raises(ColumnRefused):
        c.set("LowDoseAreas", {"Nowhere": {}})


def _to_view_frame(record: np.ndarray, rec_view, view_view, shape) -> np.ndarray:
    """The Record image resampled onto View's camera frame about the two image centres, from
    the magnification calibrations' scale and rotation (the views' true pixel and rotation):
    what SerialEM does before aligning the two to find the area offset."""
    from scipy.ndimage import affine_transform

    s = view_view.pixel_um / rec_view.pixel_um  # record px per view px (raster)
    a = view_view.rotation_rad - rec_view.rotation_rad
    c, sn = math.cos(a), math.sin(a)
    # output (view) (row, col) offsets -> input (record) (row, col) offsets
    R = s * np.array([[c, sn], [-sn, c]])
    out_c = (np.array(shape) - 1) / 2.0
    in_c = (np.array(record.shape) - 1) / 2.0
    return affine_transform(record, R, offset=in_c - R @ out_c, output_shape=shape, order=1, cval=np.nan)


def test_view_to_record_alignment_finds_the_true_area_offset():
    from scipy.ndimage import gaussian_filter

    tw = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(), seed=1,
                     optics_config=OpticsConfig.realistic(seed=5))
    tw.column.set("Magnification", 20000)
    tw.column.set("Intensity", 0.95)
    tw.column.set("LowDose", True)
    tw.column.set("LowDoseAreas", {"View": {"magnification": 5000, "defocus_offset_um": -30.0,
                                            "intensity": 0.95}})
    req = tw.request()
    o_rec = tw.optics(req)
    rec = tw.flux(req).astype(float)
    tw.column.set("LowDoseArea", "View")
    o_view = tw.optics(req)
    view = tw.flux(req).astype(float)
    assert o_rec.raster_downsample == o_view.raster_downsample == 1
    # truth: where Record's centre falls in View, in View pixels from View's centre
    r1, c1 = o_view.view.world_to_pixel(*o_rec.view.center_um)
    r0, c0 = o_view.view.world_to_pixel(*o_view.view.center_um)
    want = np.array([float(c1 - c0), float(r1 - r0)])
    assert np.hypot(*want) > 5.0, "the areas do not line up by themselves"
    # the Record image in View's frame, cross-correlated with View
    patch = _to_view_frame(rec, o_rec.view, o_view.view, view.shape)
    m = np.isfinite(patch)
    p = np.where(m, patch - np.nanmean(patch), 0.0)
    v = np.where(m.any(), view - view.mean(), 0.0)
    cc = np.fft.ifft2(np.fft.fft2(gaussian_filter(v, 2)) * np.conj(np.fft.fft2(gaussian_filter(p, 2)))).real
    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    ny, nx = cc.shape
    got = np.array([ix - nx if ix > nx // 2 else ix, iy - ny if iy > ny // 2 else iy], float)
    assert got == pytest.approx(want, abs=1.5)


def test_editing_the_current_area_keeps_its_live_changes():
    c = _col()
    c.set("LowDose", True)
    c.set("LowDoseArea", "View")
    c.set("SpotSize", 5)
    c.set("LowDoseAreas", {"View": {"intensity": 0.6}})
    s = c.state()
    assert s.spot_size == 5 and s.intensity == pytest.approx(0.6)


def test_a_bad_area_setting_is_refused_and_changes_nothing():
    c = _col()
    c.set("LowDose", True)
    before = c.get("LowDoseAreas")
    for bad in ({"View": {"image_shift": (1.0,)}}, {"View": {"intensity": "lots"}},
                {"View": {"defocus_offset_um": float("nan")}}):
        with pytest.raises(ColumnRefused):
            c.set("LowDoseAreas", bad)
    assert c.get("LowDoseAreas") == before


def test_a_refused_switch_leaves_the_column_as_it_was(monkeypatch):
    import de_twin.column.column as colmod

    c = _col()
    c.set("LowDose", True)
    s0 = c.state()
    real = colmod._SETTERS["ImageShift"]

    def refuse(col, v):
        raise ColumnRefused("no")

    monkeypatch.setitem(colmod._SETTERS, "ImageShift", refuse)
    with pytest.raises(ColumnRefused):
        c.set("LowDoseArea", "View")
    monkeypatch.setitem(colmod._SETTERS, "ImageShift", real)
    s1 = c.state()
    assert c.get("LowDoseArea") == "Record"
    assert (s1.magnification, s1.defocus_um, s1.spot_size) == (s0.magnification, s0.defocus_um, s0.spot_size)


def test_low_dose_switches_in_diffraction():
    c = _col()
    c.set("LowDose", True)
    c.set("ProjectionMode", 2)  # diffraction
    c.set("LowDoseArea", "View")
    c.set("LowDoseArea", "Record")
    c.set("LowDose", False)
    assert c.get("LowDoseAreas")["Record"]["magnification"] == 25000


def test_re_enabling_keeps_what_was_set_while_off():
    c = _col()
    c.set("LowDose", True)
    c.set("LowDose", False)
    c.set("Defocus", -3.0)
    c.set("Magnification", 40000)
    c.set("LowDose", True)
    s = c.state()
    assert (s.magnification, s.defocus_um) == (40000, -3.0)
