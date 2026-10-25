"""Column: the DummyEMInstrument-shaped simulated column."""

import math

import pytest

from de_twin.clock import ManualClock
from de_twin.column import (
    BEAM_TILT_MRAD_PER_UNIT,
    Column,
    ColumnAdapter,
    ColumnRefused,
    ladders,
)
from de_twin.state import MicroscopeState, ProbeMode, Projection, TemStem


@pytest.fixture
def clock():
    return ManualClock()


@pytest.fixture
def col(clock):
    return Column(clock=clock)


def test_defaults_are_jeol_mag1(col):
    s = col.state()
    assert s.instrument_type == "JEOL"
    assert s.mag_mode == "MAG1" and s.projection == Projection.IMAGING
    assert s.magnification in ladders.MAG1_MAGS
    assert col.get("SubMode") == 0 and col.get("ProjectionMode") == 1
    assert col.get("HighTension") == pytest.approx(200000.0)


def test_magnification_snaps_to_ladder(col):
    col.set("Magnification", 23000)
    assert col.get("Magnification") == 25000.0  # nearest by absolute difference
    col.set_magnification_index(0)
    assert col.get("Magnification") == ladders.MAG1_MAGS[0]
    col.set("MagnificationIndex", 999)
    assert col.get("Magnification") == ladders.MAG1_MAGS[-1]


def test_magnification_crosses_lowmag_and_back(col):
    col.set("Magnification", 150)
    assert col.get("MagMode") == "LowMAG" and col.get("Magnification") == 150.0
    col.set("Magnification", 1800)  # above the LowMAG ceiling -> MAG1 floor
    assert col.get("MagMode") == "MAG1" and col.get("Magnification") == 2000.0
    col.set("MagnificationIndex", -1)  # stepping below MAG1 crosses to LowMAG
    assert col.get("MagMode") == "LowMAG" and col.get("Magnification") == 1500.0


def test_tem_imaging_reaches_the_magnifications_between_lowmag_and_mag1(col):
    """500x and 1000x are LowMAG rungs on a real JEOL; DE-Server's mag.yaml lacked them."""
    for magnification in (500.0, 1000.0, 1500.0):
        col.set("Magnification", magnification)
        assert col.get("MagMode") == "LowMAG" and col.get("Magnification") == magnification


@pytest.mark.parametrize(("vendor", "largest"), [("JEOL", 9), ("FEI", 11)])
def test_spot_size_reaches_the_vendor_range(clock, vendor, largest):
    column = Column(clock=clock, instrument_type=vendor)
    column.set("SpotSize", largest)
    assert column.get("SpotSize") == largest
    column.set("SpotSize", largest + 5)
    assert column.get("SpotSize") == largest
    column.set("SpotSize", 0)
    assert column.get("SpotSize") == 1


def test_the_mag_ladder_steps_across_lowmag_and_mag1(col):
    """A client stepping `mag_ladder` rung by rung walks from LowMAG into MAG1 and back."""
    col.set("Magnification", 120)
    ladder = list(col.mag_ladder())
    assert list(ladder) == sorted(ladder) and ladder[0] == 100.0 and ladder[-1] == ladders.MAG1_MAGS[-1]
    for rung in ladder:
        col.set("Magnification", rung)
        assert col.get("Magnification") == rung
        assert list(col.mag_ladder()) == ladder
    assert col.get("MagMode") == "MAG1"
    for rung in reversed(ladder):
        col.set("Magnification", rung)
        assert col.get("Magnification") == rung
    assert col.get("MagMode") == "LowMAG"


def test_camera_length_only_in_diffraction(col):
    with pytest.raises(ColumnRefused):
        col.set("CameraLength", 20)
    col.set("ProjectionMode", 2)
    assert col.state().projection == Projection.DIFFRACTION and col.get("SubMode") == 4
    col.set("CameraLength", 21)
    assert col.get("CameraLength") == 20.0
    assert col.state().camera_length_mm == 200.0
    col.set_camera_length_mm(410)
    assert col.get("CameraLength") == 40.0
    with pytest.raises(ColumnRefused):
        col.set("Magnification", 50000)  # no magnification in TEM diffraction
    col.set("ProjectionMode", 1)
    assert col.get("MagMode") == "MAG1"  # back to the last imaging mode


def test_stem_rules(col):
    col.set("TemStemMode", 1)
    assert col.get("MagMode") == "LowMAG"  # HR_STEM
    assert col.get("Magnification") in ladders.STEM_MAGS
    with pytest.raises(ColumnRefused):
        col.set("SubMode", 0)  # MAG1 has no STEM project
    col.set("SubMode", 4)
    assert col.state().camera_length_mm in ladders.STEM_CAMERA_LENGTHS_MM
    col.set("Magnification", 1.1e6)
    assert col.get("Magnification") == 1.0e6 or col.get("Magnification") == 1.2e6


def test_stage_moves_on_the_clock(col, clock):
    col.set_stage(x=100.0, alpha=5.0)
    s = col.state()
    assert s.op_status == 1 and s.stage.x_um == 0.0
    clock.advance(0.5)
    assert col.get("StageX") == pytest.approx(50.0)  # 100 um/s
    assert col.get("StageA") == pytest.approx(5.0)  # 10 deg/s: tilt already arrived
    assert col.time_to_idle() == pytest.approx(0.5)
    assert col.wait_idle()
    assert col.state().op_status == 0
    assert col.get("StagePosition") == {"x": 100.0, "y": 0.0, "z": 0.0, "a": 5.0, "b": 0.0}


def test_stage_retarget_keeps_other_axes_and_stop_freezes(col, clock):
    col.set_stage(x=200.0)
    clock.advance(1.0)
    col.set_stage(y=-50.0)  # the x move keeps going
    clock.advance(0.5)
    x, y, *_ = col.stage_position()
    assert x == pytest.approx(150.0) and y == pytest.approx(-50.0)
    col.stop()
    clock.advance(10)
    assert col.get("StageX") == pytest.approx(150.0)
    assert col.op_state() == 4


def test_stage_limits_clamp(col, clock):
    col.set("StagePosition", {"x": 1e6, "z": -9999, "a": 90})
    col.wait_idle()
    x, _, z, a, _ = col.stage_position()
    assert (x, z, a) == (2000.0, -500.0, 70.0)
    assert clock.now() == pytest.approx(20.0, abs=1e-6)


def test_convergence_derived_from_probe_and_alpha(col):
    col.set("AlphaSelector", 1)
    col.set("ProbeMode", 3)  # CBD: 3-15 mrad
    assert col.get("ConvergenceAngle") == pytest.approx(5.0)
    assert col.state().probe_mode == ProbeMode.CBD
    col.set("ProbeMode", "NBD")  # NBD: 0.5-3 mrad
    assert col.get("ProbeMode") == 2
    col.set("ConvergenceAngle", 1.7)
    assert col.state().convergence_semi_angle_mrad == pytest.approx(1.7)
    col.set("ConvergenceAngle", 0)
    assert col.get("ConvergenceAngle") == pytest.approx(1.0)
    col.set("ProbeMode", "TEM")  # parallel illumination: 0.01-0.1 mrad
    assert 0.01 <= col.get("ConvergenceAngle") <= 0.1


def test_stem_convergence_table(col):
    col.set("TemStemMode", 1)
    for alpha, mrad in enumerate((5.0, 10.0, 15.0, 22.0, 30.0)):
        col.set("AlphaSelector", alpha)
        assert col.get("ConvergenceAngle") == pytest.approx(mrad)
        assert col.state().convergence_semi_angle_mrad == pytest.approx(mrad)
    col.set("TemStemMode", 0)
    assert col.get("ConvergenceAngle") < 0.2


def test_safety_refuses_ht(clock):
    column = Column(clock=clock)
    for name in ("HighTension", "HTState", "Filament"):
        with pytest.raises(ColumnRefused):
            column.set(name, 1)
    ok = Column(clock=clock, allow_ht=True)
    ok.set_ht_kv(300)
    assert ok.state().ht_kv == 300.0


def test_pairs_and_units(col):
    col.set("ImageShift", (0.5, -0.25))
    col.set_beam_tilt_mrad(2.0, -1.0)
    col.set("DiffractionShift", [0.1, 0.2])
    s = col.state()
    assert s.image_shift_um.x == 0.5 and s.image_shift_um.y == -0.25
    assert s.beam_tilt_mrad.x == pytest.approx(2.0)
    assert col.get("BeamTilt") == pytest.approx(
        (2.0 / BEAM_TILT_MRAD_PER_UNIT, -1.0 / BEAM_TILT_MRAD_PER_UNIT)
    )
    assert s.diffraction_shift_mrad.y == pytest.approx(2.0)
    with pytest.raises(ColumnRefused):
        col.set("BeamShift", "nonsense")
    with pytest.raises(ColumnRefused):
        col.set("ImageShift", (math.nan, 0))


def test_screen_and_blank(col):
    col.set("ScreenPosition", 1)
    assert col.state().screen_position == 1
    with pytest.raises(ColumnRefused):
        col.set("ScreenPosition", 3)
    col.set_beam_blank(True)
    assert col.state().beam_blanked and col.get("BeamBlank") == 1


def test_fei_vendor_reporting(clock):
    column = Column(clock=clock, instrument_type="FEI")
    assert column.get("SubMode") == 2  # MAG1 -> TFS "M"
    assert column.get("ScreenPosition") == 2  # up
    column.set("ScreenPosition", 3)
    assert column.state().screen_position == 1
    column.set("ProbeMode", 0)
    assert column.state().probe_mode == ProbeMode.NANOPROBE
    column.set("SubMode", 1)
    assert column.get("MagMode") == "LowMAG"


def test_initial_state_is_resnapped(clock):
    st = MicroscopeState(magnification=23456.0, tem_stem=TemStem.STEM, mag_mode="MAG1")
    column = Column(st, clock=clock)
    assert column.get("MagMode") == "LowMAG"
    assert column.get("Magnification") in ladders.STEM_MAGS


def test_on_change_and_every_property_roundtrips(col):
    seen = []
    col.on_change(lambda name, value: seen.append((name, value)))
    col.set("SpotSize", 12)
    assert seen[-1] == ("SpotSize", 9)  # clamped to a JEOL's range, readback reported
    for name in Column.property_names():
        if name in ("HighTension", "HT", "HTState", "Filament"):
            continue
        try:
            value = col.get(name)
        except KeyError:
            continue
        try:
            col.set(name, value)
        except KeyError:
            continue  # read-only
        except ColumnRefused:
            continue  # e.g. CameraLength in imaging
        assert col.get(name) == value, name
    assert isinstance(col.values(), dict)


def test_real_clock_callable():
    t = [0.0]
    column = Column(clock=lambda: t[0])
    column.set_stage(x=10)
    t[0] = 0.05
    assert column.get("StageX") == pytest.approx(5.0)


def test_column_adapter_duck_type(col, clock):
    ad = ColumnAdapter(col)
    moves, sets = [], []
    ad.on_move(lambda x, y: moves.append((x, y)))
    ad.on_set(lambda p, v: sets.append((p, v)))
    assert ad.real is False and ad.reason
    ad.move_stage(12.0, -3.0)  # blocks on the (manual) clock
    assert ad.stage_xy() == (12.0, -3.0) and moves == [(12.0, -3.0)]
    ad.tilt_to(10)
    assert ad.tilt() == 10.0
    with pytest.raises(ColumnRefused):
        ad.tilt_to(80)
    with pytest.raises(ColumnRefused):
        ad.move_stage(5000, 0)
    with pytest.raises(ColumnRefused):
        ad.set("HighTension", 1)
    ad.set("ImageShift", [0.1, 0.2])
    assert ad.get("ImageShift") == [0.1, 0.2]
    assert sets[-1] == ("ImageShift", [0.1, 0.2])
    ad.set("StageX", 20.0)
    assert ad.stage_xy()[0] == 20.0
    v = ad.values()
    assert v["InstrumentType"] == "JEOL" and isinstance(v["BeamShift"], list)
    assert ad.screen_positions() == {"up": 0, "down": 1}
    ad.stop()
    ad.close()


def test_ladders_are_lists_and_camera_lengths_are_millimeters(col):
    assert not col.cl_ladder()  # no camera length in TEM imaging
    col.set("ProjectionMode", 2)
    assert not col.mag_ladder()  # no magnification in TEM diffraction
    assert col.cl_ladder()[0] == 15.0 and col.cl_ladder()[-1] == 4000.0
