"""Scan coil voltages: the voltage range is the field, and offsets, zoom and rotation move
the probe inside it (de_twin.optics.scan_coils)."""

import dataclasses

import numpy as np
import pytest

from de_twin.optics import Calibration, OpticsConfig, derive_optics
from de_twin.optics.scan_coils import point_volts, scan_coils
from de_twin.render.testing import StubCamera
from de_twin.state import AcquisitionRequest, MicroscopeState, Roi, ScanRequest, TemStem


def _scan(**kw):
    return ScanRequest(enabled=True, **{"size": (64, 64), **kw})


class TestCoils:
    def test_the_default_range_spreads_the_grid_across_two_volts(self):
        coils = scan_coils(_scan())
        assert coils.volts_per_point == pytest.approx(2.0 / 64)
        assert coils.center_volts == (0.0, 0.0)

    def test_a_smaller_range_zooms_in_about_its_midpoint(self):
        coils = scan_coils(_scan(voltage_h=(0.2, 0.7), voltage_v=(-0.5, 0.0)))
        assert coils.volts_per_point == pytest.approx(0.5 / 64)
        assert coils.center_volts == pytest.approx((0.45, -0.25))

    def test_the_shorter_side_keeps_the_points_square(self):
        coils = scan_coils(_scan(size=(64, 16)))
        assert coils.volts_per_point == pytest.approx(2.0 / 64)
        top, bottom = point_volts(_scan(size=(64, 16)), 0, 0)[1], point_volts(_scan(size=(64, 16)), 0, 15)[1]
        assert bottom - top == pytest.approx(15 * 2.0 / 64)  # a quarter of the V range

    def test_a_pixel_offset_moves_by_whole_and_fractional_points(self):
        step = 2.0 / 64
        assert scan_coils(_scan(pixel_offset=(3.0, -1.5))).center_volts == pytest.approx(
            (3 * step, -1.5 * step)
        )

    def test_rotation_turns_the_grid_but_not_the_offset(self):
        scan = _scan(rotation_deg=90.0, pixel_offset=(2.0, 0.0))
        cx, cy = point_volts(scan, 31.5, 31.5)
        assert (cx, cy) == pytest.approx((2 * 2.0 / 64, 0.0))  # the offset stays along H
        x, y = point_volts(scan, 31.5 + 10, 31.5)  # 10 points along the scan's x ...
        assert (x - cx, y - cy) == pytest.approx((0.0, 10 * 2.0 / 64))  # ... runs along V

    def test_flips_mirror_about_the_grid_centre(self):
        h, v = point_volts(_scan(flip=(True, False)), 0, 0)
        assert h == pytest.approx(31.5 * 2.0 / 64) and v == pytest.approx(-31.5 * 2.0 / 64)


@pytest.fixture
def make_scan_optics():
    def build(scan):
        state = MicroscopeState()
        state.tem_stem = TemStem.STEM
        state.camera_length_mm = 13.0
        state.convergence_semi_angle_mrad = 10.0
        request = AcquisitionRequest(hw_roi=Roi(1920, 1920, 256, 256), hw_binning=(2, 2), scan=scan)
        return derive_optics(state, request, StubCamera(), Calibration.default(), OpticsConfig())

    return build


class TestOptics:
    def test_the_default_range_scans_the_calibrated_field(self, make_scan_optics):
        o = make_scan_optics(_scan(size=(16, 16), step_um=0.0008))
        assert o.view.pixel_um == pytest.approx(0.0008)
        base = make_scan_optics(_scan(size=(16, 16), step_um=0.0008, voltage_h=(-1.0, 1.0)))
        assert o.view.center_um == base.view.center_um

    def test_each_point_lands_where_its_voltages_say(self, make_scan_optics):
        scan = _scan(
            size=(16, 16),
            step_um=0.0008,
            voltage_h=(-0.3, 0.5),
            voltage_v=(-0.6, 0.2),
            pixel_offset=(1.25, -2.0),
            rotation_deg=30.0,
            flip=(False, True),
        )
        o = make_scan_optics(scan)
        reference = make_scan_optics(_scan(size=(16, 16), step_um=0.0008))
        nm_per_v = o.extras["scan_nm_per_volt"]
        assert nm_per_v == pytest.approx(0.8 * 16 / 2.0)
        x0, y0 = reference.view.center_um  # the optic axis: zero volts
        for column, row in ((0, 0), (15, 3), (7, 12)):
            h, v = point_volts(scan, column, row)
            wx, wy = o.view.pixel_to_world(np.array([row]), np.array([column]))
            assert (wx[0], wy[0]) == pytest.approx((x0 + h * nm_per_v / 1000, y0 + v * nm_per_v / 1000))

    def test_the_probe_sees_the_specimen_the_offset_moved_it_to(self, make_scan_optics):
        from de_twin.render import RenderConfig, Renderer
        from de_twin.render.testing import Particle, SyntheticSpecimen

        scan = _scan(size=(16, 16), step_um=0.0008)
        o = make_scan_optics(scan)
        moved = make_scan_optics(dataclasses.replace(scan, pixel_offset=(3.0, 0.0)))
        particles = []
        for column, row in ((7, 5), (11, 9)):  # on the points the moved probe reaches from (4, 5), (8, 9)
            x, y = o.view.pixel_to_world(row, column)
            particles.append(Particle(float(x), float(y), 1.5))
        renderer = Renderer(
            SyntheticSpecimen(particles=particles),
            RenderConfig(stem_model="kinematic", intensity_jitter=False, probe_footprint_blend=False),
        )
        film = renderer.render(o, scan_point=(1, 1))
        for column, row in ((4, 5), (8, 9)):
            here = renderer.render(moved, scan_point=(column, row))
            np.testing.assert_allclose(here, renderer.render(o, scan_point=(column + 3, row)), rtol=1e-5)
            assert not np.allclose(here, film, rtol=0.01)  # it really is on a particle


class TestClamp:
    def test_the_offset_stops_at_half_the_range(self):
        step = 2.0 / 64
        assert scan_coils(_scan(pixel_offset=(32.0, -32.0))).center_volts == pytest.approx(
            (32 * step, -32 * step)
        )
        assert scan_coils(_scan(pixel_offset=(100.0, -100.0))).center_volts == pytest.approx((1.0, -1.0))

    def test_the_offset_stays_inside_the_cards_output(self):
        coils = scan_coils(_scan(voltage_h=(6.0, 9.0), pixel_offset=(64.0, 0.0)))
        assert coils.center_volts[0] == pytest.approx(10.0 - 1.5)  # the scan's edge at +10 V
