"""SerialEM calibration files to and from a column definition: an exact round trip."""

from __future__ import annotations

import math

import numpy as np
import pytest

from de_twin.clock import ManualClock
from de_twin.optics import OpticsConfig
from de_twin.optics.realism import ColumnRealism
from de_twin.optics.serialem import ColumnDefinition, from_serialem, to_serialem, twin_mag_table
from de_twin.twin import DigitalTwin

CAMERA = "DESim"


@pytest.fixture
def files(tmp_path):
    return tmp_path / "SerialEMcalibrations.txt", tmp_path / "rotation_and_pixel.txt"


def test_the_file_is_a_serialem_calibration_file(files):
    cal, props = files
    to_serialem(ColumnRealism(seed=3), CAMERA, cal, props)
    text = cal.read_text().splitlines()
    assert text[0] == "SerialEMCalibrations"
    keys = {line.split()[0] for line in text[1:] if line and line.split()[0].isalpha()}
    assert {"ImageShiftMatrix", "StageToCameraMatrix", "ImageShiftOffsets",
            "CrossoverIntensity", "HighFocusMagCal", "FocusCalibration"} <= keys
    n = int(next(line for line in text if line.startswith("ImageShiftMatrix")).split()[1])
    assert n == len(twin_mag_table())
    body = props.read_text().splitlines()
    assert body[0] == "CameraProperties 0" and body[-1] == "EndCameraProperties"
    assert all(line.startswith("RotationAndPixel") for line in body[1:-1])


@pytest.mark.parametrize("with_props", [False, True])
def test_a_column_survives_the_round_trip(files, with_props):
    cal, props = files
    real = ColumnRealism(seed=11)
    to_serialem(real, CAMERA, cal, props)
    d = from_serialem(cal, props if with_props else None, camera=CAMERA)
    assert isinstance(d, ColumnDefinition)
    for mag in twin_mag_table().values():
        mode = "LowMAG" if mag <= 200 else "MAG1"
        dr = (d.rotation_rad(mode, mag) - real.rotation_rad(mode, mag) + math.pi) % (2 * math.pi) - math.pi
        assert dr == pytest.approx(0.0, abs=1e-6), mag
        assert d.pixel_scale(mode, mag) == pytest.approx(real.pixel_scale(mode, mag), rel=1e-6), mag
        np.testing.assert_allclose(d.is_matrix(mode, mag), real.is_matrix(mode, mag), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(d.mag_offset_um(mode, mag), real.mag_offset_um(mode, mag), atol=1e-6)
    for spot in (1, 3, 5):  # the twin's own probe modes: TEM (2, microprobe) and nanoprobe (0)
        for probe in (2, 0):
            assert d.crossover(spot, probe) == pytest.approx(real.crossover(spot, probe), abs=1e-7)
    assert d.hd_scale_per_um == pytest.approx(real.hd_scale_per_um, rel=1e-5)
    assert d.hd_rotation_deg_per_um == pytest.approx(real.hd_rotation_deg_per_um, rel=1e-5)


def test_a_twin_defined_by_serialem_calibrations_is_that_column(files):
    cal, props = files
    real = ColumnRealism(seed=11, backlash_um=0.0, is_coma_mrad_per_um=0.0, is_astig_nm_per_um=0.0)
    to_serialem(real, CAMERA, cal, props)
    d = from_serialem(cal, props, camera=CAMERA)
    a = DigitalTwin("Dense Au on holey C", camera=CAMERA, clock=ManualClock(), seed=1,
                    optics_config=OpticsConfig(realism=real))
    b = DigitalTwin("Dense Au on holey C", camera=CAMERA, clock=ManualClock(), seed=1,
                    optics_config=OpticsConfig(realism=d))
    for mag in (150.0, 5000.0, 20000.0, 60000.0):
        for tw in (a, b):
            tw.column.set("Magnification", mag)
            tw.column.set("ImageShift", (0.3, -0.2))
        ta, tb = a.calibration_truth(), b.calibration_truth()
        assert tb["true_pixel_nm"] == pytest.approx(ta["true_pixel_nm"], rel=1e-6)
        assert (tb["image_rotation_deg"] - ta["image_rotation_deg"] + 180) % 360 - 180 == pytest.approx(0, abs=1e-4)
        np.testing.assert_allclose(tb["is_matrix_um_per_unit"], ta["is_matrix_um_per_unit"], atol=1e-6)
        np.testing.assert_allclose(tb["mag_offset_um"], ta["mag_offset_um"], atol=1e-6)
        oa, ob = a.optics(a.request()), b.optics(b.request())
        np.testing.assert_allclose(ob.view.center_um, oa.view.center_um, atol=1e-6)
    ia = a.flux(a.request())
    ib = b.flux(b.request())
    # the same image; the file's 12 digits can tip a few texture pixels into the next cell
    assert np.corrcoef(ia.ravel(), ib.ravel())[0, 1] > 0.9995, "the same image"


# ---------------------------------------------------------- against the twin's own images
def _se_twin():
    from test_realism_calibrations import _twin

    return _twin()


def test_the_stage_matrix_is_serialems_convention_measured_on_the_twin(files):
    """SerialEM: image motion (right-handed camera px, y up) = -StageToCamera @ stage step,
    and at rotation 0 +X moves the image right, +Y up. Measured on the twin's images."""
    from test_realism_calibrations import _img, _shift_px

    cal, props = files
    tw = _se_twin()
    tw.column.backlash_um = 0.0  # backlash has its own test; here only the matrix
    to_serialem(tw.optics_config.realism, CAMERA, cal, props)
    ind = next(i for i, m in twin_mag_table().items() if m == 20000.0)
    S = next(np.array([float(v) for v in line.split()[3:7]]).reshape(2, 2)
             for line in cal.read_text().splitlines()
             if line.startswith(f"StageToCameraMatrix {ind} "))
    rot = float(next(line.split()[3] for line in props.read_text().splitlines()
                     if line.startswith(f"RotationAndPixel {ind} ")))
    assert np.linalg.det(S) > 0, "a real scope's handedness"
    assert rot == pytest.approx(math.degrees(math.atan2(-S[1, 0], -S[0, 0])), abs=1e-6)
    a = _img(tw)
    for dx, dy in ((0.1, 0.0), (0.0, 0.1)):
        s0 = tw.column.state().stage
        tw.column.move_stage(x=s0.x_um + dx, y=s0.y_um + dy)
        m = _shift_px(a, _img(tw))  # raster (y down)
        tw.column.move_stage(x=s0.x_um, y=s0.y_um)
        m_se = np.array([m[0], -m[1]])
        step_se = np.array([dx, -dy])  # SerialEM's stage y runs opposite to the twin's
        np.testing.assert_allclose(m_se, -S @ step_se, atol=0.3)


def test_the_image_shift_matrix_is_measured_on_the_twin(files):
    from test_realism_calibrations import _img, _shift_px

    cal, props = files
    tw = _se_twin()
    to_serialem(tw.optics_config.realism, CAMERA, cal, props)
    ind = next(i for i, m in twin_mag_table().items() if m == 20000.0)
    text = cal.read_text().splitlines()
    k = text.index(next(line for line in text if line.startswith("ImageShiftMatrix")))
    A = next(np.array([float(v) for v in line.split()[2:6]]).reshape(2, 2)
             for line in text[k + 1:] if line.split()[0] == str(ind))
    a = _img(tw)
    for u in ((0.2, 0.0), (0.0, 0.2)):
        tw.column.set("ImageShift", u)
        m = _shift_px(a, _img(tw))
        tw.column.set("ImageShift", (0.0, 0.0))
        np.testing.assert_allclose(np.array([m[0], -m[1]]), -A @ np.asarray(u), atol=0.3)


def test_the_reader_takes_what_serialem_writes_and_skips_what_it_does_not_define(tmp_path):
    cal = tmp_path / "cal.txt"
    props = tmp_path / "props.txt"
    cal.write_text("\n".join([
        "SerialEMCalibrations",
        "StageToCameraMatrix 14 0 -2 0 0 -2   0.000000   20000",
        "StageToCameraMatrix 14 1 -9 0 0 -9   0.000000   20000",
        "CrossoverIntensity 3 0.42",  # microprobe only
        "BeamShiftCalibration 0.5 0 0 0.5",  # an old form without a mag index
        "BeamShiftCalibration 14 0.5 0 0 0.5 -999 1 0  20000",
        "ImageShiftMatrix 1",
        "14 0 2 0 0 2   20000",
        "FocusCalibration 14 0 1 1 1.00 2 0 1 -999   20000",
        "-1 1 1",
        "1 -1 -1",
    ]) + "\n")
    props.write_text("\n".join([
        "CameraProperties 1", "RotationAndPixel 14 999 45 0.1", "EndCameraProperties",
        "CameraProperties 0", "RotationAndPixel 14 999 999 0", "EndCameraProperties",
    ]) + "\n")
    d = from_serialem(cal, props, camera=CAMERA)
    mag = 20000.0
    assert d.rotation_rad("MAG1", mag) == pytest.approx(0.0, abs=1e-9), "999: from the stage matrix"
    assert d.pixel_nm[mag] == pytest.approx(500.0), "0 / 999 pixel: from the stage matrix"
    assert d.crossover(3, 2) == pytest.approx(0.42)
    assert d.bs is not None
