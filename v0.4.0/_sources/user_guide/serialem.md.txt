# SerialEM calibrations

```python
from de_twin import DigitalTwin, ManualClock
from de_twin.optics import OpticsConfig
from de_twin.optics.realism import ColumnRealism
from de_twin.optics.serialem import from_serialem, to_serialem

# a twin's column, as the calibrations SerialEM would measure on it
to_serialem(ColumnRealism(seed=11), "DESim", "SerialEMcalibrations.txt", "rotation_and_pixel.txt")

# a microscope's calibrations, as a twin's column
column = from_serialem("SerialEMcalibrations.txt", "rotation_and_pixel.txt", camera="DESim")
twin = DigitalTwin(camera="DESim", clock=ManualClock(), optics_config=OpticsConfig(realism=column))
```

`to_serialem` writes a `SerialEMcalibrations.txt` file for a column (`ColumnRealism` or
`ColumnDefinition`):

| Entry | What it holds |
|---|---|
| `ImageShiftMatrix` | Camera pixels per image-shift unit, for each magnification index. |
| `StageToCameraMatrix` | Camera pixels per stage µm. |
| `ImageShiftOffsets` | The image shift that re-centres each magnification. |
| `CrossoverIntensity` | Per spot size: microprobe (the twin's TEM mode), then nanoprobe. |
| `HighFocusMagCal` | Image scale and rotation at −50 to −300 µm defocus. |
| `FocusCalibration` | Image displacement against defocus, per unit of beam tilt in the column's units. |

With a properties path, it also writes the camera's `CameraProperties` block of
`RotationAndPixel` lines.

`from_serialem` reads a microscope's files into a `ColumnDefinition`. This is a
table-driven column that goes in `OpticsConfig(realism=...)`, so the twin behaves like that
microscope. For each magnification it holds:

- the rotation and true pixel size, from `StageToCameraMatrix`, or from `RotationAndPixel`
  where that is defined;
- the image-shift matrix;
- the magnification offset.

It also holds a beam-shift matrix (from a real file's `BeamShiftCalibration`), the crossovers
per spot and probe, and the high-defocus scale and rotation. Anything the files do not define
is ideal.

The formats are those of SerialEM's own reader and writer (`ParameterIO.cpp`). The reader:

- treats 999 as undefined;
- reads only the requested camera's `CameraProperties` block;
- skips line forms it does not know, rather than failing.

## Conventions

The conventions follow SerialEM's help ("Image Rotation") and `ShiftManager.cpp`:

- Camera coordinates are right-handed, with x to the right and y up. The twin's raster has
  y down.
- Specimen coordinates are minus stage coordinates.
- `SpecimenToCamera = R(rotation) / pixel`, so `StageToCamera = −R(rotation) / pixel`, and
  the image rotation is `atan2(−ypx, −xpx)`.
- `IStoCamera = SpecimenToCamera @ (specimen shift per image-shift unit)`.

The twin's stage y runs opposite to SerialEM's. With that, SerialEM's image rotation is the
twin's view rotation, and every matrix has a positive determinant, as on a real scope.
`tests/test_serialem_calibrations.py` checks the written stage and image-shift matrices
against image motion measured on the twin's own images.

## Magnification indices

SerialEM numbers a scope's magnifications from 1. The twin's own numbering is its imaging
ladder, LowMAG then MAG1 (`twin_mag_table()`). For a real scope, pass its table as
`mag_table={index: magnification}` and its low-magnification ceiling as `lm_max_mag`.
A magnification that falls between table entries takes the nearest entry in its own range
(LM or not).

## Not covered

- Refining a column definition from acquired data. That belongs to the tools that drive a
  microscope (Ground Crew, de_autopilot); de-twin only converts.
- `ComaVsISCal` and the low-dose areas. These live in SerialEM's settings file, not its
  calibrations.
- The image-shift to beam-shift calibration. In the twin, image shift never moves the beam
  off the imaged area, so none is written.
- Absolute handedness. This follows SerialEM's documentation; check it once against a real
  calibration file.
