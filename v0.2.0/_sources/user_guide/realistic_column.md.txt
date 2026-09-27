# A realistic column

```python
from de_twin import DigitalTwin, ManualClock
from de_twin.optics import OpticsConfig

twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(),
                   optics_config=OpticsConfig.realistic(seed=7))
twin.column.set("Magnification", 20000)
truth = twin.calibration_truth()
truth["image_rotation_deg"], truth["true_pixel_nm"], truth["nominal_pixel_nm"]
truth["is_matrix_um_per_unit"], truth["mag_offset_um"], truth["crossover_intensity"]
```

The default twin is an ideal column. Its image never rotates, the pixel size is the nominal
one, and image shift moves the specimen one micrometre per unit along the camera axes.
Changing magnification keeps the same point on axis, and the stage lands where it is told.

A real column does none of these, and each difference is something a SerialEM calibration
measures. `OpticsConfig.realistic(seed)` gives the twin those imperfections. Every value is a
deterministic function of the seed and of the imaging mode and magnification, so the same
seed always gives the same column. `DigitalTwin.calibration_truth()` reports the true values
at the column's current state. A test can run a calibration procedure on the twin's images
and compare its answer with them.

| Effect | Model | SerialEM calibration |
|---|---|---|
| Image rotation | Per imaging mode, a base angle (LowMAG and MAG1 differ by a large angle). Per magnification, about 2° of jitter. High defocus adds 0.01°/µm. | Image & Stage Shift, stage calibration |
| True pixel size | Nominal × (1 ± about 2 %) per magnification. High defocus adds 2×10⁻⁴/µm (4 % at −200 µm). Clients such as the deapi face and DE-Server still see the nominal value. | Find Pixel Size, High-Defocus Mag |
| Image-shift matrix | Per magnification, a 2×2 of scale, skew and rotation, in specimen µm per image-shift unit. | Image Shift |
| Magnification offsets | Per magnification, an image-centre offset (2 % of a 1e5 / mag µm field). | Mag IS offsets |
| Stage backlash | The stage stops short by backlash / 2 against its approach direction, while reporting the commanded position. | Backlash correction |
| C2 crossover | The beam converges to its smallest at an Intensity that depends on spot size and probe mode, and spreads again past it. | Beam Crossover, Beam Intensity |
| Beam-shift matrix | One 2×2 for the illumination system. | Beam Shift |
| Coma from image shift | Image shift images the specimen off the coma-free axis. This acts as an effective beam tilt of about 0.43 mrad per µm of image shift, plus about 15 nm of A1 per µm. | Coma vs Image Shift |

Two effects are plain physics, so they are on in every twin:

- **A tilted specimen that is off eucentric height moves in the image.** The displacement is
  dz·sin(tilt) across the tilt axis (dz·tan(tilt) in specimen coordinates). This is what
  eucentricity routines measure.
- **Beam shift moves the beam, not the image.** In TEM imaging it moves the illuminated disc
  across the camera and leaves the specimen in place. In STEM, and for any scan request, it
  moves the probe.

The realistic column takes the magnification-specific terms (rotation, pixel size,
image-shift matrix, offsets and induced coma) from the **render mode**, not the column
state. A 4D-STEM request on a column that reports TEM is therefore treated as a scan.

## Conventions

`image_rotation_deg` is the rotation of the view from the world (specimen) frame to the
camera frame. Image features turn by minus that angle in (x right, y down) pixel
coordinates. Beam tilt is a direction in the specimen plane, because the tilt coils sit above
the specimen. Its effects on the camera turn with the image, including the axial coma, the
focus-dependent image shift and the tilt induced by image shift.

## Intensity zoom

```python
cfg = OpticsConfig(intensity_zoom=True)      # or OpticsConfig.realistic(seed, intensity_zoom=True)
```

Without intensity zoom, the dose per detector pixel goes as 1/mag². Zooming out from a
detector-safe beam then saturates the camera, and zooming in leaves only shot noise.
Intensity zoom works like TFS Intensity Zoom: in TEM imaging the condenser follows the
magnification, so the illuminated area keeps its size relative to the field of view and the
dose per pixel stays the same. Intensity still sets the beam at
`intensity_zoom_reference_mag`, and so at every magnification. Diffraction and STEM are
unaffected. Ground Crew's twin turns it on.

## Calibration specimens

`"Cross grating 2160 l/mm"` is a carbon replica cross grating with a 463 nm period. It is
the standard specimen for pixel size, and for image-shift calibrations at low
magnification. At higher magnification, use a non-periodic specimen such as
`"Dense Au on holey C"`, so cross-correlations have a unique answer.

## Closed-loop tests

`tests/test_realism_calibrations.py` runs each calibration procedure on the twin's images and
checks the result against `calibration_truth()`. It covers:

- pixel size from the grating's FFT;
- image-shift and stage matrices from cross-correlation;
- magnification offsets and backlash;
- eucentric displacement;
- beam crossover and beam shift;
- autofocus beam-tilt pairs;
- coma vs image shift;
- the high-defocus scale and rotation.

The measuring code in these tests is test-only on purpose: de-twin produces data and ground
truth, and the calibration methods belong to the tools that drive a microscope.

Some things to know when measuring on twin images:

- **Focus-step pairs.** Use a plain cross-correlation of low-passed images. Phase correlation
  whitens the spectrum, so low-pass filtering has no effect on it, and the fine contrast
  differs between focus settings.
- **Gratings.** Use a spread beam and a window function. Otherwise the fixed illuminated disc
  and the edges of the image dominate the FFT.
- **Stage steps.** Keep them to a fraction of the field of view, and set
  `column.backlash_um = 0` when the test is not about backlash.
