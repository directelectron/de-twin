# 4D-STEM data for ptychography and tcBF

STEM patterns are **coherent** for 4D-STEM geometries: the renderer propagates a probe
wavefunction (condenser aperture and the column's probe aberrations) through the specimen's
complex transmission function and records |FFT|² on the detector. The bright-field disk
interferes, so a defocused probe gives a shadow image, and ptychography and tilt-corrected
bright field (tcBF, "parallax") reconstruct the twin's data the way they do real DE data.
de-twin produces the data and the ground truth; reconstruct with the tool of your choice.

```python
from de_twin import DigitalTwin
from de_twin.state import Roi, ScanRequest, TemStem

twin = DigitalTwin("Dense Au on holey C", camera="Celeritas")
col = twin.column
col.set_tem_stem(TemStem.STEM)
col.set_convergence_mrad(15.0)
col.set_defocus_um(-0.5)                    # tcBF: a few hundred nm to ~2 um of (under)focus
col.set_camera_length_mm(40.0)              # bright-field disk ~1/3 of the detector
col.set_spot_size(8)                        # low current: no saturation, smaller source
req = twin.request(hw_roi=Roi(384, 384, 256, 256), hw_binning=(2, 2), frame_time_s=1e-4,
                   total_frames=64 * 64,
                   scan=ScanRequest(enabled=True, size=(64, 64), step_um=0.001, dwell_s=1e-4))

# raw frames exactly as the camera would deliver them (dark/gain-correct with twin.processor)
refs = twin.processor.references(req)
frames = [twin.processor.correct(raw, refs) for raw, meta in twin.frames(req)]
# ...or the noiseless data cube, (scan y, x, 128, 128) electrons / binned pixel / s
cube = twin.datacube(req)
gt = twin.ground_truth_ptychography(req)    # object t(r), probe (modes), positions, aberrations
```

For ptychography, use a focused (or slightly defocused) probe with a scan step well below
the probe size so neighbouring probe positions overlap. The same data are served through
the deapi and shared-memory faces.

## Model

- **Aberrations** come from the column: C1 is the defocus (negative = underfocus), A1 the
  condenser stigmator, C3 1.2 mm uncorrected or whatever the probe corrector leaves; beam
  tilt adds coma.
- **Partial coherence**: focal spread and the effective source size are treated as
  probe modes.
- **Sampling**: detector pixels (after ROI and binning) integrate an oversampled k-grid, so a
  probe defocused by 1 µm fits the real-space window.
- **Model choice**: `RenderConfig.stem_model` is `"auto"` by default: coherent when the binned
  pattern is at most 256² (or the scan oversamples the probe), else a fast kinematic disk
  model. Set `"coherent"` or `"kinematic"` to force one.
- **Speed**: on a 24-core workstation a 64×64 scan with 128² patterns at 20 mrad takes about
  4 s; a tcBF scan at 1 µm defocus about 1 min with 4 probe modes (`coherent_max_modes=1` is
  faster).

The test suite reconstructs this data (tcBF defocus within 10 %, ptychographic phase
correlation above 0.85 against the ground truth), which is how the model is checked.
