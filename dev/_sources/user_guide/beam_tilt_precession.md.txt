# Beam tilt and precession

```python
from de_twin import DigitalTwin, ManualClock

twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock())
col = twin.column
col.set_beam_tilt_mrad(2.0, 0.0)          # tilt the illumination
col.set("Precession", True)               # sweep the tilt round a cone...
col.set("PrecessionAngle", 20.0)          # ...of this half-angle, mrad
col.set("PrecessionFrequency", 100.0)     # Hz
col.set("PrecessionDescan", True)         # descan brings the pattern back (the default)
```

## Beam tilt

Beam tilt tilts the incident beam.

- **TEM imaging.** The tilt enters the objective transfer as χ(k + k_t) − χ(k_t), with the
  envelopes tilted the same way. This gives axial coma, tilt-induced defocus and
  astigmatism, and an image shift of about defocus × tilt. The shift is what the autofocus
  beam-tilt pairs measure: the displacement between +τ and −τ is 2·Δf·τ.
- **Diffraction (SAED and CBED / 4D-STEM).** A tilted beam moves the whole pattern by
  tilt / (mrad per pixel), and it tilts the Ewald sphere relative to the crystal. Tilting the
  beam towards −g by the Bragg angle excites +g.

## Precession

Precession is the beam-tilt coils driven round a cone, with descan below the specimen
bringing the pattern back. The twin models it that way.

- **What is averaged.** SAED and kinematic 4D-STEM patterns (PED at every scan point) average
  over the tilts swept during each frame. The cone is sampled at 24 tilts around the static
  beam tilt.
- **Short frames.** A frame shorter than one precession period covers only part of the cone,
  as on a real instrument. The phase advances with time, and it is quantised to the 24 cone
  tilts so partial frames reuse cached patterns.
- **Effect on the pattern.** Reflections further out are excited, and the intensities move
  towards kinematic values.
- **Without descan.** The pattern rides a ring of radius θ/λ. Only the Bragg spots and the
  direct beam sweep; rings and the diffuse background stay centred, which is a
  simplification.
- **Imaging is not affected.** TEM imaging does not precess, and its caches are unaffected.
  Hollow-cone imaging is not modelled.
- **Coherent STEM.** The coherent (multislice) STEM model does not sweep its probe. Only its
  kinematic Bragg add-on sees precession.
- **Noiseless summaries.** `Renderer.datacube`, `virtual_image` and `ground_truth` show the
  average over the whole cone.
