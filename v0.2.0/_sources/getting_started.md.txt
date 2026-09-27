# Getting started

## Install

```
pip install de-twin            # the twin: numpy, scipy, diffsims, orix
pip install "de-twin[deapi]"   # + the deapi face (a DE-Server stand-in for deapi clients)
```

Python 3.10 or newer. The twin is pure Python; nothing else has to be running.

## Your first images

```python
from de_twin import DigitalTwin, ManualClock

twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(), seed=0)

twin.column.set("Magnification", 20_000)
twin.column.move_stage(x=0.5, y=0.0)        # micrometres; waits for the stage
twin.column.set_defocus_um(-1.0)

img = twin.snap(1.0)                         # 1 s, split into frames, dark/gain corrected, electrons
raw = twin.raw_frame(0.025)                  # one raw uint16 frame: offset, noise, defects
truth = twin.flux(twin.request())            # noiseless electrons / pixel / second
```

* `ManualClock` makes the run deterministic and instant. The default `Clock` runs in real
  time, and `Clock(time_scale=20)` fast-forwards stage moves, holder ramps and in-situ
  anneals.
* `twin.frames(request)` yields raw frames with metadata (scan point, column state, dose)
  for any `AcquisitionRequest`: exposure mode, ROI, binning, frame time and scan.
* `de-twin list` prints the camera models and specimen presets.

## Serve it to real software

```
de-twin serve --soap 5002 --deapi 13240
```

DE-Server, de_microscope, de_autopilot and de_ground_crew talk to the twin's column
through the DE-TEM-Channel SOAP API, and any deapi client acquires images from the twin
as if from DE-Server. See {doc}`user_guide/serving`.
