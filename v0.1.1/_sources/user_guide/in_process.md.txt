# Using the twin in-process

```python
from de_twin import DigitalTwin

twin = DigitalTwin("Dense Au on holey C", camera="DE16", holder="sim-heating")
twin.column.set("Magnification", 50_000)
twin.column.move_stage(x=12.5, y=-3.0)        # um, waits for the stage
twin.column.set_defocus_um(-1.5)

img = twin.snap(1.0)                          # dark/gain-corrected electrons, like DE-Server
raw = twin.raw_frame(0.025)                   # one raw uint16 frame: offset, noise, defects
for frame, meta in twin.frames(twin.request(frame_time_s=0.01, total_frames=100)):
    ...                                       # raw frames + metadata (scan point, state, dose)
truth = twin.flux(twin.request())             # noiseless e-/px/s, for scoring automation
```

## Requests

`twin.request(...)` builds an `AcquisitionRequest`: exposure mode (`Dark`, `Gain`,
`Normal`, …), frame time, frame count, hardware ROI and binning, and a `ScanRequest` for
STEM. The same request type comes from DE-Server over shared memory or from deapi
properties, so code written against one path works on the others.

## References

`twin.processor` measures dark references through the twin, the way an operator would,
and provides a well-dosed gain reference. `twin.snap()` and `processor.acquire()` apply
them. Acquisitions in `ExposureMode.DARK` or `GAIN` measure references frame by frame, so
under-dosed or saturated references behave as they do on a real camera.

## Clocks

Every time-dependent part (stage motion, holder ramps, in-situ evolution, corrector drift,
frame pacing) reads one clock:

- `Clock()`: real time (the default).
- `Clock(time_scale=20)`: 20 simulated seconds per wall second.
- `ManualClock()`: time moves only when you advance it (or when paced code sleeps), so runs
  are deterministic and instant.

## Existing duck types

Code written for de_autopilot or de_ground_crew keeps working:
`de_twin.column.ColumnAdapter(twin.column)` looks like their `SimColumn`, and
`twin.holder` looks like autopilot's `SimHeater` / `ImpulseHeater`.
