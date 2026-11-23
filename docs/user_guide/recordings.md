# Replaying a recording

The twin can serve a recorded dataset in place of its simulated specimen: every face then
serves the recording's frames — the deapi face, and the shared memory that feeds a real
DE-Server — read where the scan puts the beam.

```python
twin.load_recording(r"D:\data\20240223_02783_movie.mrc", side=256)
...
twin.unload_recording()
```

or `de-twin serve --shm --recording <path> [--recording-side N]`.

## What it reads

| Recording | Scan size from | Frame time from |
|---|---|---|
| DE-Server movie (`…_movie.mrc`) | its `…_info.txt` (`Scan - Size X/Y`) | the info file |
| EDAX patterns (`.up1`, `.up2`) | a version-3 header, else the info file, else the `.ang` beside it | the info file |
| Autopilot run folder (`dataset.json`) | the record's field; each frame at its recorded point and pass | the info file |

Where an external controller drove the scan, DE-Server's own scan size only says how it
counted frames, and the scan size has to be given (`side=`; without it
`de_twin.recording.SideNeeded` is raised). Files are read with plain reads of contiguous
runs of frames into reused buffers, never mapped, so a recording far larger than memory
plays for as long as it is asked to with flat memory.

## Where each frame comes from

* **The sensor is the recording's frame.** A hardware ROI is cut from its middle; binning
  sums.
* **Every scan point lands on the recorded point under it**, through the coil voltages its
  scan asks for (size, `Scan - Pixel Offset X/Y`, voltage range, rotation, flips). A scan of
  the recorded size replays it point for point; a smaller one has a larger step and
  subsamples the same field; ROIs and XY arrays read the points under them; a point off the
  recorded field is zeros. Through DE-Server's shared memory the voltages are DE-Server's
  own.
* A frame with no scan is the field's middle point. With the beam off the camera (a dark
  reference, the beam blanked) the frames are zeros: the recording's dark was taken off
  before it was saved, and its frames are never corrected again.

## When: recorded time

A movie of repeated scans records an experiment as it happened, and a frame is served from
the scan the recording was on at **recorded time**:

* It does not start until the first acquisition that is not a search
  (`twin.searching(True)` around a search).
* While an acquisition runs, each frame moves it on by the recording's frame time:
  acquiring continuously plays scan 1, 2, 3… back to back.
* Between acquisitions it moves with the wall clock.
* `twin.series_started()` puts it back to 0.
* Past the end, the last scan repeats; a recording of one scan is that scan every pass.

The twin sees only acquisitions, so the driver says when a series starts and when its
acquisitions are searches.
