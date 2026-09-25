=========
Changelog
=========

All notable changes to **de-twin** are documented here.

Fragment files in ``upcoming_changes/`` are assembled into this file by
`towncrier <https://towncrier.readthedocs.io/>`_ when a release is prepared
(see ``upcoming_changes/README.rst`` for contributor instructions).

.. towncrier release notes start

0.1.2 (2026-09-25)
==================

New Features
------------

- Apollo super-resolution: ``AcquisitionRequest(super_resolution=True)`` (deapi: ``Centroiding Mode = Super-resolution``) reads a counting camera out at 2x the ROI each way (8192² for a full Apollo frame, ~20 ms). Each event lands in one of its pixel's 2x2 sub-pixels, drawn uniformly: the rendered flux has no detail finer than a sensor pixel, so that is what a real super-resolution readout of it would show.
- Counting cameras (the Apollo family) run through numba kernels too: a 4096² counted frame in ~9 ms (the numpy path took ~90 ms).
- Pinned acquisitions: ``AcquisitionRequest(seed=...)`` (and ``snap(seed=...)``) seeds frame *i* of an acquisition by ``(seed, i)`` whatever the twin exposed before; ``DigitalTwin.reset_serial()`` restarts the running frame count.
- STEM and 4D STEM patterns with large disks (a probe's disks can outgrow the detector) are drawn row by row over the detector with a numba kernel instead of stamping each disk's whole square: a Steel 4D STEM frame goes from ~3.3 s to ~50 ms.
- Shared-memory throughput: ``ShmFace(reuse=N, threads=M)`` (``de-twin serve --shm --reuse N --threads M``) renders into a pool on its own thread and publishes each frame up to N times, with the detector physics capped at M cores; frames are copied into the mapping once. A 4096² DE16 stream reaches ~170 fps (5.7 GB/s) with ``reuse=8, threads=8``, an Apollo ~250 fps (4.2 GB/s) with ``reuse=8, threads=4``. Recycled frames are always of the current view.
- TEM imaging renders a view 15 % of the field larger on each side and serves a stage move inside that margin by cropping it: a nudge costs ~2 ms instead of a render; ``RenderConfig(pan_margin=0)`` turns it off.
- The deapi face has a read-only ``Simulator Source`` property (``de-twin <version>``) so clients can stamp provenance on what they save.
- The detector simulation runs as fused numba kernels for integrating cameras: 6x faster at 4096² (170 → 28 ms a frame) and 11x at 1024² (42 → 4 ms), with the same statistics; ``DE_TWIN_NUMPY_DETECTOR=1`` keeps the numpy path. TEM images render at up to 1024² (from 2048²), which makes a re-render after a stage move ~2.7x faster on a 4096² camera. numba is now a dependency.
- ``OpticsConfig(max_raster_pixels=...)`` sets the largest TEM raster (0: render at the frame's own sampling, for CTF and MTF measurements); ``OpticsState.resolution_warning`` says when a frame is upsampled from a coarser raster.
- ``RenderConfig(interactive=True)``: while the view keeps changing (dragging the stage, zooming), a view the pan cache cannot crop is rendered at up to ``preview_side``² and upsampled, so live view keeps up; once the view has been still for ``settle_s`` it is rendered in full.


Bug Fixes
---------

- A specimen with an amorphous material in a crystalline slot (the FIB-liftout pattern's "auto" post) no longer raises in imaging or diffraction; it scatters no Bragg beams.
- A twin now starts where an operator would: the stage over the specimen (the nearest intact grid square, well, lamella or chip window, and its largest particle cluster) instead of the stage origin, which was a grid bar or chip frame for some presets and left them black; the beam at a dose the detector takes without saturating a frame; and cryo specimens 1.5 µm underfocus. ``DigitalTwin(start_on_specimen=False)`` keeps the old start, and a supplied column is never moved.
- The column's ``mag_ladder()`` in TEM imaging lists LowMAG's and MAG1's magnifications together, the ones a ``Magnification`` write can reach, so a client stepping it (Ground Crew's mag +/-) crosses from low to medium magnification instead of stopping at the top of LowMAG.
- The deapi face caches its port at start, so a face stopped before its UDP thread binds no longer calls ``getsockname`` on a closed socket.
- The deapi face starts with a 0.5 s exposure instead of a single frame, whose sparse electrons read as noise.
- The shared-memory face renders on a thread of its own and republishes the last frame while a slow render runs, so DE-Server no longer stops a live acquisition ("no frame from the producer within 1020 ms") when a view takes over a second to render.
- The shared-memory face renders one frame to warm the twin up before it attaches, so DE-Server's first acquisition no longer times out while a fresh twin starts (about 15 s on first use); ``ShmFace.ready`` is set once it is serving.
- impulsePy is imported on a thread of its own with a timeout (``holder.impulse.import_impulse``): with Impulse not running, ``connect_holder("impulse")`` falls back to the simulated holder instead of hanging.


0.1.1 (2026-09-24)
==================

Bug Fixes
---------

- The deapi face reports "Specimen Pixel Size X/Y (nanometers)" per binned pixel, including hardware and software binning, as DE-Server does.
- The deapi face treats ``start_acquisition(0)`` as live view (repeat until stopped), as
  DE-Server does, instead of a single acquisition; and repeated acquisitions no longer repeat
  the same detector noise, because noise is now seeded by a twin-wide frame counter rather than
  the frame index within each request.


0.1.0 (2026-09-24)
==================

New Features
------------

- First release of the de-twin digital twin: a simulated TEM column (JEOL/TFS-style
  modes, magnification and camera-length ladders, stage motion, aberrations and an
  optional probe/image corrector), seeded specimens (grids, thin films, proteins in ice,
  FIB lamellae, an in-situ heating chip) with real crystal orientations from diffsims and
  orix, wave-optical TEM imaging, diffraction and coherent 4D-STEM, and a detector model
  for every Direct Electron camera. ``DigitalTwin`` runs it all in-process.
- The twin can stand in for real software: a DE-TEM-Channel SOAP server
  (``de-twin serve --soap``) that DE-Server, de_microscope, de_autopilot and
  de_ground_crew drive unchanged, a deapi-compatible fake DE-Server
  (``--deapi``), and a shared-memory frame source that feeds a real DE-Server
  through its processing pipeline (``--shm``, test pattern "External Frame Source
  (Shared Memory)"). It can also follow a real or Dummy DE-TEM-Channel and a DENS
  Impulse holder instead of simulating them.


Bug Fixes
---------

- Coherent 4D-STEM data no longer depends on the machine: the probe's partial-coherence
  modes were truncated through a degenerate pair of eigenmodes, whose basis LAPACK chooses
  differently with BLAS threading, so the same request gave different patterns on different
  computers. Truncation now keeps degenerate groups whole.


Maintenance
-----------

- Packaging, documentation site and release workflows (towncrier changelog,
  Prepare Release PR, PyPI publishing when a GitHub Release is published).
- Raised the minimum supported versions to numpy 2.0, scipy 1.13 and diffpy.structure 3.2
  (and matplotlib 3.9 for the docs), the oldest set the test suite passes with. Timing
  budgets in the performance tests now scale with ``DE_TWIN_PERF_SLACK`` so shared CI
  runners can check them without being tuned to one workstation.
