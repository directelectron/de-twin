=========
Changelog
=========

All notable changes to **de-twin** are documented here.

Fragment files in ``upcoming_changes/`` are assembled into this file by
`towncrier <https://towncrier.readthedocs.io/>`_ when a release is prepared
(see ``upcoming_changes/README.rst`` for contributor instructions).

.. towncrier release notes start

0.4.0 (2026-10-26)
==================

API and Behaviour Changes
-------------------------

- Below ``CameraModel.full_stop_kv`` (40 kV) the ADU an electron deposits now falls in proportion to its energy (it stops in the sensor) instead of following the high-tension table's log-log trend, which at SEM energies gave several times more charge than the electron carries.


New Features
------------

- The twin's column became an SEM in EBSD mode: it reported an SEM's ``InstrumentType`` (``"JEOL SEM (EIK)"``), ``TemStemMode`` 1 and the new ``WorkingDistance`` / ``SampleTilt`` properties, which the DE-TEM-Channel face served too. (`#44 <https://github.com/directelectron/de-twin/pull/44>`_)
- EBSD in an SEM: ``DigitalTwin.configure_ebsd`` (``OpticsConfig.ebsd``, an ``EbsdGeometry`` with sample and detector tilt, working distance and a Bruker pattern centre that follows the beam) makes every acquisition ``RenderMode.EBSD``: kinematic Kikuchi master patterns per phase (cached, ~1 MB), projected per scan point onto the camera with a backscatter background, boundary mixing, a darker zone along grain boundaries and surface damage, through ``frames``/``flux``, ``ebsd_patterns`` and the faces (``Simulator EBSD ...`` deapi properties). ``ground_truth`` gives phase, grain, sample-frame orientation, pattern centre and band contrast per point. New bulk specimens (``BULK_PRESETS``: Ni, cube-textured Al, gamma-fibre ferrite, duplex steel, Si (001) wafer) and materials (nickel, austenite). Camera frames are made in one fused CPU pass per frame, a few frames per job on below-normal-priority worker threads (``DE_TWIN_FRAME_THREADS``): ~7,800 frames/s from ``twin.frames`` for a 512x512 binned frame on a 48-thread machine; ``DE_TWIN_FUSED_EBSD=0`` selects the general path. ``Detector.set_gain_mode("low")`` reads out at a low gain that ~100 electrons saturate, for bright SEM beams.
- EBSD master patterns of your own: ``DigitalTwin.set_master_pattern(phase, path_or_array, ...)`` reads EMsoft ``EMEBSDmaster`` HDF5 (Lambert or stereographic; an energy bin, interpolation or the Monte-Carlo energy-weighted sum), kikuchipy HDF5 or a NumPy array, keeps both hemispheres for non-centrosymmetric phases, checks the space group and point-group symmetry, and quantises it to the cache resolution (a few MB). The deapi face takes a file path (``Simulator EBSD Master Pattern File``).
- Indented Si for HR-EBSD: presets ``EBSD Si (001) Vickers indent`` and ``EBSD Si (001) Berkovich indent`` (``de_twin.specimen.indent``) carry a residual strain / rotation field, plastic zone and cracks that distort the EBSD patterns; ``ground_truth`` adds the deformation gradient, strain, rotation and a GND density proxy per scan point.


Bug Fixes
---------

- The simulated column's TEM LowMAG took a real JEOL's ladder (100x to 1500x), so 500x and 1000x became reachable, and its spot size went up to the vendor's limit (9 on JEOL, 11 on TFS) instead of stopping at 5. (`#41 <https://github.com/directelectron/de-twin/pull/41>`_)
- The deapi face took DE-Server's ``Scan - Use DE Camera`` and parked-beam properties, acknowledged ``set_xy_array`` without a 35 s stall, and clamped ``Frames Per Second`` to what the camera reaches, so a de_automate scan ran through it end to end. (`#42 <https://github.com/directelectron/de-twin/pull/42>`_)
- The shared-memory face no longer served recycled frames of the old view after a stage move or other column change, which had made an acquisition straight after a move register no shift. (`#45 <https://github.com/directelectron/de-twin/pull/45>`_)
- The twin's own column spread its beam as the operator zoomed out (the new ``OpticsConfig.intensity_zoom_out``), so frames stopped saturating at survey magnifications such as 200x and 5000x. (`#46 <https://github.com/directelectron/de-twin/pull/46>`_)
- Hardware-binned frames were rendered and exposed on the binned grid and the shared-memory face took eight numba threads by default, so a DE-Server 4D-STEM scan fed by the twin ran about three times faster on a sixth of the CPU. (`#47 <https://github.com/directelectron/de-twin/pull/47>`_)
- A parked beam behind a real DE-Server was drawn where DE-Server parked it, and followed the park when it moved. (`#48 <https://github.com/directelectron/de-twin/pull/48>`_)
- Importing ``de_twin`` sets ``NUMBA_DISABLE_CUDA=1``, so the twin never initialises CUDA (diffsims asked numba for a device on import, which started the driver in the process serving DE-Server). ``DE_TWIN_ALLOW_CUDA=1`` leaves the process alone.


0.3.0 (2026-10-23)
==================

API and Behaviour Changes
-------------------------

- Magnification and camera-length ladders are one type, ``de_twin.column.ladders.Ladder``,
  which carries both rules for picking the nearest rung (``snap_linear`` for the column's
  detents, ``snap_logarithmic`` for calibration tables). ``Column.mag_ladder()``,
  ``Column.cl_ladder()``, ``Calibration.mag_ladder()`` and ``Calibration.cl_ladder()`` return it,
  and the calibration's built-in tables moved to ``de_twin.column.ladders``.
- ``Column.mag_ladder()`` and ``Column.cl_ladder()`` now return lists, empty when the mode has
  no such ladder (they returned ``None``), and ``cl_ladder()`` is in millimeters like
  ``Calibration.cl_ladder()`` (it was centimeters). In ``de_twin.column.ladders``,
  ``DIFF_CAMLENGTHS_CM`` and ``STEM_CAMLENGTHS_CM`` became ``DIFF_CAMERA_LENGTHS_MM`` and
  ``STEM_CAMERA_LENGTHS_MM``. The ``CameraLength`` property is still centimeters, as on the
  DE-TEM-Channel protocol.
- Arguments are spelled out: ``cfg`` is ``config`` everywhere (``beam_current_pa(state,
  config=...)``, and so on), and catch-all ``**kw`` arguments have names that say what they
  carry. Several large modules were split by job, so a few names moved: the column's property
  tables, vendor tables and errors (``column.properties``, ``column.vendor``,
  ``column.errors``), the CEOS corrector (``column.ceos``), coherent-imaging setup
  (``render.coherent_setup``), TEM contrast (``render.tem_contrast``) and the deapi server's
  protocol, request, acquisition, property and result parts (``faces.deapi_*``).
- ``Calibration.default`` no longer takes ``post_magnification``, which scaled the served and the rendered pixel together and so could not express a calibration error. Set ``OpticsConfig.post_magnification`` instead. (`#39 <https://github.com/directelectron/de-twin/pull/39>`_)


New Features
------------

- ``OpticsConfig.post_magnification`` puts the camera behind a constant post-column magnification: the rendered pixel is the calibrated one divided by it at every magnification, while the served ``Specimen Pixel Size`` keeps reporting the nominal value. ``calibration_truth()`` reports the factor and the true pixel size. (`#39 <https://github.com/directelectron/de-twin/pull/39>`_)
- A scan follows DE-Server's timing for coil changes: it runs a repeat at a time, and a pixel
  offset (or voltage, rotation or flip) written mid-scan reaches the beam one
  ``Scan - Output Buffer (milliseconds)`` later, on a repeat boundary, as the new DE-Server
  property queues it. The offset is clamped to half the scan's voltage range, inside the scan
  card's +-10 V. ``get_gpu_frames()`` now maps the twin's own frames out of GPU memory, through
  DE-Server's GPU frame sharing protocol, instead of deapi's simulated stand-in.

  The shared-memory face keeps up with a fast camera: with ``reuse``, a full frame pool is a
  library that is picked from rather than rendered into, re-rendered when the column moves, and
  frames go to DE-Server a batch at a time. Its C++ reader yields instead of sleeping for the
  first few ms of a wait. A 256x256 stream reaches about 27,000 frames a second, from 580.

  In a scan, the shared-memory face follows DE-Server's scan coil voltages (``DE_ScanOutput``,
  which DE-Server's scan simulator writes as a scan card would be written): each frame is
  rendered where its point's voltages put the beam, so a pixel offset, an XY array or drift
  correction moves the beam with DE-Server's output-buffer latency. A probe held on one point
  is rendered ``pool_size`` times and then picked from (about 25,000 frames a second at 256x256).
- Calibration standards: Ted Pella's TEM calibration specimens as presets ("Ted Pella 607 - ..."), grouped in ``CALIBRATION_STANDARDS`` with their published numbers in ``specimen.standards.PRODUCTS``: the grating replicas (607, 607-A, 606, 603, 603-A, 677, 673), gold-shadowed latex (628-B), evaporated aluminum (619) and MoO3 laths tied to their crystal axis (625). The replicas are conformal carbon films shadowed with metal at an angle (embossed at low magnification; a nanocrystalline island film with dense and sparse regions and soft shadows at high magnification). New: MoO3, and an ``oriented`` grain texture.
- Faster live-view re-renders on a magnification step or a stage move: the Bragg excitation is computed once per grain and thickness and kept across views, a stage move beyond the pan margin rasterizes only the strips of specimen it brings into view (``RenderConfig.reuse_rasters``), large rasters are rasterised in parallel strips (``RenderConfig.raster_threads``), and the exit wave, crops and upsampling are threaded or numba-compiled. Padded TEM rasters now sit on a world-fixed pixel lattice, with sub-pixel view offsets applied as a phase ramp and crops interpolated to a fraction of a pixel (a move inside the margin was rounded to whole raster pixels). ``de-twin serve --interactive`` turns on the fast previews for a live view.
- STEM scans now follow DE-Server's scan coil voltages: ``Scan - Voltage H/V Min/Max`` is the
  field the full scan grid spans (a smaller range zooms in, moving it pans), and
  ``Scan - Pixel Offset X/Y`` moves the probe by whole or fractional scan points. ``Scan - Rotation``
  and ``Scan - Flip Horizontally/Vertically`` turn and mirror the grid. A live scan
  (``start_acquisition(0)``) picks up offset, voltage, rotation and flip changes at its next
  repetition, so a client can move the probe, correct drift, or pan and zoom while it scans.
- Specimen field maps can carry an amorphous layer under the primary material (``under_material``, ``under_thickness_nm``: a support film under particles, stain around protein); the renderer adds its absorption, phase, amorphous texture and diffuse scattering. New ``specimen.stamps``: organic shapes precomputed once (Cahn-Hilliard phase-separation tiles shipped as signed distances, 3D diffusion-limited aggregates) and placed fast in numba with exact coverage; ``specimen.spheres``: binned sphere sets (chords, tops, cast shadows).
- TEM lattice fringes are kinematic by default (``RenderConfig.lattice_fringe_model="kinematic"``): each resolved beam passing the objective aperture is a phase grating of its own kinematic strength, grown with thickness, times ``lattice_fringe_efficiency`` (0.5, calibrated on Ted Pella lattice images). The previous fixed amplitude (``"fixed"``, ``lattice_phase_rad``) gave every crystal the same weak fringes.


Bug Fixes
---------

- The cross grating's second and fourth orders no longer outweigh its first. The renderer removed the electrons wherever the specimen's phase changed quickly between raster pixels, and a raster pixel wider than a ridge wall (every frame larger than 1024 x 1024) put that dark rim on every edge of every hole. Rendered images change slightly wherever a specimen has steep edges: the mean frame intensity moves by under 1 % for most presets and by up to 5 % for a few. (`#39 <https://github.com/directelectron/de-twin/pull/39>`_)
- The twin no longer aborts on macOS when two threads render at once (a prefetch alongside a live frame, say). numba's only threading layer there, the workqueue, aborts the process when two threads launch a parallel kernel together, so on it the kernels are compiled serial; each still releases the GIL, and the twin's own threads run them side by side.


Removals
--------

- The ``legacy`` TEM imaging model and the ``legacy`` dose model are removed. The physical
  models were already the defaults and are now the only ones, so ``RenderConfig.tem_model``,
  ``OpticsConfig.dose_model`` and ``OpticsConfig.legacy_dose_rate_counts_per_s`` no longer exist.
- The optics state no longer has ``blur_sigma_px``, ``fresnel_gain``, ``fresnel_sign`` or
  ``fresnel_sigma_px``; only the removed legacy TEM renderer used them.


Maintenance
-----------

- Project links, the docs site URL and the trusted-publishing instructions now point at the directelectron organization (https://directelectron.github.io/de-twin/).
- Structural cleanup with no change to rendered output: one shared numba kernel decorator
  (``de_twin.parallel.numba_jit``) replaces the per-module copies, the column-realism classes
  share their derived values, and unused helpers (``log_snap``, ``resample_object``,
  ``upsample2x``, ``point_in_polygon``, ``AmorphousStructure``) are removed.


0.2.0 (2026-09-27)
==================

New Features
------------

- A tilted specimen that is off eucentric height moves in the image (by dz sin(tilt) across the tilt axis), as it does on a real stage: what eucentric-height routines measure.
- Apollo super-resolution: ``AcquisitionRequest(super_resolution=True)`` (deapi: ``Centroiding Mode = Super-resolution``) reads a counting camera out at 2x the ROI each way (8192² for a full Apollo frame, ~20 ms). Each event lands in one of its pixel's 2x2 sub-pixels, drawn uniformly: the rendered flux has no detail finer than a sensor pixel, so that is what a real super-resolution readout of it would show.
- Beam tilt reaches diffraction: a tilted beam moves the SAED / CBED pattern (tilt / mrad-per-pixel) and tilts the Ewald sphere, so tilting towards -g by the Bragg angle excites +g.
- Counting cameras (the Apollo family) run through numba kernels too: a 4096² counted frame in ~9 ms (the numpy path took ~90 ms).
- Low dose on the column: ``LowDose``, ``LowDoseArea`` (Search / View / Focus / Record) and ``LowDoseAreas``. Each area stores its magnification, spot, intensity, defocus offset from Record's focus, image shift and beam shift; switching stores the area left and applies the one entered. On a realistic column the areas do not line up by themselves, which is what SerialEM's View-to-Record low-dose calibration measures.
- Pinned acquisitions: ``AcquisitionRequest(seed=...)`` (and ``snap(seed=...)``) seeds frame *i* of an acquisition by ``(seed, i)`` whatever the twin exposed before; ``DigitalTwin.reset_serial()`` restarts the running frame count.
- Precession: the column's ``Precession``, ``PrecessionAngle`` (mrad), ``PrecessionFrequency`` (Hz) and ``PrecessionDescan`` drive the beam tilt round a cone; SAED and kinematic 4D-STEM patterns average over the tilts swept during each frame (the whole cone, or the arc a frame shorter than the period covers). Without descan the pattern sweeps round a ring.
- STEM and 4D STEM patterns with large disks (a probe's disks can outgrow the detector) are drawn row by row over the detector with a numba kernel instead of stamping each disk's whole square: a Steel 4D STEM frame goes from ~3.3 s to ~50 ms.
- Shared-memory throughput: ``ShmFace(reuse=N, threads=M)`` (``de-twin serve --shm --reuse N --threads M``) renders into a pool on its own thread and publishes each frame up to N times, with the detector physics capped at M cores; frames are copied into the mapping once. A 4096² DE16 stream reaches ~170 fps (5.7 GB/s) with ``reuse=8, threads=8``, an Apollo ~250 fps (4.2 GB/s) with ``reuse=8, threads=4``. Recycled frames are always of the current view.
- Specimen preset ``Cross grating 2160 l/mm``: a carbon replica cross grating (463 nm period), the pixel-size calibration standard.
- TEM imaging renders a view 15 % of the field larger on each side and serves a stage move inside that margin by cropping it: a nudge costs ~2 ms instead of a render; ``RenderConfig(pan_margin=0)`` turns it off.
- The deapi face has a read-only ``Simulator Source`` property (``de-twin <version>``) so clients can stamp provenance on what they save.
- The detector simulation runs as fused numba kernels for integrating cameras: 6x faster at 4096² (170 → 28 ms a frame) and 11x at 1024² (42 → 4 ms), with the same statistics; ``DE_TWIN_NUMPY_DETECTOR=1`` keeps the numpy path. TEM images render at up to 1024² (from 2048²), which makes a re-render after a stage move ~2.7x faster on a 4096² camera. numba is now a dependency.
- The realistic column (``OpticsConfig.realistic``) has a C2 crossover (the beam converges to its smallest at an Intensity that varies by spot size and probe mode, and spreads again past it) and a beam-shift matrix; ``calibration_truth()`` reports both. Closed-loop tests run SerialEM's Beam Crossover and Beam Shift calibrations against them.
- The realistic column adds image-shift-induced coma and astigmatism (image shift images the specimen off the coma-free axis: an effective beam tilt of ~0.43 mrad per micrometre, and ~15 nm A1 per micrometre) and the high-defocus change of magnification and rotation (~4 % and ~2 deg at -200 um). ``calibration_truth()`` reports them; closed-loop tests run SerialEM's autofocus, coma-vs-image-shift and high-defocus calibrations against the twin.
- ``OpticsConfig(intensity_zoom=True)``: in TEM imaging the condenser follows the magnification (like TFS Intensity Zoom), so the dose per detector pixel stays put as the operator zooms; Intensity sets the beam at ``intensity_zoom_reference_mag``. Off by default: the dose per pixel goes as 1/mag², so zooming out from a safe dose saturates the camera and zooming in drowns it in shot noise.
- ``OpticsConfig(max_raster_pixels=...)`` sets the largest TEM raster (0: render at the frame's own sampling, for CTF and MTF measurements); ``OpticsState.resolution_warning`` says when a frame is upsampled from a coarser raster.
- ``OpticsConfig.realistic(seed)``: a column with a real one's geometry, what SerialEM's calibrations measure. Per magnification an image rotation (with a large jump between LowMAG and MAG1), a true pixel size off the nominal one, an image-shift matrix (scale, skew, rotation) and an image-centre offset; stage backlash that depends on the approach direction. The ideal column stays the default. ``DigitalTwin.calibration_truth()`` reports every true value for closed-loop tests; the pan-crop cache now works for rotated views.
- ``RenderConfig(interactive=True)``: while the view keeps changing (dragging the stage, zooming), a view the pan cache cannot crop is rendered at up to ``preview_side``² and upsampled, so live view keeps up; once the view has been still for ``settle_s`` it is rendered in full.
- ``de_twin.optics.serialem``: ``to_serialem`` writes a column's calibrations as SerialEM would measure them (``SerialEMcalibrations.txt`` entries: image-shift and stage-to-camera matrices, image-shift offsets, beam-shift calibrations, crossovers, high-focus magnification and focus calibrations; plus the camera's ``RotationAndPixel`` lines), and ``from_serialem`` reads a microscope's calibrations into a ``ColumnDefinition`` that goes in ``OpticsConfig(realism=...)``, so a twin behaves like that microscope. The round trip is exact.


Bug Fixes
---------

- A specimen with an amorphous material in a crystalline slot (the FIB-liftout pattern's "auto" post) no longer raises in imaging or diffraction; it scatters no Bragg beams.
- A twin now starts where an operator would: the stage over the specimen (the nearest intact grid square, well, lamella or chip window, and its largest particle cluster) instead of the stage origin, which was a grid bar or chip frame for some presets and left them black; the beam at a dose the detector takes without saturating a frame; and cryo specimens 1.5 µm underfocus. ``DigitalTwin(start_on_specimen=False)`` keeps the old start, and a supplied column is never moved.
- In TEM imaging, beam shift moves the illuminated disc across the camera and leaves the specimen where it is (it used to move the image, like image shift). STEM and diffraction are unchanged.
- The column's ``mag_ladder()`` in TEM imaging lists LowMAG's and MAG1's magnifications together, the ones a ``Magnification`` write can reach, so a client stepping it (Ground Crew's mag +/-) crosses from low to medium magnification instead of stopping at the top of LowMAG.
- The deapi face caches its port at start, so a face stopped before its UDP thread binds no longer calls ``getsockname`` on a closed socket.
- The deapi face starts with a 0.5 s exposure instead of a single frame, whose sparse electrons read as noise.
- The shared-memory face keeps its frame pool across back-to-back requests for the same frames (DE-MC's live view is a string of short acquisitions), and republishes the last frame while a new request has nothing rendered yet, so a slow render at the start of one no longer stops DE-Server.
- The shared-memory face renders on a thread of its own and republishes the last frame while a slow render runs, so DE-Server no longer stops a live acquisition ("no frame from the producer within 1020 ms") when a view takes over a second to render.
- The shared-memory face renders one frame to warm the twin up before it attaches, so DE-Server's first acquisition no longer times out while a fresh twin starts (about 15 s on first use); ``ShmFace.ready`` is set once it is serving.
- impulsePy is imported on a thread of its own with a timeout (``holder.impulse.import_impulse``): with Impulse not running, ``connect_holder("impulse")`` falls back to the simulated holder instead of hanging.


Documentation
-------------

- User-guide pages on the realistic column (and intensity zoom), beam tilt and precession, low dose, and SerialEM calibrations; the new classes and functions are in the API reference.


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
