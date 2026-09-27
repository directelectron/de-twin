# de-twin architecture

`de-twin` is a digital twin of a Direct Electron camera on a TEM: the column,
the specimen, the in-situ holder and the detector, advancing on one clock. It
is a standalone Python package, not part of DE-Server. It replaces the
VirtualSpecimen module that was compiled into DE-Server's CameraServer.

## Goals

1. **Realistic enough to develop automation against.** Stage, magnification,
   defocus, tilt, beam/image shift, spot size and illumination all change the
   frames the way a real column does. Frames are raw detector output, with a
   dark offset, a fixed pattern, read noise, gain non-uniformity, bad pixels,
   shot noise and saturation. That means dark/gain references are meaningful
   and must be taken.
2. **Usable with nothing else installed.** In-process numpy frames, no DE-Server,
   no microscope, no vendor simulator.
3. **Connects to real software in both directions.** Each part of the twin can
   either *own* its state (simulate it) or *mirror* real software:
   - microscope: simulated `Column`, or `MirrorColumn` following a real (or
     Dummy) DE-TEM-Channel over SOAP
   - holder: simulated `SimHeatingHolder`, or `ImpulseFollower` following a
     DENS Impulse holder or DENS's own simulator via impulsePy
   - camera: frames go to DE-Server over shared memory and come out of the full
     DE-Server pipeline, or to the deapi fake server, or straight to the caller.

## Picture

```
                 state sources                      frame sinks ("faces")
              +------------------+
 DE-TEM-Chan. |  Column          |---+         +--> in-process numpy (DigitalTwin.frames)
 (real/Dummy) |  | MirrorColumn  |   |         |
     SOAP --->|                  |   v         +--> SharedMemory ring --> DE-Server GrabberSim
              +------------------+ +--------+  |        ("External Frame Source" test pattern)
 impulsePy -->|  Holder          | | Digital|--+
 (DENS/sim)   |  | ImpulseFollwr |>|  Twin  |  +--> deapi TwinFakeServer (DE-Server emulation)
              +------------------+ | clock  |
              |  Specimen        |>|        |---> TEM-Channel SOAP face on :5002
              +------------------+ +--------+     (DE-Server, de_microscope, autopilot and
              |  Detector model  |                 groundcrew all drive the twin's column)
              +------------------+
```

A single process (`de-twin serve`) can host the twin and every face at once.
Because there is one column, moving the stage over SOAP changes the frames seen
by DE-Server, by the deapi fake server and by in-process callers alike.

## Frame pipeline

```
MicroscopeState + AcquisitionRequest + Calibration --derive_optics--> OpticsState
Specimen.rasterize(OpticsState.view, layers, time) ---------------> FieldMap
Renderer(TEM | SAED | STEM).render(OpticsState, FieldMap, ...) ---> flux [e-/px/s], float32, ROI shape
Detector.expose(flux or None, exposure_s, request, frame_index) --> raw ADU frame (uint16/uint8), binned
```

* **flux** is the expected number of electrons per detector pixel per second,
  before the detector. `None` means no beam reached the detector: a dark
  reference, beam blanked, camera retracted, or column valves closed.
* The Detector adds everything the camera would add: Poisson arrivals,
  per-electron charge spread and MTF, gain map, dark offset and fixed pattern,
  dark current, read noise, bad/hot pixels, saturation, binning, and
  quantisation to the model's bit depth. Counting cameras (Apollo-type) emit
  counted events instead.
* Dark and gain references come out right because the twin honours
  `AcquisitionRequest.exposure_mode`, `beam_blank_cmd` and `shutter_cmd`.

## Module ownership and interfaces

### `de_twin.state` and `de_twin.hashing` (core, fixed)
Dataclasses: `MicroscopeState`, `HolderState`, `AcquisitionRequest`,
`ScanRequest`, `Roi`, `FrameMeta`. Enums: `ProbeMode`, `TemStem`, `Projection`,
`ExposureMode`, `RenderMode`. Units are in the field names.

The hashing helpers `hash_seed`, `splitmix64`, `mix_cell`, `uniform_from_hash`,
`normal_from_hash`, `fnv1a64` and `rng_for` are vectorised over uint64 arrays.
Hot loops use hashes. Generation-time streams use `rng_for(seed, kind, index)`.

### `de_twin.specimen`
Fixed contract: `materials.py`, `fieldmap.py` (`ViewWindow`, `FieldMap`, `GrainTable`).

`GrainTable` holds a full 3D orientation per grain id (`quaternions`, orix/diffsims
convention, see `de_twin.crystal`) plus `nucleation_u`; grain ids are
`material * 512 + k`. Orientations are drawn from `rng_for(seed, GRAIN, id)` with a
per-scene texture: uniform random (particles, polycrystals), a `<111>` fibre (~3 deg)
for deposited thin films, and a single crystal near a low-index zone (Si [110]) for
FIB lamella matrices. There is no tilt layer: misorientation follows from the real
orientation and the stage tilt at render time.

```python
@dataclass
class SpecimenConfig:
    seed: int = 42
    holder: str = "mesh_grid"         # mesh_grid | waffle_grid | fib_liftout | insitu_heating_chip
    preparation: str = "nanoparticles"  # nanoparticles | proteins | thin_film | bulk | time_evolving
    film: str = "auto"                # holey | lacey | continuous | none | vitreous_ice | auto
    options: dict = {}                # every VirtualSpecimen knob, snake_case, see specimen/options.py

class Specimen:
    def __init__(self, config: SpecimenConfig): ...
    config: SpecimenConfig
    grains: GrainTable
    def update(self, time_s: float, holder: HolderState | None) -> None   # advance scene clock / thermal budget
    def rasterize(self, view: ViewWindow, layers: frozenset[str] = frozenset()) -> FieldMap
    def crystallinity(self, grain_id: np.ndarray) -> np.ndarray   # in-situ: 0 glassy .. 1 crystalline
    def bounds_um(self) -> tuple[float, float, float, float]
    def features(self) -> list[Feature]            # navigation helpers
    def nearest_feature(self, x_um, y_um) -> Feature | None
    def overview(self, shape=(1024, 1024)) -> np.ndarray   # whole-holder thickness image (debug)

PRESETS: dict[str, SpecimenConfig]       # the 12 VirtualSpecimen presets
PATTERNS: dict[str, SpecimenConfig]      # the 6 "Virtual Specimen - ..." test patterns
def from_name(name: str) -> SpecimenConfig
```

### `de_twin.optics`
Fixed contract: `state.py` (`OpticsState`).

```python
@dataclass
class OpticsConfig: ...        # fallbacks, offsets, flips, render-mode override, aperture, eucentric height
class Calibration:             # magnification -> nm/px and camera length -> 1/nm/px, per mag mode
    @classmethod
    def from_mag_yaml(cls, path) -> "Calibration"   # DE-Server configurations/mag.yaml
    @classmethod
    def default(cls) -> "Calibration"               # built-in, JEOL-like ladders
    def specimen_pixel_nm(self, state, camera) -> float   # per unbinned detector pixel
    def recip_pixel_inv_nm(self, state, camera) -> float
    mag_ladder(mode) / cl_ladder()
def beam_current_pa(state) -> float
def derive_optics(state, request, camera: CameraModel, calibration, cfg) -> OpticsState
```

Aberrations (`aberrations.py`, CEOS notation, complex nm; `chi`, `gradient`) are wired by
`derive_optics` into `OpticsState.image_aberrations` (the column's image-side residuals, or
an uncorrected C3, plus C1 = effective defocus and A1 = objective stigmator) and
`OpticsState.probe_aberrations` (probe side, plus C1 and A1 = condenser stigmator,
`OpticsConfig.condenser_stig_nm_per_unit`). The TEM CTF evaluates the image set, and the
coherent STEM probe evaluates the probe set. `image_aberrations_of` / `probe_aberrations_of`
fall back to `cs_mm`/`defocus_um`/`astigmatism_nm` for a hand-built `OpticsState`.
`OpticsState.source_size_nm` is the effective source FWHM at the specimen (STEM spatial
coherence). `extras["hw_binning"]` carries the detector binning.

### `de_twin.crystal`
Crystallography on diffsims + orix.

* `phases.py`: full-unit-cell orix `Phase` (diffpy `Structure`) per crystalline material
  (`phase_for`); `register_cif(path)` adds a CIF phase as a new material.
* `library.py`: `CrystalLibrary(phase)` computes the reciprocal lattice to
  `max_g_inv_nm` (25 1/nm) with kinematic `|F|^2` and Debye-Waller once per phase
  (`library_for`, cached; optional on-disk cache via `DE_TWIN_CRYSTAL_CACHE`).
  `excite(matrices, lambda, t, kV, convergence)` is vectorised over orientations x
  reflections: exact Ewald excitation error (diffsims' formula), two-beam kinematic
  intensity `(pi t / xi_g)^2 <sinc^2(pi t s)>` averaged over the illumination cone.
  Positions, hkl and relative intensities match `SimulationGenerator.calculate_diffraction2d`
  exactly (`tests/test_crystal.py`). `template_library(resolution_deg)` gives a diffsims
  template library over the reduced fundamental zone for pyxem orientation mapping.
* `orientation.py`: conventions. Lab = detector frame (x column, y row, beam along -z,
  as diffsims). A grain rotation `R` is the diffsims `rotation`: crystal->lab is
  `R.to_matrix().T`. Stage tilt `S = Rx(alpha) @ Ry(beta)` (alpha about x foreshortens y,
  beta about y foreshortens x, matching `ViewWindow.cos_alpha/cos_beta`); the effective
  crystal->lab matrix is `S @ R.to_matrix().T`.

Normalisation of a pattern (per material/grain/thickness bucket, before the
`T = exp(-t/Lambda)` mass-thickness split whose `1 - T` becomes the screened-Rutherford
background): Bragg fraction `D = 0.75 (1 - exp(-P/0.75))` with `P` the kinematic sum,
spots `c D I_g / P`, glassy halo `(1 - c) A` (in-situ crystallinity `c`), direct beam the rest.
`DigitalTwin.ground_truth(request)` returns an orix `CrystalMap` of phase + effective
orientation per STEM scan point / TEM raster pixel.

### `de_twin.render`
```python
class Renderer:
    def __init__(self, specimen: Specimen, config: RenderConfig | None = None): ...
    def render(self, optics: OpticsState, *, frame_index=0, scan_point=None, time_s=0.0) -> np.ndarray
        # float32 (ny, nx) = optics.output_shape, electrons / detector pixel / second
    def invalidate(self) -> None
```
It dispatches on `optics.render_mode` to TEM imaging, SAED or STEM (parked or
4D; `scan_point=(ix, iy)` selects the probe position). Every reflection is a disk of
the column's convergence (`OpticsState.disk_radius_px = alpha / (lambda recip_px)`, PSF
limited when small) in every mode. TEM bright-field diffraction contrast is the fraction
Bragg-scattered outside the objective aperture at each grain's effective orientation. The renderer owns the
FieldMap cache (it re-rasterises only when the view or time changes) and the
diffraction-pattern LRU cache.

STEM has two models (`RenderConfig.stem_model`: `"auto"` | `"coherent"` | `"kinematic"`):

* **kinematic** (`stem.py`): cached disk patterns per (grain, thickness, crystallinity).
  It is fast, but the bright-field disk has no interference.
* **coherent** (`coherent.py`): the probe `IFFT[A(k) exp(-i chi(k))]` times the specimen
  transmission `t(r)`, then `|FFT|^2` integrated over each binned detector pixel. `t(r)`
  comes from `tem.transmission_function`, the same object model as TEM imaging, built on a
  fine world-locked grid over the scan field. Scattering beyond the object band limit
  `K_t` is added back incoherently, so totals and ADF agree with the kinematic model.
  - Partial coherence: focal spread and source size, handled as probe modes.
  - Sampling: `dx = 1/(n dk)`; the window is oversampled to hold defocused probes.
  - Rendering: patterns are computed in blocks of scan points, cached for frame-by-frame
    rendering.
  - Extra entry points: `Renderer.datacube`, `Renderer.ground_truth_ptychography` and a
    coherent `virtual_image`.

  `"auto"` picks the coherent model when the binned pattern is at most 256² (or the step is
  below 2x the probe size) and the probe window fits `coherent_max_grid`.

### Reconstruction checks (`tests/reconstruction/`)
de-twin produces data; it does not ship reconstruction methods. The test suite keeps small
numpy reconstructions (tcBF/parallax and ePIE) only to check that coherent twin data
reconstructs against `DigitalTwin.ground_truth_ptychography`.

### `de_twin.detector`
```python
@dataclass(frozen=True)
class CameraModel: name, sensor_shape, pixel_um, bit_depth, bytes_per_pixel, topology,
                   hardware_counting, max_fps, adu_per_electron, read_noise_adu, dark_offset_adu, ...
CAMERAS: dict[str, CameraModel]      # DE16, DE64, Celeritas, Apollo, DESim, ...
def camera(name) -> CameraModel

class Detector:
    def __init__(self, model: CameraModel, seed: int = 0, **defects): ...
    model: CameraModel
    def output_shape(self, request) -> tuple[int, int]          # hw_frame (after ROI and binning)
    def roi(self, request) -> Roi                               # resolved hardware ROI
    def expose(self, flux, exposure_s, request, frame_index) -> tuple[np.ndarray, dict]
        # flux: float32 (roi.h, roi.w) e-/px/s, or None for no beam
        # returns raw frame (uint16 or uint8) with the hw_frame shape, plus info
        # {"dose_e_per_px", "saturated", "blanked"}
    dark_map / gain_map / bad_pixel_mask   # ground truth for tests
```

### `de_twin.column`
```python
class Column:                        # owns a MicroscopeState and behaves like a column
    state() -> MicroscopeState       # snapshot (stage motion interpolated to "now")
    get(name) / set(name, value)     # DE-TEM-Channel property names
    set_stage(x=None, y=None, z=None, alpha=None, beta=None) ...  # plus typed setters for everything
    on_change(callback)
    mag_ladder, cl_ladder            # snapping like the Dummy instrument
class MirrorColumn(Column):          # follows a real DE-TEM-Channel (SOAP client); sets are forwarded
class SoapTemChannelClient           # stdlib-only SOAP client (executeBatchCommands, get*, set*)
class ColumnAdapter                  # de_microscope/autopilot SimColumn duck type
class Corrector                      # column.corrector when Column(corrector="probe"|"image"|"both")
```

#### Aberration corrector (`column/corrector.py`)
`Corrector` holds one `CorrectorUnit` per fitted side. Each unit models one
CEOS-RPC server as DE-TEM-Channel's `CorrectorService` sees it: probe/CESCOR on
7072, image/CETCOR on 7071.

The ground truth is `residual = native + offset + drift`, as complex nm in CEOS
notation (`optics.aberrations`):
- `native` is the uncorrected lens.
- `offset` is the corrector's elements.
- `drift` is a random walk on a fixed time grid, so its value is deterministic
  however often it is read.

`Column.state()` copies the residual into `probe_aberrations` and
`image_aberrations` (left empty for a side with no corrector), and sets
`corrector`. The residual excludes the operator's C1/A1 (defocus + stage z,
and the objective or condenser stig), which `derive_optics` adds. Measurements
see the total aberration.

Commands are posted and finish lazily on the column's clock. The unit shares
the column's `RLock`, so there is no lock-order inversion with `state()`. The
unit's `_advance()` runs a finished command at its completion time.

The tableau does not render images. It evaluates the second derivatives of chi
at the (mis-calibrated, jittered) tilts and adds noise, then fits the linear
model up to `maxFit`. So truncation, angle and measurement-range effects follow
from the physics, not from hand-made error tables.

`Column._set_local` compares `(HT, probe mode, function mode, TEM/STEM)`
before and after each write, and calls `Corrector.on_ht_change` or
`perturb("mode")`.

The SOAP face (`TemChannelServer._corrector_handlers`) serves one unit, the
probe unit unless `corrector_side="image"`, which matches the channel keeping
one endpoint. `MirrorColumn.poll_corrector` reads a real channel's corrector
every `corrector_every` polls and subtracts the operator's C1/A1 from the
measured set.

### `de_twin.holder`
The holder duck type matches autopilot's `SimHeater` and `ImpulseHeater`:
`real`, `reason`, `channels`, `busy`, `read()`, `set()`, `ramp()`, `stop()`,
`flag()`, `describe()` and `close()`, plus `state(t) -> HolderState`.
Implementations: `NoHolder`, `SimHeatingHolder` (first-order thermal model,
Pt heater resistance, optional bias), and `ImpulseFollower` (impulsePy).

### `de_twin.transport`
`shm_layout.py` is the protocol v2 layout (mirror of `cpp/ExternalFrameSource.h`).
`shm.py` provides `FrameProducer` (the twin: opens the mapping DE-Server created and
publishes frames) and `FrameConsumer` (a Python stand-in for DE-Server, used by tests).

### `de_twin.faces`
* `temchannel_soap.py`: an HTTP SOAP server that impersonates DE-TEM-Channel
  on port 5002. It is backed by a `Column`, and DE-Server polls it through
  `executeBatchCommands`.
* `deapi_server.py`: `TwinFakeServer`, a subclass of
  `deapi.simulated_server.fake_server.FakeServer` that renders through the
  twin, plus a launcher with deapi's CLI (positional port and "started" banner).
* `shm_face.py`: runs the producer loop. It reads DE-Server's request, renders
  frames and publishes them.

### `de_twin.twin`
`DigitalTwin` ties everything together on one `Clock`. The clock is real-time
or virtual, with a time scale, so in-situ experiments can be fast-forwarded.

```python
twin = DigitalTwin(specimen="Dense Au on holey C", camera="DE16")
twin.column.set("Magnification", 50_000)
for frame, meta in twin.frames(AcquisitionRequest(frame_time_s=0.05, total_frames=10)): ...
twin.acquire(request) -> np.ndarray      # summed
twin.flux(request) -> np.ndarray         # noiseless ground truth
twin.datacube(request) -> np.ndarray     # noiseless 4D-STEM (scan y, x, binned det y, x), e/px/s
twin.ground_truth_ptychography(request)  # object t(r), probe modes, positions, aberrations
twin.stem_model(request) -> str          # "coherent" | "kinematic"
```

### DE-Server side (`cpp/`)
DE-Server's only twin-related code is `ExternalFrameSource.h/.cpp` (about 220 lines)
plus a GrabberSim test pattern called "External Frame Source (Shared Memory)"; see
`cpp/README.md`. On each acquisition GrabberSim publishes the request (frame size, ROI,
binning, frame time, frame count, exposure mode, scan size) and copies each frame into
its grab buffer. From there, frames follow the normal path: frame-number stamp, crude
layout, `CheckBuffer` and processing. All simulation lives in de-twin.
