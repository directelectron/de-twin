# EBSD in an SEM

The twin can be an SEM with a Direct Electron camera as its electron backscatter diffraction
(EBSD) detector: a bulk polished specimen tilted to 70°, scanned point by point like 4D-STEM,
one Kikuchi pattern per scan point. de-twin produces the patterns and the ground truth (phase,
grain, orientation, pattern centre per point); index them with the tool of your choice.

```python
from de_twin import DigitalTwin
from de_twin.state import ScanRequest

twin = DigitalTwin("EBSD duplex steel", camera="Celeritas")
twin.configure_ebsd(beam_kv=20.0, sample_tilt_deg=70.0, pcx=0.5, pcy=0.3, pcz=0.6,
                    probe_current_na=0.1)
req = twin.request(hw_binning=(4, 4), frame_time_s=1e-3, total_frames=100 * 80,
                   scan=ScanRequest(enabled=True, size=(100, 80), step_um=0.5, dwell_s=1e-3))

for raw, meta in twin.frames(req):        # raw camera frames, 256 x 256, as DE-Server sends them
    ...
patterns = twin.ebsd_patterns(req)         # noiseless stack (80, 100, 256, 256), e- / binned px / s
xmap = twin.ground_truth(req)              # orix CrystalMap: phase, orientation, grain_id, pcx/pcy/pcz,
                                           # band_contrast per scan point (x, y in um)
```

`configure_ebsd` sets `OpticsConfig.ebsd`, an {class}`~de_twin.ebsd.EbsdGeometry`: a frozen
dataclass of plain numbers (`to_dict` / `from_dict` / `replace`, `describe()` for ranges), so a
GUI can persist it. With it set, every acquisition is `RenderMode.EBSD`; `enabled=False`
returns the twin to a TEM. The beam energy is the column's high tension (`beam_kv`; 5–30 kV).
The deapi face exposes the geometry as `Simulator EBSD ...` properties.

The pattern stack is `ny * nx * h * w * 4` bytes; pass `out=np.memmap(...)` (or `dtype`) to
`ebsd_patterns` for large scans.

## Specimens

`de_twin.specimen.BULK_PRESETS`, by name:

| preset | phases | microstructure |
| --- | --- | --- |
| `EBSD Ni polycrystal` | FCC Ni | 8 µm equiaxed grains, random texture |
| `EBSD Al cube texture` | FCC Al | 20 µm grains, {001}<100> cube texture (10°), polishing scratches |
| `EBSD ferrite gamma fibre` | BCC Fe | 10 µm grains, <111>‖ND fibre (10°) |
| `EBSD duplex steel` | BCC ferrite 55 %, FCC austenite 45 % | 6 µm grains, scratches, light damage |
| `EBSD Si (001) wafer` | Si | single crystal, [001] normal, [110] along x: pattern-centre calibration |
| `EBSD Si (001) Vickers indent` | Si | the wafer with a 10 µm Vickers indent (diagonals along ⟨110⟩) |
| `EBSD Si (001) Berkovich indent` | Si | the wafer with a Berkovich indent |

A {class}`~de_twin.specimen.BulkSurface` is an infinite polished surface: a Voronoi
tessellation of a hashed jittered lattice (deterministic, evaluated point by point), a phase
per grain from the phase fractions, and an orientation per grain drawn from the phase's
{class}`~de_twin.crystal.Texture`, unique to the grain. Build your own with
{class}`~de_twin.specimen.BulkSurfaceConfig`. A TEM specimen also works in EBSD mode (its
field map's phases and grain orientations, no boundary mixing).

## Conventions

- **Sample frame** (orientations are reported in it): `x_s` = scan x (right), `y_s` = scan y
  (down), `z_s` into the sample. The tilt axis is `x_s`; the detector is at the top of the SEM
  image. Orientations are orix rotations: crystal→sample matrix `G = R.to_matrix().T`, exactly
  as everywhere else in de-twin (`crystal_to_lab`).
- **Detector frame**: `x_d` along pattern columns, `y_d` down the rows, `z_d` from the source to
  the screen. The pattern is the screen as seen from the sample: the bright forward-scatter
  region is at the bottom.
- **Pattern centre: Bruker convention.** `pcx` from the left in units of the sensor width,
  `pcy` from the top in units of the sensor height, `pcz` = detector distance / sensor height,
  all for the whole unbinned sensor (binning and ROI leave them unchanged).
  `de_twin.ebsd.convert_pc` gives the EDAX/TSL, Oxford and EMsoft (v5) forms.
- **The pattern centre follows the beam**: `EbsdGeometry.pcx/pcy/pcz` are for the beam on the
  optic axis (scan centre) at `calibration_wd_mm`. A beam offset `(dx, dy)` on the sample moves
  the source by `S2D @ (dx, dy, 0)`: `pcx += dx/(W p)`, `pcy -= sin(σ−θ) dy/(H p)`,
  `pcz += cos(σ−θ) dy/(H p)` (σ sample tilt, θ detector tilt, p pixel pitch). A working
  distance change moves it too. The ground truth carries the pattern centre of every point.
- **Scan step** is on the tilt-corrected sample surface (`ScanRequest.step_um`, else
  `EbsdGeometry.step_um`).
- **Strain** (indented specimens): the deformation gradient `F` in the sample frame maps the
  reference lattice to the deformed one (`u = F u0`); `F = R U` with `strain = U − I` and
  `rotation` the rotation vector of `R` (radians). A deformed crystal's pattern is the reference
  pattern read along `F⁻¹ r`: zone axes move with `F`, plane normals with `F⁻ᵀ`.

## Model

- **Master patterns**: kinematic, one per phase and beam energy, built from the crystal
  library's structure factors (Lobato scattering factors, Debye–Waller) and cached
  process-wide. Every ±g pair gives a band of width 2θ_B and weight |F_g|²; higher orders nest
  inside. Band edges are blurred by the backscatter energy spread (`energy_spread_frac`) and the
  screen's angular blur, with an excess line inside and a deficiency outside each edge.
  Standardised and the bright tail log-compressed, they are stored as uint8 on a square-Lambert
  hemisphere (kinematic patterns are centrosymmetric; this is also the exchange format) and on
  a gnomonic **cube map**, which the renderer reads (one reciprocal per lookup instead of a
  square root, a division and an arctangent): 2.7 MB per phase at 0.15°, ~1 s to build.
- **Projection**: per scan point, one threaded numba pass over the binned detector grid composes
  the grain orientation, the deformation gradient, the sample and detector tilts and the point's
  own pattern centre, samples each binned pixel (`supersample` × `supersample` points), adds the
  background and writes the detector's Poisson means as it goes (`Detector.expose(means=)`):
  no full-sensor render and no second pass over the frame.
- **Signal**: backscatter yield (Arnal, by Z and tilt) × probe current × an angular
  distribution (a cosine about the surface normal plus a forward lobe about the specular
  direction, sharper at higher kV) × each pixel's solid angle (cos³γ / DD²), shadowed below the
  sample surface; bands modulate it by `1 + band_contrast × quality × m`. The default
  `band_contrast = 0.06` puts strong bands ~9–11 % and zone axes ~15 % above the background in
  raw patterns, as in real raw EBSD patterns (5–15 %).
- **Grain boundaries**: within the interaction footprint (0.02 × the Kanaya–Okayama range across
  the tilt axis, 1/cos σ longer along it; ~30 × 80 nm for Ni at 20 kV) the neighbouring grain's
  pattern is mixed in by the fraction of the footprint across the boundary. That alone touches
  only scan points within the footprint (a dotted line at 0.5 µm steps), so bulk specimens also
  have a **boundary zone** (`boundary_zone_um`, 0.2 µm σ; quality −50 % on the boundary):
  grooves, relief and distorted lattice next to a boundary, which is what makes boundaries
  continuous dark lines in real band-contrast maps. Surface damage and polishing scratches lower
  `quality` too.
- **Noise**: the camera's own detector model. At 20 kV an electron stops in the sensor and
  deposits all its energy (`CameraModel.full_stop_kv`), ~1600 ADU on a Celeritas: an
  integrating camera holds about one electron per pixel per frame, so EBSD runs at low current
  (default 0.1 nA), binned, or summed over frames.

What it does not have, compared with EMsoft dynamical master patterns: no Bloch-wave
intensities or channelling (band intensities are kinematic |F|², not dynamical), no Monte-Carlo
depth and energy distribution of the backscattered electrons (one effective energy and a fixed
edge blur), no HOLZ rings or fine zone-axis structure, no anomalous absorption, and excess /
deficiency edges that are symmetric and fixed in the crystal rather than set by the scattering
geometry. The background is a smooth heuristic, not a Monte-Carlo result. For dynamical
patterns, load your own master (below).

## Your own master patterns

```python
twin.set_master_pattern("Nickel", "Ni-master-20kV.h5")          # EMsoft EMEBSDmaster output
twin.set_master_pattern("Nickel", "ni_mp.hspy", energy_kv=20)   # kikuchipy EBSDMasterPattern.save()
twin.set_master_pattern("Silicon", array, projection="lambert", hemisphere="both", kv=20.0)
twin.master_patterns()                                           # what is in use, sizes, checks
twin.clear_master_patterns()                                     # back to kinematic
```

- **Formats**: EMsoft HDF5 (`EMData/EBSDmaster/mLPNH` / `mLPSH`, or the stereographic
  `masterSPNH`; per-site sets summed; `EkeVs`), kikuchipy / HyperSpy HDF5
  (`Experiments/<title>/data` with hemisphere and energy axes, `projection` / `hemisphere`
  attributes), or a NumPy array — (n, n), (2, n, n) north/south, (nE, n, n) with `energies`,
  (2, nE, n, n) — square Lambert or stereographic.
- **Energy**: `energy_kv` picks (interpolates) an energy bin; otherwise the Monte-Carlo
  energy-weighted sum when the file has the histogram (`EMData/MCOpenCL/accum_e`), else the
  highest energy. The master is used within 0.5 kV of its `kv`.
- **Hemispheres**: a southern hemisphere that is the inversion of the northern one is dropped;
  otherwise both are kept (non-centrosymmetric phases; six cube faces) and the south is read at
  Lambert(x, y, |z|), as EMsoft stores it.
- **Checks**: the file's space group must be the phase's; the pattern must be invariant under the
  phase's point group (`symmetry_error`; a transposed or mirrored projection is refused).
- **Memory**: resampled and quantised to `EbsdGeometry.master_resolution_deg` on load (an EMsoft
  file is read one energy slice at a time): 2.7 MB centrosymmetric, 5.4 MB with two
  hemispheres at 0.15°.
- `de_twin.ebsd.io.write_emsoft` writes the EMsoft layout (used by the round-trip test: a
  kinematic master exported and re-imported gives the same patterns, correlation > 0.998).
- **deapi**: `Simulator EBSD Master Pattern Phase`, `... File` (a path on the twin's machine:
  deapi properties are strings, so there is no upload) and `... Status`.

## Indents and HR-EBSD

`EBSD Si (001) Vickers indent` (and the Berkovich preset; {class}`~de_twin.specimen.indent.Indent`
for your own) carries the residual field of an indent
({mod}`de_twin.specimen.indent`): outside the plastic zone a traction-free surface field with a
deviatoric part decaying as 1/r² (radial compression, hoop tension, with the indenter's and the
cubic crystal's four-fold modulation), a Yoffe-blister dilatation decaying as 1/r³, and the
surface tilt of the residual uplift (2° at the plastic-zone edge, 1/r²) plus a small twist; a
plastic zone (1.8 × the contact size) with smoothly varying random rotations up to 3° and falling
pattern quality; the amorphised impression; radial cracks along ⟨110⟩ from the corners (dark
lines and a mode-I crack-tip field) and a chipped lateral-crack sector. The strain distorts the
patterns (`F⁻¹` per pixel direction) and the rotation moves them, so cross-correlation HR-EBSD
measures them. `ground_truth` adds `deformation_gradient`, `strain`, `rotation` and `gnd` (a GND
density proxy: the in-plane Nye tensor norm over b, 1/m²).

The test suite runs a test-only Wilkinson / Britton-Wilkinson HR-EBSD (FFT cross-correlation of
7 × 7 ROIs, remapping, 8-parameter deviatoric fit) against a reference 57 µm away: on
full-resolution 1024² Celeritas patterns it recovers the deviatoric strain to 1.3e-4 RMS and the
rotation to 0.007° RMS at 4000 e⁻ per pixel of shot noise (4e-5 and 0.002° at 16 000;
3.6e-5 and 0.001° noiseless). With kinematic bands (no dynamical fine structure) that is
the dose it takes; below ~1000 e⁻/px the cross-correlation fails.

## Speed and memory

On a 48-thread workstation (duplex steel on a Celeritas, masters built):

| pattern | noiseless | camera frames | noiseless stack |
| --- | --- | --- | --- |
| 128² | 0.44 ms (2260/s) | 0.56 ms (1800/s) | 13 000/s |
| 256² | 0.55 ms (1800/s) | 0.72 ms (1390/s) | 4 200/s |
| 512² | 1.38 ms (730/s) | 1.36 ms (735/s) | 925/s |

A scan costs ~80 bytes per point (+72 with strain) plus a copy of its phases' cube maps
(1.8 MB each); streaming 512² frames peaks at ~400–500 MB RSS, mostly the detector's maps.
Master patterns are built on first use of a phase and energy (~1 s) and kept up to 64 MB.

The test suite checks the geometry (zone axes and band edges where the geometry puts them, the
pattern centre moving with the beam), round-trips the ground truth, indexes noisy frames with a
test-only dictionary + refinement (~0.06° median at ~45 e⁻ per binned pixel), round-trips
master patterns through the EMsoft and kikuchipy layouts, and recovers the indent's strain with
HR-EBSD.
