# de-twin

[![Tests](https://github.com/CSSFrancis/de-twin/actions/workflows/tests.yml/badge.svg)](https://github.com/CSSFrancis/de-twin/actions/workflows/tests.yml)
[![Docs](https://github.com/CSSFrancis/de-twin/actions/workflows/docs.yml/badge.svg)](https://cssfrancis.github.io/de-twin/)
[![PyPI](https://img.shields.io/pypi/v/de-twin.svg)](https://pypi.org/project/de-twin/)
[![Python](https://img.shields.io/pypi/pyversions/de-twin.svg)](https://pypi.org/project/de-twin/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A digital twin of a Direct Electron camera on a TEM: the column, the specimen, the in-situ
holder and the detector, advancing on one clock. Develop and test automated microscope
control without a microscope, a camera or DE-Server, and plug the same twin into the real
software when you have it.

**Documentation:** <https://cssfrancis.github.io/de-twin/>

```
pip install de-twin            # numpy, scipy, diffsims, orix
pip install "de-twin[deapi]"   # + a deapi-compatible DE-Server stand-in
```

## Quick start

```python
from de_twin import DigitalTwin

twin = DigitalTwin("Dense Au on holey C", camera="DE16", holder="sim-heating")
twin.column.set("Magnification", 50_000)
twin.column.move_stage(x=12.5, y=-3.0)        # um, waits for the stage
twin.column.set_defocus_um(-1.5)

img = twin.snap(1.0)                          # dark/gain-corrected electrons, like DE-Server
raw = twin.raw_frame(0.025)                   # one raw uint16 frame: offset, noise, defects
truth = twin.flux(twin.request())             # noiseless e-/px/s, for scoring automation
```

## What is simulated

- **Column**: JEOL/TFS-style modes, magnification and camera-length ladders, stage motion,
  beam current and illumination, defocus, astigmatism, shifts and tilts, axial aberrations
  (C1 to C5), and an optional CEOS-like probe/image corrector with drift and tuning.
- **Specimen**: seeded grids, thin films, proteins in ice, FIB lamellae and an in-situ
  heating chip; every grain is a real crystal (diffsims + orix) that responds to tilt.
- **Image formation**: wave-optical TEM with a CTF, kinematic SAED/NBD/CBED, coherent
  4D-STEM for ptychography and tcBF, SEM EBSD (kinematic Kikuchi patterns of bulk
  polycrystals, with ground-truth orientations), and physical dose.
- **Detector**: every DE camera model, with dark, gain, noise, defects, saturation, binning
  and counting, so references are needed and work.
- **Holder**: a simulated DENS-style heating chip, or a real DENS Impulse holder.

## Plug it into real software

```
de-twin serve --soap 5002 --deapi 13240 --shm
```

| Face | Who talks to it |
|---|---|
| DE-TEM-Channel SOAP (`--soap`) | DE-Server, de_microscope, de_autopilot, de_ground_crew |
| deapi fake DE-Server (`--deapi`) | any deapi client |
| shared memory (`--shm`) | a real DE-Server with the test pattern *External Frame Source (Shared Memory)* |

Each part can also follow real hardware instead of simulating it: `--mirror-temchannel HOST`
for a real (or Dummy) DE-TEM-Channel, `--holder impulse` for a DENS Impulse holder.

DE-Server's side of the shared-memory frame source is two small files in [`cpp/`](cpp/).

## Contributing and releases

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are recorded as
[towncrier](https://towncrier.readthedocs.io/) fragments in `upcoming_changes/`, and releases
follow the procedure in the [development docs](https://cssfrancis.github.io/de-twin/dev/dev/index.html).

MIT licensed.
