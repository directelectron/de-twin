# de-twin

**de-twin** is a digital twin of a Direct Electron camera on a TEM: the column, the
specimen, the in-situ holder and the detector, advancing on one clock. Develop and
test automated microscope control without a microscope, a camera or DE-Server, and
plug the same twin into the real software when you have it.

```python
from de_twin import DigitalTwin

twin = DigitalTwin("Dense Au on holey C", camera="DE16")
twin.column.set("Magnification", 50_000)
twin.column.move_stage(x=12.5, y=-3.0)   # um
image = twin.snap(1.0)                   # dark/gain-corrected electrons, like DE-Server
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting started
:link: getting_started
:link-type: doc

Install de-twin and take your first images, in-process, with nothing else running.
:::

:::{grid-item-card} User guide
:link: user_guide/index
:link-type: doc

What is simulated, serving the twin to DE-Server, deapi and DE-TEM-Channel clients,
following real hardware, the aberration corrector, and 4D-STEM data.
:::

:::{grid-item-card} Example gallery
:link: auto_examples/index
:link-type: doc

Images, diffraction, tilt series, detector frames and in-situ anneals rendered by the twin.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

`DigitalTwin`, the column, specimen, detector, holders and the faces.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

getting_started
user_guide/index
auto_examples/index
api/index
architecture
dev/index
changelog
```
