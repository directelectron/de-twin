"""
Probe convergence: spots to disks
=================================

The same gold particle in diffraction as the probe converges. Parallel illumination gives
spots; nano-beam diffraction gives disks whose radius is the convergence angle; convergent
beam disks grow until they overlap.
"""

import matplotlib.pyplot as plt
import numpy as np

from de_twin import DigitalTwin, ManualClock
from de_twin.state import ProbeMode, Projection

twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(), seed=3)
twin.column.set("Magnification", 150000)
feature = twin.specimen.features()[0]
twin.column.move_stage(x=-feature.center_um[0], y=-feature.center_um[1])
twin.column.set_projection(Projection.DIFFRACTION)

probes = [(ProbeMode.TEM, 3), (ProbeMode.NBD, 1), (ProbeMode.NBD, 4), (ProbeMode.CBD, 2)]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
for ax, (mode, alpha) in zip(axes, probes):
    twin.column.set_probe_mode(mode)
    twin.column.set_alpha_selector(alpha)
    dp = np.log1p(twin.flux(twin.request()))
    ax.imshow(dp, cmap="gray", vmin=np.percentile(dp, 1), vmax=np.percentile(dp, 99.95))
    ax.set_title(f"{mode.name}, {twin.optics(twin.request()).convergence_mrad:.2g} mrad", fontsize=9)
    ax.axis("off")
fig.tight_layout()
plt.show()
