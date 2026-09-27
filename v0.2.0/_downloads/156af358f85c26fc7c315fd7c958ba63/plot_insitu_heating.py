"""
In-situ heating
===============

A simulated heating chip is stepped to 650 C. Every droplet is its own crystal, and each
nucleates as the thermal budget accumulates: the glassy halo gives way to spotty powder
rings. A ``ManualClock`` lets the anneal run as fast as the renderer allows.
"""

import matplotlib.pyplot as plt
import numpy as np

from de_twin import DigitalTwin, ManualClock
from de_twin.state import Projection

clock = ManualClock()
twin = DigitalTwin("Droplet crystallization", camera="DESim", clock=clock, holder="sim-heating", seed=3)
twin.column.set("Magnification", 20000)
feature = twin.specimen.features()[0]
twin.column.move_stage(x=-feature.center_um[0], y=-feature.center_um[1])
twin.column.set_projection(Projection.DIFFRACTION)

twin.specimen.update(0.0, twin.holder_state(0.0))
twin.holder.set(650.0)

fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
for ax, t_end in zip(axes, [0, 40, 80, 160]):
    while clock.now() < t_end:
        clock.advance(1.0)
        twin.specimen.update(clock.now(), twin.holder_state())
    dp = np.log1p(twin.flux(twin.request()))
    ax.imshow(dp, cmap="gray", vmin=np.percentile(dp, 1), vmax=np.percentile(dp, 99.95))
    ax.set_title(f"t = {t_end} s, {twin.holder_state().temperature_c:.0f} C", fontsize=9)
    ax.axis("off")
fig.tight_layout()
plt.show()
