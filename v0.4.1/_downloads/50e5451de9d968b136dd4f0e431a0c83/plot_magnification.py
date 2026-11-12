"""
Zooming in on a seeded specimen
===============================

One seeded specimen, gold particles on holey carbon over a copper mesh grid, seen from a
low-magnification atlas down to single particles. The images are the noiseless electron
flux the twin renders before the detector model.
"""

import matplotlib.pyplot as plt

from de_twin import DigitalTwin, ManualClock

twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(), seed=3)
feature = twin.specimen.features()[0]  # a particle cluster, in world micrometres

fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
for ax, (mag, intensity) in zip(axes, [(120, 0.97), (2500, 0.75), (20000, 0.5), (60000, 0.5)]):
    twin.column.set("Magnification", mag)
    twin.column.set_intensity(intensity)  # spread the beam at low magnification
    if mag >= 20000:
        twin.column.move_stage(x=-feature.center_um[0], y=-feature.center_um[1])
    flux = twin.flux(twin.request())
    optics = twin.optics(twin.request())
    ax.imshow(flux, cmap="gray")
    fov_um = optics.specimen_pixel_nm * flux.shape[1] / 1000
    ax.set_title(f"{mag:,}x, field {fov_um:.3g} um", fontsize=9)
    ax.axis("off")
fig.tight_layout()
plt.show()
