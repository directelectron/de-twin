"""
Raw frames and references
=========================

The detector model adds what a DE camera adds: a dark offset with fixed pattern, read
noise, shot noise, gain non-uniformity and defects. Integrated images are dark- and
gain-corrected with references the twin measures, the way DE-Server delivers them.
"""

import matplotlib.pyplot as plt

from de_twin import DigitalTwin, ExposureMode, ManualClock

twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(), seed=3)
twin.column.set("Magnification", 20000)
feature = twin.specimen.features()[0]
twin.column.move_stage(x=-feature.center_um[0], y=-feature.center_um[1])

dark = twin.raw_frame(0.025, exposure_mode=ExposureMode.DARK)
raw = twin.raw_frame(0.025)
corrected = twin.snap(1.0)  # 40 frames at 40 fps, corrected, in electrons

fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
panels = [(dark, "dark frame (ADU)"), (raw, "one raw 25 ms frame (ADU)"), (corrected, "1 s corrected (electrons)")]
for ax, (img, title) in zip(axes, panels):
    ax.imshow(img, cmap="gray", vmin=img.mean() - 3 * img.std(), vmax=img.mean() + 3 * img.std())
    ax.set_title(title, fontsize=9)
    ax.axis("off")
fig.tight_layout()
plt.show()
