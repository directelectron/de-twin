"""
Tilting real crystals
=====================

Every grain is a crystal with a full 3D orientation (orix), and diffraction comes from
diffsims structure factors. Tilting the stage moves grains through Bragg conditions: the
Laue circle of a silicon lamella sweeps across its pattern, and gold grains in a thin film
switch between dark and bright in bright-field images.
"""

import matplotlib.pyplot as plt
import numpy as np

from de_twin import DigitalTwin, ManualClock
from de_twin.state import Projection

lamella = DigitalTwin("PN junction lamella", camera="DESim", clock=ManualClock(), seed=3)
lamella.column.set("Magnification", 4000)
lamella.column.set_projection(Projection.DIFFRACTION)

film = DigitalTwin("Au thin film 20 nm", camera="DESim", clock=ManualClock(), seed=3)
film.column.set("Magnification", 30000)
film.column.set_intensity(0.75)

tilts = [-2.0, -1.0, 0.0, 1.0, 2.0]
fig, axes = plt.subplots(2, len(tilts), figsize=(12, 5.2))
for i, a in enumerate(tilts):
    lamella.column.move_stage(alpha=a)
    dp = np.sqrt(lamella.flux(lamella.request()))
    axes[0, i].imshow(dp, cmap="gray", vmax=np.percentile(dp, 99.97))
    axes[0, i].set_title(f"alpha {a:+.0f} deg", fontsize=9)
    film.column.move_stage(alpha=a)
    axes[1, i].imshow(film.flux(film.request()), cmap="gray")
for ax in axes.ravel():
    ax.axis("off")
fig.text(0.01, 0.74, "Si lamella\ndiffraction", va="center", fontsize=9)
fig.text(0.01, 0.26, "Au film\nbright field", va="center", fontsize=9)
fig.tight_layout(rect=(0.06, 0, 1, 1))
plt.show()
