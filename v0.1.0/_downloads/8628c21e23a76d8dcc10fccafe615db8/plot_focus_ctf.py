"""
Defocus, astigmatism and the CTF
================================

TEM imaging is wave-optical: amorphous carbon shows Thon rings whose spacing follows the
defocus and which turn elliptical with objective astigmatism, so autofocus and CTF tools
can be tested against known values.
"""

import matplotlib.pyplot as plt
import numpy as np

from de_twin import DigitalTwin, ManualClock

twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock(), seed=3)
twin.column.set("Magnification", 300000)


def power_spectrum(img):
    a = (img - img.mean()) * np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    return np.log1p(np.fft.fftshift(np.abs(np.fft.fft2(a)) ** 2))


settings = [(-0.5, (0.0, 0.0)), (-1.5, (0.0, 0.0)), (-1.5, (0.4, 0.0))]
fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
for col, (df, stig) in enumerate(settings):
    twin.column.set_defocus_um(df)
    twin.column.set_objective_stig(*stig)
    img = twin.flux(twin.request())
    ps = power_spectrum(img)
    axes[0, col].imshow(img, cmap="gray")
    axes[1, col].imshow(ps, cmap="magma", vmin=np.percentile(ps, 40), vmax=np.percentile(ps, 99.95))
    axes[0, col].set_title(f"defocus {df} um" + (", astigmatic" if stig[0] else ""), fontsize=9)
    for ax in axes[:, col]:
        ax.axis("off")
fig.tight_layout()
plt.show()
