"""Reference HR-EBSD (test helper only, not part of de_twin): Wilkinson-style cross-correlation.

1. Flatten both patterns (divide out a wide Gaussian background).
2. Shifts of ``n_roi`` square ROIs of the test pattern against the reference: Hann-windowed,
   band-passed FFT cross-correlation, sub-pixel peak by a 3x3 quadratic (log) fit.
3. Least squares for the displacement gradient ``beta`` (detector frame, ``beta_33 = 0``: the
   hydrostatic part is unobservable): a reference direction ``r = (p - pc, DD)`` moves to
   ``(I + beta) r``, i.e. by ``q = (beta r)_xy - r_xy (beta r)_z / DD`` on the screen.
4. Remapping (Britton & Wilkinson 2012): the reference is warped into the test geometry
   with the pattern-centre move (a fixed crystal direction lands at
   ``p_t = pc_t + DD_t r_xy / r_z`` from either source point) and the current ``beta``, and
   the shifts that remain update ``beta``; a few iterations remove the bias of measuring
   large shifts with fixed windows.
5. To the sample frame: ``beta_s = S2D^T beta S2D``; deviatoric strain = sym(beta_s) minus its
   trace / 3, rotation vector from the antisymmetric part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def flatten(pattern, sigma: float = 20.0) -> np.ndarray:
    p = np.asarray(pattern, np.float64)
    bg = ndimage.gaussian_filter(p, sigma, mode="nearest")
    return p / np.maximum(bg, 1e-9 * max(bg.max(), 1e-12)) - 1.0


def _bandpass(n, lo=2.0, hi=None):
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    r = np.hypot(fx, fy) * n
    hi = hi if hi is not None else n / 4
    return (1 - np.exp(-((r / lo) ** 2))) * np.exp(-((r / hi) ** 2))


def xcf_shift(ref_roi, test_roi, bp, upsample: int = 50) -> tuple[float, float]:
    """(dx, dy) the test ROI is shifted by relative to the reference (pixels): Hann-windowed,
    band-passed FFT cross-correlation, refined by evaluating the correlation on a 1/upsample
    pixel grid around its peak with a matrix DFT (Guizar-Sicairos et al. 2008)."""
    n = ref_roi.shape[0]
    win = np.outer(np.hanning(n), np.hanning(n))
    a = ref_roi - ref_roi.mean()
    b = test_roi - test_roi.mean()
    X = np.fft.fft2(b * win) * np.conj(np.fft.fft2(a * win)) * bp
    c = np.real(np.fft.ifft2(X))
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    sy = iy - n if iy > n // 2 else iy
    sx = ix - n if ix > n // 2 else ix
    m = int(1.5 * upsample)
    off = np.arange(m) / upsample - 0.75
    k = np.fft.fftfreq(n) * n
    ey = np.exp(2j * np.pi * np.outer(sy + off, k) / n)  # (m, n)
    ex = np.exp(2j * np.pi * np.outer(k, sx + off) / n)  # (n, m)
    fine = np.real(ey @ X @ ex)
    jy, jx = np.unravel_index(np.argmax(fine), fine.shape)
    return float(sx + off[jx]), float(sy + off[jy])


@dataclass
class HrebsdResult:
    beta_sample: np.ndarray  # (3, 3) displacement gradient, sample frame (up to + lambda I)
    strain_dev: np.ndarray  # (3, 3) deviatoric strain, sample frame
    rotation: np.ndarray  # (3,) rotation vector, rad, sample frame
    residual_px: float  # RMS shift residual of the fit (sensor pixels)


def roi_centres(shape, n_side: int, size: int, margin: float = 0.12):
    h, w = shape
    ys = np.linspace(margin * h + size / 2, (1 - margin) * h - size / 2, n_side)
    xs = np.linspace(margin * w + size / 2, (1 - margin) * w - size / 2, n_side)
    return [(int(round(y)), int(round(x))) for y in ys for x in xs]


def _remap(R, beta, pcr, pct, binning, origin):
    """The reference pattern as the test geometry would record it: each test pixel's direction
    ``r_t = (p - pc_t, DD_t)`` taken back by ``(I + beta)^-1`` and found on the reference screen
    (pattern centres in sensor pixels: (x0, y0, DD)); cubic-spline interpolation."""
    h, w = R.shape
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    px = origin[0] + (jj + 0.5) * binning
    py = origin[1] + (ii + 0.5) * binning
    rt = np.stack([px - pct[0], py - pct[1], np.full(px.shape, pct[2])], -1)
    rr = rt @ np.linalg.inv(np.eye(3) + beta).T
    qx = pcr[0] + pcr[2] * rr[..., 0] / rr[..., 2]
    qy = pcr[1] + pcr[2] * rr[..., 1] / rr[..., 2]
    return ndimage.map_coordinates(
        R, [(qy - origin[1]) / binning - 0.5, (qx - origin[0]) / binning - 0.5], order=3, mode="nearest"
    )


def measure(
    ref,
    test,
    pc_ref,
    pc_test,
    s2d,
    *,
    sensor_shape,
    binning: int,
    roi_origin=(0, 0),
    n_side=5,
    size=96,
    sigma=20.0,
    iterations: int = 3,
) -> HrebsdResult:
    """HR-EBSD of ``test`` against ``ref`` (patterns on a ``binning``-binned grid; Bruker pattern
    centres of each; ``s2d`` the sample -> detector rotation). Remapping: the reference is
    warped into the test's geometry with the current estimate (pattern-centre move + beta) and
    the remaining ROI shifts update beta, ``iterations`` times."""
    H, W = sensor_shape
    R, T = flatten(ref, sigma), flatten(test, sigma)
    bp = _bandpass(size)
    pcr = np.array([pc_ref[0] * W, pc_ref[1] * H, pc_ref[2] * H])
    pct = np.array([pc_test[0] * W, pc_test[1] * H, pc_test[2] * H])
    beta = np.zeros((3, 3))
    centres = roi_centres(R.shape, n_side, size)
    res = 0.0
    for _ in range(max(1, iterations)):
        Rm = _remap(R, beta, pcr, pct, binning, roi_origin)
        rows, rhs = [], []
        for cy, cx in centres:
            sl = (slice(cy - size // 2, cy - size // 2 + size), slice(cx - size // 2, cx - size // 2 + size))
            dx, dy = xcf_shift(Rm[sl], T[sl], bp)
            px = roi_origin[0] + (cx + 0.5) * binning
            py = roi_origin[1] + (cy + 0.5) * binning
            r = np.array([px - pct[0], py - pct[1], pct[2]])
            # q = (d r)_xy - r_xy (d r)_z / DD for the remaining gradient d (d_33 = 0)
            for comp, q in ((0, dx * binning), (1, dy * binning)):
                row = np.zeros(9)
                for j in range(3):
                    row[comp * 3 + j] += r[j]
                    row[6 + j] -= r[comp] * r[j] / pct[2]
                rows.append(np.delete(row, 8))
                rhs.append(q)
        A, b = np.array(rows), np.array(rhs)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        d = np.insert(sol, 8, 0.0).reshape(3, 3)
        res = float(np.sqrt(np.mean((A @ sol - b) ** 2)))
        beta = (np.eye(3) + d) @ (np.eye(3) + beta) - np.eye(3)
    bs = s2d.T @ beta @ s2d
    sym = 0.5 * (bs + bs.T)
    dev = sym - np.trace(sym) / 3.0 * np.eye(3)
    anti = 0.5 * (bs - bs.T)
    rot = np.array([anti[2, 1], anti[0, 2], anti[1, 0]])
    return HrebsdResult(bs, dev, rot, res)


def truth_relative(F_test, F_ref):
    """(deviatoric strain, rotation vector) of F_test F_ref^-1 to first order (sample frame)."""
    beta = F_test @ np.linalg.inv(F_ref) - np.eye(3)
    sym = 0.5 * (beta + beta.T)
    anti = 0.5 * (beta - beta.T)
    return sym - np.trace(sym) / 3.0 * np.eye(3), np.array([anti[2, 1], anti[0, 2], anti[1, 0]])
