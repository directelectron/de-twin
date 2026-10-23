"""Tilt-corrected bright field (tcBF, "parallax") from a 4D-STEM datacube. numpy only.

Each detector pixel ``k`` inside the bright-field disk gives a virtual image that is, for a
weak phase object, the object seen through the probe's aberrations::

    V_k(r_p) ~ const + CTF (x) phi(r_p + d_k),      d_k = grad chi(k) / 2 pi      (lab frame, nm)

(``chi`` in the twin's convention, :mod:`de_twin.optics.aberrations`; for defocus alone
``d_k = lambda C1 k``, so with underfocus, C1 < 0, the image of pixel ``k`` moves against
``k``). The images are aligned by cross-correlation, the shifts fitted to ``d_k`` (a 2x2
matrix = scan rotation x symmetric (C1, A1) part; optionally the quadratic coma B2 and
three-fold A2 terms), and shifted back and summed: the tcBF image. Every pixel image has the
same transfer ``2 sin(chi_{C1,A1}(Q))``, which ``ctf_correct`` phase-flips.

Scan frame vs lab frame: scan pixel (row i, col j) sits at ``R(theta) (j, i) * step`` in the lab
(detector) frame, theta = scan rotation. The fit returns theta (the pi ambiguity is resolved
with ``rotation_hint_rad``, default 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TcbfResult:
    image: np.ndarray  # tcBF image (scan sampling x upsample), float64
    bf_image: np.ndarray  # plain (unshifted) bright-field sum, scan sampling
    C1_nm: float  # fitted defocus (twin convention: negative = underfocus)
    A1_nm: complex  # fitted two-fold astigmatism (CEOS complex, nm)
    B2_nm: complex = 0j  # axial coma (when fit_order >= 2)
    A2_nm: complex = 0j  # three-fold astigmatism (when fit_order >= 2)
    C3_nm: float = 0.0  # spherical aberration (when fit_order >= 3)
    rotation_rad: float = 0.0  # fitted scan rotation
    shifts_px: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # measured (dx, dy) scan px
    shifts_fit_px: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    k_inv_nm: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (kx, ky) of each pixel image
    matrix: np.ndarray = field(default_factory=lambda: np.eye(2))  # scan px per 1/nm (linear fit)
    center_px: tuple = (0.0, 0.0)
    radius_px: float = 0.0
    residual_px: float = 0.0

    @property
    def aberrations(self):
        from de_twin.optics.aberrations import Aberrations
        return Aberrations({"C1": self.C1_nm, "A1": self.A1_nm, "B2": self.B2_nm, "A2": self.A2_nm,
                            "C3": self.C3_nm})


def bf_disk(mean_pattern: np.ndarray, threshold: float = 0.5) -> tuple[float, float, float]:
    """(cx, cy, radius) of the bright-field disk in detector pixels (centroid / area)."""
    p = np.asarray(mean_pattern, np.float64)
    m = p > threshold * p.max()
    y, x = np.nonzero(m)
    w = p[m]
    cx, cy = float((x * w).sum() / w.sum()), float((y * w).sum() / w.sum())
    return cx, cy, math.sqrt(m.sum() / math.pi)


def _xcorr_shift(a_hat: np.ndarray, ref_hat: np.ndarray, upsample: int = 16) -> tuple[float, float]:
    """(dx, dy) such that a(r) ~ ref(r + d) (phase correlation, parabolic subpixel)."""
    c = a_hat * np.conj(ref_hat)
    c /= np.abs(c) + 1e-9 * np.abs(c).max()
    cc = np.fft.ifft2(c).real
    ny, nx = cc.shape
    iy, ix = np.unravel_index(int(np.argmax(cc)), cc.shape)

    def sub(cm, c0, cp):
        d = cm - 2 * c0 + cp
        return 0.0 if abs(d) < 1e-12 else 0.5 * (cm - cp) / d
    fy = sub(cc[(iy - 1) % ny, ix], cc[iy, ix], cc[(iy + 1) % ny, ix])
    fx = sub(cc[iy, (ix - 1) % nx], cc[iy, ix], cc[iy, (ix + 1) % nx])
    py = ((iy + ny // 2) % ny - ny // 2) + fy
    px = ((ix + nx // 2) % nx - nx // 2) + fx
    return -px, -py  # peak at r = -d


def _fourier_shift(img_hat: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """FFT of img(r - d) given FFT of img (i.e. translate content by +d)."""
    ny, nx = img_hat.shape
    qy = np.fft.fftfreq(ny)[:, None]
    qx = np.fft.fftfreq(nx)[None, :]
    return img_hat * np.exp(-2j * math.pi * (qx * dx + qy * dy))


def _basis(kx, ky, lam, order):
    """Gradient basis (d_x, d_y in nm per unit coefficient) of the fitted aberrations, lab frame:
    grad chi / 2pi for C1, A1 (re, im) and, with order >= 2, B2 (re, im), A2 (re, im)."""
    from de_twin.optics.aberrations import Aberrations

    names = [("C1", 1), ("A1", 1), ("A1", 1j)]
    if order >= 2:
        names += [("B2", 1), ("B2", 1j), ("A2", 1), ("A2", 1j)]
    if order >= 3:
        names += [("C3", 1)]
    cols = []
    for n, v in names:
        gx, gy = Aberrations({n: v}).gradient(kx, ky, lam)
        cols.append(np.concatenate([gx, gy]) / (2 * math.pi))
    return names, np.stack(cols, 1)


def tcbf(datacube: np.ndarray, *, recip_px_inv_nm: float, scan_step_nm: float, wavelength_nm: float,
         center: tuple[float, float] | None = None, radius: float | None = None,
         edge: float = 0.85, max_pixels: int = 150, iterations: int = 3, fit_order: int = 1,
         rotation_hint_rad: float = 0.0, upsample: int = 1, ctf_correct: bool = False,
         reference_radius: float = 0.25) -> TcbfResult:
    """Parallax / tilt-corrected bright field.

    Parameters
    ----------
    datacube : (Sy, Sx, Qy, Qx) intensities (any units; noise allowed).
    recip_px_inv_nm : 1/nm per detector pixel of the datacube (after binning).
    scan_step_nm, wavelength_nm : calibrations.
    center, radius : BF disk in detector pixels (default: measured from the mean pattern).
    edge : use pixels within ``edge * radius`` of the centre (avoid the disk rim).
    max_pixels : detector pixels are grouped (b x b) until at most this many images remain.
    fit_order : 1 fits C1 + A1 (+ rotation); 2 adds B2 and A2; 3 adds C3 (needed for an
        uncorrected probe at >~10 mrad, whose C3 shifts otherwise bias C1).
    upsample : >1 splats the shifted images onto a finer grid (de-aliasing, like py4DSTEM).
    ctf_correct : phase-flip the tcBF image by ``sign(sin chi_{C1,A1}(Q))``.
    """
    data = np.asarray(datacube)
    sy, sx, qy, qx = data.shape
    mean = data.reshape(-1, qy, qx).mean(axis=0)
    if center is None or radius is None:
        c0x, c0y, r0 = bf_disk(mean)
        center = center or (c0x, c0y)
        radius = radius or r0
    cx, cy = center
    yy, xx = np.mgrid[0:qy, 0:qx]
    inside = np.hypot(xx - cx, yy - cy) < edge * radius
    # group detector pixels b x b until few enough images remain
    b = 1
    while inside.sum() / (b * b) > max_pixels:
        b += 1
    gy, gx = yy // b, xx // b
    gid = np.where(inside, gy * ((qx + b - 1) // b) + gx, -1)
    groups = np.unique(gid[gid >= 0])
    flat = data.reshape(sy * sx, qy * qx)
    ims, ks = [], []
    for g in groups:
        sel = np.flatnonzero(gid.ravel() == g)
        if len(sel) < max(1, (b * b) // 2):
            continue
        v = flat[:, sel].sum(axis=1).reshape(sy, sx).astype(np.float64)
        mu = v.mean()
        if mu <= 0:
            continue
        ims.append(v / mu - 1.0)
        ks.append(((xx.ravel()[sel].mean() - cx) * recip_px_inv_nm, (yy.ravel()[sel].mean() - cy) * recip_px_inv_nm))
    ims = np.array(ims)
    ks = np.array(ks)
    hats = np.fft.fft2(ims)
    bf = data[:, :, inside].sum(axis=-1).astype(np.float64)

    # initial reference: the pixels near the centre (small parallax)
    kr = np.hypot(ks[:, 0], ks[:, 1])
    k_edge = edge * radius * recip_px_inv_nm
    near = kr <= max(reference_radius * k_edge, np.sort(kr)[min(len(kr) - 1, 3)])
    ref_hat = hats[near].sum(axis=0)
    lam = wavelength_nm
    A = np.c_[ks, np.ones(len(ks))]

    def model(shifts, bad):
        """Linear fit -> scan rotation (polar decomposition); lab-frame aberration fit; model
        shifts back in scan pixels."""
        coef, *_ = np.linalg.lstsq(A[~bad], shifts[~bad], rcond=None)
        M = coef[:2].T  # d_scan = M k + b
        # scan rotation + symmetric aberration part: step*M = R(-theta) lambda S
        W, _, Vt = np.linalg.svd(scan_step_nm * M)
        U = W @ Vt
        if np.linalg.det(U) < 0:  # indefinite S (|A1| > |C1|): keep a proper rotation
            U = W @ np.diag([1.0, -1.0]) @ Vt
        th = -math.atan2(U[1, 0], U[0, 0])
        if abs(math.remainder(th - rotation_hint_rad, 2 * math.pi)) > math.pi / 2:
            th = th + math.pi
        th = math.remainder(th, 2 * math.pi)
        c, s_ = math.cos(th), math.sin(th)
        R = np.array([[c, -s_], [s_, c]])
        # lab-frame shifts (nm): d_lab = R(theta) step d_scan
        d_lab = (R @ (scan_step_nm * (shifts - coef[2]).T)).T
        names, B = _basis(ks[:, 0], ks[:, 1], lam, fit_order)
        keep = np.r_[~bad, ~bad]
        sol, *_ = np.linalg.lstsq(B[keep], np.concatenate([d_lab[:, 0], d_lab[:, 1]])[keep], rcond=None)
        dl = (B @ sol).reshape(2, -1).T
        fit = (R.T @ dl.T).T / scan_step_nm + coef[2]
        return fit, M, th, names, sol

    bad = np.zeros(len(ks), bool)
    fit = np.zeros((len(ks), 2))
    for it in range(max(1, iterations)):
        shifts = np.array([_xcorr_shift(h, ref_hat) for h in hats])
        if it > 0:  # robust: drop gross outliers against the previous model
            dev = np.hypot(*(shifts - fit).T)
            bad = dev > max(3.0, 5 * np.median(dev))
        fit, M, th, names, sol = model(shifts, bad)
        # new reference: every image shifted back by its fitted parallax
        ref_hat = sum(_fourier_shift(h, dx, dy) for h, (dx, dy) in zip(hats, fit))
    residual = float(np.sqrt(np.mean(np.sum((shifts - fit)[~bad] ** 2, axis=1))))
    c, s = math.cos(th), math.sin(th)
    vals: dict = {}
    for (n, v), x in zip(names, sol):
        vals[n] = vals.get(n, 0j) + v * x
    C1 = float(vals.get("C1", 0).real)
    A1 = complex(vals.get("A1", 0))

    # tcBF image
    if upsample <= 1:
        img_hat = sum(_fourier_shift(h, dx, dy) for h, (dx, dy) in zip(hats, fit))
        img = np.fft.ifft2(img_hat).real / len(ims)
    else:
        u = int(upsample)
        acc = np.zeros((sy * u, sx * u))
        wts = np.zeros_like(acc)
        iy, ix = np.mgrid[0:sy, 0:sx]
        for v, (dx, dy) in zip(ims, fit):
            py_ = ((iy - dy) * u) % (sy * u)
            px_ = ((ix - dx) * u) % (sx * u)
            y0, x0 = np.floor(py_).astype(int), np.floor(px_).astype(int)
            fy_, fx_ = py_ - y0, px_ - x0
            for oy, wy in ((0, 1 - fy_), (1, fy_)):
                for ox, wx in ((0, 1 - fx_), (1, fx_)):
                    yy2, xx2 = (y0 + oy) % (sy * u), (x0 + ox) % (sx * u)
                    np.add.at(acc, (yy2, xx2), v * wy * wx)
                    np.add.at(wts, (yy2, xx2), wy * wx)
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(acc, 0.5 * u, mode="wrap") / np.maximum(gaussian_filter(wts, 0.5 * u, mode="wrap"),
                                                                       1e-6)
    if ctf_correct:
        from de_twin.optics.aberrations import Aberrations
        ny2, nx2 = img.shape
        step = scan_step_nm / max(1, int(upsample))
        qyy = np.fft.fftfreq(ny2, step)[:, None]
        qxx = np.fft.fftfreq(nx2, step)[None, :]
        # scan-frame frequency -> lab frame
        qlx, qly = c * qxx - s * qyy, s * qxx + c * qyy
        chi = Aberrations({"C1": C1, "A1": A1}).chi(qlx, qly, lam)
        img = np.fft.ifft2(np.fft.fft2(img) * np.sign(np.sin(chi))).real
    return TcbfResult(img, bf, C1, A1, complex(vals.get("B2", 0)), complex(vals.get("A2", 0)),
                      float(complex(vals.get("C3", 0)).real), th,
                      shifts, fit, ks, M, (float(cx), float(cy)), float(radius), residual)
