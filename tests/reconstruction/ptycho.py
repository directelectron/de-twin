"""Compact single-slice ptychography (ePIE) for validating twin 4D-STEM data. numpy only.

Forward model (same conventions as :mod:`de_twin.render.coherent`)::

    psi_j(r) = P(r) O(r + r_j)                   P centred at N/2 of an N x N window
    I_j(k)   = |F[psi_j](k)|^2                   F: orthonormal FFT, k = 0 at the detector centre

Pixel size ``dx = 1 / (N dk)`` for detector pixels of ``dk`` 1/nm (N = pattern side).
The optic axis on the detector (``center``, may be fractional) is honoured exactly with a phase
ramp on the exit wave. Positions are probe centres in object pixels (row, col).

This is a reference implementation (validation, examples), not a production reconstructor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PtychoResult:
    object: np.ndarray  # complex
    probe: np.ndarray  # complex (N, N)
    errors: list = field(default_factory=list)  # relative Fourier error per iteration
    dx_nm: float = 0.0


def probe_from_aberrations(
    shape: tuple[int, int],
    recip_px_inv_nm: float,
    wavelength_nm: float,
    alpha_mrad: float,
    aberrations,
    *,
    edge_px: float = 1.0,
) -> np.ndarray:
    """Probe on the reconstruction grid (dx = 1/(N dk)), centred at (N/2, N/2), sum |P|^2 = 1."""
    ny, nx = shape
    ky = np.fft.fftfreq(ny, 1.0 / (ny * recip_px_inv_nm))[:, None]
    kx = np.fft.fftfreq(nx, 1.0 / (nx * recip_px_inv_nm))[None, :]
    ka = alpha_mrad * 1e-3 / wavelength_nm
    A = np.clip((ka - np.hypot(kx, ky)) / (edge_px * recip_px_inv_nm) + 0.5, 0, 1)
    chi = aberrations.chi(kx, ky, wavelength_nm) if aberrations else 0.0
    p = np.fft.fftshift(np.fft.ifft2(A * np.exp(-1j * chi), norm="ortho"))
    return p / math.sqrt(float((np.abs(p) ** 2).sum()))


def _prepare(datacube, center):
    """Measured amplitudes rolled so that k = 0 sits at index 0, plus the fractional ramp."""
    data = np.asarray(datacube, np.float64)
    n_p = data.shape[:-2]
    qy, qx = data.shape[-2:]
    amps = np.sqrt(np.maximum(data.reshape(-1, qy, qx), 0.0))
    if center is None:
        center = (qx // 2, qy // 2)
    cx, cy = center
    ix, iy = math.floor(cx), math.floor(cy)
    ex, ey = cx - ix, cy - iy
    amps = np.roll(amps, (-iy, -ix), axis=(1, 2))
    # model FFT index i must represent k = (i - e) dk: multiply the exit wave by exp(+2 pi i e j / N)
    ramp = (
        np.exp(2j * math.pi * ey * np.arange(qy) / qy)[:, None]
        * np.exp(2j * math.pi * ex * np.arange(qx) / qx)[None, :]
    )
    return amps, ramp, n_p


def epie(
    datacube: np.ndarray,
    positions_px: np.ndarray,
    probe: np.ndarray,
    *,
    object_shape: tuple[int, int] | None = None,
    iterations: int = 30,
    alpha: float = 1.0,
    beta: float = 0.0,
    beta_start: int = 5,
    center: tuple[float, float] | None = None,
    seed: int = 0,
    dx_nm: float = 0.0,
    mode_weights=None,
    rpie: float = 1.0,
) -> PtychoResult:
    """Extended ptychographic iterative engine, optionally mixed-state (several probe modes).

    datacube : (..., Qy, Qx) intensities, one pattern per position (scan order = positions order).
    positions_px : (P, 2) probe centres (row, col) in object pixels.
    probe : (Qy, Qx) initial probe, or (M, Qy, Qx) incoherent probe modes (partial coherence,
        e.g. :func:`probe_modes_on_grid`); scaled internally to the mean pattern intensity
        (``mode_weights`` give the modes' intensity shares, default their own norms).
    beta : probe update strength (0 = known probe); starts after ``beta_start`` iterations.
    center : optic axis on the detector (x, y) in pixels; default the array centre.
    rpie : rPIE regularisation (Maiden et al. 2017): the object step is
        ``conj(P) d / ((1 - rpie) |P|^2 + rpie max|P|^2)``; 1 = plain ePIE, ~0.1-0.3 converges
        several times faster on strong phase objects.
    """
    amps, ramp, _ = _prepare(datacube, center)
    amps = amps.astype(np.float32)
    ramp = ramp.astype(np.complex64)
    cramp = np.conj(ramp)
    npat, qy, qx = amps.shape
    pos = np.asarray(positions_px, np.float64).reshape(-1, 2)
    if len(pos) != npat:
        raise ValueError("one position per pattern")
    tl = np.rint(pos - np.array([qy // 2, qx // 2])).astype(int)  # window top-left
    off = np.maximum(0, -tl.min(axis=0))
    tl += off
    shape = object_shape or tuple(int(v) for v in tl.max(axis=0) + np.array([qy, qx]))
    O = np.ones(shape, np.complex64)
    P = np.asarray(probe, np.complex64).copy()
    if P.ndim == 2:
        P = P[None]
    norms = (np.abs(P) ** 2).sum(axis=(1, 2))
    w = (
        norms / norms.sum()
        if mode_weights is None
        else np.asarray(mode_weights, float) / np.sum(mode_weights)
    )
    total = float((amps.astype(np.float64) ** 2).sum(axis=(1, 2)).mean())
    P *= np.sqrt(total * w / norms).astype(np.float32)[:, None, None]
    rng = np.random.default_rng(seed)
    norm = float((amps.astype(np.float64) ** 2).sum())
    errors = []
    for it in range(iterations):
        err = 0.0
        pint = (np.abs(P) ** 2).sum(axis=0)
        pmax = float(pint.max())
        denom = ((1.0 - rpie) * pint + rpie * pmax).astype(np.float32)
        for j in rng.permutation(npat):
            r, c = tl[j]
            ow = O[r : r + qy, c : c + qx]
            psi = P * ow[None]
            F = np.fft.fft2(psi * ramp[None], norm="ortho")
            I = (F.real**2 + F.imag**2).sum(axis=0)
            magnification = np.sqrt(I)
            err += float(((magnification - amps[j]) ** 2).sum())
            F *= (amps[j] / np.maximum(magnification, 1e-12))[None]
            d = np.fft.ifft2(F, norm="ortho") * cramp[None] - psi
            O[r : r + qy, c : c + qx] = ow + alpha * (np.conj(P) * d).sum(axis=0) / denom
            if beta > 0 and it >= beta_start:
                P = P + (beta / max(float((np.abs(ow) ** 2).max()), 1e-12)) * np.conj(ow)[None] * d
                pint = (np.abs(P) ** 2).sum(axis=0)
                pmax = float(pint.max())
                denom = ((1.0 - rpie) * pint + rpie * pmax).astype(np.float32)
        errors.append(err / norm)
    result = PtychoResult(O, P[0] if P.shape[0] == 1 else P, errors, dx_nm)
    result.offset_px = off  # object pixel (0, 0) = position frame pixel (-off)
    return result


def probe_modes_on_grid(gt: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The ground-truth probe modes (``DigitalTwin.ground_truth_ptychography``) on a
    reconstruction grid of ``shape`` detector pixels (dx = 1/(N dk_det)): their spectra cropped
    to the detector's k grid. Returns ((M, Qy, Qx) complex, weights)."""
    s = gt["sampling"]
    modes = np.asarray(gt["probe_modes"])
    n = modes.shape[-1]
    qy, qx = shape
    ratio = s.det_recip_px[0] / s.dk  # simulation k pixels per detector pixel
    out = []
    for m in modes:
        F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(m), norm="ortho"))
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError("detector pixel is not an integer number of simulation pixels")
        r = round(ratio)
        c = n // 2
        # detector k grid: k = (i - Q/2) dk_det for i in 0..Q-1 -> simulation index c + (i - Q/2) r
        iy = c + (np.arange(qy) - qy // 2) * r
        ix = c + (np.arange(qx) - qx // 2) * r
        ok_y = (iy >= 0) & (iy < n)
        ok_x = (ix >= 0) & (ix < n)
        G = np.zeros((qy, qx), complex)
        G[np.ix_(ok_y, ok_x)] = F[np.ix_(iy[ok_y], ix[ok_x])]
        p = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G), norm="ortho"))
        out.append(p / math.sqrt(float((np.abs(p) ** 2).sum())))
    return np.array(out), np.asarray(gt["mode_weights"], float)


def align_phase(recon: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> tuple:
    """(recon x exp(-i c), c, (dy, dx)): remove the constant phase ``c`` (and report the integer
    global shift that best aligns recon onto truth)."""
    a = np.asarray(recon)
    b = np.asarray(truth)
    m = np.ones(a.shape, bool) if mask is None else mask
    A = np.fft.fft2((a - a[m].mean()) * m)
    B = np.fft.fft2((b - b[m].mean()) * m)
    cc = np.abs(np.fft.ifft2(A * np.conj(B)))
    dy, dx = np.unravel_index(int(np.argmax(cc)), cc.shape)
    ny, nx = a.shape
    dy, dx = (dy + ny // 2) % ny - ny // 2, (dx + nx // 2) % nx - nx // 2
    c = float(np.angle(np.sum((a * np.conj(b))[m])))
    return a * np.exp(-1j * c), c, (int(dy), int(dx))


def phase_correlation(recon: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Pearson correlation of the phases inside ``mask`` after removing the constant phase
    offset (phases taken relative to the truth's mean phase, so they do not wrap)."""
    a, _, _ = align_phase(recon, truth, mask)
    b = np.asarray(truth)
    m = np.ones(a.shape, bool) if mask is None else mask
    ref = np.exp(-1j * np.angle(np.mean(b[m])))
    return float(np.corrcoef(np.angle(a[m] * ref), np.angle(b[m] * ref))[0, 1])


def object_on_grid(gt: dict, dx_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """The ground-truth object (``DigitalTwin.ground_truth_ptychography``) low-passed to the
    Nyquist of pixel ``dx_nm`` and resampled onto that grid (same origin: pixel 0 = pixel 0),
    plus the scan positions in those pixels. Use it to score a reconstruction made at the
    detector's sampling ``dx = 1 / (N dk)``."""
    from scipy import fft as sfft
    from scipy.ndimage import map_coordinates

    obj = np.asarray(gt["object"])
    d0 = float(gt["dx_nm"])
    ny, nx = obj.shape
    F = sfft.fft2(obj, workers=-1)
    ky = np.fft.fftfreq(ny, d0)[:, None]
    kx = np.fft.fftfreq(nx, d0)[None, :]
    kc = 0.5 / dx_nm
    F *= (np.abs(kx) <= kc) & (np.abs(ky) <= kc)
    lp = sfft.ifft2(F, workers=-1)
    s = dx_nm / d0
    my, mx = int((ny - 1) / s) + 1, int((nx - 1) / s) + 1
    yy, xx = np.meshgrid(np.arange(my) * s, np.arange(mx) * s, indexing="ij")
    out = map_coordinates(lp.real, [yy, xx], order=3, mode="nearest") + 1j * map_coordinates(
        lp.imag, [yy, xx], order=3, mode="nearest"
    )
    return out, np.asarray(gt["positions_px"], np.float64) / s
