"""Reference EBSD indexing (test helper only, not part of de_twin): dictionary indexing over a
fundamental-zone sampling, then a Nelder-Mead refinement of each orientation, both on
normalised cross-correlation of background-flattened patterns.

The simulated patterns come from the twin's own projection (``simulate(matrix) -> pattern``),
i.e. the indexing knows the geometry and the phase, as a calibrated EBSD system does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage, optimize
from scipy.spatial.transform import Rotation as _R


def flatten(pattern: np.ndarray, sigma: float | None = None, factor: int = 1) -> np.ndarray:
    """Divide out the smooth background (a wide Gaussian), optionally bin by ``factor``, and
    normalise to zero mean, unit norm (float32, flattened)."""
    p = np.asarray(pattern, np.float32)
    s = sigma if sigma is not None else max(p.shape) / 8.0
    bg = ndimage.gaussian_filter(p, s, mode="nearest")
    p = p / np.maximum(bg, 1e-6 * max(float(bg.max()), 1e-12)) - 1.0
    if factor > 1:
        h, w = (p.shape[0] // factor) * factor, (p.shape[1] // factor) * factor
        p = p[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return normalise(p)


def normalise(p: np.ndarray) -> np.ndarray:
    v = np.asarray(p, np.float32).ravel()
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


@dataclass
class IndexResult:
    matrices: np.ndarray  # (N, 3, 3) crystal -> sample
    ncc: np.ndarray  # (N,) final correlation
    dictionary_ncc: np.ndarray  # (N,) best dictionary correlation


def dictionary_index(
    experimental: np.ndarray,
    simulate: Callable[[np.ndarray], np.ndarray],
    matrices: np.ndarray,
    chunk: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Best dictionary entry per experimental pattern: ``experimental`` (N, P) normalised
    vectors, ``simulate(matrix)`` a normalised (P,) vector. Returns (index (N,), ncc (N,))."""
    best = np.full(len(experimental), -np.inf, np.float32)
    idx = np.zeros(len(experimental), np.int64)
    for c0 in range(0, len(matrices), chunk):
        D = np.stack([simulate(m) for m in matrices[c0 : c0 + chunk]])  # (C, P)
        S = experimental @ D.T  # (N, C)
        j = S.argmax(axis=1)
        v = S[np.arange(len(S)), j]
        better = v > best
        best[better] = v[better]
        idx[better] = j[better] + c0
    return idx, best


def refine(
    experimental: np.ndarray,
    simulate: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    step_deg: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Nelder-Mead on a rotation vector applied (in the sample frame) to ``start``."""

    def cost(r):
        return -float(experimental @ simulate(_R.from_rotvec(r).as_matrix() @ start))

    s = np.radians(step_deg)
    simplex = np.vstack([np.zeros(3), np.eye(3) * s])
    res = optimize.minimize(
        cost,
        np.zeros(3),
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": 1e-4, "fatol": 1e-6, "maxiter": 400},
    )
    return _R.from_rotvec(res.x).as_matrix() @ start, -float(res.fun)


def index(
    experimental: np.ndarray,
    simulate_coarse,
    simulate_fine,
    matrices: np.ndarray,
    experimental_fine: np.ndarray,
) -> IndexResult:
    """Dictionary-index (coarse patterns) then refine (fine patterns) every pattern."""
    j, dncc = dictionary_index(experimental, simulate_coarse, matrices)
    out, ncc = [], []
    for k in range(len(experimental)):
        m, c = refine(experimental_fine[k], simulate_fine, matrices[j[k]])
        out.append(m)
        ncc.append(c)
    return IndexResult(np.array(out), np.array(ncc), dncc)


def misorientation_deg(m1: np.ndarray, m2: np.ndarray, point_group) -> np.ndarray:
    """Symmetry-reduced misorientation angles (deg) between crystal->sample matrices (orix)."""
    from orix.quaternion import Orientation

    from de_twin.crystal import rotation_from_crystal_to_lab

    o1 = Orientation(rotation_from_crystal_to_lab(m1), symmetry=point_group)
    o2 = Orientation(rotation_from_crystal_to_lab(m2), symmetry=point_group)
    return np.asarray(o1.angle_with(o2, degrees=True))
