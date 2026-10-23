"""Recycled large arrays.

A new large NumPy array costs the page faults of its first touch, and on Windows these are
slow when the first touch is a multithreaded kernel (a 14 MB complex64 spectrum written by a
threaded FFT: ~17 ms of faults for ~4 ms of work). The render allocates the same few shapes
for every new view (field-map layers, exit-wave spectra, raster images), so :func:`empty`
hands back an array of the same shape and dtype that nothing references any more (its pages
still mapped) instead of a new one. An array is free when only this pool holds it; views keep
their base alive, so a live view is never recycled. The pool keeps at most ``MAX_BYTES`` of
unreferenced arrays.
"""

from __future__ import annotations

import sys
import threading

import numpy as np

MAX_BYTES = 256 << 20
MIN_BYTES = 1 << 20  # smaller arrays: plain np.empty

_POOL: list = []
_LOCK = threading.Lock()


def _refs(i: int) -> int:
    return sys.getrefcount(_POOL[i])


# references of a pool entry nothing else holds, as `_refs` counts them (the list, the
# argument; calibrated: the count differs between Python versions)
_POOL.append(np.empty(1))
_FREE_REFS = _refs(0)
_POOL.clear()


def _free(i: int) -> bool:
    """``_POOL[i]`` is referenced by the pool only (views of it count: they keep it alive)."""
    return _refs(i) <= _FREE_REFS


def idle_bytes() -> int:
    """Bytes of pooled arrays nothing uses (kept for reuse, at most ``MAX_BYTES``)."""
    with _LOCK:
        return sum(_POOL[i].nbytes for i in range(len(_POOL)) if _free(i))


def empty(shape, dtype=np.float64) -> np.ndarray:
    """``np.empty(shape, dtype)``, recycled when a free array of that shape and dtype exists
    (uninitialised either way)."""
    shape = tuple(int(v) for v in (shape if np.ndim(shape) else (shape,)))
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    if nbytes < MIN_BYTES:
        return np.empty(shape, dtype)
    with _LOCK:
        for i in range(len(_POOL)):
            if _POOL[i].shape == shape and _POOL[i].dtype == dtype and _free(i):
                a = _POOL.pop(i)
                _POOL.append(a)  # most recently used last
                a.flags.writeable = True  # a cache may have frozen it while it held it
                return a
        a = np.empty(shape, dtype)
        a.reshape(-1).view(np.uint8)[::4096] = 0  # fault the pages in here, single-threaded
        _POOL.append(a)
        # forget the oldest free arrays beyond the budget (in use ones are not the pool's)
        total = sum(_POOL[k].nbytes for k in range(len(_POOL)) if _free(k))
        k = 0
        while total > MAX_BYTES and k < len(_POOL) - 1:
            if _free(k):
                total -= _POOL[k].nbytes
                del _POOL[k]
            else:
                k += 1
        # entries in use are dropped from the list too once it is long: an array a cache
        # still holds is the cache's, and the pool must not keep it alive after that
        if len(_POOL) > 256:
            del _POOL[:len(_POOL) - 256]
        return a


def zeros(shape, dtype=np.float64) -> np.ndarray:
    a = empty(shape, dtype)
    a.fill(0)
    return a


def full(shape, value, dtype) -> np.ndarray:
    a = empty(shape, dtype)
    a.fill(value)
    return a
