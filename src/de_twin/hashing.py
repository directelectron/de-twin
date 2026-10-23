"""Deterministic, order-independent hashing.

Ported from DE-Server's VirtualSpecimen ``Geometry.cpp`` so that the twin keeps
the same seeding scheme: every random quantity is derived from
``hash_seed(seed, kind, index, salt)`` rather than from a shared random stream,
which makes results independent of evaluation order and of parallelism.

All functions accept Python ints or numpy ``uint64`` arrays; array versions
are vectorised and wrap modulo 2**64 exactly like the C++ code.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from .parallel import numba_jit

_M64 = (1 << 64) - 1
_U64 = np.uint64

#: SplitMix64's increment and its two finalizer multipliers.
GOLDEN_GAMMA = 0x9E3779B97F4A7C15
MIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
MIX_MULTIPLIER_2 = 0x94D049BB133111EB

# Kinds used as the second argument of hash_seed (values match the C++ enum).


class SeedKind(IntEnum):
    HOLDER = 1
    SUPPORT_FILM = 2
    HOLE_POPULATION = 3
    PARTICLE = 4
    STRUCTURE = 5
    GRAIN = 6
    FIB_POST = 7
    LAMELLA = 8
    TIME_EVOLUTION = 9
    INTENSITY_JITTER = 10
    # Twin-only kinds (not present in the C++ module)
    DETECTOR = 32
    DARK = 33
    GAIN = 34
    BAD_PIXELS = 35
    SHOT_NOISE = 36
    DRIFT = 37


def mix64_int(x: int) -> int:
    """SplitMix64's finalizer (no increment) of a 64-bit int."""
    z = ((x ^ (x >> 30)) * MIX_MULTIPLIER_1) & _M64
    z = ((z ^ (z >> 27)) * MIX_MULTIPLIER_2) & _M64
    return z ^ (z >> 31)


def _splitmix64_int(x: int) -> int:
    return mix64_int((x + GOLDEN_GAMMA) & _M64)


def _splitmix64_arr(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x.astype(_U64, copy=False) + _U64(GOLDEN_GAMMA)
        z = (z ^ (z >> _U64(30))) * _U64(MIX_MULTIPLIER_1)
        z = (z ^ (z >> _U64(27))) * _U64(MIX_MULTIPLIER_2)
        return z ^ (z >> _U64(31))


@numba_jit(fastmath=False, inline="always")
def mix64(x):
    """:func:`mix64_int` for numba kernels: the same finalizer on a uint64."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(MIX_MULTIPLIER_1)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(MIX_MULTIPLIER_2)
    return x ^ (x >> np.uint64(31))


@numba_jit(fastmath=False, inline="always")
def splitmix64_kernel(x):
    """:func:`splitmix64` for numba kernels (one uint64 in, one out). Kernels that call it
    cache their compilation, and numba does not notice when this function changes."""
    return mix64(x + np.uint64(GOLDEN_GAMMA))


def splitmix64(x):
    """One SplitMix64 avalanche step (int or uint64 array)."""
    if isinstance(x, np.ndarray):
        return _splitmix64_arr(x)
    return _splitmix64_int(int(x) & _M64)


def hash_seed(seed, kind=0, index=0, salt=0):
    """Chained SplitMix64 of (seed, kind, index, salt).

    Mirrors ``HashSeed`` in the C++ module: four chained avalanches.
    Any argument may be a uint64 array (broadcasting applies).
    """
    if any(isinstance(v, np.ndarray) for v in (seed, kind, index, salt)):
        s = np.asarray(seed).astype(_U64)
        k = np.asarray(kind).astype(np.int64).astype(_U64)
        i = np.asarray(index).astype(np.int64).astype(_U64)
        t = np.asarray(salt).astype(np.int64).astype(_U64)
        h = _splitmix64_arr(s)
        h = _splitmix64_arr(h ^ k)
        h = _splitmix64_arr(h ^ i)
        return _splitmix64_arr(h ^ t)
    h = _splitmix64_int(int(seed) & _M64)
    h = _splitmix64_int(h ^ (int(kind) & _M64))
    h = _splitmix64_int(h ^ (int(index) & _M64))
    return _splitmix64_int(h ^ (int(salt) & _M64))


def mix_cell(i, j):
    """Combine two signed lattice indices into one 64-bit key."""
    if isinstance(i, np.ndarray) or isinstance(j, np.ndarray):
        with np.errstate(over="ignore"):
            a = np.asarray(i).astype(np.int64).astype(_U64) * _U64(GOLDEN_GAMMA)
            b = np.asarray(j).astype(np.int64).astype(_U64) * _U64(0xC2B2AE3D27D4EB4F)
            return a ^ b
    a = ((int(i) & _M64) * GOLDEN_GAMMA) & _M64
    b = ((int(j) & _M64) * 0xC2B2AE3D27D4EB4F) & _M64
    return a ^ b


def uniform_from_hash(h):
    """Map a 64-bit hash to a float in [0, 1)."""
    if isinstance(h, np.ndarray):
        return (h >> _U64(11)).astype(np.float64) * (1.0 / (1 << 53))
    return (int(h) >> 11) * (1.0 / (1 << 53))


def normal_from_hash(h):
    """Standard normal from a hash via Box-Muller (uses two derived uniforms)."""
    u1 = uniform_from_hash(splitmix64(h ^ 0x1) if not isinstance(h, np.ndarray) else splitmix64(h ^ _U64(1)))
    u2 = uniform_from_hash(splitmix64(h ^ 0x2) if not isinstance(h, np.ndarray) else splitmix64(h ^ _U64(2)))
    u1 = np.maximum(u1, 1e-300)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit hash, used for cache signatures."""
    h = 14695981039346656037
    for b in data:
        h ^= b
        h = (h * 1099511628211) & _M64
    return h


def rng_for(seed: int, kind: int = 0, index: int = 0, salt: int = 0) -> np.random.Generator:
    """A numpy Generator deterministically derived from ``hash_seed``.

    Use for generation-time streams (holder geometry, particle populations).
    Hot render loops should use the hash functions directly.
    """
    return np.random.Generator(np.random.PCG64(hash_seed(seed, kind, index, salt)))
