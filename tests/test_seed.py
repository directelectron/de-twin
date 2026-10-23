"""Pinning one acquisition's noise: AcquisitionRequest.seed and DigitalTwin.reset_serial."""

import numpy as np


def _frames(tw, **kw):
    req = tw.request(total_frames=3, **kw)
    return [raw for raw, _ in tw.frames(req)]


def test_a_seeded_acquisition_repeats_whatever_came_before(make_twin):
    tw = make_twin(seed=2)
    first = _frames(tw, seed=7)
    _frames(tw)  # unrelated acquisitions in between
    again = _frames(tw, seed=7)
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], first[1]), "frames within it still differ"
    assert not np.array_equal(first[0], _frames(tw, seed=8)[0])


def test_unseeded_acquisitions_differ_until_the_serial_is_reset(make_twin):
    tw = make_twin(seed=2)
    a = _frames(tw)
    b = _frames(tw)
    assert not np.array_equal(a[0], b[0])
    tw.reset_serial()
    c = _frames(tw)
    np.testing.assert_array_equal(a[0], c[0])


def test_snap_takes_a_seed(make_twin):
    tw = make_twin(seed=2)
    a = tw.snap(0.1, seed=11)
    tw.snap(0.1)
    b = tw.snap(0.1, seed=11)
    np.testing.assert_array_equal(a, b)


def test_the_numba_and_numpy_hashes_agree():
    """Kernels hash with the numba functions, the specimen's NumPy code with the others."""
    import numpy as np

    from de_twin.hashing import mix64, mix64_int, splitmix64, splitmix64_kernel

    rng = np.random.default_rng(0)
    values = np.concatenate([rng.integers(0, 2**64, 64, dtype=np.uint64),
                             np.array([0, 1, 2**63, 2**64 - 1], np.uint64)])
    expected = splitmix64(values)
    for value, want in zip(values, expected):
        assert splitmix64_kernel(value) == want
        assert splitmix64(int(value)) == int(want)
        assert mix64(value) == mix64_int(int(value))
