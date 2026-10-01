"""The recycled-array pool (de_twin.buffers)."""

import numpy as np

from de_twin import buffers


def test_a_released_array_is_recycled_and_a_viewed_one_is_not():
    a = buffers.empty((700, 900), np.float32)
    addr = a.ctypes.data
    a.flags.writeable = False  # e.g. a frame cache froze it
    del a
    b = buffers.empty((700, 900), np.float32)
    assert b.ctypes.data == addr and b.flags.writeable
    v = b[::2]
    del b
    c = buffers.empty((700, 900), np.float32)
    assert c.ctypes.data != addr  # the view keeps it alive: in use
    del v, c


def test_idle_arrays_are_capped():
    for k in range(80):
        buffers.empty((1000, 1000 + k), np.float32)  # 4 MB each, all released at once
    assert buffers.idle_bytes() <= buffers.MAX_BYTES + (8 << 20)
    assert len(buffers._POOL) <= 256
