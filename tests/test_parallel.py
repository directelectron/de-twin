"""The twin's numba kernels can be launched from several threads at once."""

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("numba")

#: Four threads launching parallel kernels at once, over and over: under numba's
#: workqueue threading layer (macOS's only one) this aborted the process.
_CONCURRENT = textwrap.dedent(
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from de_twin.parallel import PARALLEL
    from de_twin.render.fastpath import transpose
    from de_twin.render.stem_nb import window_dots

    x = np.arange(512 * 384, dtype=np.float32).reshape(512, 384)
    corners = np.arange(0, 256, 4, dtype=np.int64)
    window = np.ones((16, 16), np.float32)
    expected = np.array([x[r:r + 16, r:r + 16].sum(dtype=np.float64) for r in corners])

    def launch(_):
        for _ in range(50):
            out = np.empty((384, 512), np.float32)
            transpose(x, out)
            assert np.array_equal(out, x.T)
            dots = np.empty(len(corners))
            window_dots(x, corners, corners, window, dots)
            assert np.allclose(dots, expected)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(launch, range(4)))
    print(PARALLEL)
    """
)


def test_the_kernels_survive_concurrent_launches_on_the_workqueue_layer(tmp_path):
    env = {**os.environ, "NUMBA_THREADING_LAYER": "workqueue", "NUMBA_CACHE_DIR": str(tmp_path)}
    done = subprocess.run(
        [sys.executable, "-c", _CONCURRENT], env=env, capture_output=True, text=True, timeout=600, check=False
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == "False"  # compiled serial there
