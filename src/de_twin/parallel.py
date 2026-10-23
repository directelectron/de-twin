"""Whether numba's parallel kernels can be launched from more than one thread.

The twin renders on several threads at once (prefetch, strips, the camera's
publishing), and each may launch a parallel kernel. numba's TBB and OpenMP
threading layers take that; its fallback, the workqueue layer, aborts the whole
process when two threads launch at once. It is the only layer a pip install
has on macOS, which has neither TBB nor an OpenMP runtime, so there the kernels
are compiled serial: each still releases the GIL, and the twin's own threads
run them side by side.
"""

from __future__ import annotations


def _parallel_is_thread_safe() -> bool:
    try:
        import numba
        from numba.np.ufunc import parallel

        parallel._launch_threads()  # the layer is chosen when its threads start
        return numba.threading_layer() != "workqueue"
    except Exception:
        return True


#: Pass as ``parallel=`` in place of True.
PARALLEL = _parallel_is_thread_safe()


def numba_jit(*, fastmath: bool = True, **options):
    """``numba.njit`` with the twin's defaults; the identity decorator without numba.

    Kernels cache their compilation, release the GIL (so the publishing thread runs
    meanwhile) and take ``parallel=True`` only where :data:`PARALLEL` allows it.
    ``fastmath`` is on unless a kernel needs strict IEEE results.
    """
    try:
        import numba
    except Exception:
        return lambda function: function
    if options.get("parallel"):
        options["parallel"] = PARALLEL
    return numba.njit(cache=True, nogil=True, fastmath=fastmath, **options)
