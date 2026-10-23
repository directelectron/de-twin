"""A pool of CPU worker threads, one per logical processor, for frame-parallel numba kernels.

numba's own threads split one call's loop between them, so every call waits for the slowest
of them, and the next call cannot start until it does. Here each thread takes the next job (a
few camera frames) as soon as it finishes one, and the caller submits jobs ahead without
waiting, so no core idles at the end of a batch.

The threads run below normal priority: the twin feeds a consumer (DE-Server reading frames
out of shared memory), and a consumer that cannot get a core is a camera that drops frames.
They are not pinned to processors: pinned, at that priority, they stall for milliseconds at a
time on this kind of machine (two sockets, 48 logical processors; measured: 1,500-6,900
frames/s pinned against a steady ~7,400 not). ``DE_TWIN_FRAME_THREADS`` sets their number
(default: every logical processor).
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from typing import Callable, Optional

log = logging.getLogger(__name__)

_WINDOWS = sys.platform == "win32"


def default_threads() -> int:
    n = int(os.environ.get("DE_TWIN_FRAME_THREADS", "0") or 0)
    return max(1, n or os.cpu_count() or 1)


def _lower_priority() -> None:
    """Lower the calling thread's priority. Best effort: a refusal leaves it as it was."""
    try:
        if _WINDOWS:
            import ctypes

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentThread.restype = ctypes.c_void_p
            k32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            k32.SetThreadPriority(k32.GetCurrentThread(), -1)  # THREAD_PRIORITY_BELOW_NORMAL
    except Exception:  # noqa: BLE001 - priority is a courtesy only
        log.debug("could not lower a worker thread's priority", exc_info=True)


class Batch:
    """Jobs submitted together; :meth:`wait` returns their results in submission order."""

    def __init__(self, count: int):
        self._left = count
        self._results: list = [None] * count
        self._error: Optional[BaseException] = None
        self._done = threading.Event()
        self._lock = threading.Lock()
        if count == 0:
            self._done.set()

    def _finish(self, index: int, result, error: Optional[BaseException]) -> None:
        with self._lock:
            self._results[index] = result
            if error is not None and self._error is None:
                self._error = error
            self._left -= 1
            if self._left == 0:
                self._done.set()

    def done(self) -> bool:
        return self._done.is_set()

    def wait_done(self, timeout: Optional[float] = None) -> bool:
        """Wait for the jobs, up to *timeout*; whether they are done (errors not raised)."""
        return self._done.wait(timeout)

    def wait(self, timeout: Optional[float] = None) -> list:
        if not self._done.wait(timeout):
            raise TimeoutError("frame jobs did not finish")
        if self._error is not None:
            raise self._error
        return self._results


class Workers:
    """*threads* worker threads (default :func:`default_threads`). A job is
    ``fn(local, *args)``, where ``local`` is a dict private to the worker running it (for its
    scratch buffers)."""

    def __init__(self, threads: Optional[int] = None):
        self.threads = int(threads or default_threads())
        self._jobs: "queue.SimpleQueue" = queue.SimpleQueue()
        self._threads = [threading.Thread(target=self._run, name=f"de-twin-frames-{i}", daemon=True)
                         for i in range(self.threads)]
        for t in self._threads:
            t.start()

    def _run(self) -> None:
        _lower_priority()
        local: dict = {}
        while True:
            item = self._jobs.get()
            if item is None:
                return
            fn, args, batch, index = item
            try:
                result, error = fn(local, *args), None
            except BaseException as exc:  # noqa: BLE001 - handed to the waiter
                result, error = None, exc
            batch._finish(index, result, error)
            del item, fn, args, batch, result  # a finished job holds nothing (its frame buffer)

    def submit(self, fn: Callable, arg_list: list) -> Batch:
        """Run ``fn(local, *args)`` for each ``args`` of *arg_list*; returns their `Batch`."""
        batch = Batch(len(arg_list))
        for i, args in enumerate(arg_list):
            self._jobs.put((fn, args, batch, i))
        return batch

    def close(self) -> None:
        for _ in self._threads:
            self._jobs.put(None)


_POOL: Optional[Workers] = None
_POOL_LOCK = threading.Lock()


def pool() -> Workers:
    """The process-wide pool, started on first use."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = Workers()
        return _POOL
