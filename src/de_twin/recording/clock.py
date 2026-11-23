"""Which recorded scan a replayed frame comes from: the recording's own time.

A movie of repeated scans records an experiment as it happened (a film crystallizing as it
heats), one scan after another. Its time is **recorded time**, and a frame is served from
the scan the recording was on at that time. Recorded time moves the way the experiment
would have under the operator's hands:

* **Unstarted** until the first acquisition that is not a search: every frame comes from
  recorded time 0. A search (finding the area, picking regions) looks without starting the
  experiment.
* **Frame-driven** while an acquisition runs: each frame acquired moves it on by the
  recording's frame time, so acquiring continuously plays scan 1, 2, 3... back to back.
* **Wall-driven** between acquisitions (and during a search, once started): it moves with
  the wall clock, as the specimen goes on changing while nothing is acquired.
* **Reset** to 0, and unstarted, when a series starts (the program's first acquisition
  starts the experiment again from the beginning). A series started while an acquisition
  runs restarts it at 0 there and then.
* Past the end, the last recorded scan is served for ever; a recording of one scan is that
  scan every pass.

The twin only sees acquisitions, so whoever drives it says when a series starts and when
its acquisitions are searches (:meth:`RecordedClock.series_started`,
:meth:`RecordedClock.set_searching`).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np

#: What :meth:`RecordedClock.begin` hands back for :meth:`RecordedClock.end`.
SEARCH, ACQUIRE = "search", "acquire"


class RecordedClock:
    """Recorded time for a recording of *passes* scans of *pass_frames* frames each,
    *frame_time_s* a frame. *wall* is the wall clock in seconds (injectable for tests)."""

    def __init__(
        self,
        frame_time_s: float,
        pass_frames: int,
        passes: int,
        *,
        wall: Callable[[], float] = time.monotonic,
    ) -> None:
        self.frame_time_s = float(frame_time_s)
        self.pass_frames = max(1, int(pass_frames))
        self.passes = max(1, int(passes))
        self.period_s = self.frame_time_s * self.pass_frames
        self._wall = wall
        self._lock = threading.Lock()
        self._t = 0.0  # recorded seconds, as of `_since` when wall-driven
        self._since = 0.0
        self._started = False
        self._acquiring = 0  # non-search acquisitions running
        self._searching = False

    # ── what the driver says ──
    def series_started(self) -> None:
        """A series starts: back to recorded time 0, and unstarted unless an acquisition is
        running (then it plays on from 0)."""
        with self._lock:
            self._t = 0.0
            self._since = self._wall()
            self._started = self._acquiring > 0

    def set_searching(self, searching: bool) -> None:
        """Whether the acquisitions begun from now on are searches."""
        with self._lock:
            self._searching = bool(searching)

    @property
    def searching(self) -> bool:
        return self._searching

    # ── what the faces say ──
    def begin(self) -> str:
        """An acquisition begins; returns its kind, for :meth:`end`."""
        with self._lock:
            if self._searching:
                return SEARCH
            if self._started and self._acquiring == 0:
                self._t += self._wall() - self._since
            self._started = True
            self._acquiring += 1
            return ACQUIRE

    def end(self, kind: str) -> None:
        with self._lock:
            if kind != ACQUIRE:
                return
            self._acquiring = max(0, self._acquiring - 1)
            if self._acquiring == 0:
                self._since = self._wall()

    def take(self, count: int) -> np.ndarray:
        """The recorded scan of each of the next *count* frames (an int array), moving
        frame-driven time on by them."""
        with self._lock:
            if self._started and self._acquiring > 0:
                times = self._t + np.arange(count) * self.frame_time_s
                self._t += count * self.frame_time_s
            else:
                times = np.full(count, self._now())
        if self.passes == 1 or self.period_s <= 0:
            return np.zeros(count, np.int64)
        return np.minimum((times // self.period_s).astype(np.int64), self.passes - 1)

    # ── what it reads ──
    def _now(self) -> float:
        if not self._started:
            return 0.0
        if self._acquiring > 0:
            return self._t
        return self._t + self._wall() - self._since

    def now(self) -> float:
        """Recorded seconds now."""
        with self._lock:
            return self._now()

    def scan_now(self) -> int:
        """The recorded scan a frame taken now comes from."""
        if self.passes == 1 or self.period_s <= 0:
            return 0
        return min(int(self.now() // self.period_s), self.passes - 1)

    def state(self) -> dict:
        with self._lock:
            mode = "unstarted" if not self._started else ("frames" if self._acquiring > 0 else "wall")
            t = self._now()
        return {
            "recorded_s": t,
            "scan": 0 if self.period_s <= 0 else min(int(t // self.period_s), self.passes - 1),
            "scans": self.passes,
            "period_s": self.period_s,
            "mode": mode,
            "searching": self._searching,
        }
