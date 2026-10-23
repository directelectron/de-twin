"""C++ <-> Python: the compiled ExternalFrameSource (what DE-Server runs) reads frames the
twin's shared-memory face publishes. Build the consumer first with ``cpp\\build.bat``."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from de_twin.clock import ManualClock
from de_twin.faces.shm_face import ShmFace
from de_twin.state import ExposureMode
from de_twin.twin import DigitalTwin

EXE = Path(__file__).resolve().parents[1] / "cpp" / "build" / "test_consumer.exe"

pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="Win32 shared memory"),
    pytest.mark.skipif(not EXE.exists(), reason="build cpp/test_consumer.exe with cpp\\build.bat"),
]


@pytest.mark.parametrize("mode", [ExposureMode.NORMAL, ExposureMode.DARK])
def test_cpp_reads_twin_frames(mode):
    name = f"DE_ExternalFramesInterop_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    twin = DigitalTwin("Dense Au on holey C", camera="DESim", clock=ManualClock())
    n = 6
    face = ShmFace(twin, name=name, pace=False).start()
    try:
        out = subprocess.run(
            [str(EXE), name, str(n), "1024", "1024", str(int(mode))],
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        face.stop()
    print(out.stdout)
    assert out.returncode == 0, out.stdout + out.stderr
    frames = [
        dict(t.split("=") for t in ln.split()[1:]) for ln in out.stdout.splitlines() if ln.startswith("FRAME")
    ]
    assert [int(f["i"]) for f in frames] == list(range(n))
    mean = [int(f["sum"]) / (1024 * 1024) for f in frames]
    if mode == ExposureMode.DARK:
        assert all(300 < m < 450 for m in mean)  # dark offset only
    else:
        assert all(m > 450 for m in mean)  # the beam adds signal
    assert face.frames_published == n
