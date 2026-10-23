"""GPU frame sharing: a deapi client's get_gpu_frames() maps the twin's frames out of GPU
memory, as it does DE-Server's (de_twin.faces.gpu_frames).

Needs Windows, a GPU that can share memory between processes, CuPy, and a deapi with the
GPU frames client (``deapi.gpu_frames``). The client runs in its own process, as it does
against DE-Server.
"""

import importlib.util
import json
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("deapi")
pytestmark = pytest.mark.deapi

from de_twin.faces import gpu_frames
from de_twin.faces.deapi_server import TwinDeapiServer

needs_sharing = pytest.mark.skipif(
    sys.platform != "win32"
    or importlib.util.find_spec("cupy") is None
    or importlib.util.find_spec("deapi.gpu_frames") is None
    or not gpu_frames.sharing_available(),
    reason="needs Windows, CuPy, deapi.gpu_frames and a GPU that shares memory between processes",
)

CLIENT = textwrap.dedent("""
    import json, os, sys
    import numpy as np
    import deapi

    c = deapi.Client()
    c.usingMmf = False
    c.connect(port=int(sys.argv[1]))
    c["Scan - Enable"] = "On"
    c["Scan - Size X"] = c["Scan - Size Y"] = 4
    c["Scan - Repeats"] = 3
    frames = c.get_gpu_frames()
    c.start_acquisition(1)
    seen, columns = [], []
    for batch in frames.batches(idle_s=2.0):
        if batch is None:
            break
        images = batch.images.get()
        for image, info in zip(images, batch.info):
            seen.append([int(info["frame"]), int(info["scanX"]), int(info["scanY"]), int(info["valid"])])
            columns.append(int(np.argmax(image.mean(axis=0))))
        if len(seen) >= 48:
            break
    frames.close()
    print(json.dumps({"type": type(frames).__name__, "seen": seen, "columns": columns,
                      "shape": list(images.shape[1:])}))
    sys.stdout.flush()
    os._exit(0)
""")


@needs_sharing
def test_a_client_maps_the_twins_frames_out_of_gpu_memory():
    from test_faces_deapi import StubTwin

    twin = StubTwin()
    with TwinDeapiServer(twin, port=0, pace=True) as server:
        result = subprocess.run(
            [sys.executable, "-c", CLIENT, str(server.port)],
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
        )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["type"] == "GpuFrames"  # the zero-copy path, not deapi's simulated stand-in
    assert out["shape"] == list(twin.detector.output_shape(twin.requests[0]))
    expected = [[i, (i % 16) % 4, (i % 16) // 4, 1] for i in range(48)]
    assert out["seen"] == expected
    # the stub twin's beam is a bright column at the stage's x: the frames are the twin's
    assert set(out["columns"]) == {int(twin.column.state().stage.x_um) % twin.detector.model.sensor_shape[1]}
