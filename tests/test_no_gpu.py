"""The twin never starts CUDA: the GPU belongs to the consumer of its frames (DE-Server, then
Autopilot), and a twin holding a CUDA context there competes with them for it.

Run in a fresh interpreter, because CUDA initialisation is per process and the test process
may have started it for something else."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import uuid

SCRIPT = textwrap.dedent(
    """
    import ctypes, json, os, sys

    from de_twin import DigitalTwin
    from de_twin.clock import ManualClock
    from de_twin.faces.shm_face import ShmFace
    from de_twin.state import ScanRequest
    from de_twin.transport.shm import FrameConsumer

    name = sys.argv[1]
    twin = DigitalTwin("EBSD duplex steel", camera="DESim", seed=1, clock=ManualClock(),
                       detector_overrides={"gain_mode": "low"})
    twin.configure_ebsd(beam_kv=20.0, probe_current_na=100.0)
    req = twin.request(hw_binning=(2, 2), frame_time_s=0.125e-3, total_frames=4,
                       scan=ScanRequest(enabled=True, size=(8, 8), step_um=0.5, dwell_s=0.125e-3))
    for _ in twin.frames(req):
        pass
    consumer = FrameConsumer(name, max_frame_bytes=512 * 512 * 2)
    face = ShmFace(twin, name=name, pace=False, warm_up=False).start()
    consumer.begin(frame_shape=(512, 512), binning=(2, 2), sensor_shape=(1024, 1024),
                   frame_time_s=0.125e-3, total_frames=16, scan_size=(8, 8))
    frames = [consumer.read(timeout=60)[0] for _ in range(16)]
    face.stop()
    consumer.close()

    out = {"frames": len(frames), "numba_cuda_initialised": False, "driver": "not loaded"}
    drv = sys.modules.get("numba.cuda.cudadrv.driver")
    if drv is not None:  # imported, but asked nothing of the driver
        out["numba_cuda_initialised"] = drv.driver.pid is not None
    if sys.platform == "win32":
        k32 = ctypes.WinDLL("kernel32")
        k32.GetModuleHandleW.restype = ctypes.c_void_p
        if k32.GetModuleHandleW("nvcuda.dll"):  # loaded by something: ask without initialising
            ctx = ctypes.c_void_p()
            out["driver"] = ctypes.WinDLL("nvcuda.dll").cuCtxGetCurrent(ctypes.byref(ctx))
    print(json.dumps(out))
    sys.stdout.flush()
    os._exit(0)
    """
)

CUDA_ERROR_NOT_INITIALIZED = 3


def test_a_twin_serving_ebsd_frames_never_initialises_cuda():
    env = {k: v for k, v in os.environ.items() if k not in ("NUMBA_DISABLE_CUDA", "DE_TWIN_ALLOW_CUDA")}
    name = f"DE_ExternalFramesNoGpu_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    run = subprocess.run([sys.executable, "-c", SCRIPT, name], capture_output=True, text=True,
                         env=env, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["frames"] == 16
    assert not out["numba_cuda_initialised"]
    # cuInit never ran: the driver, if anything loaded it, reports itself uninitialised
    assert out["driver"] in ("not loaded", CUDA_ERROR_NOT_INITIALIZED), out
