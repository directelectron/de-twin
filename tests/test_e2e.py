"""One twin, several faces: a stage move over the TEM-Channel SOAP face shows up
in images acquired through the deapi face, just as moving a real column shows up in
DE-Server images."""

import time

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

deapi = pytest.importorskip("deapi")

from de_twin.column.soap import SoapTemChannelClient
from de_twin.faces.deapi_server import TwinDeapiServer
from de_twin.faces.temchannel_soap import TemChannelServer
from de_twin.twin import DigitalTwin


def xcorr_shift(a, b):
    """Integer (dx, dy) with a ~ b shifted by (dx, dy); smoothed plain cross-correlation.

    (Whitened phase correlation locks onto residual fixed-pattern noise and reports 0,
    exactly as it does on real camera data.)
    """
    a = gaussian_filter(a, 4)
    b = gaussian_filter(b, 4)
    c = np.fft.ifft2(np.fft.fft2(a - a.mean()) * np.conj(np.fft.fft2(b - b.mean()))).real
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    ny, nx = c.shape
    return (ix + nx // 2) % nx - nx // 2, (iy + ny // 2) % ny - ny // 2


def test_soap_stage_move_shifts_deapi_images():
    twin = DigitalTwin("Dense Au on holey C", camera="DESim")
    with (
        TemChannelServer(twin.column, host="127.0.0.1", port=0) as soap,
        TwinDeapiServer(twin, port=0, pace=False) as cam,
    ):
        scope = SoapTemChannelClient(port=soap.port)
        c = deapi.Client()
        c.usingMmf = False
        c.connect(port=cam.port)
        c["Frames Per Second"] = 40
        c["Frame Count"] = 8

        def image():
            c.start_acquisition(1)
            deadline = time.time() + 30
            while c.acquiring and time.time() < deadline:
                time.sleep(0.02)
            return c.get_result("sumtotal")[0].astype(np.float64)

        px_um = twin.optics(twin.request()).specimen_pixel_nm / 1000.0
        x0 = twin.column.state().stage.x_um  # the twin starts over the specimen, not at 0
        a = image()
        scope.set_stage_position(x=x0 + 60 * px_um)
        deadline = time.time() + 10
        while scope.operation_status() == 1 and time.time() < deadline:
            time.sleep(0.02)
        b = image()
        # DE-Server's metadata view of the column agrees with the SOAP face
        assert float(c["Instrument Stage Position X (micrometers)"]) == pytest.approx(
            x0 + 60 * px_um, abs=1e-3
        )
        c.disconnect()
    dx, dy = xcorr_shift(b, a)
    assert abs(abs(dx) - 60) <= 2 and abs(dy) <= 2
