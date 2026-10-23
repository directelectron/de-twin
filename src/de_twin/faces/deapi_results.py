"""What GET_RESULT, movie buffers, virtual images and GPU frames return."""

from __future__ import annotations

import logging
import time

import numpy as np

from .deapi_acquisition import _convert, _resample
from .deapi_compat import _SERVER_VERSION, _sends_scan_pattern_index, pb
from .deapi_protocol import (
    GPU_BUFFER_FRAMES,
    PIXEL_DTYPES,
    _recv_exact,
)

log = logging.getLogger(__name__)


class ResultsMixin:
    """What GET_RESULT, movie buffers, virtual images and GPU frames return."""

    # ------------------------------------------------------------ geometry
    def _image_shape(self) -> tuple[int, int]:
        return int(float(self["Image Size Y (pixels)"])), int(float(self["Image Size X (pixels)"]))

    def _masks_for(self, shape):
        key = (shape, tuple(id(m) for m in self.virtual_masks))
        if key not in self._mask_cache:
            out = []
            for k, m in enumerate(self.virtual_masks[:5]):
                mm = _resample(np.asarray(m), shape)
                try:
                    calc = str(self[f"Scan - Virtual Detector {k} Calculation"]).lower()
                except KeyError:
                    calc = "sum"
                pos = mm == 1
                neg = mm == 2
                out.append((pos, neg if neg.any() else None, calc))
            self._mask_cache = {key: out}
        return self._mask_cache[key]

    def _set_virtual_mask(self, command):
        mask_id = command.command[0].parameter[0].p_int
        w = command.command[0].parameter[1].p_int
        h = command.command[0].parameter[2].p_int
        data = _recv_exact(self.socket, w * h)
        mask = np.frombuffer(data, dtype=np.int8).reshape((h, w)).copy()
        if 0 <= mask_id < len(self.virtual_masks):
            self.virtual_masks[mask_id] = mask
        return (self._ack(command),)

    def _gpu_frames(self, command):
        """GET_GPU_FRAMES (client pid, attach): attach answers the client's handle to the
        control block, as DE-Server does. Without GPU memory sharing on this computer it
        answers "simulated", as deapi's fake server does."""
        from . import gpu_frames

        params = command.command[0].parameter
        pid = int(params[0].p_string)
        attach = bool(params[1].p_bool) if len(params) > 1 else True
        if not attach:
            if self._shared_frames is not None:
                self._shared_frames.detach()
            return (self._ack(command),)
        if self._shared_frames is None:
            if not gpu_frames.sharing_available():
                log.warning("this GPU cannot share memory with another process: GPU frames are simulated")
                return (self._ack(command, "simulated"),)
            self._shared_frames = gpu_frames.SharedFrames()
        h, w = self._frame_shape()
        try:
            handle = self._shared_frames.attach(pid, (GPU_BUFFER_FRAMES, h, w))
        except OSError as exc:
            log.warning("GPU frames: %s", exc)
            return (self._ack(command, error=True),)
        return (self._ack(command, str(handle)),)

    def _frame_shape(self) -> tuple[int, int]:
        """The shape of the frames the next acquisition will render."""
        detector = self.twin.detector
        request = self.request()
        if hasattr(detector, "output_shape"):
            return tuple(int(v) for v in detector.output_shape(request))
        return self._image_shape()

    # ------------------------------------------------------------ results
    def _result_image(self, frame_type: int, timeout_s: float):
        """(image array, frame count, eppix) for a GET_RESULT frame type."""
        acq = self._acq
        if 12 <= frame_type <= 16:  # the masks themselves
            return np.asarray(self.virtual_masks[frame_type - 12]), 1, 0.0
        if acq is None:
            shape = self._image_shape()
            return np.zeros(shape, np.uint16), 0, 0.0
        acq.wait_first(timeout_s)
        with acq.cond:
            meta = acq.last_meta
            eppix = float(getattr(meta, "electrons_per_pixel", 0.0) or 0.0)
            if 2 <= frame_type <= 8:
                image = acq.last if acq.last is not None else np.zeros(self._image_shape(), np.uint16)
                return image, acq.count, eppix
            if frame_type in (9, 10, 11):
                image = acq.sum if acq.sum is not None else np.zeros(self._image_shape())
                return image, acq.count, eppix * acq.count
            if 17 <= frame_type <= 21:
                if acq.virtual is None:
                    return np.zeros((1, 1)), acq.count, eppix
                return acq.virtual[frame_type - 17].copy(), acq.count, eppix
        if 22 <= frame_type <= 25:
            vi = getattr(self.twin, "virtual_image", None)
            if callable(vi) and acq.request.scan.enabled:
                try:
                    return np.asarray(vi(acq.request, 50.0, 200.0), np.float32), acq.count, eppix
                except Exception as exc:
                    log.info("twin.virtual_image failed (%s); using mask 0", exc)
            if acq.virtual is not None:
                return acq.virtual[0].copy(), acq.count, eppix
            return np.zeros((1, 1)), acq.count, eppix
        raise ValueError(f"Frame type {frame_type} not supported by the twin fake server")

    def _get_result(self, command):
        p = command.command[0].parameter
        frame_type = p[0].p_int
        pixel_format = p[1].p_int
        window_w, window_h = p[5].p_int, p[6].p_int
        timeout_ms = p[13].p_int if len(p) > 13 else 5000
        histo_min = p[14].p_float if len(p) > 14 else 0.0
        histo_max = p[15].p_float if len(p) > 15 else 0.0
        histo_bins = p[16].p_int if len(p) > 16 else 256
        image, count, eppix = self._result_image(frame_type, max(timeout_ms, 1000) / 1000.0)
        if pixel_format not in PIXEL_DTYPES:  # AUTO
            if image.dtype in (np.uint8, np.int8) and frame_type in (12, 13, 14, 15, 16):
                pixel_format = 1
            elif image.dtype == np.uint16 and frame_type <= 8:
                pixel_format = 5
            else:
                pixel_format = 13
        if window_w <= 0:
            window_w = image.shape[1]
        if window_h <= 0:
            window_h = image.shape[0]
        image = _resample(np.asarray(image), (window_h, window_w))
        out = _convert(image, pixel_format)
        data = out.tobytes()
        acq = self._acq
        fps = float(self._fps)
        mean = float(out.mean()) if out.size else 0.0
        hmin, hmax = (float(out.min()), float(out.max())) if out.size else (0.0, 0.0)
        mapping = [
            int(pixel_format),
            int(window_w),
            int(window_h),
            "de_twin",
            int(acq.request.acquisition_index if acq else 0),
            bool(acq is not None and acq.running),
            int(max(count - 1, 0)),
            int(count),
            hmin,
            hmax,
            mean,
            float(out.std()) if out.size else 0.0,
            float(eppix),
            float(eppix * out.size * fps),
            float(eppix * fps),
            0.0,
            float(eppix),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # incident
            0.0,
            0.0,
            0.0,  # red/orange warning, saturation
            f"{time.time():.6f}",
            hmin,
            hmax,
            1.0,
        ]
        if _sends_scan_pattern_index(_SERVER_VERSION):
            mapping.append(0)  # current_scan_pattern_idx
        if histo_min == 0 and histo_max == 0:
            histo_min, histo_max = hmin, hmax
        if histo_max <= histo_min:
            histo_max = histo_min + 1
        hist, _ = np.histogram(out.ravel(), bins=max(1, histo_bins), range=(histo_min, histo_max))
        mapping += [float(histo_min), float(histo_max), float(histo_max)]
        mapping += [int(v) for v in hist]
        ack = self._ack(command, *mapping)
        header = pb.DEPacket()
        header.type = pb.DEPacket.P_DATA_HEADER
        header.data_header.bytesize = len(data)
        return (ack, header, data)

    def _movie_buffer_info(self, command):
        h, w = self._image_shape()
        fpb = int(float(self["Grabbing - Frames Per Buffer"]))
        # client order: headerBytes, imageBufferBytes, frameIndexStartPos, imageStartPos, W, H, frames, type
        return (self._ack(command, 1024, h * w * fpb * 2, 0, 1024, w, h, fpb, 5),)

    def _movie_buffer(self, command):
        p = command.command[0].parameter
        timeout_s = (p[0].p_int if len(p) else 5000) / 1000.0
        fpb = max(1, int(float(self["Grabbing - Frames Per Buffer"])))
        acq = self._acq
        if acq is None:
            return (self._ack(command, 4, 0, 0),)
        with acq.cond:
            acq.cond.wait_for(lambda: len(acq.movie_frames) >= fpb or acq.done, timeout_s)
            n = min(fpb, len(acq.movie_frames))
            if n == 0:
                return (self._ack(command, 4 if acq.done else 3, 0, 0),)
            frames = [acq.movie_frames.popleft() for _ in range(n)]
            first = acq.movie_next
            acq.movie_next += n
        data = np.stack(frames).astype(np.uint16).tobytes()
        numbers = np.zeros(128, np.int64)
        numbers[: min(n, 128)] = np.arange(first, first + min(n, 128))
        return (self._ack(command, 5, len(data) + 1024, n), numbers.tobytes(), data)

    def _virtual_image_info(self, command):
        nx, ny = int(float(self["Scan - Size X"])), int(float(self["Scan - Size Y"]))
        return (self._ack(command, nx * ny * 2, nx, ny, 5),)

    def _virtual_image(self, command):
        vid = command.command[0].parameter[0].p_int
        acq = self._acq
        nx, ny = int(float(self["Scan - Size X"])), int(float(self["Scan - Size Y"]))
        image = np.zeros((ny, nx))
        if acq is not None and acq.virtual is not None and 0 <= vid < 5:
            with acq.cond:
                image = acq.virtual[vid].copy()
        peak = float(image.max()) if image.size else 0.0
        if peak > 65535:
            image = image * (65535.0 / peak)
        raw = _convert(image, 5).tobytes()
        return (self._ack(command, 5, len(raw), int(acq.count if acq else 0)), raw)
