"""DE-Server's GPU frame sharing, served by the twin (Windows, CUDA, same computer).

A client's ``get_gpu_frames()`` (deapi ``feat/gpu-frames``, which Autopilot uses) maps the
server's two ping-pong movie buffers straight out of GPU memory. :class:`SharedFrames`
is the server side of that hand-off, ported from DE-Server's
``GpuFrameSharing/SharedPingPong.cpp`` (protocol version 4), so a client written against
DE-Server runs unchanged against the twin:

* GET_GPU_FRAMES (client pid, attach) answers a handle to a small control block, valid
  in the client process. The buffer and semaphore handles are duplicated into the client
  and listed in the block, and listed again (with a new ``session``) whenever they change.
* Publish: frames go into buffer b, slot ``published % 2`` describes them, and ``filled``
  is released. The client releases ``free[b]`` when it is done; until then the twin waits
  before writing b again, as DE-Server's processing does.
* The client's ``detach`` event stops sharing and gives held buffers back.

Frames are copied from host memory into the shared buffers: the twin renders on the CPU.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from ctypes import c_int, c_size_t, c_uint32, c_void_p, wintypes
from ctypes import c_uint64 as u64

import numpy as np

log = logging.getLogger(__name__)

VERSION, BUFFERS, MAX_IMAGES = 4, 2, 1024  # GpuFrameSharing/Protocol.h
IMAGE = np.dtype([("frame", "<i8"), ("scanX", "<i4"), ("scanY", "<i4"), ("valid", "<i4"), ("pad", "<i4")])
PROFILE = ("produced", "released", "unshared", "holdNs", "holdMaxNs", "setupNs", "publishNs", "publishMaxNs")


class _Slot(ctypes.Structure):
    _fields_ = [
        (n, u64) for n in ("buffer", "count", "width", "height", "rowBytes", "imageBytes", "publishedNs")
    ] + [("image", ctypes.c_byte * (IMAGE.itemsize * MAX_IMAGES))]


class _ClientHandles(ctypes.Structure):
    _fields_ = [
        ("filled", u64),
        ("free", u64 * BUFFERS),
        ("buffer", u64 * BUFFERS),
        ("detach", u64),
        ("session", u64),
    ]


class _Control(ctypes.Structure):
    _fields_ = (
        [("version", c_uint32), ("device", c_uint32), ("bufferBytes", u64), ("client", _ClientHandles)]
        + [(n, u64) for n in PROFILE]
        + [("slot", _Slot * BUFFERS)]
    )


class _Location(ctypes.Structure):
    _fields_ = [("type", c_int), ("id", c_int)]


class _AllocationFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _AllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", c_int),
        ("requestedHandleTypes", c_int),
        ("location", _Location),
        ("win32HandleMetaData", c_void_p),
        ("allocFlags", _AllocationFlags),
    ]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", c_int)]


class _ObjectAttributes(ctypes.Structure):  # winternl.h; the driver reads all of it
    _fields_ = [
        ("Length", wintypes.ULONG),
        ("RootDirectory", wintypes.HANDLE),
        ("ObjectName", c_void_p),
        ("Attributes", wintypes.ULONG),
        ("SecurityDescriptor", c_void_p),
        ("SecurityQualityOfService", c_void_p),
    ]


PINNED, HANDLE_WIN32, LOCATION_DEVICE, READ_WRITE = 1, 2, 1, 3
VMM_SUPPORTED, WIN32_HANDLE_SUPPORTED = 102, 104  # CUdevice_attribute
PAGE_READWRITE, FILE_MAP_ALL_ACCESS, PROCESS_DUP_HANDLE, DUPLICATE_SAME_ACCESS = 4, 0xF001F, 0x40, 2
WAIT_OBJECT_0, WAIT_TIMEOUT = 0, 0x102

_k32 = _cu = None


def _load_dlls():
    global _k32, _cu
    if _k32 is not None:
        return
    k32, cu = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("nvcuda")
    H = wintypes.HANDLE
    k32.CreateFileMappingW.restype = H
    k32.CreateFileMappingW.argtypes = [H, c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, c_void_p]
    k32.MapViewOfFile.restype = c_void_p
    k32.MapViewOfFile.argtypes = [H, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, c_size_t]
    k32.UnmapViewOfFile.argtypes = [c_void_p]
    k32.CreateSemaphoreW.restype = H
    k32.CreateSemaphoreW.argtypes = [c_void_p, wintypes.LONG, wintypes.LONG, c_void_p]
    k32.CreateEventW.restype = H
    k32.CreateEventW.argtypes = [c_void_p, wintypes.BOOL, wintypes.BOOL, c_void_p]
    k32.SetEvent.argtypes = [H]
    k32.OpenProcess.restype = H
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.GetCurrentProcess.restype = H
    k32.DuplicateHandle.argtypes = [H, H, H, ctypes.POINTER(H), wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.ReleaseSemaphore.argtypes = [H, wintypes.LONG, c_void_p]
    k32.WaitForMultipleObjects.argtypes = [wintypes.DWORD, ctypes.POINTER(H), wintypes.BOOL, wintypes.DWORD]
    k32.WaitForMultipleObjects.restype = wintypes.DWORD
    k32.CloseHandle.argtypes = [H]
    for name in ("cuMemAddressReserve", "cuMemMap", "cuMemUnmap", "cuMemAddressFree", "cuMemcpyHtoD_v2"):
        getattr(cu, name).argtypes = {
            "cuMemAddressReserve": [ctypes.POINTER(u64), c_size_t, c_size_t, u64, u64],
            "cuMemMap": [u64, c_size_t, c_size_t, u64, u64],
            "cuMemUnmap": [u64, c_size_t],
            "cuMemAddressFree": [u64, c_size_t],
            "cuMemcpyHtoD_v2": [u64, c_void_p, c_size_t],
        }[name]
    _k32, _cu = k32, cu


def _check(result, what):
    if result:
        raise RuntimeError(f"{what}: CUDA driver error {result}")


class _Allocation:
    """GPU memory another process can map (CUDA VMM with a Win32 handle), as
    GpuFrameSharing/SharedAllocation.h."""

    def __init__(self, device: int, bytes_requested: int):
        attributes = _ObjectAttributes(Length=ctypes.sizeof(_ObjectAttributes))
        self._attributes = attributes  # the driver keeps a pointer to it until export
        prop = _AllocationProp(
            type=PINNED,
            requestedHandleTypes=HANDLE_WIN32,
            location=_Location(LOCATION_DEVICE, device),
            win32HandleMetaData=ctypes.cast(ctypes.pointer(attributes), c_void_p),
        )
        granularity = c_size_t()
        _check(
            _cu.cuMemGetAllocationGranularity(ctypes.byref(granularity), ctypes.byref(prop), 0), "granularity"
        )
        g = max(1, granularity.value)
        self.bytes = (bytes_requested + g - 1) // g * g
        self._handle, self.address, self.exported = u64(), u64(), wintypes.HANDLE()
        _check(
            _cu.cuMemCreate(ctypes.byref(self._handle), c_size_t(self.bytes), ctypes.byref(prop), u64(0)),
            "create shared GPU allocation",
        )
        _check(
            _cu.cuMemAddressReserve(ctypes.byref(self.address), self.bytes, 0, 0, 0), "reserve GPU address"
        )
        _check(_cu.cuMemMap(self.address, self.bytes, 0, self._handle, 0), "map GPU allocation")
        access = _AccessDesc(_Location(LOCATION_DEVICE, device), READ_WRITE)
        _check(
            _cu.cuMemSetAccess(self.address, c_size_t(self.bytes), ctypes.byref(access), c_size_t(1)),
            "set GPU mapping access",
        )
        _check(
            _cu.cuMemExportToShareableHandle(ctypes.byref(self.exported), self._handle, HANDLE_WIN32, u64(0)),
            "export GPU allocation",
        )

    def close(self):
        _cu.cuMemUnmap(self.address, self.bytes)
        _cu.cuMemAddressFree(self.address, self.bytes)
        _cu.cuMemRelease(self._handle)
        _k32.CloseHandle(self.exported)


def sharing_available(device: int = 0) -> bool:
    """Whether this computer's GPU can share memory with another process (as DE-Server
    checks before it allocates shareable movie buffers)."""
    try:
        _load_dlls()
        if _cu.cuInit(0):
            return False
        handle, vmm, win32 = c_int(), c_int(), c_int()
        if _cu.cuDeviceGet(ctypes.byref(handle), device):
            return False
        _cu.cuDeviceGetAttribute(ctypes.byref(vmm), VMM_SUPPORTED, handle)
        _cu.cuDeviceGetAttribute(ctypes.byref(win32), WIN32_HANDLE_SUPPORTED, handle)
        return bool(vmm.value and win32.value)
    except OSError:  # no Windows, or no CUDA driver
        return False


class SharedFrames:
    """The server side of GpuFrameSharing: two shareable GPU buffers and the hand-off to one
    client (see the module docstring)."""

    def __init__(self, device: int = 0):
        _load_dlls()
        _check(_cu.cuInit(0), "initialise CUDA")
        self.device = device
        handle = c_int()
        _check(_cu.cuDeviceGet(ctypes.byref(handle), device), "open the GPU")
        self._context = c_void_p()
        _check(_cu.cuDevicePrimaryCtxRetain(ctypes.byref(self._context), handle), "retain the GPU context")
        self._use_context()
        self._lock = threading.RLock()
        self._mapping = _k32.CreateFileMappingW(
            wintypes.HANDLE(-1), None, PAGE_READWRITE, 0, ctypes.sizeof(_Control), None
        )
        address = _k32.MapViewOfFile(self._mapping, FILE_MAP_ALL_ACCESS, 0, 0, ctypes.sizeof(_Control))
        if not self._mapping or not address:
            raise ctypes.WinError(ctypes.get_last_error())
        self.control = _Control.from_address(address)
        self.control.version, self.control.device = VERSION, device
        self._buffers: list[_Allocation] = []
        self._shape: tuple[int, int, int] | None = None  # (images, height, width) per buffer
        self._client = None
        self.attached = False
        self._filled = self._free = None
        self._held = [False] * BUFFERS
        self._published = self._session = 0
        self._detach = _k32.CreateEventW(None, False, False, None)
        self._quit = _k32.CreateEventW(None, True, False, None)
        self._new_session()
        self._watcher = threading.Thread(
            target=self._watch_detach, name="twin-gpu-frames-detach", daemon=True
        )
        self._watcher.start()

    # ------------------------------------------------------------------ setup
    def _use_context(self):
        _check(_cu.cuCtxSetCurrent(self._context), "make the GPU context current")

    def _new_session(self):
        """Fresh semaphores, so releases for dropped publications cannot free new ones."""
        for handle in (self._filled, *(self._free or ())):
            if handle:
                _k32.CloseHandle(handle)
        self._filled = _k32.CreateSemaphoreW(None, 0, BUFFERS, None)
        self._free = [_k32.CreateSemaphoreW(None, 0, 1, None) for _ in range(BUFFERS)]
        self._held = [False] * BUFFERS
        self._published = 0
        self._session += 1
        self._give_handles()

    def _give(self, handle) -> int:
        copy = wintypes.HANDLE()
        if not _k32.DuplicateHandle(
            _k32.GetCurrentProcess(),
            handle,
            self._client,
            ctypes.byref(copy),
            0,
            False,
            DUPLICATE_SAME_ACCESS,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(copy.value or 0)

    def _give_handles(self, *, control: bool = False) -> int:
        if not self.attached or not self._buffers:
            return 0
        listed = self.control.client
        listed.session = 0  # "being updated", for a client reading concurrently
        listed.filled = self._give(self._filled)
        for b in range(BUFFERS):
            listed.free[b] = self._give(self._free[b])
            listed.buffer[b] = self._give(self._buffers[b].exported)
        listed.detach = self._give(self._detach)
        handle = self._give(self._mapping) if control else 0
        listed.session = self._session  # written last: the client takes the handles now
        return handle

    def allocate(self, images: int, height: int, width: int) -> None:
        """(Re)allocate both buffers for ``images`` frames of height x width float32."""
        shape = (int(images), int(height), int(width))
        with self._lock:
            if shape == self._shape:
                return
            start = time.perf_counter_ns()
            self._use_context()
            for buffer in self._buffers:
                buffer.close()
            self._buffers = [
                _Allocation(self.device, shape[0] * shape[1] * shape[2] * 4) for _ in range(BUFFERS)
            ]
            self._shape = shape
            self.control.bufferBytes = self._buffers[0].bytes
            self._new_session()
            self.control.setupNs = time.perf_counter_ns() - start

    @property
    def capacity(self) -> int:
        """Frames one buffer holds."""
        return self._shape[0] if self._shape else 0

    # --------------------------------------------------------------- clients
    def attach(self, pid: int, shape: tuple[int, int, int]) -> int:
        """Hand the buffers to process ``pid``; returns its handle to the control block."""
        with self._lock:
            self.allocate(*shape)
            client = _k32.OpenProcess(PROCESS_DUP_HANDLE, False, int(pid))
            if not client:
                raise ctypes.WinError(ctypes.get_last_error())
            if self._client:
                _k32.CloseHandle(self._client)
            self._client, self.attached = client, True
            return self._give_handles(control=True)

    def detach(self) -> None:
        with self._lock:
            if not self.attached:
                return
            self.attached = False
            self._new_session()

    def _watch_detach(self):
        events = (wintypes.HANDLE * 2)(self._detach, self._quit)
        while _k32.WaitForMultipleObjects(2, events, False, 0xFFFFFFFF) == WAIT_OBJECT_0:
            self.detach()

    def new_acquisition(self) -> None:
        """Start of an acquisition: a buffer the client still holds is dropped with a new
        semaphore set, as DE-Server's Initialize hands both buffers back to Process()."""
        with self._lock:
            if any(self._held):
                self._new_session()

    # --------------------------------------------------------------- publish
    def _wait_free(self, buffer: int, stop: threading.Event | None) -> bool:
        """Wait until the client has given ``buffer`` back (DE-Server's processing waits the
        same way). False if sharing stopped or ``stop`` was set meanwhile."""
        while True:
            with self._lock:
                if not self.attached:
                    return False
                if not self._held[buffer]:
                    return True
                free = self._free[buffer]
                result = _k32.WaitForMultipleObjects(1, (wintypes.HANDLE * 1)(free), False, 0)
                if result == WAIT_OBJECT_0:
                    self._held[buffer] = False
                    self.control.released += 1
                    return True
            if stop is not None and stop.is_set():
                return False
            time.sleep(0.0005)

    def publish(self, images: np.ndarray, info: np.ndarray, *, stop: threading.Event | None = None) -> bool:
        """Share ``images`` (count, height, width) with their IMAGE ``info``. Without a client
        nothing is shared. Returns whether the client got them."""
        count = len(images)
        if not self.attached or not count:
            self.control.unshared += 1
            return False
        if count > MAX_IMAGES or count > self.capacity or images.shape[1:] != self._shape[1:]:
            raise ValueError(f"{images.shape} does not fit buffers of {self._shape}")
        buffer = self._published % BUFFERS
        if not self._wait_free(buffer, stop):
            self.control.unshared += 1
            return False
        with self._lock:
            if not self.attached:
                return False
            start = time.perf_counter_ns()
            self._use_context()
            host = np.ascontiguousarray(images, dtype=np.float32)
            _check(
                _cu.cuMemcpyHtoD_v2(self._buffers[buffer].address, host.ctypes.data, c_size_t(host.nbytes)),
                "copy frames to the GPU",
            )
            slot = self.control.slot[self._published % BUFFERS]
            slot.buffer, slot.count = buffer, count
            slot.height, slot.width = host.shape[1], host.shape[2]
            slot.rowBytes, slot.imageBytes = host.shape[2] * 4, host.shape[1] * host.shape[2] * 4
            ctypes.memmove(slot.image, np.ascontiguousarray(info, IMAGE).ctypes.data, count * IMAGE.itemsize)
            slot.publishedNs = start
            self._held[buffer] = True
            self._published += 1
            self.control.produced += 1
            _k32.ReleaseSemaphore(self._filled, 1, None)
            elapsed = time.perf_counter_ns() - start
            self.control.publishNs += elapsed
            self.control.publishMaxNs = max(self.control.publishMaxNs, elapsed)
            return True

    def close(self) -> None:
        _k32.SetEvent(self._quit)
        self._watcher.join(timeout=2)
        with self._lock:
            self.attached = False
            self._use_context()
            for buffer in self._buffers:
                buffer.close()
            self._buffers = []
            for handle in (self._filled, *self._free, self._detach, self._quit, self._client):
                if handle:
                    _k32.CloseHandle(handle)
            _k32.UnmapViewOfFile(ctypes.addressof(self.control))
            _k32.CloseHandle(self._mapping)
