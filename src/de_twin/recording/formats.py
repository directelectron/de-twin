"""The recordings the twin replays, and where each one's geometry and time are read from.

Three kinds, opened by :func:`open_recording`:

* **A DE-Server movie** (``…_movie.mrc``) with the ``…_info.txt`` DE-Server writes beside
  it: frames in raster order, scan after scan. The frame time is the info file's. The scan
  size is too, unless an external controller drove the scan (DE-Server then counts frames
  in a raster of its own that says nothing of the beam), in which case it has to be given.
* **An EDAX pattern file** (``.up1``, 8-bit; ``.up2``, 16-bit), as DE-Server writes EBSD
  maps: the scan size from a version-3 header, else the info file or the ``.ang`` beside
  it, else it has to be given.
* **An Autopilot run folder** (``dataset.json``): the movie files the record lists, each
  frame placed at its recorded point and pass by the record's index, so a sparse or region
  scan replays where it was taken and nothing elsewhere.

A recording is a field of ``nx`` by ``ny`` points recorded ``passes`` times. Where the
scan size has to be given, :class:`SideNeeded` says so; it is a square scan's side.
"""

from __future__ import annotations

import glob
import json
import os
import struct
from collections import OrderedDict
from collections.abc import Callable

import numpy as np

from ..state import RenderMode
from .files import FrameFiles, FramePart

MRC_HEADER_BYTES = 1024
#: The MRC modes DE-Server writes a movie in (mode 0 is its unsigned 8 bit).
MRC_MODES = {0: "u1", 1: "<i2", 2: "<f4", 6: "<u2"}
#: EDAX pattern files: their pixel types, by extension.
UP_DTYPES = {".up1": "u1", ".up2": "<u2"}
#: A version-3 UP header: version, width, height, offset, extra patterns, map columns,
#: map rows, grid type, x step, y step.
UP3_HEADER = struct.Struct("<4IB2IB2d")


class Unreadable(ValueError):
    """Not a recording this reads, or one whose files disagree with themselves."""


class SideNeeded(Unreadable):
    """The recording's scan size is not written anywhere it can be read: give the side of
    its square scan (``side=``). *frames* is how many frames it holds."""

    def __init__(self, message: str, *, frames: int, frame_shape: tuple[int, int]) -> None:
        super().__init__(message)
        self.frames = int(frames)
        self.frame_shape = tuple(frame_shape)


class Recording:
    """A field of *field* = (nx, ny) points recorded *passes* times, the frames in *files*.

    *locate* maps (pass, row, column) arrays to frame numbers in *files*, -1 where nothing
    was recorded. *frame_time_s* is None when nothing says it (a single scan needs none).
    """

    def __init__(
        self,
        path: str,
        kind: str,
        files: FrameFiles,
        field: tuple[int, int],
        passes: int,
        frame_time_s: float | None,
        render_mode: RenderMode,
        locate: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        self.path = os.fspath(path)
        self.name = os.path.basename(self.path.rstrip("\\/"))
        self.kind = kind
        self.files = files
        self.field = (int(field[0]), int(field[1]))
        self.passes = max(1, int(passes))
        self.frame_time_s = frame_time_s
        self.render_mode = render_mode
        self._locate = locate

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self.files.shape

    @property
    def dtype(self) -> np.dtype:
        return self.files.dtype

    @property
    def pass_frames(self) -> int:
        """Frames per recorded scan."""
        return max(1, self.files.total // self.passes)

    def locate(self, passes: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Frame numbers at (*rows*, *columns*) of recorded scans *passes*; -1 off the field
        or where nothing was recorded."""
        nx, ny = self.field
        rows, columns = np.asarray(rows, np.int64), np.asarray(columns, np.int64)
        inside = (rows >= 0) & (rows < ny) & (columns >= 0) & (columns < nx)
        out = np.full(len(rows), -1, np.int64)
        if inside.any():
            passes = np.broadcast_to(np.asarray(passes, np.int64), rows.shape)
            out[inside] = self._locate(passes[inside], rows[inside], columns[inside])
        return out

    def describe(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "kind": self.kind,
            "field": list(self.field),
            "scans": self.passes,
            "frames": self.files.total,
            "frame_shape": list(self.frame_shape),
            "dtype": self.dtype.name,
            "frame_time_s": self.frame_time_s,
            "scan_period_s": None if self.frame_time_s is None else self.frame_time_s * self.pass_frames,
        }

    def close(self) -> None:
        self.files.close()


def open_recording(path: str | os.PathLike, *, side: int | None = None) -> Recording:
    """The recording at *path*: an Autopilot run folder, a DE-Server MRC movie, or a
    ``.up1``/``.up2`` pattern file. *side*: the square scan's side, where the files do not
    say it (:class:`SideNeeded`); given, it is used whatever they say."""
    path = os.fspath(path)
    if os.path.isdir(path):
        return _run_folder(path)
    extension = os.path.splitext(path)[1].lower()
    if extension == ".mrc":
        return _mrc_movie(path, side)
    if extension in UP_DTYPES:
        return _up_file(path, side)
    if os.path.basename(path).lower() == "dataset.json":
        return _run_folder(os.path.dirname(path))
    raise Unreadable(f"{path}: replays an Autopilot run folder, a DE-Server .mrc movie, or a .up1/.up2 file")


# ── DE-Server's info file ────────────────────────────────────────────────────


def read_info(path: str) -> dict[str, str]:
    """A DE-Server ``_info.txt``: ``Name = Value`` lines."""
    info: dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            name, sep, value = line.partition("=")
            if sep:
                info[name.strip()] = value.strip()
    return info


def info_beside(path: str) -> dict[str, str] | None:
    """The info file DE-Server wrote with the movie *path*, or None. DE-Server names its
    files after the dataset (``<dataset>_movie.mrc``, ``<dataset>_0_movie.up2``,
    ``<dataset>_info.txt``), so the info file is the one whose dataset name starts the
    movie's."""
    folder, name = os.path.split(path)
    candidates = sorted(glob.glob(os.path.join(glob.escape(folder or "."), "*_info.txt")))
    best = None
    for candidate in candidates:
        dataset = os.path.basename(candidate)[: -len("_info.txt")]
        if name.startswith(dataset + "_") and (best is None or len(dataset) > len(best[0])):
            best = (dataset, candidate)
    return read_info(best[1]) if best is not None else None


def _number(info: dict[str, str] | None, name: str) -> float | None:
    try:
        return float(info[name]) if info is not None else None
    except (KeyError, ValueError):
        return None


def info_frame_time(info: dict[str, str] | None) -> float | None:
    """The frame time the info file says, in seconds."""
    ns = _number(info, "Frame Time (nanoseconds)")
    if ns:
        return ns * 1e-9
    fps = _number(info, "Frames Per Second")
    return 1.0 / fps if fps else None


def info_scan_size(info: dict[str, str] | None) -> tuple[int, int] | None:
    """The scan size the info file says, (nx, ny); None without a scan, or when an external
    controller drove it (DE-Server's own scan size is then only how it counted frames)."""
    if info is None:
        return None
    if str(info.get("Scan - Enable", "")).lower() not in ("enable", "on", "true", "1"):
        return None
    if "external" in str(info.get("Scan - Controller", "")).lower():
        return None
    nx, ny = _number(info, "Scan - Size X"), _number(info, "Scan - Size Y")
    return (int(nx), int(ny)) if nx and ny else None


# ── the raster formats ──────────────────────────────────────────────────────


def _raster(path, kind, files, field, frame_time_s, render_mode) -> Recording:
    """Frames in raster order, scan after scan; a partial last scan is left out."""
    nx, ny = field
    points = nx * ny
    if files.total < points:
        raise Unreadable(f"{path}: {files.total} frames, fewer than one {nx}×{ny} scan")
    passes = files.total // points

    def locate(passes_, rows, columns):
        return passes_ * points + rows * nx + columns

    return Recording(path, kind, files, field, passes, frame_time_s, render_mode, locate)


def _given_field(side: int | None, path: str, frames: int, shape) -> tuple[int, int] | None:
    if side is None:
        return None
    side = int(side)
    if side < 1 or frames % (side * side):
        raise Unreadable(f"{path}: {frames} frames are not whole scans of {side}×{side}")
    return side, side


def _mrc_movie(path: str, side: int | None) -> Recording:
    with open(path, "rb") as fh:
        head = fh.read(MRC_HEADER_BYTES)
    if len(head) < MRC_HEADER_BYTES:
        raise Unreadable(f"{path}: too short to be an MRC file")
    nx, ny, nz, mode = struct.unpack("<4i", head[:16])
    if mode not in MRC_MODES or min(nx, ny, nz) < 1:
        raise Unreadable(f"{path}: not a DE-Server MRC movie (mode {mode}, {nx}×{ny}×{nz})")
    (extended,) = struct.unpack("<i", head[92:96])
    offset = MRC_HEADER_BYTES + max(0, extended)
    dtype = np.dtype(MRC_MODES[mode])
    if os.path.getsize(path) < offset + nz * nx * ny * dtype.itemsize:
        raise Unreadable(f"{path}: shorter than its header's {nz} frames (still being written?)")
    files = FrameFiles([FramePart(path, offset, nz, 0)], (ny, nx), dtype)
    info = info_beside(path)
    if info is None:
        raise Unreadable(f"{path}: needs the _info.txt DE-Server wrote beside it (its frame time)")
    field = _given_field(side, path, nz, (ny, nx)) or info_scan_size(info)
    if field is None:
        raise SideNeeded(
            f"{os.path.basename(path)}: its scan size is not recorded (an external scan controller)",
            frames=nz,
            frame_shape=(ny, nx),
        )
    return _raster(path, "DE-Server movie", files, field, info_frame_time(info), RenderMode.STEM_4D)


def _ang_size(path: str) -> tuple[int, int] | None:
    """The scan size in the header of an ``.ang`` beside *path* with the same stem."""
    stem = os.path.splitext(path)[0]
    for candidate in (stem + ".ang", *sorted(glob.glob(glob.escape(stem) + "*.ang"))):
        if not os.path.isfile(candidate):
            continue
        header: dict[str, str] = {}
        with open(candidate, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                name, _, value = line[1:].partition(":")
                header[name.strip().upper()] = value.strip()
        try:
            return int(header["NCOLS_ODD"]), int(header["NROWS"])
        except (KeyError, ValueError):
            continue
    return None


def _up_file(path: str, side: int | None) -> Recording:
    extension = os.path.splitext(path)[1].lower()
    dtype = np.dtype(UP_DTYPES[extension])
    with open(path, "rb") as fh:
        head = fh.read(UP3_HEADER.size)
    if len(head) < 16:
        raise Unreadable(f"{path}: too short to be a {extension} file")
    version, width, height, offset = struct.unpack("<4I", head[:16])
    if min(width, height) < 1 or offset < 16:
        raise Unreadable(f"{path}: not a {extension} file (version {version}, {width}×{height})")
    frames = (os.path.getsize(path) - offset) // (width * height * dtype.itemsize)
    files = FrameFiles([FramePart(path, offset, frames, 0)], (height, width), dtype)
    info = info_beside(path)
    field = _given_field(side, path, frames, (height, width))
    if field is None and version >= 3 and len(head) == UP3_HEADER.size:
        columns, rows = UP3_HEADER.unpack(head)[5:7]
        if columns and rows and columns * rows <= frames:
            field = (int(columns), int(rows))
    field = field or info_scan_size(info) or _ang_size(path)
    if field is None:
        raise SideNeeded(
            f"{os.path.basename(path)}: its scan size is not recorded",
            frames=frames,
            frame_shape=(height, width),
        )
    return _raster(
        path, extension[1:].upper() + " patterns", files, field, info_frame_time(info), RenderMode.EBSD
    )


# ── an Autopilot run ────────────────────────────────────────────────────────

#: Recorded scans of a run whose point -> frame tables are kept.
RUN_TABLES = 8


def _run_folder(folder: str) -> Recording:
    """An Autopilot run: the camera's complete files, in movie order, and each frame's
    point and pass from the record's index."""
    record_path = os.path.join(folder, "dataset.json")
    try:
        with open(record_path, encoding="utf-8") as fh:
            record = json.load(fh)
    except FileNotFoundError:
        raise Unreadable(f"{folder}: no dataset.json; not an Autopilot run folder") from None
    if record.get("format") != "de_autopilot.dataset":
        raise Unreadable(f"{record_path}: not an Autopilot dataset record")
    scan = record["scan"]
    rows_, columns_ = (int(n) for n in scan["field"])
    positions = np.load(os.path.join(folder, scan["positions"]))  # (point, 2) of (row, column)
    entries = [f for f in record["detectors"]["camera"]["files"] if f.get("complete")]
    if not entries:
        raise Unreadable(f"{folder}: the run has no complete movie file to replay")
    parts, points, passes_ = [], [], []
    shape = dtype = None
    for entry in sorted(entries, key=lambda f: int(f["first"])):
        file_path = os.path.join(folder, entry["path"])
        part, file_shape, file_dtype = _movie_part(file_path, int(entry["first"]))
        if shape is None:
            shape, dtype = file_shape, file_dtype
        elif (file_shape, file_dtype) != (shape, dtype):
            raise Unreadable(f"{file_path}: frames of {file_shape} {file_dtype}, where the run's are {shape}")
        parts.append(part)
        index = np.load(os.path.join(folder, entry["index"]))
        points.append(np.asarray(index["point"], np.int64)[: part.frames])
        passes_.append(np.asarray(index["pass"], np.int64)[: part.frames])
    files = FrameFiles(parts, shape, dtype)
    # every frame's movie number, recorded point and pass
    movie = np.concatenate([np.arange(p.first, p.first + len(pt)) for p, pt in zip(parts, points)])
    point = np.concatenate(points)
    passes_all = np.concatenate(passes_)
    placed = point >= 0
    movie, point, passes_all = movie[placed], point[placed], passes_all[placed]
    passes_all -= passes_all.min() if len(passes_all) else 0
    passes = int(passes_all.max()) + 1 if len(passes_all) else 1
    flat = positions[point, 0] * columns_ + positions[point, 1]
    order = np.argsort(passes_all, kind="stable")
    movie, flat, passes_all = movie[order], flat[order], passes_all[order]
    bounds = np.searchsorted(passes_all, np.arange(passes + 1))
    tables: OrderedDict[int, np.ndarray] = OrderedDict()

    def table(k: int) -> np.ndarray:
        found = tables.get(k)
        if found is None:
            found = np.full(rows_ * columns_, -1, np.int64)
            lo, hi = bounds[k], bounds[k + 1]
            found[flat[lo:hi]] = movie[lo:hi]  # the latest frame at a point wins
            tables[k] = found
            while len(tables) > RUN_TABLES:
                tables.popitem(last=False)
        tables.move_to_end(k)
        return found

    def locate(passes_k, rows, columns):
        out = np.empty(len(rows), np.int64)
        for k in np.unique(passes_k):
            at = passes_k == k
            out[at] = table(int(k))[rows[at] * columns_ + columns[at]]
        return out

    info = info_beside(parts[0].path)
    workflow = ""
    try:
        with open(os.path.join(folder, "run.json"), encoding="utf-8") as fh:
            workflow = str(json.load(fh).get("workflow", ""))
    except (OSError, ValueError):
        pass
    mode = RenderMode.EBSD if workflow.lower() == "ebsd" else RenderMode.STEM_4D
    return Recording(
        folder, "Autopilot run", files, (columns_, rows_), passes, info_frame_time(info), mode, locate
    )


def _movie_part(path: str, first: int) -> tuple[FramePart, tuple[int, int], np.dtype]:
    """One movie file of a run: where its frames start, how many, their shape and type."""
    extension = os.path.splitext(path)[1].lower()
    if extension == ".mrc":
        with open(path, "rb") as fh:
            head = fh.read(MRC_HEADER_BYTES)
        nx, ny, nz, mode = struct.unpack("<4i", head[:16])
        if mode not in MRC_MODES:
            raise Unreadable(f"{path}: MRC mode {mode} is not a movie this reads")
        (extended,) = struct.unpack("<i", head[92:96])
        return (
            FramePart(path, MRC_HEADER_BYTES + max(0, extended), nz, first),
            (ny, nx),
            np.dtype(MRC_MODES[mode]),
        )
    if extension in UP_DTYPES:
        dtype = np.dtype(UP_DTYPES[extension])
        with open(path, "rb") as fh:
            _, width, height, offset = struct.unpack("<4I", fh.read(16))
        frames = (os.path.getsize(path) - offset) // (width * height * dtype.itemsize)
        return FramePart(path, offset, frames, first), (height, width), dtype
    raise Unreadable(f"{path}: a run's movie this does not read")
