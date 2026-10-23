"""User master patterns: read EMsoft / kikuchipy HDF5 or a NumPy array, write the EMsoft layout.

A user master replaces the kinematic one for its phase at its beam energy
(:meth:`de_twin.twin.DigitalTwin.set_master_pattern`). On load it is

1. reduced to one pattern per hemisphere: an energy bin (``energy_kv``: the nearest, linearly
   interpolated between the two nearest), else the Monte-Carlo energy-weighted sum when the
   file has the energy histogram (EMsoft ``accum_e``), else the highest energy (the beam's);
   EMsoft's per-site sets (``numset``) are summed;
2. resampled to the square-Lambert grid of the twin's master resolution (area-averaged when
   coarser) — a stereographic input is reprojected — and to the renderer's cube map;
3. standardised (zero mean, unit variance over the sphere) and quantised to uint8: a few MB
   whatever the file's size (an EMsoft file is read one energy slice at a time);
4. checked: a southern hemisphere that is the inversion of the northern one is dropped (the
   pattern is centrosymmetric); otherwise both are kept (six cube faces). The pattern's
   invariance under the phase's point group is measured (``symmetry_error``: RMS difference
   between symmetry-equivalent directions over the pattern's standard deviation) and a
   pattern that is not invariant (a wrong frame, transposed or mirrored) is refused.

Conventions (EMsoft's, which kikuchipy shares): the northern hemisphere ``z >= 0`` of the
crystal's Cartesian frame (orix's: x || a, z || c* ... for the cubic phases here the standard
frame) on ``[-L, L]^2``, ``L = sqrt(pi / 2)``, rows = Lambert Y, columns = Lambert X (an
HDF5 dataset read in C order); the southern array is read at Lambert(x, y, |z|).
Stereographic: ``(x, y) / (1 + |z|)`` on ``[-1, 1]^2``, same row/column order.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

from . import kernels
from .master import MasterPattern, cube_side, lambert_sample, lambert_side, quantise

#: A pattern whose symmetry-equivalent directions differ by more than this (RMS / std) is refused.
SYMMETRY_REFUSE = 0.35
SYMMETRY_WARN = 0.15
#: Southern hemisphere within this (RMS / std) of the inverted northern one: centrosymmetric.
CENTRO_TOLERANCE = 0.03


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - h5py comes with orix
        raise ImportError("reading HDF5 master patterns needs h5py (pip install h5py)") from exc
    return h5py


# ---------------------------------------------------------------------- energy reduction
def _reduce_energy(get_slice, energies, weights, energy_kv):
    """One float64 (n, n) array from energy slices ``get_slice(i)`` (read one at a time)."""
    ne = len(energies)
    if ne == 1:
        return np.asarray(get_slice(0), np.float64)
    e = np.asarray(energies, np.float64)
    if energy_kv is not None:
        order = np.argsort(e)
        es = e[order]
        x = float(np.clip(energy_kv, es[0], es[-1]))
        k = int(np.clip(np.searchsorted(es, x) - 1, 0, ne - 2))
        t = (x - es[k]) / (es[k + 1] - es[k]) if es[k + 1] > es[k] else 0.0
        return (1 - t) * np.asarray(get_slice(int(order[k])), np.float64) + t * np.asarray(
            get_slice(int(order[k + 1])), np.float64
        )
    if weights is not None and np.sum(weights) > 0:
        w = np.asarray(weights, np.float64) / np.sum(weights)
        acc = None
        for i in range(ne):
            if w[i] > 0:
                s = w[i] * np.asarray(get_slice(i), np.float64)
                acc = s if acc is None else acc + s
        return acc
    return np.asarray(get_slice(int(np.argmax(e))), np.float64)


# ---------------------------------------------------------------------- readers
def read_emsoft(path, energy_kv=None) -> dict:
    """The master pattern(s) of an EMsoft ``EMEBSDmaster`` HDF5 file: dict with ``north``,
    ``south`` (or None), ``projection``, ``energies``, ``kv``, ``space_group``, ``lattice``."""
    h5py = _h5py()
    with h5py.File(path, "r") as f:
        g = f["EMData/EBSDmaster"]
        if "mLPNH" in g:
            names, projection = ("mLPNH", "mLPSH"), "lambert"
        elif "masterSPNH" in g:
            names, projection = ("masterSPNH", "masterSPSH"), "stereographic"
        else:
            raise ValueError(f"{path}: no mLPNH / masterSPNH in EMData/EBSDmaster")
        energies = np.atleast_1d(np.asarray(g["EkeVs"][()], np.float64)) if "EkeVs" in g else None
        weights = None
        if "EMData/MCOpenCL/accum_e" in f:  # Monte-Carlo energy histogram (nE, ...)
            acc = np.asarray(f["EMData/MCOpenCL/accum_e"][()], np.float64)
            weights = acc.reshape(acc.shape[0], -1).sum(axis=1)
        out = {"projection": projection}
        for key, name in zip(("north", "south"), names):
            if name not in g:
                out[key] = None
                continue
            d = g[name]
            shape = d.shape
            if len(shape) == 2:
                out[key] = np.asarray(d[()], np.float64)
                continue
            ne = shape[-3]
            en = energies if energies is not None and len(energies) == ne else np.arange(ne, dtype=float)
            w = weights if weights is not None and len(weights) == ne else None

            def get(i, d=d):
                s = d[..., i, :, :]
                return s.sum(axis=tuple(range(s.ndim - 2))) if s.ndim > 2 else s

            out[key] = _reduce_energy(get, en, w, energy_kv)
        cd = f.get("CrystalData")
        out["space_group"] = (
            int(np.asarray(cd["SpaceGroupNumber"][()]).ravel()[0])
            if cd is not None and "SpaceGroupNumber" in cd
            else None
        )
        out["lattice"] = (
            tuple(float(v) for v in np.asarray(cd["LatticeParameters"][()]).ravel())
            if cd is not None and "LatticeParameters" in cd
            else None
        )
        out["energies"] = energies
        out["kv"] = (
            float(energy_kv)
            if energy_kv is not None
            else (float(energies.max()) if energies is not None else None)
        )
    return out


def read_kikuchipy(path, energy_kv=None) -> dict:
    """A kikuchipy ``EBSDMasterPattern`` saved to HDF5 (HyperSpy ``.hspy`` layout:
    ``Experiments/<title>/data`` with ``axis-i`` groups; attributes ``projection`` and
    ``hemisphere`` anywhere under the experiment)."""
    h5py = _h5py()
    with h5py.File(path, "r") as f:
        exps = f["Experiments"]
        exp = exps[next(iter(exps))]
        d = exp["data"]
        axes = []
        for i in range(d.ndim):
            a = exp.get(f"axis-{i}")
            attrs = dict(a.attrs) if a is not None else {}
            name = attrs.get("name", "")
            name = name.decode() if isinstance(name, bytes) else str(name)
            axes.append((name.lower(), float(attrs.get("offset", 0.0)), float(attrs.get("scale", 1.0))))
        found = {}

        def visit(_, obj):
            for k in ("projection", "hemisphere"):
                if k in obj.attrs and k not in found:
                    v = obj.attrs[k]
                    found[k] = v.decode() if isinstance(v, bytes) else str(v)

        exp.visititems(visit)
        names = [a[0] for a in axes]
        hemi_ax = names.index("hemisphere") if "hemisphere" in names else (0 if d.ndim == 4 else None)
        en_ax = names.index("energy") if "energy" in names else None
        if en_ax is None and d.ndim - (hemi_ax is not None) == 3:
            en_ax = 1 if hemi_ax == 0 else 0
        ne = d.shape[en_ax] if en_ax is not None else 1
        energies = (axes[en_ax][1] + axes[en_ax][2] * np.arange(ne)) if en_ax is not None else None
        hemis = d.shape[hemi_ax] if hemi_ax is not None else 1
        out = {"projection": found.get("projection", "lambert").lower()}
        hemi_name = found.get("hemisphere", "both" if hemis == 2 else "upper").lower()
        for h in range(hemis):

            def get(i, h=h):
                idx = [slice(None)] * d.ndim
                if hemi_ax is not None:
                    idx[hemi_ax] = h
                if en_ax is not None:
                    idx[en_ax] = i
                return d[tuple(idx)]

            arr = _reduce_energy(get, energies if energies is not None else [0.0], None, energy_kv)
            key = "north" if (h == 0 and hemi_name in ("upper", "north", "both")) else "south"
            out[key] = arr
        out.setdefault("north", None)
        out.setdefault("south", None)
        out["energies"] = energies
        out["kv"] = (
            float(energy_kv)
            if energy_kv is not None
            else (float(np.max(energies)) if energies is not None else None)
        )
        out["space_group"], out["lattice"] = None, None
    return out


def read_master_file(path, energy_kv=None) -> dict:
    """EMsoft or kikuchipy HDF5, by layout."""
    h5py = _h5py()
    with h5py.File(path, "r") as f:
        emsoft = "EMData/EBSDmaster" in f
        kikuchipy = "Experiments" in f
    if emsoft:
        return read_emsoft(path, energy_kv)
    if kikuchipy:
        return read_kikuchipy(path, energy_kv)
    raise ValueError(
        f"{path}: neither an EMsoft master pattern (EMData/EBSDmaster) nor a kikuchipy/HyperSpy "
        "file (Experiments/...)"
    )


def from_array(
    array, *, projection="lambert", hemisphere=None, energies=None, energy_kv=None, weights=None
) -> dict:
    """A NumPy master: (n, n) one hemisphere; (2, n, n) north and south; (nE, n, n) with
    ``energies``; (2, nE, n, n). ``hemisphere``: "north"/"upper", "south"/"lower" or "both"."""
    a = np.asarray(array, np.float64)
    hemi = (
        hemisphere
        or ("both" if (a.ndim == 4 or (a.ndim == 3 and energies is None and a.shape[0] == 2)) else "north")
    ).lower()
    if a.ndim == 2:
        a = a[None, None]
    elif a.ndim == 3:
        a = a[:, None] if hemi == "both" else a[None]
    if hemi in ("north", "upper", "south", "lower") and a.shape[0] != 1:
        raise ValueError("one hemisphere named but the array has two")
    ne = a.shape[1]
    en = np.asarray(energies, np.float64) if energies is not None else np.arange(ne, dtype=float)
    if len(en) != ne:
        raise ValueError(f"{len(en)} energies for {ne} energy slices")
    reduced = [_reduce_energy(lambda i, h=h: a[h, i], en, weights, energy_kv) for h in range(a.shape[0])]
    if hemi in ("south", "lower"):
        north, south = None, reduced[0]
    else:
        north, south = reduced[0], (reduced[1] if len(reduced) > 1 else None)
    return {
        "north": north,
        "south": south,
        "projection": str(projection).lower(),
        "energies": energies,
        "kv": float(energy_kv)
        if energy_kv is not None
        else (float(np.max(energies)) if energies is not None else None),
        "space_group": None,
        "lattice": None,
    }


# ---------------------------------------------------------------------- resampling
def _stereo_to_lambert(st, n, south: bool) -> np.ndarray:
    dirs = kernels.lambert_directions(n)
    z = np.abs(dirs[..., 2])
    X, Y = dirs[..., 0] / (1.0 + z), dirs[..., 1] / (1.0 + z)
    m = st.shape[0]
    return ndimage.map_coordinates(
        st, [(Y + 1) * (m - 1) / 2, (X + 1) * (m - 1) / 2], order=1, mode="nearest"
    )


def _resample_lambert(a, n) -> np.ndarray:
    m = a.shape[0]
    if m == n:
        return a
    if m > n:  # area-average before decimating
        a = ndimage.gaussian_filter(a, 0.5 * (m - 1) / (n - 1), mode="nearest")
    c = np.linspace(0, m - 1, n)
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return ndimage.map_coordinates(a, [yy, xx], order=1, mode="nearest")


def symmetry_error(north, south, point_group, n_dirs: int = 4000, seed: int = 0) -> float:
    """RMS difference of the pattern between symmetry-equivalent directions, over its std."""
    if point_group is None:
        return 0.0
    ops = np.asarray(point_group.proper_subgroup.to_matrix())  # the rotations (a master need not be
    # centrosymmetric, so the improper operations are not tested)
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n_dirs, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    v0 = lambert_sample(north, south, d)
    sd = float(np.std(v0)) or 1.0
    diffs = []
    for op in ops:
        if np.allclose(op, np.eye(3)):
            continue
        diffs.append(lambert_sample(north, south, d @ op.T) - v0)
    if not diffs:
        return 0.0
    return float(np.sqrt(np.mean(np.square(diffs))) / sd)


def master_from_hemispheres(
    raw: dict,
    material: int,
    kv: float,
    resolution_deg: float,
    *,
    centrosymmetric: Optional[bool] = None,
    point_group=None,
    source: str = "array",
    validate: bool = True,
) -> MasterPattern:
    """A cached-format :class:`MasterPattern` from float hemispheres (see the module docstring)."""
    north, south = raw.get("north"), raw.get("south")
    if north is None and south is None:
        raise ValueError("no master pattern data")
    proj = raw.get("projection", "lambert")
    if proj not in ("lambert", "stereographic"):
        raise ValueError(f"unknown projection {proj!r} (lambert or stereographic)")
    n = lambert_side(resolution_deg)

    def to_lambert(a, is_south):
        if a is None:
            return None
        a = np.asarray(a, np.float64)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"a master hemisphere must be square, got {a.shape}")
        return _stereo_to_lambert(a, n, is_south) if proj == "stereographic" else _resample_lambert(a, n)

    north, south = to_lambert(north, False), to_lambert(south, True)
    if north is None:  # only the south given: the north is its inversion (Friedel)
        north, south = south[::-1, ::-1].copy(), None
    if south is not None:
        inv = north[::-1, ::-1]
        rel = float(np.sqrt(np.mean((south - inv) ** 2)) / (np.std(north) or 1.0))
        centro = rel < CENTRO_TOLERANCE if centrosymmetric is None else bool(centrosymmetric)
        if centro:
            south = None
    elif centrosymmetric is False:
        raise ValueError("a non-centrosymmetric master needs its southern hemisphere")
    elif point_group is not None and not getattr(point_group, "contains_inversion", True):
        warnings.warn(
            "one hemisphere of a non-centrosymmetric phase: assuming Friedel symmetry", stacklevel=3
        )
    both = [north] if south is None else [north, south]
    mu = float(np.mean(both))
    sd = float(np.std(both)) or 1.0
    north = (north - mu) / sd
    south = None if south is None else (south - mu) / sd
    err = symmetry_error(north, south, point_group) if validate else float("nan")
    if validate and err > SYMMETRY_REFUSE:
        raise ValueError(
            f"master pattern is not invariant under the phase's point group (symmetry error "
            f"{err:.2f}): wrong phase, or a transposed / mirrored projection?"
        )
    if validate and err > SYMMETRY_WARN:
        warnings.warn(f"master pattern symmetry error {err:.2f}", stacklevel=3)
    nf = cube_side(resolution_deg)
    faces = 3 if south is None else 6
    cube = lambert_sample(north, south, kernels.cube_directions(nf, faces).reshape(-1, 3)).reshape(
        faces, nf, nf
    )
    arrays = [north, cube] + ([south] if south is not None else [])
    lo, scale, codes = quantise(*arrays)
    mp = MasterPattern(
        int(material),
        float(kv),
        codes[0],
        lo,
        scale,
        0,
        0.0,
        float(resolution_deg),
        south=codes[2] if south is not None else None,
        source=source,
        cube=codes[1],
    )
    mp.symmetry_error = err
    return mp


# ---------------------------------------------------------------------- writer
def write_emsoft(
    path,
    north,
    south=None,
    energies=(20.0,),
    *,
    space_group: int = 225,
    lattice=(0.4, 0.4, 0.4, 90.0, 90.0, 90.0),
    weights=None,
) -> None:
    """Write master hemispheres (n, n) or (nE, n, n) in EMsoft's EMEBSDmaster layout (mLPNH /
    mLPSH as (1, nE, n, n) float32, EkeVs, CrystalData; ``weights``: an energy histogram as
    ``EMData/MCOpenCL/accum_e``). ``south`` None: the inversion of the north."""
    h5py = _h5py()
    nh = np.asarray(north, np.float32)
    nh = nh[None] if nh.ndim == 2 else nh
    sh = nh[:, ::-1, ::-1] if south is None else np.asarray(south, np.float32)
    sh = sh[None] if sh.ndim == 2 else sh
    with h5py.File(path, "w") as f:
        g = f.create_group("EMData/EBSDmaster")
        g["mLPNH"] = nh[None]
        g["mLPSH"] = np.ascontiguousarray(sh)[None]
        g["EkeVs"] = np.asarray(energies, np.float32)
        g["numEbins"] = np.int32(len(energies))
        if weights is not None:
            f["EMData/MCOpenCL/accum_e"] = np.asarray(weights, np.int32).reshape(-1, 1, 1)
        cd = f.create_group("CrystalData")
        cd["SpaceGroupNumber"] = np.int32(space_group)
        cd["LatticeParameters"] = np.asarray(lattice, np.float32)
        f.create_group("EMheader/EBSDmaster")["ProgramName"] = np.bytes_("de-twin")


# ---------------------------------------------------------------------- one call
def load_master_pattern(
    source,
    material: int,
    *,
    resolution_deg: float = 0.15,
    energy_kv=None,
    kv: Optional[float] = None,
    projection=None,
    hemisphere=None,
    energies=None,
    weights=None,
    centrosymmetric: Optional[bool] = None,
    validate: bool = True,
    point_group=None,
) -> MasterPattern:
    """A user :class:`MasterPattern` for ``material`` from a file path (EMsoft / kikuchipy HDF5)
    or a NumPy array (with ``projection``, ``hemisphere``, ``energies``). ``kv``: the beam energy
    it is for (default: the selected energy, else the file's highest)."""
    from ..crystal import phase_for
    from ..specimen.materials import material as material_by_id

    if isinstance(source, (str, Path)):
        raw = read_master_file(str(source), energy_kv)
        src = str(source)
    else:
        raw = from_array(
            source,
            projection=projection or "lambert",
            hemisphere=hemisphere,
            energies=energies,
            energy_kv=energy_kv,
            weights=weights,
        )
        src = "array"
    if projection is not None and isinstance(source, (str, Path)):
        raw["projection"] = projection
    beam = kv if kv is not None else raw.get("kv")
    if beam is None:
        raise ValueError("the master pattern's beam energy is unknown: pass kv=")
    phase = phase_for(int(material))
    if phase is None:
        raise ValueError(f"material {material} is not crystalline")
    crystal = material_by_id(int(material)).crystal
    if (
        raw.get("space_group")
        and crystal is not None
        and crystal.space_group
        and int(raw["space_group"]) != int(crystal.space_group)
    ):
        raise ValueError(
            f"the file's space group {raw['space_group']} is not material {material}'s "
            f"({crystal.space_group})"
        )
    if raw.get("lattice") and crystal is not None and crystal.lattice:
        a_file, a_mat = float(raw["lattice"][0]), float(crystal.lattice[0])
        if a_file > 0 and abs(a_file - a_mat) / a_mat > 0.03:
            warnings.warn(
                f"the file's lattice a = {a_file:g} nm differs from the material's {a_mat:g} nm", stacklevel=2
            )
    pg = point_group if point_group is not None else phase.point_group
    return master_from_hemispheres(
        raw,
        int(material),
        float(beam),
        resolution_deg,
        centrosymmetric=centrosymmetric,
        point_group=pg,
        source=src,
        validate=validate,
    )


__all__ = [
    "load_master_pattern",
    "master_from_hemispheres",
    "read_emsoft",
    "read_kikuchipy",
    "read_master_file",
    "from_array",
    "symmetry_error",
    "write_emsoft",
]
