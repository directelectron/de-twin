"""Per-raster-pixel diffraction descriptors shared by the SAED and STEM renderers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hashing import mix_cell
from ..specimen.fieldmap import grain_id_from_hash
from ..specimen.materials import GRAINS_PER_MATERIAL, MATERIALS, absorption_lengths_nm
from .diffraction import thickness_bin_for

FALLBACK_GRAIN_CELL_NM = 25.0
MIN_TRANSMISSION = 1.0e-4
DIFFUSE_CUTOFF_LENGTHS = 10.0  # beyond ~10 mean free paths nothing comes out (thick bars)


@dataclass
class Samples:
    mat: np.ndarray  # uint8
    t: np.ndarray  # float32, tilt-corrected projected thickness
    T: np.ndarray  # float32 transmission exp(-t/Lambda), clamped (the pattern weight)
    own_gid: np.ndarray  # int64, -1 = none (rim rule applied)
    gid: np.ndarray  # int64, own grain or (STEM) the fallback cell grain
    thickness_bin: np.ndarray  # int64
    crystallinity: np.ndarray  # float64 per pixel (1 where no grain)
    diffuse_w: np.ndarray  # float32 fraction redistributed into the diffuse background
    under_mat: np.ndarray = None  # uint8 material of the amorphous under layer (0: none)
    under_w: np.ndarray = None  # float32 its diffuse fraction


def describe(
    fm, optics, grains, crystallinity, *, fallback_grains: bool = True, diffuse: bool = True
) -> Samples:
    mat = fm.material_id
    t = fm.thickness_nm.astype(np.float32) * np.float32(optics.thickness_tilt_factor)
    lam = absorption_lengths_nm(optics.ht_kv)
    lam_px = lam[mat]
    T = np.exp(-t / lam_px).astype(np.float32)
    umat = uw = None
    if fm.under_thickness_nm is not None:
        # the under layer scatters first: its diffuse share, and what it lets through
        umat = fm.under_material
        tu = fm.under_thickness_nm.astype(np.float32) * np.float32(optics.thickness_tilt_factor)
        Tu = np.exp(-tu / lam[umat]).astype(np.float32)
        uw = (
            ((1.0 - Tu) * np.exp(-tu / (DIFFUSE_CUTOFF_LENGTHS * lam[umat]))).astype(np.float32)
            if diffuse
            else np.zeros_like(tu)
        )
        uw = np.where(umat > 0, uw, np.float32(0.0))

    gid = fm.grain_id.astype(np.int64)
    n_grains = len(grains) if grains is not None else 0
    own_gid = np.where((gid >= 0) & (gid < n_grains) & ((gid // GRAINS_PER_MATERIAL) == mat), gid, -1)

    diff_gid = own_gid.copy()
    crystalline = np.array([m.crystalline for m in MATERIALS])[mat]
    if fallback_grains and grains is not None:
        need = (own_gid < 0) & crystalline
        if need.any():
            rows, cols = np.nonzero(need)
            x_um, y_um = fm.view.pixel_to_world(rows, cols)
            cx = np.floor(x_um * 1000.0 / FALLBACK_GRAIN_CELL_NM).astype(np.int64)
            cy = np.floor(y_um * 1000.0 / FALLBACK_GRAIN_CELL_NM).astype(np.int64)
            diff_gid[rows, cols] = grain_id_from_hash(mat[rows, cols].astype(np.int64), mix_cell(cx, cy))

    cryst = np.ones(t.shape)
    if crystallinity is not None:
        m = diff_gid >= 0
        if m.any():
            cryst[m] = np.asarray(crystallinity(diff_gid[m]), float)
        ungrained = ~m & crystalline
        if ungrained.any():  # the specimen decides: glassy (in-situ, not nucleated) or powder
            cryst[ungrained] = float(np.asarray(crystallinity(np.array([-1])), float)[0])
    dw = (
        ((1.0 - T) * np.exp(-t / (DIFFUSE_CUTOFF_LENGTHS * lam_px))).astype(np.float32)
        if diffuse
        else np.zeros_like(t)
    )
    if umat is not None:  # the primary sees only what the under layer let through
        T = T * Tu
        dw = dw * Tu
    return Samples(
        mat,
        t,
        np.maximum(T, MIN_TRANSMISSION).astype(np.float32),
        own_gid,
        diff_gid,
        thickness_bin_for(t),
        cryst,
        dw,
        umat,
        uw,
    )
