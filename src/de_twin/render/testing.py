"""Tiny stand-ins for the specimen and the camera, for tests and notebooks.

``SyntheticSpecimen`` implements the documented ``Specimen`` duck type
(``rasterize``, ``grains``, ``crystallinity``, ``config.seed``) with a
continuous film, circular holes and spherical particles at fixed world
positions. ``StubCamera`` has the ``CameraModel`` attributes the optics use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from ..specimen.fieldmap import FieldMap, GrainTable, ViewWindow
from ..specimen.materials import GRAINS_PER_MATERIAL, MaterialId


@dataclass(frozen=True)
class StubCamera:
    name: str = "DE16"
    sensor_shape: tuple[int, int] = (4096, 4096)
    pixel_um: float = 6.5
    bit_depth: int = 12


@dataclass
class Particle:
    x_um: float
    y_um: float
    radius_nm: float
    material: int = MaterialId.GOLD
    grain: int = 0  # k within the material's grain block (grain id = material * 512 + k)


@dataclass
class SyntheticSpecimen:
    film_material: int = MaterialId.AMORPHOUS_CARBON
    film_thickness_nm: float = 20.0
    holes: list = field(default_factory=list)  # (x_um, y_um, radius_um)
    particles: list = field(default_factory=list)  # Particle
    seed: int = 42
    crystallinity_value: float = 1.0
    grains: GrainTable | None = None
    generation: int = 0
    rasterize_calls: int = 0

    def __post_init__(self):
        self.config = SimpleNamespace(seed=self.seed)
        if self.grains is None:
            self.grains = _cached_grains(self.seed)

    def update(self, time_s: float, holder=None) -> None:  # static specimen (DigitalTwin duck type)
        pass

    def crystallinity(self, grain_id):
        return np.full(np.shape(grain_id), self.crystallinity_value, np.float64)

    def rasterize(self, view: ViewWindow, layers: frozenset = frozenset()) -> FieldMap:
        self.rasterize_calls += 1
        field_map = FieldMap.empty(view, layers)
        field_map.generation = self.generation
        x, y = view.world_grid()
        field_map.material_id[:] = self.film_material
        field_map.thickness_nm[:] = self.film_thickness_nm if self.film_material else 0.0
        for hx, hy, hr in self.holes:
            m = (x - hx) ** 2 + (y - hy) ** 2 < hr * hr
            field_map.material_id[m] = MaterialId.VACUUM
            field_map.thickness_nm[m] = 0.0
        for p in self.particles:
            r_um = p.radius_nm / 1000.0
            bx0, by0, bx1, by1 = p.x_um - r_um, p.y_um - r_um, p.x_um + r_um, p.y_um + r_um
            xmin, ymin, xmax, ymax = view.bounds_um()
            if bx1 < xmin or bx0 > xmax or by1 < ymin or by0 > ymax:
                continue
            d2 = ((x - p.x_um) ** 2 + (y - p.y_um) ** 2) / (r_um * r_um)
            m = d2 < 1.0
            if not m.any():
                continue
            field_map.material_id[m] = p.material
            field_map.thickness_nm[m] = (2.0 * p.radius_nm * np.sqrt(1.0 - d2[m])).astype(np.float32)
            gid = int(p.material) * GRAINS_PER_MATERIAL + int(p.grain)
            field_map.grain_id[m] = gid
        return field_map


def on_zone_grains(grain_id: int, zone=(0, 0, 1), seed: int = 42, in_plane_rad: float = 0.0) -> GrainTable:
    """A copy of the seed's grain table with ``grain_id`` exactly on zone axis ``zone``
    (the [uvw] direction along the beam, rotated ``in_plane_rad`` about it)."""
    from ..crystal.orientation import align, rot_z, rotation_from_crystal_to_lab

    g = _cached_grains(seed)
    q = g.quaternions.copy()
    q[grain_id] = rotation_from_crystal_to_lab(rot_z(in_plane_rad) @ align(zone, (0.0, 0.0, 1.0)))[0]
    return GrainTable(q, g.nucleation_u)


_GRAINS: dict = {}


def _cached_grains(seed: int) -> GrainTable:
    if seed not in _GRAINS:
        _GRAINS[seed] = GrainTable.generate(seed)
    return _GRAINS[seed]


# ----------------------------------------------------------- analysis helpers
def radial_power_spectrum(
    image: np.ndarray,
    pixel_nm: float,
    sector_rad: float | None = None,
    half_width_deg: float = 15.0,
    nbins: int = 256,
):
    """(k [1/nm], radially averaged |FFT|^2), optionally within a +-sector (mod 180 deg)."""
    a = np.asarray(image, np.float64)
    f = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean()))) ** 2
    ny, nx = a.shape
    ky = (np.arange(ny) - ny // 2)[:, None] / (ny * pixel_nm)
    kx = (np.arange(nx) - nx // 2)[None, :] / (nx * pixel_nm)
    k = np.hypot(kx, ky)
    m = np.ones(k.shape, bool)
    if sector_rad is not None:
        ang = np.arctan2(np.broadcast_to(ky, k.shape), np.broadcast_to(kx, k.shape))
        d = np.angle(np.exp(2j * (ang - sector_rad))) / 2.0
        m = np.abs(d) < np.radians(half_width_deg)
    edges = np.linspace(0.0, 0.5 / pixel_nm, nbins + 1)
    index = np.digitize(k[m], edges)
    s = np.bincount(index, weights=f[m], minlength=nbins + 2)[1 : nbins + 1]
    n = np.bincount(index, minlength=nbins + 2)[1 : nbins + 1]
    return 0.5 * (edges[1:] + edges[:-1]), s / np.maximum(n, 1)


def ctf2(k, defocus_nm, wavelength_nm, cs_mm=1.2, amplitude_contrast=0.07):
    """(sin chi - kappa cos chi)^2 with the twin's sign convention (df < 0 underfocus)."""
    kap = amplitude_contrast / np.sqrt(1.0 - amplitude_contrast**2)
    chi = np.pi * wavelength_nm * defocus_nm * k**2 + 0.5 * np.pi * cs_mm * 1e6 * wavelength_nm**3 * k**4
    return (np.sin(chi) - kap * np.cos(chi)) ** 2


def fit_defocus(
    image,
    pixel_nm,
    wavelength_nm,
    df_grid_nm,
    cs_mm=1.2,
    k_range=(0.3, 3.0),
    sector_rad=None,
    amplitude_contrast=0.07,
):
    """Brute-force CTF fit: the defocus whose CTF^2 best correlates with the
    background-flattened radial power spectrum. Returns (defocus_nm, score)."""
    k, p = radial_power_spectrum(image, pixel_nm, sector_rad)
    sel = (k > k_range[0]) & (k < k_range[1])
    pr = p[sel]
    flat = pr / np.convolve(pr, np.ones(31) / 31.0, "same")
    best = (-2.0, float("nan"))
    for df in df_grid_nm:
        c = np.corrcoef(ctf2(k[sel], df, wavelength_nm, cs_mm, amplitude_contrast), flat)[0, 1]
        if c > best[0]:
            best = (float(c), float(df))
    return best[1], best[0]


def phase_correlation_shift(a, b) -> tuple[int, int]:
    """Integer (dx, dy) such that ``a`` ~ ``b`` shifted by (dx, dy)."""
    A = np.fft.fft2(a - np.mean(a))
    B = np.fft.fft2(b - np.mean(b))
    r = A * np.conj(B)
    r /= np.abs(r) + 1e-12
    c = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    ny, nx = c.shape
    return int((ix + nx // 2) % nx - nx // 2), int((iy + ny // 2) % ny - ny // 2)
