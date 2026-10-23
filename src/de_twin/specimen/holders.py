"""Holders: the macroscopic support geometry (port of VirtualSpecimen ``Holder.cpp``).

Four holders, generated from the master seed:

* ``mesh_grid`` - 3.05 mm, 200-mesh Cu grid: seeded pitch/bar jitter (+/-3 %), rotation
  (+/-0.15 rad), lattice offset, 120 um rim, rounded hole corners, 2-6 torn squares with a bent
  bar fragment, and a support film (holey R2/2, lacey log-normal holes, continuous, vitreous ice
  with edge thickening) with 8 % broken squares and 3 nm / 2 nm value-noise granularity;
* ``waffle_grid`` - the same generator with the waffle parameter block (62.5 um pitch, 32 um bars,
  25 um thick, sharp windows, 0-2 torn squares);
* ``fib_liftout`` - the authored half-moon FIB grid (``samples/FIB Liftout.svg``): body
  silhouette, 15 white through-holes, 3-5 flags each carrying one lamella (Curtain-profile body,
  Cu support rails/tabs, dark Pt backing strip) with a seeded BulkSample kind permutation;
* ``insitu_heating_chip`` - the authored MEMS heater (``samples/Insitu2.svg``): SiN membrane,
  Pt serpentine heater, ten transparent openings (six ellipses, two with SiN collars, and four
  rounded rectangles) which are the placement areas.

Bars, rim, bodies and membranes are sampled analytically per pixel (never rasterised as
primitives), so the holder pass costs O(pixels) at every magnification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..hashing import SeedKind, hash_seed, rng_for, uniform_from_hash
from . import raster_nb as _rnb
from .geometry import (
    AABB,
    NM_PER_UM,
    _separable_noise_tables,
    polygon_mask_rows,
    rotated_rect_half_extent,
    value_noise,
    value_noise_separable,
)
from .materials import MaterialId
from .options import SpecimenOptions
from .raster import Layer, PrimitiveSet, Profile, Shape, iter_patches

# -- tuning constants (Holder.cpp) ------------------------------------------------------------
MESH_JITTER_FRAC = 0.03
MESH_ROTATION_RAD = 0.15
DAMAGED_MIN, DAMAGED_MAX = 2, 6
WAFFLE_DAMAGED_MIN, WAFFLE_DAMAGED_MAX = 0, 2
TORN_BAR_LENGTH_FRAC = 0.55
TORN_BAR_WIDTH_FRAC = 0.16
LACEY_HOLES_PER_AREA = 64
LACEY_SIGMA_LOG = 0.45
LACEY_MIN_UM, LACEY_MAX_UM = 0.5, 5.0
LAMELLA_THICKNESS_JITTER = 0.20
ICE_EDGE_GRADIENT = 1.5
CHIP_FILM_GRANULARITY_NM = 1.0

BULK_SAMPLE_KINDS = ("PrecipitateMatrix", "PNJunction", "StrainedInclusion", "Polycrystal", "Multilayer")
_POST_SAMPLE_TO_KIND = {
    "precipitate_matrix": 0,
    "pn_junction": 1,
    "strained_inclusion": 2,
    "polycrystal": 3,
    "multilayer": 4,
}

# -- authored FIB liftout geometry (samples/FIB Liftout.svg) -----------------------------------
FIB_SVG_ORIGIN = (61.1207465, 149.5847275)
FIB_SVG_SCALE_UM = 36.69351224175573  # 41.5604805 SVG units -> 1525 um
FIB_SVG_DISK_CENTRE = (69.903229, 145.57257)
FIB_SVG_DISK_RADIUS = 41.5604805
FIB_SVG_RIM_HALF_WIDTH = 1.0
FIB_FLAG_ROTATION_RAD = -0.8795157414429432  # -50.39254 deg
FIB_BODY_SVG = np.array(
    [
        (37.25806, 167.83871),
        (40.21251, 165.47516),
        (43.16695, 163.11160),
        (46.08245, 162.20126),
        (48.90608, 164.71964),
        (51.72971, 167.23801),
        (54.27490, 164.57620),
        (56.81231, 161.76967),
        (59.34973, 158.96313),
        (61.88714, 156.15660),
        (62.49906, 153.46021),
        (59.65818, 150.96130),
        (60.99023, 149.81795),
        (63.94356, 152.18288),
        (66.46169, 150.23921),
        (65.07919, 147.62961),
        (63.43065, 144.73717),
        (66.56245, 146.36795),
        (69.26618, 146.76331),
        (70.23351, 143.96756),
        (67.29079, 141.58943),
        (69.16525, 140.50232),
        (72.10726, 142.88132),
        (74.62798, 140.30058),
        (72.55608, 137.79021),
        (71.03048, 134.96309),
        (74.20041, 136.67845),
        (76.99306, 137.60977),
        (78.13245, 134.79231),
        (75.31084, 132.27191),
        (77.07565, 131.36911),
        (80.02646, 133.46063),
        (82.54283, 130.63522),
        (85.05921, 127.80981),
        (87.57559, 124.98441),
        (90.09197, 122.15900),
        (89.68165, 119.63356),
        (86.64443, 117.37734),
        (85.83750, 114.55228),
        (87.42303, 111.11697),
        (89.82352, 110.13489),
        (93.14945, 111.93855),
        (96.47540, 113.74221),
        (99.41704, 115.95802),
        (101.44856, 119.14989),
        (103.48009, 122.34175),
        (105.51161, 125.53363),
        (107.54313, 128.72549),
        (108.76100, 132.26706),
        (109.64684, 135.94540),
        (110.47752, 139.63654),
        (111.13319, 143.36089),
        (110.76568, 147.12286),
        (110.32033, 150.88010),
        (109.83283, 154.63202),
        (109.20853, 158.36246),
        (107.61985, 161.78891),
        (106.08661, 165.24783),
        (104.57909, 168.71806),
        (102.22690, 171.62183),
        (99.46504, 174.20723),
        (96.58164, 176.65634),
        (93.57850, 178.95699),
        (90.45088, 181.08505),
        (87.19140, 183.00467),
        (83.79286, 184.66484),
        (80.25256, 185.99455),
        (76.52153, 186.52827),
        (72.74637, 186.77957),
        (68.96896, 186.99435),
        (65.23999, 186.57301),
        (61.57895, 185.61833),
        (57.92746, 184.62741),
        (54.28049, 183.62013),
        (50.63694, 182.60049),
        (47.66239, 180.28658),
        (44.72361, 177.90358),
        (41.78482, 175.52057),
        (38.84604, 173.13757),
        (35.90728, 170.75458),
    ]
)
FIB_HOLE_CENTRES_SVG = np.array(
    [
        (80.501411, 136.03064),
        (81.341064, 136.73569),
        (80.308540, 135.09038),
        (81.426682, 135.26735),
        (82.298325, 135.83516),
        (65.024193, 154.38809),
        (68.477554, 149.84677),
        (69.170219, 149.14583),
        (72.487564, 144.41895),
        (73.359207, 144.98676),
        (76.153564, 140.50383),
        (77.025208, 141.07164),
        (76.887497, 139.77454),
        (77.759140, 140.34235),
        (74.377823, 145.69026),
    ]
)
FIB_HOLE_RADIUS_SVG = (0.33266127, 0.29939514)
FIB_LAMELLA_CENTRES_SVG = np.array(
    [
        (61.1207465, 149.5847275),
        (65.1199105, 144.7731350),
        (69.1088905, 140.1525650),
        (72.9288125, 135.3073900),
        (76.8644895, 130.8921500),
    ]
)

# -- authored in-situ heater geometry (samples/Insitu2.svg) ------------------------------------
INSITU_SVG_ORIGIN = (71.7487375, 146.579695)
INSITU_SVG_SCALE_UM = 3.5
INSITU_HEATER_SVG = np.array(
    [
        (31.17406, 155.06852),
        (37.22682, 154.84947),
        (41.91255, 151.18835),
        (43.18968, 145.27960),
        (44.19959, 139.28394),
        (45.79398, 133.43251),
        (49.13371, 128.38281),
        (53.59978, 124.26817),
        (58.56631, 120.76513),
        (64.03767, 118.14211),
        (70.00534, 117.12712),
        (76.04483, 117.73001),
        (81.87776, 119.39834),
        (87.28070, 122.18241),
        (91.75598, 126.26872),
        (95.26329, 131.19884),
        (96.77731, 137.07013),
        (97.39105, 143.11736),
        (96.59690, 149.09421),
        (94.70253, 154.86901),
        (92.01094, 160.30967),
        (87.75664, 164.62212),
        (82.60559, 167.83016),
        (76.89665, 169.88308),
        (70.87325, 170.34785),
        (64.95303, 168.99779),
        (59.57547, 166.25817),
        (55.02739, 162.26646),
        (52.48707, 156.81034),
        (51.57911, 150.81728),
        (52.09288, 144.77919),
        (54.04424, 139.03857),
        (57.37334, 133.96932),
        (62.01079, 130.08507),
        (67.63784, 127.84455),
        (73.68220, 127.47098),
        (79.53948, 128.97053),
        (84.30708, 132.68922),
        (87.26109, 137.88354),
        (86.90368, 143.91057),
        (83.62919, 148.89026),
        (78.17062, 151.46997),
        (72.14186, 151.87090),
        (66.07680, 151.64676),
        (61.52648, 154.78786),
        (63.58577, 160.29012),
        (69.07029, 162.73925),
        (75.12055, 162.80201),
        (80.80717, 160.75188),
        (85.41813, 156.83578),
        (88.60283, 151.68422),
        (90.21236, 145.84185),
        (90.03379, 139.79869),
        (88.02630, 134.07879),
        (84.13100, 129.45450),
        (78.85798, 126.52120),
        (72.89419, 125.35852),
        (66.88450, 126.10016),
        (61.24312, 128.26961),
        (56.35290, 131.87255),
        (53.22592, 137.05344),
        (50.55963, 142.51797),
        (50.32751, 148.46569),
        (51.21632, 154.47927),
        (52.76801, 160.35104),
        (56.07314, 165.39570),
        (60.79954, 169.19467),
        (66.38003, 171.57461),
        (72.40497, 172.07411),
        (78.39966, 171.09212),
        (84.15789, 169.17186),
        (89.17822, 165.78039),
        (93.28583, 161.30941),
        (96.40819, 156.11696),
        (98.13644, 150.29102),
        (99.32714, 144.34529),
        (99.39208, 138.26570),
        (100.36659, 132.28738),
        (104.72137, 128.47401),
        (108.09312, 131.57950),
        (108.28625, 137.65337),
        (103.65431, 139.33753),
        (102.11994, 145.21675),
        (100.86942, 151.16766),
        (98.61490, 156.80558),
        (95.65128, 162.11107),
        (91.82003, 166.77640),
        (86.79445, 170.19913),
        (81.36260, 172.84187),
        (75.51415, 174.44704),
        (69.44791, 174.19252),
        (63.51555, 172.99110),
        (58.20153, 170.05757),
        (53.51165, 166.22163),
        (50.04294, 161.24444),
        (48.30817, 155.46694),
        (47.41431, 149.45428),
        (47.45344, 143.39185),
        (49.14210, 137.55976),
        (51.96345, 132.18907),
        (56.05387, 127.72381),
        (61.38696, 124.84953),
        (67.21566, 123.13440),
        (73.23956, 122.42847),
        (79.17378, 123.55239),
        (84.53142, 126.40069),
        (89.01888, 130.46912),
        (91.79712, 135.83492),
        (93.24242, 141.73452),
        (92.95723, 147.76794),
        (90.64365, 153.38127),
        (87.55559, 158.59840),
        (83.04519, 162.62897),
        (77.44389, 164.95766),
        (71.42322, 165.55541),
        (65.50338, 164.28242),
        (60.45572, 160.97903),
        (57.38014, 155.81390),
        (57.15899, 149.79007),
        (59.75287, 144.38498),
        (64.85447, 141.15812),
        (70.81736, 140.53151),
        (76.85341, 141.23920),
        (81.97260, 138.85270),
        (80.78777, 133.28043),
        (75.47591, 130.41948),
        (69.48616, 130.30980),
        (63.84264, 132.53945),
        (58.83614, 135.88640),
        (55.69550, 141.08231),
        (54.11746, 146.93201),
        (54.38761, 152.96654),
        (56.94741, 158.45271),
        (60.58544, 163.31958),
        (65.60918, 166.56735),
        (71.62408, 167.25307),
        (77.62550, 166.37406),
        (83.21981, 164.02683),
        (88.12713, 160.46315),
        (91.61571, 155.52561),
        (93.85143, 149.87375),
        (95.02359, 143.93748),
        (94.25955, 137.91752),
        (92.24334, 132.19567),
        (88.85063, 127.17688),
        (84.02219, 123.52597),
        (78.37782, 121.32948),
        (72.38550, 120.37252),
        (66.34138, 120.95615),
        (60.71719, 123.18339),
        (55.61794, 126.48224),
        (51.24903, 130.70453),
        (48.40968, 136.02282),
        (46.87910, 141.90377),
        (45.89393, 147.90331),
        (45.85366, 153.94176),
        (44.74172, 159.86445),
        (39.69631, 163.16114),
        (33.78515, 164.49476),
        (31.02036, 161.11198),
    ]
)
INSITU_ELLIPSE_RADIUS_SVG = (1.490399, 1.4882812)
INSITU_ELLIPSE_CENTRES_SVG = np.array(
    [
        (63.932060, 137.004560),
        (68.748108, 136.981370),
        (73.605888, 137.054170),
        (78.335320, 136.971500),
        (68.115868, 146.571210),
        (75.381607, 146.588180),
    ]
)
INSITU_GRAY_CENTRES_SVG = np.array([(68.103004, 146.541400), (75.349014, 146.651690)])
INSITU_GRAY_RADIUS_SVG = (1.7948507, 1.7923003)
INSITU_RECT_CENTRES_SVG = np.array(
    [
        (67.6671825, 154.0371123),
        (74.6124863, 154.0371065),
        (69.1223905, 156.1207123),
        (75.6873666, 156.1537723),
    ]
)
INSITU_RECT_HALF_SVG = np.array(
    [
        (2.9765625, 0.8764323),
        (3.1088543, 0.8102865),
        (2.9765625, 0.8764323),
        (1.9678386, 0.8764323),
    ]
)


def _fib_svg_to_world(p):
    p = np.asarray(p, np.float64)
    return (p[..., 0] - FIB_SVG_ORIGIN[0]) * FIB_SVG_SCALE_UM, (
        p[..., 1] - FIB_SVG_ORIGIN[1]
    ) * FIB_SVG_SCALE_UM


def _insitu_svg_to_world(p):
    p = np.asarray(p, np.float64)
    return (
        (p[..., 0] - INSITU_SVG_ORIGIN[0]) * INSITU_SVG_SCALE_UM,
        (p[..., 1] - INSITU_SVG_ORIGIN[1]) * INSITU_SVG_SCALE_UM,
    )


def default_film_thickness_nm(film: str) -> float:
    return {"continuous": 12.0, "holey": 20.0, "lacey": 15.0, "vitreous_ice": 60.0, "auto": 20.0}.get(
        film, 0.0
    )


@dataclass
class PlacementArea:
    """A region a preparation is placed into: a grid hole, a lamella, a chip opening."""

    index: int
    center: tuple[float, float]
    half: tuple[float, float]
    bounds: AABB
    rotation: float = 0.0
    film_covered: bool = False
    film_broken: bool = False
    film_thickness_nm: float = 0.0
    post_index: int = -1
    feature_kind: int | None = None  # BulkSampleKind index for lamellae

    @property
    def half_span(self) -> float:
        return max(abs(self.half[0]), abs(self.half[1]))


@dataclass
class FibPost:
    index: int
    base: tuple[float, float]
    width_um: float
    height_um: float
    lamella_count: int
    first_lamella_area: int
    bulk_kind: int
    matrix_material: int


@dataclass
class Lamella:
    index: int
    post_index: int
    area_index: int
    center: tuple[float, float]
    half: tuple[float, float]
    thickness_nm: float
    matrix_material: int
    bulk_kind: int


@dataclass
class MeshParams:
    disk_radius_um: float = 1525.0
    pitch_um: float = 125.0
    bar_width_um: float = 35.0
    bar_thickness_nm: float = 20000.0
    rotation_rad: float = 0.0
    offset_um: tuple[float, float] = (0.0, 0.0)
    rim_width_um: float = 120.0
    hole_corner_radius_frac: float = 0.18
    damaged_square_count: int = 4
    bar_material: int = MaterialId.COPPER


def waffle_defaults() -> MeshParams:
    return MeshParams(
        pitch_um=62.5,
        bar_width_um=32.0,
        bar_thickness_nm=25000.0,
        hole_corner_radius_frac=0.02,
        rim_width_um=100.0,
        damaged_square_count=1,
    )


@dataclass
class FilmParams:
    type: str = "holey"
    material: int = MaterialId.AMORPHOUS_CARBON
    thickness_nm: float = 20.0
    hole_diameter_um: float = 2.0
    hole_pitch_um: float = 4.0
    lacey_mean_hole_um: float = 1.6
    lacey_open_fraction: float = 0.55
    broken_fraction: float = 0.08
    granularity_nm: float = 3.0
    granularity_correlation_nm: float = 2.0


class Holder:
    """A generated holder: analytic bulk + film samplers, placement areas and a few primitives."""

    def __init__(self, kind: str, seed: int, film: str, options: SpecimenOptions):
        self.kind = kind
        self.seed = int(seed)
        self.opts = options
        self.mesh = MeshParams()
        self.film = FilmParams()
        self.areas: list[PlacementArea] = []
        self.posts: list[FibPost] = []
        self.lamellae: list[Lamella] = []
        self.primitives = PrimitiveSet(0)
        self._lacey = None  # (cx, cy, r, area) arrays
        if kind in ("mesh_grid", "waffle_grid"):
            if kind == "waffle_grid":
                self.mesh = waffle_defaults()
            self._generate_mesh(film)
        elif kind == "fib_liftout":
            self._generate_fib(film)
        elif kind == "insitu_heating_chip":
            self._generate_chip(film)
        else:
            raise ValueError(f"unknown holder {kind!r}")
        self._area_arrays()

    # ------------------------------------------------------------------------------------------
    def _area_arrays(self):
        a = self.areas
        self.area_bounds = np.array([ar.bounds.as_tuple() for ar in a], np.float64).reshape(-1, 4)
        self.area_cx = np.array([ar.center[0] for ar in a], np.float64)
        self.area_cy = np.array([ar.center[1] for ar in a], np.float64)
        self.area_hx = np.array([ar.half[0] for ar in a], np.float64)
        self.area_hy = np.array([ar.half[1] for ar in a], np.float64)
        self.area_film_ok = np.array([ar.film_covered and not ar.film_broken for ar in a], bool)
        self.area_film_nm = np.array([ar.film_thickness_nm for ar in a], np.float64)

    def query_areas(self, box: AABB) -> np.ndarray:
        b = self.area_bounds
        if b.shape[0] == 0:
            return np.zeros(0, np.int64)
        m = (b[:, 0] <= box.xmax) & (b[:, 2] >= box.xmin) & (b[:, 1] <= box.ymax) & (b[:, 3] >= box.ymin)
        return np.flatnonzero(m)

    def query_primitives(self, box: AABB) -> PrimitiveSet:
        if self.primitives.n == 0:
            return self.primitives
        return self.primitives.take(np.flatnonzero(self.primitives.intersects(box)))

    # -- mesh / waffle --------------------------------------------------------------------------
    def _generate_mesh(self, film: str):
        m = self.mesh
        rng = rng_for(self.seed, SeedKind.HOLDER, 0)
        m.rotation_rad = rng.uniform(-MESH_ROTATION_RAD, MESH_ROTATION_RAD)
        m.pitch_um *= rng.uniform(1 - MESH_JITTER_FRAC, 1 + MESH_JITTER_FRAC)
        m.bar_width_um *= rng.uniform(1 - MESH_JITTER_FRAC, 1 + MESH_JITTER_FRAC)
        m.offset_um = (
            rng.uniform(-0.5 * m.pitch_um, 0.5 * m.pitch_um),
            rng.uniform(-0.5 * m.pitch_um, 0.5 * m.pitch_um),
        )
        if self.kind == "waffle_grid":
            m.damaged_square_count = int(rng.integers(WAFFLE_DAMAGED_MIN, WAFFLE_DAMAGED_MAX + 1))
        else:
            m.damaged_square_count = int(rng.integers(DAMAGED_MIN, DAMAGED_MAX + 1))

        f = self.film
        f.type = film
        f.material = MaterialId.VITREOUS_ICE if film == "vitreous_ice" else MaterialId.AMORPHOUS_CARBON
        ov = self.opts.film_thickness_nm
        f.thickness_nm = min(max(ov, 0.0), 1000.0) if ov > 0 else default_film_thickness_nm(film)
        if film == "vitreous_ice" and not ov > 0 and self.opts.ice_thickness_nm > 0:
            f.thickness_nm = min(self.opts.ice_thickness_nm, 1000.0)
        if film == "none":
            f.thickness_nm = 0.0

        self._cos, self._sin = math.cos(m.rotation_rad), math.sin(m.rotation_rad)
        self._half = 0.5 * (m.pitch_um - m.bar_width_um)
        self._corner = min(max(m.hole_corner_radius_frac, 0.0), 0.5) * self._half
        usable = max(0.0, m.disk_radius_um - m.rim_width_um)
        self._disk2 = m.disk_radius_um**2
        self._usable2 = usable**2
        self._noise_scale = (
            NM_PER_UM / f.granularity_correlation_nm if f.granularity_correlation_nm > 0 else 0.0
        )

        span = math.ceil(m.disk_radius_um / m.pitch_um) + 2
        self._cell_origin = -span
        self._cell_count = 2 * span + 1
        self._cell_area = np.full((self._cell_count, self._cell_count), -1, np.int32)
        center_r = usable + self._half
        rot_half = self._half * (abs(self._cos) + abs(self._sin))
        jj, ii = np.meshgrid(np.arange(-span, span + 1), np.arange(-span, span + 1), indexing="ij")
        lx = ii * m.pitch_um + m.offset_um[0]
        ly = jj * m.pitch_um + m.offset_um[1]
        wx = lx * self._cos - ly * self._sin
        wy = lx * self._sin + ly * self._cos
        keep = (wx * wx + wy * wy) <= center_r * center_r  # row-major (j outer, i inner) = C++ order
        idx = np.flatnonzero(keep.ravel())
        covered = film != "none"
        u_broken = uniform_from_hash(
            hash_seed(np.uint64(self.seed), SeedKind.SUPPORT_FILM, np.arange(idx.size, dtype=np.int64))
        )
        for k, flat in enumerate(idx):
            j, i = divmod(int(flat), self._cell_count)
            cx, cy = float(wx.flat[flat]), float(wy.flat[flat])
            broken = covered and bool(u_broken[k] < f.broken_fraction)
            self.areas.append(
                PlacementArea(
                    index=k,
                    center=(cx, cy),
                    half=(self._half, self._half),
                    bounds=AABB.from_center_half(cx, cy, rot_half, rot_half).pad(self._corner),
                    film_covered=covered,
                    film_broken=broken,
                    film_thickness_nm=f.thickness_nm if covered and not broken else 0.0,
                )
            )
            self._cell_area[j, i] = k

        # torn squares: a bent bar fragment hanging into the hole, film of that square gone
        n_damaged = min(m.damaged_square_count, len(self.areas))
        damaged: list[int] = []
        prim = []
        for _ in range(n_damaged):
            pick = -1
            for _attempt in range(8):
                cand = int(rng.integers(0, len(self.areas)))
                if cand not in damaged:
                    pick = cand
                    break
            if pick < 0:
                continue
            damaged.append(pick)
            area = self.areas[pick]
            area.film_broken = True
            area.film_thickness_nm = 0.0
            edge = int(rng.integers(0, 4))
            toward, along = {
                0: ((0.0, -self._half), 0.0),
                1: ((0.0, self._half), 0.0),
                2: ((-self._half, 0.0), 0.5 * math.pi),
                3: ((self._half, 0.0), 0.5 * math.pi),
            }[edge]
            reach = rng.uniform(0.55, 0.95)
            tx, ty = toward[0] * reach, toward[1] * reach
            cx = area.center[0] + tx * self._cos - ty * self._sin
            cy = area.center[1] + tx * self._sin + ty * self._cos
            rot = m.rotation_rad + along + rng.uniform(-0.35, 0.35)
            hw, hh = self._half * TORN_BAR_LENGTH_FRAC, self._half * TORN_BAR_WIDTH_FRAC
            prim.append((cx, cy, hw, hh, rot, int(hash_seed(self.seed, SeedKind.HOLDER, pick, 1))))
        if prim:
            a = np.array(prim, dtype=object)
            self.primitives = PrimitiveSet.make(
                a[:, 0].astype(float),
                a[:, 1].astype(float),
                a[:, 2].astype(float),
                a[:, 3].astype(float),
                rot=a[:, 4].astype(float),
                shape=Shape.RECT,
                thickness=m.bar_thickness_nm,
                profile=Profile.FLAT,
                layer=Layer.HOLDER_BULK,
                material=m.bar_material,
                seed=np.array([p[5] for p in prim], np.uint64),
            )
        # The rim (a Ring primitive in the C++) is not emitted: the analytic bulk pass already
        # paints it, and painting both doubled the rim thickness.

        if film == "lacey" and self.areas:
            target = min(max(f.lacey_open_fraction, 0.0), 1.0) * 4.0 * self._half * self._half
            cxs, cys, rs, ais = [], [], [], []
            for area in self.areas:
                if not area.film_covered or area.film_broken:
                    continue
                r = rng_for(self.seed, SeedKind.SUPPORT_FILM, area.index, 1)
                d = np.clip(
                    f.lacey_mean_hole_um * np.exp(LACEY_SIGMA_LOG * r.standard_normal(LACEY_HOLES_PER_AREA)),
                    LACEY_MIN_UM,
                    LACEY_MAX_UM,
                )
                rad = 0.5 * d
                hx = area.center[0] + r.uniform(-area.half[0], area.half[0], LACEY_HOLES_PER_AREA)
                hy = area.center[1] + r.uniform(-area.half[1], area.half[1], LACEY_HOLES_PER_AREA)
                cum = np.cumsum(np.pi * rad * rad)
                nkeep = int(min(LACEY_HOLES_PER_AREA, np.searchsorted(cum, target, side="left") + 1))
                cxs.append(hx[:nkeep])
                cys.append(hy[:nkeep])
                rs.append(rad[:nkeep])
                ais.append(np.full(nkeep, area.index, np.int32))
            if cxs:
                self._lacey = tuple(np.concatenate(v) for v in (cxs, cys, rs, ais))
        self.bounds = AABB.from_center_radius(0.0, 0.0, m.disk_radius_um)

    # -- FIB liftout ----------------------------------------------------------------------------
    def _generate_fib(self, film: str):
        o = self.opts
        self.film = FilmParams(type="none", thickness_nm=0.0, granularity_nm=0.0)
        self._fib_disk_r = FIB_SVG_DISK_RADIUS * FIB_SVG_SCALE_UM
        rim = FIB_SVG_RIM_HALF_WIDTH * FIB_SVG_SCALE_UM
        post_count = int(min(max(o.fib_post_count, 3), 5))
        nominal = min(max(o.lamella_thickness_nm, 1.0), 5000.0) if o.lamella_thickness_nm > 0 else 110.0
        dcx, dcy = _fib_svg_to_world(FIB_SVG_DISK_CENTRE)
        self._fib_centre = (float(dcx), float(dcy))
        self._disk2 = self._fib_disk_r**2
        self._usable2 = max(0.0, self._fib_disk_r - rim) ** 2
        bx, by = _fib_svg_to_world(FIB_BODY_SVG)
        self._fib_body = (bx, by)
        self._fib_body_box = AABB(bx.min(), by.min(), bx.max(), by.max())
        hx, hy = _fib_svg_to_world(FIB_HOLE_CENTRES_SVG)
        self._fib_holes = (
            hx,
            hy,
            FIB_HOLE_RADIUS_SVG[0] * FIB_SVG_SCALE_UM,
            FIB_HOLE_RADIUS_SVG[1] * FIB_SVG_SCALE_UM,
        )
        self.mesh.bar_material = MaterialId.COPPER
        self.mesh.bar_thickness_nm = 20000.0

        perm = rng_for(self.seed, SeedKind.FIB_POST, 0xFFFF).permutation(5)
        forced = _POST_SAMPLE_TO_KIND.get(o.fib_post_sample)
        lam_half = (0.168 * FIB_SVG_SCALE_UM, 0.104 * FIB_SVG_SCALE_UM)
        rot = FIB_FLAG_ROTATION_RAD
        fu = (math.cos(rot), math.sin(rot))
        fv = (-fu[1], fu[0])
        support_half = 0.290 * FIB_SVG_SCALE_UM
        upper_c, upper_hw = 0.1605 * FIB_SVG_SCALE_UM, 0.0565 * FIB_SVG_SCALE_UM
        tab_c, tab_hl = 0.229 * FIB_SVG_SCALE_UM, 0.061 * FIB_SVG_SCALE_UM
        dark_c, dark_hw = -0.128 * FIB_SVG_SCALE_UM, 0.024 * FIB_SVG_SCALE_UM
        rects = []  # cx, cy, hx, hy, thickness, profile, layer, material, seed
        lx, ly = _fib_svg_to_world(FIB_LAMELLA_CENTRES_SVG)
        for k in range(post_count):
            prng = rng_for(self.seed, SeedKind.FIB_POST, k)
            kind = forced if forced is not None else int(perm[k % 5])
            if kind == 0:
                mat = MaterialId.IRON
            elif kind == 3:
                mat = MaterialId.ALUMINUM if prng.uniform() < 0.5 else MaterialId.GOLD
            else:
                mat = MaterialId.SILICON
            cx, cy = float(lx[k]), float(ly[k])
            self.posts.append(
                FibPost(k, (cx, cy), 0.58 * FIB_SVG_SCALE_UM, 0.37 * FIB_SVG_SCALE_UM, 1, k, kind, int(mat))
            )
            thick = nominal * rng_for(self.seed, SeedKind.LAMELLA, k).uniform(
                1 - LAMELLA_THICKNESS_JITTER, 1 + LAMELLA_THICKNESS_JITTER
            )
            lam_seed = hash_seed(self.seed, SeedKind.LAMELLA, k)
            bh = rotated_rect_half_extent(lam_half[0], lam_half[1], rot)
            self.areas.append(
                PlacementArea(
                    index=k,
                    center=(cx, cy),
                    half=lam_half,
                    bounds=AABB.from_center_half(cx, cy, *bh),
                    rotation=rot,
                    post_index=k,
                    feature_kind=kind,
                )
            )
            self.lamellae.append(Lamella(k, k, k, (cx, cy), lam_half, float(thick), int(mat), kind))
            cu, pt = MaterialId.COPPER, MaterialId.PLATINUM
            rects += [
                (
                    cx + fv[0] * upper_c,
                    cy + fv[1] * upper_c,
                    support_half,
                    upper_hw,
                    400.0,
                    Profile.FLAT,
                    Layer.HOLDER_BULK,
                    cu,
                    hash_seed(lam_seed, SeedKind.LAMELLA, 10),
                ),
                (
                    cx - fu[0] * tab_c,
                    cy - fu[1] * tab_c,
                    tab_hl,
                    lam_half[1],
                    400.0,
                    Profile.FLAT,
                    Layer.HOLDER_BULK,
                    cu,
                    hash_seed(lam_seed, SeedKind.LAMELLA, 11),
                ),
                (
                    cx + fu[0] * tab_c,
                    cy + fu[1] * tab_c,
                    tab_hl,
                    lam_half[1],
                    400.0,
                    Profile.FLAT,
                    Layer.HOLDER_BULK,
                    cu,
                    hash_seed(lam_seed, SeedKind.LAMELLA, 12),
                ),
                (
                    cx + fv[0] * dark_c,
                    cy + fv[1] * dark_c,
                    support_half,
                    dark_hw,
                    1200.0,
                    Profile.FLAT,
                    Layer.HOLDER_BULK,
                    pt,
                    hash_seed(lam_seed, SeedKind.LAMELLA, 13),
                ),
                (
                    cx,
                    cy,
                    lam_half[0],
                    lam_half[1],
                    float(thick),
                    Profile.CURTAIN,
                    Layer.PREPARATION,
                    int(mat),
                    lam_seed,
                ),
            ]
        column = list(zip(*rects))
        self.primitives = PrimitiveSet.make(
            np.array(column[0]),
            np.array(column[1]),
            np.array(column[2]),
            np.array(column[3]),
            rot=rot,
            shape=Shape.RECT,
            thickness=np.array(column[4]),
            profile=np.array(column[5]),
            layer=np.array(column[6]),
            material=np.array(column[7]),
            seed=np.array(column[8], dtype=np.uint64),
        )
        self.bounds = AABB.from_center_radius(dcx, dcy, self._fib_disk_r)

    # -- in-situ heating chip ---------------------------------------------------------------------
    def _generate_chip(self, film: str):
        self.film = FilmParams(
            type="none",
            material=MaterialId.SILICON_NITRIDE,
            thickness_nm=0.0,
            granularity_nm=CHIP_FILM_GRANULARITY_NM,
            broken_fraction=0.0,
        )
        self._chip_membrane_nm = 50.0
        self._chip_trace_nm = 100.0
        hx, hy = _insitu_svg_to_world(INSITU_HEATER_SVG)
        self._chip_heater = (hx, hy)
        self._chip_heater_box = AABB(hx.min(), hy.min(), hx.max(), hy.max())
        self._chip_membrane_box = self._chip_heater_box.pad(4.0 * INSITU_SVG_SCALE_UM)
        s = INSITU_SVG_SCALE_UM
        ops = []  # (cx, cy, rx, ry, ellipse, corner, gcx, gcy, grx, gry)
        ex, ey = _insitu_svg_to_world(INSITU_ELLIPSE_CENTRES_SVG)
        gx, gy = _insitu_svg_to_world(INSITU_GRAY_CENTRES_SVG)
        for i in range(6):
            if i >= 4:
                g = (
                    float(gx[i - 4]),
                    float(gy[i - 4]),
                    INSITU_GRAY_RADIUS_SVG[0] * s,
                    INSITU_GRAY_RADIUS_SVG[1] * s,
                )
            else:
                g = (float(ex[i]), float(ey[i]), 0.0, 0.0)
            ops.append(
                (
                    float(ex[i]),
                    float(ey[i]),
                    INSITU_ELLIPSE_RADIUS_SVG[0] * s,
                    INSITU_ELLIPSE_RADIUS_SVG[1] * s,
                    True,
                    0.0,
                )
                + g
            )
        rx_, ry_ = _insitu_svg_to_world(INSITU_RECT_CENTRES_SVG)
        for i in range(4):
            hxr, hyr = INSITU_RECT_HALF_SVG[i] * s
            ops.append(
                (
                    float(rx_[i]),
                    float(ry_[i]),
                    float(hxr),
                    float(hyr),
                    False,
                    float(hyr),
                    float(rx_[i]),
                    float(ry_[i]),
                    0.0,
                    0.0,
                )
            )
        self._chip_openings = ops
        for k, (cx, cy, rx, ry, ell, corner, *_g) in enumerate(ops):
            half = (0.7 * rx, 0.7 * ry) if ell else (max(0.1, rx - 0.30 * corner), max(0.1, ry * 0.70))
            self.areas.append(
                PlacementArea(
                    index=k, center=(cx, cy), half=half, bounds=AABB.from_center_half(cx, cy, rx, ry)
                )
            )
        self.mesh.bar_material = MaterialId.SILICON
        self.bounds = self._chip_membrane_box

    # ------------------------------------------------------------------------------------------
    # Per-pixel sampling
    # ------------------------------------------------------------------------------------------
    def paint(self, context) -> None:
        """Holder bulk pass then support-film pass (FieldMap::Rasterize steps 1 and 2)."""
        nx = context.nx
        if not self.bounds.intersects(context.aabb):
            return
        film_on = self.film.type != "none" and bool(self.areas) and bool(self.area_film_ok.any())
        lacey = film_on and self.film.type == "lacey"
        cols = np.arange(nx, dtype=np.float64)[None, :]
        for r0, r1 in context.row_blocks(0, context.ny, nx):
            rows = np.arange(r0, r1, dtype=np.float64)[:, None]
            mat, thick, area = self._sample_bulk(context, rows, cols, r0, r1)
            sl = slice(r0 * nx, r1 * nx)
            context.material[sl] = mat.ravel()
            context.thick[sl] = thick.ravel()
            context.area[sl] = area.ravel()
            if film_on and not lacey:
                self._film_block(context, rows, cols, r0, r1, None)
        if lacey:
            removed = self._lacey_mask(context)
            for r0, r1 in context.row_blocks(0, context.ny, nx):
                rows = np.arange(r0, r1, dtype=np.float64)[:, None]
                self._film_block(context, rows, cols, r0, r1, removed[r0 * nx : r1 * nx])

    def _lattice(self, context, rows, cols):
        """Mesh lattice coordinates (rotate by -rotation, remove offset) as affine outer sums."""
        c, s = self._cos, self._sin
        ox, oy = self.mesh.offset_um
        lx = context.linear(c, s, -ox, rows, cols)
        ly = context.linear(-s, c, -oy, rows, cols)
        return lx, ly

    def _mesh_block_single_hole(self, context, r0, r1):
        """Area index when every pixel centre of the block lies in one hole's safe square."""
        rr = np.array([r0, r0, r1 - 1, r1 - 1], np.float64)
        cc = np.array([0, context.nx - 1, 0, context.nx - 1], np.float64)
        X, Y = context.world(rr, cc)
        if np.any(X * X + Y * Y > self._usable2):
            return None
        m = self.mesh
        lx = X * self._cos + Y * self._sin - m.offset_um[0]
        ly = -X * self._sin + Y * self._cos - m.offset_um[1]
        qx = np.floor(lx / m.pitch_um + 0.5)
        qy = np.floor(ly / m.pitch_um + 0.5)
        if np.ptp(qx) != 0 or np.ptp(qy) != 0:
            return None
        safe = self._half - self._corner
        if np.any(np.abs(lx - qx * m.pitch_um) >= safe) or np.any(np.abs(ly - qy * m.pitch_um) >= safe):
            return None
        i = int(qx[0]) - self._cell_origin
        j = int(qy[0]) - self._cell_origin
        if not (0 <= i < self._cell_count and 0 <= j < self._cell_count):
            return -1
        return int(self._cell_area[j, i])

    def _sample_bulk(self, context, rows, cols, r0, r1):
        shape = (rows.shape[0], cols.shape[1])
        if self.kind in ("mesh_grid", "waffle_grid"):
            m = self.mesh
            k = self._mesh_block_single_hole(context, r0, r1)
            if k is not None:
                return (np.zeros(shape, np.uint8), np.zeros(shape, np.float32), np.full(shape, k, np.int32))
            if _rnb.AVAILABLE and shape[1] == context.nx:
                c, s = self._cos, self._sin
                ox, oy = m.offset_um
                LC = np.empty(6)
                for q, (p1, p2, kk) in enumerate(((c, s, -ox), (-s, c, -oy))):
                    LC[3 * q] = kk + p1 * context.ox + p2 * context.oy
                    LC[3 * q + 1] = p1 * context.axc + p2 * context.ayc
                    LC[3 * q + 2] = p1 * context.axr + p2 * context.ayr
                mat = np.empty(shape, np.uint8)
                thick = np.empty(shape, np.float32)
                area = np.empty(shape, np.int32)
                _rnb.mesh_bulk(
                    r0,
                    r1,
                    context.nx,
                    LC,
                    np.array([context.ox, context.oy, context.axc, context.ayc, context.axr, context.ayr]),
                    float(m.pitch_um),
                    float(self._half),
                    float(self._corner),
                    float(self._disk2),
                    float(self._usable2),
                    np.uint8(m.bar_material),
                    np.float32(m.bar_thickness_nm),
                    float(self._cell_origin),
                    float(self._cell_count),
                    np.ascontiguousarray(self._cell_area, np.int32),
                    mat,
                    thick,
                    area,
                )
                return mat, thick, area
            lx, ly = self._lattice(context, rows, cols)
            inv = 1.0 / m.pitch_um
            qx = np.floor(lx * inv + 0.5)
            qy = np.floor(ly * inv + 0.5)
            u = np.abs(lx - qx * m.pitch_um)
            v = np.abs(ly - qy * m.pitch_um)
            h = self._half
            hole = (u < h) & (v < h)
            if self._corner > 0:
                ic = h - self._corner
                du = np.maximum(u - ic, 0.0)
                dv = np.maximum(v - ic, 0.0)
                hole &= (du * du + dv * dv) < self._corner**2
            X, Y = context.world(rows, cols)
            r2 = X * X + Y * Y
            in_disk = r2 <= self._disk2
            hole &= r2 <= self._usable2
            bar = in_disk & ~hole
            mat = (bar * np.uint8(m.bar_material)).astype(np.uint8)
            thick = (bar * np.float32(m.bar_thickness_nm)).astype(np.float32)
            n = self._cell_count
            ci = qx - self._cell_origin
            cj = qy - self._cell_origin
            okc = hole & (ci >= 0) & (ci < n) & (cj >= 0) & (cj < n)
            ci = np.clip(ci, 0, n - 1).astype(np.int64)
            cj = np.clip(cj, 0, n - 1).astype(np.int64)
            area = np.where(okc, self._cell_area[cj, ci], -1).astype(np.int32)
            return mat, thick, area
        X, Y = context.world(rows, cols)
        X = np.broadcast_to(X, shape)
        Y = np.broadcast_to(Y, shape)
        mat = np.zeros(shape, np.uint8)
        thick = np.zeros(shape, np.float32)
        area = np.full(shape, -1, np.int32)
        rr = np.array([r0, r0, r1 - 1, r1 - 1], np.float64)
        cc = np.array([0, context.nx - 1, 0, context.nx - 1], np.float64)
        bx, by = context.world(rr, cc)
        bb = AABB(bx.min(), by.min(), bx.max(), by.max())  # world box of the block's pixel centres
        if self.kind == "fib_liftout":
            cx0, cy0 = self._fib_centre
            ddx = max(bb.xmin - cx0, 0.0, cx0 - bb.xmax)
            ddy = max(bb.ymin - cy0, 0.0, cy0 - bb.ymax)
            if ddx * ddx + ddy * ddy > self._disk2:
                return mat, thick, area  # entirely off the grid
            if np.max((bx - cx0) ** 2 + (by - cy0) ** 2) < self._usable2:
                in_disk = True
                opaque = np.zeros(shape, bool)
            else:
                r2 = (X - cx0) ** 2 + (Y - cy0) ** 2
                in_disk = r2 <= self._disk2
                opaque = in_disk & (r2 >= self._usable2)
            box = self._fib_body_box
            if box.intersects(bb):
                body = self._poly_mask(context, self._fib_body, r0, r1) & in_disk & box.contains(X, Y)
                hx, hy, hrx, hry = self._fib_holes
                near = np.flatnonzero(
                    (hx + hrx >= bb.xmin)
                    & (hx - hrx <= bb.xmax)
                    & (hy + hry >= bb.ymin)
                    & (hy - hry <= bb.ymax)
                )
                for k in near:
                    body &= ((X - hx[k]) / hrx) ** 2 + ((Y - hy[k]) / hry) ** 2 > 1.0
                opaque |= body
            mat[opaque] = MaterialId.COPPER
            thick[opaque] = 20000.0
            return mat, thick, area
        # in-situ chip
        mb = self._chip_membrane_box
        if not mb.intersects(bb):
            return mat, thick, area
        inside_all = mb.xmin <= bb.xmin and bb.xmax <= mb.xmax and mb.ymin <= bb.ymin and bb.ymax <= mb.ymax
        inm = np.ones(shape, bool) if inside_all else mb.contains(X, Y)
        mat[inm] = MaterialId.SILICON_NITRIDE
        thick[inm] = self._chip_membrane_nm
        if self._chip_heater_box.intersects(bb):
            on_heater = (
                inm
                & self._poly_mask(context, self._chip_heater, r0, r1)
                & self._chip_heater_box.contains(X, Y)
            )
            mat[on_heater] = MaterialId.PLATINUM
            thick[on_heater] = self._chip_trace_nm + self._chip_membrane_nm
        # first matching opening wins (SVG paint order); paint in reverse so earlier ones win
        for k in range(len(self._chip_openings) - 1, -1, -1):
            cx, cy, rx, ry, ell, corner, gcx, gcy, grx, gry = self._chip_openings[k]
            if not AABB.from_center_half(cx, cy, max(rx, grx), max(ry, gry)).intersects(bb):
                continue
            if grx > 0:
                column = inm & (((X - gcx) / grx) ** 2 + ((Y - gcy) / gry) ** 2 <= 1.0)
                mat[column] = MaterialId.SILICON_NITRIDE
                thick[column] = 2.0 * self._chip_membrane_nm
                area[column] = -1
            dx, dy = X - cx, Y - cy
            if ell:
                ins = (dx / rx) ** 2 + (dy / ry) ** 2 <= 1.0
            else:
                ax, ay = np.abs(dx), np.abs(dy)
                ins = (
                    (ax <= rx)
                    & (ay <= ry)
                    & (
                        (ax <= rx - corner)
                        | (ay <= ry - corner)
                        | ((ax - (rx - corner)) ** 2 + (ay - (ry - corner)) ** 2 <= corner * corner)
                    )
                )
            ins &= inm
            mat[ins] = 0
            thick[ins] = 0.0
            area[ins] = k
        return mat, thick, area

    def _poly_mask(self, context, poly, r0, r1):
        r, c = context.to_pixel(poly[0], poly[1])
        return polygon_mask_rows(c, r, r0, r1, context.nx)

    def _lacey_mask(self, context) -> np.ndarray:
        removed = np.zeros(context.npix, bool)
        if self._lacey is None:
            return removed
        cx, cy, rad, ai = self._lacey
        box = context.aabb
        vis = np.flatnonzero(
            (cx + rad >= box.xmin) & (cx - rad <= box.xmax) & (cy + rad >= box.ymin) & (cy - rad <= box.ymax)
        )
        if vis.size == 0:
            return removed
        cx, cy, rad, ai = cx[vis], cy[vis], rad[vis], ai[vis]
        for sub, rows, cols, valid in iter_patches(context, cx - rad, cy - rad, cx + rad, cy + rad):
            X, Y = context.world(rows, cols)
            d2 = (X - cx[sub][:, None, None]) ** 2 + (Y - cy[sub][:, None, None]) ** 2
            flat = np.broadcast_to(rows * context.nx + cols, d2.shape)
            m = valid & (d2 < (rad[sub] ** 2)[:, None, None])
            m &= context.area[np.where(m, flat, 0)] == ai[sub][:, None, None]
            removed[flat[m]] = True
        return removed

    def _film_block_fast(self, context, rows, cols, r0, r1, removed, amin, amax) -> None:
        """`_film_block` through :func:`..raster_nb.film_block` (the same arithmetic)."""
        f = self.film
        nx = context.nx
        same = -1
        if amin == amax and removed is None:
            if not self.area_film_ok[amin]:
                return
            same = amin
        holey = False
        LC = np.zeros(6)
        if f.type == "holey" and f.hole_pitch_um > 0:
            state = self._holey_block_state(context, r0, r1)
            if state == "perforated":
                return
            if state != "film":
                holey = True
                c, s = self._cos, self._sin
                ox, oy = self.mesh.offset_um
                for k, (p, q, kk) in enumerate(((c, s, -ox), (-s, c, -oy))):
                    LC[3 * k] = kk + p * context.ox + q * context.oy
                    LC[3 * k + 1] = p * context.axc + q * context.ayc
                    LC[3 * k + 2] = p * context.axr + q * context.ayr
        noise_mode = 0
        lodg = 0.0
        G = np.zeros((2, 2))
        gix = giy = np.zeros(1, np.int64)
        gtx = gty = np.zeros(1)
        narr = np.zeros(1)
        if f.granularity_nm > 0 and self._noise_scale > 0:
            cpp = context.step_um * self._noise_scale
            lod = 1.0 if cpp <= 0.5 else (0.0 if cpp >= 1.0 else 2.0 * (1.0 - cpp))
            if lod > 0:
                sc = self._noise_scale
                lodg = lod * f.granularity_nm
                sep = None
                if context.axis_aligned:
                    xs = (context.ox + context.axc * cols[0]) * sc
                    ys = (context.oy + context.ayr * rows[:, 0]) * sc
                    sep = _separable_noise_tables(self.seed, xs, ys)
                if sep is not None:
                    noise_mode = 1
                    G, gix, gtx, giy, gty = sep
                else:
                    noise_mode = 2
                    if context.axis_aligned:
                        narr = value_noise_separable(self.seed, xs, ys).ravel()
                    else:
                        X, Y = context.world(rows, cols)
                        shape = (rows.shape[0], cols.shape[1])
                        narr = value_noise(
                            self.seed, np.broadcast_to(X, shape) * sc, np.broadcast_to(Y, shape) * sc
                        ).ravel()
        W = np.array([context.ox, context.oy, context.axc, context.ayc, context.axr, context.ayr])
        rem = np.ascontiguousarray(removed) if removed is not None else np.zeros(1, bool)
        _rnb.film_block(
            context.thick,
            context.material,
            context.area,
            nx,
            r0,
            r1,
            same,
            self.area_film_ok,
            self.area_film_nm,
            rem,
            removed is not None,
            holey,
            LC,
            float(f.hole_pitch_um),
            (0.5 * f.hole_diameter_um) ** 2,
            f.type == "vitreous_ice",
            self.area_hx,
            self.area_hy,
            self.area_cx,
            self.area_cy,
            W,
            ICE_EDGE_GRADIENT,
            noise_mode,
            lodg,
            np.ascontiguousarray(G, np.float64),
            gix,
            gtx,
            giy,
            gty,
            narr,
            np.uint8(f.material),
        )

    def _holey_block_state(self, context, r0, r1) -> str:
        """'film' if no perforation touches the block, 'perforated' if one covers it, else 'mixed'."""
        f = self.film
        rr = np.array([r0, r0, r1 - 1, r1 - 1], np.float64)
        cc = np.array([0, context.nx - 1, 0, context.nx - 1], np.float64)
        X, Y = context.world(rr, cc)
        m = self.mesh
        lx = X * self._cos + Y * self._sin - m.offset_um[0]
        ly = -X * self._sin + Y * self._cos - m.offset_um[1]
        p, r = f.hole_pitch_um, 0.5 * f.hole_diameter_um
        kx = np.arange(math.floor(lx.min() / p) - 1, math.ceil(lx.max() / p) + 2)
        ky = np.arange(math.floor(ly.min() / p) - 1, math.ceil(ly.max() / p) + 2)
        if kx.size * ky.size > 64:
            return "mixed"
        hx = kx[None, :] * p
        hy = ky[:, None] * p
        # distance from each perforation centre to the block's lattice-space bounding box
        dx = np.maximum(np.maximum(lx.min() - hx, hx - lx.max()), 0.0)
        dy = np.maximum(np.maximum(ly.min() - hy, hy - ly.max()), 0.0)
        if np.all(dx * dx + dy * dy >= r * r):
            return "film"
        # one perforation containing all four corners (disk is convex -> whole block)
        d2 = (lx[None, None, :] - hx[..., None]) ** 2 + (ly[None, None, :] - hy[..., None]) ** 2
        if np.any(np.all(d2 < r * r, axis=-1)):
            return "perforated"
        return "mixed"

    def _film_block(self, context, rows, cols, r0, r1, removed) -> None:
        """SampleFilmAt over one row block: film thickness ADDED, material only claims vacuum."""
        nx = context.nx
        sl = slice(r0 * nx, r1 * nx)
        a = context.area[sl]
        f = self.film
        R, W = rows.shape[0], cols.shape[1]
        amin, amax = int(a.min()), int(a.max())
        if amax < 0:
            return
        if _rnb.AVAILABLE:
            self._film_block_fast(context, rows, cols, r0, r1, removed, amin, amax)
            return
        if amin == amax and removed is None:
            # the whole block is one placement area (the usual high-magnification case)
            if not self.area_film_ok[amin]:
                return
            ai = np.full(a.shape, amin, np.int32)
            t = np.full((R, W), self.area_film_nm[amin])
            keep = np.ones((R, W), bool)
        else:
            ok = a >= 0
            ai = np.where(ok, a, 0)
            ok &= self.area_film_ok[ai]
            if removed is not None:
                ok &= ~removed
            if not ok.any():
                return
            t = self.area_film_nm[ai].reshape(R, W)
            keep = ok.reshape(R, W)
        if f.type == "holey" and f.hole_pitch_um > 0:
            state = self._holey_block_state(context, r0, r1)
            if state == "perforated":
                return
            if state != "film":
                lx, ly = self._lattice(context, rows, cols)
                p = f.hole_pitch_um
                hu = lx - np.floor(lx / p + 0.5) * p
                hv = ly - np.floor(ly / p + 0.5) * p
                keep = keep & ((hu * hu + hv * hv) >= (0.5 * f.hole_diameter_um) ** 2)
        if f.type == "vitreous_ice":
            X, Y = context.world(rows, cols)
            aa = ai.reshape(R, W)
            hx = np.where(self.area_hx[aa] > 0, self.area_hx[aa], 1.0)
            hy = np.where(self.area_hy[aa] > 0, self.area_hy[aa], 1.0)
            d = np.clip(np.hypot((X - self.area_cx[aa]) / hx, (Y - self.area_cy[aa]) / hy), 0.0, 1.0)
            t = t * (1.0 + ICE_EDGE_GRADIENT * d * d)
        if f.granularity_nm > 0 and self._noise_scale > 0:
            cpp = context.step_um * self._noise_scale
            lod = 1.0 if cpp <= 0.5 else (0.0 if cpp >= 1.0 else 2.0 * (1.0 - cpp))
            if lod > 0:
                sc = self._noise_scale
                if context.axis_aligned:
                    xs = (context.ox + context.axc * cols[0]) * sc
                    ys = (context.oy + context.ayr * rows[:, 0]) * sc
                    n = value_noise_separable(self.seed, xs, ys)
                else:
                    X, Y = context.world(rows, cols)
                    n = value_noise(
                        self.seed, np.broadcast_to(X, (R, W)) * sc, np.broadcast_to(Y, (R, W)) * sc
                    )
                t = t + lod * f.granularity_nm * (n - 0.5)
        t = np.maximum(t, 0.0)
        if keep.all():
            context.thick[sl] = np.clip(context.thick[sl] + t.ravel(), 0.0, 65535.0)
            m = context.material[sl]
            context.material[sl] = np.where(m == 0, np.uint8(f.material), m)
            return
        kf = np.flatnonzero(keep.ravel())
        flat = kf + r0 * nx
        context.add_thickness(flat, t.ravel()[kf])
        vac = context.material[flat] == 0
        context.material[flat[vac]] = f.material
