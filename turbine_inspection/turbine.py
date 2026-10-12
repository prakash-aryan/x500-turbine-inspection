"""Procedural horizontal-axis wind turbine.

Dimensions follow the public NREL 5 MW reference turbine (Jonkman et al., 2009): 90 m hub height,
126 m rotor diameter, 61.5 m blades, tower base diameter 6 m tapering to 3.87 m. Geometry is generated
here as triangle meshes, so the model carries no third-party licence.

World frame: metres, Z up. The rotor faces the wind coming from +X, so the rotor plane is the Y-Z plane
at x = hub_x and the blade pressure side faces +X.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TurbineSpec:
    tower_height: float = 87.6
    tower_base_radius: float = 3.0
    tower_top_radius: float = 1.935
    hub_height: float = 90.0
    overhang: float = 5.0                 # hub centre ahead of the tower axis (+X)
    hub_radius: float = 1.5
    blade_length: float = 61.5
    blade_azimuths_deg: tuple = (180.0, 60.0, 300.0)   # Y pose: blade 0 points straight down
    nacelle_size: tuple = (11.0, 4.2, 4.0)             # length (x), width (y), height (z)
    position: tuple = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0                  # yaw of the whole turbine about Z (0: rotor faces +X)
    rotor_azimuth_deg: float = 0.0        # extra rotation of the whole rotor (used for distant turbines)

    @property
    def rotor_radius(self) -> float:
        return self.hub_radius + self.blade_length

    def hub_center(self) -> np.ndarray:
        return self.to_world(np.array([self.overhang, 0.0, self.hub_height]))

    def to_world(self, p: np.ndarray) -> np.ndarray:
        """Turbine-local point(s) -> world (applies yaw and position)."""
        c, s = np.cos(np.radians(self.yaw_deg)), np.sin(np.radians(self.yaw_deg))
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return np.asarray(p) @ R.T + np.asarray(self.position)

    def dir_to_world(self, d: np.ndarray) -> np.ndarray:
        c, s = np.cos(np.radians(self.yaw_deg)), np.sin(np.radians(self.yaw_deg))
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return np.asarray(d) @ R.T


@dataclass
class Mesh:
    points: np.ndarray                    # (N, 3) float
    faces: np.ndarray                     # (M, 3) int, counter-clockwise seen from outside
    uv: np.ndarray = None                 # (N, 2) per-vertex
    name: str = "mesh"
    meta: dict = field(default_factory=dict)

    def normals(self) -> np.ndarray:
        p, f = self.points, self.faces
        fn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
        vn = np.zeros_like(p)
        for k in range(3):
            np.add.at(vn, f[:, k], fn)
        return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def grid_faces(n_rows: int, n_cols: int, wrap: bool) -> np.ndarray:
    """Quad grid (rows along a sweep, cols around a section) split into triangles."""
    cols = n_cols if wrap else n_cols - 1
    f = []
    for i in range(n_rows - 1):
        for j in range(cols):
            a = i * n_cols + j
            b = i * n_cols + (j + 1) % n_cols
            c = (i + 1) * n_cols + (j + 1) % n_cols
            d = (i + 1) * n_cols + j
            f.append((a, b, c))
            f.append((a, c, d))
    return np.array(f, dtype=np.int64)


def cap_faces(center_index: int, ring_start: int, n: int, flip: bool) -> np.ndarray:
    f = []
    for j in range(n):
        a, b = ring_start + j, ring_start + (j + 1) % n
        f.append((center_index, b, a) if not flip else (center_index, a, b))
    return np.array(f, dtype=np.int64)


def signed_volume(m: Mesh) -> float:
    p, f = m.points - m.points.mean(axis=0), m.faces
    return float(np.sum(np.einsum("ij,ij->i", p[f[:, 0]], np.cross(p[f[:, 1]], p[f[:, 2]]))) / 6.0)


def orient_outward(m: Mesh) -> None:
    """Flip the winding of a closed mesh if its signed volume is negative (normals pointing inward)."""
    if signed_volume(m) < 0:
        m.faces = m.faces[:, ::-1].copy()


# ---- tower ---------------------------------------------------------------------------------------------------------

def tower_radius(spec: TurbineSpec, z: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(z) / spec.tower_height, 0.0, 1.0)
    return spec.tower_base_radius + (spec.tower_top_radius - spec.tower_base_radius) * t


def tower_mesh(spec: TurbineSpec, n_around: int = 64, n_up: int = 40) -> Mesh:
    z = np.linspace(0.0, spec.tower_height, n_up)
    th = np.linspace(0.0, 2 * np.pi, n_around, endpoint=False)
    r = tower_radius(spec, z)
    pts = np.stack([r[:, None] * np.cos(th)[None], r[:, None] * np.sin(th)[None],
                    np.repeat(z[:, None], n_around, 1)], -1).reshape(-1, 3)
    faces = grid_faces(n_up, n_around, wrap=True)
    top = len(pts)
    pts = np.vstack([pts, [[0.0, 0.0, spec.tower_height]], [[0.0, 0.0, 0.0]]])
    faces = np.vstack([faces, cap_faces(top, (n_up - 1) * n_around, n_around, flip=True),
                       cap_faces(top + 1, 0, n_around, flip=False)])
    uv = np.zeros((len(pts), 2))
    uv[:-2, 0] = np.tile(th / (2 * np.pi), n_up)
    uv[:-2, 1] = np.repeat(z / spec.tower_height, n_around)
    return Mesh(spec.to_world(pts), faces, uv, "tower")


# ---- nacelle and hub ------------------------------------------------------------------------------------------------

def rounded_box_mesh(size, center, radius_frac=0.28, n_around=48, n_len=12) -> Mesh:
    """Superellipse cross-section swept along X (a rounded nacelle body), with closed ends."""
    L, W, H = size
    t = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    e = 2.0 / (1.0 + 6.0 * radius_frac)          # superellipse exponent: smaller -> boxier
    cy = np.sign(np.cos(t)) * np.abs(np.cos(t)) ** e * W / 2
    cz = np.sign(np.sin(t)) * np.abs(np.sin(t)) ** e * H / 2
    xs = np.linspace(-L / 2, L / 2, n_len)
    taper = 1.0 - 0.22 * np.clip((xs - L * 0.15) / (L * 0.35), 0, 1)   # front tapers towards the hub
    pts = []
    for x, k in zip(xs, taper):
        pts.append(np.stack([np.full(n_around, x), cy * k, cz * k], -1))
    pts = np.vstack(pts)
    faces = grid_faces(n_len, n_around, wrap=True)
    i0 = len(pts)
    pts = np.vstack([pts, [[xs[0], 0, 0]], [[xs[-1], 0, 0]]])
    faces = np.vstack([faces, cap_faces(i0, 0, n_around, flip=False),
                       cap_faces(i0 + 1, (n_len - 1) * n_around, n_around, flip=True)])
    return Mesh(pts + np.asarray(center), faces, None, "nacelle")


def spinner_mesh(radius, length, center, n_around=48, n_len=16) -> Mesh:
    """Nose cone pointing +X (upwind), base at center."""
    s = np.linspace(0, 1, n_len)
    x = length * s
    r = radius * np.sqrt(np.clip(1 - s ** 2, 0, 1)) ** 0.9
    r[-1] = 0.0
    th = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    pts = np.stack([np.repeat(x[:, None], n_around, 1), r[:, None] * np.cos(th)[None],
                    r[:, None] * np.sin(th)[None]], -1).reshape(-1, 3)
    faces = grid_faces(n_len, n_around, wrap=True)
    i0 = len(pts)
    pts = np.vstack([pts, [[0, 0, 0]]])
    faces = np.vstack([faces, cap_faces(i0, 0, n_around, flip=False)])
    return Mesh(pts + np.asarray(center), faces, None, "spinner")


# ---- blades ---------------------------------------------------------------------------------------------------------

# NREL 5 MW blade: (radius from rotor centre [m], chord [m], twist [deg]); thickness ratio picked from the
# airfoil families listed for each station (cylinder at the root, DU and NACA64 sections outboard).
_NREL_STATIONS = np.array([
    # r,      chord, twist, t/c
    [1.50,   3.542, 13.308, 1.00],
    [2.87,   3.542, 13.308, 1.00],
    [5.60,   3.854, 13.308, 0.90],
    [8.33,   4.167, 13.308, 0.60],
    [11.75,  4.557, 13.308, 0.40],
    [15.85,  4.652, 11.480, 0.35],
    [19.95,  4.458, 10.162, 0.30],
    [24.05,  4.249,  9.011, 0.25],
    [28.15,  4.007,  7.795, 0.25],
    [32.25,  3.748,  6.544, 0.21],
    [36.35,  3.502,  5.361, 0.21],
    [40.45,  3.256,  4.188, 0.18],
    [44.55,  3.010,  3.125, 0.18],
    [48.65,  2.764,  2.319, 0.18],
    [52.75,  2.518,  1.526, 0.18],
    [56.17,  2.313,  0.863, 0.18],
    [58.90,  2.086,  0.370, 0.18],
    [61.63,  1.419,  0.106, 0.18],
    [63.00,  0.600,  0.000, 0.18],
])


def blade_stations(spec: TurbineSpec, s: np.ndarray):
    """Chord, twist (deg) and thickness ratio at span fraction s in [0, 1] (0 = root, 1 = tip)."""
    r = spec.hub_radius + s * spec.blade_length
    scale = (spec.hub_radius + spec.blade_length) / 63.0
    st = _NREL_STATIONS.copy()
    st[:, 0] = spec.hub_radius + (st[:, 0] - 1.5) / 61.5 * spec.blade_length
    chord = np.interp(r, st[:, 0], st[:, 1]) * scale
    twist = np.interp(r, st[:, 0], st[:, 2])
    tc = np.interp(r, st[:, 0], st[:, 3])
    return r, chord, twist, tc


def airfoil_profile(n: int, tc: float, camber: float = 0.02):
    """Closed airfoil outline in chord units, leading edge at x=0, trailing edge at x=1.

    NACA 4-digit thickness law blended towards a circle for thick root sections. Returns (x, y, u) with
    u the perimeter parameter in [0, 1): 0 = trailing edge, 0.25 = suction side, 0.5 = leading edge,
    0.75 = pressure side.
    """
    u = np.linspace(0, 1, n, endpoint=False)
    beta = 2 * np.pi * u
    xc = 0.5 * (1 + np.cos(beta))               # 1 at TE, 0 at LE (cosine spacing)
    upper = u < 0.5
    t = min(tc, 0.40)
    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc ** 2 + 0.2843 * xc ** 3 - 0.1036 * xc ** 4)
    yc = camber * (1 - (2 * xc - 1) ** 2) * (1 - tc)
    y = np.where(upper, yc + yt, yc - yt)
    x = xc
    if tc > 0.40:                                # thick root: blend towards a circle of diameter = chord
        w = np.clip((tc - 0.40) / 0.60, 0, 1)
        cx, cy = 0.5 + 0.5 * np.cos(beta), 0.5 * np.sin(beta)
        x = (1 - w) * x + w * cx
        y = (1 - w) * y + w * cy
    return x, y, u


def blade_frame(spec: TurbineSpec, azimuth_deg: float):
    """Unit vectors (span, tangential, axial) of a blade in turbine-local coordinates.

    span: from hub towards tip in the rotor plane; axial: +X (upwind); tangential: span x axial, the
    direction the leading edge points when the rotor turns."""
    a = np.radians(azimuth_deg + spec.rotor_azimuth_deg)
    span = np.array([0.0, np.sin(a), np.cos(a)])
    axial = np.array([1.0, 0.0, 0.0])
    tang = np.cross(axial, span)
    return span, tang, axial


def blade_surface(spec: TurbineSpec, azimuth_deg: float, s: np.ndarray, u: np.ndarray, pitch_deg: float = 0.0):
    """Surface points (turbine-local) and outward normals at span fractions s (S,) and perimeter params u (U,).

    Returns arrays of shape (S, U, 3)."""
    s, u = np.atleast_1d(np.asarray(s, float)), np.atleast_1d(np.asarray(u, float))
    if len(s) < 2 or len(u) < 3:
        # evaluate on a small neighbourhood so finite-difference normals exist, then pick the request
        eps = 1e-3
        s3 = np.concatenate([[max(si - eps, 0.0), si, min(si + eps, 1.0)] for si in s]) if len(s) < 2 else s
        u3 = np.concatenate([[(ui - eps) % 1.0, ui, (ui + eps) % 1.0] for ui in u]) if len(u) < 3 else u
        p3, n3 = blade_surface(spec, azimuth_deg, s3, u3, pitch_deg)
        si = np.arange(1, len(s3), 3) if len(s) < 2 else np.arange(len(s))
        ui = np.arange(1, len(u3), 3) if len(u) < 3 else np.arange(len(u))
        return p3[np.ix_(si, ui)], n3[np.ix_(si, ui)]
    span, tang, axial = blade_frame(spec, azimuth_deg)
    r, chord, twist, tc = blade_stations(spec, s)
    hub = np.array([spec.overhang, 0.0, spec.hub_height])
    pts = np.zeros((len(s), len(u), 3))
    for i in range(len(s)):
        x, y, _ = airfoil_profile(4 * len(u), tc[i])      # dense profile, resampled by u below
        uu = np.linspace(0, 1, len(x), endpoint=False)
        xi = np.interp(u, uu, x, period=1.0)
        yi = np.interp(u, uu, y, period=1.0)
        # chord coordinate: leading edge forward (+tang), pitch axis at 25% chord; thickness along -axial
        # for the suction side (y>0 faces downwind), so the pressure side faces the wind (+X).
        cx = (0.25 - xi) * chord[i]
        cy = -yi * chord[i]
        ang = np.radians(twist[i] + pitch_deg)
        ca, sa = np.cos(ang), np.sin(ang)
        tx, ty = ca * cx - sa * cy, sa * cx + ca * cy
        pts[i] = hub + r[i] * span + tx[:, None] * tang + ty[:, None] * axial
    # normals from finite differences on the (s, u) grid
    du = np.roll(pts, -1, axis=1) - np.roll(pts, 1, axis=1)
    ds = np.gradient(pts, axis=0)
    n = np.cross(du, ds)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    # orient outward: away from the local section centroid
    cen = pts.mean(axis=1, keepdims=True)
    flip = np.sum(n * (pts - cen), axis=-1) < 0
    n[flip] *= -1
    return pts, n


def blade_mesh(spec: TurbineSpec, index: int, n_span: int = 90, n_around: int = 72) -> Mesh:
    s = np.linspace(0.0, 1.0, n_span) ** 1.15            # denser towards the root where shape changes
    u = np.linspace(0.0, 1.0, n_around, endpoint=False)
    pts, _ = blade_surface(spec, spec.blade_azimuths_deg[index], s, u)
    pts = pts.reshape(-1, 3)
    faces = grid_faces(n_span, n_around, wrap=True)
    # tip and root caps (the root sits inside the spinner; closing it makes the orientation test exact)
    i0 = len(pts)
    tip = pts[-n_around:].mean(axis=0)
    root = pts[:n_around].mean(axis=0)
    pts = np.vstack([pts, tip, root])
    faces = np.vstack([faces, cap_faces(i0, (n_span - 1) * n_around, n_around, flip=True),
                       cap_faces(i0 + 1, 0, n_around, flip=False)])
    uv = np.zeros((len(pts), 2))
    uv[:-2, 0] = np.tile(u, n_span)
    uv[:-2, 1] = np.repeat(s, n_around)
    m = Mesh(spec.to_world(pts), faces, uv, f"blade_{index}")
    orient_outward(m)
    return m


def turbine_meshes(spec: TurbineSpec) -> list:
    meshes = [tower_mesh(spec)]
    L, W, H = spec.nacelle_size
    nac_center = np.array([spec.overhang - L / 2 - 0.6, 0.0, spec.hub_height + 0.2])
    nac = rounded_box_mesh(spec.nacelle_size, nac_center)
    nac.points = spec.to_world(nac.points)
    meshes.append(nac)
    spin = spinner_mesh(spec.hub_radius * 1.25, 3.6, np.array([spec.overhang - 1.2, 0.0, spec.hub_height]))
    spin.points = spec.to_world(spin.points)
    meshes.append(spin)
    for i in range(len(spec.blade_azimuths_deg)):
        meshes.append(blade_mesh(spec, i))
    for m in meshes:
        orient_outward(m)
    return meshes
