"""Synthetic surface defects at known positions on the turbine.

Each defect is a thin patch that follows the blade or tower surface 4 mm above it, textured with a
procedurally generated image. The image background equals the paint colour, so only the damage is
visible. Ground truth (component, span fraction, perimeter range, world centroid, size) is exported
as JSON for scoring inspection results.
"""
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image
from scipy import ndimage

from .turbine import Mesh, TurbineSpec, blade_surface, grid_faces, tower_radius

PAINT_SRGB = np.array([232, 234, 236], dtype=np.float64)     # light grey-white turbine paint


@dataclass
class Defect:
    id: str
    kind: str                 # leading_edge_erosion | crack | lightning_strike | oil_streak | delamination
    component: str            # blade_0..2 | tower
    s_range: tuple            # span fraction (blades) or height range in metres (tower)
    u_range: tuple            # perimeter parameter (blades) or angle range in degrees (tower)
    severity: str
    texture: str = ""
    centroid: tuple = ()
    size_m: tuple = ()


def default_defects() -> list:
    return [
        Defect("D1", "leading_edge_erosion", "blade_1", (0.80, 0.93), (0.40, 0.60), "medium"),
        Defect("D2", "crack", "blade_0", (0.330, 0.352), (0.60, 0.90), "high"),
        Defect("D3", "lightning_strike", "blade_2", (0.952, 0.982), (0.78, 0.96), "high"),
        Defect("D4", "delamination", "blade_1", (0.50, 0.53), (0.66, 0.82), "low"),
        Defect("D5", "oil_streak", "tower", (76.0, 87.0), (170.0, 196.0), "medium"),
    ]


# ---- textures ---------------------------------------------------------------------------------------------------------

def _noise(h, w, scale, rng):
    z = rng.random((h // scale + 2, w // scale + 2))
    return ndimage.zoom(z, scale, order=3)[:h, :w]


def _fade_mask(h, w, margin=0.18):
    """1 in the middle, 0 at the patch border, so edges blend into the paint."""
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    fy = np.clip(np.minimum(y, 1 - y) / margin, 0, 1)
    fx = np.clip(np.minimum(x, 1 - x) / margin, 0, 1)
    return (fy * fx) ** 1.5


def make_texture(kind: str, h: int, w: int, seed: int) -> np.ndarray:
    """RGB uint8 image (rows = span direction, columns = perimeter / angle direction)."""
    rng = np.random.default_rng(seed)
    img = np.ones((h, w, 3)) * PAINT_SRGB
    fade = _fade_mask(h, w)
    if kind == "leading_edge_erosion":
        # pitting concentrated on the leading-edge line (middle column), worse towards the tip (top rows)
        x = np.linspace(-1, 1, w)[None, :]
        y = np.linspace(0, 1, h)[:, None]
        band = np.exp(-(x / 0.35) ** 2) * (0.4 + 0.6 * y)
        pits = (rng.random((h, w)) < 0.09 * band) | (_noise(h, w, 5, rng) * band > 0.45)
        pits = ndimage.binary_dilation(pits, iterations=1)
        rough = _noise(h, w, 3, rng)
        dark = np.array([96, 84, 70]) + 30 * rough[..., None]
        a = (pits * band * 0.9 + 0.25 * band * rough)[..., None] * fade[..., None]
        img = img * (1 - a) + dark * a
    elif kind == "crack":
        # thin jagged dark line across the chord (columns), with a faint stress halo
        yc = h / 2 + np.cumsum(rng.normal(0, 0.6, w))
        yc = ndimage.gaussian_filter1d(yc, 3)
        rows = np.arange(h)[:, None]
        d = np.abs(rows - yc[None, :])
        line = np.exp(-(d / 1.3) ** 2)
        halo = np.exp(-(d / 7.0) ** 2) * 0.18
        ext = np.clip(1.5 * np.sin(np.linspace(0, np.pi, w))[None, :], 0, 1)
        a = np.clip(line * 0.95 + halo, 0, 1) * ext * _fade_mask(h, w, 0.08)
        img = img * (1 - a[..., None]) + np.array([40, 36, 34]) * a[..., None]
    elif kind == "lightning_strike":
        y, x = np.mgrid[0:h, 0:w]
        cy, cx = h * 0.5, w * 0.45
        r = np.hypot((y - cy) / h, (x - cx) / w)
        ang = np.arctan2(y - cy, x - cx)
        ragged = 0.20 + 0.06 * np.sin(7 * ang + rng.random() * 6) + 0.04 * _noise(h, w, 8, rng)
        burn = np.clip(1 - r / ragged, 0, 1) ** 0.7
        hole = (r < 0.045).astype(float)
        soot = np.array([46, 38, 32]) + 40 * _noise(h, w, 4, rng)[..., None]
        a = (burn * 0.95)[..., None] * fade[..., None]
        img = img * (1 - a) + soot * a
        img = img * (1 - hole[..., None]) + np.array([20, 16, 14]) * hole[..., None]
    elif kind == "delamination":
        # gelcoat flaked off, showing the yellowish laminate underneath, with a darker rim
        n = _noise(h, w, 10, rng) * fade
        patch = n > 0.42
        rim = ndimage.binary_dilation(patch, iterations=2) & ~patch
        fibre = np.array([214, 196, 150]) + 18 * _noise(h, w, 2, rng)[..., None]
        img[patch] = fibre[patch]
        img[rim] = img[rim] * 0.55 + np.array([120, 110, 90]) * 0.45
    elif kind == "oil_streak":
        # dark vertical runs from the top edge (row 0 = top of the patch) fading downwards
        x = np.arange(w)
        a = np.zeros((h, w))
        for _ in range(9):
            cx = rng.uniform(0.2, 0.8) * w
            width = rng.uniform(1.5, 5.0)
            length = rng.uniform(0.4, 1.0) * h
            col = np.exp(-((x - cx) / width) ** 2)[None, :]
            rows = np.arange(h)[:, None]
            run = np.clip(1 - rows / length, 0, 1) ** 0.8
            a = np.maximum(a, col * run * rng.uniform(0.5, 0.9))
        a = ndimage.gaussian_filter(a, 1.0) * _fade_mask(h, w, 0.10)
        img = img * (1 - a[..., None]) + np.array([58, 48, 30]) * a[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


# ---- patch geometry ---------------------------------------------------------------------------------------------------

def defect_patch(spec: TurbineSpec, d: Defect, offset: float = 0.004, n_s: int = 24, n_u: int = 32) -> Mesh:
    if d.component.startswith("blade"):
        idx = int(d.component.split("_")[1])
        s = np.linspace(d.s_range[0], d.s_range[1], n_s)
        u = np.linspace(d.u_range[0], d.u_range[1], n_u)
        pts, nrm = blade_surface(spec, spec.blade_azimuths_deg[idx], s, u)
        pts = pts + offset * nrm
        pts = spec.to_world(pts.reshape(-1, 3))
        ref_normal = spec.dir_to_world(nrm.reshape(-1, 3).mean(axis=0))
    else:
        z = np.linspace(d.s_range[1], d.s_range[0], n_s)          # top row first (matches the texture)
        th = np.radians(np.linspace(d.u_range[0], d.u_range[1], n_u))
        r = tower_radius(spec, z) + offset
        pts = np.stack([r[:, None] * np.cos(th)[None], r[:, None] * np.sin(th)[None],
                        np.repeat(z[:, None], n_u, 1)], -1).reshape(-1, 3)
        pts = spec.to_world(pts)
    faces = grid_faces(n_s, n_u, wrap=False)
    uu, vv = np.meshgrid(np.linspace(0, 1, n_u), np.linspace(1, 0, n_s))   # row 0 of the image at v=1
    m = Mesh(pts, faces, np.stack([uu.ravel(), vv.ravel()], -1), f"defect_{d.id}")
    # face winding: outward like the surface it sits on
    p, f = m.points, m.faces
    fn = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
    if d.component.startswith("blade"):
        ref = ref_normal
    else:
        c = p.mean(axis=0)
        ref = c - np.array([spec.position[0], spec.position[1], c[2]])
    if np.mean(fn @ ref > 0) < 0.5:
        m.faces = f[:, ::-1].copy()
    c = m.points.mean(axis=0)
    ext = m.points.max(axis=0) - m.points.min(axis=0)
    d.centroid = tuple(float(v) for v in c.round(3))
    d.size_m = tuple(float(v) for v in ext.round(2))
    return m


def write_textures(defects: list, out_dir) -> None:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sizes = {"leading_edge_erosion": (1024, 128), "crack": (128, 512), "lightning_strike": (256, 256),
             "delamination": (256, 256), "oil_streak": (512, 256)}
    for i, d in enumerate(defects):
        h, w = sizes[d.kind]
        path = out / f"defect_{d.id}_{d.kind}.png"
        Image.fromarray(make_texture(d.kind, h, w, seed=100 + i)).save(path)
        d.texture = str(path)


def defects_json(defects: list) -> list:
    return [asdict(d) for d in defects]
