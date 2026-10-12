"""Post-flight analysis of the inspection photos.

Each photo comes with its depth image (distance to the image plane, metres), the exact camera pose (USD camera
convention: x right, y up, looking along -z) and the intrinsics. From these:

- coverage: which surface samples of the turbine were seen, how sharply (ground sample distance, mm/px) and how
  often, per component;
- defects: which ground-truth defects were imaged, in how many photos, at what GSD, with the best crop;
- 3D model: the depth images back-projected into a coloured point cloud in the world frame (voxel-averaged).
"""
import glob
import json
import os

import numpy as np
from PIL import Image


def load_photos(photo_dir):
    """List of (meta dict, image path, depth path) sorted by waypoint index."""
    out = []
    for meta_path in sorted(glob.glob(os.path.join(photo_dir, "wp*.json"))):
        base = meta_path[:-5]
        if os.path.exists(base + ".jpg") and os.path.exists(base + "_depth.npy"):
            out.append((json.load(open(meta_path)), base + ".jpg", base + "_depth.npy"))
    return out


def project(points, meta):
    """Project world points into a photo. Returns (u, v, depth_along_view) arrays."""
    T = np.asarray(meta["camera_to_world_usd"])
    R, t = T[:3, :3], T[:3, 3]
    pc = (points - t) @ R                     # world -> camera (rows)
    z = -pc[:, 2]                              # distance along the viewing axis
    k = meta["intrinsics"]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k["cx"] + k["fx"] * pc[:, 0] / z
        v = k["cy"] - k["fy"] * pc[:, 1] / z
    return u, v, z


def visible(points, normals, meta, depth, depth_tol=0.3, max_incidence_deg=75.0):
    """Mask of points seen in this photo (in frame, in front, not occluded, not too oblique) and their GSD (mm/px)."""
    k = meta["intrinsics"]
    u, v, z = project(points, meta)
    W, H = k["width"], k["height"]
    ok = (z > 0.2) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui = np.clip(u.astype(int), 0, W - 1)
    vi = np.clip(v.astype(int), 0, H - 1)
    d = depth[vi, ui].astype(np.float32)
    ok &= np.isfinite(d) & (np.abs(d - z) < depth_tol)
    if normals is not None:
        T = np.asarray(meta["camera_to_world_usd"])
        ray = points - T[:3, 3]
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        cos_inc = -np.sum(ray * normals, axis=1)
        ok &= cos_inc > np.cos(np.radians(max_incidence_deg))
    gsd = z / k["fx"] * 1000.0
    return ok, gsd, u, v


def coverage_from_photos(photos, samples, gsd_limit_mm=8.0):
    pts, nrm, lab = samples["points"], samples["normals"], samples["labels"]
    views = np.zeros(len(pts), int)
    best = np.full(len(pts), np.inf)
    for meta, _, dpath in photos:
        depth = np.load(dpath).astype(np.float32)
        ok, gsd, _, _ = visible(pts, nrm, meta, depth)
        views += ok
        best = np.where(ok, np.minimum(best, gsd), best)
    report = {}
    for c in sorted(set(lab.tolist())):
        m = lab == c
        seen = views[m] > 0
        sharp = best[m] <= gsd_limit_mm
        report[c] = {"samples": int(m.sum()), "seen": round(float(seen.mean()), 4),
                     f"seen_at_or_below_{gsd_limit_mm:g}mm_per_px": round(float(sharp.mean()), 4),
                     "median_gsd_mm": round(float(np.median(best[m][seen])), 2) if seen.any() else None,
                     "mean_views": round(float(views[m][seen].mean()), 2) if seen.any() else 0.0}
    return report, views, best


def defect_check(photos, defects, crop_dir=None):
    """For each defect: photos that see its centroid, best GSD, and a crop from the sharpest one."""
    results = []
    for d in defects:
        c = np.asarray(d["centroid"], float)[None]
        hits = []
        for meta, ipath, dpath in photos:
            depth = np.load(dpath).astype(np.float32)
            ok, gsd, u, v = visible(c, None, meta, depth, depth_tol=0.5)
            if ok[0]:
                hits.append((float(gsd[0]), float(u[0]), float(v[0]), meta["index"], ipath))
        hits.sort()
        r = {"id": d["id"], "kind": d["kind"], "component": d["component"], "photos": len(hits),
             "best_gsd_mm": round(hits[0][0], 2) if hits else None,
             "best_photo": os.path.basename(hits[0][4]) if hits else None}
        if hits and crop_dir:
            gsd, u, v, _, ipath = hits[0]
            size_m = max(d.get("size_m", [1.0])[1:] or [1.0])
            half = int(np.clip(size_m * 1000.0 / gsd * 0.9, 60, 360))
            img = Image.open(ipath)
            box = (max(0, int(u) - half), max(0, int(v) - half), min(img.width, int(u) + half),
                   min(img.height, int(v) + half))
            os.makedirs(crop_dir, exist_ok=True)
            out = os.path.join(crop_dir, f"{d['id']}_{d['kind']}.jpg")
            img.crop(box).save(out, quality=93)
            r["crop"] = out
        results.append(r)
    return results


def fuse_point_cloud(photos, voxel=0.05, stride=3, max_depth=30.0):
    """Back-project every photo's depth into the world frame and voxel-average the coloured points."""
    all_p, all_c = [], []
    for meta, ipath, dpath in photos:
        depth = np.load(dpath).astype(np.float32)
        rgb = np.asarray(Image.open(ipath).convert("RGB"), np.float32)
        k = meta["intrinsics"]
        vs, us = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        z = depth[vs, us]
        ok = np.isfinite(z) & (z > 0.2) & (z < max_depth)
        us, vs, z = us[ok], vs[ok], z[ok]
        pc = np.stack([(us - k["cx"]) / k["fx"] * z, -(vs - k["cy"]) / k["fy"] * z, -z], axis=1)
        T = np.asarray(meta["camera_to_world_usd"])
        all_p.append(pc @ T[:3, :3].T + T[:3, 3])
        all_c.append(rgb[vs, us])
    P = np.concatenate(all_p)
    C = np.concatenate(all_c)
    keys = np.floor(P / voxel).astype(np.int64)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    pts = np.stack([np.bincount(inv, P[:, i]) for i in range(3)], 1) / cnt[:, None]
    cols = (np.stack([np.bincount(inv, C[:, i]) for i in range(3)], 1) / cnt[:, None]).clip(0, 255).astype(np.uint8)
    return pts, cols


def write_ply(path, pts, cols):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.zeros(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
        rec["r"], rec["g"], rec["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
        f.write(rec.tobytes())
