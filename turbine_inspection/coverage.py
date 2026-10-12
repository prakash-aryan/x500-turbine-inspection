"""Inspection path planning and photo coverage for the wind turbine.

The planner flies vertical passes around the tower and four passes along every blade (pressure side,
leading edge, suction side, trailing edge) at a fixed stand-off distance, with the gimbal camera
pointed at the closest surface point. Waypoints closer than the minimum clearance to any part of the
turbine are pushed outward; passes that would run into the tower are rotated sideways around the
blade. Coverage is the fraction of surface samples seen by at least one photo at a usable incidence
angle and ground sample distance, with occlusion from a point-based z-buffer.
"""
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.spatial import cKDTree

from .turbine import TurbineSpec, blade_frame, blade_stations, blade_surface, tower_radius, turbine_meshes


@dataclass
class CameraSpec:
    width: int = 4000                    # 12 MP inspection camera on the gimbal
    height: int = 3000
    hfov_deg: float = 50.0
    max_range: float = 25.0
    max_incidence_deg: float = 70.0      # angle between the surface normal and the view ray
    max_gsd_mm: float = 3.0              # coarser than this does not count as inspected

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(np.tan(np.radians(self.hfov_deg) / 2) * self.height / self.width)))


@dataclass
class PlannerConfig:
    standoff: float = 9.0
    min_clearance: float = 5.5
    step: float = 4.0                    # distance between photo waypoints along a pass
    tower_azimuths_deg: tuple = (36.0, 108.0, 180.0, 252.0, 324.0)   # no pass straight upwind: blade 0 hangs there
    tower_z: tuple = (8.0, 84.0)
    blade_r_start: float = 9.0           # metres from the rotor centre where blade passes start
    safe_x: float = 32.0                 # transit plane in front of the rotor
    home: tuple = (10.0, -10.0, 0.3)        # on the gravel pad, upwind of the tower
    cruise_speed: float = 2.0
    photo_stop_s: float = 1.0
    arc_step_deg: float = 15.0


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    yaw_deg: float
    gimbal_pitch_deg: float
    segment: str
    photo: bool
    target: list = field(default_factory=list)


def look_angles(c, t):
    d = np.asarray(t) - np.asarray(c)
    yaw = np.degrees(np.arctan2(d[1], d[0]))
    pitch = np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))
    return float(yaw), float(pitch)


class Planner:
    def __init__(self, spec: TurbineSpec, cfg: PlannerConfig = PlannerConfig()):
        self.spec, self.cfg = spec, cfg
        pts = np.vstack([m.points for m in turbine_meshes(spec)])
        self.tree = cKDTree(pts)

    # -- safety ---------------------------------------------------------------------------------------------------------
    def clearance(self, p) -> float:
        return float(self.tree.query(np.asarray(p))[0])

    def push_clear(self, p, direction):
        p = np.asarray(p, float)
        direction = np.asarray(direction, float) / np.linalg.norm(direction)
        for _ in range(60):
            if self.clearance(p) >= self.cfg.min_clearance:
                return p
            p = p + 0.5 * direction
        return p

    # -- passes ---------------------------------------------------------------------------------------------------------
    def tower_point(self, az_deg, z):
        sp, cfg = self.spec, self.cfg
        a = np.radians(az_deg)
        radial = sp.dir_to_world(np.array([np.cos(a), np.sin(a), 0.0]))
        axis = np.array([sp.position[0], sp.position[1], z])
        r = float(tower_radius(sp, z))
        c = self.push_clear(axis + radial * (r + cfg.standoff), radial)
        return c, axis + radial * r

    def tower_passes(self):
        cfg = self.cfg
        wps = []
        zs = np.arange(cfg.tower_z[0], cfg.tower_z[1] + 1e-6, cfg.step)
        azs = list(cfg.tower_azimuths_deg)
        for k, az in enumerate(azs):
            col = zs if k % 2 == 0 else zs[::-1]
            for z in col:
                c, t = self.tower_point(az, z)
                yaw, pitch = look_angles(c, t)
                wps.append(Waypoint(*c.tolist(), yaw, pitch, f"tower_{int(az)}", True, t.round(3).tolist()))
            if k + 1 < len(azs):                         # arc around the tower to the next side
                z = col[-1]
                for a_deg in np.arange(az + cfg.arc_step_deg, azs[k + 1], cfg.arc_step_deg):
                    c, t = self.tower_point(a_deg, z)
                    yaw, pitch = look_angles(c, t)
                    wps.append(Waypoint(*c.tolist(), yaw, pitch, "tower_arc", False, []))
        return wps

    def blade_point(self, index, r, ang_deg, side_u):
        """Camera position around blade `index` at radius r, offset direction ang_deg in the section plane
        (0 = upwind/pressure side, 90 = leading edge), aimed at the surface point with perimeter param side_u."""
        sp, cfg = self.spec, self.cfg
        span, tang, axial = (sp.dir_to_world(v) for v in blade_frame(sp, sp.blade_azimuths_deg[index]))
        hub = sp.hub_center()
        s_r = float(np.clip((r - sp.hub_radius) / sp.blade_length, 0, 1))
        surf, _ = blade_surface(sp, sp.blade_azimuths_deg[index], np.array([s_r]), np.array([side_u]))
        t = sp.to_world(surf.reshape(3))
        a = np.radians(ang_deg)
        off = np.cos(a) * axial + np.sin(a) * tang
        return t + off * cfg.standoff, t, hub + r * span

    def blade_passes(self, index: int):
        cfg, sp = self.cfg, self.spec
        rs = np.arange(cfg.blade_r_start, sp.rotor_radius + 1.5, cfg.step)
        sides = [("pressure", 0.0, 0.75), ("leading_edge", 90.0, 0.5), ("suction", 180.0, 0.25),
                 ("trailing_edge", 270.0, 0.0)]
        wps, last = [], None
        for k, (name, ang0, u_side) in enumerate(sides):
            order = list(range(len(rs))) if k % 2 == 0 else list(range(len(rs) - 1, -1, -1))
            for j, i in enumerate(order):
                best, best_clr, best_ang = None, -1.0, ang0
                for dang in (0, 15, -15, 30, -30, 45, -45, 60, -60):   # rotate around the blade if blocked
                    c, t, axis_pt = self.blade_point(index, rs[i], ang0 + dang, u_side)
                    clr = self.clearance(c)
                    if clr > best_clr:
                        best, best_clr, best_ang, best_t, best_axis = c, clr, ang0 + dang, t, axis_pt
                    if clr >= cfg.min_clearance:
                        break
                c = self.push_clear(best, best - best_axis)
                if j == 0 and last is not None:              # arc from the previous side at this radius
                    a0, a1 = last
                    step = cfg.arc_step_deg if best_ang > a0 else -cfg.arc_step_deg
                    for a_deg in np.arange(a0 + step, best_ang, step):
                        ca, ta, axa = self.blade_point(index, rs[i], a_deg, u_side)
                        ca = self.push_clear(ca, ca - axa)
                        yaw, pitch = look_angles(ca, ta)
                        wps.append(Waypoint(*ca.tolist(), yaw, pitch, f"blade{index}_arc", False, []))
                yaw, pitch = look_angles(c, best_t)
                wps.append(Waypoint(*c.tolist(), yaw, pitch, f"blade{index}_{name}", True, best_t.round(3).tolist()))
                last = (best_ang, None)
        return wps

    def transit(self, a: Waypoint, b: Waypoint):
        """Leave a pass outward to the transit plane in front of the rotor, move, and come back in."""
        cfg = self.cfg
        out_a = np.array([max(cfg.safe_x, a.x), a.y, a.z])
        out_b = np.array([max(cfg.safe_x, b.x), b.y, b.z])
        res = []
        for p in (out_a, out_b):
            yaw, pitch = look_angles(p, self.spec.hub_center())
            res.append(Waypoint(*p.tolist(), yaw, pitch, "transit", False, []))
        return res

    def plan(self):
        cfg = self.cfg
        segments = [self.tower_passes()] + [self.blade_passes(i) for i in (0, 2, 1)]
        home = np.array(cfg.home, float)
        route = [Waypoint(*home.tolist(), 0.0, 0.0, "takeoff", False, [])]
        first = segments[0][0]
        route.append(Waypoint(home[0], home[1], first.z, 0.0, 0.0, "climb", False, []))
        prev = None
        for seg in segments:
            if prev is not None:
                route += self.transit(prev, seg[0])
            route += seg
            prev = seg[-1]
        route += self.transit(prev, Waypoint(home[0], home[1], 30.0, 0, 0, "", False))
        route.append(Waypoint(home[0], home[1], 30.0, 0.0, 0.0, "return", False, []))
        route.append(Waypoint(*home.tolist(), 0.0, 0.0, "land", False, []))
        return route

    def leg_clearance(self, a: Waypoint, b: Waypoint) -> float:
        pa, pb = np.array([a.x, a.y, a.z]), np.array([b.x, b.y, b.z])
        n = max(2, int(np.linalg.norm(pb - pa) / 0.5))
        return float(self.tree.query(np.linspace(pa, pb, n))[0].min())


# ---- coverage ---------------------------------------------------------------------------------------------------------

def surface_samples(spec: TurbineSpec, n_s=140, n_u=56, n_z=70, n_th=56):
    """Points, normals and component labels on the surfaces to inspect (blades and tower)."""
    P, N, L = [], [], []
    s = np.linspace(0.02, 0.995, n_s)
    u = np.linspace(0, 1, n_u, endpoint=False)
    for i, az in enumerate(spec.blade_azimuths_deg):
        p, n = blade_surface(spec, az, s, u)
        P.append(spec.to_world(p.reshape(-1, 3)))
        N.append(spec.dir_to_world(n.reshape(-1, 3)))
        L += [f"blade_{i}"] * len(s) * len(u)
    z = np.linspace(2.0, spec.tower_height - 1.0, n_z)
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    r = tower_radius(spec, z)
    p = np.stack([r[:, None] * np.cos(th)[None], r[:, None] * np.sin(th)[None], np.repeat(z[:, None], n_th, 1)],
                 -1).reshape(-1, 3)
    n = np.stack([np.cos(th)[None].repeat(n_z, 0), np.sin(th)[None].repeat(n_z, 0), np.zeros((n_z, n_th))],
                 -1).reshape(-1, 3)
    P.append(spec.to_world(p))
    N.append(spec.dir_to_world(n))
    L += ["tower"] * len(p)
    return np.vstack(P), np.vstack(N), np.array(L)


def occluder_points(spec: TurbineSpec, density: float = 120.0, seed: int = 0):
    """Area-weighted random points on every turbine triangle (blades, tower, nacelle, spinner)."""
    rng = np.random.default_rng(seed)
    out = []
    for m in turbine_meshes(spec):
        p, f = m.points, m.faces
        a, b, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        n = rng.poisson(area * density)
        idx = np.repeat(np.arange(len(f)), n)
        r1, r2 = rng.random(len(idx)), rng.random(len(idx))
        sq = np.sqrt(r1)
        pts = (1 - sq)[:, None] * a[idx] + (sq * (1 - r2))[:, None] * b[idx] + (sq * r2)[:, None] * c[idx]
        out.append(pts)
    return np.vstack(out)


def camera_basis(c, t):
    fwd = np.asarray(t, float) - np.asarray(c, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fwd, right, up


def coverage(route, spec: TurbineSpec, cam: CameraSpec = CameraSpec(), bins=(192, 108), depth_tol=0.30):
    P, N, L = surface_samples(spec)
    occ = occluder_points(spec)
    tx = np.tan(np.radians(cam.hfov_deg) / 2)
    ty = np.tan(np.radians(cam.vfov_deg) / 2)
    seen = np.zeros(len(P), dtype=np.int32)
    best_gsd = np.full(len(P), np.inf)
    cos_inc = np.cos(np.radians(cam.max_incidence_deg))
    for w in route:
        if not w.photo:
            continue
        c = np.array([w.x, w.y, w.z])
        fwd, right, up = camera_basis(c, w.target)

        def project(X):
            d = X - c
            z = d @ fwd
            x = (d @ right) / np.maximum(z, 1e-6) / tx
            y = (d @ up) / np.maximum(z, 1e-6) / ty
            ok = (z > 0.2) & (np.abs(x) <= 1) & (np.abs(y) <= 1)
            bx = np.clip(((x + 1) / 2 * bins[0]).astype(int), 0, bins[0] - 1)
            by = np.clip(((y + 1) / 2 * bins[1]).astype(int), 0, bins[1] - 1)
            return z, ok, bx * bins[1] + by
        zo, oko, bo = project(occ)
        zbuf = np.full(bins[0] * bins[1], np.inf)
        np.minimum.at(zbuf, bo[oko], zo[oko])
        zp, okp, bp = project(P)
        dist = np.linalg.norm(P - c, axis=1)
        view = (c - P) / dist[:, None]
        inc = np.sum(N * view, axis=1)
        vis = okp & (dist <= cam.max_range) & (inc >= cos_inc) & (zp <= zbuf[bp] + depth_tol)
        gsd = dist * 2 * tx / cam.width * 1000.0          # mm per pixel at the sample
        vis &= gsd <= cam.max_gsd_mm
        seen[vis] += 1
        best_gsd[vis] = np.minimum(best_gsd[vis], gsd[vis])
    report = {}
    for comp in sorted(set(L)):
        m = L == comp
        cov = float(np.mean(seen[m] > 0))
        report[comp] = {"coverage": round(cov, 4),
                        "median_gsd_mm": round(float(np.median(best_gsd[m][np.isfinite(best_gsd[m])])), 2)
                        if np.any(np.isfinite(best_gsd[m])) else None,
                        "mean_views": round(float(np.mean(seen[m])), 2)}
    return report, {"points": P, "normals": N, "labels": L, "views": seen, "gsd": best_gsd}


def route_stats(route, cfg: PlannerConfig):
    p = np.array([[w.x, w.y, w.z] for w in route])
    length = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    photos = sum(w.photo for w in route)
    return {"waypoints": len(route), "photos": photos, "path_length_m": round(length, 1),
            "est_flight_time_min": round((length / cfg.cruise_speed + photos * cfg.photo_stop_s) / 60.0, 1)}


def route_json(route):
    return [asdict(w) for w in route]
