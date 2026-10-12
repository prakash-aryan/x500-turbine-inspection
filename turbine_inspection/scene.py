"""Write the inspection scene as USD: turbine with defects, wind-farm neighbours, terrain, sky and sun.

Only pxr (usd-core) and numpy are needed; the stage opens in Isaac Sim 6.1.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from .defects import PAINT_SRGB, default_defects, defect_patch, defects_json, write_textures
from .turbine import Mesh, TurbineSpec, turbine_meshes


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# ---- USD helpers --------------------------------------------------------------------------------------------------

def add_mesh(stage, path: str, m: Mesh, material=None, collider: bool = False, double_sided: bool = False):
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(m.points.astype(np.float32)))
    g.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(m.faces)))
    g.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(m.faces.astype(np.int32).ravel()))
    g.CreateSubdivisionSchemeAttr("none")
    g.CreateDoubleSidedAttr(double_sided)
    nrm = m.normals().astype(np.float32)
    g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nrm))
    g.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if m.uv is not None:
        pv = UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(m.uv.astype(np.float32)))
    lo, hi = m.points.min(axis=0), m.points.max(axis=0)
    g.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())]))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(material)
    if collider:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr("none")
    return g


def preview_material(stage, path: str, color_linear=None, roughness=0.5, metallic=0.0, texture: str = None,
                     opacity: float = 1.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    if texture:
        st = UsdShade.Shader.Define(stage, path + "/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, path + "/tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(v) for v in color_linear]))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


# ---- procedural ground textures -------------------------------------------------------------------------------------

def _periodic_noise(n, octaves, rng, falloff=1.8):
    """Tileable fractal noise in [0, 1] from filtered white noise in the Fourier domain."""
    f = np.fft.fftfreq(n)
    k = np.hypot(f[:, None], f[None, :])
    k[0, 0] = 1.0
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / k ** falloff
    spec[0, 0] = 0
    lo = 1.0 / (2 ** octaves)
    spec[k < lo / n] = 0
    z = np.real(np.fft.ifft2(spec))
    return (z - z.min()) / (z.max() - z.min())


def meadow_texture(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    big = _periodic_noise(n, 9, rng, 2.4)
    mid = _periodic_noise(n, 9, rng, 1.7)
    fine = _periodic_noise(n, 9, rng, 0.8)
    greens = np.array([[52, 74, 34], [66, 90, 40], [82, 102, 46], [104, 110, 58], [122, 112, 70]], float)
    t = np.clip(0.60 * big + 0.30 * mid + 0.10 * fine, 0, 1) ** 1.2 * (len(greens) - 1)
    i0 = np.floor(t).astype(int).clip(0, len(greens) - 2)
    w = (t - i0)[..., None]
    col = greens[i0] * (1 - w) + greens[i0 + 1] * w
    col *= (0.80 + 0.40 * fine)[..., None]                    # blade-of-grass scale brightness jitter
    return np.clip(col, 0, 255).astype(np.uint8)


def gravel_texture(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = _periodic_noise(n, 8, rng, 0.6)
    b = _periodic_noise(n, 8, rng, 1.4)
    base = np.array([118, 113, 106], float)
    col = base * (0.75 + 0.35 * a + 0.15 * b)[..., None]
    return np.clip(col, 0, 255).astype(np.uint8)


def terrain_mesh(size: float, n: int, flat_radius: float, amplitude: float, tile: float, seed: int,
                 center=(0.0, 0.0)) -> Mesh:
    rng = np.random.default_rng(seed)
    h = _periodic_noise(n, 8, rng, 2.4) - 0.5
    xs = np.linspace(-size / 2, size / 2, n) + center[0]
    ys = np.linspace(-size / 2, size / 2, n) + center[1]
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    r = np.hypot(X, Y)
    ramp = np.clip((r - flat_radius) / (size / 2 - flat_radius), 0, 1) ** 1.5
    Z = amplitude * h * ramp - 0.05
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    f = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            f += [(a, a + 1, a + n + 1), (a, a + n + 1, a + n)]
    uv = np.stack([X.ravel() / tile, Y.ravel() / tile], -1)
    return Mesh(pts, np.array(f, dtype=np.int64), uv, "terrain")


def disc_mesh(radius: float, n: int, z: float, tile: float, center=(0.0, 0.0)) -> Mesh:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([center[0] + radius * np.cos(th), center[1] + radius * np.sin(th), np.full(n, z)], -1)
    pts = np.vstack([[center[0], center[1], z], ring])
    f = np.array([(0, 1 + j, 1 + (j + 1) % n) for j in range(n)], dtype=np.int64)
    uv = pts[:, :2] / tile
    return Mesh(pts, f, uv, "pad")


def strip_mesh(p0, p1, width: float, z: float, tile: float) -> Mesh:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    nrm = np.array([-d[1], d[0]]) * width / 2
    pts = np.array([[*(p0 + nrm), z], [*(p0 - nrm), z], [*(p1 - nrm), z], [*(p1 + nrm), z]])
    f = np.array([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
    return Mesh(pts, f, pts[:, :2] / tile, "road")


# ---- scene ------------------------------------------------------------------------------------------------------------

FARM_LAYOUT = [((0.0, 720.0), 15.0), ((0.0, -700.0), 95.0), ((-880.0, 360.0), 40.0), ((-900.0, -380.0), 75.0),
               ((-1750.0, 0.0), 20.0), ((820.0, 1350.0), 50.0)]


def build_scene(out_dir, hdri: str = None, farm: bool = True, seed: int = 7) -> dict:
    out = Path(out_dir)
    tex_dir = out / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(meadow_texture(4096, seed)).save(tex_dir / "meadow.png")
    Image.fromarray(gravel_texture(512, seed + 1)).save(tex_dir / "gravel.png")

    spec = TurbineSpec()
    defects = default_defects()
    write_textures(defects, tex_dir)

    stage = Usd.Stage.CreateNew(str(out / "turbine_scene.usd"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")

    paint = preview_material(stage, "/World/Looks/BladePaint", srgb_to_linear(PAINT_SRGB), roughness=0.32)
    tower_paint = preview_material(stage, "/World/Looks/TowerPaint", srgb_to_linear(PAINT_SRGB) * 0.97, roughness=0.42)
    meadow = preview_material(stage, "/World/Looks/Meadow", roughness=0.92, texture="./textures/meadow.png")
    gravel = preview_material(stage, "/World/Looks/Gravel", roughness=0.95, texture="./textures/gravel.png")

    # main turbine (inspection target)
    UsdGeom.Xform.Define(stage, "/World/Turbine")
    for m in turbine_meshes(spec):
        mat = tower_paint if m.name in ("tower",) else paint
        add_mesh(stage, f"/World/Turbine/{m.name}", m, mat, collider=True)
    UsdGeom.Scope.Define(stage, "/World/Turbine/Defects")
    for d in defects:
        dm = defect_patch(spec, d)
        tex_mat = preview_material(stage, f"/World/Looks/Defect_{d.id}",
                                   roughness=0.55 if d.kind != "oil_streak" else 0.28,
                                   texture="./textures/" + Path(d.texture).name)
        add_mesh(stage, f"/World/Turbine/Defects/{d.id}", dm, tex_mat)

    # neighbours (clean, different rotor positions)
    if farm:
        UsdGeom.Xform.Define(stage, "/World/WindFarm")
        for k, ((x, y), az) in enumerate(FARM_LAYOUT):
            sp = TurbineSpec(position=(x, y, 0.0), rotor_azimuth_deg=az)
            for m in turbine_meshes(sp):
                add_mesh(stage, f"/World/WindFarm/T{k}_{m.name}", m,
                         tower_paint if m.name == "tower" else paint)

    # ground: flat meadow around the site, rolling terrain further out, gravel pad and access road
    terrain = terrain_mesh(5000.0, 220, 420.0, 70.0, 95.0, seed)
    add_mesh(stage, "/World/Ground/Terrain", terrain, meadow, collider=True)
    add_mesh(stage, "/World/Ground/Pad", disc_mesh(16.0, 96, 0.02, 6.0), gravel)
    add_mesh(stage, "/World/Ground/Road", strip_mesh((-14.0, -4.0), (-420.0, -160.0), 5.0, 0.015, 6.0), gravel)

    # light
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(4200.0)
    sun.CreateAngleAttr(0.53)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.90))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(52.0, 0.0, 128.0))
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateIntensityAttr(520.0)
    if hdri:
        dome.CreateTextureFileAttr(hdri)
        dome.CreateTextureFormatAttr("latlong")
    else:
        dome.CreateColorAttr(Gf.Vec3f(0.55, 0.70, 0.95))

    stage.GetRootLayer().Save()
    gt = {"turbine": {"hub_center": spec.hub_center().round(3).tolist(), "hub_height": spec.hub_height,
                      "rotor_radius": spec.rotor_radius, "blade_azimuths_deg": list(spec.blade_azimuths_deg),
                      "tower_height": spec.tower_height},
          "defects": defects_json(defects)}
    import json
    (out / "defects_ground_truth.json").write_text(json.dumps(gt, indent=2))
    return {"usd": str(out / "turbine_scene.usd"), "ground_truth": str(out / "defects_ground_truth.json"),
            "spec": spec, "defects": defects}
