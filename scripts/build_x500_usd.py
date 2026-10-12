"""Build the Holybro X500 V2 vehicle model (USD) for Isaac Sim and the Pegasus Simulator.

    uv run --with "usd-core==25.5.*" --with numpy --with trimesh --with pycollada --with cascadio \
        python scripts/build_x500_usd.py --px4 ~/PX4-gazebo-models [--step ~/Downloads/x500v2-frame.step]

Writes assets/x500/x500.usda (plus assets/x500/meshes/*.usdc), which scripts/sim_city.py loads.

Meshes:
  * PX4's gz x500 model (BSD-3-Clause), from a PX4-gazebo-models or PX4-Autopilot checkout: the propellers
    always, the frame and motors when no CAD frame is given.
  * Optional: the Holybro X500 V2 frame CAD (x500v2-frame.step from Holybro's download page). It is tessellated
    with cascadio and coloured by part name: moulded plastic clamps and mounts blue, foam feet black, the rest
    black carbon. --cad-obj takes an OBJ already converted from that STEP instead.
Physics: PX4 gz x500_base values (body 2.0 kg, inertia 0.021667 / 0.021667 / 0.04, arms 0.174 m, rotors
0.016077 kg) and the PX4 quad-X rotor order. Prim layout for Pegasus: /x500/body, /x500/rotor0..3 with
joint0..3, propeller visuals at /x500/rotorK/visual (centred on the hub, so they can be spun about Z).
All materials are UsdPreviewSurface. The RealSense D455 model is added at run time by sim_city.py from the
Isaac Sim asset library.
"""
import argparse
import os
import re
import tempfile

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--px4", default=os.environ.get("PX4_MODELS", ""),
                help="PX4-gazebo-models or PX4-Autopilot checkout (contains x500_base/meshes)")
ap.add_argument("--step", default=os.environ.get("X500_STEP", ""), help="Holybro x500v2-frame.step (optional)")
ap.add_argument("--cad-obj", default=os.environ.get("X500_CAD_OBJ", ""), help="OBJ converted from that STEP")
ap.add_argument("--out", default=os.path.join(HERE, "assets", "x500"))
args = ap.parse_args()

ARM, ROTOR_Z = 0.174, 0.06
ROTORS = [("rotor0", ARM, -ARM, True), ("rotor1", -ARM, ARM, True),     # (name, x, y, ccw), PX4 quad-X
          ("rotor2", ARM, ARM, False), ("rotor3", -ARM, -ARM, False)]
PROP_SCALE = 0.85
COLORS = {  # diffuse, roughness, metallic
    "carbon": ((0.030, 0.030, 0.035), 0.5, 0.0),
    "blue": ((0.050, 0.420, 0.850), 0.5, 0.0),
    "foam": ((0.025, 0.025, 0.025), 0.9, 0.0),
    "prop": ((0.020, 0.020, 0.025), 0.5, 0.0),
    "motor": ((0.060, 0.060, 0.065), 0.35, 0.6),
}


def find_meshes(root):
    for sub in ("models/x500_base/meshes", "Tools/simulation/gz/models/x500_base/meshes", "x500_base/meshes", ""):
        d = os.path.join(os.path.expanduser(root), sub)
        if os.path.exists(os.path.join(d, "1345_prop_ccw.stl")):
            return d
    raise SystemExit(f"x500_base/meshes not found under '{root}' (pass --px4 with a PX4-gazebo-models checkout)")


def material(stage, path, name):
    rgb, rough, metal = COLORS[name]
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def add_mesh(stage, path, pts, tris, normals=None, mat=None):
    """Triangle mesh; normals per vertex (len == len(pts)) or flat face normals when None."""
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.reshape(-1).astype(np.int32)))
    if normals is None:
        fn = np.cross(pts[tris[:, 1]] - pts[tris[:, 0]], pts[tris[:, 2]] - pts[tris[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(fn, 3, axis=0).astype(np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    else:
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*pts.min(0).tolist()), Gf.Vec3f(*pts.max(0).tolist())]))
    if mat is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


def mesh_layer(path, parts):
    """Write {name: (points, tris, normals, colour)} as one .usdc layer whose default prim is /part."""
    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/part")
    stage.SetDefaultPrim(root.GetPrim())
    mats = {}
    for name, (pts, tris, nrm, col) in parts.items():
        if col not in mats:
            mats[col] = material(stage, f"/part/Looks/{col}", col)
        add_mesh(stage, f"/part/{name}", pts, tris, nrm, mats[col])
    stage.GetRootLayer().Save()


def load_trimesh(path):
    """Mesh in metres (trimesh applies the COLLADA unit); all geometries of a scene merged."""
    sc = trimesh.load(path, force="scene")
    m = trimesh.util.concatenate(sc.dump()) if len(sc.geometry) > 1 else sc.dump()[0]
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def read_cad_obj(path):
    """OBJ from cascadio (mm, Y-up, one group per CAD part) -> {colour: (points m Z-up, tris, normals)}."""
    with open(path) as f:
        lines = f.read().splitlines()
    v, vn, faces, colour_of_face, colour = [], [], [], [], "carbon"
    for ln in lines:
        if ln.startswith("v "):
            v.append(ln[2:])
        elif ln.startswith("vn "):
            vn.append(ln[3:])
        elif ln.startswith("f "):
            faces.append(ln[2:])
            colour_of_face.append(colour)
        elif ln.startswith("g "):
            name = ln[2:].upper()
            colour = "blue" if "MUJU" in name else "foam" if "EVA" in name else "carbon"
    V = np.array(" ".join(v).split(), np.float64).reshape(-1, 3)
    N = np.array(" ".join(vn).split(), np.float64).reshape(-1, 3) if vn else None
    tok = np.array(re.split(r"[ /]+", " ".join(faces).strip()), np.int64)
    per = len(tok) // len(faces)                       # 3 (v), 6 (v//n) or 9 (v/t/n) indices per triangle
    F = tok.reshape(len(faces), per)
    tris = F[:, ::per // 3] - 1
    ntri = F[:, per // 3 - 1::per // 3] - 1 if (N is not None and per > 3) else None
    rot = lambda a: np.stack([a[:, 0], -a[:, 2], a[:, 1]], 1)   # Y-up -> Z-up (+90 deg about X)
    V, N = rot(V) * 0.001, (rot(N) if N is not None else None)
    out = {}
    cof = np.array(colour_of_face)
    for col in ("carbon", "blue", "foam"):
        sel = cof == col
        if not sel.any():
            continue
        used, inv = np.unique(tris[sel].reshape(-1), return_inverse=True)
        nrm = None
        if ntri is not None and np.array_equal(ntri[sel], tris[sel]):
            nrm = N[used]
        out[col] = (V[used], inv.reshape(-1, 3), nrm)
    return out


def main():
    mesh_dir = find_meshes(args.px4) if args.px4 else find_meshes(os.path.expanduser("~/PX4-gazebo-models"))
    os.makedirs(os.path.join(args.out, "meshes"), exist_ok=True)

    # propellers: hub moved to the origin, scaled for appearance (thrust comes from the PX4 rotor model)
    for kind in ("ccw", "cw"):
        p, t = load_trimesh(os.path.join(mesh_dir, f"1345_prop_{kind}.stl"))
        c = (p.min(0) + p.max(0)) / 2
        p = (p - [c[0], c[1], 0.0]) * PROP_SCALE + [0.0, 0.0, 0.005]
        mesh_layer(os.path.join(args.out, "meshes", f"prop_{kind}.usdc"), {"prop": (p, t, None, "prop")})

    cad_obj = args.cad_obj
    if args.step and not cad_obj:
        import cascadio
        cad_obj = os.path.join(tempfile.mkdtemp(), "x500v2_frame.obj")
        print(f"[build] tessellating {args.step}")
        cascadio.step_to_obj(os.path.expanduser(args.step), cad_obj, tol_linear=0.6, tol_angular=0.5)
    if cad_obj:
        parts = {k: (p, t, n, k) for k, (p, t, n) in read_cad_obj(os.path.expanduser(cad_obj)).items()}
        print(f"[build] CAD frame: {sum(len(t) for _, t, _, _ in parts.values())} triangles")
    else:
        p, t = load_trimesh(os.path.join(mesh_dir, "NXP-HGD-CF.dae"))
        p = np.stack([-p[:, 0], -p[:, 1], p[:, 2] + 0.025], 1)          # rotated 180 deg about Z, raised
        parts = {"carbon": (p, t, None, "carbon")}
        bp, bt = load_trimesh(os.path.join(mesh_dir, "5010Base.dae"))
        a = np.radians(-25.8)
        rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        for k, (mx, my) in enumerate([(ARM, ARM), (-ARM, ARM), (ARM, -ARM), (-ARM, -ARM)]):
            parts[f"motor{k}"] = (bp @ rz.T + [mx, my, 0.032], bt, None, "motor")
        print("[build] PX4 frame and motors (no CAD frame given)")
    mesh_layer(os.path.join(args.out, "meshes", "frame.usdc"), parts)
    if not cad_obj:
        bp, bt = load_trimesh(os.path.join(mesh_dir, "5010Bell.dae"))
        mesh_layer(os.path.join(args.out, "meshes", "motor_bell.usdc"), {"bell": (bp, bt, None, "motor")})

    out = os.path.join(args.out, "x500.usda")
    if os.path.exists(out):
        os.remove(out)
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/x500")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    root.GetPrim().AddAppliedSchema("PhysxArticulationAPI")

    body = UsdGeom.Xform.Define(stage, "/x500/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    m = UsdPhysics.MassAPI.Apply(body)
    m.CreateMassAttr(2.0)
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(0.021667, 0.021667, 0.04))
    m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    UsdGeom.Xform.Define(stage, "/x500/body/visual")
    stage.DefinePrim("/x500/body/visual/mesh", "Xform").GetReferences().AddReference("./meshes/frame.usdc")

    # invisible colliders: a box over the arm footprint down to the feet, and one disc per propeller
    col = UsdGeom.Cube.Define(stage, "/x500/body/collision")
    col.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(col).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.07))
    UsdGeom.Xformable(col).AddScaleOp().Set(Gf.Vec3f(0.36, 0.36, 0.30))
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    col.CreatePurposeAttr(UsdGeom.Tokens.guide)
    col.MakeInvisible()

    for i, (name, x, y, ccw) in enumerate(ROTORS):
        r = UsdGeom.Xform.Define(stage, f"/x500/{name}")
        r.AddTranslateOp().Set(Gf.Vec3d(x, y, ROTOR_Z))
        UsdPhysics.RigidBodyAPI.Apply(r.GetPrim())
        rm = UsdPhysics.MassAPI.Apply(r.GetPrim())
        rm.CreateMassAttr(0.016077)
        rm.CreateDiagonalInertiaAttr(Gf.Vec3f(3.85e-07, 2.66e-05, 2.65e-05))
        pc = UsdGeom.Cylinder.Define(stage, f"/x500/{name}/prop_collision")
        pc.CreateRadiusAttr(0.14)
        pc.CreateHeightAttr(0.03)
        pc.CreateAxisAttr("Z")
        UsdPhysics.CollisionAPI.Apply(pc.GetPrim())
        pc.CreatePurposeAttr(UsdGeom.Tokens.guide)
        pc.MakeInvisible()
        if not cad_obj:      # the CAD frame already has the motors
            stage.DefinePrim(f"/x500/{name}/bell", "Xform").GetReferences().AddReference("./meshes/motor_bell.usdc")
        UsdGeom.Xform.Define(stage, f"/x500/{name}/visual")
        stage.DefinePrim(f"/x500/{name}/visual/mesh", "Xform").GetReferences().AddReference(
            f"./meshes/prop_{'ccw' if ccw else 'cw'}.usdc")
        j = UsdPhysics.RevoluteJoint.Define(stage, f"/x500/{name}/joint{i}")
        j.CreateBody0Rel().SetTargets(["/x500/body"])
        j.CreateBody1Rel().SetTargets([f"/x500/{name}"])
        j.CreateAxisAttr("Z")
        j.CreateLocalPos0Attr(Gf.Vec3f(x, y, ROTOR_Z))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")   # free spinning; Pegasus spins it for visuals
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateDampingAttr(0.0)
        drive.CreateStiffnessAttr(0.0)
    stage.GetRootLayer().Save()
    print(f"[build] wrote {out}")


if __name__ == "__main__":
    main()
