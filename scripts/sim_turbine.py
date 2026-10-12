#!/usr/bin/env python3
"""Wind turbine inspection scene for Isaac Sim 6.1: the X500 (Pegasus PX4 MAVLink backend) with its sensors.

Publishes to ROS 2 (domain 0):
  /clock
  /quadrotor/camera/{rgb,depth,camera_info}   forward RGB-D camera (RealSense D455-like, 640x480, 87 deg HFOV)
  /gimbal/{rgb,depth,camera_info}             gimbal inspection camera (1280x720, 50 deg HFOV)
  /lidar/points                               RTX lidar point cloud (sensor frame "lidar")

The gimbal is stabilised: every frame it sits under the drone body (on top of it for shots pitched more than
20 deg up, like the upper gimbal of inspection drones) and looks at the target of the current route
waypoint (assets/plan/waypoints.json). The mission node writes the current waypoint index and photo requests into
RUN_DIR; the sim answers each photo request by saving the gimbal image, its depth and its exact pose.

Env:
  RUN_DIR=<dir>          shared with the mission container (wp_index.txt, capture_request.json, photos/)
  RECORD=<file.npz>      per-frame drone and gimbal poses, for replay renders
  X500_USD=<usd>         vehicle (default assets/x500/x500.usda, built by scripts/build_x500_usd.py)
  SCENE_USD, WAYPOINTS   scene and route (defaults assets/scene/turbine_scene.usd, assets/plan/waypoints.json)
  SPAWN_X, SPAWN_Y, SPAWN_Z, ISAAC_HEADLESS, CHASE_CAM, VIEWPORT_OFF, LIDAR=0|1
"""
import json
import math
import os
import time

import numpy as np
from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_HEADLESS", "0") == "1"
simulation_app = SimulationApp({"headless": HEADLESS})

import carb  # noqa: E402
import omni.graph.core as og  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("isaacsim.ros2.bridge")
enable_extension("isaacsim.core.nodes")
for _ in range(30):
    simulation_app.update()

_PEG_EXT = os.environ.get("PEGASUS_EXT_PATH", "")
if _PEG_EXT:
    import omni.kit.app
    _mgr = omni.kit.app.get_app().get_extension_manager()
    _mgr.add_path(_PEG_EXT)
    for _ in range(20):
        simulation_app.update()
    _mgr.set_extension_enabled_immediate("pegasus.simulator", True)
    for _ in range(40):
        simulation_app.update()

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_USD = os.environ.get("SCENE_USD", os.path.join(ROOT, "assets", "scene", "turbine_scene.usd"))
WAYPOINTS = os.environ.get("WAYPOINTS", os.path.join(ROOT, "assets", "plan", "waypoints.json"))
X500_USD = os.path.expanduser(os.environ.get("X500_USD", os.path.join(ROOT, "assets", "x500", "x500.usda")))
RUN_DIR = os.environ.get("RUN_DIR", "")
RECORD = os.environ.get("RECORD", "")
PX4_DIR = os.path.expanduser(os.environ.get("PX4_DIR", "~/px4_jazzy_ws/PX4-Autopilot"))
SPAWN = [float(os.environ.get("SPAWN_X", "10.0")), float(os.environ.get("SPAWN_Y", "-10.0")),
         float(os.environ.get("SPAWN_Z", "0.3"))]

CAM_W, CAM_H, CAM_FOV = 640, 480, 87.0
GIM_W, GIM_H = int(os.environ.get("GIMBAL_W", "1280")), int(os.environ.get("GIMBAL_H", "720"))
GIM_FOV = float(os.environ.get("GIMBAL_HFOV", "50.0"))
GIMBAL_OFFSET = np.array([0.0, 0.0, -0.12])   # lower gimbal under the body, metres in the world frame
GIMBAL_TOP = np.array([0.0, 0.0, 0.16])      # upper gimbal on top of the body, for upward shots
GIMBAL_TOP_PITCH = math.radians(20.0)       # planned pitch above which the upper gimbal is used
APERTURE = 20.955


def build_clock_graph():
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/World/ClockGraph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.SET_VALUES: [("Context.inputs:domain_id", 0), ("PublishClock.inputs:topicName", "/clock")],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("SimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ],
        },
    )


def set_intrinsics(cam_prim, width, height, hfov_deg, near=0.05, far=2000.0):
    cam = UsdGeom.Camera(cam_prim)
    cam.CreateHorizontalApertureAttr().Set(APERTURE)
    cam.CreateVerticalApertureAttr().Set(APERTURE * height / width)
    cam.CreateFocalLengthAttr().Set(APERTURE / (2.0 * math.tan(math.radians(hfov_deg) / 2.0)))
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(near, far))


def camera_publishers(graph, rp_path, topics, frame_id, skip=0):
    """ROS2CameraHelper nodes (rgb, depth, camera_info) on an existing render product."""
    keys = og.Controller.Keys
    nodes = [("OnTick", "omni.graph.action.OnPlaybackTick"), ("Context", "isaacsim.ros2.bridge.ROS2Context"),
             ("rgb", "isaacsim.ros2.bridge.ROS2CameraHelper"), ("depth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
             ("info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper")]
    values = [("Context.inputs:domain_id", 0)]
    connect = []
    for n, kind in (("rgb", "rgb"), ("depth", "depth"), ("info", None)):
        values += [(f"{n}.inputs:renderProductPath", rp_path), (f"{n}.inputs:topicName", topics[n]),
                   (f"{n}.inputs:frameId", frame_id), (f"{n}.inputs:frameSkipCount", skip)]
        if kind:
            values.append((f"{n}.inputs:type", kind))
        connect += [("OnTick.outputs:tick", f"{n}.inputs:execIn"), ("Context.outputs:context", f"{n}.inputs:context")]
    og.Controller.edit({"graph_path": graph, "evaluator_name": "execution"},
                       {keys.CREATE_NODES: nodes, keys.SET_VALUES: values, keys.CONNECT: connect})


def look_rotation(forward, up=(0.0, 0.0, 1.0)):
    """Rotation matrix (columns = camera x right, y up, z backward; USD camera looks along -z)."""
    f = np.asarray(forward, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, (1.0, 0.0, 0.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f], axis=1)


def set_pose(xform_op, R, p):
    M = Gf.Matrix4d(1.0)
    for i in range(3):
        M.SetRow(i, Gf.Vec4d(float(R[0, i]), float(R[1, i]), float(R[2, i]), 0.0))
    M.SetRow(3, Gf.Vec4d(float(p[0]), float(p[1]), float(p[2]), 1.0))
    xform_op.Set(M)


def add_realsense_visual(body_path):
    """Intel RealSense D455 model on the front mount (visual only), from the Isaac Sim asset library.
    REALSENSE_USD overrides the path, e.g. with a local copy of the asset pack."""
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(body_path + "/realsense").IsValid():
        return
    path = os.environ.get("REALSENSE_USD", "")
    if not path:
        try:
            from isaacsim.storage.native import get_assets_root_path
            root = get_assets_root_path()
        except Exception:
            root = None
        if not root:
            print("[veh] Isaac Sim asset library not reachable, no RealSense model", flush=True)
            return
        path = root + "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
    rs = stage.DefinePrim(body_path + "/realsense", "Xform")
    rs.GetReferences().AddReference(path)
    UsdGeom.Xformable(rs).AddTranslateOp().Set(Gf.Vec3d(0.13, 0.0, -0.05))
    print(f"[veh] RealSense model: {path}", flush=True)


def read_text(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def main():
    if not os.path.exists(X500_USD):
        raise SystemExit(f"[veh] {X500_USD} not found: build it with scripts/build_x500_usd.py (see README)")
    stage = omni.usd.get_context().get_stage()
    pg = PegasusInterface()
    pg._world = World(**pg._world_settings)
    world = pg.world

    scene = stage.DefinePrim("/World/Scene", "Xform")
    scene.GetReferences().AddReference(SCENE_USD)
    world.scene.add_default_ground_plane(z_position=0.0)
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/defaultGroundPlane")):
        if prim.IsA(UsdGeom.Mesh):
            UsdGeom.Imageable(prim).MakeInvisible()
    print(f"[world] {os.path.basename(SCENE_USD)} + invisible collision ground", flush=True)

    config = MultirotorConfig()
    config.backends = [PX4MavlinkBackend(PX4MavlinkBackendConfig({
        "vehicle_id": 0, "px4_autolaunch": False, "px4_dir": PX4_DIR,
        "px4_vehicle_model": "gazebo-classic_x500", "enable_lockstep": False}))]
    Multirotor("/World/quadrotor", X500_USD, 0, SPAWN,
               Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(), config=config)
    add_realsense_visual("/World/quadrotor/body")
    world.reset()
    build_clock_graph()

    # forward RGB-D camera (as in the X500 asset), for RTAB-Map
    fwd_path = "/World/quadrotor/body/camera"
    from isaacsim.sensors.camera import Camera
    fwd_cam = Camera(prim_path=fwd_path, frequency=30.0, resolution=(CAM_W, CAM_H))
    fwd_cam.initialize()
    fwd_cam.set_local_pose(np.array([0.2, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), camera_axes="world")
    set_intrinsics(stage.GetPrimAtPath(fwd_path), CAM_W, CAM_H, CAM_FOV)
    fwd_rp = rep.create.render_product(fwd_path, (CAM_W, CAM_H))
    camera_publishers("/World/ForwardCamGraph", fwd_rp.path,
                      {"rgb": "/quadrotor/camera/rgb", "depth": "/quadrotor/camera/depth",
                       "info": "/quadrotor/camera/camera_info"}, "camera_optical_link")

    # stabilised gimbal camera (free prim, posed every frame)
    gim = UsdGeom.Camera.Define(stage, "/World/GimbalCam")
    set_intrinsics(gim.GetPrim(), GIM_W, GIM_H, GIM_FOV, near=0.1, far=3000.0)
    gim_op = UsdGeom.Xformable(gim.GetPrim()).AddTransformOp()
    gim_rp = rep.create.render_product("/World/GimbalCam", (GIM_W, GIM_H))
    camera_publishers("/World/GimbalCamGraph", gim_rp.path,
                      {"rgb": "/gimbal/rgb", "depth": "/gimbal/depth", "info": "/gimbal/camera_info"},
                      "gimbal_optical", skip=int(os.environ.get("GIMBAL_SKIP", "5")))
    ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    ann_rgb.attach([gim_rp.path])
    ann_depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    ann_depth.attach([gim_rp.path])
    fx = (GIM_W / 2.0) / math.tan(math.radians(GIM_FOV) / 2.0)
    intrinsics = {"width": GIM_W, "height": GIM_H, "fx": fx, "fy": fx, "cx": GIM_W / 2.0, "cy": GIM_H / 2.0}

    # RTX lidar on top of the body
    if os.environ.get("LIDAR", "1") == "1":
        try:
            from isaacsim.sensors.experimental.rtx import Lidar
            lidar = Lidar.create("/World/quadrotor/body/lidar", config=os.environ.get("LIDAR_CONFIG", "OS1"),
                                 variant=os.environ.get("LIDAR_VARIANT", "OS1_REV6_32ch10hz1024res"),
                                 translations=[[0.0, 0.0, 0.12]])
            lrp = rep.create.render_product(lidar.paths[0], (128, 128))
            keys = og.Controller.Keys
            og.Controller.edit({"graph_path": "/World/LidarGraph", "evaluator_name": "execution"}, {
                keys.CREATE_NODES: [("OnTick", "omni.graph.action.OnPlaybackTick"),
                                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                                    ("pcl", "isaacsim.ros2.bridge.ROS2RtxLidarHelper")],
                keys.SET_VALUES: [("Context.inputs:domain_id", 0), ("pcl.inputs:renderProductPath", lrp.path),
                                  ("pcl.inputs:topicName", "/lidar/points"), ("pcl.inputs:type", "point_cloud"),
                                  ("pcl.inputs:frameId", "lidar"), ("pcl.inputs:fullScan", True)],
                keys.CONNECT: [("OnTick.outputs:tick", "pcl.inputs:execIn"),
                               ("Context.outputs:context", "pcl.inputs:context")]})
            print(f"[lidar] {lidar.paths[0]} -> /lidar/points", flush=True)
        except Exception as e:
            print(f"[lidar] disabled: {e}", flush=True)

    waypoints = json.load(open(WAYPOINTS))
    if RUN_DIR:
        os.makedirs(os.path.join(RUN_DIR, "photos"), exist_ok=True)
        json.dump(intrinsics, open(os.path.join(RUN_DIR, "gimbal_intrinsics.json"), "w"))

    chase_op = None
    if not HEADLESS and os.environ.get("CHASE_CAM", "1") == "1":
        chase = UsdGeom.Camera.Define(stage, "/World/ChaseCam")
        chase.CreateFocalLengthAttr(18.0)
        chase.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1.0e5))
        chase_op = UsdGeom.Xformable(chase.GetPrim()).AddTransformOp()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    print("[run] simulation playing", flush=True)

    body = stage.GetPrimAtPath("/World/quadrotor/body")
    step_dt = world.get_rendering_dt()      # one world.step(render=True) advances the sim by this much
    wall0 = time.time()
    frame = 0
    gim_dir = np.array([1.0, 0.0, 0.0])
    rec = {"t": [], "pos": [], "quat": [], "gim_pos": [], "gim_quat": [], "wp": []}
    last_seq = None
    pending = None
    viewport_set = False
    while simulation_app.is_running():
        world.step(render=True)
        if os.environ.get("REALTIME", "1") == "1":
            slack = (wall0 + (frame + 1) * step_dt) - time.time()
            if slack > 0:
                time.sleep(slack)
        frame += 1
        wm = omni.usd.get_world_transform_matrix(body)
        p = np.array(wm.ExtractTranslation())
        q = wm.ExtractRotationQuat()
        fwd = np.array(wm.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)))
        yaw = math.atan2(fwd[1], fwd[0])

        # gimbal aim: target of the current waypoint, else level along the body heading with the planned pitch
        k = read_text(os.path.join(RUN_DIR, "wp_index.txt")) if RUN_DIR and frame % 3 == 0 else None
        if k is not None:
            try:
                wp_idx = int(k)
            except ValueError:
                wp_idx = -1
        else:
            wp_idx = rec["wp"][-1] if rec["wp"] else -1
        up_shot = 0 <= wp_idx < len(waypoints) and math.radians(waypoints[wp_idx]["gimbal_pitch_deg"]) > GIMBAL_TOP_PITCH
        gpos = p + (GIMBAL_TOP if up_shot else GIMBAL_OFFSET)
        if 0 <= wp_idx < len(waypoints) and waypoints[wp_idx].get("target"):
            want = np.array(waypoints[wp_idx]["target"]) - gpos
        else:
            pitch = math.radians(waypoints[wp_idx]["gimbal_pitch_deg"]) if 0 <= wp_idx < len(waypoints) else 0.0
            want = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
        want /= np.linalg.norm(want)
        gim_dir = gim_dir + 0.25 * (want - gim_dir)     # gimbal slew
        gim_dir /= np.linalg.norm(gim_dir)
        Rg = look_rotation(gim_dir)
        set_pose(gim_op, Rg, gpos)

        if chase_op is not None:
            if not viewport_set and frame == 30:
                from omni.kit.viewport.utility import get_active_viewport
                get_active_viewport().camera_path = "/World/ChaseCam"
                viewport_set = True
            back = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            eye = p - 7.0 * back + np.array([0.0, 0.0, 2.5])
            set_pose(chase_op, look_rotation(p - eye), eye)

        if RECORD:
            rec["t"].append(frame * step_dt)
            rec["pos"].append(p)
            rec["quat"].append([q.GetReal(), *q.GetImaginary()])
            rec["gim_pos"].append(gpos)
            rec["gim_quat"].append(Rotation.from_matrix(Rg).as_quat(scalar_first=True))
            rec["wp"].append(wp_idx)
            if frame % 600 == 0:
                np.savez(RECORD, **{k: np.asarray(v) for k, v in rec.items()})
        else:
            rec["wp"] = [wp_idx]

        # photo requests from the mission node: aim, let the renderer settle a few frames, then save
        if RUN_DIR and frame % 5 == 0 and pending is None:
            req = read_text(os.path.join(RUN_DIR, "capture_request.json"))
            if req:
                try:
                    r = json.loads(req)
                    if r.get("seq") != last_seq:
                        pending = (r, frame + 12)
                except json.JSONDecodeError:
                    pass
        if pending is not None and frame >= pending[1]:
            r = pending[0]
            pending = None
            last_seq = r.get("seq")
            rgb = ann_rgb.get_data()
            depth = ann_depth.get_data()
            name = f"wp{int(r['index']):03d}"
            out = os.path.join(RUN_DIR, "photos")
            if rgb is not None and np.asarray(rgb).size:
                Image.fromarray(np.asarray(rgb)[..., :3]).save(os.path.join(out, name + ".jpg"), quality=93)
                np.save(os.path.join(out, name + "_depth.npy"), np.asarray(depth, dtype=np.float32).astype(np.float16))
                T = np.eye(4)
                T[:3, :3] = Rg                       # USD camera axes: x right, y up, looks along -z
                T[:3, 3] = gpos
                json.dump({"index": int(r["index"]), "seq": r.get("seq"), "sim_time": frame * step_dt,
                           "camera_to_world_usd": T.tolist(), "drone_pos": p.tolist(),
                           "drone_quat_wxyz": [q.GetReal(), *q.GetImaginary()], "intrinsics": intrinsics},
                          open(os.path.join(out, name + ".json"), "w"))
            with open(os.path.join(RUN_DIR, "capture_done.json"), "w") as f:
                json.dump({"seq": r.get("seq"), "index": r.get("index")}, f)

    if RECORD:
        np.savez(RECORD, **{k: np.asarray(v) for k, v in rec.items()})
    carb.log_warn("closing")
    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
