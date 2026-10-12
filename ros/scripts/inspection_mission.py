#!/usr/bin/env python3
"""Fly the turbine inspection route with PX4 offboard position setpoints and take a gimbal photo at each photo
waypoint.

Waypoints (assets/plan/waypoints.json) are in the world frame; PX4's local frame is ENU from the take-off point,
so local = world - home + ground, where ground is the EKF position while standing on the pad before arming (the EKF
height origin is not exactly the pad height). The setpoint is a carrot that moves along each leg at `speed` with velocity feedforward,
and the yaw turns towards the waypoint's heading. At a photo waypoint the drone holds still, then writes a photo
request into the shared run directory and waits for the simulator to save the gimbal image, depth and pose.

Topics: /inspection/wp_index (std_msgs/Int32), /inspection/progress (std_msgs/String, JSON).
Files in out_dir: wp_index.txt, capture_request.json (written), capture_done.json (read), telemetry.csv.
"""
import json
import math
import os
import time

import rclpy
from nav_msgs.msg import Odometry
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String

PX4_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
NAN = float("nan")


def parse_ranges(text, n):
    idx = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        lo, hi = int(a), int(b or a)
        idx += [i for i in range(lo, hi + 1) if 0 <= i < n]
    return idx


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class InspectionMission(Node):
    def __init__(self):
        super().__init__("inspection_mission")
        p = self.declare_parameter
        p("waypoints", "/plan/waypoints.json")
        p("ranges", "2-21,118-189")       # waypoint indices to fly (climb, return and landing are added)
        p("home_x", 10.0)
        p("home_y", -10.0)
        p("home_z", 0.25)
        p("speed", 2.5)
        p("photo_tolerance", 0.6)
        p("pass_tolerance", 1.5)
        p("settle_time", 1.0)
        p("capture_timeout", 6.0)
        p("out_dir", "/out")
        g = lambda k: self.get_parameter(k).value  # noqa: E731
        self.wps = json.load(open(g("waypoints")))
        n = len(self.wps)
        body = parse_ranges(g("ranges"), n)
        climb = [i for i, w in enumerate(self.wps) if w["segment"] == "climb"]
        tail = [i for i, w in enumerate(self.wps) if w["segment"] in ("return", "land")]
        # leave the last inspected part by its planned transit (straight out from the rotor), then the final
        # descent transit, return and landing
        exit_ = []
        j = body[-1] + 1 if body else len(self.wps)
        while j < n and self.wps[j]["segment"] == "transit":
            exit_.append(j)
            j += 1
        last_transit = [i for i, w in enumerate(self.wps) if w["segment"] == "transit"][-1:]
        self.route = climb + body + [i for i in exit_[:1] + last_transit if i not in body] + tail
        self.home = (g("home_x"), g("home_y"), g("home_z"))
        self.speed = float(g("speed"))
        self.tol_photo, self.tol_pass = float(g("photo_tolerance")), float(g("pass_tolerance"))
        self.settle_time, self.capture_timeout = float(g("settle_time")), float(g("capture_timeout"))
        self.out = g("out_dir")
        os.makedirs(self.out, exist_ok=True)

        self.pose = None
        self.ground = None                # EKF position on the pad, averaged before arming
        self.vel = (0.0, 0.0, 0.0)
        self.k = 0
        self.armed = False
        self.ri = 0                       # position in self.route
        self.sp = None                    # carrot (local ENU)
        self.sp_yaw = 0.0
        self.state = "INIT"
        self.state_t = 0.0
        self.seq = 0
        self.photos = 0
        self.t0 = time.time()
        self.tel = open(os.path.join(self.out, "telemetry.csv"), "w")
        self.tel.write("t,state,wp,x,y,z,sp_x,sp_y,sp_z\n")

        self.ocm = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_QOS)
        self.tsp = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", PX4_QOS)
        self.vcmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", PX4_QOS)
        self.wp_pub = self.create_publisher(Int32, "/inspection/wp_index", 10)
        self.prog_pub = self.create_publisher(String, "/inspection/progress", 10)
        self.create_subscription(Odometry, "/px4_odom", self._on_odom, PX4_QOS)
        self.create_timer(0.05, self.tick)
        self.set_wp_index(self.route[0])
        self.get_logger().info(f"route: {len(self.route)} waypoints "
                               f"({sum(1 for i in self.route if self.wps[i]['photo'])} photos)")

    # ---- helpers -----------------------------------------------------------------------------------------------
    def local(self, i):
        w = self.wps[i]
        g = self.ground or (0.0, 0.0, 0.0)
        return (w["x"] - self.home[0] + g[0], w["y"] - self.home[1] + g[1], w["z"] - self.home[2] + g[2])

    def yaw_of(self, i):
        return math.radians(self.wps[i]["yaw_deg"])

    def _on_odom(self, m):
        p = m.pose.pose.position
        new = (p.x, p.y, p.z)
        if self.pose is not None:
            self.vel = tuple(0.8 * v + 0.2 * (a - b) / 0.05 for v, a, b in zip(self.vel, new, self.pose))
        if not self.armed:
            self.ground = new if self.ground is None else tuple(0.9 * g + 0.1 * n for g, n in zip(self.ground, new))
        self.pose = new

    def now(self):
        return time.time() - self.t0

    def cmd(self, command, p1=0.0, p2=0.0):
        m = VehicleCommand()
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        m.command, m.param1, m.param2 = command, p1, p2
        m.target_system, m.target_component, m.source_system, m.source_component = 1, 1, 255, 1
        m.from_external = True
        self.vcmd.publish(m)

    def publish_sp(self, pos_enu, vel_enu, yaw_enu):
        t = int(self.get_clock().now().nanoseconds / 1000)
        o = OffboardControlMode()
        o.timestamp, o.position, o.velocity = t, True, vel_enu is not None
        self.ocm.publish(o)
        s = TrajectorySetpoint()
        s.timestamp = t
        s.position = [float(pos_enu[1]), float(pos_enu[0]), float(-pos_enu[2])]
        s.velocity = [float(vel_enu[1]), float(vel_enu[0]), float(-vel_enu[2])] if vel_enu else [NAN] * 3
        s.acceleration = [NAN] * 3
        s.yaw = float(wrap(math.pi / 2.0 - yaw_enu))
        self.tsp.publish(s)

    def set_wp_index(self, i):
        with open(os.path.join(self.out, "wp_index.txt"), "w") as f:
            f.write(str(i))
        self.wp_pub.publish(Int32(data=int(i)))

    def go(self, state):
        self.state, self.state_t = state, self.now()

    # ---- main loop ---------------------------------------------------------------------------------------------
    def tick(self):
        if self.pose is None:
            o = OffboardControlMode()
            o.timestamp, o.position = int(self.get_clock().now().nanoseconds / 1000), True
            self.ocm.publish(o)
            return
        self.k += 1
        x, y, z = self.pose
        if self.sp is None:
            self.sp = [x, y, z]
        if self.state == "LAND":
            if z < (self.ground or (0.0, 0.0, 0.0))[2] + 0.3:
                self.landed_k = getattr(self, "landed_k", 0) + 1
                if self.landed_k > 100:          # 5 s on the ground at 20 Hz
                    self.get_logger().info(f"LANDED, {self.photos} photos; mission finished")
                    raise SystemExit
            return
        if self.state in ("INIT", "CLIMB") and z < 0.6 and (self.k == 30 or (self.k > 30 and self.k % 60 == 0)):
            self.cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
            self.cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            if not self.armed:
                self.armed = True
                self.get_logger().info(f"arming + offboard, EKF pad position {tuple(round(v, 2) for v in self.ground)}")
                self.go("CLIMB")

        i = self.route[self.ri]
        target = self.local(i)
        vel = None
        if self.state in ("INIT", "CLIMB", "MOVE"):
            # carrot along the leg towards the current waypoint
            d = [t - s for t, s in zip(target, self.sp)]
            dist = math.sqrt(sum(v * v for v in d))
            step = self.speed * 0.05 if self.state == "MOVE" else 0.05 * 1.5
            if dist > step:
                self.sp = [s + v / dist * step for s, v in zip(self.sp, d)]
                vel = tuple(v / dist * (step / 0.05) for v in d)
            else:
                self.sp = list(target)
            self.sp_yaw += max(-0.04, min(0.04, wrap(self.yaw_of(i) - self.sp_yaw)))
        self.publish_sp(self.sp, vel, self.sp_yaw)

        err = math.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2 + (z - target[2]) ** 2)
        photo = self.wps[i]["photo"]
        tol = self.tol_photo if photo else self.tol_pass
        if self.state == "CLIMB" and self.armed and err < tol and self.sp == list(target):
            self.get_logger().info("at climb point, starting route")
            self.next_wp()
        elif self.state == "MOVE" and self.sp == list(target) and err < tol:
            if photo:
                self.go("SETTLE")
            else:
                self.next_wp()
        elif self.state == "SETTLE":
            speed = math.sqrt(sum(v * v for v in self.vel))
            if self.now() - self.state_t > self.settle_time and speed < 0.3:
                self.seq += 1
                with open(os.path.join(self.out, "capture_request.json"), "w") as f:
                    json.dump({"index": i, "seq": self.seq}, f)
                self.go("CAPTURE")
        elif self.state == "CAPTURE":
            done = {}
            try:
                done = json.load(open(os.path.join(self.out, "capture_done.json")))
            except (OSError, json.JSONDecodeError):
                pass
            if done.get("seq") == self.seq:
                self.photos += 1
                self.next_wp()
            elif self.now() - self.state_t > self.capture_timeout:
                self.get_logger().warn(f"photo at wp {i} timed out")
                self.next_wp()

        if self.k % 10 == 0:
            self.tel.write(f"{self.now():.2f},{self.state},{i},{x:.3f},{y:.3f},{z:.3f},"
                           f"{self.sp[0]:.3f},{self.sp[1]:.3f},{self.sp[2]:.3f}\n")
        if self.k % 40 == 0:
            self.get_logger().info(f"[{self.state}] wp {i} ({self.ri + 1}/{len(self.route)}) "
                                   f"pos=({x:.1f},{y:.1f},{z:.1f}) err={err:.2f} photos={self.photos}")
            self.prog_pub.publish(String(data=json.dumps(
                {"wp": i, "segment": self.wps[i]["segment"], "done": self.ri, "total": len(self.route),
                 "photos": self.photos})))

    def next_wp(self):
        if self.ri + 1 >= len(self.route):
            self.get_logger().info(f"ROUTE DONE, {self.photos} photos; landing")
            self.cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.tel.flush()
            self.go("LAND")
            return
        self.ri += 1
        i = self.route[self.ri]
        self.set_wp_index(i)
        if self.wps[i]["segment"] == "land":
            self.get_logger().info(f"ROUTE DONE, {self.photos} photos; landing")
            self.cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.tel.flush()
            self.go("LAND")
            return
        self.go("MOVE")


def main():
    rclpy.init()
    n = InspectionMission()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        n.tel.close()
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
