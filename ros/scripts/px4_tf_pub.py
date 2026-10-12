#!/usr/bin/env python3
"""
Publish the drone's pose from PX4's (robust) EKF as TF odom(ENU)->base_link AND a
nav_msgs/Odometry on /px4_odom, replacing RTAB-Map's fragile visual odometry as the pose
source for nvblox + the planner. (RTAB-Map RGB-D VO froze under aggressive motion - the
Phase 1.1b lesson; PX4's estimate stays valid.)

Frames: PX4 is NED/FRD; ROS odom is ENU and base_link is FLU. Position NED->ENU is
(E=y, N=x, U=-z); attitude FRD->NED is converted to FLU->ENU with the mavros static rotations.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation

PX4_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
# /px4_odom published RELIABLE so RELIABLE subscribers (Ego-Planner FSM + grid_map) match; a
# RELIABLE publisher also satisfies BEST_EFFORT subscribers (the bridge, the older planners).
ODOM_PUB_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
NED_ENU = Rotation.from_euler("xyz", [np.pi, 0.0, np.pi / 2.0])
ENU_NED = NED_ENU.inv()
FLU_FRD = Rotation.from_euler("xyz", [np.pi, 0.0, 0.0])


class Px4TfPub(Node):
    def __init__(self):
        super().__init__("px4_tf_pub")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.br = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, "/px4_odom", ODOM_PUB_QOS)
        self.pos = None
        self.q_ned = None
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self._on_pos, PX4_QOS)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self._on_att, PX4_QOS)
        self.create_timer(0.05, self.tick)
        self.get_logger().info("px4_tf_pub: PX4 NED -> odom(ENU)->base_link TF + /px4_odom")

    def _on_pos(self, m):
        self.pos = m

    def _on_att(self, m):
        self.q_ned = m.q   # [w, x, y, z], FRD body -> NED world

    def _enu_pose(self):
        ex, ey, ez = float(self.pos.y), float(self.pos.x), float(-self.pos.z)   # NED -> ENU position
        if self.q_ned is not None and np.isfinite(self.q_ned[0]):
            q = Rotation.from_quat([self.q_ned[1], self.q_ned[2], self.q_ned[3], self.q_ned[0]])  # x,y,z,w
            qe = (ENU_NED * q * FLU_FRD).as_quat()   # FLU body -> ENU world, (x,y,z,w)
        else:
            qe = [0.0, 0.0, 0.0, 1.0]
        return (ex, ey, ez), qe

    def tick(self):
        if self.pos is None:
            return
        (ex, ey, ez), qe = self._enu_pose()
        now = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = ex, ey, ez
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = \
            float(qe[0]), float(qe[1]), float(qe[2]), float(qe[3])
        self.br.sendTransform(t)

        o = Odometry()
        o.header.stamp = now
        o.header.frame_id = self.odom_frame
        o.child_frame_id = self.base_frame
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = ex, ey, ez
        o.pose.pose.orientation.x, o.pose.pose.orientation.y, o.pose.pose.orientation.z, o.pose.pose.orientation.w = \
            float(qe[0]), float(qe[1]), float(qe[2]), float(qe[3])
        self.odom_pub.publish(o)


def main():
    rclpy.init()
    n = Px4TfPub()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
