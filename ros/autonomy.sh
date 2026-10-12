#!/usr/bin/env bash
# Inspection stack, run inside the x500-turbine-inspection container (network host) against the Isaac Sim side:
#   PX4 SITL + MicroXRCEAgent -> px4_tf_pub (PX4 EKF pose -> TF + /px4_odom)
#   RTAB-Map (RGB-D map from the forward camera, PX4 odometry)
#   inspection_mission (route -> PX4 offboard setpoints, photo requests to the simulator)
set +e
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTDDS_BUILTIN_TRANSPORTS=UDPv4
X=/x500
CAM=/quadrotor/camera
log() { echo "[inspection] $*"; }

log "MicroXRCEAgent (udp 8888)"
MicroXRCEAgent udp4 -p 8888 > /tmp/agent.log 2>&1 &

log "PX4 SITL (gazebo-classic_x500 airframe, MAVLink to Isaac Sim on tcp 4560)"
( mkdir -p /tmp/px4run && cd /tmp/px4run && \
  PX4_SIM_MODEL=gazebo-classic_x500 /opt/PX4-Autopilot/build/px4_sitl_default/bin/px4 \
    /opt/PX4-Autopilot/ROMFS/px4fmu_common/ -s /opt/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/rcS -i 0 -d \
    > /tmp/px4.log 2>&1 ) &
sleep 14

log "px4_tf_pub (PX4 EKF -> /px4_odom + TF odom->base_link)"
/usr/bin/python3 -u $X/scripts/px4_tf_pub.py --ros-args -p use_sim_time:=true > /tmp/px4_tf.log 2>&1 &
ros2 run tf2_ros static_transform_publisher --x 0.2 --y 0.0 --z 0.0 --qx -0.5 --qy 0.5 --qz -0.5 --qw 0.5 \
  --frame-id base_link --child-frame-id camera_optical_link --ros-args -p use_sim_time:=true > /tmp/cam_tf.log 2>&1 &
ros2 run tf2_ros static_transform_publisher --x 0.0 --y 0.0 --z 0.12 \
  --frame-id base_link --child-frame-id lidar --ros-args -p use_sim_time:=true > /tmp/lidar_tf.log 2>&1 &
sleep 2

if [ "${RTABMAP:-1}" = "1" ]; then
  log "RTAB-Map (RGB-D map from the forward camera, PX4 odometry)"
  ros2 launch rtabmap_launch rtabmap.launch.py \
    args:="--delete_db_on_start --Grid/RangeMax 20 --Grid/CellSize 0.2 --Rtabmap/DetectionRate 1" \
    visual_odometry:="false" odom_topic:="/px4_odom" frame_id:="base_link" odom_frame_id:="odom" map_frame_id:="map" \
    rgb_topic:="$CAM/rgb" depth_topic:="$CAM/depth" camera_info_topic:="$CAM/camera_info" \
    approx_sync:="true" approx_sync_max_interval:="0.06" use_sim_time:="true" qos:="2" \
    rviz:="false" rtabmap_viz:="false" publish_tf_map:="true" > /tmp/rtabmap.log 2>&1 &
  sleep 4
fi

if [ "${BAG:-1}" = "1" ]; then
  log "recording odometry, progress, map and lidar topics to /out/bag"
  rm -rf /out/bag
  ros2 bag record -o /out/bag --use-sim-time /px4_odom /inspection/wp_index /inspection/progress \
    /rtabmap/cloud_map /lidar/points /tf /tf_static > /tmp/bag.log 2>&1 &
  BAG_PID=$!
fi

# stop the recorder cleanly (bag index) when the mission ends or the container is stopped
finish() {
  log "stopping the recorder"
  [ -n "${BAG_PID:-}" ] && kill -TERM $BAG_PID 2>/dev/null   # background jobs of a script ignore SIGINT
  for _ in $(seq 60); do
    { [ -n "${BAG_PID:-}" ] && kill -0 $BAG_PID 2>/dev/null; } || break
    sleep 1
  done
  exit 0
}
trap finish INT TERM

log "inspection_mission: ranges=${RANGES:-2-21,118-189} speed=${SPEED:-2.5}"
/usr/bin/python3 -u $X/scripts/inspection_mission.py --ros-args -p use_sim_time:=true \
  -p waypoints:=/plan/waypoints.json -p ranges:="${RANGES:-2-21,118-189}" -p speed:=${SPEED:-2.5} \
  -p home_x:=${HOME_X:-10.0} -p home_y:=${HOME_Y:--10.0} -p out_dir:=/out &
wait $!
log "mission finished"
finish
