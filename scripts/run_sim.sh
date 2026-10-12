#!/usr/bin/env bash
# Start Isaac Sim 6.1 with the turbine scene, the X500 (Pegasus PX4 MAVLink backend) and its ROS 2 sensors.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_PY="${ISAAC_PY:-$HOME/IsaacLab3/.venv/bin/python}"
ISAAC_PKG="${ISAAC_PKG:-$(dirname "$(dirname "$ISAAC_PY")")/lib/python3.12/site-packages/isaacsim}"
export PEGASUS_EXT_PATH="${PEGASUS_EXT_PATH:-$HOME/PegasusSimulator_isaac61/extensions}"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_fastrtps_cpp FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export ROS_DOMAIN_ID="${X500_ROS_DOMAIN:-0}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${ISAAC_PKG}/exts/isaacsim.ros2.core/jazzy/lib"
export PX4_AUTOLAUNCH="${PX4_AUTOLAUNCH:-0}" PX4_ROBOT="${PX4_ROBOT:-x500}" PX4_AIRFRAME="${PX4_AIRFRAME:-gazebo-classic_x500}"
export REALTIME="${REALTIME:-1}" ISAAC_HEADLESS="${ISAAC_HEADLESS:-0}" ISAAC_CAMERA="${ISAAC_CAMERA:-1}"
exec "$ISAAC_PY" "$HERE/sim_turbine.py"
