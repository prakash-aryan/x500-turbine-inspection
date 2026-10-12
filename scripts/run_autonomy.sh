#!/usr/bin/env bash
# Start the PX4 + ROS 2 inspection container (needs the Isaac Sim side from run_sim.sh running).
#   RUN_DIR   shared run directory (same as for run_sim.sh): photo requests, photos, telemetry, rosbag
#   IMAGE     container image (default x500-turbine-inspection:latest)
#   RANGES    waypoint index ranges to fly, e.g. "2-21,118-189" (tower face + blade 0) or "2-337" (full route)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
mkdir -p "$RUN_DIR"
docker rm -f x500_turbine >/dev/null 2>&1 || true
exec docker run --rm --name x500_turbine --network host --ipc=host \
  -e RANGES="${RANGES:-2-21,118-189}" -e SPEED="${SPEED:-2.5}" -e RTABMAP="${RTABMAP:-1}" -e BAG="${BAG:-1}" \
  -v "$HERE/ros":/x500:ro -v "$HERE/assets/plan":/plan:ro -v "$RUN_DIR":/out \
  "${IMAGE:-x500-turbine-inspection:latest}" bash /x500/autonomy.sh
