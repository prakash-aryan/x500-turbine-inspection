# PX4 SITL + ROS 2 Jazzy + RTAB-Map, used by scripts/run_autonomy.sh.
#   docker build -t x500-turbine-inspection:latest docker/
FROM ros:jazzy-ros-base
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates gnupg lsb-release sudo cmake ninja-build build-essential ccache \
        python3-pip python3-colcon-common-extensions python3-rosdep python3-numpy python3-scipy python3-empy \
        libasio-dev libtinyxml2-dev libeigen3-dev \
        ros-jazzy-rtabmap-ros ros-jazzy-rtabmap-launch ros-jazzy-tf2-tools ros-jazzy-tf2-ros \
        ros-jazzy-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*

# uXRCE-DDS agent (PX4 <-> ROS 2)
RUN git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git /opt/Micro-XRCE-DDS-Agent \
    && cd /opt/Micro-XRCE-DDS-Agent && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DUAGENT_USE_SYSTEM_FASTDDS=OFF && make -j"$(nproc)" && make install && ldconfig

# px4_msgs
RUN mkdir -p /root/ros2_ws/src && git clone -b release/1.16 https://github.com/PX4/px4_msgs.git /root/ros2_ws/src/px4_msgs \
    && source /opt/ros/jazzy/setup.bash && cd /root/ros2_ws \
    && colcon build --packages-select px4_msgs --cmake-args -DCMAKE_BUILD_TYPE=Release

# PX4 SITL v1.16.2 with the X500 airframe used by the Isaac Sim side
RUN git clone -b v1.16.2 --recursive https://github.com/PX4/PX4-Autopilot.git /opt/PX4-Autopilot \
    && cd /opt/PX4-Autopilot && bash ./Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools \
    && make px4_sitl_default && rm -rf build/px4_sitl_default/.px4-sitl
COPY airframes/4500_gazebo-classic_x500 /opt/PX4-Autopilot/build/px4_sitl_default/etc/init.d-posix/airframes/4500_gazebo-classic_x500
COPY airframes/4500_gazebo-classic_x500 /opt/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/4500_gazebo-classic_x500
RUN chmod +x /opt/PX4-Autopilot/build/px4_sitl_default/etc/init.d-posix/airframes/4500_gazebo-classic_x500

WORKDIR /root
CMD ["bash"]
