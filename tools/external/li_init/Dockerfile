# LI-Init (hku-mars/LiDAR_IMU_Init) as an external baseline for Calibrex.
#
# LI-Init is GPL-2.0.  It is cloned into this image at a pinned commit and run
# as a separate process; Calibrex only reads the result file it writes, so no
# GPL code enters src/calibrex (see docs/concepts/license_boundaries.md).
FROM ros:noetic-perception-focal

ARG LI_INIT_COMMIT=66b157afa30bf6637d7f9defa6ddba46f417000b

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libceres-dev libgoogle-glog-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /catkin_ws/src
# Only the livox_ros_driver message definitions are needed; the Livox SDK is not.
COPY livox_ros_driver /catkin_ws/src/livox_ros_driver
RUN git clone https://github.com/hku-mars/LiDAR_IMU_Init.git \
    && git -C LiDAR_IMU_Init checkout ${LI_INIT_COMMIT}

WORKDIR /catkin_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make -j4 -DCMAKE_BUILD_TYPE=Release"

COPY run_li_init.sh /run_li_init.sh
RUN chmod +x /run_li_init.sh
ENTRYPOINT ["/run_li_init.sh"]
