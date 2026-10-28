#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 run direct_visual_lidar_calibration calibrate /data/matrix-20261028/2011_09_28_0002-0000000281-negative --background --auto_quit
