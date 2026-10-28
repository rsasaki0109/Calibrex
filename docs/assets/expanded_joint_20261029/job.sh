#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 run direct_visual_lidar_calibration calibrate /data/expanded-joint-20261029/2011_09_26_0011-negative --background --auto_quit
