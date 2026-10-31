#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
exec "$@"
