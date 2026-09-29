#!/bin/bash
# Usage: run_li_init.sh BAG CONFIG_YAML OUTPUT_DIR [TIMEOUT_S]
# Plays BAG into LI-Init headless with CONFIG_YAML and copies its result files.
set -euo pipefail
BAG="$1"; CONFIG="$2"; OUT="$3"; TIMEOUT="${4:-3600}"
set +u
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash
set -u
PKG=/catkin_ws/src/LiDAR_IMU_Init
mkdir -p "$OUT" "$PKG/result"
rm -f "$PKG/result/"*
roscore > "$OUT/roscore.log" 2>&1 &
sleep 3
rosparam load "$CONFIG"
rosparam set point_filter_num 2
rosparam set max_iteration 5
rosparam set cube_side_length 2000
rosrun lidar_imu_init li_init > "$OUT/li_init.log" 2>&1 &
sleep 3
timeout "$TIMEOUT" rosbag play --quiet "$BAG" > "$OUT/rosbag_play.log" 2>&1 || true
sleep 10
cp -r "$PKG/result/." "$OUT/" 2>/dev/null || true
kill %2 %1 2>/dev/null || true
ls -la "$OUT"
