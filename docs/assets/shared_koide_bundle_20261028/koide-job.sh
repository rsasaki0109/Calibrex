#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
mkdir -p /data/paired-output
ros2 run direct_visual_lidar_calibration preprocess /data/paired-input /data/paired-output -a -d --camera_model plumb_bob --camera_intrinsics 721.5377,721.5377,609.5593,172.854 --camera_distortion_coeffs 0,0,0,0,0
python3 - <<'PY'
import json
from pathlib import Path
p=Path('/data/paired-output/calib.json')
a=json.loads(p.read_text())
a['results']={'init_T_lidar_camera':json.loads(Path('/data/calib.json').read_text())['results']['init_T_lidar_camera']}
p.write_text(json.dumps(a,indent=2))
Path('/data/paired-output/calib-seeded.json').write_text(json.dumps(a,indent=2))
PY
ros2 run direct_visual_lidar_calibration calibrate /data/paired-output --background --auto_quit
