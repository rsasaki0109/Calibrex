#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
python3 - <<'PY'
import json
from pathlib import Path
p=Path('/data/aligned-stereo/calib.json')
a=json.loads(p.read_text())
a['results']={'init_T_lidar_camera':json.loads(Path('/data/calib.json').read_text())['results']['init_T_lidar_camera']}
p.write_text(json.dumps(a,indent=2))
Path('/data/aligned-stereo/calib-seeded.json').write_text(json.dumps(a,indent=2))
PY
ros2 run direct_visual_lidar_calibration calibrate /data/aligned-stereo --background --auto_quit
