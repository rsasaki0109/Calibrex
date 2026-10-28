import json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
root=Path('/workspace/work/external-comparison-20261028')
p=root/'preprocessed/calib.json'
shutil.copyfile(p,root/'calib-unseeded.json')
r=json.loads(p.read_text())
meta=json.load(open('/workspace/work/camera-lidar-recovery-20261027/bags/2011_09_26_0005/calibrex_conversion.json'))
reference=np.array(meta['camera']['t_cam_velo'])
start=reference.copy();start[:3,:3]=Rotation.from_euler('xyz',[1,-1,0.5],degrees=True).as_matrix() @ start[:3,:3];start[:3,3]+=[0.1,-0.05,0.05]
inv=np.linalg.inv(start)
r['results']={'init_T_lidar_camera':[*inv[:3,3].tolist(),*Rotation.from_matrix(inv[:3,:3]).as_quat().tolist()]}
(root/'calib-seeded.json').write_text(json.dumps(r,indent=2))
(root/'seed.json').write_text(json.dumps(dict(scope='development runtime smoke, not benchmark',reference_T_camera_lidar=reference.tolist(),initial_T_camera_lidar=start.tolist(),rotation_injection_xyz_deg=[1,-1,0.5],translation_injection_m=[0.1,-0.05,0.05]),indent=2))
# Replace a generated container-owned file through its user-owned parent.
p.unlink();shutil.copyfile(root/'calib-seeded.json',p)
