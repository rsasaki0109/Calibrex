import hashlib,json,sqlite3
from pathlib import Path
from calibrex.data.rosbag2_writer import Rosbag2Writer
root=Path('/workspace/work/external-comparison-20261028')
src=Path('/workspace/work/camera-lidar-recovery-20261027/bags/2011_09_26_0005/2011_09_26_0005_0.db3')
db=sqlite3.connect(f'file:{src}?mode=ro',uri=True)
rows=[]; digests=[]
with Rosbag2Writer(root/'input'/'spent-0005-three-frames') as writer:
 for tid,name,typ in db.execute('select id,name,type from topics'):
  if name not in ['/camera/image_raw','/camera/camera_info','/velodyne_points']:continue
  writer.add_topic(name,typ)
  for timestamp,data in db.execute('select timestamp,data from messages where topic_id=? order by timestamp,id limit 3',(tid,)):
   rows.append((timestamp,name,data));digests.append(dict(topic=name,timestamp=timestamp,sha256=hashlib.sha256(data).hexdigest()))
 assert len(rows)==9,(len(rows),list(db.execute('select name from topics')))
 for timestamp,name,data in sorted(rows):writer.write(name,timestamp,data)
(root/'smoke-input.json').write_text(json.dumps(dict(scope='spent input execution smoke only',source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),messages=digests),indent=2))
