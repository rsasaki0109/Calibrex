# Camera–LiDAR: fixed joint estimator on additional spent recordings

Freeze the joint three-frame assisted-input setup from
[the previous experiment](camera_lidar_joint_assisted.md), then apply it to two
additional already-spent KITTI raw drives: 2011-09-26 0009 and 0011. These are the
next two eligible same-day IDs after the existing 0005 pilot; they were selected
without inspecting images or calibration outcomes. Both appear in the historical
55-drive development inventory. They are new to this experiment, not fresh or
independent evaluation data.

The [plan](../assets/expanded_joint_20261029/plan.json) and
[schema](../assets/expanded_joint_20261029/plan.schema.json) pin source, archive
identity, frames and the unchanged two perturbation signs:

| Drive | Archive frame count | `floor((N-1)*q/4)`, q=1,2,3 |
| --- | ---: | --- |
| 0009 | 447 | 111, 223, 334 |
| 0011 | 233 | 58, 116, 174 |

Acquire only those left/right images and clouds, complete timestamp files and
OXTS samples from frame `f-2` through `f+2` for each selected frame. Retain original
frame IDs and timestamps. Require the selected downloads to fit within 64 MiB
per drive, validate ZIP CRC and SHA-256, and pin S3 size/ETag before payload reads.
Do not interpolate outside local pose coverage or substitute another frame if
preparation fails. The seed-only input directories are transport for the joint
runner, **not** evidence that single-frame calibration ran on these drives.

The only frontend code change supports sparse acquisition: select a cloud by
its exact original KITTI timestamp rather than its ordinal position in the bag.
The old selector returned byte-identical payloads to the new selector for all
six previously evaluated frames. Tests reject missing and duplicate timestamps.
Motion interpolation, stereo settings, solver defaults, seeds, four-CPU ceilings,
180-second timeouts and strict `<0.5 deg / <0.20 m` recovery criteria are unchanged.
The comparison has four new joint cases per method. Require all three native
frames to support the result, and retain failed cases in the denominator.

Report per-case rotation and translation, recovery, and change from the seed.
Do not tune settings, switch seeds or select only successful drives after the
run. The native arm still has stereo depth while Koide does not; both share
OXTS-assisted motion. This is not evidence of monocular superiority. Even after
successful execution, current assisted-input coverage is only 4 of 55 drives,
not an independent-family or fresh-data audit.
