# Check a deployed calibration (`calibrex check`)

Status: **Phase A, plan only.** The command reads the calibration deployed on a
robot, works out which sensor pairs a bag can audit, and says why the others
cannot be. No pair solver runs yet; later phases will run the existing native
evidence methods per pair and add pass / warn / fail / inconclusive verdicts.

```bash
calibrex check my_bag/ --plan --output check.json
calibrex check my_bag/ --tf rig.urdf --vehicle-frame base_link --plan  # ground vehicle
calibrex validate check.json
```

Without `--plan` the command behaves the same and prints a notice that solvers
are not wired yet.

## Candidate calibration

Candidate extrinsics come from the bag's `/tf_static` (the latest transform per
child frame; a child that appears under two different parents is an error) and
from any number of `--tf FILE` options. A file overrides `/tf_static` for the
same child frame (explicit user input wins), and the override is listed in the report. Two files that
define the same frame differently are rejected. Every source is recorded with
its SHA-256 (for `/tf_static`, the hash of the raw messages read).

| `--tf` format | Notes |
| --- | --- |
| URDF | Fixed joints only (`origin xyz` and `rpy`, ROS fixed-axis convention). Run xacro first. |
| `slac.check_frames/v0.1` YAML | `frames: [{name, parent, translation_m, rotation_quat_xyzw}]` as `T_parent_name`, plus optional `topic_frames` and `role_frames` hints. |
| Kalibr camchain-imucam | `camN.T_cam_imu` is inverted to `T_imu_cam`. The IMU frame is named `imu`; camera frames take their `rostopic` as name, which also maps the topic to its frame. |
| RTK-SLAM `calib.yaml` | Camera as Kalibr; `lidarN.T_lidar_imu` is read as the repository reads it (LiDAR points into the IMU frame); the GNSS antenna CAD offset becomes a `gnss_antenna` frame with identity rotation. `base_center` is not turned into a vehicle frame. |
| Hilti `lidar_calibration.yaml` | `sensors: {name: {parent, extrinsics: {quaternion, translation}}}`. The quaternion order is x, y, z, w (the identity `imu` entry is `[0, 0, 0, 1]`). |

Transforms follow `T_parent_child`. A pair named `A-B` carries `T_A_B` composed
through the tree.

## Topic roles and frames

Topics are classified by message type: `lidar` (PointCloud2, Livox CustomMsg),
`imu`, `camera` (Image, CompressedImage), `gnss` (NavSatFix), `odometry`,
`twist` and `tf_static`. Odometry topics are labelled `wheel` or `ins` from
tokens in their name, otherwise `unknown`.

Each sensor topic is mapped to a tree frame in this order: `--frame-map
TOPIC=FRAME`; the first message's `header.frame_id` when it is in the tree (for
odometry, `child_frame_id`); a topic hint from a calibration file; a role
default from a calibration file (only when the role's remaining topics are one
sensor). Topics that stay unmapped are listed. Several topics that map to the
same frame (for example a PointCloud2 and a CustomMsg of one LiDAR) are one
sensor.

## Pairs

Candidates: `imu-lidar`, `lidar-lidar`, `camera-imu`, `camera-focal`,
`gnss-lidar`, `gnss-imu`, `lidar-vehicle`, `imu-vehicle`, `ins-lidar`,
`lidar-wheel_odometry`. A pair is `planned` when both frames are in the tree and
connected and the topics exist. Otherwise it is `skipped` with a reason code:

| Code | Meaning |
| --- | --- |
| `missing_topic` | no topic for a required role (or fewer than two LiDARs for `lidar-lidar`) |
| `no_candidate_calibration` | the bag has no `/tf_static` and no `--tf` was given |
| `no_vehicle_frame` | `--vehicle-frame` was not given, so vehicle pairs are not checked |
| `frame_not_in_tree` | a sensor topic maps to no frame, or to a frame the tree lacks |
| `frames_not_connected` | the two frames are in different trees |
| `method_not_wired` | no check method exists for the pair yet |

## Vehicle pairs are opt-in

`lidar-vehicle` and `imu-vehicle` rest on non-holonomic ground-vehicle motion.
A `base_link` frame alone does not say the rig is a ground vehicle (on a
hand-held rig it is just the IMU body frame), so these pairs are skipped with
`no_vehicle_frame` unless you pass `--vehicle-frame <frame>`. If that frame is
not in the tree the pairs are skipped with `frame_not_in_tree`. An automatic
ground-vehicle motion test may replace the flag in Phase C. The artifact
records `vehicle_frame` (absent when not given).

## Artifact

`--output` writes `slac.calibration_check/v0.1`: the bag path with a digest
(`metadata.yaml` in full; storage files by name, size and first 64 MiB, the
same scope used by other bag-based artifacts), candidate sources, topics with
roles and frames, the frame tree, per-pair records with the candidate
transform, status and reason, a summary, and provenance. Fields for evidence
artifacts and verdicts are reserved and empty in Phase A.
