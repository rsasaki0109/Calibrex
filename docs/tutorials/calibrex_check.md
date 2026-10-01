# Check a deployed calibration (`calibrex check`)

Status: **Phase B** (imu-lidar, lidar-lidar, camera-imu). The command reads the
calibration deployed on a robot, works out which sensor pairs a bag can audit,
and, without `--plan`, runs the existing native estimator of each wired pair and
judges the deployed (candidate) transform against it: `pass`, `warn`, `fail` or
`inconclusive` per pair, with the per-axis numbers behind it. Other pairs stay
`skipped` with `method_not_wired`.

```bash
calibrex check my_bag/ --plan --output check.json                 # fast: what could be checked
calibrex check my_bag/ --tf rig.urdf --output check.json          # run the estimators, judge
calibrex check my_bag/ --pairs imu-lidar,camera-imu --max-duration-s 120 --output check.json
calibrex check my_bag/ --tf rig.urdf --vehicle-frame base_link --plan  # ground vehicle
calibrex validate check.json
```

`--plan` is unchanged and runs no estimator. Without `--output` the estimator
artifacts go to `./calibrex_check_evidence/`.

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
| `not_selected` | a wired pair that `--pairs` or `--camera` left out |
| `unsupported_sensor` | the sensor cannot be read by the estimator: a LiDAR that is not `PointCloud2`, a LiDAR without a per-point time field (scans cannot be deskewed), a compressed camera image |
| `missing_intrinsics` | `camera-imu` needs intrinsics: a Kalibr camchain given with `--tf`, or a `CameraInfo` topic next to the image topic |
| `missing_dependency` | `camera-imu` needs OpenCV (`pip install 'calibrex[opencv]'`) |

A run can also leave a pair `inconclusive` with one of `estimator_error` (the
estimator raised; the message is recorded and the other pairs still run),
`estimator_failed` (no solution, or the estimator failed its own held-out check,
so its estimate is not a reliable yardstick) or `no_judgeable_axes` (every axis
was unobservable).

## Vehicle pairs are opt-in

`lidar-vehicle` and `imu-vehicle` rest on non-holonomic ground-vehicle motion.
A `base_link` frame alone does not say the rig is a ground vehicle (on a
hand-held rig it is just the IMU body frame), so these pairs are skipped with
`no_vehicle_frame` unless you pass `--vehicle-frame <frame>`. If that frame is
not in the tree the pairs are skipped with `frame_not_in_tree`. An automatic
ground-vehicle motion test may replace the flag in Phase C. The artifact
records `vehicle_frame` (absent when not given).

## Verdicts (Phase B)

| Pair | Estimator | Compared transform | Judged axes |
| --- | --- | --- | --- |
| `imu-lidar` | `imu-lidar` rotation (gyro against LiDAR odometry) and, unless `--no-imu-lidar-translation`, lever arm (accelerometer) | `T_lidar_imu` (the planner's `T_imu_lidar` inverted) | roll, pitch, yaw, x, y, z |
| `lidar-lidar` | map registration, started at the candidate | `T_first_second` | roll, pitch, yaw, x, y, z |
| `camera-imu` | targetless camera rotation against the gyro | `T_cam_imu` | roll, pitch, yaw |

Per axis `i` the estimator reports an estimate, a standard deviation `std_i`
(the larger of its analytic and jackknife values) and whether the axis is
`estimated` or `unobservable`. Only estimated axes are judged; the rest are
listed as `unchecked_axes` with the reason.

* `delta_i` is candidate minus estimate. For rotations it is the component of
  `rotvec(R_candidate R_estimate^T)` about the axes of the compared transform's
  parent frame, in degrees; this is the convention the estimators report `std`
  in. For translations it is the difference of the components, in metres.
* `tolerance_i = max(k * std_i, floor)`, with `k = --sigma-k` (3),
  rotation floor `--rotation-floor-deg` (0.5 deg) and translation floor
  `--translation-floor-m` (0.02 m). Each axis records which one dominated
  (`tolerance_source`).
* Axis: `pass` if `|delta_i| <= tolerance_i`, `fail` if `|delta_i| > 2 * tolerance_i`,
  otherwise `warn`.
* Pair: `fail` if any judged axis fails, else `warn` if any warns, else `pass`.
  `inconclusive` when no axis could be judged, the estimator did not solve, or
  it failed its own held-out check. The estimator's clock offset is reported
  (`time_offset`) but not judged.
* Overall: the worst pair verdict in the order `fail > warn > inconclusive > pass`.
  Skipped pairs do not count. If no pair ran the overall verdict is
  `inconclusive`, never a silent pass. The exit status is 1 when the overall
  verdict is at least `--fail-on` (default `fail`; `never` always exits 0).

The estimator is told nothing about the candidate except, for `lidar-lidar`, its
starting value (the solver is local, so a candidate outside its capture range
shows up as a failed held-out check, not as a large `delta`); for the other two
pairs the estimate is independent of the candidate. The estimator's own
`reference` fields carry the candidate, so each evidence artifact also holds its
`error_to_reference`.

### What error could this bag detect?

Every judged axis records a detection-power self-test that re-solves nothing.
If the candidate were wrong by a further `e` on that axis, either sign, the
error would be `delta_i + e`; it is flagged (beyond tolerance) for both signs
exactly when `e > tolerance_i + |delta_i|`. That bound is recorded as
`detectable_error` (degrees or metres), and for rotation axes
`detects_perturbation` says whether a `--detection-probe-deg` (1) error would
have been flagged. A bag whose axes carry `detectable_error` of 0.8 deg catches
1 deg errors; one with 2 deg cannot, and a `pass` from it only means "no
error larger than that was seen".

### Runtime controls

`--pairs a,b` restricts pairs; `--camera TOPIC` restricts `camera-imu` to one
image topic (or camera frame); `--max-duration-s S` analyses the first `S`
seconds of each sensor stream (images, scans, IMU samples within that window;
the IMU is still read in full but only the covered span is used);
`--acceleration-unit` states the IMU unit for the lever arm (`g` for Livox
`livox_ros_driver2` recordings). Progress is printed to stderr.

LiDAR support: `PointCloud2` with a per-point time field. The field (`offset_time`,
`t`, `time`, `timestamp`, ...) and its meaning (offset in seconds, absolute
seconds as Hesai writes it, absolute nanoseconds as Livox writes it) are
detected from the first message by comparing its values with the header stamp. A
LiDAR without such a field is `unsupported_sensor`; deskewing needs it, and an
undeskewed scan on a moving platform would be a biased yardstick. Livox
`CustomMsg` is not read yet.

## Artifact

`--output` writes `slac.calibration_check/v0.1`: the bag path with a digest
(`metadata.yaml` in full; storage files by name, size and first 64 MiB, the
same scope used by other bag-based artifacts), candidate sources, topics with
roles and frames, the frame tree, per-pair records with the candidate
transform, status and reason, a summary, and provenance. A run adds, as
optional fields: `options` (thresholds and runtime controls), `evidence_dir`,
`overall_verdict`, and per pair `compared_transform`, `axes`, `unchecked_axes`,
`time_offset`, `estimator*`, `runtime_s` and `evidence`.

Each estimator's own schema'd artifact (`slac.imu_lidar_rotation/v0.1`,
`slac.imu_lidar_translation/v0.1`, `slac.lidar_lidar_extrinsic/v0.1`) is written
to `<output stem>_evidence/<pair>_<sensors>[_role].yaml`, referenced from the
pair by relative path and SHA-256 (`evidence`; `evidence_artifact` repeats the
first path). Those files hold the per-DoF estimates, jackknife and known-bad
controls, and policy status.

No HTML report yet: the existing HTML helpers render calibration results, not
check artifacts.
