# Check a deployed calibration (`calibrex check`)

Status: **Phase C2** (imu-lidar, lidar-lidar, camera-imu, the GNSS pairs gnss-lidar and
gnss-imu, and the opt-in vehicle pairs lidar-vehicle, imu-vehicle, ins-lidar,
lidar-wheel_odometry). The command reads the
calibration deployed on a robot, works out which sensor pairs a bag can audit,
and, without `--plan`, runs the existing native estimator of each wired pair and
judges the deployed (candidate) transform against it: `pass`, `warn`, `fail` or
`inconclusive` per pair, with the per-axis numbers behind it. The camera-focal
pair stays `skipped` with `method_not_wired`.

```bash
calibrex check my_bag/ --plan --output check.json                 # fast: what could be checked
calibrex check my_bag/ --tf rig.urdf --output check.json          # run the estimators, judge
calibrex check my_bag/ --pairs imu-lidar,camera-imu --max-duration-s 120 --output check.json
calibrex check my_bag/ --tf rig.urdf --vehicle-frame base_link --plan  # ground vehicle
calibrex validate check.json
calibrex check my_bag/ --tf rig_v2.urdf --output check_v2.json   # same bag, other candidate: cache hit
calibrex check my_bag/ --tf rig.urdf --no-cache --output check.json  # recompute everything
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
`twist` and `tf_static`. Odometry and twist topics are labelled `wheel` or `ins`
from tokens in their name, otherwise `unknown`; `--topic-kind TOPIC=wheel|ins`
(repeatable) sets the kind explicitly, is recorded in the topic's notes and in
`options.topic_kinds`, and must name an odometry or twist topic of the bag.

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
| `missing_dependency` | `camera-imu` needs OpenCV (`pip install 'calibrex[opencv]'`); `gnss-imu` needs this check's own `gnss-lidar` and `imu-lidar` results |

A run can also leave a pair `inconclusive` with one of `estimator_error` (the
estimator raised; the message is recorded and the other pairs still run),
`estimator_failed` (no solution, or the estimator failed its own held-out check,
so its estimate is not a reliable yardstick) or `no_judgeable_axes` (every axis
was unobservable).

## Vehicle pairs are opt-in

`lidar-vehicle`, `imu-vehicle`, `ins-lidar` and `lidar-wheel_odometry` rest on
ground-vehicle motion (`ins-lidar` needs only an INS, but is grouped with them).
A `base_link` frame alone does not say the rig is a ground vehicle (on a
hand-held rig it is just the IMU body frame), so `lidar-vehicle`, `imu-vehicle`
and `lidar-wheel_odometry` are skipped with `no_vehicle_frame` unless you pass
`--vehicle-frame <frame>`. If that frame is not in the tree the pairs are
skipped with `frame_not_in_tree`. The artifact records `vehicle_frame` (absent
when not given).

## Phase C1: vehicle pairs

| Pair | Needs in the bag | Estimator | Compared transform | Judged axes |
| --- | --- | --- | --- | --- |
| `lidar-vehicle` | a `PointCloud2` LiDAR | `slac.vehicle_frame_rotation` on LiDAR odometry (non-holonomic motion) | `T_vehicle_lidar` (planner `T_lidar_vehicle` inverted) | roll, pitch, yaw |
| `imu-vehicle` | an IMU in frame F and an `ins` Odometry/Twist topic expressed in F | same solver on the INS's own body-frame velocity and angular rate | `T_vehicle_imu` | roll, pitch, yaw |
| `ins-lidar` | an `ins` `nav_msgs/Odometry` (pose) and a `PointCloud2` LiDAR | trajectory hand-eye `slac.ins_lidar_hand_eye` | `T_ins_lidar` | roll, pitch, yaw, x, y (z only with a prior: unchecked) |
| `lidar-wheel_odometry` | a `PointCloud2` LiDAR and a `wheel` Odometry/Twist topic | `slac.lidar_wheel_odometry` (speed `linear.x`, yaw rate `angular.z`) | `T_wheel_lidar` | roll, pitch, yaw (lever, speed scale and clock offset are in the evidence, not judged) |

* **LiDAR odometry** is the one the KITTI runners use (`scan_to_scan_odometry`,
  default options, sweeps registered as rigid snapshots). A per-point time field
  is neither needed nor used, so KITTI-style clouds work; the three LiDAR pairs
  share one odometry pass per run. A silence of more than 1 s in the stream
  starts an independent segment: odometry is not chained across it, motions and
  blocks never span it. This is how several recordings in one bag are pooled.
* **`imu-vehicle` does not use the raw IMU.** A gyro and accelerometer carry no
  velocity; the estimator needs the INS's body-frame velocity and rate. It reads
  them from the `ins` topic's twist (Odometry twist is in `child_frame_id`), and
  the topic must map to the same frame as the IMU, otherwise the pair is skipped
  (`unsupported_sensor`; no INS topic at all is `missing_topic`).
* **`ins-lidar`** takes the pose track of an `ins` `nav_msgs/Odometry`; a twist
  cannot supply it (`unsupported_sensor`).
* An axis the drive does not constrain, or whose held-out known-bad control was
  not detected, is listed as unchecked and the pair is `partial`; a prior-only
  axis is unchecked as well. Every estimate is independent of the candidate, so
  the estimator artifacts are cached.
* The verdict rule is unchanged (`check/verdict.py`).

### Converting a KITTI raw drive

```bash
calibrex convert kitti-raw 2011_09_26_drive_0022_sync --calib-dir 2011_09_26 --output kitti_0022_bag
calibrex check kitti_0022_bag --vehicle-frame base_link --topic-kind /oxts/twist=wheel \
  --pairs lidar-vehicle,imu-vehicle,ins-lidar,lidar-wheel_odometry --output check.json
```

Several drives of one calibration may be given; they share one bag, with the
minutes between them as stream gaps. Topics and frames:

| Topic | Type | Frame | Content |
| --- | --- | --- | --- |
| `/velodyne_points` | `PointCloud2` | `velo_link` | x, y, z, intensity (float32). **No per-point time field**: KITTI raw has none and none is invented, so `imu-lidar` is `unsupported_sensor` |
| `/oxts/imu` | `Imu` | `imu_link` | OXTS body-frame `wx..wz`, `ax..az`; no orientation |
| `/oxts/fix` | `NavSatFix` | `imu_link` | OXTS lat/lon/alt. KITTI's GNSS solution is at the OXTS unit; the antenna lever arm is not published |
| `/oxts/odometry` | `Odometry` | `odom` to `imu_link` | the INS pose as in `kitti_oxts_pose_world_imu` (Mercator, origin at each drive's first packet) and, in the twist, the **body-frame** velocity and rate |
| `/oxts/twist` | `TwistStamped` | `base_link` | the same body-frame velocity and rate, as a wheel-odometry **proxy** (KITTI has no wheel odometry). By name it reads as an INS topic, so declare it with `--topic-kind /oxts/twist=wheel` |
| `/tf_static` (transient local) | `TFMessage` | | `base_link` to `imu_link` (identity); `imu_link` to `velo_link` = inverse of `calib_imu_to_velo.txt` |

KITTI's `vf, vl, vu` and `wf, wl, wu` are level-frame quantities
(level = `Ry(pitch) Rx(roll)` body); the converter rotates them back with each
packet's roll and pitch (`oxts_body_frame_motion`, the same function the KITTI
runners use) before declaring them in `imu_link`/`base_link`.

**`base_link` is defined as the OXTS/IMU frame**, KITTI's own convention (its
calibration files give sensors relative to the OXTS unit and no vehicle frame is
published). The vehicle pairs therefore report how far the sensor is from the
OXTS frame taken as the vehicle; the OXTS unit itself sits about 1 deg in roll
and 0.5 deg in pitch off the motion-defined vehicle frame
([KITTI LiDAR-vehicle](../benchmarks/kitti_lidar_vehicle.md)), so a candidate
with that definition is expected to sit near the pitch tolerance. The bag has a
`calibrex_conversion.json` sidecar (source drives and digests, command,
generator version and commit, frame and topic definitions). The db3 and
`metadata.yaml` follow the rosbag2 sqlite3 layout and are read back by
calibrex; they have not been opened with a ROS 2 install.

### Phase C1 results on KITTI raw (development drives only)

Drives 0005, 0009, 0014, 0015, 0022 of 2011_09_26, converted to bags; candidate
= the vendor `calib_imu_to_velo` as `/tf_static`, `base_link` = OXTS frame.
Drives 0027 to 0059 were not used. Defaults (`k = 3`, floors 0.5 deg).
Entries are `|delta| / tolerance` in degrees; `[w]`/`[f]` mark warn/fail.

**Bag path against the KITTI-text path.** For `lidar-vehicle`, `imu-vehicle` and
`ins-lidar`, on all five single drives and on the five pooled in one bag, the
bag run reproduces the text CLI (`calibrex lidar-vehicle kitti`, `imu-vehicle
kitti`, `ins-lidar kitti`, same drives): rotation, every DoF value, every std
and every held-out control statistic differ by exactly 0 (bit-identical; the
INS pose is written as the quaternion the text path builds its matrix from, so
no round trip enters). `lidar-wheel` differs on purpose: the text wheel proxy
uses the level-frame `vf` and `wu`, the bag declares the body-frame speed and
yaw rate. Rotation differences: 0.031 deg (0005), 0.021 (0009), 0.009 (0022),
0.028 (0014), 0.024 (pooled), and 0.84 deg on 0015 where roll is unobservable
(std 2.9 deg) and so moves with any input change. With the text path run on
the body-frame proxy (`tools/compare_check_kitti_text.py --body-frame-wheel`) the
difference is exactly 0 on 0005, 0022 and 0015.

**Verdicts** (tolerances 0.50 deg unless noted):

| Bag | `lidar-vehicle` | `imu-vehicle` | `ins-lidar` | `lidar-wheel_odometry` |
| --- | --- | --- | --- | --- |
| 5 drives pooled | pass, partial: pitch 0.50/0.50, yaw 0.31; roll unchecked | inconclusive | pass, partial: roll 0.11, pitch 0.06; yaw, x, y, z unchecked | pass, partial: pitch 0.50/0.50, yaw 0.31; roll unchecked |
| 0005 (15 s) | inconclusive | pass, partial: pitch 0.18; roll, yaw unchecked (yaw control not detected) | inconclusive | pass, partial: pitch 0.27, yaw 0.50/0.50 |
| 0009 (44 s) | pass, partial: yaw 0.26 | inconclusive | inconclusive | pass, partial: yaw 0.26 |
| 0014 | pass, partial: pitch 0.34, yaw 0.17 | pass, partial: pitch 0.47, yaw 0.05 | inconclusive | pass, partial: pitch 0.34, yaw 0.17 |
| 0015 | inconclusive | pass, partial: pitch 0.41 | inconclusive | pass, partial: yaw 0.30 |
| 0022 (~80 s) | pass, partial: pitch 0.45 | inconclusive | pass, partial: roll 0.14, pitch 0.08 | pass, partial: pitch 0.45 |

The overall verdict is `inconclusive` on every bag at baseline: no pair is
fully covered, and `imu-lidar` is `skipped` (`unsupported_sensor`, as expected
for a cloud without per-point time). `imu-vehicle` is `inconclusive` whenever its
axes have std above 0.1 deg, which is most single drives. The pooled
`lidar-vehicle` pitch error (0.50 deg) is the known OXTS-versus-vehicle offset
of the `base_link = imu_link` definition, sitting on the tolerance; it is not a
conversion error.

**Known-bad candidates**, +1 and +3 deg yaw on `velo_link` (rotation about the
`imu_link` z axis, `tools/make_check_frames_perturbation.py`, passed with `--tf`):

| Bag | +1 deg | +3 deg |
| --- | --- | --- |
| pooled | `lidar-vehicle` fail (yaw 1.31 [f], pitch 0.51 [w]); `lidar-wheel` fail (yaw 1.31 [f]); **`ins-lidar` pass (yaw unchecked)**; overall fail | same with yaw 3.31 [f]; `ins-lidar` still pass |
| 0009 | `lidar-vehicle` and `lidar-wheel` fail (yaw 1.26 [f]) | yaw 3.26 [f] |
| 0014 | both fail (yaw 1.17 [f]) | yaw 3.17 [f] |
| 0005 | `lidar-wheel` fail (yaw 1.50 [f]); `lidar-vehicle` inconclusive | yaw 3.50 [f] |
| 0015 | `lidar-wheel` fail (yaw 1.30 [f]); `lidar-vehicle` inconclusive | yaw 3.29 [f] |
| **0022** | **no flip**: `lidar-vehicle` and `lidar-wheel` pass, partial (pitch only, yaw unchecked) | **no flip** |

`imu-vehicle` is unaffected by the LiDAR mount, as it should be. A flip needs
the yaw axis to be observable: on 0022 the drive is long but yaw is unchecked
(std 0.11 deg, above the 0.1 deg observability limit), so a 3 deg error passes
with `coverage: partial`. A `pass (partial)` verdict says nothing about the
unchecked axes, and `ins-lidar` never saw the yaw error on any bag because yaw
is unchecked there. These are limits of 15 to 80 s drives; the pooled
`lidar-vehicle` and `lidar-wheel` do see it.

**Runtime** (8-core shared machine; the numbers marked "alone" ran with nothing
else active). Drive 0022 alone, no cache: 116 s wall clock (LiDAR odometry
101 s for `lidar-vehicle`, `ins-lidar` 11 s, `lidar-wheel` 1 s, `imu-vehicle` 0 s),
peak 196 MB. Pooled five drives with two jobs in parallel: 389 s. The two known-bad
re-judgements of each bag: 3 s (cache). The odometry pass is shared by the three
LiDAR pairs; the KITTI text CLIs repeat it per command
(0022: 106 s + 110 s + 234 s + 362 s for the four CLIs).

## Verdicts (Phase B)

| Pair | Estimator | Compared transform | Judged axes |
| --- | --- | --- | --- |
| `imu-lidar` | `imu-lidar` rotation (gyro against LiDAR odometry) and, unless `--no-imu-lidar-translation`, lever arm (accelerometer) | `T_lidar_imu` (the planner's `T_imu_lidar` inverted) | roll, pitch, yaw, x, y, z |
| `lidar-lidar` | map registration, started at the candidate | `T_first_second` | roll, pitch, yaw, x, y, z |
| `camera-imu` | targetless camera rotation against the gyro | `T_cam_imu` | roll, pitch, yaw |
| vehicle pairs | see [Phase C1](#phase-c1-vehicle-pairs) | | |

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

### Coverage

A pair is `partial` when any rotation axis, or any translation axis the
estimator attempted, is unchecked: because it is unobservable, or because the
estimator's own known-bad control on that axis was not detected on held-out data
(`control_not_detected`; the estimate is then not trusted as a yardstick). The
table prints `pass (partial: roll only)`, the pair record carries
`coverage: partial`, and the summary counts `partial_pairs`. A run whose pairs
all pass but some are partial keeps the overall verdict `pass`, and the CLI
prints a warning; a `pass` covers only the judged axes.

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

### Estimator cache

The `imu-lidar` and `camera-imu` estimates do not depend on the candidate: the
candidate is only compared with the estimate afterwards. `calibrex check`
therefore caches each estimator's artifact in `--cache-dir` (default
`$XDG_CACHE_HOME/calibrex/check`, i.e. `~/.cache/calibrex/check`) under a key of

* the bag digest (the one recorded in the check artifact),
* the estimator (`imu_lidar_rotation`, `imu_lidar_translation`,
  `camera_imu_rotation`),
* every estimator option: topics, per-point time field, acceleration unit,
  `--max-duration-s`, camera intrinsics, and all solver / windowing options, and
* the calibrex version, plus the git revision (and a digest of uncommitted
  changes) when it runs from a source checkout.

Changing any of these misses the cache. The lever-arm artifact is keyed on the
same ingredients plus its own options (it is solved from the cached rotation).
Re-checking a bag against another candidate, or with other `--sigma-k` or floor
values, then costs seconds: a hit rewrites only the candidate-dependent
`reference_value` / `error_to_reference` of the cached artifact for the current
candidate, so the evidence artifact still describes this check. A hit is flagged
in the check artifact (`pairs[].evidence_from_cache`, `evidence[].from_cache`),
and `runtime_s` is then the lookup, not the estimator. `lidar-lidar` starts its
solver at the candidate, so its result is not candidate-independent and is not
cached. `gnss-lidar` is cached the same way (the fit uses the candidate only as
the reference it is compared with afterwards; the key holds the NavSatFix topic,
the GNSS filter settings, the per-point time field, the span and all windowing
and solver options, and a hit rebases the reference fields). `gnss-imu` is a
cheap composition of the run's own results and is recomputed each time.
`--no-cache` recomputes everything and writes nothing; `calibrex check`
never deletes cache entries (remove the directory to clear it).

### Runtime controls

`--gnss-max-duration-s S` sets the span for `gnss-lidar` (and so the GNSS input
of `gnss-imu`) separately from `--max-duration-s`: the antenna lever arm needs
minutes of RTK-fixed windows, `imu-lidar` a couple of minutes (see Phase C2).
`--pairs a,b` restricts pairs; `--camera TOPIC` restricts `camera-imu` to one
image topic (or camera frame); `--max-duration-s S` analyses the first `S`
seconds of each sensor stream (images, scans, IMU samples within that window;
the IMU is still read in full but only the covered span is used);
`--scan-memory-mb MB` bounds the memory that keeps decoded LiDAR scans across
the estimator's passes (default 2048; `0` re-reads the bag every pass);
`--acceleration-unit` states the IMU unit for the lever arm (`g` for Livox
`livox_ros_driver2` recordings). Progress is printed to stderr.

LiDAR support: `PointCloud2` with a per-point time field. The field (`offset_time`,
`t`, `time`, `timestamp`, ...) and its meaning (offset in seconds, absolute
seconds as Hesai writes it, absolute nanoseconds as Livox writes it) are
detected from the first message by comparing its values with the header stamp. A
LiDAR without such a field is `unsupported_sensor`; deskewing needs it, and an
undeskewed scan on a moving platform would be a biased yardstick. Livox
`CustomMsg` is not read yet.

## Phase C2: GNSS pairs

| Pair | Needs in the bag | Estimator | Compared | Judged axes |
| --- | --- | --- | --- | --- |
| `gnss-lidar` | `NavSatFix` topic and a PointCloud2 LiDAR with per-point time | windowed variable projection of GNSS positions against deskewed LiDAR odometry (`slac.gnss_lidar_lever_arm/v0.1`) | antenna position in the LiDAR frame (translation of `T_lidar_gnss`) | x, y, z |
| `gnss-imu` | the same, plus `imu-lidar` (rotation and lever arm) in the same run | composition of the three artifacts as `calibrex gnss-imu compose` does (`slac.gnss_imu_lever_arm/v0.1`) | antenna position in the IMU frame (translation of `T_imu_gnss`) | x, y, z |

The roll, pitch and yaw of both pairs are listed as **unchecked** with the reason
that an antenna has no defined orientation (the RTK-SLAM CAD offset is a point,
with identity rotation assumed). `gnss-imu` needs `gnss-lidar` and `imu-lidar`
selected and solved in the same run (it is skipped with `missing_dependency`
otherwise, also when either failed its held-out check), and it needs the
`imu-lidar` lever arm, so `--no-imu-lidar-translation` skips it.

**GNSS from the bag.** `calibrex.data.navsatfix_track` builds the estimator's
track from the `NavSatFix` topic; no `rtk.txt` is needed.

* *Time*: the header stamp. On RTK-SLAM it equals `rtk.txt`'s timestamp, and the
  LiDAR scans use the same clock, so no offset is applied. (The estimator also
  fits a clock offset; it is reported, not judged.)
* *Position*: latitude, longitude and the altitude **above the WGS84 ellipsoid**,
  which is what the ENU conversion uses (RTK-SLAM's `/gnss/fix` altitude equals
  `rtk.txt`'s `height` to the millimetre).
* *Quality*: `NavSatFix.status` cannot tell RTK-fixed from RTK-float. RTK-SLAM's
  driver publishes RTK-fixed as `0` and everything else (`rtk.txt` status 1 and
  2) as `-1`; a fix is usable when `status >= 0` **and** its 3-D standard
  deviation is at most 0.15 m, so ordinary single-point GPS is not mistaken for
  RTK.
* *Standard deviation*: `sqrt(trace(position_covariance))`, which reproduces
  `rtk.txt`'s `blt_std` exactly (ratio 1.0000, max difference 0). With
  `position_covariance_type` 0 (unknown) the covariance is not used: only
  `GBAS` (2) fixes are kept, with a 5 cm std, and the count is recorded.
* A bag with no usable fixes (KITTI `/oxts/fix` carries a 0.74 m covariance)
  makes the pair `skipped` (`unsupported_sensor`) with the counts in the reason.

**Candidate and caching.** The fit does not use the candidate (it is compared
afterwards), so it is cached; see the estimator cache above. The planner's
candidate is `T_gnss_lidar`; the estimate is its inverse's translation, which
for the RTK-SLAM calibration equals `rtk_slam_reference_lever_arm` exactly.

### Phase C2 results on RTK-SLAM

Rig: Livox MID360 on a hand-held pole, candidate = `calib.yaml` (CAD antenna
offset, MID360 manual IMU position), `--max-duration-s 120` for `imu-lidar`.
All four sequences were already spent for claims; these are validation runs. Only
the first minutes of each sequence have long RTK-fixed stretches (construction:
about 220 s, then 4 % fixed; stadtgarten seq1: 54 %, seq2: 40 %).

**Equivalence with the `rtk.txt` path.** The same windows, the same code, GNSS
from `/gnss/fix` against `rtk.txt` (`calibrex gnss-lidar rtk-slam --max-seconds`),
same scans (cut at the same seconds):

| Run | Windows | x | y | z | time offset |
| --- | ---: | --- | --- | --- | --- |
| stadtgarten_seq2, 600 s: `rtk.txt` | 29 | 4.937 cm ± 1.65 | -0.224 cm ± 0.41 | 3.945 cm ± 5.49 | -23.69 ms ± 8.7 |
| stadtgarten_seq2, 600 s: bag | 29 | 4.949 cm ± 1.61 | -0.227 cm ± 0.42 | 3.949 cm ± 5.51 | -23.80 ms ± 8.6 |
| construction_seq1, 240 s: `rtk.txt` | 17 | 4.087 cm ± 2.42 | 0.348 cm ± 1.46 | 7.522 cm ± 4.32 | 6.39 ms ± 21.6 |
| construction_seq1, 240 s: bag | 17 | 4.070 cm ± 2.43 | 0.337 cm ± 1.45 | 7.519 cm ± 4.33 | 6.41 ms ± 21.6 |

The estimates differ by at most 0.17 mm, 0.1 % to 2 % of their std. The ENU
positions of the common fixes agree to 0.3 mm rms in height (`rtk.txt` rounds
height to 1 mm) and exactly in the horizontal, and the stds are identical. The
only data difference: seq2's bag path drops 10 of 3541 fixes whose covariance
std exceeds 0.15 m although the driver reports them fixed (3531 used). The
policy status and the held-out residual (2.20 against 2.21 cm) agree as well.

**Verdicts against the CAD antenna offset** (`--gnss-max-duration-s` as stated):

| Bag, spans | `gnss-lidar` | `gnss-imu` | Estimator evidence |
| --- | --- | --- | --- |
| stadtgarten_seq2, `imu-lidar` 120 s, gnss 900 s (whole bag) | pass, partial: y 0.28 / 2.0 cm; x, z unchecked | pass, partial: y 0.73 / 2.9 cm; x, z unchecked | 37 windows; x 5.13 +/- 1.04 cm (just over the 1 cm bound), z 4.0 +/- 4.1 cm |
| stadtgarten_seq1, `imu-lidar` 120 s, gnss 600 s | **warn**, partial: x 2.15 / 2.0 cm; y, z unchecked | inconclusive (`no_judgeable_axes`) | 45 windows; x 5.55 +/- 0.64 cm; y 1.4 cm and z 1.3 cm std over the bound |
| construction_seq1, gnss 240 s | inconclusive (`no_judgeable_axes`) | not run | 17 windows; x std 2.4 cm |
| KITTI 0009 (`/oxts/fix`) | skipped (`unsupported_sensor`) | skipped (`missing_dependency`) | the LiDAR has no per-point time; the OXTS covariance is 0.74 m (not RTK) |

On a candidate that is the dataset's own CAD calibration, `gnss-lidar` finds the
antenna **1.7 to 2.2 cm further out in x** on both sequences (seq1 5.55 cm,
seq2 5.13 cm, CAD 3.4 cm). The 2 cm translation floor sits right at that
discrepancy, so seq1 is a `warn`; the benchmark page already notes this as a
tension to resolve (the antenna phase centre is poorly defined by CAD). Read it
as the check not being able to separate a CAD rounding from a miscalibration at
2 cm, not as a miscalibrated rig. `gnss-imu` adds the `imu-lidar` lever arm's
uncertainty (x std 1.2 cm, y 1.8 cm on seq1, both over its bound), which is why
it judges only y on seq2 and nothing on seq1. Time offsets are reported, not
judged, and none is observable.

**Known-bad antenna offsets** (the CAD offset moved in a `--tf` copy of
`calib.yaml`; the estimates are re-used from the cache, 3 s per run):

| Perturbation | seq1 `gnss-lidar` | seq2 `gnss-lidar` / `gnss-imu` |
| --- | --- | --- |
| baseline | warn (x 2.15 cm [w]) | pass / pass (y only) |
| x +5 cm | warn (x 2.85 cm [w]) | **pass / pass, x unchecked** |
| x +15 cm | **fail** (x 12.9 cm [f]) | **pass / pass, x unchecked** |
| y +5 cm | not run (y is unchecked at baseline) | **fail** (y 5.3 cm [f]) / **warn** (y 4.3 cm [w]) |
| y +15 cm | not run | **fail** (15.3 cm) / **fail** (14.3 cm) |
| z +5 cm, z +15 cm | pass/warn unchanged: z unchecked | pass / pass: z unchecked |

The detectable error (tolerance + current delta, in the artifact as
`detectable_error`) is what the data can see: x 4.2 cm on seq1, y 2.3 cm
(`gnss-lidar`) or 3.7 cm (`gnss-imu`) on seq2. An axis the estimator leaves
unconstrained detects nothing: **z (std 1.3 to 4 cm) is never checked on these
bags, and x on seq2 sits just over the bound**, so a `pass (partial)` here says
nothing about them. The estimator's own 5 cm control is detected on every lever-arm axis
(delta chi-square 30 to 220; the 20 ms clock-offset control is not on seq2), but the check judges only axes whose reported std is
at most 1 cm.

**Runtime** (8-core shared machine, two jobs in parallel): `gnss-lidar` on
seq2's 900 s took 9.5 min and on seq1's 600 s 14 min (odometry runs only on the
RTK-fixed stretches; construction's 240 s took 5 min); `imu-lidar` 120 s with its lever arm took 21 to 27 min,
which is why the two pairs get separate spans. `gnss-imu` costs 0 s;
re-checking with another candidate costs 3 s.

Choosing the spans: the lever arm needs about 30 or more 10 s windows inside
RTK-fixed stretches, which on these sequences means 600 to 900 s of recording;
the first 240 s of construction (17 windows) are not enough for any axis.

## Phase B results on real recordings

All runs are on development or already-spent recordings (Hilti exp21, NTU VIRAL
tnp_01, RTK-SLAM construction_seq1); none touches a held-out recording. The
candidate is each dataset's own deployed calibration. "Known-bad" rows add a
deliberate yaw error to the candidate and re-judge the same bag. Defaults:
`k = 3`, floors 0.5 deg and 0.02 m. Runtimes are wall-clock on a shared 8-core
machine that was oversubscribed during these runs, so read them as upper
bounds.

| Recording / pair | Candidate | Verdict | Per-axis `|delta|` / tolerance | Unchecked | Runtime |
| --- | --- | --- | --- | --- | ---: |
| Hilti exp21, cam0 / IMU | Kalibr `calib_3_cam0-1` | `pass` | roll 0.19/0.50, pitch 0.33/0.50, yaw 0.25/0.50 deg | - | 5.8 min |
| Hilti exp21, cam1 / IMU | Kalibr `calib_3_cam0-1` | `pass` | roll 0.15/0.50, pitch 0.33/0.50, yaw 0.19/0.53 deg | - | 5.8 min |
| Hilti exp21, cam0 / IMU | known-bad, +1 deg yaw | `fail` | roll 0.19/0.50, pitch 0.33/0.50, **yaw 1.25/0.50** deg | - | 3.2 min |
| Hilti exp21, cam0 / IMU | known-bad, +3 deg yaw | `fail` | roll 0.18/0.50, pitch 0.34/0.50, **yaw 3.25/0.50** deg | - | 3.3 min |
| Hilti exp21 (first 40 s), IMU / Hesai PandarXT-32 | `lidar_calibration.yaml` | `pass (partial: roll only)` | roll 0.12/0.50 deg | pitch (std 0.13 deg), yaw (0.19 deg) | 25 min |
| Hilti exp21 (full 153 s, with lever arm), IMU / Hesai | `lidar_calibration.yaml` | not run to completion | stopped after 2.5 h; see Cost | - | - |
| NTU tnp_01 (first 240 s), horz / vert LiDAR | design `T_Body2Lidar` | `fail` | pitch 0.52/0.50 (warn), yaw 0.19/0.50, x 0.011/0.020, **y 0.074/0.029, z 0.058/0.020** | roll (std 0.11 deg) | 34 min |
| NTU tnp_01 (first 240 s) | known-bad, +1 deg yaw | `fail` | same, **yaw 1.19/0.50** (fail) | roll | 34 min |
| NTU tnp_01 (first 240 s) | known-bad, +3 deg yaw | `fail` | same, **yaw 3.19/0.50** (fail) | roll | 34 min |
| RTK-SLAM construction_seq1 (first 180 s), IMU / LiDAR | `calib.yaml` | `pass (partial: roll, pitch, yaw, z only)` | roll 0.22/0.50, pitch 0.03/0.50, yaw 0.20/0.50 deg; z 0.001/0.020 m | x (`control_not_detected`), y (std 14 mm) | 94 min |
| RTK-SLAM construction_seq1 (first 180 s) | known-bad, +1 deg yaw, rotation only | `warn` | yaw **0.80/0.50** (warn), roll 0.22, pitch 0.03 | - | 87 min |
| RTK-SLAM construction_seq1 (first 180 s) | known-bad, +3 deg yaw, rotation only | `fail` | yaw **2.80/0.50** (fail), roll 0.22, pitch 0.04 | - | 86 min |

What the table shows, including what does not look good:

* **Known-bad flips.** On cam0 the verdict goes `pass` to `fail` at +1 deg (the
  yaw error is 0.25 deg at baseline, so the added 1 deg exceeds twice the 0.5 deg
  floor). On RTK-SLAM the rotation axes pass at baseline and the yaw axis goes
  `pass` (0.20 deg) to `warn` (0.80 deg) at +1 deg and `fail` (2.80 deg) at +3 deg.
  On NTU the yaw axis flips `pass` to `fail` at +1 deg; the pair was already
  `fail` from translation, so the pair verdict does not move there. The
  non-perturbed axes keep their baseline numbers to within 0.01 deg, as they should.
* **The NTU design transform fails.** The deployed `T_Body2Lidar` values are
  rounded design numbers; the registration puts the second LiDAR 7.4 cm and
  5.8 cm away in y and z, and 0.5 deg in pitch, with standard deviations of
  1 cm or less, and the held-out chi-square rises by about 22,700 when the
  design value replaces the estimate. This matches the documented development
  result (about 0.5 deg of pitch and 5 cm in y and z, see the
  [NTU LiDAR-LiDAR benchmark](../benchmarks/ntu_viral_lidar_lidar.md)). No
  independent measurement says which is right; the check says the data do not
  support the design values at the stated accuracy. The estimator's own policy is
  `inconclusive` (roll is unobservable), and roll is left unchecked.
* **RTK-SLAM is a partial pass.** The first run judged x as a `warn` (3.2 cm
  against 2.8 cm). The estimator's own known-bad 20 mm shift on x was not
  detected on held-out windows, so x is now left unchecked with reason
  `control_not_detected` (the same evidence, re-judged), and the pair is `pass`
  with partial coverage. The y axis was not constrained. The clock offset
  (10.1 ms) is reported, not judged. The +1 and +3 deg rows were rotation only
  and are unaffected.
* **Detection power.** At baseline every judged rotation axis has a
  `detectable_error` between 0.53 and 0.83 deg, except the NTU pitch axis
  (1.03 deg, because the candidate is already 0.52 deg off): a 1 deg error on
  any other axis would have been flagged. The floors dominate the tolerances: at
  0.5 deg the rule cannot see errors much below that, whatever the estimator's
  `std`.
* **Hesai works, but one axis is not a full check.** The Hesai `timestamp`
  field (float64 absolute seconds) is detected and read as absolute seconds. On
  the first 40 s of exp21 only roll cleared the 0.1 deg observability bound, so the
  pair is `pass` on a single axis and lists pitch and yaw as unchecked. Read the
  `unchecked` column: a `pass` covers only the axes that were judged. The
  per-axis rule is applied exactly as stated; whether a pair that judged fewer
  than all rotation axes should be reported differently is an open question.
* **Cost.** The LiDAR estimators dominate. IMU-LiDAR runs one odometry pass
  plus up to five gyro-deskew refinement passes and a feedback check, each pass
  about 0.4 s per scan on these streams: 180 s of Livox data took 87 to 94 minutes
  and 40 s of Hesai data (58,000 points per scan) took 25 minutes on a loaded
  machine, and a full 153 s Hesai run with the lever arm was stopped after 2.5
  hours (as first measured; see below for the current cost). The cost does not
  scale well with the window, so bound it with `--max-duration-s`, `--pairs` and
  `--no-imu-lidar-translation`. LiDAR-LiDAR took 34 minutes for 240 s.
  Camera-IMU took 3 to 6 minutes for 153 s.

### Runtime after the speed-up

The table above was measured before the estimator cache and the odometry
speed-ups. The IMU-LiDAR estimator spends its time in the odometry passes (one
constant-velocity pass, up to five gyro-deskew passes and a feedback check, each
registering every scan). A profile of the first 15 to 25 s showed where it went:
voxel downsampling (`np.unique` over rows) 17 to 41 %, the per-point gyro
rotation model of the deskew 15 to 29 %, reading the whole IMU topic once for
the rotation and twice more for the lever arm 11 to 27 % (the SQLite reader
fetched every topic's payload), normals, KD-tree and registration the rest.
Three changes that leave every output **bit-identical** (compared artifact by
artifact: 0 differences outside provenance) removed most of it: a faster
voxel key, the gyro rotation computed once per scan and from one shared start
orientation, and an SQL topic filter plus one IMU read per run. Decoded scans
are also kept in memory across passes (`--scan-memory-mb`). The remaining
cost is the registration itself, which the gyro-deskew iteration needs: its
yaw estimate moves by 0.5 deg in the second pass and still by 0.06 to 0.1 deg
in the fifth (reported std 0.05 to 0.2 deg) before it settles, so fewer passes or a coarser voxel do not reproduce the estimate and
the check does not offer a "fast" mode. Same recordings, same machine (shared,
loaded), same options:

| Recording / run | Before | After (first run) | Re-check with another candidate (cache hit) |
| --- | ---: | ---: | ---: |
| Hilti exp21 first 40 s, IMU / Hesai, rotation | 25.4 min | 8.5 min | 3 s |
| RTK-SLAM construction_seq1 first 180 s, rotation + lever arm | 94 min | 28 min | 3 s |
| RTK-SLAM, known-bad +1 deg yaw (`warn`) | 87 min | 3 s (cache) | - |
| RTK-SLAM, known-bad +3 deg yaw (`fail`) | 86 min | 3 s (cache) | - |
| Hilti exp21 cam0 / IMU first 60 s, camera-IMU | - | 32 s | 3 s (+1 deg yaw: `pass` to `warn`) |

The estimator artifacts of the first runs equal the earlier ones in every field
but provenance, and the cache-served runs reproduce the earlier +1 and +3 deg
verdicts and per-axis numbers (roll, pitch and yaw errors agree to 1e-15; the
reference fields are recomputed for the candidate, so they differ from a fresh
solve only in the last bit). Peak memory was 0.75 GB (Hilti) and 1.2 GB (RTK-SLAM)
with the default scan budget.

## Artifact

`--output` writes `slac.calibration_check/v0.1`: the bag path with a digest
(`metadata.yaml` in full; storage files by name, size and first 64 MiB, the
same scope used by other bag-based artifacts), candidate sources, topics with
roles and frames, the frame tree, per-pair records with the candidate
transform, status and reason, a summary, and provenance. A run adds, as
optional fields: `options` (thresholds and runtime controls), `evidence_dir`,
`overall_verdict`, and per pair `compared_transform`, `axes`, `unchecked_axes`,
`time_offset`, `estimator*`, `runtime_s`, `evidence` (each with `from_cache`)
and `evidence_from_cache`; `options.topic_kinds` records `--topic-kind`.

Each estimator's own schema'd artifact (`slac.imu_lidar_rotation/v0.1`,
`slac.imu_lidar_translation/v0.1`, `slac.lidar_lidar_extrinsic/v0.1`; for the vehicle pairs
`slac.vehicle_frame_rotation/v0.1`, `slac.ins_lidar_hand_eye/v0.1`,
`slac.lidar_wheel_odometry/v0.1`) is written
to `<output stem>_evidence/<pair>_<sensors>[_role].yaml`, referenced from the
pair by relative path and SHA-256 (`evidence`; `evidence_artifact` repeats the
first path). Those files hold the per-DoF estimates, jackknife and known-bad
controls, and policy status.

No HTML report yet: the existing HTML helpers render calibration results, not
check artifacts.
