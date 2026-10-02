# Check a deployed calibration (`calibrex check`)

Status: **Phase D** (rig closure, HTML report, the 1 degree demo) on top of **Phase C2** (imu-lidar, lidar-lidar, camera-imu, the GNSS pairs gnss-lidar and
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
| `degenerate_frames` | two sensors of a pair (or a sensor and `--vehicle-frame`) are stamped in the same frame, typically streams already transformed into `base_link`; the candidate would be the identity by construction, so the pair is not run (use `--frame-map` or `--tf` to name the physical sensor frames) |
| `method_not_wired` | no check method exists for the pair yet |
| `not_selected` | a wired pair that `--pairs` or `--camera` left out |
| `unsupported_sensor` | the sensor cannot be read by the estimator: a LiDAR that is not `PointCloud2`, a LiDAR without a per-point time field for `lidar-lidar` and `gnss-lidar` (`imu-lidar` runs such clouds as rigid scans), a compressed camera image |
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
| `/velodyne_points` | `PointCloud2` | `velo_link` | x, y, z, intensity (float32). **No per-point time field**: KITTI raw has none and none is invented, so `imu-lidar` runs on rigid scans (no deskew; see "Clouds without per-point time") |
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
LiDAR without such a field is `unsupported_sensor` for `lidar-lidar` and `gnss-lidar`
(`imu-lidar` falls back to rigid scans, see "Clouds without per-point time"); deskewing needs it, and an
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

## Phase D: closure, HTML report and the 1 degree demo

### Rig closure

Each pair is judged against the *candidate* on its own. Closure adds a
candidate-free test: the run's own estimates form a graph of frames, and every
loop of it must multiply to the identity.

- **Edges** are the per-pair estimates `T_parent_child` (rotation, and translation
  where the estimator reports it) with the estimator's per-axis std. Two methods
  estimating the same frame pair (lidar-vehicle and lidar-wheel_odometry both give
  `R_vehicle_lidar`) are parallel edges; lidar-vehicle, imu-vehicle and ins-lidar
  form a cycle through the vehicle, IMU and LiDAR frames.
- **Derived estimates are excluded.** `gnss-imu` is composed from this run's
  gnss-lidar and imu-lidar results, so it is not independent evidence. It is listed
  as `derived` in `closures.edges`. An estimate with no observed axis is `unusable`.
- **Loops** are a fundamental cycle basis of the remaining graph (a spanning forest,
  one loop per other edge). Loops that share a member are correlated.
- **Error and std.** The closure error is `rotvec` of the loop product in degrees
  (and the translation of the loop product in metres, only if every member estimates
  it), in the axes of the loop's first frame. The std is propagated to first order
  assuming independent members, which is optimistic when members share an input: the
  three vehicle pairs that use LiDAR motion read the same LiDAR odometry, and a loop
  with two of them carries a note saying so.
- **Verdict.** The rule of the pairs: `tolerance = max(sigma_k * std, floor)`, pass
  within 1x, fail beyond 2x. A closure axis is judged only if every member observes
  everything that enters it (an unobserved member axis with a first-order weight above
  0.05 makes the axis `unchecked`).
- **Overall verdict.** A failing loop raises `overall_verdict` to at least `warn`,
  never to `fail`. A closure contains no candidate: it says the estimators disagree
  with each other, which makes their verdicts less trustworthy, not that the candidate
  is wrong. A `warn` loop is reported but does not change the overall verdict.

The result is the optional `closures` field (`edges`, `loops` with members,
per-axis judgements, unchecked axes and notes, and the worst loop `verdict`).
Without a loop (fewer than two independent estimates sharing frames) it is empty.

On the pooled KITTI development bag (drives 0005, 0009, 0014, 0015, 0022) the one loop is
lidar-wheel_odometry against lidar-vehicle: pitch -0.0003 deg, yaw +0.004 deg, `pass`; roll is
unchecked because neither estimate observes it. That is small because both read the same LiDAR
odometry, so it is a weak test of the odometry itself. The cycle through imu-vehicle and
ins-lidar does not form: imu-vehicle is `inconclusive` on KITTI (its std is over the bound,
the 10 Hz INS velocities are noisy), hence `unusable`, and ins-lidar does not observe yaw. The
documented
[KITTI closure](../benchmarks/kitti_lidar_vehicle.md#ins-vehicle-calibrex-imu-vehicle-kitti)
(LiDAR-vehicle composed with KITTI's `calib_imu_to_velo` against INS-vehicle, within
(-0.03, -0.03, +0.09) deg) uses estimates `calibrex check` refuses to judge for the same
reason, so it stays a separate, benchmark-level statement.

### HTML report

```bash
calibrex check my_bag/ --tf rig.urdf --output check.json --html check.html
calibrex render check.json --html check.html     # re-render a saved artifact
```

One self-contained page (no scripts, no external assets, light and dark): the overall
verdict banner, a table of the judged pairs (verdict, coverage, `|error| / tolerance` per
axis, unchecked axes with reasons, detectable error, evidence file with SHA-256 and a link
relative to the report), the skipped pairs with reasons, the closure loops, the candidate frame
tree, and the provenance (bag digest and its scope, candidate sources, command, version).

### The 1 degree demo

`tools/check_tf_injection_demo.py` turns `velo_link` of the KITTI pooled bag by +1 and +3
degrees about its parent's z axis and runs `calibrex check` on each (details and the table in
the [README](https://github.com/rsasaki0109/Calibrex#check-a-deployed-calibration)). Outputs, with absolute paths
shortened, are in `docs/assets/calibrex_check_demo/`. The estimator cache key includes a digest
of the working tree, so editing calibrex invalidates the cache; the first variant costs about
6 minutes and the others seconds.

## Clouds without per-point time (rigid scans)

`imu-lidar` normally deskews every sweep with the cloud's per-point time field
(a LiDAR constant-velocity pass, then up to five gyro-deskew passes and a
feedback check). Many clouds have no such field: Autoware concatenated clouds,
depth-camera clouds, KITTI Velodyne, many drivers. Those used to end
`unsupported_sensor`. `--imu-lidar-deskew` now chooses:

| Mode | Behaviour |
| --- | --- |
| `auto` (default) | per-point time field present: gyro deskew (unchanged, bit-identical); absent: **rigid scans** |
| `gyro` | requires the field; skips the pair (`unsupported_sensor`) without it |
| `none` | forces rigid scans, even when the field exists (used to measure the bias) |

A cloud that has a time field is never silently run rigid: `auto` only falls
back when the field is missing, and the pair record says which mode ran
(`deskew: gyro | none`, plus a note in the table).

**Rigid scans** treat every point as stamped at the scan's header stamp: one
odometry pass, no constant-velocity or gyro deskew, no feedback check. The
`slac.imu_lidar_rotation/v0.1` artifact records `options.deskew: none`, one
`rigid_none` pass and the limitations below; the gyro-mode artifact is unchanged.
Two consequences are documented rather than corrected:

* the motion during a sweep is not compensated, so the rotation rates (and the
  lever arm) are biased in proportion to how fast the rig turns;
* the header stamp may mark the start, the middle or the end of the sweep
  depending on the driver, and the clock offset absorbs that constant, so the
  reported time offset includes about half a sweep (about 50 ms for a 10 Hz
  LiDAR) of stamp convention. It is reported, never judged.

### Measured bias

Same bag, same window, the check run with `--imu-lidar-deskew gyro` and `none`
(the gyro result is the reference for the bias; neither is ground truth).
`delta` is the rigid estimate minus the gyro estimate, in the sensor axes of the
estimator. Runtimes are wall-clock on a shared, oversubscribed 8-core machine.

| Recording | Axis | gyro estimate (std) | rigid estimate (std) | delta |
| --- | --- | ---: | ---: | ---: |
| RTK-SLAM construction_seq1, first 180 s (hand-held Livox, 18 windows) | roll | -0.224 (0.057) deg | -0.426 (0.269) deg | -0.20 deg |
| | pitch | 0.031 (0.038) deg | 0.038 (0.220) deg | +0.01 deg |
| | yaw | -0.199 (0.045) deg | -0.639 (0.335) deg, not observable | -0.44 deg |
| | time offset | 10.1 ms | 61.3 ms | +51.2 ms |
| | lever arm | x 43 mm, z -43 mm (estimated), y not observable | not observable (std 21-26 mm) | - |
| | runtime | 38.8 min | 8.4 min | |
| Hilti exp21, first 40 s (hand-held Hesai PandarXT-32, 4 windows) | roll | -179.924 (0.068) deg | -179.873 (0.671) deg | +0.05 deg |
| | pitch | 0.122 (0.126) deg | 0.187 (0.600) deg | +0.07 deg |
| | yaw | -89.498 (0.193) deg | -88.975 (0.671) deg | +0.52 deg |
| | time offset | 1.8 ms | 49.7 ms | +47.9 ms |
| | lever arm | z 60 mm (std 3 mm) | not observable (std 40-135 mm) | - |
| | runtime | 14.6 min | 1.9 min | |
| MID360 driving (rosbag2_2024_04_16-14_17_01, first 120 s, vehicle, 12 windows; API run, no reference calibration) | roll | -0.376 (0.131) deg | -0.256 (0.181) deg | +0.12 deg |
| | pitch | 0.127 (0.142) deg | 0.026 (0.187) deg | -0.10 deg |
| | yaw | 0.679 (0.426) deg | 0.610 (0.513) deg | -0.07 deg |
| | time offset | 2.6 ms | 53.6 ms | +51.0 ms |
| | runtime | 32.7 min | 5.3 min | |
| Synthetic rotating rig (unit test; pure rotation, 0.3 and 1.0 x the base motion, sweep 0.1 s) | roll / pitch / yaw error against the truth | at most 0.10 deg | at most 0.10 deg | none beyond the gyro result's own error |
| | time offset | 7.6 / 11.9 ms | 57.3 / 62.1 ms | +49.7 / +50.2 ms |

What the measurements say:

* The rotation bias of rigid scans against the gyro deskew is **at most 0.52 deg**
  on the two hand-held recordings (yaw is the worst axis on both), 0.12 deg on the
  driving recording and about 0.1 deg or less on the synthetic rig. Every
  estimate on the driving recording is within the gyro-mode std of the gyro one,
  so it is not distinguishable from the gyro result there; fast hand-held motion
  is where the bias shows. It is also much smaller than the
  error of the LiDAR constant-velocity deskew that the gyro mode starts
  from (RTK-SLAM yaw after that first pass: -2.8 deg against -0.2 deg at
  the end), because the rigid scan does not impose a motion model at all.
* The clock offset moves by +48 to +51 ms, about half a 0.1 s sweep, on all
  three LiDARs, consistent with header stamps at the start of the sweep.
* The reported standard deviations grow 3.5 to 10 times on the hand-held
  recordings and 1.2 to 1.4 times on the vehicle (jackknife, not analytic): the
  windows that deskewing makes consistent are noisier without it. Hilti's four
  windows leave every axis unobservable; the lever arm is not observable on
  either recording in rigid mode. That, not the bias, is what limits the verdict.
* Bias is not a std: the rigid yaw offsets (0.44, 0.52 deg) are 3 to 10 times the
  gyro-mode std of the same axis (0.045, 0.19 deg). Reading a rigid-scan estimate
  at gyro-mode resolution would therefore mis-judge. Bias grows with rotation
  rate; the hand-held recordings turn faster than a vehicle does.

### Policy

Chosen from those numbers:

* **Rotation floor 1.5 deg for rigid scans** (`--rigid-scan-rotation-floor-deg`,
  default 1.5; the global `--rotation-floor-deg` 0.5 still applies when it is
  larger). 1.5 deg = the largest measured bias (0.52 deg) plus three times the
  largest std that is still called estimated (0.3 deg), rounded up, so a bias of
  the measured size cannot turn a good calibration into a `warn`. The tolerance
  of such an axis is `max(3 std, 1.5 deg)` and `tolerance_source` says `floor`.
* **Axes are called estimated at std <= 0.3 deg** (0.1 deg in gyro mode, a fifth
  of its floor; the same ratio to the 1.5 deg floor), and the estimator's own
  known-bad control is 1.5 deg (not 1 deg) so it must detect a shift as large as
  the floor. An axis whose control is not detected is unchecked
  (`control_not_detected`), as in gyro mode.
* Translation keeps its normal floor; in practice the rigid lever arm is
  unobservable, so the translation axes are listed as unchecked.
* The time offset is reported with the stamp-convention caveat and never judged.
* What a pass means changes: with the 1.5 deg floor a rigid-scan `pass` cannot
  resolve a 1 deg error (`detects_perturbation: false` in the record); it can
  resolve about 2 to 3 deg and larger. The detection power is in the table.

Because rigid mode costs a fraction of the gyro runtime (one odometry pass), the
estimate is also cached like the gyro one (its cache key includes the deskew
mode, so the two never mix).

### Result on Koide `indoor_easy_01` and `indoor_easy_02`

These bags used to end `unsupported_sensor` for `imu-lidar`. The "LiDAR" is a
**depth camera** (`/points2/decompressed`, plain x/y/z, `depth_camera_link`), not a
spinning LiDAR, so this is a depth-camera-IMU check on a hand-held rig, and the
median held-out rate residual is high (0.07 to 0.09 rad/s, against 0.01 for the
LiDARs above). Zero-config run, rigid scans selected automatically:

| Sequence | Verdict | Judged, `|delta|` / tolerance | Unchecked | Runtime | Held-out rate residual (median) |
| --- | --- | --- | --- | ---: | ---: |
| `indoor_easy_01` | `pass (partial: roll, pitch, yaw only)` | roll 0.08/1.5, pitch 0.66/1.5, yaw 0.07/1.5 deg | x, y, z (std 38, 30, 24 mm, `unobservable`) | 5.3 min | 0.069 rad/s |
| `indoor_easy_02` | `pass (partial: roll, pitch, yaw only)` | roll 0.32/1.5, pitch 0.29/1.5, yaw 0.40/1.5 deg | x, y, z (std 18, 25, 28 mm, `unobservable`) | 6.9 min | 0.091 rad/s |

Reported std 0.19 to 0.29 deg on every judged axis; the detectable error
(tolerance plus the candidate's error) is 1.6 to 2.2 deg. Peak memory 2.3 GB with
the default scan budget.

The IMU-camera extrinsic from `/tf_static` agrees with the estimate to within
0.7 deg on every axis, and the pair is `pass (partial: roll, pitch, yaw only)`;
the lever arm is not observable. A deliberate +1 deg yaw error on
`imu_link` (`tools/make_check_frames_perturbation.py ... imu_link 1`) is
**not detected** (`pass`, yaw 0.94 / 1.5): that is the stated resolution of
rigid scans, not a failure to look. +2 deg and +3 deg give `warn` and +4 deg
gives `fail` on `indoor_easy_01`; each known-bad re-check is served from the
cache in about 4 s. Autoware `all-sensors-bag1` is unchanged: `imu-lidar` is
still skipped `degenerate_frames` (cloud and IMU are both in `base_link`).

`gnss-lidar` and `lidar-lidar` still require a per-point time field; rigid
scans are implemented for `imu-lidar` only.

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

## Real bags with /tf_static

The zero-config path (no `--tf`, candidate read from the bag's `/tf_static`) was
first exercised only on synthetic bags and our own KITTI conversions. These are
third-party ROS 2 bags that ship their own `/tf_static`, run with calibrex 0.5.0
plus the fixes listed below. Nothing here is a benchmark: none of these bags
yields a `pass`, `warn` or `fail`, and that is the honest result. Licenses were
not shipped with the local copies and were not verified; check each source before
redistributing anything. No data from these bags is committed.

Bugs found by this exercise (all fixed, with unit tests):

- **CameraInfo with end padding.** Autoware's CDR `CameraInfo` carries 3 trailing
  zero bytes (alignment padding after the final `uint8`); the decoder rejected it
  (`payload has 3 trailing byte(s)`) and `camera-imu` ended `estimator_error` on
  every camera. Zero padding to a 4-byte multiple is now accepted for `CameraInfo`
  and `Image`; any other tail is still rejected.
- **Draco-encoded cloud picked as the LiDAR.** Koide's `/points2/compressed` is a
  Draco blob typed `sensor_msgs/msg/PointCloud2`, sorted before its decompressed
  twin, so the estimator tried to read it as points (garbage, `invalid value
  encountered in cast`). Topics whose first message is not a raw point array are
  now recorded with `ignored_reason` (optional topic field) and fill no sensor slot.
- **Identity candidates presented as a check.** Autoware's concatenated cloud and
  its IMU are both stamped in `base_link`, so `imu-lidar` and the vehicle pairs
  planned an identity transform `base_link`/`base_link`. They are now skipped with
  the new reason code `degenerate_frames` (an added enum value; the artifact
  stays `slac.calibration_check/v0.1`).

### Autoware `all-sensors-bag1`

- **Origin.** `migrated/autoware_data/all-sensors-bag1` (Autoware sample data, a
  real vehicle; sqlite3, 36.5 s). No license file accompanies the local copy;
  not verified.
- **Frame tree.** `/tf_static` (1 message) gives 14 frames, one root `base_link`:
  `base_link` > `sensor_kit_base_link` > {`camera_{left,right,top}/camera_link` >
  `.../camera_optical_link`, `gnss_ins_link`, `velodyne_{front,left,right}_base_link`
  > `velodyne_*`}. Sensor frame ids are found in the tree except `/gnss/fix`
  (`POS_REF`).
- **Ignored gracefully.** The applanix custom messages and `VelodyneScan` packets
  have no sensor role and are not listed; `/tf` (dynamic) is not used.
- **Topic mapping.** The cameras map by header to their `camera_*/camera_link`
  frames (not the optical frames; the check composes through the tree either way).
  The concatenated cloud and the IMU map to `base_link`. `/gnss/fix` is
  `UNMAPPED`.

| Pair | Plan (`--vehicle-frame base_link`) | Result |
| --- | --- | --- |
| `imu-lidar` | skipped `degenerate_frames` (both in `base_link`) | (before the fix: `unsupported_sensor`, no per-point time) |
| `camera-imu` x3 | planned (camera link / `base_link`) | inconclusive, `no_judgeable_axes`: roll/pitch/yaw std 15-46 deg over 36 s |
| `camera-focal` | planned | `method_not_wired` |
| `gnss-lidar`, `gnss-imu` | skipped `frame_not_in_tree` (`POS_REF`) | with `--frame-map /gnss/fix=gnss_ins_link`: skipped `unsupported_sensor` (cloud has no per-point time) |
| `lidar-vehicle`, `imu-vehicle` | skipped `degenerate_frames` | (before the fix: `lidar-vehicle` inconclusive in 98 s, no axis constrained) |
| `lidar-lidar`, `ins-lidar`, `lidar-wheel_odometry` | skipped `missing_topic` | - |

Runtime: plan 2.7 s; all three `camera-imu` pairs 67 s (OpenCV, intrinsics from
`CameraInfo`); a repeat with a changed candidate 28 s.

**Known-bad.** A `--tf` file turning `camera_left/camera_link` by +1 degree about
its z axis leaves `camera-imu` for that camera `inconclusive`
(`no_judgeable_axes`), the same as the deployed candidate. The verdict does not
flip because the estimator constrains no axis on 36 s of driving; the check
refuses to judge rather than passing or failing. Whether the verdict would flip on a longer
recording was not tested.

Limitations: the 36 s bag is too short for the camera and GNSS estimators; the
concatenated cloud has no per-point time, so no LiDAR pair is solvable; because
the cloud and the IMU are in `base_link`, the physical Velodyne and IMU
extrinsics are not checkable from this bag at all. The IMU "frame" of
`camera-imu` is `base_link`, i.e. the check compares the camera with the
already-rotated IMU stream.

### Koide hard-localization `indoor_easy_01` and `indoor_easy_02`

- **Origin.** `koide_hard_localization/sequences/indoor_easy_0{1,2}` (Koide's hard
  point-cloud localization dataset, hand-held; sqlite3, 139 s). License not
  recorded with the local copy; not verified. The directory also holds a
  zero-byte `indoor_easy_0N_0.db3`; the real storage is the file named by
  `metadata.yaml`.
- **Frame tree.** `/tf_static` (2 messages) gives 6 frames, root `camera_base`:
  `camera_base` > {`camera_body`, `camera_visor`, `depth_camera_link`} and
  `depth_camera_link` > {`imu_link`, `rgb_camera_link`}. `/imu` maps to
  `imu_link` and both clouds to `depth_camera_link`.
- **Clouds.** `/points2/decompressed` is a plain x/y/z `PointCloud2`;
  `/points2/compressed` is Draco and is ignored (`ignored_reason`).

| Pair | Plan | Result |
| --- | --- | --- |
| `imu-lidar` | planned (`imu_link` / `depth_camera_link`) | before: skipped `unsupported_sensor` (no per-point time field); now runs on rigid scans, see "Clouds without per-point time" |
| `lidar-vehicle` (`--vehicle-frame imu_link`) | planned | not run (hand-held rig) |
| `imu-vehicle` | skipped `degenerate_frames` | - |
| others | skipped `missing_topic` | - |

Both sequences give the same plan (2.4-2.6 s). Limitation: the "LiDAR" is a depth
camera cloud without per-point time, so the IMU-LiDAR estimator, which deskews
each scan, cannot run in its normal mode; the rigid-scan result is in "Clouds without
per-point time" above.

### Aqua `beach_pond_ros2` (plan only)

- **Origin.** `aqua_localization/mbes_slam/beach_pond_ros2` (an underwater
  multibeam-sonar SLAM recording, converted to MCAP/ROS 2 from
  `seaward.science/files/pos-datasets/bag/beach_pond.tar.gz`; 9474 s). License not
  recorded locally; not verified.
- **Frame tree.** `/tf_static` (1 message) gives 14 frames, root `base_link`
  (`norbit`, `ustrain_imu`, `gps_port`, `gps_stbd`, `wassp`, `nortek_dvl`,
  `rbr_ctd`, `gantry`, ...). The MCAP + `/tf_static` path works end to end: all
  sensor topics map by header (`/norbit/detections` to `norbit`, both IMU topics
  to `ustrain_imu`, `/nav/sensors/navsat/ubx_pos/fix` to `gps_stbd`).
- **Plan.** `imu-lidar` (`ustrain_imu` / `norbit`), `gnss-lidar` (`gps_stbd` /
  `norbit`) and `gnss-imu` are `planned`; `ins-lidar` and `lidar-wheel_odometry`
  are skipped `missing_topic` and name `/nav/processed/odometry` as an odometry
  topic of unknown kind (`--topic-kind`); the rest are skipped. Plan time 2.9 s on
  a 9474 s bag.
- **Limitation.** The role comes from the message type, so the sonar
  `PointCloud2` counts as a "lidar" and the plan lists pairs that make no physical
  sense for a sonar. No estimator was run on it.

### Remaining gaps

- Per-point time is mandatory for `lidar-lidar` and `gnss-lidar`. `imu-lidar` runs
  clouds without it as rigid scans (see "Clouds without per-point time"); the
  Autoware concatenated cloud has none and is skipped `degenerate_frames` there.
- No bag here has the length or motion to give a verdict: the real-bag verdict
  tables remain the Phase B/C results above.
- `CompressedImage` cameras remain `unsupported_sensor` (none of these bags had
  one); no bag here exercised Livox `CustomMsg`.
- Sensor-class is not inferred beyond the message type (sonar counts as LiDAR).

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
