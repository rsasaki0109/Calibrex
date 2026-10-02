# Changelog

## Unreleased

- **`calibrex check` runs `lidar-lidar` and `gnss-lidar` on clouds without
  per-point time** as rigid scans, like `imu-lidar` (#99). Both pairs used the
  field only to deskew LiDAR odometry (`lidar-lidar`: the reference LiDAR's map;
  the target scans were never deskewed, so the target no longer needs the field;
  `gnss-lidar`: the odometry compared with GNSS), so rigid scans are defensible.
  `--lidar-lidar-deskew` and `--gnss-lidar-deskew` `auto|constant_velocity|none`
  (default `auto`: the field when present, bit-identical default path; rigid
  otherwise or when forced). `LidarLidarMapOptions.deskew` and
  `GnssLidarRunOptions.deskew`; artifacts record `options.deskew: none` and a
  limitation only in rigid mode. Measured rigid minus deskewed: lidar-lidar on
  NTU VIRAL tnp_01 at most 0.07 deg and 5 mm (stds unchanged); gnss-lidar on
  RTK-SLAM stadtgarten_seq2/seq1 y +41.5/+26.4 mm, x -2.7/-4.2 mm, z -5.3/-9.8 mm,
  time offset +64 ms, stds 1.7 to 2.5 times larger. Policy: lidar-lidar floors
  0.5 deg / 0.03 m with default thresholds; gnss-lidar (and gnss-imu) estimated at
  std <= 2 cm, control and translation floor 0.11 m. Optional schema fields:
  `CheckOptions.lidar_lidar_deskew`, `gnss_lidar_deskew`; `CheckPairRecord.deskew`
  also takes `constant_velocity` (`slac.calibration_check/v0.1` unchanged). No
  real bag we hold newly becomes judgeable (Autoware: one concatenated cloud in
  `base_link`, `gnss-lidar` with `--frame-map` is `inconclusive` on 36 s; Koide:
  one cloud, no GNSS). See the tutorial section "lidar-lidar and gnss-lidar on
  rigid scans".
- **`calibrex check` wires `camera-focal`**: the deployed focal lengths (Kalibr
  camchain or `CameraInfo`) are judged against the gyro as a scale
  `s = f_estimate / f_candidate`, `|s - 1|` against `max(k * std, floor)` with the
  new `--focal-scale-floor` (default 0.005). It consumes this run's `camera-imu`
  estimate (skipped `missing_dependency` when that is absent, failed or looser
  than 0.5 deg), is cached under the camera model and the `camera-imu` estimate,
  and equals `calibrex camera-imu focal` bit for bit on Hilti exp21 cam0-cam4.
  On exp21 only cam1 (fx, fy) and cam0 (fy) are judgeable; +3 % known-bad
  flips cam1 to `fail`, +1 % to `warn`; the other cameras stay `inconclusive`.
  Optional schema fields: `CheckPairRecord.focal_scale` (new `CheckFocalScale`),
  `CheckOptions.focal_scale_floor`; `slac.calibration_check/v0.1` unchanged
  otherwise. `run_camera_focal_check` splits out from the standalone runner
  (`max_seconds` option).
- **`calibrex check` runs `imu-lidar` on clouds without per-point time** as rigid
  scans (no deskew) instead of skipping them `unsupported_sensor`.
  `--imu-lidar-deskew auto|gyro|none` (default `auto`: the per-point time field
  when the cloud has one, so the gyro path is unchanged and bit-identical; rigid
  scans only when it is absent or when `none` is forced). Measured against the
  gyro deskew on RTK-SLAM construction_seq1 and Hilti exp21, the rigid rotation
  differs by at most 0.52 deg (yaw), the reported std grows 3.5 to 10 times, the
  lever arm becomes unobservable and the time offset absorbs about half a sweep
  (+48 to +51 ms). Policy: rotation floor 1.5 deg for rigid scans
  (`--rigid-scan-rotation-floor-deg`), axes estimated only at std <= 0.3 deg, a
  1.5 deg known-bad control. Optional schema fields:
  `CheckPairRecord.deskew`, `CheckOptions.imu_lidar_deskew` and
  `rigid_scan_rotation_floor_deg` (`slac.calibration_check/v0.1` unchanged);
  `ImuLidarRunOptions.deskew` and `ImuLidarTranslationOptions.deskew`; the
  rotation/translation artifacts record `options.deskew: none` only in rigid mode.
  Koide indoor_easy_01/02 (a depth-camera cloud) now give `pass (partial: roll,
  pitch, yaw only)`. See the tutorial section "Clouds without per-point time".
- README: a new top with a `calibrex check` hero image (`docs/assets/readme-check-hero.svg`,
  generated from the committed demo summary by `tools/generate_readme_check_hero.py`,
  light and dark aware, digest-bound to its sources), a feature strip, and a shorter nav.
- **`calibrex check` on real third-party bags with `/tf_static`** (Autoware
  all-sensors-bag1, Koide indoor_easy, Aqua beach_pond MCAP; see the tutorial's
  "Real bags with /tf_static"). Fixes: ROS 2 `CameraInfo`/`Image` payloads with
  CDR end padding no longer fail; PointCloud2 topics that are not raw point
  arrays (Draco) are ignored (`ignored_reason`, optional field on topic
  records); pairs whose two frames coincide are skipped with the new reason code
  `degenerate_frames` (enum widening, `slac.calibration_check/v0.1` unchanged).

## 0.5.0 - 2026-10-02

Highlights:

- **`calibrex check <bag>`**: a zero-config audit of the calibration deployed on a
  bag. It reads `/tf_static` (or `--tf`), infers topic roles, runs native
  estimators for 10 sensor pairs including GNSS and vehicle pairs, and reports
  per-axis verdicts with partial coverage and detection power. It caches
  estimators, checks rig closure, writes a self-contained HTML report (`--html`),
  and `calibrex convert kitti-raw` turns KITTI raw drives into bags.
- Pre-registered SOTA audits and a leaderboard: imu-lidar and lidar-vehicle are
  supported, lidar-lidar is refuted.
- Native methods for all 10 target sensor pairs.
- A browser calibration page.
- Camera-IMU static-window exclusion and a camera bound.

- **`calibrex check`, Phase D: rig closure, HTML report, the 1 degree demo.** The
  estimates of a run are now also checked against each other: parallel estimates of
  one frame pair and cycles (a fundamental cycle basis) are judged with the pairs'
  rule (`max(sigma_k * std, floor)`, first-order std assuming independent members,
  axes unobserved by any member left unchecked); composed pairs (`gnss-imu`) are
  excluded as derived. New optional `closures` field in `slac.calibration_check/v0.1`;
  a failing loop raises `overall_verdict` to at least `warn`, never `fail`. New
  `calibrex check --html PATH` and `calibrex render check.json --html PATH` write a
  self-contained HTML report. `tools/check_tf_injection_demo.py` and
  `docs/assets/calibrex_check_demo/` reproduce "a +1 degree yaw in the deployed tf
  fails" on the pooled KITTI development drives; the README gains a `calibrex check`
  section.

- **`calibrex check`, Phase C2: `gnss-lidar` and `gnss-imu` from the bag's
  NavSatFix topic.** New `calibrex.data.navsatfix_track` turns a `NavSatFix` topic
  into the GNSS track the lever-arm estimator uses (header-stamp time, ellipsoidal
  altitude, usable = `status >= 0` and `sqrt(trace(covariance)) <= 0.15 m`, which
  reproduces RTK-SLAM's `blt_std` exactly; unknown covariance keeps only GBAS
  fixes with a recorded fallback std). `gnss-lidar` judges the antenna lever arm in
  the LiDAR frame (x, y, z; the antenna's rotation axes are listed as unchecked)
  and is cached like `imu-lidar` (the fit does not use the candidate). `gnss-imu`
  composes this run's `gnss-lidar` and `imu-lidar` evidence as `calibrex gnss-imu
  compose` does, and is skipped (`missing_dependency`) when either is missing or
  failed. New `--gnss-max-duration-s` gives the GNSS pairs their own span (they
  need minutes of RTK-fixed windows); `calibrex gnss-lidar rtk-slam` gains
  `--max-seconds`. `slac.calibration_check/v0.1` gains the optional
  `options.gnss_max_duration_s`. On RTK-SLAM the bag path reproduces the `rtk.txt`
  path to 0.2 mm (same windows); against the CAD antenna offset the pairs judge only
  the axes with std under 1 cm (y on stadtgarten seq2, x on seq1), a +15 cm
  error on an observable axis fails and z is never checked. See
  `docs/tutorials/calibrex_check.md`.

- **`calibrex check`, Phase C1: vehicle pairs, validated on KITTI raw.** With
  `--vehicle-frame`, `lidar-vehicle`, `imu-vehicle`, `ins-lidar` and
  `lidar-wheel_odometry` now run their native estimators on bag data and are
  judged by the existing verdict rule (unobservable axes, such as roll without
  turns, are listed as unchecked). LiDAR odometry comes from `PointCloud2` scans
  (one pass shared by the three LiDAR pairs; a stream gap over 1 s starts a new
  segment, so several recordings can share a bag), the INS from a
  `nav_msgs/Odometry` topic, the wheel from an Odometry or `TwistStamped` topic;
  `imu-vehicle` uses the INS's body-frame velocity, not the raw IMU. New
  `--topic-kind TOPIC=wheel|ins` classifies odometry and twist topics; twist
  topics now fill the `ins`/`wheel` slots. `slac.calibration_check/v0.1` gains
  the optional `options.topic_kinds`.
  New `calibrex convert kitti-raw DRIVE... --output BAG [--calib-dir DIR]` writes
  a KITTI raw drive as a rosbag2 (`/velodyne_points`, `/oxts/{imu,fix,odometry,twist}`,
  transient-local `/tf_static` from `calib_imu_to_velo.txt`, `base_link` = the OXTS
  frame, no invented per-point time) with a provenance sidecar, using a new
  ROS-free rosbag2 sqlite3 writer (`calibrex.data.rosbag2_writer`) and CDR
  encoders (`ros_cdr_writer`, shared with the tests) and a `TwistStamped` decoder.
  On the development drives the bag run reproduces the KITTI-text CLIs exactly for
  lidar-vehicle, imu-vehicle and ins-lidar (wheel: differs only by the declared
  body-frame wheel proxy); +1 and +3 deg yaw on the Velodyne mount flips the
  pooled lidar-vehicle and lidar-wheel verdicts to `fail`, and is not seen on
  single drives whose yaw is unobservable. See
  `docs/tutorials/calibrex_check.md`.

- **`calibrex check` is 3x faster on a first run and instant on a re-check.**
  The `imu-lidar` and `camera-imu` estimates do not depend on the candidate, so
  their artifacts are cached (`--cache-dir`, default
  `$XDG_CACHE_HOME/calibrex/check`; `--no-cache` disables it) under a key of the
  bag digest, the estimator, every estimator option, the camera intrinsics and
  the calibrex version and source revision. A hit rewrites only the
  candidate-dependent reference fields; `lidar-lidar`, which starts at the
  candidate, is not cached. `slac.calibration_check/v0.1` gains optional
  `pairs[].evidence_from_cache` and `evidence[].from_cache`. Re-judging RTK-SLAM
  construction_seq1 with a +1 and +3 deg yaw error took 87 and 86 minutes and now
  takes 3 seconds, with identical verdicts and per-axis numbers. The IMU-LiDAR
  odometry passes are faster with bit-identical output (Hilti 40 s: 25 to 8.5 min;
  RTK-SLAM 180 s with lever arm: 94 to 28 min): voxel downsampling packs the
  voxel key into one int64 instead of a row-wise `np.unique`; the gyro rotation
  model is evaluated once per scan from a shared start orientation; the rosbag2
  SQLite reader filters by topic in SQL (the IMU topic of a long bag no longer
  drags every point cloud through Python), and the check reads the IMU once and
  keeps decoded scans in memory across passes (`--scan-memory-mb`). Defaults of
  every estimator are unchanged, so benchmark artifacts reproduce.

- **`calibrex check`, Phase B (verdicts).** Without `--plan`, `calibrex check`
  now runs the native estimator of `imu-lidar` (rotation, plus the lever arm
  unless `--no-imu-lidar-translation`), `lidar-lidar` (map registration started
  at the candidate) and `camera-imu` (needs OpenCV and intrinsics from a Kalibr
  camchain or a `CameraInfo` topic) and judges the deployed transform against it.
  Per estimated axis: `tolerance = max(k * std, floor)` (`--sigma-k` 3,
  `--rotation-floor-deg` 0.5, `--translation-floor-m` 0.02), `pass` within one
  tolerance, `fail` beyond two, `warn` between; unobservable axes are listed as
  unchecked, and a pair with no judgeable axis, or whose estimator failed its own
  held-out check, is `inconclusive`. Each axis records a detection-power
  self-test (`detectable_error = tolerance + |delta|`, and whether a 1 deg error
  would be flagged) without re-solving. The overall verdict is the worst pair
  (`fail > warn > inconclusive > pass`; `inconclusive` when nothing ran). Each
  estimator's schema'd artifact is written to `<output stem>_evidence/` and
  referenced by relative path and SHA-256. New options: `--pairs`,
  `--max-duration-s`, `--camera`, `--evidence-dir`, `--fail-on`,
  `--acceleration-unit`, `--detection-probe-deg`. `slac.calibration_check/v0.1`
  gains optional fields only (`options`, `evidence_dir`, per-pair `axes`,
  `unchecked_axes`, `time_offset`, `evidence`, ...) and new reason codes; plan-only
  artifacts stay valid. LiDAR support is generalised from Livox to any
  `PointCloud2` with a per-point time field (offset seconds, absolute seconds as
  Hesai writes it, or absolute nanoseconds), and the IMU-LiDAR and LiDAR-LiDAR
  estimators accept a time window. Real-data validation (Hilti, NTU VIRAL,
  RTK-SLAM, including known-bad yaw perturbations) is in
  `docs/tutorials/calibrex_check.md`. A pair with any unchecked attempted axis is
  marked `coverage: partial` (table: `pass (partial: roll only)`, summary
  `partial_pairs`, CLI warning), and an axis whose estimator known-bad control was
  not detected is left unchecked (`control_not_detected`).
- **`calibrex check`, Phase A (plan only).** `calibrex check BAG [--tf FILE]
  [--vehicle-frame FRAME] [--frame-map TOPIC=FRAME] [--plan]` audits the
  calibration deployed on a robot. It reads candidate extrinsics from the bag's
  `/tf_static` and from `--tf` files (URDF fixed joints, a small
  `slac.check_frames/v0.1` YAML, Kalibr camchain-imucam, RTK-SLAM `calib.yaml`,
  Hilti `lidar_calibration.yaml`), classifies sensor topics (LiDAR, IMU, camera,
  GNSS, odometry, twist), maps them to tree frames, and lists the pairs that can
  be checked, with a machine-readable reason for each skipped pair. No solver
  runs yet: every runnable pair is `planned`. New schemas
  `slac.calibration_check/v0.1` and `slac.check_frames/v0.1`. The rosbag2
  reader now decodes `tf2_msgs/msg/TFMessage` and `sensor_msgs/msg/NavSatFix`
  and can list a bag's topics and read one topic's first messages without a
  full scan. Vehicle pairs (`lidar-vehicle`, `imu-vehicle`) are opt-in: without
  `--vehicle-frame` they are skipped with `no_vehicle_frame`, because they assume
  ground-vehicle motion. See `docs/tutorials/calibrex_check.md`.
- **README refresh.** The README now covers all ten target sensor pairs, the
  pre-registered SOTA audits and leaderboard standings, and current honest
  verdicts; the duplicated quickstart and gallery GIF are removed and the
  solid-state section is condensed.
- **Camera-IMU evidence settings.** Near-static windows (accumulated rotation
  under 1.0 deg) are left out of the fit and the jackknife, and the rotation
  observability bound is 0.3 deg for cameras instead of the 0.1 deg inherited
  from IMU-LiDAR (`ImuLidarRunOptions.min_window_rotation_deg`, default 0 and
  so unchanged for IMU-LiDAR; `CameraImuRunOptions` defaults; CLI
  `--min-window-rotation-deg` and `--observable-rotation-std-deg`). The
  exclusion count, its reason, and the threshold are recorded in the artifact
  options; no schema changed.
  - Hilti 2022 development (exp21, exp07): verdicts go from 10
    `inconclusive` to 5 `pass`, 1 `warn`, 4 `inconclusive`. exp07 changes come
    from the threshold alone. The jackknife is not removed by the exclusion,
    accuracy against Kalibr is not systematically better, and cam2-cam4 on
    exp07 and cam3 yaw on exp21 stay unconstrained.
- **Camera-IMU std diagnosis.** `tools/diagnose_camera_imu_std.py` shows the
  reported std that keeps every Hilti verdict `inconclusive` is the window
  jackknife, not the analytic std: the tracked camera rotation carries a
  per-window scale of about 1-8 % on the forward cameras, and leaving a
  window out moves the fit by a similar amount. The 0.1 deg bound is
  inherited from IMU-LiDAR and is tighter than image-tracked rotations
  support (`docs/benchmarks/hilti_camera_imu.md`).
- **A pre-registered `lidar-lidar` audit is refuted** on three held-back
  NTU VIRAL recordings (`ntu-lidar-lidar-v1`, 2/4 gates).
  - Calibrex's fitted extrinsic is 0.49 deg and 7.7 cm from the rounded
    design value (gates pass), but it does not beat concurrent scan-to-scan
    (paired CI low −0.005) and the cross-recording consistency of
    0.3 deg / 3 cm is violated (1.14 on average).
  - Development was tnp_01 only; the claim was about accuracy against the
    design value and reproducibility, not a margin over a baseline.
  - Tools: `tools/score_ntu_lidar_lidar.py`,
    `tools/build_ntu_lidar_lidar_audit.py`.
- **A pre-registered `lidar-vehicle` audit is supported** on eight unseen
  KITTI drives (`kitti-lidar-vehicle-v1`, 3/3 gates).
  - Calibrex's motion-only rotation fits held-out LiDAR motion better than
    KITTI's calib_imu_to_velo (paired CI low 0.051, 9 of 9 blocks).
  - Its pitch and yaw agree with the OXTS-motion vehicle frame within
    0.21 deg on average (threshold 0.5).
  - Tools: `tools/score_kitti_lidar_vehicle.py`,
    `tools/build_kitti_lidar_vehicle_audit.py`.
- `calibrex camera-imu focal` checks a camera's focal lengths against the
  gyro, without a target (`slac.camera_focal_scale/v0.1`).
  - Per-axis camera/gyro rate ratios give `fx`/`fy` estimates. The
    optical-axis ratio is the control.
  - On Hilti 2022 exp21 the forward cameras agree with Kalibr within 0.5 %
    (cam1 `pass`); the side cameras scatter by 1-2 %.
- **Test fix:** the synthetic rotating-camera test rendered almost black
  frames (the texture offset was inverted). It now renders the full view.
- `calibrex lidar-wheel` calibrates a LiDAR against wheel odometry
  (`slac.lidar_wheel_odometry/v0.1`).
  - It estimates the LiDAR-to-vehicle rotation, the wheel-speed scale, the
    clock offset (started by a yaw-rate scan), and the lever.
  - `trajectory` takes any TUM trajectory and a wheel CSV. `kitti` uses the
    OXTS speed and yaw rate as a stand-in for wheels, since KITTI has none.
  - On KITTI dev drives the rotation matches lidar-vehicle within 0.004 deg.
    The verdict is `warn`: the held-out blocks prefer a different speed
    scale, and the 74 ms offset describes the OXTS velocity output.
- `calibrex imu-vehicle kitti` estimates `R_vehicle_imu` for the KITTI OXTS
  unit from its body-frame velocity and rates with the non-holonomic solver.
  It includes a closure reference against LiDAR-vehicle composed with
  `calib_imu_to_velo`; the closure agrees within 0.1 deg.
- **Fix:** the OXTS-motion vehicle frame used KITTI's level-frame
  `vf, vl, vu` / `wf, wl, wu` as if they were body-frame. They are now
  rotated into the body frame with each packet's roll and pitch.
  - LiDAR-vehicle pitch and yaw now agree with that frame within 0.06 and
    0.15 deg.
  - The earlier "HDL-64 odometry drift" explanation of a 0.5 deg pitch
    disagreement was wrong.
- `calibrex gnss-imu compose` composes GNSS-LiDAR and IMU-LiDAR artifacts
  into the GNSS antenna position in the IMU frame
  (`slac.gnss_imu_lever_arm/v0.1`).
  - Inputs are digest-pinned, std is propagated, and axes inherit
    unobservable inputs.
  - On RTK-SLAM the estimated axes agree with CAD within 1 cm (both
    recordings `inconclusive`).
- `calibrex lidar-vehicle kitti` estimates `R_vehicle_velodyne` from the
  vehicle's non-holonomic motion, as `slac.vehicle_frame_rotation/v0.1`
  (`calibrex.solvers.vehicle_frame_solver`).
  - It is compared both with KITTI's OXTS frame and with a vehicle frame
    derived from the OXTS velocities by the same solver.
  - On five development drives, pitch and yaw are estimated (0.04 and 0.05
    deg std) and roll is not (too few turns).
  - Yaw is 0.15 deg from the OXTS-motion vehicle frame. Pitch is 0.5 deg from
    both references, which points at the LiDAR odometry's vertical drift.
  - No claim (`docs/benchmarks/kitti_lidar_vehicle.md`).
- `calibrex lidar-lidar ros2` calibrates `T_reference_target` for two LiDARs
  in one ROS 2 bag, as `slac.lidar_lidar_extrinsic/v0.1`.
  - Target scans are registered to local maps from the reference LiDAR's
    odometry and solved jointly by robust point-to-plane Gauss-Newton, with
    held-out blocks, a block jackknife, and 1 deg / 5 cm controls.
  - On NTU VIRAL tnp_01 (development), five of six DoFs are estimated
    (roll's std 0.108 deg is just over 0.1). The held-out data reject the
    rounded design extrinsic (delta chi-square 48,901), about 0.5 deg of
    pitch and 5 cm in y and z away from the estimate.
  - Motion-based hand-eye could not recover the translation there, which is
    documented (`docs/benchmarks/ntu_viral_lidar_lidar.md`).
- The trajectory hand-eye solver also starts from the rotation that aligns
  the motion rotation axes, so arbitrary mountings converge. Its jackknife
  and reference differences use small rotations about the parent axes.
- Camera-IMU tracking now equalizes contrast (CLAHE) and tracks up to 600
  features at quality 0.001.
  - On the Hilti exp07 corridor, failed frame pairs dropped from 577 to 16
    of 1321 (cam0), and the forward cameras' std from 0.4-0.9 to 0.1-0.24 deg.
  - The side-looking cameras did not improve, and two got worse on exp21.
  - `tools/analyze_camera_imu_consistency.py` separates a rig-level
    IMU-frame offset from per-camera errors. On exp21 it finds about -0.4 deg
    about the IMU vertical axis, shared by all cameras; the cause is not
    identified.
- `calibrex camera-imu rotation`: targetless camera-IMU rotation, clock
  offset, and gyro bias from ROS 2 image and IMU topics (needs
  `calibrex[opencv]`).
  - Camera rotations come from tracked features, and are aligned with the
    gyro by the IMU-LiDAR rotation evidence.
  - The rotation artifact gains `sensor_modality` (`lidar`, `camera`, or
    `trajectory`; default `lidar`).
- On Hilti 2022 development recordings (exp21, exp07), the clock offset
  agrees with Kalibr within 0.35 ms, and the rotation is 0.0-0.7 deg from
  Kalibr on exp21.
  - Every camera is `inconclusive`: the std exceeds 0.1 deg.
  - A negative bias about the camera y axis appears on all five cameras,
    unexplained.
  - Tracking fails often in the exp07 corridor.
  - No camera-IMU SOTA claim is made (`docs/benchmarks/hilti_camera_imu.md`).
- Rotation jackknife spreads and reference differences are now small
  rotations about the sensor axes rather than Euler-angle differences, which
  broke near a +-90 deg pitch.
- Browser calibration at <https://rsasaki0109.github.io/Calibrex/app/>.
  - Upload an IMU CSV and a sensor trajectory (TUM), or generate a synthetic
    hand-held example. The page calibrates the IMU rotation, clock offset,
    gyro bias, and lever arm, with held-out evidence.
  - Calibrex, NumPy, and SciPy run under Pyodide, so no data leaves the page.
  - The page offers the schema-valid artifacts for download.
  - The docs workflow builds the wheel it installs
    (`tools/build_browser_wheel.py`).
- `calibrex imu-lidar trajectory` runs the same computation from files
  (`calibrex.evaluation.imu_trajectory`, `calibrex.data.imu_trajectory`,
  `calibrex.data.imu_trajectory_synthetic`).
  - The rotation and lever-arm artifact builders are now shared helpers
    (`rotation_artifact_from_evaluation`,
    `translation_artifact_from_evaluation`).
- `accelerometer_lever_arm/v0.2`, now the default, fits gravity per 2 s
  segment instead of per 10 s window, because the odometry's tilt drifts
  within a window.
  - Segments need at least 10 scans, and the segment sensitivity uses 3 s and
    4 s refits.
  - It was revised on spent recordings only, after six hypotheses were
    tested; the negative results are documented.
- A second pre-registered `imu-lidar` extrinsic audit
  (`mid360-imu-lidar-extrinsic-vs-li-init-v2`) is **supported** on the last
  unseen RTK-SLAM recording, `construction_seq2`.
  - Calibrex beats LI-Init by 4.13 mm mean span RMS (CI low 2.79 mm), and
    LI-Init's lever arm by 4.11 mm (CI low 2.87 mm).
  - The lever arm is (27.0, 17.9, -33.0) ± (5.6, 4.6, 7.9) mm; x is 16 mm
    from the design value.
  - `tools/build_mid360_imu_lidar_extrinsic_audit.py --round v2` rebuilds it.
    The scoring tool pins the first round's nuisances.
- A pre-registered `imu-lidar` extrinsic claim (rotation, lever arm, and
  clock offset against LI-Init) is **refuted** on two unseen RTK-SLAM
  recordings (`stadtgarten_seq1`, `construction_seq1`).
  - Calibrex's lever-arm run was `inconclusive` on both: x's segment-duration
    sensitivity is 11.0 and 13.3 mm, above the 10 mm bound. The
    pre-registration counts that as no estimate.
  - The rotation passes on both and replicates seq2 to within 0.05 deg.
  - Candidates are scored on one common odometry per recording
    (`score_imu_lidar_translation_candidate`, `tools/score_mid360_imu_lidar_extrinsics.py`,
    `tools/build_mid360_imu_lidar_extrinsic_audit.py`).
  - The leaderboard lists the refuted claim next to the supported rotation
    claim.
- `calibrex imu-lidar livox-translation` estimates the IMU lever arm
  (translation of `T_lidar_imu`) from the accelerometer, on top of a
  digest-pinned rotation artifact, as `slac.imu_lidar_translation/v0.1`.
  - Segment double integration against LiDAR odometry is linear in the lever
    arm. Gravity and the accelerometer bias are fitted per window and the
    velocity per segment.
  - The reported std includes a segment-duration sensitivity alongside the
    analytic and jackknife std, and held-out windows must detect a 20 mm
    shift of each axis.
  - MID360 hand-held (RTK-SLAM seq2): `pass`, at (21.3, 24.3, -38.5)
    ± (6.1, 5.2, 5.3) mm against the design value (11.0, 23.3, -44.1) mm.
    The vehicle recording is `inconclusive`: its planar motion leaves every
    axis unobservable.
- The first supported SOTA claim: `imu-lidar` rotation and clock offset of
  the Livox MID360 built-in IMU, against LI-Init, on two recordings from two
  dataset families (`docs/benchmarks/mid360_imu_lidar.md`).
  - LI-Init (GPL-2.0) runs only in a pinned container
    (`tools/external/li_init/`). `calibrex.importers.li_init` imports its
    result as an external-run artifact.
  - The claim, metric, and thresholds were pre-registered
    (`docs/benchmarks/mid360_imu_lidar_preregistration.yaml`) before any
    LI-Init score was computed. The audit protocol pins that file's SHA-256.
  - On RTK-SLAM seq2, the mean held-out span residual is 0.0123 rad/s for
    Calibrex and 0.148 rad/s for LI-Init (paired improvement 95 % CI
    0.090-0.185). On the driving recording, LI-Init returned no result.
  - `score_imu_lidar_candidate` scores a fixed rotation and clock offset,
    refitting only the gyro bias. Its `holdout_spans_s` scores every
    candidate on the same held-out time spans. The first audit build paired
    windows by each method's own segmentation, which shifted LI-Init's
    held-out windows and made the audit `refuted`. The thresholds were not
    changed by the fix.
- The MID360 hand-held IMU-LiDAR rotation now passes. On RTK-SLAM seq2,
  roll, pitch, and yaw are within 0.16 deg of the design value (std 0.02-0.03
  deg), the clock offset is 9.8 ms, and held-out windows detect every 1 deg
  control.
  - The cause of the earlier 1.4 deg yaw discrepancy was LiDAR-only
    constant-velocity deskewing, which cannot follow fast hand-held rotation,
    not coning as first reported. Odometry now accepts a `rotation_model`,
    and the evaluation deskews with gyro rotations mapped by the current
    estimate.
  - It iterates until every axis moves less than 0.25 std. It refuses a pass
    while the last pass still moves an axis by more than its std.
  - It records a `deskew_feedback_ratio`: the share of a deliberate 2 deg
    deskew error that survives in the estimate.
  - Rotation increments are compared with on-manifold gyro preintegration
    (cumulative orientations and first-order bias Jacobians); mean rates
    remain as the `mean_rate` model.
- Added IMU-LiDAR rotation, clock-offset, and gyro-bias evidence
  (`slac.imu_lidar_rotation/v0.1`, `calibrex imu-lidar livox`).
  - Mean LiDAR odometry rates between consecutive scans are aligned with
    interval means of the gyro (a cumulative integral, so evaluation is
    vectorized).
  - A Procrustes initialization handles arbitrary mountings.
  - Yaw-only vehicle motion leaves yaw unobservable, and the artifact reports
    it as such.
  - Livox ROS 2 bags are read without ROS through explicit stream profiles:
    point-time field and encoding (offset seconds or absolute nanoseconds)
    and acceleration units (m/s^2 or g). The MID360 design `T_lidar_imu` is
    the reference.
- LiDAR odometry can motion-compensate each sweep from per-point capture
  times under a constant-velocity model, re-registering once with the refined
  motion. On RTK-SLAM MID360 data the one-second distance error against RTK
  drops from 1.94 cm to 1.53 cm (p90 5.89 cm to 3.95 cm).
  The RTK-SLAM GNSS-LiDAR results were re-run with deskewing:
  - x now agrees across both sequences (5.1 and 5.2 cm).
  - The ~42 ms clock offset reported before was an artefact of undeskewed
    sweeps: they are stamped at sweep start, but their geometry sits
    mid-sweep.
- Odometry windowing for multi-sensor evaluations moved to
  `calibrex.evaluation.odometry_windows`.
- Added GNSS antenna lever-arm and clock-offset evidence against LiDAR
  odometry (`slac.gnss_lidar_lever_arm/v0.1`, `calibrex gnss-lidar rtk-slam`)
  with an RTK-SLAM dataset reader (RTK `rtk.txt`, Livox rosbag2 scans, CAD
  reference from `calib.yaml`).
  - The solver profiles out one ENU alignment per odometry window by weighted
    Procrustes, so the outer robust fit has four unknowns.
  - GNSS epochs are used as observed and the odometry is interpolated to them,
    with a blend-dependent pose variance. Interpolating the noisy GNSS track
    instead biased the clock offset toward mid-epoch values.
  - LiDAR odometry gains an incremental scan-to-local-map mode
    (`local_map_scans`) for sparse Livox scanners; on MID360 it cut the
    one-second distance error against RTK from 7.3 cm to 2.2 cm.
- Added INS/GNSS-LiDAR trajectory hand-eye calibration evidence
  (`slac.ins_lidar_hand_eye/v0.1`, `calibrex ins-lidar kitti`).
  - LiDAR motions come from a new vectorized point-to-plane scan-to-scan
    odometry that never uses the reference as a prior.
  - A joint solver estimates `T_ins_lidar`, the clock offset, and a reference
    trajectory scale. On planar vehicle motion it recovers yaw from the
    translation equations, which a rotation-first method cannot do.
  - Every DoF is classified as estimated, prior, or unobservable using the
    larger of its analytic and block-jackknife std. The model rejects a pass
    verdict while any extrinsic DoF is unobservable.
  - Held-out time blocks must detect each estimated DoF's known-bad shift
    (delta chi-square).
  - Same-rig KITTI drives can be pooled, and reference gaps are never
    interpolated.
  - Real-data findings are in `docs/benchmarks/kitti_ins_lidar.md`, including
    that pairing KITTI `sync` frames by directory position misaligns Velodyne
    and OXTS by hundreds of milliseconds after a dropped frame; frames are now
    paired by file index and timestamp.
- Generalized the SOTA claim audit to any sensor pair.
  `slac.sota_audit_protocol/v0.1` scopes a claim by modalities (camera, LiDAR,
  IMU, GNSS, INS, radar, RGB-D, wheel odometry, vehicle, robot arm), estimated
  quantities, and a method category, and `calibrex sota audit` evaluates it.
  Camera-LiDAR audits now run through the same engine, and their published
  schemas are unchanged. `calibrex sota leaderboard` builds
  `slac.sota_leaderboard/v0.1` standings per sensor pair from digest-pinned
  audit results, keeps refuted claims counted beside supported ones, and lists
  target pairs without a claim. The first leaderboard
  (`docs/benchmarks/sota_leaderboard.md`) records that no pair has a supported
  claim yet.
- Declared every artifact kind once in `calibrex.core.schema_registry`.
  Validation, `calibrex schema`, and auto-detection previously drew on seven
  hand-maintained tables that had drifted: `slac.remote_archive_selection/v0.1`
  and `slac.camera_lidar_confidence_calibration/v0.1` were accepted by their
  models but failed auto-detection, and `continuous-time-imu-intrinsics` was
  missing from `ValidationKind`. Accepted versions are now derived from each
  model's `schema_version` annotation. `calibrex schema all` also writes
  `schemas/schema_ledger.json` (kinds, current and accepted versions, schema
  digests, retired versions, and migrations), and `calibrex schema ledger`
  prints it. Retired versions fail with the retiring release and a remedy;
  `slac.doctor/v0.1` (replaced by `slac.environment_readiness/v0.1`) is the
  first, since its workflows lack the now-required `template_path`.
- Added `tools/check_schema_compat.py` and a CI job that fail when a released
  schema version is no longer readable, is narrowed without a version bump, or
  a released artifact no longer validates. Decided narrowings are recorded in
  `tools/schema_compat_acknowledged.yaml`.
- `slac.external_calibration_run/v0.1` keeps requiring a sha256
  `container_digest`, accepted as a v0.1 fix: a mutable image tag cannot
  reproduce a container run. External runs recorded with a tag must be
  re-imported with the resolved digest.
- The published `slac.result/v0.1` schema accepts an empty legacy provenance
  map again, matching the model and the omitted-map case. Such results remain
  blocked from production admission by `calibrex validate`.
- Fixed the A2D2 Pandey dataset manifest, which had violated
  `slac.dataset_manifest/v0.1` since v0.4.1. A new test validates every
  committed `slac.*` artifact; the digest-bound raw-replay stub config remains
  a tracked known violation.
- `calibrex doctor` now reports Python path isolation: ROS site-packages on
  `sys.path`, dependencies resolved from outside the active environment, and
  imported ROS modules. `core_ros_independent` is measured instead of always
  `true`, and a shadowed environment downgrades readiness to `warn`.
- pytest no longer autoloads third-party plugins, so the suite runs in a shell
  where ROS is sourced (ROS pytest plugins previously aborted collection). CI
  adds a ROS Jazzy job that runs the tests and `calibrex doctor` with
  `setup.bash` sourced, plus static and runtime checks that the package never
  imports ROS modules.
- Pinned ruff and mypy exactly and moved the strict mypy gate to
  `tools/typecheck-requirements.txt`, shared by CI and developers.

- Added independent evidence for imported Camera--IMU calibrations.
  `calibrex external-run evaluate-camera-imu` evaluates a Kalibr (or other
  external-run) `T_cam_imu` and `timeshift_cam_imu` on a separate
  `slac.camera_imu_motion_recording/v0.1`. It uses a middle-block temporal
  holdout, a train-only gyro bias, excitation gates, and eight signed
  rotation/time known-bad controls. The result is
  `slac.external_camera_imu_evidence/v0.1` with
  pass/warn/fail/inconclusive/blocked status and digest-bound provenance. A
  recording whose bytes match a declared fitting input is blocked.
  `calibrex external-run synthesize-camera-imu-fixture` writes the
  nonphysical Kalibr fixture used to prove the external-run contract with a
  Kalibr producer.

- Switched the public installation path to the verified GitHub Release wheel
  and removed the automatic PyPI publishing workflow following the maintainer
  decision to skip PyPI distribution.

- Added the continuous-time trajectory foundation. `slac.continuous_time_trajectory/v0.1`
  is a typed, schema-valid trajectory contract that declares the interpolation
  model, knot-domain validity (queries outside the domain are rejected, never
  clamped), and clock/capture-time semantics, and converts losslessly to the
  existing piecewise linear-slerp adapter. Analytic SE(3) manifold operations
  (`se3_manifold`) provide right-trivialized exp/log, the SO(3)/SE(3) left
  Jacobians and adjoint, the point-transform Jacobian, and two-knot screw
  interpolation Jacobians, all verified against finite differences. A sparse
  Gauss--Newton/Levenberg--Marquardt fitter (`continuous_time_sparse`)
  assembles the normal equations as a block-banded system in which every
  factor touches at most its two bracketing knots, estimating all knots on the
  manifold from body-frame point measurements and pose measurements (poses
  become four anchored point factors using only the analytic point Jacobian).
  `calibrex trajectory build-contract` and `calibrex trajectory fit` expose the
  workflow, and the fit result validates against
  `slac.continuous_time_trajectory_fit/v0.1` with provenance pinned to both
  input digests.

- Added schema-valid empirical SE(3) uncertainty evidence
  (`slac.empirical_se3_uncertainty/v0.1`). `calibrex camera-lidar
  empirical-uncertainty` refits the native probabilistic multi-frame refiner on
  deterministic contiguous temporal-block subsamples that never split a block
  across the train/holdout boundary, reports tangent-space intervals with
  Sidak-corrected joint family coverage against injected truth, retains weak or
  unobservable directions instead of dropping them, and applies an
  overconfident interval control that must be refuted before the
  PASS/WARN/FAIL policy can certify the intervals. Without a declared reference
  (`--stability-only`) the artifact reports the empirical spread with an
  INCONCLUSIVE policy. Every resample refit is retained as a digest-bound
  schema-valid result, and the artifact validates against its generated schema.

## 0.4.1 - 2026-08-11

- Promoted the no-download KITTI-shaped Camera-LiDAR evidence path into a
  strict five-minute demo. Its deterministic, generator-bound 300x300 fixture
  now detects 16/24 mandatory ±1 deg / ±0.10 m perturbation cases, passes all
  six falsification-policy gates without relaxing thresholds, and remains
  explicitly synthetic rather than a real-sensor accuracy claim. Added direct
  generator parity and strict end-to-end tests plus README commands that
  validate the result and verify its digest-bound provenance bundle.
- Added an external-only I2PNet KITTI-large adapter with pinned MIT source,
  checkpoint/archive/input digests, an explicit non-reference initial-transform
  contract, a Windows-safe pure-Torch neighbor backend, deterministic
  frame-derived NumPy/Torch sampling, enforced deterministic CUDA execution,
  and schema-valid diagnostic correspondence and aggregate pose outputs. The
  first two-frame KITTI raw development audit rejects the diffuse cost-volume
  correspondences
  (mean reliability `0.000032333` below the fixed `0.01` gate), while the
  intended direct pose output improves the fixed `10 deg / 1.37049 m`
  perturbation to `0.04847 deg / 0.08211 m`. Added a frozen, schema-valid pose
  initializer protocol and subprocess runner that binds the problem, adapter,
  checkpoint, frames, perturbations, failure policy, and every generated
  initial/manifest/pose digest. Its disjoint three-frame, four-perturbation
  development matrix completes all trials but records 0/4 strict hits at the
  prespecified `<0.5 deg` and `<0.2 m` gate; mean error changes from
  `8.1046 deg / 0.40395 m` to `6.7043 deg / 0.37994 m`. This remains
  initializer-development evidence and does not alter a release or SOTA gate.
- Added a digest-bound pose-initializer failure-analysis artifact and external
  I2PNet analyzer. It replays all 12 frame corrections, reproduces benchmark
  metrics, and decomposes each required/predicted correction into signed
  parallel gain and orthogonal leakage. The frozen single-axis rotation gains
  are `x=0.002904`, `y=0.957952`, and `z=-0.000960`; translation gains are
  `x=1.374653`, `y=-0.023960`, and `z=0.949139`. This identifies a strong
  development-set axis anisotropy without claiming a causal network defect.
- Added a schema-valid, development-only Camera-LiDAR provider-support
  comparison and CLI. It requires every confidence calibration to share the
  exact problem, threshold grid, split seeds, support gates, and excluded
  evaluation datasets; ranks a support-first diagnostic point while keeping
  rejected calibrations runtime-ineligible; and records complete source
  digests and provider identity. The first KITTI-360 drive 0002 comparison
  rejects adapter v0.3, v0.4, and v0.5 because every candidate has zero
  worst-split geometric holdout inliers. Adapter v0.5 is retained only as the
  diagnostic leader, not as a runtime lock or SOTA result. A second disjoint
  development capture on drive 0003 also rejects the outdoor and indoor
  SuperGlue checkpoints: outdoor retains 10/1,049 geometric inliers at its
  diagnostic point and indoor retains 0/276, while both have zero worst-split
  holdout support. Diagnostic tie-breaking now prefers frame and accepted
  support when every geometric metric is tied, instead of reporting an empty
  highest-threshold operating point.
- Completed the Camera-LiDAR v0.7 `integrated-r5-numba-v03-v02` release audit
  without retuning the two evaluation sequences. At the frozen
  `(0.5 deg, 0.5 m)` six-DoF cell, KITTI raw 0018 reached 85/200 strict hits
  and failed its 88% gate, while KITTI-360 drive 0000 reached 161/200 and
  passed its 76.5% gate; both had zero D2D execution failures. The paired
  five-seed falsification packs retain all 1,000 cases per dataset. Raw records
  529 failure-aware refinement failures, while KITTI-360 rolls back all 1,000
  unsupported candidates. Both raw-recomputed 1,007-artifact bundles verify,
  and the digest-frozen learned-targetless SOTA audit returns `refuted` with
  external comparable baselines and independent-rig evidence still missing.
- Added schema-valid post-hoc Camera-LiDAR correspondence-quality diagnostics
  for initializer, candidate, and selected poses. The final reports expose the
  provider bottleneck directly: 112/2,834 accepted correspondences and 7/25
  supported frames on raw, versus 33/2,269 and 2/25 on KITTI-360. These reports
  record that holdout was not used for selection and are provenance-bound to
  the exact refinement result and correspondence artifact.
- Added the Phase 3 provider-neutral probabilistic 2D--3D correspondence
  artifact and an optional Apache-2.0 OpenCV PnP-RANSAC adapter with
  confidence filtering, covariance-aware diagnostics, tests, and complete
  provider/input provenance. The correspondence contract now distinguishes
  dataset-license provenance from the provider's code/model license; ablation
  evidence is marked unverified when the dataset license or source digests
  are undeclared.
- Added a D2D-initialized native multi-frame probabilistic pose refiner with
  covariance/outlier/reliability weighting, robust loss, bounded SE(3)
  correction, disjoint frame holdout, complete candidate traces, and explicit
  uncertainty ablation switches.
- Added the v0.2 integrated falsification contract around that refiner. A
  candidate is selected from fit evidence only and rolls back to the D2D
  initializer on non-convergence, negligible improvement, correspondence
  support loss, non-finite evidence, or correction-bound proximity. Rejected
  candidates and holdout outcomes remain in the evidence, complete trace
  directories must share one solver/backend identity, and reusable run IDs no
  longer embed the obsolete `v06` release label.
- Added explicit D2D candidate-trace initialization for probabilistic
  multi-frame refinement and its paired ablations. The trace must pin the same
  problem digest and frame pair; results record its ID, digest, solver status,
  and hit decision. Runs without a trace declare `problem_initial_transform`
  and are not mislabeled as solved-D2D initialization evidence. The generic
  method ID is now
  `probabilistic_multiframe_refinement/v0.3`; the v0.2 and old v0.1 IDs remain
  accepted for artifact compatibility. Generated result and benchmark method
  provenance now record the same v0.3 implementation version.
- Added a paired probabilistic-refinement ablation benchmark that executes the
  full estimator and three one-factor removals over fixed frame-split seeds,
  retains every schema-valid result, and reports failure-aware paired
  bootstrap comparisons with both input digests. Results and ablations now
  include translation distance and quaternion-geodesic rotation error to the
  problem's digest-pinned reference, making the planned cm/deg gates directly
  auditable alongside held-out reprojection.
- Added a ROS-independent joint camera--LiDAR continuous-time solver boundary
  using per-point firing times, supplied body twists, probabilistic image
  residuals, bounded extrinsic/clock refinement, disjoint holdout evidence,
  and an explicit weak-motion rejection gate.
- Added rolling-shutter row-time deskew plus fixed-clock, no-per-point-time,
  no-rolling-shutter, and no-covariance temporal ablations. The paired
  multi-split benchmark retains every result, failure, source digest, and
  bootstrap comparison. Optional frozen extrinsic/time references now produce
  initial/final rotation, translation, and clock errors in each result and
  paired ablation, making injected-recovery gates directly auditable.
  Continuous-time problems separately declare their dataset license; absent
  license provenance marks the ablation benchmark unverified.
- Added a ROS-independent, provenance-pinned piecewise SE(3) body-trajectory
  adapter with linear translation, shortest-arc quaternion interpolation, and
  strict no-extrapolation behavior. Continuous-time camera--LiDAR refinement
  now uses this non-constant motion model when supplied while retaining the
  constant-twist input as a backward-compatible fallback. A CLI adapter
  attaches existing schema-valid `T_world_body` trajectory artifacts while
  preserving both input digests and frame semantics.
- Added a digest-frozen Camera-LiDAR SOTA audit protocol/result and CLI. It
  evaluates every numerical/integrity gate, dataset-family and independent-rig
  coverage, and can emit only supported, refuted, or incomplete verdicts;
  schema validation prevents an unsupported success label.
- Extended benchmark distributions with p90 and p95 and made mean, median,
  p90, p95, maximum, and failure rate individually addressable by frozen SOTA
  audit requirements, so tail failures cannot be hidden behind a mean. Paired
  bootstrap improvement confidence bounds can also be frozen against an
  explicit reference/candidate method pair, while runtime and peak-memory
  mean/p95 statistics are directly auditable. Evidence whose benchmark
  provenance declares `data_verified: false` is rejected, and result
  validation now enforces the semantics of all three verdicts rather than
  guarding only `supported`. Optional smoke requirements cannot inflate the
  achieved dataset-family or independent-rig coverage counts.
- Added a schema-valid Camera-LiDAR D2D research pipeline with frozen
  Fibonacci-sphere protocols, complete candidate traces, Bull's Eye artifacts,
  an external FeatDepth provider adapter, rectified KITTI problem construction,
  deterministic parallel execution, and a 200/200-hit KITTI raw 0018
  rotation-only reproduction gate.
- Replaced the D2D z-buffer's global stable sort with an exactly equivalent
  linear reduction and added an optional fused Numba CPU projection adapter.
  Six-DoF v0.3 traces pin the backend and dependency version, refuse mixed
  backend resume or mixed full-trace falsification input, and preserve the
  NumPy default; full 325-evaluation and official-frame regressions retain
  identical numerical results.
- Completed the frozen KITTI raw 0018 rotation matrix at `1, 2, 10, 20 deg`
  with 200 deterministic directions per level. The `1, 2, 10 deg` levels
  recovered 200/200; `20 deg` recovered 178/200 with zero computational
  failures, while its `17.0067 deg` p90 and `28.4920 deg` p95 retain the
  direction-dependent capture failures that its `0.2425 deg` median would
  otherwise hide. All new traces and aggregate artifacts are schema- and
  digest-validated.
- Added the KITTI-360 half of the D2D reproduction path: a pinned external
  MiDaS v3.1 provider, official MEI fisheye projection, official
  pose/azimuth-based Velodyne motion compensation with a schema-valid
  provenance manifest, KITTI-360 problem construction, and CLI support. The
  frozen `10 deg` gate completed at 200/200 hits with zero failures (mean
  `0.3812 deg`, maximum `0.4676 deg`), matching the paper's 100% target.
- Completed the frozen KITTI-360 rotation matrix at `1, 2, 10, 20 deg` with
  200 deterministic directions per level. The `1, 2, 10 deg` levels recovered
  200/200; `20 deg` recovered 184/200 with zero computational failures. Its
  `0.3882 deg` median contrasts with a `28.4409 deg` p95 and `45.4505 deg`
  maximum, retaining the direction-dependent capture failures. All new trace,
  benchmark, and Bull's Eye artifacts are schema- and digest-validated.
- Added a digest-pinned A2D2 MiDaS/provider and pre-registered camera-view D2D
  smoke path. The frozen five-direction `10 deg` run completed 5/5 trials but
  failed recovery at 0/5 hits; the two-frame/pre-registration limitation and
  negative result are retained rather than weakening the gate.
- Added digest-checked resumable Camera-LiDAR trial execution plus a bounded
  six-DoF D2D solver, paired Fibonacci rotation/translation protocols,
  schema-valid candidate traces, benchmark aggregation, and CLI commands. The
  first real KITTI-360 `(0.5 deg, 0.5 m)` smoke converged to
  `0.2505 deg / 2.08e-17 m` and passed the frozen hit gate.
- Completed the first full six-DoF matrix cell on KITTI raw 0018 at the frozen
  `(0.5 deg, 0.25 m)` perturbation. All 200 trials converged with zero
  computational failures, but only 84/200 passed the strict
  `<0.5 deg / <0.20 m` accuracy gate. Rotation error was `0.5230 deg` mean,
  `0.5239 deg` median, and `0.7228 deg` p95; translation error was `0.1447 m`
  mean, `0.1421 m` median, and `0.1803 m` p95. The negative 42% hit-rate result,
  all trace identities/digests, and schema-valid aggregate provenance are
  retained without weakening the gate.
- Corrected the UniCalib adapter's default provenance URL to the official WACV
  2026 `han-15/UniCalib` repository and locked it with a unit assertion. Updated
  the SOTA evidence review for TLC-Calib's May 2026 code/data release while
  retaining its non-commercial, external-process-only license boundary.

- Added a digest-locked fixed-frame KITTI raw Camera-LiDAR input artifact and
  recovery benchmark comparing a scalar-reference Pandey I2I optimizer with a
  vectorized, safety-gated deterministic coarse-to-fine path. On the fixed
  official KITTI raw 0005 input, all paired accuracy metrics were identical
  while mean trial runtime improved by 1.719x.
## 0.4.0 - 2026-07-29

- Extended `calibrex doctor` from an environment check into an optional
  dataset-readiness workflow with type inference, quality and degeneracy
  diagnostics, compatible-workflow suggestions, a versioned
  `slac.doctor/v0.1` artifact, generation provenance, and validation support.
- Added local and GitHub-hosted Calibration CI with `calibrex ci`, a composite
  action, Step Summary rendering, enforced falsification and protocol gates,
  digest-bound inputs and custom policies, provenance-bound SVGs, and the
  versioned `slac.calibration_ci/v0.1` decision artifact.
- Added the schema-versioned `slac.external_calibration_run/v0.1` contract for
  digest-bound subprocess, container, precomputed, and imported calibration
  results; migrated the Koide adapter to it and added a ROS-free Kalibr
  camchain importer with explicit frame/time conventions and failure states.
- Added a reproducible full-scale KITTI raw Camera-LiDAR falsification runner
  with a frozen reference/known-bad protocol, selected-frame and raw-input
  SHA-256 provenance, independently verified evidence bundles, an honest
  PASS/FAIL/INCONCLUSIVE decision artifact, and an opt-in official-data test.
- Prepared the public launch surface with a concrete five-minute quickstart,
  an approachable documentation home, a repository social preview, citation
  metadata, and corrected issue links.
- Added guarded PyPI trusted publishing after a GitHub release is published,
  including tag-to-package-version checks and clean wheel smoke tests.
- Replaced the abbreviated license notice with the complete Apache-2.0 text and
  added a distributable `NOTICE`.
- Restored the Calibrex product, Python distribution/package, CLI, and source
  path. Existing `slac.*` schema IDs, `slac_version`, `slac_native`, and
  versioned protocol/policy IDs remain the stable legacy wire namespace.
- Added provenance-bound SVG evidence cards via
  `calibrex render --format evidence-card` and schema-source comparison tables
  via `calibrex compare --format evidence-table`, with deterministic README
  artifacts and drift tests.
- Reworked the README around a concise evidence-first story, generated visual
  summaries, an honest accepted-vs-known-bad comparison, public-data demos, and
  a compact calibration coverage matrix.

## 0.3.0

- Trajectory artifact (`slac.trajectory/v0.1`, schema #18) with ground-truth-free
  quality gates (kinematic health, interpolation clamp fraction, cross-segment
  drift proxy) on motion-compensated online runs.
- Two-pass and native-deskew KISS-ICP odometry modes at rosbag conversion;
  native deskew cut Indoor02 identity selftest translation error ~9.5 cm → ~4.3 cm
  (~55%) with modest rotation regression (~1.2° → ~2.0°); two-pass odometry
  FAILed its own trajectory gate (cross-segment RMSE 0.366 m).
- Anchored temporal time-offset estimation (`time_offset_anchor: initial`),
  dual-mode adapted/anchored provenance, joint extrinsic/temporal separability
  evidence row, and validation-only `inject_time_offset_s` (anchored mode
  tracked +50 ms injection exactly on selftest; adapted mode absorbed it).
- LiDAR-IMU rotation evidence (`lidar_imu` family): `sensor_msgs/Imu` CDR
  decoding, `--imu-topic` bag conversion with OS1 IMU restamp, symmetric
  rotation-rate smoothing (holdout RMSE ~8 deg/s → ~3.6 deg/s, passing 5 deg/s
  gate), per-axis observability, known-bad rotation probes, and supporting-only
  gravity consistency (4.42° on Indoor02).

## 0.2.0

- Pure-Python rosbag2/MCAP reader (sqlite3 + MCAP, CDR PointCloud2 and Odometry).
- Odometry-aware online motion compensation for rosbag2 replays with world-frame
  source maps, odometry-extrapolation gate, and clock-domain restamping.
- KISS-ICP rig-frame odometry path and TIERS Indoor02 real-data validation
  (identity self-consistency control: ~10× extrinsic error reduction with motion
  compensation vs static control).
- Per-point deskew via `sensors.<name>.point_time_field` on rosbag2 online runs.
- Camera-LiDAR projection promoted to ADR-0004-protocol evidence rows with
  family-aware assessment policy gates and KITTI committed-sample demonstration.
- Temporal time-offset perturbation probes and 1D holdout-RMSE estimator on
  motion-compensated online runs (synthetic ±5 ms recovery; honest real-data
  degeneracy reporting).
- Project rename Calibrex → slac (ADR 0006).

## 0.1.0-alpha.1

- Project bootstrap.
- Typed config and result models.
- CLI skeleton with `doctor`, `schema`, `validate`, `verify`, `assess`, `evidence`, `init`, `inspect`, `calibrate`, `evaluate`, `render`, `visualize`, `compare`, `report`, and `export`.
- Frame graph and SE3 utilities.
- Dataset manifest schema, filesystem dataset adapter, timestamp normalization, and MCAP adapter boundary.
- HTML report, report sidecars, and PASS/WARN/FAIL quality aggregation.
- Metric registry, deterministic holdout splitting, and threshold-based metric grading profiles.
- RGB-D Open3D SLAC adapter boundary and example config.
- Backend-neutral graph problem compiler and `slac compile`.
- Public dataset catalog, TUM RGB-D config, KITTI raw config, and direct TUM downloader helper.
- KITTI raw loader and fixed-mounted LiDAR prior factor.
- KITTI `calib_velo_to_cam.txt` importer for fixed-LiDAR initial transforms.
- Autonomous-driving starter profile with Radar and Autoware export surface.
- Public Livox Horizon-Horizon PCD evidence demo for rigidly mounted solid-state LiDAR pairs.
- LiDAR pair evidence metrics, known-bad perturbation cases, holdout point-to-plane summaries, and evidence protocol metadata.
- Machine-readable `summary.json`, `metrics.json`, `observability.json`, `degeneracy.json`, and `evidence.json` report sidecars.
- Machine-readable `comparison.json` artifacts with metric, transform, evidence summary, and evidence protocol compatibility comparisons.
- Static JSON schemas for config, result, comparison, dataset manifest, and report sidecars.
- Bulk schema generation via `slac schema all --output-dir schemas`, with schema drift tests and CI smoke coverage.
- 3D rig viewer artifact for reference, candidate, and estimated extrinsic comparison.
- Falsification assessment framework: `slac assess` applying a policy to evidence, policy artifacts, `--enforce` exit codes, per-DoF known-bad challenge summaries, a fixed support denominator, mandatory challenge verification, and assessment recomputation verification inside evidence bundles.
- Evidence bundle verification: `slac verify`, an `evidence-bundle-verification` schema, a `--require-raw-recomputed` raw recomputation gate, a verified-raw-inputs requirement, and structured verification claims.
- `slac render` and `slac evidence` commands for producing report and evidence artifacts from an existing result without recomputing metrics.
- Protocol and policy JSON schemas, and `compare --enforce-compatible` protocol compatibility enforcement.
- README GIF gallery generation (`generate_calibration_evidence_gif.py --readme-gallery`) with a provenance manifest and schema-validated visual modes.
- Local release smoke helper, wheel smoke coverage in CI, and a draft GitHub release workflow.
- Configurable public-dataset frame sampling: `DatasetConfig.sample_limit` and `slac inspect --sample-limit`.
- N-way `slac report-compare` command with labeled results, protocol-compatibility gating, metric-family rankings, and a `report_comparison` schema (16 schemas total).
- Pure-Python rosbag1 (v2.0) reader with PointCloud2 decoding (`none`/`bz2`, optional `lz4` extra), `slac inspect --type rosbag1`, and the TIERS LidarsCali example config.
- Native LiDAR point-to-plane solver backend (`solver.backend: native_lidar_point_to_plane`) wired into `slac calibrate`, with real rank/condition-number/weak-DoF observability replacing the `uncomputed_alpha_backend` stub and point-to-plane metrics in results.
