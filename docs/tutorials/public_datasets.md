# Public Datasets

Calibrex examples should prefer public datasets over synthetic-only workflows.
Large datasets are not committed to the repository. Instead, Calibrex keeps
small manifests, configs, and source metadata.

List known public datasets:

```bash
calibrex public-datasets list
calibrex public-datasets show tum_rgbd_freiburg1_xyz
```

RGB-D / Open3D SLAC:

```bash
python3 tools/download_public_dataset.py tum_rgbd_freiburg1_xyz --output-dir data/public
calibrex calibrate examples/public_datasets/tum_rgbd_freiburg1_xyz/config.yaml
```

Multi-plane LiDAR-camera calibration on ACFR's public real VLP-16 target poses:

```bash
python3 tools/download_public_dataset.py acfr_vlp_plane_poses --output-dir data/public
calibrex calibrate examples/public_datasets/acfr_vlp_plane_poses/config.yaml
calibrex verify outputs/acfr_vlp_plane_poses/bundle.json
```

The run evaluates two independent native baselines on the same capture split:
Zhang-Pless plane-only alignment and a Verma-style board-centre/normal robust
Procrustes solver. The latter reports centre/normal/offset holdout closure, a
six-DoF joint spectrum, and twelve known-bad controls. On the pinned real VLP-16
poses its gates PASS (1.17 cm centre RMSE, 1.68° normal RMSE, 12/12 controls),
while the 1.76 cm / 0.219° inter-method transform delta remains an ungated WARN
diagnostic. The upstream ROS/PCL feature extractor stays outside the core and
its pinned commit, Apache-2.0 license, input digest, and source URL are recorded
as provenance.

Seven native hand-eye baselines plus Zhuang-Roth-Sudhakar, Shah, Li-Wang-Wu,
and the closed-form and nonlinear Dornaika-Horaud robot-world/hand-eye
estimators on ETHZ ASL's real robot-arm pose streams:

```bash
python3 tools/download_public_dataset.py ethz_hand_eye_robot_arm_real \
  --output-dir data/public
calibrex calibrate \
  examples/public_datasets/ethz_hand_eye_robot_arm_real/config.yaml
calibrex verify outputs/ethz_hand_eye_robot_arm_real/bundle.json
```

The comparison runs Park-Martin, Tsai-Lenz, Daniilidis, Horaud-Dornaika
closed-form/nonlinear, Chou-Kamel, and the incremental Andreff-Horaud-Espiau
method with one disjoint relative-motion protocol. Zhuang-Roth-Sudhakar, Shah,
Li-Wang-Wu, and both Dornaika-Horaud methods solve absolute-pose `AX=YB`
(`Z` in the latter papers)
for both transforms on one common leakage-safe split. The Zhuang implementation
reports its paper-specific `a0`/`z0` failure margins; Dornaika-Horaud reports
quaternion double-cover sign synchronization and the nonlinear estimator's full
24-column local spectrum. Closure and all minimum-width gates pass on the public
sample, while the default known-bad gate remains INCONCLUSIVE; the documented
negative result is not converted into a PASS by threshold tuning.

Solid-state LiDAR-to-LiDAR evidence demo:

The Livox workflow now carries a schema-validated solid-state profile through
the result.  See [solid-state LiDAR calibration](../concepts/solid_state_lidar.md)
for point-time/integration semantics, temperature declarations, distance/FOV
coverage, and the independent-holdout limitation of this two-PCD sample.

```bash
calibrex demo livox-evidence --output-dir outputs/livox_horizon_horizon_pcd_sample
```

The demo command downloads the public Livox PCD sample when needed, writes a
materialized `demo_config.yaml`, recomputes `result.yaml`, renders
`evidence.json`, `protocol.json`, `transforms.json`, `assessment.json`, and
`policy.json`, and verifies `bundle.json`.

The same flow can be run step by step:

```bash
python3 tools/download_public_dataset.py livox_horizon_horizon_pcd_sample --output-dir data/public
python3 tools/generate_calibration_evidence_gif.py --readme-gallery
calibrex public-datasets show livox_horizon_horizon_pcd_sample --json
calibrex inspect data/public/livox_horizon_horizon_pair --type livox-pcd --json
calibrex calibrate examples/public_datasets/livox_horizon_horizon_pcd_sample/config.yaml
calibrex evidence outputs/livox_horizon_horizon_pcd_sample/result.yaml \
  --output outputs/livox_horizon_horizon_pcd_sample/evidence.json
calibrex render examples/public_datasets/livox_horizon_horizon_pcd_sample/cached_evidence_result.yaml \
  --output-dir outputs/livox_horizon_horizon_pcd_sample
calibrex validate outputs/livox_horizon_horizon_pcd_sample/evidence.json --kind report-evidence
calibrex validate outputs/livox_horizon_horizon_pcd_sample/assessment.json --kind assessment
calibrex validate outputs/livox_horizon_horizon_pcd_sample/protocol.json --kind protocol
calibrex validate outputs/livox_horizon_horizon_pcd_sample/transforms.json --kind transforms
calibrex validate outputs/livox_horizon_horizon_pcd_sample/policy.json --kind policy
calibrex assess outputs/livox_horizon_horizon_pcd_sample/evidence.json \
  --policy outputs/livox_horizon_horizon_pcd_sample/policy.json \
  --output outputs/livox_horizon_horizon_pcd_sample/reassessment.json
calibrex assess outputs/livox_horizon_horizon_pcd_sample/evidence.json \
  --policy outputs/livox_horizon_horizon_pcd_sample/policy.json \
  --enforce
calibrex verify outputs/livox_horizon_horizon_pcd_sample/bundle.json \
  --output outputs/livox_horizon_horizon_pcd_sample/verification.json
```

## TIERS LidarsCali ROS 1 bag (Livox Horizon + Avia)

The TIERS `LidarsCali` sequence records a Livox Horizon
(`/livox/lidar`) and a Livox Avia (`/avia/livox/lidar`) in one ROS 1 bag.
Calibrex reads the bag directly with a pure-Python rosbag v2.0 parser
(`slac.data.rosbag1`); no ROS installation is required. Decoding
`sensor_msgs/PointCloud2` payloads to arrays requires numpy
(`pip install "Calibrex[rosbag1]"`); bags with `lz4` chunk compression
additionally need `pip install "Calibrex[rosbag1-lz4]"` (`none` and `bz2`
compression work out of the box).

The bag is multiple gigabytes, so Calibrex never downloads it automatically.
Fetch it manually from the upstream dataset:

1. Open the TIERS dataset repository:
   <https://github.com/TIERS/tiers-lidars-dataset> and follow its download
   table to the `LidarsCali` sequence (hosted on the University of Turku
   SharePoint; the direct link is recorded in
   `examples/public_datasets/tiers_livox_lidars_cali/manifest.yaml` under
   `provenance.download_url`).
2. Save the bag as `data/public/tiers_lidars_cali/LidarsCali.bag`
   (the path declared in the example config).

Then inspect and run the LiDAR-pair evidence config:

```bash
calibrex public-datasets show tiers_livox_lidars_cali --json
calibrex inspect data/public/tiers_lidars_cali/LidarsCali.bag --type rosbag1 --json
calibrex calibrate examples/public_datasets/tiers_livox_lidars_cali/config.yaml
```

The offline config uses the native ROS-independent point-to-plane adapter. It
reserves later target capture windows for holdout and records whether the
source map was cut at the holdout time boundary. With
`require_independent_holdout: true`, too few target windows or an unverifiable
Livox/ROS clock relationship is an inconclusive evidence outcome, not a
successful calibration claim.

In the downloaded TIERS bag, Livox `timebase` values are in a since-boot-like
domain while ROS bag records are in the recording epoch. The ROS 1 adapter
therefore uses the bag record timestamp as the replay clock and applies
`capture_ns = bag_record_timestamp_ns + offset_time_ns`; it does not substitute
the Livox timebase with a message header stamp. Results record
`point_time_clock_mapping_by_sensor`, including the residual and stable/unstable
decision. The current TIERS run measured approximately 1 ms residuals and
classified both Livox mappings as stable.

For the point-time ablation, materialize signed timing controls and both deskew
modes from the online mixed-sensor config:

```bash
python tools/run_livox_time_ablation.py \
  examples/public_datasets/tiers_livox_lidars_cali/online_mixed_config.yaml \
  --output-dir outputs/tiers_livox_time_ablation \
  --include-no-deskew
calibrex validate outputs/tiers_livox_time_ablation/ablation_manifest.yaml \
  --kind livox-time-ablation
```

The manifest binds every materialized config/result by SHA-256. A nonzero
`inject_time_offset_s` is a validation-only known-bad timing control; it is not
reported as a recovered clock estimate. The `deskew_off` variants still
decode and report `offset_time`, but do not consume it for pose interpolation.

### Camera-LiDAR measured capture-time evidence

The same bag contains RealSense color images, Velodyne VLP-16 PointCloud2
messages with a measured `time` field, and a VRPN pose stream. Run the native,
ROS-independent capture-time adapter with:

```bash
calibrex calibrate \
  examples/public_datasets/tiers_livox_lidars_cali/camera_lidar_capture_time_config.yaml
calibrex validate outputs/tiers_camera_lidar_capture_time/result.yaml --kind result
calibrex verify outputs/tiers_camera_lidar_capture_time/bundle.json
```

This example deliberately does not estimate an extrinsic or clock offset. The
bag has no declared camera-to-LiDAR point identities, and the constant transform
between the VRPN `UWBTest` rigid body and Velodyne is unpublished. It instead
tests whether measured per-point firing times can be applied at real camera
exposures and whether the sequence makes a scalar clock perturbation observable.

| Evidence | Train | Holdout | Verdict |
| --- | ---: | ---: | --- |
| Paired captures / points | 18 / 2,304 | 5 / 640 | disjoint split |
| Mean point-time span | 100.82 ms | 100.81 ms | expected VLP-16 scan duration |
| Mean camera/LiDAR stamp delta | 11.52 ms | 8.48 ms | within 60 ms pairing gate |
| Deskew displacement RMS | 0.91 mm | 1.90 mm | diagnostic only |
| Time sensitivity | 0.01562 m/s | shared train diagnostic | rank 0; below 0.05 m/s gate |
| Signed ±5/10/20 ms controls | — | 0 / 6 detected | below 1 mm margin |

The result is **INCONCLUSIVE**, not PASS: the firing-time data are present and
authenticated, but this short sequence has insufficient motion excitation for
the declared timing gate. `camera_lidar_time_offset_estimated=0` records the
separate correspondence limitation. Bundle verification recomputes the full
7.18 GB bag SHA-256 and reports zero issues.

Online/streaming Horizon-to-Avia calibration replays the same bag without loading
it into memory. Early Horizon messages build the fixed source voxel map; Avia
messages stream in time order as target batches. The bounded replay window and
message budgets are declared in the example config factor options and recorded in
`result.yaml` provenance (`rosbag1_*` keys).

```bash
pip install -e ".[dev,rosbag1-lz4]"
calibrex calibrate examples/public_datasets/tiers_livox_lidars_cali/online_config.yaml \
  --online \
  --batch-size 500 \
  --accumulation-batches 3 \
  --max-accumulated-train-points 8000
```

`outputs/tiers_livox_lidars_cali_online/result.yaml` and `timeline.json` capture
per-batch gate status, batch-only vs accumulated observability rank, holdout RMSE,
and rolling RMSE. Gate thresholds are declared under
`pipeline.factors.lidar_rig_point_to_plane.options` (`online_gate_*` keys) and
recorded in `result.yaml` provenance. The example config uses a 60 s Avia replay
slice (`max_replay_duration_s`) so a multi-gigabyte bag stays stream-bounded.

### Livox point-time and deskew ablation

Livox `CustomMsg` carries a `timebase` for the first point and an integer
nanosecond `offset_time` for each point. The online adapter decodes and records
both. With `use_point_time_offsets: true` it uses those offsets for pose
interpolation when `/vrpn_client_node/UWBTest/pose` is available; with the flag
false it still records the offsets but deliberately uses message-level timing.
Signed `inject_time_offset_s` values are validation-only perturbations.

Materialize and run the comparison set with provenance and SHA-256 digests for
every variant:

```bash
python tools/run_livox_time_ablation.py \
  examples/public_datasets/tiers_livox_lidars_cali/online_config.yaml \
  --output-dir outputs/tiers_livox_time_ablation \
  --include-no-deskew
```

`ablation_manifest.yaml` compares deskew on/off at zero shift and the requested
signed timing controls. The default tool set is 0, ±5, ±10, and ±20 ms; a
bounded replay can request only `--offset-s 0 -0.005` as in the checked
TIERS evidence bundle. A shifted case is a negative control and must not be
reported as an estimated clock offset. Check each result's
`point_time_reference_by_sensor`, clock-domain note, and
`point_time_clock_mapping_by_sensor` and
`rosbag1_source_map_temporal_boundary_applied` before interpreting it. The
tool reuses a completed result when its config SHA still matches, so a long
bag experiment can be resumed without silently changing completed variants.

When `report.html` is generated for an online run, an **Online timeline**
section visualizes the same `timeline.json` data: a per-batch gate verdict strip
(pass / fail / inconclusive), inline SVG charts of holdout and rolling RMSE with
threshold lines, a per-batch observability table (batch-only vs accumulated rank,
condition number, retention), and a session summary (batch counts, final gate
status, final estimate, thresholds). Offline runs omit this section.

On a bounded replay of the same bag (12 source messages / 8k source points, 36
target messages / 54k target points, 60 s budget,
`--batch-size 500 --accumulation-batches 3 --max-accumulated-train-points 8000`),
the Horizon→Avia run produced 108 target batches (500 points each) from 36 Avia
messages. All 108 batches passed the gates with full rank 6, per-batch holdout
RMSE between 0.05 m and 0.23 m (mean ~0.14 m), and final rolling RMSE near
0.14 m.

#### Checked TIERS `LidarsCali.bag` result (2026-08-02)

The full 14-variant check (deskew on/off × 0, ±5, ±10, and ±20 ms validation
shift) completed and is recorded in
[`ablation_manifest.yaml`](C:/Users/rsasa/Workspace/Calibrex/outputs/tiers_livox_time_ablation/ablation_manifest.yaml).
Every variant passed the final online calibration gate and detected all known
bad controls (`known_bad_detectable_fraction: 1.0`). The independent holdout
RMSE range was 0.15778–0.15857 m and the final rolling RMSE range was
0.17377–0.17444 m. At zero shift, deskew on/off was 0.158479/0.173897 m vs
0.158435/0.174256 m (holdout/rolling), so this sequence does not support a
deskew benefit claim: the difference is below 1 mm and is smaller than the
trajectory-quality uncertainty.

The Livox point-time mapping was `stable` for all variants, with a maximum
clock-mapping residual of 1.089 ms and a decoded point-time span of 99.905 ms.
The result quality grade remains `fail` because the recorded trajectory exceeds
the current kinematic gate (3.383 m/s maximum linear speed and
2897.130 deg/s maximum angular speed) and no cross-segment source scans were
available. The solid-state coverage was 10.20 m in distance, 81.7° in azimuth,
and 28.1° in elevation; only one of four distance bins was supported, and the
bag has no integration-window or temperature profile. These are evidence gaps,
not reasons to relax the trajectory gate.

#### AgRob Modular-e ROS 2 smoke check

The `2023-06-14T174658Z` sequence from the public
[AgRob Modular-e dataset](https://zenodo.org/records/8083431) was downloaded
and checksum-verified (`f8e52a9ee09f4013773f56893fc67b8b`). It contains 3855
Livox messages over 386.15 s and 7539 `/odom` messages, but `/livox/lidar` is
`sensor_msgs/msg/PointCloud2` with only `x`, `y`, `z`, `intensity`, `tag`, and
`line` fields. No per-point time field is present. Its odometry also contains
large outliers (maximum linear speed 98.69 m/s), so this sequence is retained
as an adapter smoke fixture only and is not used for the point-time or deskew
calibration evidence.

The larger AgRob sequences expose the ROS 2 Livox topic as
`livox_interfaces/msg/CustomMsg` alongside `/rslidar_points`. Calibrex's
ROS-independent rosbag2 adapter decodes that message directly: it preserves
`timebase`, `offset_time`, reflectivity, tag, and line, and maps the Livox clock
to the rosbag2 record clock for online deskew. A ROS installation or vendor
driver is not required. Configure the Livox sensor with
`point_time_field: offset_time`; leave the RS-LiDAR PointCloud2 sensor's time
field unset unless its fields provide one.

The reproducible paired-sequence entry point is
[`agrob_modular_e/manifest.yaml`](../../examples/public_datasets/agrob_modular_e/manifest.yaml)
and
[`agrob_modular_e/online_config.yaml`](../../examples/public_datasets/agrob_modular_e/online_config.yaml).
The manifest pins the `2023-06-13T170852Z.zip` archive (2,548,542,803 bytes,
MD5 `fa0cea9ea4aafd5e7a0d75676e8e064d`), which contains 2455 Livox
`CustomMsg` messages, 2470 `/rslidar_points` messages, and 4932 `/odom`
messages according to the Zenodo record. The downloader resumes exact byte
ranges and promotes the archive only after the MD5 check succeeds:

```bash
python tools/download_public_dataset.py agrob_modular_e \
  --agrob-workers 256 --agrob-chunk-kib 256
calibrex validate examples/public_datasets/agrob_modular_e/manifest.yaml --kind dataset-manifest
calibrex calibrate examples/public_datasets/agrob_modular_e/online_config.yaml --online
```

The parallel options are useful when the public Zenodo endpoint throttles each
individual Range response. They write a size-shaped, but explicitly
unverified, sidecar plus a resumable state ledger; only the final MD5 check
creates `2023-06-13T170852Z.zip`. A partial file must never be treated as a
valid bag or extracted.

This is the first public paired run in the project that exercises the ROS 2
Livox `CustomMsg` adapter against an independent RS-LiDAR stream. A successful
solver exit is not by itself an accuracy claim: retain the generated result's
holdout, known-bad, trajectory-quality, point-time mapping, and provenance
gates when interpreting it.

For the checked three-window joint replay, the standalone
[`trajectory_window_drift.yaml`](../../outputs/agrob_modular_e_livox_rslidar_online_joint/trajectory_window_drift.yaml)
artifact reports full-span RMSE 0.349 m (FAIL) and local RMSEs 0.361 m (FAIL),
0.288 m (PASS), and 0.330 m (FAIL). Its
`local_inconsistency_suspected` interpretation means the joint failure cannot
be attributed to long-span accumulation alone; it remains a ground-truth-free
diagnostic rather than an absolute extrinsic-accuracy claim.

The deskew-off control is preserved in
[`trajectory_window_drift_deskew_off.yaml`](../../outputs/agrob_modular_e_livox_rslidar_online_joint/trajectory_window_drift_deskew_off.yaml):
full-span RMSE rises to 0.358 m and the three windows become 0.364/0.310/0.352
m (all FAIL). The odometry stream has 4932 raw and retained poses with zero
sub-millisecond burst removals, so `preserve`, `keep_first`, and `keep_last`
are equivalent for this recording. Window 0/2 also have much less rotational
excitation than window 1 (endpoint rotation 6.1°/1.8° versus 170.5°), which
is a stronger next suspect than burst preprocessing alone.

#### AIST GLIM versatile Livox trajectory fixture

The official [GLIM cross-sensor evaluation dataset](https://zenodo.org/records/6864654)
provides a smaller, reproducible Livox Avia fixture. The downloaded
`livox.bag` is 613,350,674 bytes with MD5
`fb17a6d4e0bc1765e276f0dcca8a5063`, and `gt/livox.txt` is the accompanying
trajectory file. `calibrex inspect --type rosbag1` found 618 messages each on
`/livox/lidar` (`livox_ros_driver/CustomMsg`) and `/livox/points`
(`sensor_msgs/PointCloud2`); both expose point time over approximately 99.85 ms
per message. The bag span is 61.693 s, while the 618 GT poses span 61.700 s.

The GT trajectory has a maximum linear speed of 0.749 m/s and maximum angular
speed of 56.6 deg/s under the current kinematic policy, so it is suitable for
ROS 1 CustomMsg/PointCloud2 decoding, point-time mapping, and deskew replay
regression. It is not a LiDAR-to-LiDAR absolute-extrinsic reference: this small
fixture supplies one Livox sensor and a trajectory, not a second calibrated
LiDAR. The raw bag remains an external input under `data/public/glim_versatile/`.
The checked stream contract is
[`glim_versatile/manifest.yaml`](../../examples/public_datasets/glim_versatile/manifest.yaml)
and validates with `calibrex validate --kind dataset-manifest`.

The ROS 1 PointCloud2 header carries the relative Livox clock (about 110 s),
while bag records use epoch time. The reproducible conversion therefore
restamps `/livox/points` with the measured median record/header offset and
maps the external GT with the same explicit offset. This is a clock-domain
normalization, not a recovered time-offset estimate:

```bash
python tools/rosbag1_to_rosbag2_online_pair.py \
  --src data/public/glim_versatile/livox.bag \
  --dst data/public/glim_versatile/livox_rosbag2_gt_restamped \
  --topic /livox/points \
  --odom-source external-trajectory \
  --trajectory-file data/public/glim_versatile/gt/livox.txt \
  --trajectory-time-offset-s 1655977530.260883 \
  --restamp-topic /livox/points \
  --duplicate-topic /livox/points:/livox/points_copy
```

The duplicate topic is an identity self-consistency control. The checked
zero-shift ablation is recorded in
`outputs/glim_versatile_livox_time_ablation/ablation_manifest.yaml`:

| Mode | Holdout RMSE (m) | Final rolling RMSE (m) | Final gate | Trajectory gate |
|------|------------------:|------------------------:|------------|-----------------|
| deskew on | 0.164552 | 0.161424 | pass | pass |
| deskew off | 0.153536 | 0.149489 | pass | pass |

Both variants detect the declared known-bad controls (1.0), but the off-mode
is slightly better on this identity control. This does not establish a deskew
benefit or absolute extrinsic accuracy; it establishes that the external-GT
adapter, clock normalization, PointCloud2 `t` decoding, and trajectory gates
are executable on a public solid-state fixture.

### Spinning × solid-state (Velodyne VLP-16 → Livox Horizon)

The same bag also carries spinning-LiDAR `sensor_msgs/PointCloud2` topics
(`/velodyne_points`, `/os_cloud_node/points`, `/os_cloud_nodee/points`) beside
the Livox `CustomMsg` streams. `calibrex inspect --type rosbag1` reports message
counts and sampled decoded point counts for every LiDAR topic without loading the
full bag.

Online mixed-pair calibration uses the Velodyne as the fixed 360° source map and
streams Livox Horizon target batches—the same source/target split recommended for
heterogeneous pairs where the denser scanner should anchor the voxel map:

```bash
calibrex calibrate examples/public_datasets/tiers_livox_lidars_cali/online_mixed_config.yaml \
  --online \
  --batch-size 500 \
  --accumulation-batches 3 \
  --max-accumulated-train-points 8000
```

The mixed config seeds `frames.livox_horizon.transform.initial` from the TIERS
dataset README GICP extrinsics (`velo_sensor` and `hori_frame` relative to
`base_link` in section 5.4); it is not hand-tuned to the online estimate. The
rig baseline is small (~0.30 m translation magnitude in that nominal seed).

On the same bounded replay budgets as the Horizon→Avia run above, a Velodyne→Horizon
replay produced 108 Horizon batches (500 points each) over ~3.6 s of bag time
(36 target messages consumed). All 108 batches passed (`online_gate_min_rank: 6`,
`online_gate_max_holdout_rmse_m: 0.40`, `online_gate_max_rolling_regression_m:
0.15`). Batch-only and accumulated observability stayed at rank 6 with condition
number ~8–14 and no weak DoF warnings. Per-batch holdout RMSE ranged 0.09–0.23 m
(mean ~0.16 m); rolling RMSE settled near 0.16 m. Gate thresholds match the
Horizon→Avia example because measured residuals sit in the same band—there was no
reason to relax them for the heterogeneous pair.

Compared with solid-state→solid-state (Horizon→Avia), the spinning×solid-state pair
shows slightly higher and more variable batch holdout RMSE (ring-sampled Velodyne
planes vs non-repetitive Horizon stripes) but still passes the same gates. FOV
overlap is sufficient in the static room sequence: the Velodyne 360° map covers
structure that reappears in forward-facing Horizon scans. Final estimate
`T_velodyne_vlp16_livox_horizon` (translation ~[0.12, 0.11, −0.16] m) sits about
0.16 m from the TIERS nominal GICP seed — a shift on the order of the measured
holdout residuals, so the two references agree only to that residual level rather
than confirming each other tightly. It is recorded in
`outputs/tiers_livox_lidars_cali_online_mixed/result.yaml` with full rosbag replay
provenance.

The mixed offline config follows the same bounded native adapter protocol, with
Velodyne as the source map and Horizon as the target. Its PointCloud2 `time`
field is recorded as a spinning-LiDAR control; it is not substituted for
Livox `offset_time`. KITTI/A2D2 and Indoor02 remain comparison controls: they
test spinning-LiDAR geometry, public-data reproducibility, and temporal
holdout machinery, but cannot establish solid-state temperature or internal
beam-calibration accuracy.

`calibrex inspect --type rosbag1` lists every supported PointCloud2 or Livox
`CustomMsg` topic with message counts, and samples a few messages per topic to
report decoded point counts, intensity presence, spatial bounds, range,
azimuth/elevation FOV, point-time availability/range, and first/last ROS
timestamps (normalized to integer nanoseconds). Other message types are
ignored by design; the reader remains ROS-independent and point-cloud scoped.

## ROS 2 bag (rosbag2) reader

Calibrex reads rosbag2 recordings directly with a pure-Python parser
(`slac.data.rosbag2`); no ROS installation is required. Supported storage
backends are **sqlite3** (``.db3``, the default through Humble) and **mcap**
(``.mcap``, the default from Iron). A bag directory with `metadata.yaml` or a
bare ``.db3`` / ``.mcap`` file path both work.

Decoding ``sensor_msgs/msg/PointCloud2`` and ``nav_msgs/msg/Odometry`` payloads
requires numpy (`pip install "Calibrex[rosbag2]"`). MCAP bags with chunked
``lz4`` or ``zstd`` compression additionally need
`pip install "Calibrex[rosbag2-compression]"` (`none` / uncompressed chunks
work out of the box).

```bash
calibrex inspect path/to/bag --type rosbag2 --json
```

`calibrex inspect --type rosbag2` reports per-topic message counts, decoded
`frame_id`, sampled point counts/bounds, Livox `offset_time` availability and
timebase range, plus an all-message Odometry motion audit (time span, path
length, linear/angular speed p95/max, duplicate timestamps, and the default replay
kinematic gate). This separates a valid message decode from a trustworthy
motion-compensation premise.

For ROS 2 online calibration, the replay applies a deterministic temporal
boundary: source-map messages at or after the final 20% target-capture boundary
are excluded from the source map, while the target stream retains per-batch
point holdouts. The result records the boundary timestamp, train/holdout window
counts, and whether the clock-domain comparison was actually applied. Livox
`CustomMsg` is also included in the independent first-half/second-half
trajectory drift evidence; no ROS vendor driver is needed.

For a joint replay over several time ranges, replace the legacy single-window
options with a factor option such as:

```yaml
capture_windows:
  - start_timestamp_ns: 1686676331234969730
    end_timestamp_ns: 1686676371234969729
  - start_timestamp_ns: 1686676371234969730
    end_timestamp_ns: 1686676411234969729
capture_window_sampling_policy: evenly_spaced
```

`capture_windows` is an inclusive union. With `evenly_spaced`, source and
target records are selected per window using decoded LiDAR capture timestamps;
the legacy `max_source_messages`, `max_target_messages`, and source-point cap
are applied per window. The result provenance records each boundary and the
adopted source/target message and point counts. The default `first` policy
preserves the historical single-window behavior. Set `max_replay_duration_s`
to `null` when windows span more than that duration.

### Moving platform (TIERS Indoor02, rosbag2 + odometry)

The TIERS `Indoor02` sequence records a Velodyne VLP-16 (`/velodyne_points`),
an Ouster OS1 (`/os_cloud_nodee/points`, note the upstream typo), and a VRPN
MOCAP `geometry_msgs/PoseStamped` stream (`/vrpn_client_node/UWBTest/pose`) on
a moving office platform (~42 s). The ROS 1 bag is multiple gigabytes; fetch it
manually from the upstream dataset:

1. Open the TIERS dataset repository:
   <https://github.com/TIERS/tiers-lidars-dataset> and follow its download table
   to the `Indoor02` sequence (direct link recorded in
   `examples/public_datasets/tiers_lidars_dataset_indoor02/manifest.yaml` under
   `provenance.download_url`).
2. Save the bag as `data/public/tiers_lidars_dataset/indoor02.bag`.

The repository helper performs the public SharePoint cookie handshake and
resumable Range download automatically:

```bash
python tools/download_public_dataset.py tiers_lidars_dataset_indoor02 \
  --output-dir data/public
```

Reproduction audit (2026-08-02): a cookie-preserving public SharePoint session
retrieved the raw bag to an external data volume (the 17,974,826,130-byte file
is intentionally not committed). Its first-MiB SHA-256 matches the manifest:
`164c2f8dcfd42463ad4981ba25a97668f015ced776db2781fc1e4211fd8fd574`.
`calibrex inspect --type rosbag1` then reported 420 Velodyne messages, 423
Ouster messages, and Livox/other LiDAR streams. The pure-Python ROS 1 reader
also had to be made tolerant of the official bag's padding between top-level
records; the regression test and the raw-bag replay now cover that path.
The generated Indoor02 results below are therefore a fresh, schema-validated
run from the public archive. The archive itself remains an external input.

Convert the ROS 1 bag to rosbag2 with synthesized odometry (the pose trajectory
is real MOCAP; only the `nav_msgs/msg/Odometry` envelope is authored). The
upstream ROS 1 bag mixes clock domains: `/velodyne_points` and the VRPN pose use
the recording epoch, while `/os_cloud_nodee/points` stamps with the Ouster
since-boot clock (~753 s). Add `--restamp-topic /os_cloud_nodee/points` so the
converter shifts each Ouster header stamp by the per-topic median
`bag_receive_time − header_stamp` (measured ≈ 1.645×10⁹ s on Indoor02), moving
it into the recording clock domain while preserving the sensor's own relative
timing. The constant offset includes mean transport/assembly latency, so
restamped absolute stamps carry a bias of that order; residual error is roughly
latency × platform speed.

```bash
uv run tools/rosbag1_to_rosbag2_online_pair.py \
  --src data/public/tiers_lidars_dataset/indoor02.bag \
  --dst data/public/tiers_lidars_dataset/indoor02_rosbag2 \
  --topic /velodyne_points \
  --topic /os_cloud_nodee/points \
  --restamp-topic /os_cloud_nodee/points \
  --pose-topic /vrpn_client_node/UWBTest/pose \
  --odom-source pose-topic \
  --odom-topic /odom \
  --child-frame-id base_link \
  --storage sqlite3 \
  --compress none
```

Rig-frame odometry from KISS-ICP on the source Velodyne (no external body
alignment; poses are `T_world_velo_sensor` with `T_base_source = identity`):

```bash
uv run tools/rosbag1_to_rosbag2_online_pair.py \
  --src data/public/tiers_lidars_dataset/indoor02.bag \
  --dst data/public/tiers_lidars_dataset/indoor02_rosbag2_kissicp \
  --topic /velodyne_points \
  --topic /os_cloud_nodee/points \
  --restamp-topic /os_cloud_nodee/points \
  --odom-source kiss-icp \
  --kiss-icp-topic /velodyne_points \
  --kiss-icp-max-range 30.0 \
  --child-frame-id base_link \
  --storage sqlite3 \
  --compress none
```

An MCAP variant with per-message zstd compression exercises the message-mode
decompression path on real independently-encoded CDR:

```bash
uv run tools/rosbag1_to_rosbag2_online_pair.py \
  --src data/public/tiers_lidars_dataset/indoor02.bag \
  --dst data/public/tiers_lidars_dataset/indoor02_rosbag2_mcap \
  --topic /velodyne_points \
  --topic /os_cloud_nodee/points \
  --restamp-topic /os_cloud_nodee/points \
  --pose-topic /vrpn_client_node/UWBTest/pose \
  --odom-source pose-topic \
  --odom-topic /odom \
  --child-frame-id base_link \
  --storage mcap \
  --compress zstd
```

Inspect the converted bags (topics, counts, odometry pose sanity):

```bash
calibrex inspect data/public/tiers_lidars_dataset/indoor02_rosbag2 --type rosbag2 --json
calibrex inspect data/public/tiers_lidars_dataset/indoor02_rosbag2_kissicp --type rosbag2 --json
calibrex inspect data/public/tiers_lidars_dataset/indoor02_rosbag2_mcap --type rosbag2 --json
```

Online motion-compensated calibration with MOCAP odometry (A), the static-rig
control (B), and the optional KISS-ICP rig-frame odometry (C):

```bash
pip install -e ".[dev,rosbag1-lz4,rosbag2,rosbag2-compression]"
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_motion_config.yaml --online
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_static_config.yaml --online
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_kissicp_config.yaml --online
```

On the fresh bounded replay (12 Velodyne source messages / 8k source points,
36 Ouster target messages / 1.5k target points, 60 s budget), the MOCAP run (A)
and static control (B) each produced 108 target batches. Before restamping,
Ouster header stamps were in a different clock domain from `/odom`; the
converter's measured per-topic offset was `1645462364594977536` ns. After
restamping, A had zero odometry clamping and zero extrapolation.

The recomputed evidence is:

| Run | Odometry | Batches adopted | Holdout RMSE (m) | Final rolling RMSE (m) | Trajectory/evidence verdict |
|-----|----------|-----------------|------------------|-------------------------|-----------------------------|
| A motion | MOCAP `/odom` | 108 / 108 | 0.319675 | 0.281411 | kinematic gate **fail**; max 240.387 m/s and 115546.218 deg/s |
| B static | none | 107 / 108 | 0.302346 | 0.286878 | trajectory skipped; static evidence **warn** |

Both runs detected all six declared known-bad extrinsic perturbations (6/6),
and both final online gates passed. The result, timeline, trajectory (A),
assessment, and evidence bundle validate against their schemas; `calibrex
verify` reports 10/10 artifacts and zero failed claims for each run. The
artifacts are under `outputs/tiers_lidars_dataset_indoor02_online_motion/` and
`outputs/tiers_lidars_dataset_indoor02_online_static/`.

The trajectory failure in A is intentional evidence, not a relaxed threshold:
the VRPN pose stream contains kinematic outliers under the current 3 m/s and
120 deg/s policy. A's LiDAR holdout and interpolation numbers are retained,
but A is **not** an absolute-accuracy certification. B is a useful static
control, not ground truth. The TIERS README GICP seed (inter-sensor distance
about 0.37 m) remains a nominal reference rather than an independently verified
truth; the current estimates differ from that seed by about 2.853 m / 45.4° (A)
and 0.862 m / 17.6° (B). These gaps prevent claiming absolute extrinsic accuracy.

Run C (KISS-ICP rig-frame odometry) remains a reproducible optional experiment
via the command above, but its numeric result is not included in this audit
until that conversion and bundle are rerun from the same raw archive.

#### Indoor02 pose-burst and frame-semantics audit

The official TIERS OptiTrack CSV is available at
<https://github.com/TIERS/tiers-lidars-dataset/blob/main/data/ground_truth/indoor02_optitrack.csv>.
An audit of that file and the converted `/odom` stream found 854 consecutive
pose intervals shorter than 1 ms. The burst is present in the official source
trajectory itself; it is not introduced by the rosbag2 conversion. The raw
track remains the input of record.

Two schema-validated controls make the handling explicit:

```bash
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_motion_pose_burst_config.yaml --online
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_motion_baseframe_config.yaml --online
```

Both set `dataset.odometry_preprocessing.burst_policy: keep_last` and
`min_interval_s: 0.001`. This retains the latest pose in each sub-millisecond
burst, without modifying the bag. The result provenance records 5,065 raw
poses, 4,211 retained poses, and 854 removed poses.

| Run | Frame graph | Batches adopted | Final rolling RMSE (m) | Linear max (m/s) | Angular max (deg/s) | Trajectory verdict |
|-----|-------------|-----------------|------------------------|------------------|---------------------|--------------------|
| A raw baseline | Velodyne root | 108 / 108 | 0.281411 | 240.387 | 115546.218 | **FAIL** |
| A1 pose-burst control | Velodyne root | 108 / 108 | 0.309474 | 1.745 | 600.717 | **FAIL** |
| A2 pose-burst + explicit base frame | `base_link → velodyne → ouster` | 106 / 108 | 0.285919 | 1.745 | 600.717 | **FAIL** |

The burst policy removes the artificial linear-speed failure, but the official
orientation stream still exceeds the declared 120 deg/s angular gate. No
orientation smoothing or threshold relaxation is applied, so A1/A2 are
improved reproducible analyses rather than absolute-accuracy certification.
The A2 output keeps the documented fixed `T_base_link_velodyne_vlp16` edge and
estimates `T_velodyne_vlp16_ouster_os1`; the online pair selector is covered by
a regression test for chained non-root LiDAR frames.

The corrected A2 bundle is under
`outputs/tiers_lidars_dataset_indoor02_online_motion_baseframe_pose_burst/`.
It has a passing final online gate, a passing cross-segment RMSE of 0.2973 m,
and a failing overall quality grade solely because of the angular kinematic
gate. Verify it with:

```bash
calibrex verify outputs/tiers_lidars_dataset_indoor02_online_motion_baseframe_pose_burst/bundle.json
calibrex validate outputs/tiers_lidars_dataset_indoor02_online_motion_baseframe_pose_burst/trajectory.json --kind trajectory
```

This validation established that the public archive exercised three real-input
reader paths that synthetic mirror-image tests did not: ROS 1 top-level record
padding between chunks, CDR encapsulation endianness keyed off the wrong header
byte, and message-mode zstd compression declared in `metadata.yaml` being
ignored. With those fixes, `calibrex inspect` decodes the raw bag and the
regenerated sqlite3/MCAP bags with standard CDR headers and per-message
decompression.

| Run | Odometry | Batches adopted | Holdout RMSE (m) | Final Δ vs TIERS seed |
|-----|----------|-----------------|------------------|------------------------|
| A motion | MOCAP `/odom` | 108 / 108 | 0.319675 | 2.853 m, 45.4°; kinematic gate fail |
| B static | none | 107 / 108 | 0.302346 | 0.862 m, 17.6°; warn |
| C kiss-icp | KISS-ICP `/odom` | not rerun in this audit | — | optional, no current claim |

The identity/KISS-ICP and deskew subsections below are historical optional
evidence records from earlier Indoor02 conversions. They are retained as
reproduction targets, but their numeric claims are not part of the fresh
2026-08-02 A/B audit above because those KISS-ICP bundles were not regenerated
in this run. Re-run the commands and verify the resulting bundles before using
any of those numbers as current release evidence.

### Identity self-consistency control

Duplicate the source Velodyne on the kiss-icp rosbag2 variant so ground truth
is exactly identity: calibrate `/velodyne_points` (root) vs
`/velodyne_points_copy` with an identity initial transform. Same factor
options and gates as the A/B/C runs above (voxel 0.5 m, correspondence gate
1.5 m, 12 source messages / 8k source points, 36 target messages / 1.5k target
points, holdout 0.2, prior_sigma 0.3 m / 15°).

```bash
uv run tools/rosbag1_to_rosbag2_online_pair.py \
  --src data/public/tiers_lidars_dataset/indoor02.bag \
  --dst data/public/tiers_lidars_dataset/indoor02_rosbag2_selftest \
  --topic /velodyne_points \
  --topic /os_cloud_nodee/points \
  --restamp-topic /os_cloud_nodee/points \
  --odom-source kiss-icp \
  --kiss-icp-topic /velodyne_points \
  --kiss-icp-max-range 30.0 \
  --child-frame-id base_link \
  --storage sqlite3 \
  --compress none \
  --duplicate-topic /velodyne_points:/velodyne_points_copy

calibrex inspect data/public/tiers_lidars_dataset/indoor02_rosbag2_selftest --type rosbag2 --json

calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_selftest_motion_config.yaml --online
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_selftest_static_config.yaml --online
```

Two additional odometry modes improve the synthesized `/odom` track at bag
conversion time (before calibration). Both use the Velodyne per-point `time`
field (float32 seconds, ~99% negative offsets, span ≈ 0.10 s on Indoor02).

**Native deskew** (`--kiss-icp-native-deskew` with `--odom-source kiss-icp`):
single-pass KISS-ICP with `cfg.data.deskew = True` and per-point timestamps
normalized to [0, 1] within each scan before `register_frame`.

```bash
uv run tools/rosbag1_to_rosbag2_online_pair.py \
  --src data/public/tiers_lidars_dataset/indoor02.bag \
  --dst data/public/tiers_lidars_dataset/indoor02_rosbag2_selftest_kissdeskew \
  --topic /velodyne_points \
  --topic /os_cloud_nodee/points \
  --restamp-topic /os_cloud_nodee/points \
  --odom-source kiss-icp \
  --kiss-icp-topic /velodyne_points \
  --kiss-icp-max-range 30.0 \
  --kiss-icp-native-deskew \
  --kiss-icp-time-field time \
  --child-frame-id base_link \
  --storage sqlite3 \
  --compress none \
  --duplicate-topic /velodyne_points:/velodyne_points_copy
```

**Two-pass** (`--odom-source kiss-icp-two-pass`): pass 1 runs KISS-ICP on raw
scans; each scan is rigidified to its message stamp with the pass-1 track
(`p_rigid = T(t_msg)⁻¹ T(t_msg + dt_i) p_i`, pose interpolation with clamping
at track ends); pass 2 runs a fresh KISS-ICP instance on the deskewed scans.

```bash
uv run tools/rosbag1_to_rosbag2_online_pair.py \
  --src data/public/tiers_lidars_dataset/indoor02.bag \
  --dst data/public/tiers_lidars_dataset/indoor02_rosbag2_selftest_twopass \
  --topic /velodyne_points \
  --topic /os_cloud_nodee/points \
  --restamp-topic /os_cloud_nodee/points \
  --odom-source kiss-icp-two-pass \
  --kiss-icp-topic /velodyne_points \
  --kiss-icp-max-range 30.0 \
  --kiss-icp-time-field time \
  --child-frame-id base_link \
  --storage sqlite3 \
  --compress none \
  --duplicate-topic /velodyne_points:/velodyne_points_copy
```

Example configs for the new bags:

```bash
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_selftest_kissdeskew_config.yaml --online
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_selftest_twopass_config.yaml --online
```

On the bounded replay budget above, motion-compensated selftest (KISS-ICP
`/odom`) adopted all 104 batches with zero odometry clamping and zero
extrapolation and recovered identity to ~9.6 cm / ~1.2°. The static control
(no odometry) also adopted all 104 batches and passed every internal gate —
yet its final estimate landed ~98 cm / ~37° from the known identity truth.
Motion compensation cut the extrinsic error roughly 10× versus the static
run; the residual ~10 cm / ~1.2° was attributed to no per-point deskew and
LiDAR-odometry drift on a moving platform.

Per-point deskew is now available for rosbag2 online runs via
`sensors.<name>.point_time_field` (for example `time` on Velodyne VLP-16,
`t` on Ouster OS1). When `dataset.odometry_topic` is set, capture time is
`header_stamp + offset` with offsets decoded as float seconds or integer
nanoseconds from the named PointCloud2 field. Replay provenance records
`deskew_applied`, `deskew_time_field`, and `deskew_span_s` (max − min offset
seen). Without odometry, `point_time_field` is ignored
(`deskew_ignored_no_odometry: true`).

On the Velodyne selftest bag, per-point `time` offsets are ~99% negative
(span ≈ 0.10 s, consistent with a VLP-16 sweep where the header stamp sits
near the end of rotation). A deskew replay with `point_time_field: time` on
both Velodyne sensors adopted 104 / 104 batches and landed ~10.5 cm / ~1.5°
from identity — not a meaningful improvement over the ~9.5 cm / ~1.2°
per-message motion-compensated baseline on this control. The modest regression
is consistent with KISS-ICP odometry drift dominating the residual once
intra-scan timing is corrected.

| Run | Odometry | Deskew | Batches adopted | Final Δ vs identity (truth) |
|-----|----------|--------|-----------------|-------------------------------|
| Selftest motion | KISS-ICP `/odom` | no | 104 / 104 | ~9.5 cm, ~1.2° |
| Selftest deskew | KISS-ICP `/odom` | velodyne `time` | 104 / 104 | ~10.5 cm, ~1.5° |
| Selftest static | none | — | 104 / 104 | ~98 cm, ~37° |
| Ground truth | — | — | — | 0 cm, 0° |

**Odometry deskew variants (bag conversion, identity selftest):** on the same
bounded replay budget, all three motion-compensated variants adopted 104 / 104
batches. Native-deskew KISS-ICP odometry tightened translation recovery
meaningfully versus the v0.2 baseline (~4.3 cm vs ~9.5 cm) at the cost of a
modest rotation increase (~2.0° vs ~1.2°). Two-pass odometry also beat the
baseline on translation (~6.2 cm) but did not improve rotation and degraded
cross-segment trajectory consistency (RMSE 0.366 m, trajectory verdict FAIL).
**Native deskew wins on identity translation recovery** (~5 cm below baseline);
two-pass is intermediate on translation but worse on trajectory evidence.

| Variant | Odometry mode | Batches adopted | Final Δ vs identity | Holdout RMSE (m) | Cross-segment RMSE (m) | Trajectory verdict |
|---------|---------------|-----------------|---------------------|------------------|------------------------|-------------------|
| Baseline | kiss-icp raw | 104 / 104 | 9.54 cm, 1.17° | 0.059 – 0.184 | 0.143 (PASS) | PASS |
| Native deskew | kiss-icp + `--kiss-icp-native-deskew` | 104 / 104 | 4.34 cm, 1.96° | 0.062 – 0.165 | 0.160 (PASS) | PASS |
| Two-pass | kiss-icp-two-pass | 104 / 104 | 6.16 cm, 2.01° | 0.060 – 0.176 | 0.366 (FAIL) | FAIL |

KISS-ICP rig-frame Velodyne→Ouster run C with deskew (`time` + `t`) adopted
106 / 108 batches and landed ~76 cm / ~28° from the TIERS GICP seed versus
~85 cm / ~28° without deskew — a modest translation improvement with rotation
unchanged, still well above the holdout residual band and not confirmation of
the nominal seed.

| Run | Odometry | Deskew | Batches adopted | Final Δ vs TIERS seed |
|-----|----------|--------|-----------------|------------------------|
| C kiss-icp | KISS-ICP `/odom` | no | 106 / 108 | ~85 cm, ~28° |
| C kiss-icp deskew | KISS-ICP `/odom` | `time` + `t` | 106 / 108 | ~76 cm, ~28° |

Example configs:

```bash
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_selftest_deskew_config.yaml --online
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_kissicp_deskew_config.yaml --online
```

This is the first genuine absolute-accuracy validation on real moving-platform
data with exact ground truth: motion compensation reduces extrinsic error from
~0.98 m / ~37° to ~9.6 cm / ~1.2°. The honest negative finding is that the
static run passed all internal gates (holdout RMSE 0.40 m, rolling regression,
rank 6) while being ~1 m / ~37° wrong — on this scene the online gates measure
internal consistency of a self-consistently wrong map, not absolute accuracy.
Per-point deskew is implemented but did not materially tighten the identity
selftest residual on this sequence when applied only during online replay;
remaining error was attributed primarily to LiDAR-odometry drift once intra-scan
timing is handled. **Native-deskew odometry at bag conversion** does materially
reduce translation drift on the same control (~4.3 cm vs ~9.5 cm baseline).

Reconciling the earlier A/B/C discussion: the Velodyne→Ouster ~85 cm residual
in run C is now attributable primarily to cross-sensor effects (Ouster stamp
semantics and restamp bias, LiDAR-odometry drift, nominal GICP seed), not a
failure of the motion-compensation pipeline itself. Deskew narrows the C
translation gap modestly (~9 cm) but does not establish absolute accuracy
against the TIERS reference.

### Temporal evidence (time-offset probes and estimator)

After a motion-compensated rosbag2 online run completes, optional post-run
temporal evidence re-evaluates holdout point-to-plane RMSE while shifting target
capture timestamps. This is evaluation-only — it never re-solves the extrinsic.

**Sign convention:** positive `delta_t_s` evaluates
`T_world_source(t_capture + delta_t_s)`; positive offset means target capture
timestamps are treated as lagging the odometry clock.

Factor options under `lidar_rig_point_to_plane`:

| Option | Default | Role |
|--------|---------|------|
| `time_offset_probe_s` | `[]` | Declared probe offsets (empty = probes off) |
| `time_offset_probe_min_rmse_increase` | `0.10` | Relative RMSE increase for a probe to count as detected |
| `online_gate_min_time_offset_probe_detection_ratio` | `1.0` | Fraction of declared probes that must be detected |
| `estimate_time_offset` | `false` | Run 1D holdout-RMSE search (±`time_offset_search_bound_s`, ~1 ms resolution) |
| `time_offset_search_bound_s` | `0.20` | Search half-width in seconds |
| `online_gate_max_abs_time_offset_s` | `0.05` | Pass when \|δt̂\| ≤ gate; fail when above with significant RMSE improvement |
| `time_offset_anchor` | `final` | `final` = evaluate at the adopted online estimate (v0.2); `initial` = hold extrinsic at the configured initial transform (Pillar 2 anchored mode) |
| `inject_time_offset_s` | `0.0` | **Validation-only:** shift target capture timestamps at load time (requires `dataset.odometry_topic`; recorded loudly in provenance) |

When temporal options are set, evidence is computed in **both** modes every run:
`adapted` (final adopted extrinsic) and `anchored` (configured initial). Top-level
metrics and gate verdicts follow `time_offset_anchor` (`final` by default).
`run.provenance.temporal_evidence` always records `adapted` and `anchored`
sub-blocks, `time_offset_anchor`, `anchored_transform`, optional
`time_offset_injected_s`, and a `separability` joint-observability verdict.

**Separability verdict** (`online_temporal_separability`, pass for
`separable`/`consistent`, warn for `degenerate`):

| Verdict | Condition |
|---------|-----------|
| `separable` | Anchored curve localizes δt while adapted curve is flat or has absorbed the offset (anchored \|δt̂\| ≫ adapted \|δt̂\| ≈ 0) — v0.2 degeneracy signature reversed |
| `degenerate` | Both holdout-RMSE curves are flat (motion cannot excite δt) |
| `consistent` | Both curves localize a minimum |

Reports also emit a `temporal` evidence-summary row:
**Extrinsic/Temporal Separability** (metric ids include
`online_temporal_separability`).

When temporal options are set but `dataset.odometry_topic` is unset, provenance
records `temporal_evidence_skipped_no_odometry: true` and no temporal verdict is
emitted. Non-zero `inject_time_offset_s` without odometry raises `ConfigError`.

Example configs and commands:

```bash
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_kissicp_temporal_anchored_config.yaml --online
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_selftest_motion_temporal_anchored_config.yaml --online
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_selftest_motion_temporal_anchored_injected_config.yaml --online
```

On the bounded Indoor02 replay budget (12 source messages / 8k source points, 36
target messages / 1.5k target points, 60 s, probes `[0.05, -0.05, 0.10]`,
`estimate_time_offset: true`, gate 0.05 s, **`time_offset_anchor: initial`**):

**Pillar 2 acceptance read (honest):** Anchored mode restores more probe power
than v0.2 adapted-only runs on the injected selftest (+0.10 s probe detects;
δt̂ moves toward the injected bias), but ±0.05 s probes still miss on real
Indoor02 and the anchored estimator does not yet meet the ±10 ms recovery target
on injection. Cross-sensor KISS-ICP anchored curves remain below the 10 %
flatness margin (separability **degenerate**); δt̂ ≈ −140 ms is in the Ouster
restamp bias order of magnitude but estimate gate stays **INCONCLUSIVE**. The
machinery, dual-mode provenance, and separability row are in place; full Pillar 2
acceptance on real Indoor02 is **not yet met**.

| Run | Anchor δt̂ | Anchor flatness | Anchor probes | Adapted δt̂ | Adapted flatness | Adapted probes | Separability |
|-----|-----------|-----------------|---------------|------------|------------------|----------------|--------------|
| Selftest clean (anchored) | −25 ms | 0.31 | 0 / 3 | −71 ms | 0.34 | 1 / 3 | consistent |
| Selftest injected +50 ms | −75 ms | 0.44 | 1 / 3 | −44 ms | 0.32 | 1 / 3 | consistent |
| KISS-ICP cross-sensor (anchored) | −140 ms | 0.032 | 0 / 3 | −18 ms | 0.012 | 0 / 3 | degenerate |

The differential response is the strongest signal in this table: injecting
+50 ms moved the anchored estimate from −25 ms to −75 ms — a shift of exactly
−50 ms. The anchored estimator therefore tracks injected offsets precisely on
real data; it is the absolute baseline (−25 ms on a shared-clock selftest)
that stays offset. Interpretation (not a measurement): VLP-16 per-point
`time` offsets on this bag span ≈ [−0.10, 0] s with the header stamp at sweep
end, so the effective mean capture time sits ≈ 50 ms before the stamp; a
residual-minimizing δt̂ between 0 and −50 ms is consistent with that scan-time
semantics rather than with estimator error. By contrast the adapted-mode
estimate did not track the injection (−71 ms → −44 ms, a +27 ms shift against
a −50 ms truth change) — the adapted extrinsic partially absorbs the offset,
which is exactly the degeneracy the anchor mode was built to expose.

v0.2 adapted-only reference (same replay budget, `time_offset_anchor: final`):

| Run | δt̂ (adapted) | Curve flatness | Probe detection | Temporal verdict |
|-----|---------------|----------------|-----------------|------------------|
| KISS-ICP temporal | −18 ms | 0.012 | 0 / 3 | FAIL (probe) |
| Selftest temporal | −71 ms | 0.34 | 1 / 3 | FAIL (probe) |

Synthetic moving-rig fixtures (unit tests) recover injected biases to ±5 ms in
anchored mode with FREE solver while adapted δt̂ ≈ 0 (separability
**separable**); zero-offset and static-odometry controls assert
**consistent** and **degenerate** respectively.

### Trajectory evidence (motion-compensated online runs)

When `dataset.odometry_topic` is set, motion-compensated online runs emit a
schema-validated `trajectory.json` artifact (`slac.trajectory/v0.1`) beside
`timeline.json`. The artifact records subsampled `T_world_base` poses (default
budget 500 samples, evenly spaced in time), odometry source metadata, frame
semantics, interpolation health (`interpolation_count`, `clamp_count`,
`max_extrapolation_s`), and a `quality` block with graded metrics and recorded
gate thresholds. Provenance registers `trajectory_path`; runs without odometry
set `trajectory_evidence_skipped_no_odometry: true`.

Three ground-truth-free metric families feed `trajectory_gate_verdict` (worst of
the three: pass &lt; inconclusive &lt; fail):

| Option | Default | Gate |
|--------|---------|------|
| `trajectory_gate_max_speed_mps` | `3.0` | Kinematic: max single-step linear speed (m/s) from consecutive odometry poses |
| `trajectory_gate_max_angular_speed_dps` | `120.0` | Kinematic: max single-step angular speed (deg/s) |
| `trajectory_gate_max_clamp_fraction` | `0.05` | Interpolation: `clamp_count / interpolation_count` |
| `trajectory_gate_max_cross_segment_rmse_m` | `0.30` | Cross-segment drift proxy: evaluation-only stream of the source PointCloud2 topic over the full odometry track span (independent of calibration replay budgets). The track midpoint splits two ~21 s windows; up to 40 scans per half are chosen with `evenly_spaced_time` sampling, up to 4000 world-frame points per half after per-scan subsampling. First-half points build a voxel plane map; second-half points are scored point-to-plane with identity extrinsic. Provenance and `trajectory.json` record per-half scan/point counts, time windows, correspondence count, RMSE, and `cross_segment_sampling_policy`. |

Fewer than 10 odometry samples → kinematic **INCONCLUSIVE**; fewer than 50
cross-segment correspondences → cross-segment **INCONCLUSIVE**.

Before starting a solid-state LiDAR calibration, run the capture-readiness
wizard. It checks window-local motion excitation, voxel-plane normal diversity,
and a six-DoF point-to-plane observability proxy, then emits a concise
`PROCEED`, `REVIEW`, or `RECAPTURE` recommendation:

```bash
calibrex calibration-readiness \
  examples/public_datasets/agrob_modular_e/online_joint_config.yaml \
  --output outputs/agrob_modular_e_livox_rslidar_online_joint/capture_readiness.yaml
calibrex validate \
  outputs/agrob_modular_e_livox_rslidar_online_joint/capture_readiness.yaml
```

For the AgRob Modular-e three-window replay, the readiness artifact reports
window 0: **RECAPTURE** (6.1° endpoint rotation), window 1: **PROCEED**
(170.5°), and window 2: **RECAPTURE** (1.8°). All three windows have full
local geometry evidence in this recording (normal rank 3, estimated observable
DoF 6, and 3567–3713 proxy correspondences), so the actionable problem is
motion excitation rather than missing planes. The artifact is
`slac.capture_readiness/v0.1` and includes the config/bag digests, thresholds,
deskew policy, and recapture instructions.

When the readiness result is usable, the continuous-time LiDAR-pair baseline
can profile a scalar clock offset while re-optimizing the extrinsic at every
candidate offset:

```bash
calibrex continuous-time-lidar-pair \
  examples/public_datasets/agrob_modular_e/online_joint_config.yaml \
  --output outputs/agrob_modular_e_livox_rslidar_continuous_time_ablation_hybrid_support6_iter10/adaptive_mad/continuous_time_lidar_pair.yaml
```

This artifact uses the latest target capture frame group as holdout and records
the sign convention, offset profile, train/holdout RMSE, local rank, robust
rejection count, map strategy, and input digests. The checked AgRob run uses a
range-adaptive voxel map plus a robust uniform fallback, deterministic MAD
rejection, and one correspondence-refinement pass. It uses caps of 6,000
source records and 6,000 target points per split, a six-point minimum adaptive
voxel support, and ten clock-profile iterations; it converged at `+37.5 ms`,
with train/holdout RMSE `0.2507/0.2246 m`, rank 6, and 491 rejected
correspondences. These are ground-truth-free fit diagnostics, not an absolute
accuracy claim. It is a fixed-odometry continuous-time refinement baseline:
the piecewise-SE(3) trajectory is interpolated but its knots are not estimated.
That boundary is explicit so it is not confused with iKalibr-style joint
trajectory estimation. Set `continuous_time_voxel_strategy: uniform` and
`continuous_time_outlier_policy: none` for a declared baseline comparison.

To materialize the fair public-data comparison in one user-facing command, run
the uniform/no-rejection baseline and the adaptive/MAD variant with identical
capture windows, temporal holdout, and solver budgets:

```bash
python tools/run_continuous_time_lidar_ablation.py \
  examples/public_datasets/agrob_modular_e/online_joint_config.yaml \
  --output-dir outputs/agrob_modular_e_livox_rslidar_continuous_time_ablation_hybrid_support6_iter10
calibrex validate \
  outputs/agrob_modular_e_livox_rslidar_continuous_time_ablation_hybrid_support6_iter10/ablation_manifest.yaml \
  --kind continuous-time-lidar-ablation
```

The checked manifest records both config/result SHA-256 digests. On a later
rerun, add `--reuse-only` to verify the materialized evidence without
recomputing the solver. On this AgRob recording, `adaptive_mad` converged at
`+37.5 ms` with `0.2507/0.2246 m` train/holdout RMSE and 491 rejected
correspondences; `uniform_none` converged at `+10 ms` with
`0.3113/0.2907 m` and no deterministic rejection. The adaptive holdout RMSE
is 22.7% lower on this recording. This is a ground-truth-free ablation result,
not evidence that either policy is universally superior.

### Cross-dataset solid-state benchmark

The same comparison can be aggregated across the locally available public
recordings. The declaration labels whether each dataset is a real sensor pair,
an identity control, or only a trajectory reference; the output does not turn
temporal holdout into absolute extrinsic ground truth.

```bash
calibrex validate examples/public_datasets/solid_state_cross_dataset_benchmark.yaml \
  --kind solid-state-cross-dataset-benchmark-config
python tools/run_solid_state_benchmark_replicates.py \
  examples/public_datasets/solid_state_cross_dataset_benchmark.yaml \
  --output-root outputs/solid_state_benchmark_v02_replicates \
  --output-spec examples/public_datasets/solid_state_cross_dataset_benchmark_v02.yaml \
  --split-id middle_holdout --split-id late_holdout \
  --seed 0 --seed 17
python tools/run_solid_state_cross_dataset_benchmark.py \
  examples/public_datasets/solid_state_cross_dataset_benchmark_v02.yaml \
  --output outputs/solid_state_cross_dataset_benchmark_v02.yaml \
  --markdown-output outputs/solid_state_cross_dataset_benchmark_v02.md \
  --html-output outputs/solid_state_cross_dataset_benchmark_v02.html
calibrex validate outputs/solid_state_cross_dataset_benchmark_v02.yaml \
  --kind solid-state-cross-dataset-benchmark
```

The checked v0.2 run uses four paired replicates per dataset (two temporal
holdout boundaries × two sampling seeds). Lower holdout RMSE is better:

| Dataset | Replicates | Adaptive win rate | Mean improvement (95% CI) | Winner | Reference label |
|---|---:|---:|---:|---|---|
| AgRob Modular-e | 4 | 0.25 | -3.63% (-12.68, 5.94) | uniform | real pair, trajectory-only |
| TIERS LidarsCali | 4 | 1.00 | 79.19% (74.19, 84.18) | adaptive | real pair, trajectory-only |
| AIST GLIM | 4 | 1.00 | 65.17% (62.18, 67.99) | adaptive | identity control |

Across 12 scored replicates, adaptive won 9/12 (win rate 0.75), with mean
improvement 46.91% and bootstrap 95% CI [25.36, 65.86]%. AgRob is a useful
counterexample: its adaptive result is not stable across the tested splits and
seeds. TIERS and GLIM still expose `max_iterations` in some artifacts, so the
failure classification remains visible even when the holdout comparison is
scored. These results are temporal-holdout evidence, not a universal SOTA
claim; independently surveyed extrinsic ground truth and unseen public
sequences remain the next validation gate.

To inspect the AgRob counterexample without tuning on its holdout, generate a
digest-bound diagnostic artifact:

```bash
python tools/run_solid_state_failure_analysis.py \
  outputs/solid_state_cross_dataset_benchmark_v02.yaml \
  --dataset-id agrob_modular_e \
  --output outputs/agrob_solid_state_failure_analysis.yaml \
  --markdown-output outputs/agrob_solid_state_failure_analysis.md \
  --html-output outputs/agrob_solid_state_failure_analysis.html
calibrex validate outputs/agrob_solid_state_failure_analysis.yaml \
  --kind solid-state-failure-analysis
```

The current analysis identifies three uniform-winning AgRob replicates where
adaptive has lower train RMSE but higher holdout RMSE, and records sampling-seed
and clock-profile variation as follow-up hypotheses. It also confirms that all
eight variants are rank 6 and converged, so the counterexample is not explained
by a simple rank or termination failure. Residual histograms and per-iteration
correspondence counts are not yet present in the v0.1 result artifact; the
findings are therefore explicitly labelled candidate causes.

As a controlled semantic ablation, `--adaptive-no-uniform-fallback` was also
run over the same 12 split/seed replicates. It was not adopted: AgRob mean
improvement changed from `-3.63%` to `-41.72%`, and the GLIM identity control
changed from `+65.17%` to `-82.22%`. The current hybrid fallback remains the
declared default while the next investigation targets adaptive plane support
and clock-profile selection.

For a multi-window capture, recompute the same map/odometry evidence per
window without rerunning the extrinsic solver:

```bash
calibrex trajectory-window-drift \
  examples/public_datasets/agrob_modular_e/online_joint_config.yaml \
  --reference-result outputs/agrob_modular_e_livox_rslidar_online_joint/result.yaml \
  --output outputs/agrob_modular_e_livox_rslidar_online_joint/trajectory_window_drift.yaml
calibrex validate \
  outputs/agrob_modular_e_livox_rslidar_online_joint/trajectory_window_drift.yaml
```

The `slac.trajectory_window_drift/v0.1` artifact records odometry coverage and
motion for every configured window, its local cross-segment RMSE and
correspondence count, the full-span comparison, calculation parameters, and
input digests. Use `--deskew on|off` and
`--odometry-burst-policy preserve|keep_first|keep_last` for controlled
diagnostic variants; the selected values are recorded in the artifact.
`local_inconsistency_suspected` means at least one window fails;
`long_span_inconsistency_suspected` is reserved for a full-span-only failure.
Both are ground-truth-free hypotheses and do not uniquely identify whether
odometry, map geometry, or sensor timing caused the residual.

The cross-segment pass is a separate bounded bag decode (~80 source messages on
Indoor02) and does not reuse the calibration replay's `max_source_messages`
budget.

Example (same bounded KISS-ICP replay budget as run C above):

```bash
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_kissicp_config.yaml --online
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_selftest_motion_config.yaml --online
```

**Indoor02 KISS-ICP Velodyne→Ouster:** `trajectory.json` — 420 poses recorded
(42.26 s span, 1 clamp / 39 interpolations, max extrapolation 0.024 s).
Kinematic **PASS** (p95 linear 0.96 m/s, max 1.07 m/s; p95 angular 34.8 deg/s,
max 50.9 deg/s). Interpolation **PASS** (clamp fraction 0.026). Cross-segment
**PASS** — 40 + 40 scans, 4000 + 4000 points (`evenly_spaced_time` over
21.13 s halves), 3799 correspondences, RMSE **0.143 m** (gate 0.30 m). Overall
trajectory verdict **PASS**. The proxy now spans the full KISS-ICP track and
produces a real RMSE; at voxel size 0.5 m the first-half map and second-half
returns are consistent within the 0.30 m gate. That does **not** contradict the
documented KISS-ICP drift narrative for absolute extrinsic accuracy (~85 cm
vs the TIERS seed in run C): cross-segment measures leave-segment-out map
consistency under the estimated odometry frame, not agreement with an external
reference.

**Velodyne selftest (duplicate topic):** same odometry track and source topic
geometry — kinematic **PASS**, interpolation **PASS** (zero clamps),
cross-segment **PASS** (3799 correspondences, RMSE 0.143 m). Verdict **PASS**.

| Run | Kinematic | Interpolation | Cross-segment | Trajectory verdict |
|-----|-----------|---------------|---------------|-------------------|
| KISS-ICP | PASS | PASS (clamp 0.026) | PASS (0.143 m, 3799 corr.) | PASS |
| Selftest motion | PASS | PASS (clamp 0) | PASS (0.143 m, 3799 corr.) | PASS |

### LiDAR-IMU rotation evidence (ADR 0007 Pillar 3)

The Indoor02 source ROS 1 bag carries `/os_cloud_nodee/imu` (OS1 internal IMU,
~100 Hz, 4231 messages). Convert a kiss-icp replay subset with IMU passthrough
and restamp both Ouster points and IMU (they share the Ouster since-boot clock):

```bash
uv run tools/rosbag1_to_rosbag2_online_pair.py \
  --src data/public/tiers_lidars_dataset/indoor02.bag \
  --dst data/public/tiers_lidars_dataset/indoor02_rosbag2_kissicp_imu \
  --topic /velodyne_points \
  --topic /os_cloud_nodee/points \
  --restamp-topic /os_cloud_nodee/points \
  --restamp-topic /os_cloud_nodee/imu \
  --imu-topic /os_cloud_nodee/imu \
  --odom-source kiss-icp \
  --kiss-icp-topic /velodyne_points \
  --kiss-icp-max-range 30.0 \
  --child-frame-id base_link \
  --storage sqlite3 \
  --compress none
```

**Clock-domain check (measured on the source bag):** median
`bag_receive_time − header.stamp` for `/os_cloud_nodee/imu` is
**1645462364.473 s** (p05–p95 span 0.015 s, σ ≈ 0.004 s) — the same
since-boot epoch as `/os_cloud_nodee/points` (~753 s offset from the recording
clock). **Decision: restamp** `/os_cloud_nodee/imu` alongside the points topic.

Evaluate an externally supplied `R_velo_imu` candidate (not IMU calibration):

```bash
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/imu_rotation_evidence_config.yaml
```

**Channel semantics**

| Channel | Role | Gate |
|---------|------|------|
| Rotation-rate consistency (primary) | `ω_odo = log(R_i⁻¹ R_{i+1})/Δt` and interval-mean `ω_imu`, each symmetrically low-pass filtered with the same ±N pose-interval centered moving average (`imu_rotation_rate_smoothing_intervals`, default 2 ≈ 0.5 s at 10 Hz) before holdout RMSE | `imu_gate_max_holdout_rotation_rate_rmse_dps` (default 5.0 deg/s) on contiguous-block holdout (every 5th block) |
| Known-bad rotation probes | ±5° / ±10° about roll, pitch, yaw (12 cases); detectable when holdout RMSE increases ≥ 20% relative | fraction ≥ 0.5 pass, ≥ 0.1 warn |
| Gravity support (supporting-only) | 0.5 s low-pass linear acceleration rotated to world vs upward gravity; **never fails** the decision boundary | warn if angle > 10° |

**Approximation (provenance):** the evaluated candidate is the documented TIERS
GICP seed `R_velo_os1` applied as `R_velo_imu`. Ouster documents a small lever
arm; the OS1 internal IMU is approximately rotation-aligned with the os1 sensor
frame. This is an honest engineering approximation, not metrology ground truth.

**Measured clean run** (`imu_rotation_evidence_config.yaml`, seed quat
`[-0.004, 0.021, 0.357, 0.934]`, symmetric MA ±2 intervals,
method `pose_interval_mean_imu_vs_odometry_log_symmetric_ma2`):

| Evidence row | Status | Values (threshold) |
|--------------|--------|-------------------|
| Candidate Support | pass | 4231 IMU samples, 419 pose intervals, excitation p95 **36.8 deg/s** (floor 5); per-axis p95 \|ω\| x **12.3** y **11.6** z **35.9 deg/s** — dominant **z** (yaw probes weak) |
| Holdout Rotation Consistency | pass | holdout RMSE **3.56 deg/s** (gate ≤ 5), mean \|Δω\| **2.64 deg/s** |
| Known-Bad Controls | warn | detectable fraction **2 / 12** (margin 0.20 relative); roll −10° **+34%**, pitch +10° **+40%** detected; all yaw probes **≈ 0%** (z-dominated excitation) |
| Gravity Support | pass | angle **4.42 deg** (warn > 10) |
| Decision Boundary | warn | holdout pass; known-bad fraction below 0.5 pass threshold |

**10° probe table (clean baseline RMSE 3.56 deg/s):**

| Axis | Angle | Perturbed RMSE | Relative Δ |
|------|-------|----------------|------------|
| roll | −10° | 4.77 deg/s | +34% |
| roll | +10° | 4.25 deg/s | +19% |
| pitch | −10° | 3.98 deg/s | +12% |
| pitch | +10° | 5.00 deg/s | +40% |
| yaw | −10° | 3.56 deg/s | ≈ 0% |
| yaw | +10° | 3.56 deg/s | ≈ 0% |

**Known-bad control** (`imu_rotation_evidence_known_bad_config.yaml`, seed +10°
yaw → quat `[-0.006, 0.020, 0.437, 0.899]`):

| Evidence row | Status | Values |
|--------------|--------|--------|
| Holdout Rotation Consistency | pass | holdout RMSE **3.56 deg/s** (Δ ≈ 0 vs clean — seed yaw error is weakly observable under z-dominated motion) |
| Decision Boundary | warn | same as clean (known-bad seed not separated by holdout RMSE) |

**Honest read:** Symmetric low-pass filtering of both rate series drops the
differentiation noise floor from ~8 deg/s to **~3.6 deg/s**, confirming the
pose-differentiation theory. The GICP seed `R_velo_os1 ≈ R_velo_imu` now passes
the 5 deg/s holdout gate. Roll/pitch probes regain falsification power; yaw
probes remain uninformative because excitation is z-dominated (p95 \|ω_z\| ≈ 36
deg/s vs \|ω_x\|, \|ω_y\| ≈ 12 deg/s). The +10° yaw known-bad seed is not
detectable — consistent with axis observability, not a metric-power failure.

### Online calibration with odometry motion compensation

`run_online_calibration` accepts `dataset.type: rosbag2` with the same bounded
replay budgets as rosbag1 (`max_source_messages`, `max_source_points`,
`max_target_messages`, `max_target_points`, `max_replay_duration_s` under the
`lidar_rig_point_to_plane` factor options). Set `dataset.odometry_topic` to a
`nav_msgs/msg/Odometry` topic to enable motion compensation during replay.

Formulation with optional per-point deskew (ROS 1/ROS 2, when
`sensors.<name>.point_time_field` is set and `dataset.odometry_topic` is
configured):

* `T_world_base(t)` — interpolated odometry pose (linear translation, quaternion
  slerp on the shortest arc; out-of-range queries clamp to the nearest pose).
* Source map points use capture time `t_i = message_stamp + offset_i` for ROS 2
  and `t_i = bag_record_timestamp + offset_i` for ROS 1 when deskew is active;
  otherwise the corresponding message/record time. Points transform to the
  world frame as `p_world = T_world_base(t_i) * T_base_source * p_sensor`
  before voxel-plane map construction.
* Target points keep sensor-frame coordinates and record per-point
  `T_world_source(t_j_point)` when deskew is active, else per-message
  `T_world_source(t_msg)` for correspondence (`T_world_source *
  T_hat_source_target * p` against world-frame planes). The solver still
  estimates one constant `T_source_target`.
* When `dataset.odometry_topic` is unset, the static-rig path is unchanged
  (identity frame poses) and `point_time_field` is ignored.

Replay provenance records the odometry topic, message count, time coverage,
interpolation method, clamp count, total interpolation count,
`odometry_interpolation_max_extrapolation_s`, a `motion_compensated` flag, and
when `dataset.odometry_preprocessing` is configured, the raw/retained pose
counts, selected burst policy, interval, and removed count. The raw bag is
never rewritten by this preprocessing stage. It also records
when configured `deskew_applied`, `deskew_time_field`, `deskew_span_s`, or
`deskew_ignored_no_odometry`.
Batches whose target timestamps extrapolate beyond
`online_gate_max_odometry_extrapolation_s` (default 0.25 s) outside the odometry
track are excluded with gate reason `odometry_extrapolation`; source messages
beyond the same tolerance raise `DatasetError` because the map premise is broken
globally. The online timeline artifact (`slac.online_timeline/v0.3`) records
gate reasons as free-form strings on each batch snapshot (no schema change).

```yaml
dataset:
  type: rosbag2
  path: path/to/bag_dir_or.db3
  odometry_topic: /odom
sensors:
  lidar_map:
    type: lidar
    topic: /source/lidar
  lidar_stream:
    type: lidar
    topic: /target/lidar
```

`--readme-gallery` regenerates the Livox and A2D2 README GIF assets from public
raw samples. It downloads the small A2D2 range sample when needed and refuses to
use built-in fallback geometry unless `--allow-metadata-fallback` is passed.
The README gallery includes an online-style A2D2 replay that shows live point
batches, a converging extrinsic estimate, rolling residuals, holdout checks, and
provenance-backed policy gates. To render only that hero asset, run:

```bash
python3 tools/generate_calibration_evidence_gif.py \
  --source a2d2 \
  --a2d2-source-id 0 \
  --a2d2-target-id 3 \
  --visual online \
  --output docs/assets/online-calibration-loop.gif
```

The cached Livox evidence result is a report-rendering fixture, not a claim
that raw observations were reread and recomputed.
`calibrex evidence` materializes an `evidence.json` sidecar from an existing
result and does not recompute metrics from raw observations.
Its `calibrex render --json` payload includes `render_only: true`,
`recomputed_metrics: false`, and `evidence_case_count`; the cached fixture keeps
representative roll/pitch/yaw/x/y/z cases, while raw
`calibrex calibrate` recomputes the full configured perturbation set when the
public PCD files are available locally.
Generated `summary.json` and `evidence.json` include a `materialization` block
with `metrics_origin`, `data_verified`, `computed_at`, and
`report_generated_at`. Cached fixtures show `metrics_origin: cached` and
`data_verified: false`, and the HTML report displays a cached-evidence banner.
Raw Livox recomputation also writes `evidence.json.input_files` with the PCD
paths, SHA-256 digests, byte sizes, and known public source URLs.
The built-in falsification policy only passes the raw recomputation gate when
`metrics_origin: recomputed`, `data_verified: true`, and SHA-backed
`input_files` are all present.
Use `calibrex verify --require-raw-recomputed bundle.json` when that same
requirement should be enforced as an integrity gate instead of only reported as
assessment evidence.
The Livox public demo applies this gate before reporting `bundle_valid: true`.
It also writes `verification.json` beside `bundle.json` so the gate result can
be reviewed without rerunning `verify`.
Generated report directories also include `bundle.json`, which records SHA-256
digests for the HTML report and machine-readable sidecars, and
`verification.json`, which materializes the default bundle integrity check.
`protocol.json` records the declared evidence protocol. `policy.json` records
the falsification gates and thresholds, while `assessment.json` records the
applied policy result. `transforms.json` records candidate, reference, and
output transform estimate sets as a standalone artifact. Cached or
non-independent evidence can score useful known-bad controls while still
returning `INCONCLUSIVE` because raw observations were not recomputed or the
holdout split is not independent.
`calibrex assess --policy policy.json` reapplies the declared policy to an
existing `evidence.json`, so reviewers can verify that the verdict came from a
versioned artifact rather than hidden CLI defaults. By default, `assess`
returns success when it writes a valid assessment artifact, even if the
assessment status is `FAIL` or `INCONCLUSIVE`; add `--enforce` when a non-pass
assessment should fail a shell pipeline.
`calibrex verify` checks those digests and catches stale or mixed report
artifacts. For raw recomputation artifacts with `input_files`, it also checks
the referenced raw file sizes and SHA-256 digests. The verify JSON includes
`input_file_count`, `checked_input_file_count`, and `checked_input_files`;
cached report fixtures normally report zero input files. It also includes
`verification_summary` with total, ok, failed, skipped, and per-scope claim
counts, so automated review can fail quickly without parsing every claim.
Verification also checks semantic links between sidecars: `protocol.json` must
match the protocols embedded in `evidence.json`, and `policy.json` must match
the policy and gate fingerprints recorded by `assessment.json`.
The detailed record remains in
`verification_claims`, where each claim records a scope, status, method,
expected values, observed values, and scoped issues for artifact digests,
run consistency, assessment source links, and raw input digests.
Without `--json`, `calibrex verify` prints the same rollup as a compact
review summary before any issue list.
The saved `verification.json` is schema-versioned and can be validated or
stored as a review artifact. It records the verified bundle path, SHA-256,
size, schema version, and run id under `source_bundle`, plus the primary
evidence `metrics_origin` / `data_verified` values under
`primary_evidence_materialization`. You can also pass the saved
`verification.json` back to `calibrex verify`; Calibrex recomputes the source
bundle check and reports `verification_record` claims if the saved record is
stale or edited. Saved verification artifacts use relative `source_bundle`
paths when the bundle is colocated with, or near, the verification file.
If `--require-raw-recomputed` is added while checking a saved verification,
Calibrex applies that stricter raw-input gate as an additional claim without
treating the saved default verification record as stale.
Derived sidecars such as `summary.json`, `metrics.json`, `observability.json`,
and `degeneracy.json` also include `source_evidence` so reviewers can verify
which immutable `evidence.json` they summarize.
Use `calibrex calibrate` or future dataset-backed `evaluate` flows when metrics
must be recomputed from raw observations.
When comparing two results, `calibrex compare` reports
`protocol_compatibility` as `compatible`, `warning`, or `not_comparable`, and
prints `left_materialization` / `right_materialization` in non-JSON output.
This keeps cached evidence, recomputed evidence, and different holdout
protocols from being silently ranked as if they were produced under the same
conditions. By default, comparison artifact generation succeeds even when the
protocols are not comparable; add `--enforce-compatible` when protocol warnings
or non-comparable inputs should fail a shell pipeline.

Autonomous driving:

```bash
calibrex inspect examples/public_datasets/kitti_raw_2011_09_26_drive_0005 --type kitti-raw
Calibrex compile examples/public_datasets/kitti_raw_2011_09_26_drive_0005/config.yaml
calibrex inspect data/public/nuscenes --type nuscenes
```

KITTI raw data and nuScenes require their official download flows and terms.
After downloading, point the dataset manifest paths at the local extracted
dataset.

KITTI raw camera-LiDAR evaluated projection evidence demo:

```bash
calibrex calibrate examples/public_datasets/kitti_lidar_camera_evidence/config.yaml
calibrex demo kitti-lidar-camera-evidence --output-dir outputs/kitti_lidar_camera_evidence
```

Because KITTI raw data requires the official login-gated download flow,
Calibrex cannot fetch it automatically the way it does for the Livox pair
demo. This demo instead defaults to a small synthetic fixture bundled at
`examples/public_datasets/kitti_lidar_camera_evidence/` that mirrors the
KITTI raw directory layout, so the command is runnable end to end with no
setup. Point it at a real, locally downloaded KITTI raw sequence with
`--dataset-path /path/to/2011_09_26/2011_09_26_drive_0005_sync` to evaluate
real data instead. The demo writes a materialized `demo_config.yaml`,
recomputes `result.yaml` using the dataset's `calib_velo_to_cam.txt` extrinsic
as the reference/output transform (unless
`evaluation.kitti.use_frame_graph_candidate: true`), renders `evidence.json`,
`assessment.json`, `protocol.json`, `policy.json`, and `transforms.json`, and
verifies `bundle.json`.

Before running the full-scale benchmark, lock the prespecified raw inputs:

```bash
calibrex kitti lock-benchmark-input \
  /path/to/2011_09_26/2011_09_26_drive_0005_sync \
  --output outputs/kitti_0005/kitti-benchmark-input.json
calibrex validate outputs/kitti_0005/kitti-benchmark-input.json
```

The lock uses 20 frame pairs fixed in advance at indices `0:154:8`, spanning
the complete drive. It records each selected image and Velodyne file, both
timestamp files, both required calibration files, their byte sizes and
SHA-256 digests, plus a deterministic aggregate input digest. The command
rejects the bundled two-frame synthetic fixture and incomplete official
downloads rather than silently changing the selected frames. Raw KITTI data
is never copied into the output artifact.

Compare the scalar-reference I2I optimizer with the vectorized,
safety-gated deterministic coarse-to-fine path on the same locked frames and
perturbations:

```bash
calibrex kitti benchmark-i2i \
  outputs/kitti_0005/kitti-benchmark-input.json \
  --definition-output outputs/kitti_0005/i2i-benchmark.definition.json \
  --output outputs/kitti_0005/i2i-benchmark.json
calibrex validate outputs/kitti_0005/i2i-benchmark.json
```

Both methods receive the same six prespecified `+0.10 m` or `+10 deg`
single-axis initial errors and the same seeded frame holdout. The resulting
standard benchmark artifacts retain failures, runtime, translation/rotation
error to the KITTI dataset reference, held-out normalized MI, the exact input
digest, executed source digest, and producer/execution provenance. The command
also reports `accuracy_non_degraded`, `runtime_improved`, and
`performance_gate_passed`; non-degradation is checked for every paired trial,
not only for aggregate means. KITTI calibration is treated as a dataset
reference rather than independent metrology.

The demo reports the `lidar_camera_projection_*`,
`lidar_camera_edge_alignment_score`, and `lidar_camera_perturbation_*`
projection evidence metrics described below, plus `koide_lidar_camera_*`
adapter readiness metrics — the demo does not execute the external Koide-style
adapter, so those metrics reflect availability, not a computed result. This
is an evaluation layer for externally produced camera-LiDAR candidates, not a
standalone camera calibration algorithm.

`lidar_camera_baseline_comparison` validates a structured LiDAR capture-time
policy and records camera exposure as the reference instant. KITTI `.bin`
payloads have no per-point firing offsets, so the demo deliberately reports
`lidar_camera_capture_time_deskew_applied=0` (WARN) and does not infer offsets
from point order. Every non-reference candidate is also compared with
`kitti_dataset_reference` in SE(3): translation/rotation distance and
candidate-minus-reference holdout edge evidence are stored together. These are
diagnostics, not PASS thresholds; a small evidence delta can reflect weak
observability rather than transform agreement.

Evidence rows (protocol `kitti_lidar_camera_projection_edge_holdout/v0.1`) use
recorded thresholds from `evaluation.kitti.evidence_gate_*` (defaults mirror
metric grading: holdout edge-alignment `>= 0.20`, depth-edge holdout
`>= 0.20`, perturbation detectable fraction `>= 0.50`, mandatory probe
detections `>= 8` of the declared ±1 deg / ±0.10 m cases). On the committed
76 KB synthetic fixture with the dataset reference extrinsic (baseline run):

| Check | Measured (holdout where applicable) | Verdict |
| --- | --- | --- |
| Candidate Support | 12 projected points, ratio 1.0, h/v coverage 0.2 | PASS |
| Holdout Edge Alignment | edge 0.5, depth-edge 0.5 (threshold >= 0.2) | PASS |
| Known-Bad Controls | detectable fraction 0.0, mandatory 0/24 (threshold >= 0.5 / >= 8) | FAIL |
| Decision Boundary | support + holdout pass, controls fail | WARN (INCONCLUSIVE) |
| Overall quality | evidence decision boundary inconclusive | FAIL |

The baseline FAIL on this fixture is the discipline working, not a defect: with
zero detectable perturbations the protocol has no falsification power, and the
framework refuses to certify a candidate it could not have caught being wrong
(the same `>= 0.50 / >= 0.10` detectable-fraction grading the LiDAR-pair family
uses). On full-scale KITTI frames the perturbation probes are expected to be
detectable, making PASS reachable.

Known-bad candidate probe: set `evaluation.kitti.use_frame_graph_candidate:
true` and perturb the frame-graph candidate (example: `+2.0 m` x translation on
`lidar0.initial`). A `+2 deg` yaw rotation does not move holdout scores on
this tiny synthetic fixture; `+2.0 m` x translation drives holdout edge-alignment
to `0.0` and flips that row to FAIL while overall quality remains FAIL. Set
`evaluation.kitti.use_frame_graph_candidate: true` whenever the candidate
under test comes from the config rather than `calib_velo_to_cam.txt`.

A cached example result is available for report rendering without rerunning
the pipeline:

```bash
calibrex render examples/public_datasets/kitti_lidar_camera_evidence/cached_evidence_result.yaml \
  --output-dir outputs/kitti_lidar_camera_evidence
```

nuScenes inspection reads the JSON metadata tables directly, without requiring
the nuScenes SDK. It discovers sensor streams from `sensor.json`,
`calibrated_sensor.json`, and `sample_data.json`, then reports channel counts,
modality counts, keyframe counts, missing local sample files, ego pose table
coverage, and calibrated sensor transforms normalized to Calibrex `xyzw`
quaternions. This is the first nuScenes step for validating that Camera, LiDAR,
Radar, and ego-pose metadata fit the same Calibrex dataset inspection model
used by KITTI.

### nuScenes mini Radar yaw evidence

nuScenes requires registration and acceptance of its current dataset terms.
Download the mini split manually from the [official nuScenes page](https://www.nuscenes.org/nuscenes),
review the [terms of use](https://www.nuscenes.org/terms-of-use), and do not
commit or redistribute the downloaded archives through slac. Extract the data
so the local tree contains:

```text
data/public/nuscenes/
├── v1.0-mini/
│   ├── calibrated_sensor.json
│   ├── ego_pose.json
│   ├── sample_data.json
│   └── sensor.json
├── samples/
│   └── RADAR_FRONT/
└── sweeps/
```

Then run the preflight and evidence workflow:

```bash
calibrex inspect data/public/nuscenes --type nuscenes --json
calibrex calibrate examples/public_datasets/nuscenes_mini/config.yaml \
  --output-dir outputs/nuscenes_mini_radar --seed 0
calibrex evidence outputs/nuscenes_mini_radar/result.yaml \
  --output outputs/nuscenes_mini_radar/evidence.json \
  --sidecars-dir outputs/nuscenes_mini_radar
calibrex validate outputs/nuscenes_mini_radar/evidence.json --json
calibrex assess outputs/nuscenes_mini_radar/evidence.json \
  --policy outputs/nuscenes_mini_radar/policy.json \
  --output outputs/nuscenes_mini_radar/reassessment.json
calibrex verify outputs/nuscenes_mini_radar/bundle.json
```

The example evaluates the dataset-provided `T_ego_radar` reference, uses a
seeded frame-disjoint 20% holdout, and applies the mandatory -10, -5, +5, and
+10 degree yaw controls across `RADAR_FRONT`, `RADAR_FRONT_LEFT`,
`RADAR_FRONT_RIGHT`, `RADAR_BACK_LEFT`, and `RADAR_BACK_RIGHT`. Frame IDs are
prefixed by the configured sensor name, so simultaneous acquisitions from
different Radar units cannot leak across the split. The absolute residual limits remain deliberately
unset, so the decision boundary stays `INCONCLUSIVE` even when support,
observability, and controls pass. Do not tune those limits on the reported
holdout. Freeze them on predeclared development scenes, then evaluate once on
untouched validation scenes and record negative results unchanged.

The same config also runs `radar_spatiotemporal_velocity` for `RADAR_FRONT`.
It recomputes planar scan ego velocity from static raw `vx`/`vy` Doppler
returns, differentiates the corresponding `ego_pose` trajectory, and passes
those timestamped signals to the native lever-arm/clock-offset solver. The
factor declares `radar_time + dt_radar = reference_time`, a ±0.10 s search,
and a 2 ms grid resolution. Full 3D translation is accepted only when the
stacked angular-rate cross-product matrix has rank three. Normal road driving
often supplies yaw-only motion and should therefore produce an honest
`INCONCLUSIVE` lever-arm result rather than an invented vertical translation.
The accepted solution must also have rank four in the probe-scaled joint
`(tx, ty, tz, dt)` Jacobian. Its condition number and the fraction of the time
column explained by the translation subspace expose lever-arm/clock
compensation that a low residual or positive one-dimensional curvature can
miss. All consumed PCD and metadata files are recorded with SHA-256 digests;
the supplied Radar rotation and solver options are stored in provenance.

The same timestamped velocities also run an independent three-axis rotation
stage from Wise et al. (ICRA 2021), using the configured translation for the
lever-arm term. Its holdout RMSE, three-axis information rank/condition, and
roll/pitch/yaw ±5 degree controls are reported. The estimated rotation is not
fed into the lever-arm/clock stage in that run. Ordinary road motion can still
be poorly conditioned; such data remain `INCONCLUSIVE` with the weak spectrum
preserved instead of being reduced to a yaw-only success.

A third, independent path jointly refines all six Radar extrinsic components
and the bounded clock offset with the backend-neutral LM optimizer. It uses
the same eligible scan IDs and train/holdout split as the staged solver, which
is checked by `radar_joint_spatiotemporal_common_split_consistent`. Rank must
be 7/7 and all fourteen signed spatial/temporal controls remain visible; the
joint estimate is diagnostic and is not automatically applied on weak road
motion. Its configured initial SE(3)/clock state and optimized state are both
scored on the unchanged holdout IDs. The absolute and fractional RMSE
improvements therefore expose a converged estimate that fails to outperform
the supplied calibration without feeding holdout evidence back into fitting.

`input_diagnostics` records missing extrinsics, insufficient records, missing
ego poses, low-motion exclusions, missing or malformed PCD payloads, and frames
without eligible static returns. Counts are complete, while example events are
capped at 100. Stored payload references are dataset-relative; exception text
and personal absolute paths are intentionally omitted.

For reproducibility, retain the downloaded archive name and SHA-256 locally,
the access date and terms URL, the exact config and commit, environment lock,
selected frame IDs, split seed, calibrated-sensor tokens, all four probe
results, and the generated bundle verification. Avoid publishing personal
absolute paths or licensed raw point data.

When `calibrex calibrate` runs on `dataset.type: nuscenes`, Calibrex imports
those `calibrated_sensor` entries into result `reference_extrinsics` with
`T_parent_child` convention, `ego` as parent, and channel-derived child frame
names such as `lidar_top`, `cam_front`, and `radar_front`. These are dataset
reference values, not optimized Calibrex estimates, so they stay separate from
`transforms` and future `optimized_extrinsics` outputs.
Calibrex also stores config-derived initial transforms in `candidate_extrinsics`.
When a candidate and reference share the same parent/child edge, the report
adds `extrinsic_reference_*` metrics for pair count, maximum translation delta,
and maximum rotation delta. This is the first comparison path for evaluating
dataset calibration, manual candidate files, and future external baseline
outputs without mixing their meanings in one field.
External candidates can be supplied without editing the dataset config:

```bash
calibrex calibrate config.yaml --candidate-extrinsics candidates/manual.yaml
```

The candidate file may contain a top-level `candidate_extrinsics` mapping, a
`transforms` mapping from an existing Calibrex-style result, or a direct mapping
from transform name to `T_parent_child` transform fields. Imported candidates
override config-derived candidates with the same transform name and are recorded
under `run.provenance.external_candidate_extrinsics`.

For downloaded KITTI sequences, `calibrex inspect --type kitti-raw --json`
includes sampled Velodyne diagnostics: frame count, sampled point count, XYZ
bounds, intensity range, local planarity, roughness, map sharpness proxy, and
malformed file warnings.
Without `--json`, the same command prints a human-readable summary with streams,
warnings, LiDAR quality, point-to-plane metrics, and collection recommendations.
When KITTI OXTS files are present, `inspect` also reports motion excitation
diagnostics such as duration, mean speed, speed range, and yaw/pitch/roll
excitation. KITTI timestamp files are also compared to report Camera-LiDAR and
LiDAR-OXTS nearest timestamp deltas.
When image and Velodyne files are present, `inspect` also records concrete
camera-LiDAR frame pairs selected by nearest timestamp. These pairs include the
image path, `.bin` path, signed `lidar_time - camera_time` delta, and LiDAR point
count so overlay artifacts can reference real public-dataset frames.
When KITTI `calib_velo_to_cam.txt` and `calib_cam_to_cam.txt` are available,
`calibrex calibrate` and `calibrex visualize --export-html` project a sampled
Velodyne frame into the selected camera image inside
`artifacts/camera_lidar_overlay.html`.
The same projection feeds report metrics for projected point count, projection
ratio, median projected depth, depth span, and normalized horizontal/vertical
image coverage. A lightweight edge-alignment proxy also checks how many
projected LiDAR points land near image intensity edges.
For depth-aware checking, Calibrex also marks projected LiDAR points with
nearby depth jumps and scores whether those depth discontinuities land near
image edges. When more than one inspected camera-LiDAR frame pair is available,
these projection metrics are reported with train/holdout values instead of
judging the fixed-LiDAR calibration from a single frame.
Use `evaluation.kitti.max_projection_pairs` to increase the number of public
dataset frames used for those metrics. For quick checks, 3 frames is enough; for
fixed vehicle LiDAR validation, use 10 to 50 frames when the sequence is
available locally.
KITTI evaluation can also run known perturbation sweeps around the dataset
reference calibration:

```yaml
evaluation:
  kitti:
    perturbation_rotation_deg: [0.5, 1.0]
    perturbation_translation_m: [0.05, 0.10]
```

The resulting `lidar_camera_perturbation_*` metrics show whether roll, pitch,
yaw, x, y, and z offsets make projection and edge diagnostics worse. Treat this
as sensitivity and ranking evidence, not as absolute ground truth.

Those diagnostics feed the first LiDAR quality metrics in result files:
`lidar_frame_coverage`, `lidar_point_coverage`, and
`lidar_spatial_coverage_m`, plus geometry proxies such as
`lidar_local_planarity`, `lidar_map_roughness_m`, and `lidar_map_sharpness`.
The same local voxel planes also provide the alpha
`lidar_point_to_plane_rmse_m` metric with train/holdout values.
When KITTI OXTS packets are available, Calibrex additionally builds an
OXTS-projected LiDAR train map and scores holdout frames against that map. The
resulting `lidar_world_map_point_to_plane_rmse_m`,
`lidar_world_map_point_to_plane_median_holdout_m`, and
`lidar_world_map_point_to_plane_p95_holdout_m` metrics are the first
fixed-rig LiDAR map-consistency signals for vehicle extrinsic validation.
The matching `lidar_world_map_perturbation_*` metrics rerun that OXTS-projected
map consistency check after known extrinsic perturbations, so the report can
say whether a sequence has enough signal to rank bad roll, pitch, yaw, x, y, or
z candidates worse than the reference.
Calibrex also reports `lidar_world_map_sensitivity_*` and
`lidar_world_map_weak_dof_count`. A weak DoF warning means the evaluated public
dataset segment did not move the world-map holdout metric enough for that
direction, so the result should be treated as under-observed rather than
trusted.
The HTML report includes a dedicated LiDAR World-Map Diagnostics section with
the summary metrics and a DoF sensitivity table, so weak directions are visible
without digging through the full metric list.
For fixed vehicle LiDAR ranking, Calibrex also reports `lidar_perturbation_*`
metrics. These rerun the LiDAR voxel-plane proxy after known roll, pitch, yaw,
x, y, and z perturbations, then report train/holdout RMSE deltas. Positive
deltas mean the dataset reference calibration ranked better than the perturbed
candidate under the current LiDAR-only proxy.
Calibrex also turns these diagnostics into provisional degeneracy warnings, for
example when KITTI Velodyne samples contain too few planar neighborhoods, weak
vertical structure, or holdout point-to-plane degradation.
The report recommendations then say what to recollect, such as a longer
fixed-LiDAR sequence, more vertical structure, static planar surfaces, or
cleaner holdout passes with less dynamic traffic. Weak OXTS motion excitation
adds recommendations for longer driving logs, acceleration/braking, and turns.
Large timestamp deltas add a recommendation to verify synchronization or enable
time-offset estimation.

The KITTI raw example models the Velodyne LiDAR as a fixed-mounted sensor on
the vehicle and compiles a `fixed_lidar_mount_prior`
factor together with camera reprojection, LiDAR surfel, LiDAR-camera alignment,
and IMU preintegration descriptors.

KITTI is also the first public dataset target for SceneCalib/SST-Calib-inspired
development: targetless camera-LiDAR calibration, time-offset estimation, and
fixed-rig LiDAR quality gates should be exercised here before adding private or
synthetic-only workflows.

Import KITTI fixed-LiDAR calibration initial values:

```bash
Calibrex kitti import-calib /path/to/2011_09_26 --output /tmp/kitti_transforms.yaml
```

This reads `calib_velo_to_cam.txt` and exports Calibrex `T_parent_child`
transforms such as `T_camera0_lidar0`.

Official sources:

- TUM RGB-D dataset: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
- KITTI raw data: https://www.cvlibs.net/datasets/kitti/raw_data.php
- nuScenes: https://www.nuscenes.org/nuscenes

Research references:

- SceneCalib: https://arxiv.org/abs/2304.05530
- SST-Calib: https://arxiv.org/abs/2207.03704
- Decentralized multi-LiDAR SLAC: https://arxiv.org/abs/2007.01483
- M-LOAM: https://arxiv.org/abs/2010.14294
- Koide et al. LiDAR-camera calibration: https://arxiv.org/abs/2302.05094
