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

Solid-state LiDAR-to-LiDAR evidence demo:

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
(`calibrex.data.rosbag1`); no ROS installation is required. Decoding
`sensor_msgs/PointCloud2` payloads to arrays requires numpy
(`pip install "calibrex[rosbag1]"`); bags with `lz4` chunk compression
additionally need `pip install "calibrex[rosbag1-lz4]"` (`none` and `bz2`
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

`calibrex inspect --type rosbag1` lists every `sensor_msgs/PointCloud2` topic
with message counts, and samples a few messages per topic to report decoded
point counts, intensity presence, spatial bounds, and first/last ROS
timestamps (normalized to integer nanoseconds). Other message types are
ignored by design; the alpha reader is scoped to point clouds.

## ROS 2 bag (rosbag2) reader

Calibrex reads rosbag2 recordings directly with a pure-Python parser
(`calibrex.data.rosbag2`); no ROS installation is required. Supported storage
backends are **sqlite3** (``.db3``, the default through Humble) and **mcap**
(``.mcap``, the default from Iron). A bag directory with `metadata.yaml` or a
bare ``.db3`` / ``.mcap`` file path both work.

Decoding ``sensor_msgs/msg/PointCloud2`` and ``nav_msgs/msg/Odometry`` payloads
requires numpy (`pip install "calibrex[rosbag2]"`). MCAP bags with chunked
``lz4`` or ``zstd`` compression additionally need
`pip install "calibrex[rosbag2-compression]"` (`none` / uncompressed chunks
work out of the box).

```bash
calibrex inspect path/to/bag --type rosbag2 --json
```

`calibrex inspect --type rosbag2` reports per-topic message counts, sampled
decoded point counts for PointCloud2 topics, and a pose sample for Odometry
topics (position, orientation `xyzw`, and pose covariance diagonal entries from
the first sampled message).

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
control (B), and KISS-ICP rig-frame odometry (C):

```bash
pip install -e ".[dev,rosbag1-lz4,rosbag2,rosbag2-compression]"
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_motion_config.yaml --online
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_static_config.yaml --online
calibrex calibrate examples/public_datasets/tiers_lidars_dataset_indoor02/online_kissicp_config.yaml --online
```

On a bounded replay of the restamped converted bag (12 Velodyne source messages
/ 8k source points, 36 Ouster target messages / 1.5k target points, 60 s budget),
all three runs produced 108 Ouster batches (500 points each). Before restamping,
every target interpolation was clamped to the odometry track start
(`odometry_interpolation_clamp_count: 36` / 36 target messages in run C) because
Ouster header stamps sat in a different clock domain than `/odom`; motion
compensation was frozen at the first pose and any compensated run was not
measuring real accuracy. Calibrex now gates on odometry extrapolation
(`online_gate_max_odometry_extrapolation_s`, default 0.25 s): batches whose
target timestamps lie beyond that tolerance outside the odometry track are
excluded with reason `odometry_extrapolation` instead of silently clamping.

After restamping, motion-compensated runs A and C show near-zero extrapolation
(clamp count 0–1, `odometry_interpolation_max_extrapolation_s` ≤ 0.025 s). Run A
(MOCAP odometry) adopted all 108 batches. The static control (B) adopted 106 and
rejected 2 batches whose holdout RMSE exceeded the 0.40 m gate (batches 66 and
105). KISS-ICP rig-frame odometry (C) adopted 106 and rejected 2 batches for
holdout RMSE (batches 66 and 102) — not for extrapolation. Per-batch holdout
RMSE ranged 0.20–0.38 m for A (mean ~0.28 m), 0.13–0.42 m for B (mean ~0.28 m),
and 0.17–0.41 m for C (mean ~0.28 m). Final rolling RMSE settled near 0.28 m
(A), 0.29 m (B), and 0.31 m (C). All runs kept rank 6 with condition number
~4–9.

The TIERS README GICP seed (`frames.ouster_os1.transform.initial`, inter-sensor
distance ~0.37 m) is a nominal reference, not ground truth. Final estimates
drifted substantially from that seed for the MOCAP odometry run (A: ~287 cm
translation / ~46° rotation), consistent with an unknown constant offset between
the MOCAP rigid-body frame and `base_link` — rotation smears the world map with
scene distance and platform excursion. On that sequence the MOCAP motion-compensated
final extrinsic is **not** validated against the TIERS reference: a ~2.9 m
translation drift on a physically ~0.37 m sensor pair is not agreement at any
level. The static control (B) drifted ~106 cm / ~15° from the seed. Run C now
measures genuinely motion-compensated accuracy (restamped timestamps, extrapolation
gate clean) but still lands ~85 cm / ~28° from the seed — larger than the
inter-sensor baseline (~0.37 m) and well outside the holdout residual band.

Run C removes the external MOCAP body-alignment ambiguity structurally: KISS-ICP
poses are `T_world_velo_sensor` with the Velodyne as the root frame. The
KISS-ICP trajectory itself is sane on this sequence (~7 m office-scale excursion,
unit quaternions, no consecutive-pose translation jumps above 0.11 m). C is
closer to the TIERS GICP seed than A (~85 cm vs ~287 cm translation) but still
~2.3× the inter-sensor baseline and further from the seed than B in translation
(85 cm vs 106 cm is better, not worse). **Absolute extrinsic accuracy against the
TIERS reference is not established** by C: the residual ~85 cm / ~28° gap is
larger than the measured holdout residuals (~0.28 m mean) and cannot be read as
confirmation of the GICP seed. The online gates measure internal consistency of
the (possibly biased) world map, not absolute extrinsic accuracy. The VRPN
rigid-body (`UWBTest`) alignment caveat applies only to run A.

This validation did establish that real rosbag2 bags from an independent encoder
(`rosbags`) exposed two reader bugs that synthetic mirror-image tests could not:
CDR encapsulation endianness was keyed off the wrong header byte, and
message-mode zstd compression declared in `metadata.yaml` was ignored. With both
fixed, `calibrex inspect` decodes the regenerated sqlite3 and MCAP+zstd bags
with standard CDR headers and per-message decompression.

| Run | Odometry | Batches adopted | Holdout RMSE (m) | Final Δ vs TIERS seed |
|-----|----------|-----------------|------------------|------------------------|
| A motion | MOCAP `/odom` | 108 / 108 | 0.20 – 0.38 (mean ~0.28) | ~287 cm, ~46° |
| B static | none | 106 / 108 | 0.13 – 0.42 (mean ~0.28) | ~106 cm, ~15° |
| C kiss-icp | KISS-ICP `/odom` | 106 / 108 | 0.17 – 0.41 (mean ~0.28) | ~85 cm, ~28° |

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

On the bounded replay budget above, motion-compensated selftest (KISS-ICP
`/odom`) adopted all 104 batches with zero odometry clamping and zero
extrapolation. The static control (no odometry) also adopted all 104 batches
and passed every internal gate — yet both landed far from identity truth in
opposite ways. Motion compensation cut the extrinsic error roughly 10× versus
the static run; the residual ~10 cm / ~1.2° is consistent with no per-point
deskew and LiDAR-odometry drift on a moving platform.

| Run | Odometry | Batches adopted | Final Δ vs identity (truth) |
|-----|----------|-----------------|-------------------------------|
| Selftest motion | KISS-ICP `/odom` | 104 / 104 | ~9.6 cm, ~1.2° |
| Selftest static | none | 104 / 104 | ~98 cm, ~37° |
| Ground truth | — | — | 0 cm, 0° |

This is the first genuine absolute-accuracy validation on real moving-platform
data with exact ground truth: motion compensation reduces extrinsic error from
~0.98 m / ~37° to ~9.6 cm / ~1.2°. The honest negative finding is that the
static run passed all internal gates (holdout RMSE 0.40 m, rolling regression,
rank 6) while being ~1 m / ~37° wrong — on this scene the online gates measure
internal consistency of a self-consistently wrong map, not absolute accuracy.
Tighter correspondence gating and per-point deskew are candidate future work;
they are not implemented here.

Reconciling the earlier A/B/C discussion: the Velodyne→Ouster ~85 cm residual
in run C is now attributable primarily to cross-sensor effects (Ouster stamp
semantics and restamp bias, no per-point deskew, nominal GICP seed), not a
failure of the motion-compensation pipeline itself.

### Online calibration with odometry motion compensation

`run_online_calibration` accepts `dataset.type: rosbag2` with the same bounded
replay budgets as rosbag1 (`max_source_messages`, `max_source_points`,
`max_target_messages`, `max_target_points`, `max_replay_duration_s` under the
`lidar_rig_point_to_plane` factor options). Set `dataset.odometry_topic` to a
`nav_msgs/msg/Odometry` topic to enable motion compensation during replay.

Formulation (per-message poses only; per-point deskew inside a message is out of
scope):

* `T_world_base(t)` — interpolated odometry pose (linear translation, quaternion
  slerp on the shortest arc; out-of-range queries clamp to the nearest pose).
* Source map points at time `t_i` are transformed to the world frame as
  `p_world = T_world_base(t_i) * T_base_source * p_sensor` before voxel-plane
  map construction.
* Target messages at time `t_j` keep points in the target sensor frame and
  record `T_world_source(t_j) = T_world_base(t_j) * T_base_source` for
  correspondence (`T_world_source * T_hat_source_target * p` against
  world-frame planes). The solver still estimates one constant `T_source_target`.
* When `dataset.odometry_topic` is unset, the static-rig path is unchanged
  (identity frame poses).

Replay provenance records the odometry topic, message count, time coverage,
interpolation method, clamp count, total interpolation count,
`odometry_interpolation_max_extrapolation_s`, and a `motion_compensated` flag.
Batches whose target timestamps extrapolate beyond
`online_gate_max_odometry_extrapolation_s` (default 0.25 s) outside the odometry
track are excluded with gate reason `odometry_extrapolation`; source messages
beyond the same tolerance raise `DatasetError` because the map premise is broken
globally. The online timeline artifact (`calibrex.online_timeline/v0.3`) records
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
calibrex compile examples/public_datasets/kitti_raw_2011_09_26_drive_0005/config.yaml
calibrex inspect data/public/nuscenes --type nuscenes
```

KITTI raw data and nuScenes require their official download flows and terms.
After downloading, point the dataset manifest paths at the local extracted
dataset.

KITTI raw camera-LiDAR diagnostic overlay evidence demo:

```bash
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
as the reference/output transform, renders `evidence.json`, `assessment.json`,
`protocol.json`, `policy.json`, and `transforms.json`, and verifies
`bundle.json`. It reports the `lidar_camera_projection_*`,
`lidar_camera_edge_alignment_score`, and `lidar_camera_perturbation_*`
diagnostic overlay metrics described below, plus `koide_lidar_camera_*`
adapter readiness metrics — the demo does not execute the external Koide-style
adapter, so those metrics reflect availability, not a computed result. As with
the rest of this document, treat these camera-LiDAR numbers as diagnostic
overlay evidence on the LiDAR candidate extrinsic, not a standalone camera
calibration.
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
calibrex kitti import-calib /path/to/2011_09_26 --output /tmp/kitti_transforms.yaml
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
