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
