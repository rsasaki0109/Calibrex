# Changelog

## Unreleased

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
