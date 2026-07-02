# Changelog

## 0.1.0-alpha.1

- Project bootstrap.
- Typed config and result models.
- CLI skeleton with `doctor`, `schema`, `validate`, `verify`, `assess`, `evidence`, `init`, `inspect`, `calibrate`, `evaluate`, `render`, `visualize`, `compare`, `report`, and `export`.
- Frame graph and SE3 utilities.
- Dataset manifest schema, filesystem dataset adapter, timestamp normalization, and MCAP adapter boundary.
- HTML report, report sidecars, and PASS/WARN/FAIL quality aggregation.
- Metric registry, deterministic holdout splitting, and threshold-based metric grading profiles.
- RGB-D Open3D SLAC adapter boundary and example config.
- Backend-neutral graph problem compiler and `calibrex compile`.
- Public dataset catalog, TUM RGB-D config, KITTI raw config, and direct TUM downloader helper.
- KITTI raw loader and fixed-mounted LiDAR prior factor.
- KITTI `calib_velo_to_cam.txt` importer for fixed-LiDAR initial transforms.
- Autonomous-driving starter profile with Radar and Autoware export surface.
- Public Livox Horizon-Horizon PCD evidence demo for rigidly mounted solid-state LiDAR pairs.
- LiDAR pair evidence metrics, known-bad perturbation cases, holdout point-to-plane summaries, and evidence protocol metadata.
- Machine-readable `summary.json`, `metrics.json`, `observability.json`, `degeneracy.json`, and `evidence.json` report sidecars.
- Machine-readable `comparison.json` artifacts with metric, transform, evidence summary, and evidence protocol compatibility comparisons.
- Static JSON schemas for config, result, comparison, dataset manifest, and report sidecars.
- Bulk schema generation via `calibrex schema all --output-dir schemas`, with schema drift tests and CI smoke coverage.
- 3D rig viewer artifact for reference, candidate, and estimated extrinsic comparison.
- Falsification assessment framework: `calibrex assess` applying a policy to evidence, policy artifacts, `--enforce` exit codes, per-DoF known-bad challenge summaries, a fixed support denominator, mandatory challenge verification, and assessment recomputation verification inside evidence bundles.
- Evidence bundle verification: `calibrex verify`, an `evidence-bundle-verification` schema, a `--require-raw-recomputed` raw recomputation gate, a verified-raw-inputs requirement, and structured verification claims.
- `calibrex render` and `calibrex evidence` commands for producing report and evidence artifacts from an existing result without recomputing metrics.
- Protocol and policy JSON schemas, and `compare --enforce-compatible` protocol compatibility enforcement.
- README GIF gallery generation (`generate_calibration_evidence_gif.py --readme-gallery`) with a provenance manifest and schema-validated visual modes.
- Local release smoke helper, wheel smoke coverage in CI, and a draft GitHub release workflow.
- Configurable public-dataset frame sampling: `DatasetConfig.sample_limit` and `calibrex inspect --sample-limit`.
- N-way `calibrex report-compare` command with labeled results, protocol-compatibility gating, metric-family rankings, and a `report_comparison` schema (16 schemas total).
- Pure-Python rosbag1 (v2.0) reader with PointCloud2 decoding (`none`/`bz2`, optional `lz4` extra), `calibrex inspect --type rosbag1`, and the TIERS LidarsCali example config.
- Native LiDAR point-to-plane solver backend (`solver.backend: native_lidar_point_to_plane`) wired into `calibrex calibrate`, with real rank/condition-number/weak-DoF observability replacing the `uncomputed_alpha_backend` stub and point-to-plane metrics in results.
