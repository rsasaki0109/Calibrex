# Changelog

## 0.1.0-alpha.1

- Project bootstrap.
- Typed config and result models.
- CLI skeleton with `doctor`, `schema`, `validate`, `init`, `inspect`, `calibrate`, `evaluate`, `visualize`, `compare`, `report`, and `export`.
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
