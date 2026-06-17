# Changelog

## 0.1.0-alpha

- Project bootstrap.
- Typed config and result models.
- CLI skeleton with `doctor`, `schema`, `init`, `inspect`, `calibrate`, `evaluate`, `visualize`, and `export`.
- Frame graph and SE3 utilities.
- Dataset manifest schema, filesystem dataset adapter, timestamp normalization, and MCAP adapter boundary.
- HTML report and PASS/WARN/FAIL quality aggregation.
- Metric registry, deterministic holdout splitting, and threshold-based metric grading profiles.
- RGB-D Open3D SLAC adapter boundary and example config.
- Backend-neutral graph problem compiler and `calibrex compile`.
- Public dataset catalog, TUM RGB-D config, KITTI raw config, and direct TUM downloader helper.
- KITTI raw loader and fixed-mounted LiDAR prior factor.
- KITTI `calib_velo_to_cam.txt` importer for fixed-LiDAR initial transforms.
- Autonomous-driving starter profile with Radar and Autoware export surface.
