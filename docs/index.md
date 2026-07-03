# Calibrex Docs

Calibrex is a universal sensor calibration framework for robotics and
autonomous-driving systems. The v0.1 alpha is LiDAR-main: LiDAR extrinsic
candidate evaluation (holdout point-to-plane, perturbation challenges,
observability/degeneracy reporting) is the delivered capability, camera-LiDAR
projection metrics are an experimental overlay, and IMU/radar candidate
evaluation are roadmap items (see
`docs/adr/0004-lidar-main-v0-1-evaluation-protocol.md`).

Start with:

- `docs/concepts/slac.md`
- `docs/concepts/lidar_slac.md`
- `docs/concepts/frame_conventions.md`
- `docs/concepts/license_boundaries.md`
- `docs/concepts/problem_builder.md`
- `docs/tutorials/public_datasets.md`
- `docs/tutorials/open3d_slac.md`
- `docs/tutorials/lidar_camera_adapter.md`
- `schemas/assessment.schema.json`
- `schemas/comparison.schema.json`
- `schemas/evidence_bundle.schema.json`
- `schemas/transforms.schema.json`
- `schemas/report_evidence.schema.json`
- `schemas/dataset_manifest.schema.json`
- `docs/adr/0001-core-is-ros-independent.md`
- `docs/adr/0004-lidar-main-v0-1-evaluation-protocol.md`
- `docs/adr/0005-v0-2-motion-camera-temporal-direction.md`

Regenerate committed schemas with `calibrex schema all --output-dir schemas`.
