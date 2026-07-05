# slac Docs

slac is a universal sensor calibration framework for robotics and
autonomous-driving systems. v0.3 extends the v0.2 moving-platform stack (ADR
0005) with trajectory evidence, anchored temporal estimation, and LiDAR-IMU
rotation evidence (ADR 0007): schema-validated `trajectory.json` with
ground-truth-free gates, native-deskew KISS-ICP odometry at bag conversion,
`time_offset_anchor` dual-mode temporal evaluation with joint separability
reporting, and `lidar_imu` rotation-rate consistency on TIERS Indoor02. Full
radar extrinsic calibration remains experimental.

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
- `docs/adr/0006-rename-to-slac.md`
- `docs/adr/0007-v0-3-slac-direction.md`

Regenerate committed schemas with `slac schema all --output-dir schemas`.
