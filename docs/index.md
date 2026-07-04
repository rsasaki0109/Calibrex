# slac Docs

slac is a universal sensor calibration framework for robotics and
autonomous-driving systems. v0.2 extends the v0.1 LiDAR-main evaluation protocol
(ADR 0004) with moving-platform online calibration (rosbag2, odometry motion
compensation, per-point deskew), camera-LiDAR evaluated evidence rows, and
temporal time-offset probes (ADR 0005). IMU and full radar candidate evaluation
remain roadmap items.

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
