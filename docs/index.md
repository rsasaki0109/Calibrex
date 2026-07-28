# Calibrex Docs

Calibrex is a ROS-independent calibration evidence framework for robotics and
autonomous-driving systems. It turns candidate extrinsics, time offsets, and
trajectories into reproducible **PASS / WARN / FAIL** evidence.

Most calibration tools answer “what transform did the optimizer return?”
Calibrex also asks:

- Does it generalize to held-out measurements?
- Can the available data detect a deliberately wrong calibration?
- Which degrees of freedom are weak or unobservable?
- Can the result and every generated report be traced to their inputs?

## Start here

1. Run the [five-minute quickstart](https://github.com/rsasaki0109/Calibrex#five-minute-quickstart).
2. Choose a path from [calibration methods](concepts/calibration_methods.md).
3. Reproduce an evaluation from [public datasets](tutorials/public_datasets.md).
4. Review [frame conventions](concepts/frame_conventions.md) before integrating
   transforms.

## Integration paths

- [Open3D SLAC adapter](tutorials/open3d_slac.md)
- [Targetless LiDAR-camera adapter](tutorials/lidar_camera_adapter.md)
- [Problem builder](concepts/problem_builder.md)
- [Schema reference](reference/schemas.md)
- [License boundaries](concepts/license_boundaries.md)

## Current status

Calibrex is alpha research software. LiDAR-LiDAR and hand-eye paths have native
solvers, evaluation, and public examples. Camera-LiDAR, radar, RGB-D, and some
online paths remain experimental. Failed controls are reported rather than
hidden; see the [changelog](changelog.md) and
[development roadmap](development_roadmap.md) for current limitations.

Regenerate committed schemas with `calibrex schema all --output-dir schemas`.
