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

## Public real-data benchmark

Five `AX=YB` methods were recomputed on the
[ETHZ ASL real robot-arm dataset](https://projects.asl.ethz.ch/datasets/hand-eye-calibration-2017/)
with one shared 80/20 split.

| Method | Rotation holdout RMSE ↓ | Translation holdout RMSE ↓ |
|---|---:|---:|
| Shah | **0.585795°** | 10.062827 mm |
| Li–Wang–Wu | 0.587280° | 17.457974 mm |
| Dornaika–Horaud | 0.585795° | 10.062824 mm |
| Zhuang–Roth–Sudhakar | 0.590747° | 10.131954 mm |
| **Calibrex nonlinear refinement** | 0.587210° | **10.039379 mm** |

Calibrex's nonlinear refinement is best on translation in this benchmark;
Shah is marginally best on rotation. These are held-out closure errors, not
ground-truth extrinsic errors. Read the [benchmark protocol and
provenance](benchmarks/ethz_robot_world_hand_eye.md).

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
