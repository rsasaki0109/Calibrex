# ADR 0004: LiDAR-Main v0.1 Evaluation Protocol

## Status

Accepted for v0.1 alpha planning.

## Context

Calibrex should not compete with targetless LiDAR-camera SOTA methods first.
For v0.1, the stronger OSS value is a reproducible evaluation and failure
diagnosis layer for fixed-mounted vehicle LiDAR calibration candidates.

The alpha statement is:

> On public driving datasets, Calibrex compares LiDAR, camera, IMU, and radar
> extrinsic candidates through one CLI, one schema, one holdout protocol, and
> one report, while explaining which DoF are reliable and which are degenerate.

## Decision

The next development phase prioritizes evaluation over new calibration
algorithms.

P0 work:

- LiDAR point-to-plane train/holdout consistency.
- Configurable public-dataset frame splits and frame counts.
- Perturbation sensitivity for roll, pitch, yaw, x, y, and z candidates.
- Observability and degeneracy reporting from factor information.
- Clear separation of reference, candidate, external baseline, and optimized
  extrinsics in the result schema.

P1 work:

- A small native `LidarRigPointToPlaneFactor` using dataset ego pose as fixed
  or weak prior.
- nuScenes mini or selected-scene metadata loader for LiDAR, camera, radar,
  ego pose, and calibrated sensor tables.
- Solid-state multi-LiDAR evidence on public Livox data, starting with the
  TIERS `LidarsCali` sequence containing Livox Horizon and Livox Avia streams.
- Common report comparison for dataset calibration, perturbed candidates,
  Koide-style adapter output, and native Calibrex output.

A2D2 remains useful as a public multi-LiDAR smoke test, but it uses Velodyne
VLP-16 spinning LiDARs and must not be presented as a solid-state dataset.

## Consequences

Calibrex v0.1 will not claim SOTA absolute calibration accuracy. It will claim
repeatable calibration evaluation, candidate ranking, holdout consistency, and
failure diagnosis on public driving datasets.

The core remains ROS-independent and avoids GPL code. Dataset-specific frame
names stay in adapters. Deep or semantic factors stay optional until metric and
schema behavior are stable.

## Non-Goals

- Reimplement Koide-style targetless LiDAR-camera calibration in core.
- Build a full SLAM engine.
- Make radar native calibration part of v0.1.
- Treat public dataset calibration as absolute ground truth.
- Use visual overlay as a substitute for holdout metrics.
