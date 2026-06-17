# LiDAR SLAC

Calibrex now treats fixed-mounted LiDAR calibration as the primary autonomous
driving path. Public driving logs such as KITTI and nuScenes are preferred over
synthetic-only examples for this path.

## Target Scope

The first LiDAR-centered target is an offline fixed-rig problem:

- one or more fixed-mounted LiDARs on a vehicle body
- optional cameras with known or estimated intrinsics
- optional IMU/OXTS/GNSS motion priors
- spatial extrinsics and time offsets as explicit variables
- LiDAR map quality, camera-LiDAR overlay quality, and holdout residuals as
  required outputs

Online calibration, fleet drift monitoring, and multi-LiDAR failover are later
extensions. They should reuse the same config, result schema, factors, and
quality gates.

## Paper-to-Design Mapping

| Paper | What Calibrex Should Learn |
| --- | --- |
| [SceneCalib](https://arxiv.org/abs/2304.05530) | Targetless multi-camera + LiDAR calibration needs joint camera intrinsics, camera-camera constraints, and camera-LiDAR consistency in one graph. |
| [SST-Calib](https://arxiv.org/abs/2207.03704) | `dt_camera` and `dt_lidar` must be optimized and reported with uncertainty, not treated as fixed preprocessing assumptions. |
| [Decentralized multi-LiDAR SLAC](https://arxiv.org/abs/2007.01483) | LiDAR calibration should estimate pose, velocity, mapping state, and LiDAR extrinsics together when enough motion exists. |
| [M-LOAM](https://arxiv.org/abs/2010.14294) | Online extrinsic refinement and convergence detection belong in a future `calibrex online` mode. |
| [Koide et al. ICRA 2023](https://arxiv.org/abs/2302.05094) | Single-shot targetless LiDAR-camera calibration should be wrapped as an adapter/baseline and evaluated through Calibrex result schemas. |

## Required Variables

LiDAR-first SLAC configs should be able to compile these variables:

- `T_base_lidarN` fixed-rig extrinsics
- `T_base_cameraN` and `T_lidar_camera` derived or explicit extrinsics
- `dt_lidarN` and `dt_cameraN`
- vehicle pose and velocity trajectory
- camera intrinsics when configured as estimable
- LiDAR map primitives, initially surfels or point-to-plane neighborhoods
- IMU bias and gravity alignment when IMU/OXTS data is present

## Required Factors

The problem builder must keep these as backend-neutral factor descriptors:

- `fixed_lidar_mount_prior`
- `lidar_point_to_surfel`
- `lidar_motion_consistency`
- `lidar_camera_mutual_information`
- `lidar_camera_semantic_alignment`
- `camera_reprojection`
- `cross_camera_consistency`
- `time_offset_motion_consistency`
- `imu_preintegration`

Some of these can start as descriptors before native residual implementations
exist. The important part is that configs, compiled problems, result provenance,
and evaluation reports already name the intended factor family.

## Adapter Boundary

Calibrex core should not copy external calibration implementations. Strong
methods such as Koide's LiDAR-camera toolbox should enter through an optional
adapter or subprocess wrapper that converts inputs and outputs into Calibrex
schemas.

This keeps the core small, ROS-independent, and license-safe while still making
Calibrex useful on real data early.

The first adapter boundary is `koide_lidar_camera`. It records external command
availability, camera/LiDAR input readiness, optional precomputed transform
imports, and the license boundary in result provenance.

## MVP Order

1. KITTI raw and nuScenes manifests for public autonomous-driving logs.
2. KITTI fixed-LiDAR calibration import as initial transforms and priors.
3. Velodyne point cloud inspection, bounds, intensity stats, and frame coverage.
4. LiDAR point-to-plane / surfel metric scaffolding with holdout reporting.
5. Koide-style targetless LiDAR-camera adapter boundary.
6. SST-Calib-style time-offset variables and temporal quality checks.
7. Multi-LiDAR problem descriptors for decentralized SLAC and M-LOAM-style
   online refinement.

## Quality Gates

A LiDAR calibration result is incomplete without diagnostics. Minimum reports:

- LiDAR frame count and point count coverage
- sampled spatial coverage from point cloud XYZ bounds
- local planarity, roughness, and map sharpness proxies from voxelized point
  neighborhoods
- point-to-plane RMSE proxy with train/holdout split
- map sharpness or local planarity score
- camera-LiDAR overlay score when cameras are present
- camera-LiDAR overlay readiness and proxy score from stream availability,
  transform pairs, timestamp alignment, and LiDAR point coverage until native
  image projection scoring is implemented
- estimated time offset and uncertainty when timestamps are optimized
- observability warnings for weak yaw, weak vertical excitation, low overlap, or
  insufficient vehicle motion
- LiDAR degeneracy warnings for missing point clouds, too few planar
  neighborhoods, weak vertical structure, weak local planarity, low map
  sharpness, and holdout point-to-plane degradation
- motion excitation warnings from KITTI OXTS duration, speed range, and yaw
  excitation
- timestamp alignment checks for Camera-LiDAR and LiDAR-OXTS streams, following
  the SST-Calib principle that time offset must be explicit
- recommendation text for recollection, such as longer fixed-LiDAR logs,
  vertical structures, static planar surfaces, repeated static holdout passes,
  or richer cross-modal overlap
