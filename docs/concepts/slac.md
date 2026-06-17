# SLAC

In Calibrex, SLAC means simultaneous localization and calibration as an
architecture principle rather than one fixed algorithm.

The core representation must be able to express trajectory, extrinsics, timing,
intrinsics, map variables, correspondences, priors, and evaluation diagnostics.

Calibrex should not hard-code one SLAC algorithm into the core. Instead, it
compiles a calibration problem from config into backend-neutral variables,
factors, priors, gauges, and evaluation requirements. Sensor-specific methods
then enter as plugins, optional solver adapters, or baseline subprocesses.

## Research Anchors

Autonomous-driving and fixed-rig LiDAR calibration work is guided by these papers:

- [SceneCalib](https://arxiv.org/abs/2304.05530): targetless multi-camera and
  LiDAR self-calibration for autonomous driving. It motivates joint camera
  intrinsics, camera-camera extrinsics, and camera-LiDAR extrinsics.
- [SST-Calib](https://arxiv.org/abs/2207.03704): simultaneous spatial-temporal
  LiDAR-camera calibration. It makes time offset a first-class calibration
  variable rather than metadata.
- [Decentralized multi-LiDAR SLAC](https://arxiv.org/abs/2007.01483): joint
  vehicle pose, velocity, map, and multi-LiDAR extrinsic estimation. It is the
  cleanest reference for LiDAR-centered SLAC.
- [M-LOAM](https://arxiv.org/abs/2010.14294): online multi-LiDAR odometry,
  mapping, and extrinsic calibration. It informs a future `calibrex online`
  design, but not the first offline MVP.
- [Koide et al. ICRA 2023](https://arxiv.org/abs/2302.05094): general,
  single-shot, targetless, automatic LiDAR-camera calibration. Calibrex should
  support it as an adapter or baseline before trying to duplicate it natively.

See [LiDAR SLAC](lidar_slac.md) for the concrete design consequences.
