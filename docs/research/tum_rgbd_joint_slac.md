# TUM RGB-D multi-capture joint SLAC research note

## Primary sources and scope

This series extends the backend-neutral joint graph from a synchronized LiDAR
pair to a real multi-capture RGB-D sequence with externally measured camera
poses. The target dataset is TUM RGB-D `freiburg1_xyz`, introduced by Sturm et
al., *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012. The
joint-estimation structure follows Zhou and Koltun, *Simultaneous Localization
and Calibration: Self-Calibration of Consumer Depth Cameras*, CVPR 2014.

Primary references:

- TUM RGB-D benchmark and paper:
  <https://cvg.cit.tum.de/data/datasets/rgbd-dataset>
- Official TUM file and projection conventions:
  <https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats>
- Zhou and Koltun CVPR paper:
  <https://openaccess.thecvf.com/content_cvpr_2014/html/Zhou_Simultaneous_Localization_and_2014_CVPR_paper.html>

No TUM benchmark tools, Open3D implementation code, or GPL graph code is copied
into `src/calibrex`.

## Native frontend contract

The official TUM format stores depth as non-interlaced 640 by 480, 16-bit
grayscale PNG. Zero means missing depth and integer values are divided by 5000
to obtain meters. The provided depth maps are already registered to the color
camera, and ground truth is the camera pose in a fixed world frame.

`read_tum_depth_png` implements the required PNG scanline filters with Python's
standard library and returns raw integer depth without a ROS, OpenCV, Pillow,
or Open3D dependency. `sample_tum_depth_points` performs deterministic pinhole
back-projection with explicit intrinsics, scale, depth limits, and point budget.
`associate_depth_groundtruth` performs a bounded nearest-time association and
retains the absolute timestamp delta and typed `T_world_camera` pose.

Synthetic tests cover exact 16-bit decoding, zero-depth rejection, metric
scaling, pinhole coordinates, nearest-pose association, and SE(3) conversion.
The checked local public sample is also decoded directly before factor
integration.

## Joint factor integration

`NativeTUMJointSlacSolver` reserves three disjoint map frames, builds
world-frame voxel planes, and creates one pose block for each of eight query
frames plus a shared camera mounting block and a shared two-value depth block.
The depth convention is `z_corrected = exp(log_scale) * z_nominal + bias_m`.
The disjoint map depths remain the fixed metric reference. Ground-truth pose
priors are train-only. Query frames, not pixels, define train/holdout groups. Numeric
linearization uses declared factor ownership, so perturbing one pose block only
re-evaluates that frame's factors while the shared extrinsic still touches all
frames.

Synthetic multi-capture tests jointly recover trajectory-supported extrinsic,
depth log-scale, and bias truth within `2e-5`, reach rank 32/32, and detect all
16 signed shared-parameter probes.

The public fr1/xyz depth-enabled run converges with 0.0271 m train and 0.0277 m
held-out point-to-plane RMSE. The augmented 56-dimensional joint system is full
rank. With optimized poses fixed, train geometry gives extrinsic rank 6/6 and
combined extrinsic/depth rank 8/8, with local condition numbers 6.04 and 16.99.
The recovered depth multiplier is 0.99839, only 0.161 percent from the official
pre-scaled reference and therefore PASS.

The additive bias is -0.0755 m, outside the unchanged 0.03 m reference gate.
It co-varies with a 0.112 m / 1.25 degree mounting correction, also outside the
translation PASS/WARN boundary. Thirteen of sixteen signed shared-extrinsic and
depth perturbations worsen held-out RMSE (0.8125). The public result is therefore
honestly FAIL despite its lower geometric residual and full local rank. This is
evidence of model/reference tension and bias-versus-z-translation correlation,
not a reason to relax gates.

The result records depth-index and trajectory hashes, hashes for all 11 selected
PNGs, timestamp association deltas, disjoint frame IDs, map support, intrinsics,
depth convention, options, iterations, both data-only observability evaluations,
split groups, and every known-bad probe.

This is independently trajectory-supported joint refinement, not
trajectory-from-scratch SLAM: the 56-dimensional rank includes measured-pose
priors and is therefore reported as augmented. Separate rank-6 and rank-8
metrics diagnose data-only shared extrinsic and extrinsic/depth geometry. The
next extension is multi-window replication and a spatial depth-correction basis,
with an explicit ablation to distinguish persistent bias from map/frontend
systematics.
