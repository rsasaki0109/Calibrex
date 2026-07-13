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
frames plus a shared camera mounting block. Ground-truth pose priors are
train-only. Query frames, not pixels, define train/holdout groups. Numeric
linearization uses declared factor ownership, so perturbing one pose block only
re-evaluates that frame's factors while the shared extrinsic still touches all
frames.

The public fr1/xyz run converges with 0.0298 m train and 0.0318 m held-out
point-to-plane RMSE. The augmented 54-dimensional joint system is full rank.
With optimized poses fixed, the train geometry gives shared-extrinsic rank 6
and normalized condition number 14.1. TUM ground truth already describes the
camera frame, so the mounting reference is identity; the recovered shared
transform is 0.0376 m and 0.205 degrees from that reference, inside the declared
0.05 m / 1 degree gates.

Nine of twelve signed shared-extrinsic perturbations worsen held-out RMSE, a
detectable fraction of 0.75. This remains WARN under the unchanged requirement
of 1.0. The result records depth-index and trajectory hashes, hashes for all 11
selected PNGs, timestamp association deltas, disjoint frame IDs, map support,
intrinsics, options, iterations, split groups, and every known-bad probe.

This is independently trajectory-supported joint refinement, not
trajectory-from-scratch SLAM: the 54-dimensional rank includes measured-pose
priors and is therefore reported as augmented. The separate rank-6 metric is
the data-only shared-extrinsic diagnostic. The next extension is a shared depth
scale/bias or spatial depth-correction block, closer to the correction function
estimated in Zhou and Koltun's original SLAC formulation.
