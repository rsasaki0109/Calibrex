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

### Multi-window replication and transfer

The same frontend and unchanged gates are also run at paired-frame start indices
60, 180, and 300. All three independent problems converge. Their depth
scale/bias estimates are respectively `(0.99924, -0.01160 m)`,
`(0.99839, -0.07554 m)`, and `(1.02352, -0.03293 m)`. Their identity-mounting
translation errors are 0.0129 m, 0.1120 m, and 0.0522 m. Only one of three
windows passes all four unchanged depth-scale, depth-bias, translation, and
rotation reference gates. The scale range is 2.513 percent and the bias range
is 0.06394 m. Thus the large primary-window bias is not a stable sequence-wide
constant; temporal map/frontend systematics are materially involved.

Each source window's shared extrinsic and depth correction is transferred onto
each other target window's held-out factors while retaining the target's
optimized pose blocks. This deliberately tests shared-parameter transfer, not
independent pose transfer. Only one of six ordered transfers stays within the
declared 0.005 m non-degradation margin. The worst holdout RMSE increase is
0.01514 m. The cross-window transfer result therefore FAILs and prevents a
locally low residual from being presented as a persistent calibration.

The result records depth-index and trajectory hashes, per-window hashes for all
selected PNGs, timestamp association deltas, disjoint frame IDs, map support,
intrinsics, depth convention, options, iterations, data-only observability,
split groups, every known-bad probe, and all six transfer scores.

### Trilinear depth-lattice foundation

Zhou and Koltun define a static calibration mapping `C: P3 -> P3`, shared by
all frames. Equation (3) samples it on a regular 3D lattice and evaluates a
point with trilinear weights, while Equation (4) supplies a shape-preserving
elastic regularizer. Their model is a full three-vector displacement field.

Calibrex now provides a deliberately constrained ablation factor:
`z_corrected = z_nominal + b + sum_l gamma_l(p) delta_z_l`, with the lattice
offsets constrained to zero mean. It uses the same
bounded regular-lattice/trilinear interpolation structure but estimates one
ray-depth displacement per control point, not the paper's full XYZ
displacement. A separate train-only first-order neighbor factor penalizes
scalar control differences; it is not claimed to reproduce the paper's local
SE(3) elasticity term. This separation keeps the approximation explicit and
lets later public-data evaluation compare no regularization, scalar
smoothness, and a future full-vector elastic lattice without changing factor
semantics.

On a synthetic 2 by 2 by 2 lattice, the backend-neutral optimizer recovers the
constant bias and all eight nonuniform zero-mean control offsets within
`1e-7 m`, obtains rank 9/9, yields held-out RMSE below `1e-8 m`, and detects all
18 signed bias/control perturbations.

The public ablation uses a fixed 2 by 2 by 2 sensor-coordinate lattice spanning
`[-3.1, 3.1] x [-2.4, 2.4] x [0.2, 5.0] m`, a 0.05 m neighbor-difference sigma,
and an explicit `1e-4 m` zero-mean constraint. A separate scalar block carries
the constant bias, so the lattice cannot reproduce that mode. All three
augmented systems converge at rank 63/63. However, only the start-60 window
improves scalar-model holdout RMSE; start 180 and 300 worsen it by 0.000018 m
and 0.000505 m.

After centering, the maximum fitted spatial contrast over all 24 controls is
only 0.000158 m and therefore PASS. In contrast, the separated constant biases
are -0.01262 m, -0.07733 m, and -0.01006 m, a 0.06727 m temporal range that
FAILs. This isolates the earlier apparent lattice deformation as an unstable
constant mode rather than evidence for a repeatable coarse spatial pattern.

The weakest window detects 15 of 30 signed extrinsic/bias/lattice probes. Only
one of six ordered spatial transfers remains within the unchanged 0.005 m
margin, and the worst transfer increases holdout RMSE by 0.01511 m. Thus the
extra spatial degrees of freedom do not resolve temporal instability; the
spatial ablation remains honestly FAIL.

This is independently trajectory-supported joint refinement, not
trajectory-from-scratch SLAM: the 56-dimensional rank includes measured-pose
priors and is therefore reported as augmented. Separate rank-6 and rank-8
metrics diagnose data-only shared extrinsic and extrinsic/depth geometry. The
next extension evaluates a rematching frontend, since constant/zero-mean
separation shows the remaining failure is dominated by the window-dependent
offset and fixed map/correspondence construction rather than coarse spatial
distortion.
