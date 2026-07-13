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

### Full XYZ calibration lattice

The next ablation implements the paper's three-vector calibration field rather
than a ray-depth-only approximation. Each of eight regular controls carries an
XYZ displacement, for 24 field dimensions. A fixed-map point-to-plane factor
evaluates `C(p) = p + sum_l gamma_l(p) delta_l`; the generic graph also exposes
the paper's two-sided `Ti C(p) - Tj C(q)` factor. Directed neighbor residuals
use a local proper rotation frozen inside each optimizer round. After the first
solve, each node's rotation is updated by a train-fitted Procrustes estimate and
the graph is warm-started for a second round.

The elastic term intentionally assigns zero cost to a rigid field. Because
those six modes can exchange with the camera extrinsic, a separate train-only
gauge constrains mean field translation and infinitesimal rotation moment. The
augmented pose/extrinsic/field systems are then rank 78/78 in every window.
Data-only diagnostics remain separate: with pose and extrinsic fixed the field
is rank 24/24, while the combined extrinsic/field system is rank 24/30 and
names both blocks as weak. The latter is WARN, not PASS; the six missing modes
are the predicted rigid exchange ambiguity rather than evidence created by the
gauge.

Both frozen-local-rotation solves converge in all three windows. The fitted
field is extremely small: the largest control component is 91.3 micrometers,
and the largest local rotation used in the second round is 0.000347 degrees.
Compared with the scalar ray-depth lattice, held-out RMSE changes are
-0.000130 m, +0.004047 m, and +0.000142 m at starts 60, 180, and 300. Only one
of three windows improves and the worst change therefore FAILs.

Final frozen-factor signed control probes detect 20/48, 14/48, and 11/48
perturbations. The minimum 11/48 is WARN. Thus the extra lateral degrees of
freedom are locally rank-complete only after fixing pose/extrinsic, but are
weakly falsifiable and do not provide repeatable public-data improvement.
Every control vector, local rotation, optimizer round, split, probe, augmented
rank, and data-only rank is retained in provenance. The fixed map remains an
explicit specialization and is evaluated separately from the two-sided branch.

The shared extrinsic and all 24 field components are also transferred in all
six ordered directions while retaining each target window's optimized poses
and final held-out data factors. Shape, local-rotation, and rigid-gauge
residuals are excluded from transfer scoring. Only start 180 to 300 and start
300 to 180 remain within the unchanged 0.005 m margin, with increases of
0.000477 m and 0.003023 m. The other four directions increase RMSE by
0.00763--0.01175 m; the worst is start 60 to 180 at +0.011746 m. The
nondegrading fraction is therefore 2/6 and FAIL. The small fitted field and
full field-only rank do not establish a sequence-wide calibration function.

### Two-sided Zhou--Koltun Equation (2)

A separate public graph now evaluates the paper residual
`(T_i C(p) - T_j C(q)) dot n_world` on adjacent query-query frame pairs. For
each directed edge, target depth samples form a target-camera voxel-plane map.
Source samples are transformed with the initial `T_target_source`; accepted
target centroids provide `q`, and target normals are rotated into the world
frame. Both `p` and `q` are calibrated by the same 24-dimensional trilinear XYZ
field. No extrinsic block is inserted into this paper-faithful ablation.

The frontend caps each edge at 60 deterministic correspondences and retains
414, 405, and 403 factors at starts 60, 180, and 300. Complete pair IDs are the
split groups, producing five train and two held-out edges. This is factor- and
edge-disjoint, but adjacent edges can share endpoint frames; provenance states
that limitation instead of calling it frame-disjoint. The first pose correction
is fixed as the world gauge. Ground-truth pose priors, the Equation (4) shape
term, and the rigid-field gauge are train-only.

Both frozen-local-rotation rounds converge in all three windows. With all poses
fixed, train data observes the field at rank 24/24. With non-gauge poses free,
the data-only joint rank is 57/66 and is WARN. Train-only regularization makes
the augmented graph rank 66/66, but that rank is never reported as geometric
observability. Only one of three held-out pair objectives improves from the
identity field/pose initialization; the worst change is `+3.85e-6 m`, so the
unchanged non-degradation comparison FAILs. Signed non-gauge pose and field
probes detect only 40/132, 40/132, and 39/132 perturbations.

All six ordered transfers replace only the source window's field while keeping
the target's optimized pose corrections and held-out pair factors. Every
direction remains within the unchanged 0.005 m margin; the worst increase is
`4.67e-8 m`. This PASS is recorded alongside, not in place of, the weak probe
and joint-rank evidence. Pair counts, frame IDs, exact train/holdout edges,
field controls, two optimizer rounds, local rotations, all probes, both rank
diagnostics, and transfer scores are serialized in provenance. Each fixed
association also records its source sample index, pair group, stable target
voxel ID, and initial centroid distance, allowing reconstruction from the
hashed raw frame and declared deterministic sampler.

### Optimized correspondence rematching

Held-out optimized points are rematched to the same voxel-plane map under the
unchanged 0.15 m gate. Query retention remains above 98 percent in all three
windows, yet exact query/voxel pair Jaccard drops from 0.769 at start 60 to
0.529 and 0.478 at starts 180 and 300. The minimum same-target fraction is
0.653. Rematched RMSE is up to 0.00292 m lower, but this is a
selection-dependent diagnostic rather than independent improvement. The low
pair stability despite high retention demonstrates that the fixed-assignment
linearization is not a stable surrogate for the rematched objective.

Central-difference curvature over shared extrinsic plus constant bias confirms
the mismatch. Fixed-correspondence Hessians are positive definite and rank 7/7
in every window. Rematched Hessians remain full rank but contain 3, 6, and 6
negative eigenvalues. The maximum normalized Frobenius difference between
fixed and rematched Hessians is 10.68. Consequently, augmented rank and a
positive fixed-assignment Hessian cannot be treated as evidence that the
association-aware calibration objective is locally convex.

Five deterministic full-7D searches from the optimized center and signed
extrinsic-z/depth-bias starts all converge, but form two competitive basins.
Their best objectives differ by `3.73e-6` and their representatives are 1.92
declared physical scales apart. The best basin trades roughly +0.03 m of
extrinsic z against -0.03 m of constant depth bias relative to the competing
basin. Multi-start evidence therefore confirms a genuine translation/depth
ambiguity rather than mere failure of one optimizer trajectory.

### Leakage-safe iterative reassociation

The spatial model is now also evaluated inside a backend-neutral outer loop.
At each round, every query point is transformed with the current pose,
extrinsic, constant bias, and trilinear depth offsets, reassociated to the
nearest voxel plane under the unchanged 0.15 m gate, and warm-started through
the same joint optimizer. Stable query IDs and seeded frame-level
train/holdout groups remain invariant. Only train pair Jaccard and train query
retention may stop fitting; holdout correspondence stability is diagnostic.

All inner optimizations converge in both configured rounds for all three
windows. Nevertheless, none reaches the unchanged 0.99 train pair-Jaccard
gate. Terminal train pair Jaccards at starts 60, 180, and 300 are 0.9140,
0.8037, and 0.8469, while minimum train retention remains 0.9987 and passes its
0.95 gate. The outer-loop convergence fraction is therefore honestly 0/3 and
FAIL, rather than relabelling high query retention as stable association.

Final reassociated holdout RMSE changes from the fixed spatial baselines are
+0.000874 m, -0.005127 m, and -0.002893 m. Terminal holdout pair Jaccards are
0.9023, 0.8343, and 0.8516; these values are never used for stopping. The
weakest final frozen-correspondence known-bad detection fraction is 8/15.
This remains a separately named frozen-factor WARN metric.

Each of the same 30 signed extrinsic/bias/lattice probes per window is now also
evaluated after rebuilding correspondences. The original holdout query set is
the fixed population; unmatched queries contribute the declared 0.15 m
residual penalty. Baseline population retention is 1.0000, 0.9891, and 0.9699.
All 90 evaluations are valid and none crosses the unchanged 0.95 support
collapse gate. Nevertheless, only 18/30, 13/30, and 22/30 probes are detected.
The weakest reassociation-aware fraction is therefore 0.4333, below the frozen
minimum of 0.5333, and remains WARN. Perturbed pair Jaccard falls as low as
0.4763, confirming that the two probe objectives are materially different.

Full round histories, factor identities, assignments, split groups, optimizer
results, physical parameter deltas, frozen probes, fixed-population aware
probes, support accounting, and terminal stability are serialized for every
window.

This is independently trajectory-supported joint refinement, not
trajectory-from-scratch SLAM: the 56-dimensional rank includes measured-pose
priors and is therefore reported as augmented. Separate rank-6 and rank-8
metrics diagnose data-only shared extrinsic and extrinsic/depth geometry. The
next extension can evaluate reassociation for the two-sided branch and increase
lattice resolution only if public support remains observable.
