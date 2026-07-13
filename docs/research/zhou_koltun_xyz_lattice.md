# Zhou--Koltun full-XYZ calibration lattice research note

## Primary source and scope

This implementation follows Qian-Yi Zhou and Vladlen Koltun,
*Simultaneous Localization and Calibration: Self-Calibration of Consumer Depth
Cameras*, CVPR 2014:

- paper and project page:
  <https://openaccess.thecvf.com/content_cvpr_2014/html/Zhou_Simultaneous_Localization_and_2014_CVPR_paper.html>
- archival PDF:
  <https://openaccess.thecvf.com/content_cvpr_2014/papers/Zhou_Simultaneous_Localization_and_2014_CVPR_paper.pdf>

No implementation code from the authors, Open3D, or a GPL graph backend is
copied into `src/calibrex`. The implementation is ROS-independent and uses the
typed backend-neutral factor contract.

## Paper-to-code contract

The paper estimates one static calibration function `C: P3 -> P3` shared by
all range images. Equation (2) aligns a correspondence with
`(Ti C(p) - Tj C(q))^T n`. Equation (3) represents `C` by calibrated 3D
positions at regular lattice nodes and trilinearly interpolates them. Equation
(4) prevents collapse with a shape-preserving neighbor residual around a local
rigid linearization at each node.

Calibrex stores a flattened three-vector displacement `delta_l` at each
identity-lattice node and evaluates
`C(p) = p + sum_l gamma_l(p) delta_l`. A regular trilinear lattice reproduces
`p = sum_l gamma_l(p) v_l`, so this is algebraically equivalent to the paper's
`sum_l gamma_l(p) C(v_l)` while retaining zero as the identity initialization.

`make_joint_trilinear_xyz_pair_factor` implements both calibrated sides of
Equation (2), with two pose blocks and one shared field block. The fixed-map
`make_joint_trilinear_xyz_point_to_plane_factor` is a separately named
specialization for frontends such as the current TUM evaluation; it does not
claim that a fixed target is the paper's pairwise objective.

`make_joint_xyz_lattice_shape_factor` evaluates directed lattice-neighbor
residuals
`C(u) - (C(v) + R_v (u-v))`, divided by a declared metric sigma. `R_v` is a
frozen local rotation supplied by the frontend for the current Gauss--Newton
round, matching the paper's policy of freezing local linearizations inside a
round and updating them between rounds. The factor is always train-only. It
does not silently estimate rotations from holdout data.

`estimate_xyz_lattice_local_rotations` performs an independent proper-rotation
Procrustes fit at every node from its current calibrated neighbor edges. A
frontend can therefore rebuild Equation (4) between optimizer rounds without
embedding mutable state in a residual callback. A synthetic rigid field
recovers the same 0.12 rad rotation at all eight nodes to numerical precision.

The full field also has six rigid modes that can be exchanged with camera
poses or a separately estimated extrinsic. Calibrex does not hide this behind
the elastic term, which correctly assigns zero cost to rigid deformation.
`make_joint_xyz_lattice_rigid_gauge_factor` explicitly constrains mean field
translation and the best infinitesimal rotation moment with separately
declared metric/radian sigmas. Tests isolate all three translation and all
three rotation modes. This factor is train-only, and data-only field rank must
be reported separately from gauge-augmented rank.

## Synthetic evidence

Tests cover regular-node ordering, dense trilinear weights, the fixed-map XYZ
factor, the two-sided pair factor, shape term, rigid gauge, and local-rotation
update. A global XYZ translation has exactly zero shape residual over all 24
directed edges of a 2 by 2 by 2 lattice, while a single-node distortion is
detected.

The two-sided optimizer recovery test creates five complete synthetic capture
groups whose target poses make the paper residual zero only at a declared
24-dimensional field truth. A seeded group split holds out one whole capture
group. The native optimizer recovers every control within `1e-8 m`, reaches
rank 24/24, has holdout RMSE below `1e-9 m`, and detects all 48 signed field
probes. This is distinct from the fixed-map recovery test below.

The optimizer recovery test estimates all 24 XYZ control dimensions from five
frame-level observation groups. The seeded split leaves one whole group for
holdout. It recovers every control displacement within `1e-8 m`, obtains data
rank 24/24, achieves held-out RMSE below `1e-9 m`, and detects all 48 signed
known-bad control probes. Fixed pose and extrinsic blocks are explicit gauges
and do not contribute to the reported rank.

## Public TUM evidence and current boundary

The public `freiburg1_xyz` adapter now performs one train-fitted local-rotation
update in each of three temporal windows. All six inner solves converge. The
field-only data rank is 24/24, while extrinsic plus field is rank 24/30 because
of the expected rigid exchange modes. The gauge-augmented full graph is rank
78/78 but is never substituted for either data-only diagnostic.

Only one of three windows improves on the scalar ray-depth holdout. The worst
full-XYZ-minus-scalar change is +0.004047 m and FAILs. Maximum displacement is
91.3 micrometers, maximum updated local rotation is 0.000347 degrees, and the
weakest frozen-factor probe detection is 11/48. The public result therefore
does not claim useful lateral calibration despite full field-only rank.

The fixed-map branch still uses disjoint map planes and remains explicitly
labelled as a specialization of Equation (2). Reassociation-aware probes lower
the weakest detection fraction from 0.533 frozen to 0.433 with no support
collapse. Ordered extrinsic/field transfer passes the unchanged 0.005 m margin
in only 2/6 directions; the worst holdout increase is 0.011746 m.

A public two-sided branch now builds seven adjacent query-query edges per
window. Source points are transformed by the initial `T_target_source` and
matched to target-local voxel planes. The real target depth sample nearest each
plane centroid supplies `q`, so both endpoints pass through the same `C` and
the target has a stable point ID. Target normals are rotated into world
coordinates.

The split holds out complete pair edges: five train and two holdout edges per
window. It is factor-disjoint but honestly not frame-disjoint because adjacent
edges share endpoint frames. With 60 deterministically distributed matches per
edge, the three windows retain 414, 405, and 403 correspondences. Both frozen
local-rotation rounds converge. Field-only rank is 24/24; data-only joint rank
is 57/66 after fixing the first-pose world gauge, so it remains WARN. Train-only
pose, shape, and rigid-field priors raise the augmented systems to 66/66 but do
not replace that diagnostic.

Only one of three pair holdouts improves over identity initialization; the
worst increase is `6.39e-6 m`, hence FAIL under the unchanged zero-degradation
comparison. Frozen signed pose/field probes detect 40/132, 42/132, and 40/132;
the weakest 0.3030 fraction is WARN. In contrast, all six ordered field-only
transfers stay inside the unchanged 0.005 m margin; the worst increase is only
`5.26e-8 m`. This
establishes the missing public two-sided execution path while refusing to turn
augmented rank or near-zero transfer deltas into a claim of strong calibration
observability. Every fixed association retains its source and target sample
indexes, pair group, stable target point ID, and initial centroid distance
alongside the raw frame hashes and deterministic sampling policy.

The two-sided branch now also rebuilds each target-local plane map from
calibrated `C(q)` points. A source query is evaluated with the current source
pose and field, transformed through the inverse current target pose, and
associated in calibrated target coordinates. The first reassociation round
passes the unchanged train gates in all windows: minimum pair Jaccard is
0.99322 and retention is 1.0. Holdout pair Jaccard is 1.0 and is never used for
stopping. The terminal rematched holdout differs from the fixed score by at
most `7.76e-8 m`.

Reassociation-aware signed probes use the original held-out source population;
unmatched queries pay the declared 0.15 m penalty. All 396 probes are valid,
baseline retention is 1.0, and no probe causes support collapse. Detection is
55/132, 64/132, and 43/132, so the weakest 0.3258 remains WARN. Perturbed pair
Jaccard falls to 0.6571. Stable fitted assignments therefore do not imply that
known-bad perturbations preserve the same objective.
