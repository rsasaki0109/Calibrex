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

The adapter still uses fixed disjoint map planes, which is explicitly labelled
as a specialization of Equation (2). Reassociation-aware probes lower
the weakest detection fraction from 0.533 frozen to 0.433 with no support
collapse. Ordered extrinsic/field transfer passes the unchanged 0.005 m margin
in only 2/6 directions; the worst holdout increase is 0.011746 m. A public
two-sided correspondence frontend remains the next missing paper-level
evidence component.
