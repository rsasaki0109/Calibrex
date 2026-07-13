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

## Synthetic evidence

Tests cover regular-node ordering, dense trilinear weights, the fixed-map XYZ
factor, the two-sided pair factor, and the shape term. A global XYZ translation
has exactly zero shape residual over all 24 directed edges of a 2 by 2 by 2
lattice, while a single-node distortion is detected.

The optimizer recovery test estimates all 24 XYZ control dimensions from five
frame-level observation groups. The seeded split leaves one whole group for
holdout. It recovers every control displacement within `1e-8 m`, obtains data
rank 24/24, achieves held-out RMSE below `1e-9 m`, and detects all 48 signed
known-bad control probes. Fixed pose and extrinsic blocks are explicit gauges
and do not contribute to the reported rank.

## Current boundary

This change establishes the license-safe full-XYZ factor and regularizer
foundation. It is not yet a public-data claim. The next adapter revision must
update local rotations between optimization rounds, emit the field and its
regularization policy in result provenance, compare scalar-depth and XYZ
holdout evidence on public TUM windows, and preserve honest failure when
temporal transfer or reassociation gates remain unstable.
