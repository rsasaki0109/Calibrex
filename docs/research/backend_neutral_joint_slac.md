# Backend-neutral joint SLAC research note

## Scope and primary references

Calibrex now has a numerical graph core that can jointly optimize trajectory,
extrinsic, temporal, intrinsic, bias, or map tangent blocks without importing a
particular SLAM backend. This is infrastructure for SLAC factors; it is not yet
a claim that every descriptor produced by `build_problem` has a production
measurement frontend.

The design follows the joint-estimation structure demonstrated by:

- Zhou and Koltun, *Simultaneous Localization and Calibration:
  Self-Calibration of Consumer Depth Cameras*, CVPR 2014, which jointly
  optimizes camera trajectory and a calibration correction function.
- Reinke, Camurri, and Semini, *A Factor Graph Approach to Multi-Camera
  Extrinsic Calibration on Legged Robots*, arXiv `1811.01254`, which places
  robot poses and camera extrinsics in one modular factor graph.
- Brookshire and Teller, *Automatic Calibration of Multiple Coplanar Sensors*,
  RSS 2011, which emphasizes suitable motion and explicit calibration
  observability rather than treating solver convergence as identifiability.

No code from those systems or from GPL graph optimizers is copied into
`src/calibrex`.

## Numerical contract

`JointParameterBlock` owns a named Euclidean tangent vector. Manifold-valued
states remain the factor/frontend's responsibility: an SE(3) factor can decode
six tangent values with its declared retraction, while a time offset can use a
one-dimensional block. Fixed blocks implement explicit gauges and never enter
the normal equations.

`JointResidualBlock` declares its variable names, measurement family,
observation group, weight, and residual callback. The optimizer computes
factor-local central-difference Jacobians, assembles a joint train Jacobian,
applies scalar Huber weights, and uses damped Gauss-Newton/LM steps. The
implementation depends only on NumPy and typed Calibrex contracts.

Observation groups, not individual scalar residuals, are deterministically
split. Camera, LiDAR, IMU, and Radar factors from one capture can therefore be
kept together and cannot leak across train and holdout merely because they are
different modalities.

## Observability and falsification

The final train Jacobian supplies singular values, numerical rank, condition
number, and weak parameter-block names. These are diagnostics and are never
serialized as covariance. A fixed gauge is distinct from measured
observability: fixing the world origin removes a coordinate freedom but does
not create information about an extrinsic or clock offset.

Each free block may declare per-dimension known-bad steps. Both signs are
applied only after fitting and scored on unchanged holdout factors. Synthetic
tests jointly recover two trajectory values, two extrinsic values, and one
time offset while preserving a fixed world gauge. They also prove grouped
holdout isolation, ten perturbation controls, and a coupled rank-one failure
whose weak directions span both extrinsic and time blocks.

## Remaining integration

The next layer translates concrete existing Calibrex LiDAR plane, hand-eye,
Radar velocity, and Camera-LiDAR evidence into these residual blocks and emits
schema-valid transforms/time offsets from one joint adapter. Public-data
evaluation must then compare this joint result with the already implemented
independent baselines without weakening their gates.
