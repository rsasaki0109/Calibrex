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

Version `v0.3` also permits a typed square-root information matrix per factor.
The backend-neutral residual contract is `r_w = sqrt(w) L r_raw`, where
`L^T L` is the declared information matrix. Dimensions and finite values are
checked before optimization. This supports correlated residual whitening
without embedding a backend covariance class in the core. Existing diagonal
pose priors now use this path, so A2D2 and TUM public runs exercise the same
contract. Every whitened train factor serializes its ID, family, scalar weight,
and exact matrix; the matrix is not presented as an estimated covariance.

Observation groups, not individual scalar residuals, are deterministically
split. Camera, LiDAR, IMU, and Radar factors from one capture can therefore be
kept together and cannot leak across train and holdout merely because they are
different modalities.

`BackendNeutralJointReassociation` adds an association-aware outer loop without
making the core depend on a point-cloud backend. A frontend callback returns
one deterministic factor and stable target ID per query. Each changed state is
warm-started through the same optimizer, while train/holdout observation groups
must remain byte-for-byte identical. Only train assignment pair Jaccard and
train retained-query fraction may stop the loop; holdout assignment stability
is recorded but cannot decide iteration count. Complete group drops, query
identity changes, and inner optimizer failures are explicit terminal statuses.

## Observability and falsification

The final train Jacobian supplies singular values, numerical rank, condition
number, and weak parameter-block names. These are diagnostics and are never
serialized as covariance. A fixed gauge is distinct from measured
observability: fixing the world origin removes a coordinate freedom but does
not create information about an extrinsic or clock offset.

Rank uses a declared relative threshold, `tau * sigma_max`, and serializes the
actual threshold beside the spectrum. Uniformly changing residual units can
therefore scale the spectrum without changing rank. Condition numbers remain
parameterization- and unit-dependent and are still labelled as diagnostics;
the relative rank policy does not turn them into covariance or physical
uncertainty. Weak-block attribution includes the full right nullspace when a
graph has fewer residual dimensions than free parameters; a rank failure can
therefore no longer serialize an empty weak-direction list merely because a
thin SVD omitted structural null vectors.

Each free block may declare per-dimension known-bad steps. Both signs are
applied only after fitting and scored on unchanged holdout factors. Synthetic
tests jointly recover two trajectory values, two extrinsic values, and one
time offset while preserving a fixed world gauge. They also prove grouped
holdout isolation, ten perturbation controls, and a coupled rank-one failure
whose weak directions span both extrinsic and time blocks.

## Concrete factors and public-data integration

Concrete residual builders now translate pose–extrinsic LiDAR point-to-plane
measurements, Radar Doppler with lever-arm/time terms, and diagonal tangent
priors into joint residual blocks. The LiDAR factor evaluates
`nᵀ(T_world_body T_body_sensor p-q)`. The Radar factor evaluates static-target
Doppler after applying body acceleration over the clock offset and
`omega cross t_body_radar` at the Radar origin. Tests prove that each factor is
zero at its coupled synthetic truth and responds to incorrect pose,
extrinsic, or time blocks.

`NativeJointSlacSolver` assembles those point-to-plane builders on the public
A2D2 front-left/front-right sample and emits a schema-valid extrinsic through
`solver.backend: native_joint_slac`. Target correspondences are split by
source-plane spatial regions. The source plane map is shared, so provenance
states that the holdout tests target-side generalization rather than a fully
independent mapping frontend. Diagonal priors are train-only and cannot leak
into the holdout objective.

A single synchronized pair observes only the composition of source pose and
target extrinsic. The adapter therefore introduces a declared source-pose
nuisance block with a configurable prior; it never calls this data-only
observability. On the documented A2D2 sample, the augmented Jacobian has rank
12 and condition number about 80.7. Empirical point-to-plane RMSE is about
0.321 m train and 0.216 m spatial holdout. Eighteen of 24 signed known-bad
pose/extrinsic probes worsen holdout RMSE (0.75), so that falsification metric
is honestly WARN rather than weakening its gate. The joint extrinsic differs
from the independent fixed-trajectory native baseline by about 0.057 m and
0.362 degrees. Raw NPZ hashes, split groups, prior policy, full iteration
history, probes, and baseline status are retained in result provenance.

The TUM RGB-D fr1/xyz integration adds eight query-pose blocks, a shared camera
mounting block, three disjoint map frames, independent ground-truth trajectory
support, a scalar depth model, and a constrained trilinear ray-depth lattice.
Three temporal windows expose unstable bias and cross-window transfer. The
same windows now exercise the generic reassociation loop for two warm-start
rounds. Inner optimization converges each time, but terminal train pair
Jaccard remains 0.804--0.914 against the unchanged 0.99 gate, so public TUM
honestly reports 0/3 outer convergence while retaining more than 99.8 percent
of queries. Holdout association stability remains diagnostic and cannot stop
fitting. Full assignments, split groups, iteration deltas, optimizer evidence,
and final frozen-correspondence probes are retained in provenance. A separate
fixed-population evaluator now rebuilds correspondence for every known-bad
step and penalizes unmatched holdout queries by 0.15 m. All 90 public probes
are valid with no support collapse, but the weakest detection fraction falls
from 0.533 frozen to 0.433 aware while perturbation pair Jaccard reaches 0.476.
The two falsification semantics remain separately named.

The Zhou--Koltun full-XYZ extension now adds 24 shared trilinear field
dimensions, the paper's two-sided correspondence factor, a fixed-map public
specialization, directed shape-preserving residuals, per-node local-rotation
updates, and an explicit six-mode field gauge. On public TUM, all final rounds
converge and field-only rank is 24/24, but extrinsic plus field is rank 24/30
and therefore WARN. Only one of three windows improves scalar-lattice holdout;
the worst increase is 0.004047 m and the weakest control-probe detection is
11/48. The full-XYZ ablation remains honestly FAIL rather than treating its
gauge-augmented 78/78 rank as data observability. Ordered transfer of the
shared extrinsic and field passes the unchanged 0.005 m margin in only 2/6
directions, with a worst +0.011746 m holdout increase. Local-rotation and gauge
residuals are excluded from that score, so regularization cannot manufacture
transfer success.
