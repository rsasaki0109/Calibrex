# Iterative registration correspondence stability research note

## Primary sources and scope

Besl and McKay's ICP alternates closest-point association with rigid transform
estimation. Segal, Haehnel, and Thrun's Generalized-ICP retains iterative
association while replacing the geometric error model with a probabilistic
plane-to-plane formulation.

Primary references:

- Besl and McKay, *A Method for Registration of 3-D Shapes*, PAMI 1992:
  <https://graphics.stanford.edu/courses/cs164-09-spring/Handouts/paper_icp.pdf>
- Segal, Haehnel, and Thrun, *Generalized-ICP*, RSS 2009:
  <https://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf>

No implementation code from either source is copied into `src/calibrex`.

## Diagnostic contract

A fixed-correspondence optimum is not sufficient evidence for an iterative
closest-point frontend: changing the transform can change both which queries
remain inside the gate and which target each retained query selects. Calibrex
therefore represents assignments as typed `(query_id, target_id)` pairs and
reports query retention, query-set Jaccard, exact pair Jaccard, and same-target
fraction separately. It also records the exact dropped, added, and reassigned
query IDs. Empty comparisons remain explicitly undefined rather than being
reported as perfect stability, and duplicate query IDs are rejected.

## Public TUM optimized rematching

Each held-out query is transformed with its optimized pose, extrinsic,
constant depth bias, and zero-mean spatial lattice, then reassociated to the
nearest voxel plane under the unchanged 0.15 m gate. Integer voxel coordinates
provide deterministic target IDs. The start-60, 180, and 300 windows retain
99.86, 99.56, and 98.05 percent of fixed-assignment queries, respectively, but
their exact pair Jaccards are only 0.769, 0.529, and 0.478. Same-target
fractions fall to 0.870, 0.694, and 0.653.

Rematching lowers held-out RMSE by 0.99, 2.12, and 2.92 mm. This decrease is
selection-dependent and is not presented as independent accuracy improvement.
Instead, high retention combined with low pair Jaccard shows that most points
remain inside the gate while many select different local planes. The minimum
pair-Jaccard metric therefore FAILs its 0.5 WARN threshold. Every per-query
drop, addition, and reassignment is retained in provenance.

The next step uses repeated reassociation during optimization and compares its
numerical curvature against the fixed-correspondence Hessian approximation.

## Numerical-curvature contract

`evaluate_numerical_curvature` evaluates the objective itself at symmetric
finite-difference probes. It does not freeze correspondences and does not label
the result covariance. The serialized result includes the full Hessian,
eigenvalues, positive/negative/near-zero counts, numerical rank, positive
condition number, physical perturbation steps, and exact objective-evaluation
count. Synthetic tests recover a coupled convex quadratic Hessian within
`1e-9`, detect one negative direction in a saddle, retain an independent flat
direction, and reject non-finite probes.

For public TUM, curvature is evaluated over the shared extrinsic tangent and
constant depth bias (seven dimensions), with optimized poses and zero-mean
lattice fixed. Unmatched probes receive the unchanged 0.15 m gate as a fixed
population penalty. Each fixed and rematched Hessian uses 99 objective
evaluations. All fixed-correspondence Hessians are positive definite and rank
7/7. In contrast, the rematched Hessians have 3, 6, and 6 negative eigenvalues
at starts 60, 180, and 300. Their Frobenius differences from the fixed Hessians,
normalized by fixed-Hessian norm, are 1.92, 10.12, and 10.68. Thus full rank
does not imply a locally convex rematched objective, and fixed `J^T J` curvature
materially misrepresents the association-aware landscape.

## Multi-start basin contract

Multi-start outcomes are clustered in parameter space after division by
declared physical scales. Only converged starts participate. Each basin is
represented by its lowest-objective member and records every contributing
start plus its basin fraction. Objective-competitive basins use separately
declared absolute and relative tolerances. More than one separated competitive
basin is reported as ambiguity, with best-to-second objective gap and maximum
normalized separation. Synthetic tests distinguish repeated starts in one
basin from two objective-equivalent symmetric minima separated by 20 physical
scale units, and retain non-converged start IDs in provenance.

The native local-search companion uses deterministic best signed-coordinate
probes, physical per-parameter steps, explicit improvement tolerance, and
step-halving termination. Its output records every accepted objective value,
final steps, sweeps, and exact evaluation count. Synthetic tests recover a
coupled quadratic minimum from a distant start and preserve the two distinct
minima of a symmetric double-well from positive and negative starts.
