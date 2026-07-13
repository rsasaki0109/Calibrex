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

The generic backend-neutral outer loop now performs this repeated
reassociation with invariant train/holdout groups and train-only stopping. On
the three public TUM windows, two warm-start rounds retain at least 99.87
percent of train queries but reach terminal train pair Jaccards of only 0.914,
0.804, and 0.847. All three therefore hit the declared outer limit instead of
passing the unchanged 0.99 gate. Holdout pair Jaccards of 0.902, 0.834, and
0.852 are recorded solely as diagnostics.

## Reassociation-aware known-bad probes

Frozen-correspondence perturbations can overstate falsification when the
frontend would select a different target after a calibration change. Calibrex
therefore provides a separate reassociation-aware probe evaluator. Each signed
declared parameter step reruns the deterministic frontend callback and records
query/target stability, retention, support collapse, fixed-population RMSE,
delta, detectability, and callback errors.

The population is the original seeded holdout query set. Each matched factor
contributes its own whitened residual RMS, so factors are query-balanced even
when residual dimensions differ. Each unmatched query contributes a declared
residual penalty. Dropping difficult queries therefore cannot lower the score;
support collapse remains a separate diagnostic rather than being confused
with geometric improvement.

Synthetic tests distinguish the two probe semantics. A fixed target detects
both signed perturbations, but a rematched symmetric target follows the state
and reduces reassociation-aware detection to 0/2 while pair Jaccard falls to
zero. A second test drops half the holdout population under one perturbation:
the penalty still detects the probe and explicitly marks support collapse.
Frozen and reassociation-aware probe results are never merged under one metric
name.

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

The public TUM primary window declares five starts: the optimized center,
signed 0.03 m extrinsic-z perturbations, and signed 0.02 m depth-bias
perturbations. All searches optimize the full seven shared directions under
the same rematched fixed-population objective. Five of five searches converge,
but they form two objective-competitive basins. Their best objectives differ
by only `3.73e-6`, while their representatives are separated by 1.92 declared
physical scale units. Basin occupancy is 0.4 versus 0.6. The better basin
couples an approximately +0.03 m extrinsic-z shift with an approximately
-0.03 m depth-bias shift relative to the competing basin. This independently
confirms the translation/depth ambiguity indicated by the negative rematched
curvature, so the public multi-start ambiguity metric FAILs.
