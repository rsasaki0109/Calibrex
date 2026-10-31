# Analytic stopping diagnostic results

[Protocol](analytic_stopping_protocol.md) frozen at `bf5d96e0`. Probed initial and
final states of all 18 archived truth-start cases; no optimization was rerun and
no historical pose, convergence or recovery verdict was changed.

## Six wrong negative-bias stops

| Scene | Population | Final correction m | Cost | Cost after -1 cm bias step | Exact bias gradient | Fixed-grid best bias m | Fixed-grid cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | exact_z_visible | -0.000104 | 96.80952 | 94.39428 | 233.327 | -0.25 | 0.000531 |
| original | exact_z_all | -0.000100 | 99.04593 | 96.63073 | 233.323 | -0.25 | 2.224961 |
| near_foreground | exact_z_visible | -0.003076 | 93.79572 | 91.23502 | 247.381 | -0.24 | 0.110754 |
| near_foreground | exact_z_all | -0.003082 | 97.05922 | 94.49786 | 247.444 | -0.24 | 3.353853 |
| staggered | exact_z_visible | -0.000561 | 96.17171 | 93.73983 | 234.933 | -0.25 | 0.009470 |
| staggered | exact_z_all | -0.000537 | 98.85625 | 96.42552 | 234.823 | -0.25 | 2.666471 |

## A feasible descent remains at every failed stop

All six negative-bias truth-start fits terminated with SciPy status 3 (`xtol`),
with estimated correction still between approximately -.00010 and -.00308 m
instead of -.25 m. At those unchanged final poses, the analytic derivative of the
same objective with respect to correction is **+233.3 to +247.4**. The fixed -.01 m
probe decreases cost in all six cases; this is a feasible descent independent of
reference pose or target-bias selection. The full preregistered grid finds still
lower costs at -.25 m (original/staggered) or -.24 m (near-foreground).

The grid is a diagnostic, not a replacement estimate; no fit is retroactively
called successful. Its minimum is not proof of global pose/bias optimality.
Nevertheless, these results rule out "no better feasible correction exists" as an
explanation for the six stops. Step-size termination did not certify stationarity.

## Pose derivatives cross surface discontinuities

At a pose-coordinate perturbation of **1e-8 degree/metre**, each failed final
state has a raw-residual jump greater than 1 m for at least one point. All such
probes retain exactly the same support population. In this continuous first-surface
sampler that indicates a change in the sampled surface, not a missing-point count
or a pixel interpolation boundary. The actual optical first-surface measurement
can be discontinuous at silhouettes even with an exact raycaster.

The forward residual-Jacobian maximum singular value ranges from approximately
**9.42e8 to 1.85e9** at that step. Forward versus central Jacobian relative differences
are approximately one; the largest derivatives strongly depend on the finite-
difference step. At 1e-6 they remain of order 1e7, and at 1e-4 of order 1e5, consistent
with a finite jump divided by the step. This is not evidence of huge smooth physical
sensitivity or a meaningful Gaussian pose covariance.

In contrast, the finite-difference bias-gradient component closely agrees with the
analytic bias derivative at all probed steps, and bias perturbations have no >1 m
residual jumps. Bias-only movement is smooth here while the jointly optimized pose
coordinates encounter discontinuities. This identifies a concrete mismatch with
smooth trust-region linearization near the archived stops. The probes do not record
SciPy's internal trust-region state or exactly reproduce its machine-relative step
policy, so they do not prove the entire causal chain inside that implementation.

## Next controlled change

Test separating the smooth scalar bias update from pose updates, with the original
objective and fixed samplers retained. Use a bounded bias-only update based on
measured cost, count every call within a predeclared total budget, and never initialize
with the injected correction or adopt the diagnostic grid as a truth lookup.
Require actual objective decrease and retain all original correct/incorrect cases,
including positive/zero biases and true starts. Compare returned pose and correction,
not merely the solver's convergence label. Freeze this algorithm before scoring.

Do not simultaneously introduce boundary-aware image sampling or change tolerances:
that would obscure whether the stopping failure was addressed. This diagnosis also
reinforces that local Jacobian rank near a surface switch is not sufficient validation.
No core/default change, production promotion, new recovery count or SOTA claim.

[Full probes](../assets/analytic_stopping_20261029/evaluation.json) and
[schema](../assets/analytic_stopping_20261029/evaluation.schema.json) retain all
18 cases, both states, all seven axes/three step sizes and all grid costs. Six
focused tests, repository Ruff, strict documentation build, frozen hashes and
complete-probe checks pass. The sign/cost formula has an independent finite-
difference unit check; unsupported penalties do not contribute a bias derivative.
