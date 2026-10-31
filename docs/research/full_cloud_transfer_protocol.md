# Full-cloud control of real-data transfer

Pre-registered 2026-10-30. Remove only the file-order point thinning from the
spent-data transfer input. Retain all points in each archived PLY, all three
restored SGBM depth arrays, original intrinsics, six drives and 18 starts
(positive/negative perturbations and reference). Hash-pin the same inputs.

Run unchanged bilinear metric, existing MI defaults, and edge+scalar-rescue:
54 fits. The rescue's complete nested baseline supplies the edge-only result,
so do not redundantly refit edge. No change to loss, interpolation, correction,
search bounds, support/identifiability, budgets or gates. MI's 800 evaluation
limit and metric's 800 actual residual calls have different accounting semantics.
All clouds remain full input; each algorithm retains its own correspondence rules.

Report 12 perturbed recoveries and six reference-start controls separately,
.5 degree/.20 m gates, convergence/support/rank rejection, pose errors, runtime,
full cloud counts and scalar rescue behavior. Compare full-cloud MI outputs to
all 12 archived SGBM outputs in joint_learned_stereo_20261029/evaluation.json,
including transforms and convergence, to check historical 6/12 reproducibility.
Compare metric outputs to the prior 4096-point controls without replacing them.
No true bias labels exist in real data; report nuisance estimates, not bias accuracy.

For bilinear and final rescue, retain the previous four-state diagnostics:
reference/fitted poses × zero/fitted bias, supported raw residual distribution,
total/support-only cost and common-support cost. Fixed fitted-bias comparisons
do not prove a global optimum or identify a unique sensor error cause. Reference
poses are offline controls only; no truth enters an online acceptance rule.

No new data, thresholds, model changes or production adoption. Abort on hash
mismatch or execution failure and preserve logs/partial report. Do not drop hard
cases or retry with smaller clouds. Runtime includes each solver's sensitivity
checks but excludes loading and the separate four-state diagnostics. This is
spent-data diagnosis, not independent SOTA evidence.
