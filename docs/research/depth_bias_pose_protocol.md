# Camera-depth bias versus extrinsic translation

Question: can a constant camera-depth bias be told apart from a shift of the
camera–LiDAR translation, using the frozen [one-sided metric fit](occlusion_pose_protocol.md)?
The earlier [symmetric metric fit](joint_metric_results.md) moved the translation by
about 0.222 m under a +0.25 m depth bias. This protocol measures that confounding
first, then tests identifiability, and only then fits a bias variable, as a separate
stage. Everything below is fixed before any fit on these conditions.

## Frozen inputs

- **Estimator.** `tools/joint_occlusion_pose.py` exactly as frozen at `a2e22ce5`:
  a supported residual r > 0 becomes r/sqrt(1+(r/0.2)^2), and front and unsupported
  penalties are unchanged. There is no new saturation, threshold or layout.
- **Data.** The three already-used analytic layouts of `tools/occlusion_geometry.py`
  (original, near_foreground, staggered), both populations (`exact_z_visible`,
  `exact_z_all`) and the same three views and extrinsic. Camera depth reads exact
  optical z plus b, with **b ∈ {0, +0.25, −0.25} m**. LiDAR points are exact.
  With r = LiDAR z − camera z, the residual at the truth is −b. A positive bias puts
  returns in front of the surface (full penalty). A negative bias puts them behind
  it, where the residual saturates. These are synthetic development controls on
  layouts already used for tuning. No real data is read, and none of this is unseen
  evaluation.
- **Starts.** The prior signed perturbation, ±(1.0, −0.5, 0.75) deg and
  ±(0.15, −0.1, 0.1) m, plus the truth start. The bias variable starts at 0. For
  b ≠ 0, the joint fit adds one *full-truth* start, with the true pose and β₀ = b,
  as a drift control.
- **Pass gate (unchanged).** Converged, with support retained, and rotation error
  < 0.5 deg and translation error < 0.20 m. Bias error is reported separately. A
  bias counts as **recovered** when |β̂ − b| < 0.05 m.
- **Reference use.** The reference pose generates the data, labels the outcomes,
  sets the declared truth starts and is the evaluation point of the stage B oracle
  diagnostic. It never enters an estimator, a candidate choice or a parameter choice.

## Stages

**A: confounding (54 fits).** The frozen pose-only fit on every scene × population ×
bias × start. The 18 zero-bias fits must reproduce the published one-sided results
exactly, or the run is invalid. Recorded per fit: the pose errors, the gate, the
camera-frame translation error, the absorption ratio a = Δt_z / b, support per
view, behind/front counts over 20 cm, actual residual calls and runtime.
Translation is said to absorb the bias when the median of a over biased fits is in
[0.5, 1.5]. This label is descriptive.

**B: identifiability (18 evaluations, 90 fits).**

1. *Local sensitivity (oracle diagnostic).* At the reference pose with β = b,
   central differences of the one-sided residual. Steps are 1e-3 deg, 1e-4 m and
   1e-4 m. Rows are points supported at that point and under every probe, weighted by
   sqrt(ρ′) of the soft-L1 loss (f = 0.1 m), as the solver sees them. Reported: the
   fraction e of the bias column that no pose change reproduces, R², the alias pose
   change per metre of bias, Cramér–Rao-type standard deviations per unit residual
   noise (joint and pose-known), the β–t_z correlation and the unit-column condition
   number. A condition is **distinguishable** when e ≥ 0.04, meaning at least 1 cm RMS
   of a 0.25 m bias signature cannot be imitated by pose. At the true parameters
   this Jacobian does not depend on b except by rounding, so the three biases repeat
   one measurement per scene × population.
2. *Bias profile.* The frozen pose-only fit from the truth start on depth − β, for
   β ∈ {−0.5, −0.25, 0, +0.25, +0.5} m, to see whether different bias/pose pairs reach
   similar objective values. The values are descriptive. β = 0 must reproduce the
   stage A truth start exactly.

**C: separate joint bias fit (66 fits).** `tools/depth_bias_pose.py` adds a single
β to the frozen residual. It uses the same soft-L1 (0.1 m), TRF, tolerances,
`max_nfev = 100`, pose bounds and scales, and the same support rule. The bias is
bounded to β₀ ± 1 m with scale 0.1, and the 800 actual-call cap includes Jacobian
probes. Reported separately: the pose errors and the gate, β̂ and its error, and the
reference-free sensitivity of B.1 evaluated at the estimate itself. Its agreement
with the oracle label is reported. It is not a gate.

## Decision rules (`summarize`, frozen in code)

1. `invalid_reproduction_failed` if the 18 zero-bias stage A fits or the β = 0
   profile fits do not exactly reproduce.
2. `not_identifiable` if no biased condition is distinguishable.
3. `not_supported` if any zero-bias joint fit breaks a stage A pass or has
   |β̂| ≥ 0.05 m, or if any start in a distinguishable biased condition fails the
   pose gate or the bias tolerance.
4. Otherwise `supported_synthetic_diagnostic`.

Joint results in non-distinguishable conditions are kept and reported, but never
count as support. No verdict promotes anything to production or to real data. A
supported verdict still describes three noiseless analytic scenes with a constant
bias only.

## Budget and execution

Each fit gets 100 `nfev` and at most 800 actual residual calls; a fit that exceeds
the cap is a failure. Each sensitivity evaluation uses exactly 15 residual
evaluations. The ceiling is (54 + 90 + 66) × 800 + (18 + 66) × 15 = 169,260 residual
calls. The run happens once, single-threaded (`OPENBLAS_NUM_THREADS=1`), with no
retries. Every exception is recorded as a failed fit, not dropped. If the run
crashes because of a code defect, the crash is disclosed, the fix gets its own
commit, and the whole run is repeated from scratch. Gates, tolerance, steps, bias
values and layouts are not changed after results.

`docs/assets/depth_bias_pose_20261029/plan.json` pins the SHA-256 of every source,
this protocol and the prior evaluation used for reproduction. It also pins the
conditions, and the evaluator compares them with its constants before it runs.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  tools/evaluate_depth_bias_pose.py docs/assets/depth_bias_pose_20261029/plan.json \
  docs/assets/depth_bias_pose_20261029/evaluation.json
```

## Out of scope

Depth noise, scale or range-dependent bias, per-view bias, real stereo depth and
external methods are not tested here. Neither the 0.20 m gate nor the 0.05 m bias
tolerance moves for a near miss.
