# Camera–LiDAR: distinguish search failures from remaining ambiguity

**Finding:** the two lost 0005 controls contain missed training-score candidates;
the targeted 0002 pitch miss has a different failure mode. This cross-scoring
analysis does not refit poses or demonstrate recovered detection coverage.

## Diagnostic scope

Select the two lost controls (roll and yaw on 0005 pitch +3) and the previously
targeted pitch control (0002 pitch +3). Keep all six folds of each case. Score
both methods' saved baseline and two profiled controls under both the original
scalar and new geometry objectives, separately on training and held-out frames.
Also score the dataset reference as a diagnostic; it must not become an estimator
initialization. The scalar method's saved fits use training-only initialization.

Snapshot and estimator hashes are checked. The saved own-objective held-out drops
reproduce within absolute 1e-9 for all 36 fit/fold combinations. There is no new
optimization, threshold change, data selection within a case or new accuracy claim.
All cross-scores, pose norms and hashes are retained in the
[report](../assets/geometry_failures_20261028/report.json),
[summary](../assets/geometry_failures_20261028/summary.json) and
[manifest](../assets/geometry_failures_20261028/manifest.json).

## Lost controls: explicit missed candidates

The following gains use the **geometry objective** for both poses. Positive gain
means the saved scalar fit scores higher than the geometry fit. Fold indices
are zero-based.

| Lost control | Fold | Geometry fit rotation error | Scalar fit rotation error | Training gain from scalar fit | Held-out gain from scalar fit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0005 pitch +3, roll axis | 1 | 2.002 deg | 0.122 deg | +0.09250 | +0.08741 |
| 0005 pitch +3, yaw axis | 5 | 2.101 deg | 0.150 deg | +0.09149 | +0.07684 |

These are concrete search deficits: an already-known training-only candidate
has a higher score under the new objective. Even the geometry method's own
nuisance-refitted controls have higher training scores than its baseline in
these folds, by +0.01582 and +0.05414 respectively. Both are inside the existing
search domain. The baseline search did not select them because it followed a
different local search path.

In each lost control, this is also the only fold where a profiled control beats
the baseline on held-out frames. This links the detection loss to a bad baseline
fit, but replacing that baseline changes the signed controls that must be refitted.
Therefore these cross-scores do **not** prove that either lost detection is recovered.

## Remaining pitch miss: not the same search deficit

For 0002 pitch +3, none of the six saved scalar baselines or the new method's
own profiled controls beats the geometry baseline on training frames. The scalar
baselines have training deficits from 0.00084 to 0.00888 under the geometry
objective. Both profiled controls have negative training gains in every fold.

Yet a control beats the baseline on held-out frames in folds 1, 3 and 4. The
baseline full rotation errors remain within 0.354 deg. This is training-to-held
objective disagreement among the inspected candidates, not evidence that a
higher-training-score restart will fix the pitch miss. It does not prove global
optimality or rule out other unexplored basins.

## Next implementation hypothesis

Test a bounded, training-only baseline restart when a nuisance-refitted control
already outranks the baseline under the geometry objective. Release the tested
axis, refine the saved control seeds, keep the best training score, and then
recompute the controls. The existing control-seeded-baseline option provides
this mechanism. Its earlier scalar-objective experiment failed, so do not assume
success under the geometry objective; retain a new paired comparison on all
fourteen inputs. Do not use held-out scores or reference poses to select seeds.

This targets the demonstrated search deficits without claiming to solve the
separate 0002 pitch ambiguity. Native defaults, acceptance thresholds and
previous audit verdicts remain unchanged.
