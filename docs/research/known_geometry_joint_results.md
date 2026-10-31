# Known-geometry joint diagnostic results

The [protocol](known_geometry_joint.md) and generator were frozen at `9ffd9aee`.
All 15 fits completed and report convergence. **Perturbed recovery is 1/10**;
the five reference-start checks are reported separately and are not counted as
perturbation successes. This is one analytic scene observed from three positions,
not five independent scenes or a real-data validation.

| Condition | Seed | Rotation deg | Translation m | Pass | Truth MI | Final MI |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| exact_z_visible | +1 | 1.0459 | 0.2160 | False | 1.767995 | 1.720658 |
| exact_z_visible | -1 | 0.8972 | 0.1700 | False | 1.767995 | 1.756297 |
| exact_z_visible | +0 | 0.0349 | 0.0063 | True | 1.767995 | 1.769543 |
| exact_z_all | +1 | 1.0717 | 0.2349 | False | 1.718652 | 1.679094 |
| exact_z_all | -1 | 1.3748 | 0.2359 | False | 1.718652 | 1.713213 |
| exact_z_all | +0 | 0.0271 | 0.0000 | True | 1.718652 | 1.721869 |
| exact_range_visible | +1 | 0.4725 | 0.0718 | True | 2.245966 | 2.244564 |
| exact_range_visible | -1 | 0.8184 | 0.2767 | False | 2.245966 | 2.210985 |
| exact_range_visible | +0 | 0.2500 | 0.0000 | True | 2.245966 | 2.248469 |
| biased_z_visible | +1 | 1.0459 | 0.2160 | False | 1.767995 | 1.720658 |
| biased_z_visible | -1 | 0.8972 | 0.1700 | False | 1.767995 | 1.756297 |
| biased_z_visible | +0 | 0.0349 | 0.0063 | True | 1.767995 | 1.769543 |
| holes_z_visible | +1 | 1.1449 | 0.2278 | False | 1.790869 | 1.742738 |
| holes_z_visible | -1 | 1.2791 | 0.2236 | False | 1.790869 | 1.773147 |
| holes_z_visible | +0 | 0.0884 | 0.0153 | True | 1.790869 | 1.795028 |

## What this isolates

With exact optical-z depth and oracle-visible points, both perturbed initializations
fail the full-pose gates (0.897–1.046 degrees rotation error). The exact-reference
start drifts only 0.035 degrees and 6.25 mm. Thus this scene exposes a failure to
recover from perturbation even without learned stereo error, camera occlusions,
missing depth or sensor noise. It does not prove that every KITTI failure has
this cause, nor that the analytic scene has general real-world observability.

For **all ten perturbed fits**, the known reference has higher MI than the returned
solution under the same fixed histogram ranges, including the one passing pose.
The optimizer therefore did not reach even this known feasible better objective.
This is evidence of a search limitation for this objective/scene. We have not
computed its global maximum or established that the optimum equals the true pose.
Reference-start runs increase MI slightly while drifting; the largest drift is
0.25 degrees with radial depth. Objective bias/discretization also remain relevant.

A constant +0.25 m optical-depth shift gives identical traces and poses to exact
optical depth: automatic histogram endpoints shift with the camera values, leaving
histogram assignments unchanged. This control demonstrates affine-depth
invariance here; it does not show that biased metric depth is physically correct.
The radial-range representation helps one seed (1/2), but the other still fails.
Neither observation justifies changing the production depth convention.

All-return and missing-rectangle variants each recover 0/2. The oracle filter
retains 3783/3776/3780 of 3840 LiDAR returns across the three views. The missing
rectangle removes 9000 of 61440 camera pixels per view while keeping the same
visible LiDAR points. Point counts and complete traces are retained in the
[evaluation](../assets/known_geometry_joint_20261029/evaluation.json), with a
[report schema](../assets/known_geometry_joint_20261029/evaluation.schema.json).
The protocol document remains unchanged because its hash is pinned in the plan.

## Decision

Do not replace the stereo model again on the assumption that better depth alone
will fix recovery. Next compare the unchanged objective with a deterministic
broader search, using a declared equal evaluation budget and these frozen
synthetic cases first. Include reference-start drift and objective values, then
return to matched spent real data only if controlled recovery improves. Keep the
existing single-start baseline, count all attempts, and report runtime/budget;
no cherry-picked restarts or hidden reference-guided initialization. This is an
optimizer investigation, not authorization to promote an unvalidated solver.

Five focused unit tests (analytic geometry plus existing six-DoF behavior),
repository Ruff, report validation and strict documentation build pass. The core
solver and production defaults are unchanged. No SOTA claim.
