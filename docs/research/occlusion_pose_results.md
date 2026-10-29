# One-sided occlusion residual results

[Protocol](occlusion_pose_protocol.md) frozen at `a2e22ce5`. All 36 fits completed.
All six original-scene symmetric result objects exactly reproduce the previous
metric-depth experiment. Scene layouts and saturation were not changed after scoring.

| Scene | Population | Seed | Symmetric deg / m | Pass | Asymmetric deg / m | Pass |
| --- | --- | ---: | --- | --- | --- | --- |
| original | exact_z_visible | +1 | 0.1415 / 0.0363 | True | 0.2116 / 0.0488 | True |
| original | exact_z_visible | -1 | 0.1433 / 0.0366 | True | 0.1985 / 0.0471 | True |
| original | exact_z_visible | +0 | 0.1414 / 0.0363 | True | 0.1986 / 0.0471 | True |
| original | exact_z_all | +1 | 0.7519 / 0.1054 | False | 0.2116 / 0.0487 | True |
| original | exact_z_all | -1 | 0.7518 / 0.1055 | False | 0.2118 / 0.0487 | True |
| original | exact_z_all | +0 | 0.7523 / 0.1054 | False | 0.1980 / 0.0469 | True |
| near_foreground | exact_z_visible | +1 | 0.2174 / 0.0380 | True | 0.1219 / 0.0336 | True |
| near_foreground | exact_z_visible | -1 | 0.2174 / 0.0380 | True | 0.1218 / 0.0336 | True |
| near_foreground | exact_z_visible | +0 | 0.2172 / 0.0380 | True | 0.1088 / 0.0319 | True |
| near_foreground | exact_z_all | +1 | 0.7523 / 0.0859 | False | 0.1318 / 0.0347 | True |
| near_foreground | exact_z_all | -1 | 0.7307 / 0.0870 | False | 0.1318 / 0.0347 | True |
| near_foreground | exact_z_all | +0 | 0.7244 / 0.0878 | False | 0.1318 / 0.0347 | True |
| staggered | exact_z_visible | +1 | 0.1095 / 0.0237 | True | 0.0684 / 0.0234 | True |
| staggered | exact_z_visible | -1 | 0.1084 / 0.0236 | True | 0.0637 / 0.0228 | True |
| staggered | exact_z_visible | +0 | 0.1090 / 0.0236 | True | 0.0623 / 0.0227 | True |
| staggered | exact_z_all | +1 | 0.4288 / 0.0521 | True | 0.0724 / 0.0237 | True |
| staggered | exact_z_all | -1 | 0.4476 / 0.0526 | True | 0.0715 / 0.0235 | True |
| staggered | exact_z_all | +0 | 0.4440 / 0.0526 | True | 0.0714 / 0.0236 | True |

## Outcome

Perturbed recovery across both visibility populations improves **8/12 → 12/12**.
The key all-return subset improves **2/6 → 6/6**, while oracle-visible controls
remain **6/6**. Reference-start controls improve **4/6 → 6/6** (all-return subset
1/3 → 3/3). Each asymmetric fit converges and retains support. No retry, failure
omission or threshold relaxation was needed. The prior original-scene all-return
~0.752-degree error falls to ~0.20–0.21 degrees / 4.7–4.9 cm.

Passing a gate is not absence of bias. In the original visible-only population,
asymmetric errors (~0.20–0.21 deg / 4.7–4.9 cm) are worse than the symmetric
baseline (~0.14 deg / 3.6 cm). All truth-start outputs move away from exact truth,
though within the declared limits. Saturation weakens genuine positive residuals
as well as occluded returns; that tradeoff remains even without missing geometry.
Do not state that this method is uniformly more accurate or unbiased.

Actual residual calls span 131–238 for asymmetric versus 111–341 for symmetric,
below the fixed 800 cap. Total fit times for 18 cases per method are 5.71 s versus
5.53 s on this CPU. All points retain residual slots and missing-support penalties;
reported >20 cm counts are geometric disagreement diagnostics, not inferred truth
occlusion labels. The estimator never receives oracle visibility or reference
extrinsics; oracle filtering exists only in the favorable input control.

## Decision and next step

This is a positive controlled result for an experimental approximation. Keep it
outside production: three sphere arrangements share the same analytic renderer,
background plane, extrinsics and noiseless sensor assumptions. No real-data or
SOTA claim follows. Prior metric-depth bias failure is not resolved by a test
that contains only exact depth.

Next isolate camera-depth bias and extrinsic-translation confounding, keeping this
visibility treatment fixed. Before adding a nuisance bias variable, measure whether
known depth offsets can be distinguished from pose changes across these scenes and
whether fitting a bias harms exact-depth recovery. Include positive/negative bias,
zero-bias and reference-start controls; report bias and pose separately. Freeze
settings and test observability/ambiguity, not just whether training residual falls.
Do not add more saturation tuning or promote solely from the 12/12 synthetic count.

[Full evidence](../assets/occlusion_pose_20261029/evaluation.json) and
[schema](../assets/occlusion_pose_20261029/evaluation.schema.json) preserve all
methods, statuses, support and signed population counts. Seven focused tests,
repository Ruff, strict documentation build, source-hash/report/call validation and
exact original baseline reproduction pass. Core code and defaults are unchanged.
