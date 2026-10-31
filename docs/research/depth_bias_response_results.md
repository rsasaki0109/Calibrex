# Depth-bias response and ambiguity results

[Protocol](depth_bias_response_protocol.md) frozen at `ae51adee`. All 54 fits
completed, including 36 perturbed trials and 18 separately counted truth-start
controls. All 18 zero-bias result objects exactly reproduce the previous occlusion
study. No fitted bias parameter, retries or saturation changes were introduced.

| Injected depth bias | Perturbed recovery | Truth-start within gates | Perturbed median rotation | Perturbed median translation |
| --- | ---: | ---: | ---: | ---: |
| 0 m | 12/12 | 6/6 | 0.1268 deg | 0.0341 m |
| +0.25 m | 0/12 | 0/6 | 0.3875 deg | 0.2324 m |
| -0.25 m | 12/12 | 6/6 | 0.2452 deg | 0.0488 m |

## All trials

| Scene | Population | Bias m | Seed | Rotation deg | Translation m | Pass | Residual calls |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| original | exact_z_visible | +0.00 | +1 | 0.2116 | 0.0488 | True | 187 |
| original | exact_z_visible | +0.00 | -1 | 0.1985 | 0.0471 | True | 144 |
| original | exact_z_visible | +0.00 | +0 | 0.1986 | 0.0471 | True | 220 |
| original | exact_z_visible | +0.25 | +1 | 0.4664 | 0.2625 | False | 292 |
| original | exact_z_visible | +0.25 | -1 | 0.5172 | 0.2660 | False | 187 |
| original | exact_z_visible | +0.25 | +0 | 0.4708 | 0.2624 | False | 189 |
| original | exact_z_visible | -0.25 | +1 | 0.3006 | 0.0658 | True | 252 |
| original | exact_z_visible | -0.25 | -1 | 0.3006 | 0.0658 | True | 164 |
| original | exact_z_visible | -0.25 | +0 | 0.3004 | 0.0657 | True | 184 |
| original | exact_z_all | +0.00 | +1 | 0.2116 | 0.0487 | True | 186 |
| original | exact_z_all | +0.00 | -1 | 0.2118 | 0.0487 | True | 173 |
| original | exact_z_all | +0.00 | +0 | 0.1980 | 0.0469 | True | 163 |
| original | exact_z_all | +0.25 | +1 | 0.5005 | 0.2634 | False | 168 |
| original | exact_z_all | +0.25 | -1 | 0.5629 | 0.2672 | False | 227 |
| original | exact_z_all | +0.25 | +0 | 0.4715 | 0.2624 | False | 245 |
| original | exact_z_all | -0.25 | +1 | 0.3573 | 0.0726 | True | 214 |
| original | exact_z_all | -0.25 | -1 | 0.3574 | 0.0727 | True | 168 |
| original | exact_z_all | -0.25 | +0 | 0.3573 | 0.0727 | True | 150 |
| near_foreground | exact_z_visible | +0.00 | +1 | 0.1219 | 0.0336 | True | 200 |
| near_foreground | exact_z_visible | +0.00 | -1 | 0.1218 | 0.0336 | True | 179 |
| near_foreground | exact_z_visible | +0.00 | +0 | 0.1088 | 0.0319 | True | 137 |
| near_foreground | exact_z_visible | +0.25 | +1 | 0.2940 | 0.2324 | False | 586 |
| near_foreground | exact_z_visible | +0.25 | -1 | 0.5751 | 0.2352 | False | 535 |
| near_foreground | exact_z_visible | +0.25 | +0 | 0.3326 | 0.2375 | False | 216 |
| near_foreground | exact_z_visible | -0.25 | +1 | 0.2451 | 0.0487 | True | 221 |
| near_foreground | exact_z_visible | -0.25 | -1 | 0.2456 | 0.0488 | True | 220 |
| near_foreground | exact_z_visible | -0.25 | +0 | 0.2450 | 0.0483 | True | 176 |
| near_foreground | exact_z_all | +0.00 | +1 | 0.1318 | 0.0347 | True | 163 |
| near_foreground | exact_z_all | +0.00 | -1 | 0.1318 | 0.0347 | True | 143 |
| near_foreground | exact_z_all | +0.00 | +0 | 0.1318 | 0.0347 | True | 145 |
| near_foreground | exact_z_all | +0.25 | +1 | 0.3086 | 0.2323 | False | 592 |
| near_foreground | exact_z_all | +0.25 | -1 | 0.4800 | 0.2287 | False | 167 |
| near_foreground | exact_z_all | +0.25 | +0 | 0.2985 | 0.2341 | False | 316 |
| near_foreground | exact_z_all | -0.25 | +1 | 0.2452 | 0.0488 | True | 278 |
| near_foreground | exact_z_all | -0.25 | -1 | 0.2452 | 0.0488 | True | 347 |
| near_foreground | exact_z_all | -0.25 | +0 | 0.2438 | 0.0485 | True | 205 |
| staggered | exact_z_visible | +0.00 | +1 | 0.0684 | 0.0234 | True | 213 |
| staggered | exact_z_visible | +0.00 | -1 | 0.0637 | 0.0228 | True | 220 |
| staggered | exact_z_visible | +0.00 | +0 | 0.0623 | 0.0227 | True | 183 |
| staggered | exact_z_visible | +0.25 | +1 | 0.2489 | 0.2237 | False | 432 |
| staggered | exact_z_visible | +0.25 | -1 | 0.1414 | 0.2235 | False | 233 |
| staggered | exact_z_visible | +0.25 | +0 | 0.1528 | 0.2272 | False | 574 |
| staggered | exact_z_visible | -0.25 | +1 | 0.1217 | 0.0243 | True | 130 |
| staggered | exact_z_visible | -0.25 | -1 | 0.1143 | 0.0263 | True | 150 |
| staggered | exact_z_visible | -0.25 | +0 | 0.1217 | 0.0243 | True | 94 |
| staggered | exact_z_all | +0.00 | +1 | 0.0724 | 0.0237 | True | 238 |
| staggered | exact_z_all | +0.00 | -1 | 0.0715 | 0.0235 | True | 216 |
| staggered | exact_z_all | +0.00 | +0 | 0.0714 | 0.0236 | True | 131 |
| staggered | exact_z_all | +0.25 | +1 | 0.2497 | 0.2237 | False | 334 |
| staggered | exact_z_all | +0.25 | -1 | 0.0793 | 0.2266 | False | 383 |
| staggered | exact_z_all | +0.25 | +0 | 0.1288 | 0.2233 | False | 264 |
| staggered | exact_z_all | -0.25 | +1 | 0.1229 | 0.0241 | True | 149 |
| staggered | exact_z_all | -0.25 | -1 | 0.1229 | 0.0241 | True | 165 |
| staggered | exact_z_all | -0.25 | +0 | 0.1229 | 0.0241 | True | 86 |

## Interpretation

Positive bias is consistently absorbed into pose: the median camera-axis z
translation error over perturbed trials rises from +0.0215 m (zero bias) to
+0.2236 m (+0.25 m depth bias). All 18 positive-bias outputs fail the accuracy
gates, including truth starts. Three reach max_nfev; the other 15 pass numerical
convergence/support but remain inaccurate. Across all biases 51/54 converge;
actual residual calls span 86–592, below 800. No failed fits were removed.

Negative bias instead gives positive depth residuals, whose influence the fixed
visibility approximation suppresses. The twelve perturbed poses remain within
gates, but their median rotation/translation errors increase versus zero bias.
This is sign-dependent tolerance, not recovery of the missing depth correction.
No bias value is estimated, and the one-sided residual cannot alone distinguish
a camera-occluded point from a systematically underestimated depth.

## Local identifiability diagnostic

The fraction of the bias Jacobian column unexplained by the six pose columns is
0.9539–0.9842 across all returned poses; none meets the preregistered <0.05 local
near-confounding flag. The frontoparallel-plane unit control does expose exact
axial-translation/bias degeneracy. On these sphere/plane layouts, raw supported
residuals therefore have a locally distinguishable bias direction under this
finite-difference model, even though the six-parameter optimizer displaces the pose.

These statements are compatible: the current optimizer cannot change a bias
parameter and minimizes a one-sided, robustly weighted objective, not the raw
Jacobian norm used by the diagnostic. Scene edges, bilinear sampling and active
support can strongly affect local derivatives. This diagnostic does not establish
global uniqueness, noise-conditioned bias precision or observability after robust
weighting. Thirteen diagnostic residual passes per case are separate from fitting
calls/time. Do not infer from a large unexplained fraction that bias correction is
already safe to deploy.

## Next decision

Proceed to a separately frozen **experimental** joint pose-plus-depth-bias fit,
not production integration. Keep visibility treatment fixed, initialize correction
at zero without exposing injected bias to the estimator, and include zero/positive/
negative bias, both signs and truth starts. Report estimated bias error and pose
error separately; check both raw and actual robust-objective sensitivity. Include
a plane-only degeneracy control and reject any claim of uniqueness there. Compare
against this unchanged six-parameter baseline; do not use holdout labels to tune
bias bounds or regularization. Zero-bias degradation, hidden behind-surface fits,
and truth-start drift must remain explicit failure criteria.

The Japanese review at `780a61ea` is a historical snapshot: its proposed stages
A/B have now been executed here; stage C (joint bias fitting) remains unimplemented.
The printed PDF is not silently rewritten as though these results were known earlier.

[Evidence](../assets/depth_bias_response_20261029/evaluation.json) and
[schema](../assets/depth_bias_response_20261029/evaluation.schema.json) contain
all outputs, xyz errors, support/call counts, singular spectra and local projection
coefficients. Nine focused tests, repository Ruff, strict docs build, source-hash
checks, exact zero-bias reproduction and complete-denominator validation pass.
No new real data, core/default change or SOTA claim.
