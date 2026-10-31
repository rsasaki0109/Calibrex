# Camera-depth bias versus translation results

[Protocol](depth_bias_pose_protocol.md) frozen at `9bee5c16` and run once from that
commit: Linux, Python 3.12.3, NumPy 2.5.2, SciPy 1.18.0, one OpenBLAS thread, about
90 s of wall time. All 210 fits and 84 sensitivity evaluations completed, with no
exceptions or retries. The 18 zero-bias stage A fits exactly reproduce the
published [one-sided results](occlusion_pose_results.md) on a different machine.
The β = 0 profile fits exactly reproduce the stage A truth starts.

**Frozen verdict: `not_supported`.** The joint bias variable keeps every pose
within the gate, but it does not separate bias from camera-z translation to the
declared 0.05 m. It also degrades exact-depth (zero-bias) accuracy.

## A: pose-only fit under bias

| Scene | Bias m | Pass (perturbed / truth start) | Rotation deg | Translation m | Δt_z / b |
| --- | ---: | --- | --- | --- | --- |
| original | 0 | 4/4 / 2/2 | 0.198–0.212 | 0.047–0.049 | — |
| original | +0.25 | **0/4 / 0/2** | 0.466–0.563 | 0.262–0.267 | 0.96–0.97 |
| original | −0.25 | 4/4 / 2/2 | 0.300–0.357 | 0.066–0.073 | −0.16 to −0.15 |
| near_foreground | 0 | 4/4 / 2/2 | 0.109–0.132 | 0.032–0.035 | — |
| near_foreground | +0.25 | **0/4 / 0/2** | 0.294–0.575 | 0.229–0.237 | 0.85–0.89 |
| near_foreground | −0.25 | 4/4 / 2/2 | 0.244–0.246 | 0.048–0.049 | −0.11 |
| staggered | 0 | 4/4 / 2/2 | 0.062–0.072 | 0.023–0.024 | — |
| staggered | +0.25 | **0/4 / 0/2** | 0.079–0.250 | 0.223–0.227 | 0.89–0.91 |
| staggered | −0.25 | 4/4 / 2/2 | 0.114–0.123 | 0.024–0.026 | −0.08 to −0.07 |

Each row pools the visible and all-return populations. A **positive** bias (camera
depth too long) fails every case, 0/18 including the truth starts. The translation
moves along the camera axis by 85–97 % of the bias, and the error lands just above
the 0.20 m gate. In 15 of these 18 failures the fit converges and retains support.
The other 3 stop at `max_nfev`. A **negative** bias passes 18/18 for the
same reason the frozen fit handles occlusion. The returns appear behind the
surface, where the residual saturates, so the pose barely moves. This is masking,
not robustness. Errors still grow relative to zero bias (original 0.20 deg /
4.7 cm → 0.30–0.36 deg / 6.6–7.3 cm), and nothing in the fit reports the bias.

The pre-declared pooled absorption label is **false**: the median Δt_z/b over the
36 biased fits is 0.39. The two signs behave in opposite ways, so the pooled
median sits between them. The per-sign description above is post hoc and does not
change any verdict.

## B: identifiability

**Local sensitivity at the reference** (oracle diagnostic). The values agree across
the three biases to 1e-14, as expected.

| Scene | Unexplained fraction e | R² on pose | Alias Δt_z per m | Bias sd inflation | β–t_z correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 0.899 | 0.19 | 0.18 | 1.11 | −0.44 |
| near_foreground | 0.943 | 0.11 | 0.10 | 1.06 | −0.33 |
| staggered | 0.929 | 0.14 | 0.12 | 1.08 | −0.37 |

Visible and all-return populations agree to 0.001. The rows used are
11,255–11,520, with no probe-unstable rows. All 18 conditions pass the frozen
e ≥ 0.04 rule and are labeled *distinguishable*, so no joint result is set aside
as ambiguous.

The local measure contradicts stage A. It predicts a pose-only translation shift
of 0.10–0.18 × b, but the actual nonlinear fit moved 0.85–0.97 × b under positive
bias. A plausible explanation, not tested here, is that the local Jacobian gives
silhouette rows full weight, and their strong pose sensitivity holds only for
infinitesimal moves. **A favorable local identifiability number therefore did not
predict the finite-shift confounding**, and it must not be used as a safety check
on its own.

**Bias profile** (frozen pose-only fit from the truth start on depth − β). On the
0.25 m grid the lowest cost is at the true bias in 18/18 conditions, and that fit
passes 18/18. The profile is strongly one-sided:

- **Bias corrected too little** (β = b − 0.25). The cost is only 34–145 % higher
  than at the truth. The translation lands 0.223–0.262 m off, and **0/18** pass.
- **Bias corrected too much** (β = b + 0.25). The cost is 266–400 % higher, yet the
  pose stays within the gate in **18/18**.

So the objective does prefer the true bias. Yet a modest cost increase buys a
gate-failing translation in one direction only. Nine profile fits stop at
`max_nfev`. The grid is too coarse to locate a continuous minimum.

## C: separate pose plus bias fit

| Scene | Pose pass | Rotation deg | Translation m | β̂ − b m | Bias within 0.05 m | Pose-only zero-bias translation m |
| --- | --- | --- | --- | --- | --- | --- |
| original | 22/22 | 0.221–0.263 | 0.078–0.087 | −0.069 to −0.059 | 0/22 | 0.047–0.049 |
| near_foreground | 22/22 | 0.154–0.200 | 0.061–0.068 | −0.061 to −0.052 | 0/22 | 0.032–0.035 |
| staggered | 22/22 | 0.064–0.072 | 0.044–0.045 | −0.044 to −0.042 | 22/22 | 0.023–0.024 |

Per scene, the 22 cases are 6 zero-bias fits and 16 biased fits, including 4
full-truth starts. Pose and bias outcomes are kept separate:

- **Pose:** 66/66 pass. All fits converge and retain support, and the largest
  run takes 417 actual calls.
- **Bias:** 22/66 within 0.05 m. Of the biased cases, 16/48 recover the bias; at
  zero bias, 6/18 report |β̂| < 0.05 m.
- **Trade-off:** every estimate satisfies β̂ − b ≈ −Δt_z within ±2 mm. The joint
  fit slides 4–7 cm along a bias/camera-z trade direction. The slide is the same
  for every true bias, start and population. Adding b only reparameterizes the
  joint problem, so this is a property of the objective, not of the bias value.
- **Zero-bias damage:** with exact depth, the joint fit's translation error is
  4.4–8.5 cm, against 2.3–4.9 cm for the frozen pose-only fit. Rotation also
  worsens on original and near_foreground. Everything still passes the gate, but
  the extra variable costs exact-depth accuracy.
- **Reference-free check:** the stage B sensitivity evaluated at each estimate
  says *distinguishable* every time (e 0.69–0.90), matching the oracle label in
  66/66. Yet 44 of those estimates miss the bias tolerance. The diagnostic does
  not detect the systematic offset.

## Decision

Rule 3 applies. There are 12 zero-bias regressions (|β̂| ≥ 0.05 m on original and
near_foreground), and 32 of the 48 distinguishable biased cases miss the bias
tolerance. **Do not adopt the bias variable.** Do not quote the 66/66 pose passes
as solving depth bias, and do not widen the 0.05 m tolerance after seeing these
estimates, which sit about 0.04–0.07 m away. The frozen one-sided pose-only fit
remains the research baseline, outside production.

Answer to the question for these noiseless analytic scenes:

- **Pose-only fit:** cannot tell a positive constant depth bias from a camera-z
  translation shift (0/18), and hides a negative bias behind its occlusion
  saturation.
- **Joint fit:** the objective prefers the true bias, but the fit trades bias
  against translation almost 1:1, with a 4–7 cm offset.
- **Overall:** bias and translation are **not separated** to the declared
  accuracy.

Positive camera-depth bias is an open blocker before any real-data rerun of this
estimator.

Not run, and requiring a new frozen protocol:

- Find what fixes the 4–7 cm β/t_z offset at zero bias. Candidates are the
  interpolation residual floor at silhouettes and the one-sided asymmetry.
- Test whether geometry or views that break the trade direction separate them
  better.
- Test detecting bias rather than estimating it. The sign-asymmetric
  front/behind residual populations are one candidate signal. It would need
  wrong-candidate controls, because earlier validators accepted wrong poses.

Real stereo bias is often range-dependent rather than constant, so a constant
bias is the easiest case and a pass here would not have transferred directly.

## Limits and evidence

- **Scenes:** three noiseless sphere/plane layouts that were already used for
  development. They share one renderer, one extrinsic and the same views. They are
  not independent scenes, not real data and not SOTA evidence.
- **Oracle diagnostic:** the local test is evaluated at the reference pose. It
  never fed an estimator or a selection.
- **Budget:** actual residual calls are 12,534 for stage A, 22,705 for the
  profile, 15,229 for stage C and 1,260 for the sensitivity evaluations. That is
  51,728 in total, against the 169,260 ceiling. The most calls in any fit was 628
  (in the profile), so none reached the 800 cap. Fitting time summed to 21.3 s
  (A), 38.4 s (profile) and 26.2 s (C).

[Full evidence](../assets/depth_bias_pose_20261029/evaluation.json) and
[schema](../assets/depth_bias_pose_20261029/evaluation.schema.json) keep every
fit: status, message, pose, camera-frame translation error, bias estimate, support
per view, behind/front counts, actual calls, runtime and both sensitivity
evaluations. Core code and defaults are unchanged.
