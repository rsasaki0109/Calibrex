# Larger surface support improves precision but leaves model-bias failures

Frozen `f3672551`, evaluated 2026-10-30. See
[protocol](coherent_surface_protocol.md),
[plan](../assets/coherent_surface_20261030/plan.json),
[all108 scale/frame conditions](../assets/coherent_surface_20261030/evaluation.json)
and [schema](../assets/coherent_surface_20261030/evaluation.schema.json).

Larger support substantially improves the usefulness of these synthetic surface
estimates. However, no fixed scale or adaptive rule passes every noisy frame,
and the angular uncertainty still fails to cover curved/boundary model error.
Do not transfer this construction to real data or rerun ICP as an accepted filter.

| Support | Seed | Median selected joint precision | Median retention | Frames passing all quality/support gates |
| --- | ---: | ---: | ---: | ---: |
| 5 | 730 | 0% | .15% | 0/9 |
| 5 | 731 | 0% | .16% | 0/9 |
| 11 | 730 | 37.05% | 3.89% | 1/9 |
| 11 | 731 | 37.44% | 3.83% | 1/9 |
| 21 | 730 | 80.94% | 30.65% | 5/9 |
| 21 | 731 | 81.69% | 30.81% | 5/9 |
| Smallest passing scale | 730 | 79.02% | 33.99% | 3/9 |
| Smallest passing scale | 731 | 79.47% | 34.22% | 4/9 |

Joint precision requires smoothed-center XYZ error<=.2m AND normal error<=15deg.
All methods use a common ten-pixel inset domain, and input hashes match the earlier
27 measurements. These numbers are not a paired comparison to prior raw-pixel XYZ
scoring: the fitted plane now smooths depth too. The5-pixel output can accept
small boundary populations because the fitted-center variance is smaller than
raw depth variance, yet those selected points have zero joint precision.

The21-pixel noisy precision ranges68.94–91.46% across frames. Retention ranges
21.30–36.15%, occupying15–16 cells. Adaptive output occupies all16 cells, but
accepting the smallest eligible scale loses precision relative to21 in several
conditions; choosing the best scale using truth was not allowed. Neither is
selected as a deployment winner after observing these results.

## Curvature and boundary bias remain

For clean controls,21-pixel median precision is74.65%, range65.55–84.84%; only2/9
frames pass all gates. Adaptive clean precision median73.33%, also2/9 passing.
Thus reduced random error alone does not account for all wrong surfaces. Model
bias is visible even without injected noise. The1px plane RMS criterion does not
certify that a curved or mixed patch yields the correct center normal/position.

At21 pixels, observed angular coverage under the conservative Gaussian plane
bound is63.33–87.48% in noisy conditions (median76.19/75.96% for seeds730/731).
Adaptive coverage is61.98–87.46%. These are not calibrated95% confidence ranges
for this nonplanar geometry. The bound construction assumes an exact affine plane;
its covariance does not describe curvature/discontinuity error. Enlarging support
improves random-noise precision but cannot remove that model mismatch by itself.

Four focused tests, repository Ruff and strict documentation build pass. Tests
verify exact affine-disparity reconstruction across all scales, unchanged analytic
plane normals/depths, decreasing plane-noise bound with support and missing-patch
rejection. Validation checks schema,108-case completion, all frozen hashes and
paired input-depth identity. No calibration fits, real-data tuning, production
change or SOTA claim occurred.

Next focus on measurement-model validation, not another solver: require independent
spatial patch prediction or curvature-aware fitting to identify model error before
using a noise-only angular bound. Freeze that construction and tests against both
exact planes and curved/boundary controls, keeping all failed frames and coverage/
spatial-support gates. Do not retune window/RMS thresholds on the18 real frames.
This pilot establishes a useful larger-support direction, not a ready correspondence
provider or demonstrated calibration recovery.
