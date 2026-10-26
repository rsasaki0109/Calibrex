# Empirical intervals from independent complete-pipeline errors

The P0 pilot exposed undercoverage in conditional, fixed-map covariance. This
follow-up implements frozen per-axis empirical error radii, evaluates them on a
separate cohort, and corrects a world-frame mistake in the earlier gravity
comparison. It does not change the optimizer, replace its covariance, or enable
automatic deployment.

## Method and abstention

For nominal marginal coverage p and n independent calibration recordings, sort
absolute calibration errors and use rank `ceil((n+1)*p)` (one-based) as the
symmetric interval half-width. This is the split-conformal order statistic for
an already-fixed estimator. No Gaussian assumption or bias correction is fitted.
The complete pipeline, including initialization, maps and correspondence search,
runs independently for every recording.

Unavailable calibration errors count as infinity. If the required rank is not
available, the model records `inconclusive` and no finite width. For 95% coverage,
at least 19 attempts with adequate observations are needed; with 20 complete
attempts the radius is the largest calibration error. An observation with zero
formal std can still supply an empirical absolute error. Returned stalled or
quality-failing fits are retained, not selected away.

The finite-sample marginal statement requires exchangeability of calibration and
future complete-pipeline errors. It is not conditional 95% coverage for every
fitted calibration cohort, simultaneous coverage of all axes, or a guarantee of
transfer across motions, rigs, temperatures, or real sensors. Matching scope
hashes is necessary in this implementation, but does not establish exchangeability.

The interval artifact records calibration seeds, input report and result digests,
method source hashes and settings. Evaluation rejects overlapping seeds,
overlapping result artifacts, the same input report, and incompatible pipeline
scope or axes. Scope binds producer, duration, map rounds, exact numerical source
hashes and eligible axis set. This is a conservative compatibility check.
Evaluating a model never refits its widths. Failures remain in all-attempt
coverage denominators. The original Gaussian intervals and a 0.1-width control
are reported alongside the empirical intervals; width is visible as well as
coverage. Per-axis Wilson intervals are not simultaneous significance tests.

## Gravity-frame erratum

The bag solver builds an odometry-local world frame. The synthetic generator's
world gravity `(0, 0, -9.81)` is in a different frame; the rig is already tilted
at the first retained scan. Comparing those components directly is invalid.
The prior P0 gravity undercoverage rows therefore do **not** demonstrate covariance
undercoverage. They are superseded by this correction, not silently removed.

There is no recorded independent transform, with uncertainty, aligning the fitted
world to the generator world. Gravity x/y/z are consequently excluded from new
synthetic-bag collection and empirical interval models. Even apparently perfect
gravity coverage would not make those rows admissible. Known-plane experiments
retain gravity because their plane map defines a common world frame.

Sensor-relative mounting rotation, lever arm, IMU biases and clock offset do not
change with this world gauge. Their errors, formal std and calibration widths
are unchanged. Historical raw reports retain their original gravity rows for
traceability; the interval builder explicitly excludes them. The evidence
archive includes a machine-readable `gravity-frame-erratum.json`.

## Reproduction

Install the checkout and run:

```bash
python tools/empirical_intervals.py fit \
  docs/assets/empirical_intervals/calibration/coverage.json \
  --output work/interval-model.json
python tools/empirical_intervals.py evaluate \
  docs/assets/empirical_intervals/evaluation/coverage.json \
  --model work/interval-model.json --output work/interval-evaluation.json
calibrex validate work/interval-model.json
calibrex validate work/interval-evaluation.json
```

Output paths must be new. `--nominal` belongs only to `fit`; evaluation uses the
frozen model. The input report's `protocol.json` and result files must be beside
it; result file digests are verified before fitting or evaluating. Both artifacts
are registered in `calibrex schema all` and `calibrex validate`.

The raw cohorts were produced with `tools/evaluate_p0_uncertainty.py` from base
commit `e19f9b6c`, with `--producer synthetic_bag`, 8 seconds and four map rounds.
Calibration seeds are 43001–43020, evaluation seeds 44001–44010. These are fresh
noise draws relative to P0, on the same fixed synthetic motion and truth. Data
collection ran while the method was implemented; evaluation errors were first
scored after the first model had been saved. This is development evidence, not
a pre-registered independent audit. The gravity-frame correction was made after
that first scoring and replayed without retuning eligible widths. The original
16-axis model and evaluation remain under `superseded_gauge_mismatch/`.

## Outcomes and limits

Twenty calibration draws produce finite radii for 13 sensor-relative axes. On
ten separate evaluation draws, empirical intervals contain truth 8–10 times
per axis. For example:

| Axis | Formal interval hits | Empirical interval hits | Empirical half-width |
| --- | ---: | ---: | ---: |
| mounting rotation z | 0/10 | 10/10 | 0.136 deg |
| lever arm y | 0/10 | 10/10 | 1.79 cm |
| clock offset | 6/10 | 10/10 | 0.624 ms |
| gyro bias x | 0/10 | 8/10 | 0.000553 rad/s |
| gyro bias z | 7/10 | 8/10 | 0.000304 rad/s |
| accel bias y | 8/10 | 8/10 | 0.00918 m/s² |

The three 8/10 cases have a Wilson upper bound below 0.95 and retain the
`undercoverage` label. The others remain `inconclusive`; 10/10 is not proof of
95% coverage. Correlated axes must not be pooled into 130 independent trials.
No radius or target was adjusted using these evaluation outcomes.

The data support a reproducible interval-construction workflow and a narrower
statement of where it helped on this synthetic cohort. They do not establish a
production uncertainty model. In particular, width can reflect systematic error;
it does not repair that error. Before adoption, use more calibration recordings,
varied excitation/noise/motion, independently aligned ground truth, and a fresh
frozen evaluation cohort. The current evaluation data are now spent development
data. The existing stationarity, observability and quality gates remain required.

Complete inputs/results, frozen models, evaluator source snapshots and a SHA-256
manifest are under [the evidence archive](../assets/empirical_intervals/). Numerical
producer hashes match the recorded base commit; no large raw bags are committed.
