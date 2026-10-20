# Reuse in sparse screw interpolation

The native `screw_linear` fitter prepares each fixed knot interval once per
objective or Jacobian evaluation. Factors at the same interpolation fraction
share a pose and its knot Jacobians; accelerometer factors also share repeated
discrete body-motion samples. An evaluation owns its workspace and discards it
before any state update. A pose-only evaluation never prepares SE(3) Jacobians.

The interpolated pose and its derivative share the same exponential. Fixed
3x3 block placement replaces general nested matrix assembly, and geometric
RMSE scoring skips discarded derivatives. This removes repeated relative-pose
logarithms, interval Jacobians and matrix inverses. The original FP64 multiplication order, sampled kinematics, clock
differences, weighted reductions, sparse solve, damping, stopping criteria,
covariance and serialized schemas stay unchanged. Interpolation keys preserve
the sign of zero. Shared internal arrays reject writes. Public function
signatures and standalone interpolation behavior are unchanged.

This is an additional execution improvement over the maintained source after
the accelerometer, gyro, information-prior and final-gradient corrections.
Its ratios must not be multiplied by the
[earlier residual-evaluation measurements](screw_residual_evaluation.md).

## Frozen comparison

The baseline is `297acf16310500c1a6644a584e15e7b9ec28cf27`; the measured
candidate is `c017754233f22bdd921f0b6f76a9d333bbc1f686`. All 401 Python
modules from each source are retained. Only `continuous_time_sparse.py` and
`se3_manifold.py` differ in the runtime source capsules. Both sources and the
measurement tools were committed before timing.

The unchanged benchmark runs six fresh workers per case, ordered
baseline/candidate, candidate/baseline, baseline/candidate. It uses one BLAS
thread, zero warmups, and an unreserved host. The declared adoption gate is
2x on the median of the three paired complete API ratios in **each** case.
No iteration limit, tolerance, precision or quality gate is adjusted.

The synthetic LiDAR–IMU case uses the same already-spent seeds 20261010,
20261011 and 20261012. Its full API includes trajectory and calibration
fitting, covariance, interleaved holdout, separability and all 32 signed
known-bad controls. The public GLIM case uses the same 1,013 fixed planes
from eight already-spent Livox Avia registration captures, five perturbed
knots, and a 30-iteration limit. Its assigned four-second timestamps and
capture-local plane associations form a performance fixture; they do not
reconstruct a recording's physical trajectory or an end-to-end bag calibration.

API time excludes imports, source verification, input preparation and result
serialization. Fresh-worker time includes those tasks, process startup and
exit. Parent-side preparation of the fixed geometry and source capsules is
outside both clocks. No large dataset is downloaded for this comparison.

## Results and retained first attempt

| Scope | Baseline median | Candidate median | Median paired speedup |
| --- | ---: | ---: | ---: |
| Synthetic LiDAR–IMU API | 53.169 s | 22.663 s | 2.395x |
| Synthetic LiDAR–IMU, fresh worker | 60.000 s | 32.534 s | 1.923x |
| Public geometry fit API | 30.055 s | 10.518 s | 2.765x |
| Public geometry fit, fresh worker | 34.474 s | 14.858 s | 2.271x |

The ratio column is the median of the three pair ratios, not the quotient
of the separately reported timing medians. Both declared complete-API gates
pass. Fresh-worker ratios are separate observations; the adoption gate applies
to the complete API and does not establish a 2x gate for startup overhead.

Joint API paired ratios range from 2.159x to 2.690x.
Public geometry API paired ratios range from 2.679x to 2.858x.
Every complete input/options record and full fitted/output record agrees
exactly within each pair, excluding provenance only. Equality includes
trajectory, calibration, covariance, final gradient, stopping state,
holdout, separability and all controls. All three joint cases converge
in 21, 23 and 29 iterations and pass the existing policy with 32/32 controls
detected. All six public geometry fits converge in 15 iterations.

The first frozen candidate, `2e5da540f1b81b22d8dff900bb1201289fa4a8f7`,
measured 2.225x for joint recovery and 1.900x for public geometry.
The geometry case **missed the unchanged 2x gate**. This was not adopted
as a passing comparison. A complete-fit diagnostic profile, with exact
input/output parity, then identified remaining duplicate exponentials,
general matrix assembly and discarded RMSE Jacobians. The revised source
was committed and the same cohort and gates were frozen again before timing.
Both complete twelve-worker cohorts, all logs and the diagnostic profile
are retained. This follow-up selection is post hoc development.

See the [complete archive inventories and limits](../assets/screw_interpolation_reuse/README.md).

## Reproduce and audit

With the eight original spent registration captures and their hash receipts:

```bash
python tools/benchmark_screw_residual_evaluation.py \
  --baseline-commit 297acf16310500c1a6644a584e15e7b9ec28cf27 \
  --geometry-folder outputs/odometry-search-profile-v3 \
  --output outputs/screw-interpolation-comparison
```

The command freezes the current committed candidate; use the measured
candidate commit above to reproduce that source. Always choose a new output
directory. Failed observations and logs remain in that directory.

Replay the complete published records without the original recording,
calibration execution or a GPU:

```bash
python tools/audit_screw_residual_archive.py \
  docs/assets/screw_interpolation_reuse/complete-evidence.tar.gz
```

The auditor checks the bounded archive index, both source manifests, actual
imported-module hashes, all input/result/log bindings and closed record
schemas, then recomputes exact complete pair equality, medians and gate
decisions. It neither imports Calibrex nor executes archived Python source.

The 62 new maintained checks include an independent copy of the previous
Jacobian expression, small and near-pi rotations, endpoint and extrapolated
fractions, signed zero, clamped and boundary-crossing accelerometer stencils,
mounted/unmounted gyro windows with nonunit scales, immutable shared arrays,
fixed-versus-general matrix assembly, derivative-free geometric RMSE,
and complete fits through successive state updates with two damping values.
The 210-check targeted run also covers existing residual, gyro,
accelerometer and final-gradient behavior. An initial lazy-pose test wrongly
blocked the 3x3 inverse already used by the existing Log; its failed output
and corrected checks are retained with the successful cohort.

These are spent-data development measurements. They establish neither new
calibration accuracy nor a SOTA claim. Reliability under a fixed compute
budget still needs a separately frozen claim, protocol and unspent data.
