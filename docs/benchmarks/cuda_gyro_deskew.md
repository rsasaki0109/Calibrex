# GPU gyro deskew: rejected after whole-pipeline measurement

The prototype improves whole-worker medians from **68.414 s to 65.699 s
(1.041x, 4.0% shorter)** across three alternating pairs. This falls short of the
substantial end-to-end improvement sought here. Its runtime changes are reverted;
the measured code, device tests and complete results remain in the
[frozen evidence archive](../assets/cuda_gyro_deskew/README.md).

The public GLIM Livox workload covers the first six seconds and four map rounds.
All 12 full fitted trajectories, covariances and costs, holdout values and all 32
signed controls per pair match exactly. There is no reduction in point, neighbor,
iteration or validation budgets. Independent extrinsic truth is unavailable, so
the real-data verdicts remain INCONCLUSIVE.

The trial moves per-point gyro rotations to FP64 CUDA kernels, caches gyro
prefixes on the device and reuses rotations within each scan's deskew rounds.
Native repairs preserve rounding near geometry grid boundaries. Each worker
executes 392 device rotation calls over 9,408,000 point rows, 784 device translation
calls and 606,953 native point repairs. GPU execution alone does not demonstrate
a useful whole-pipeline improvement.

The benchmark now measures complete odometry, including gyro work, preprocessing,
target construction, normal preparation and all registration calls. The prior
`__call__` preprocessing timer excludes the trial's new route; the auditor checks
that distinction and does not present its ratio as total preprocessing speed.
The unchanged fit's timing also varies, so these three pairs do not establish a
precise causal gain for every workload.

No foreign or unknown CUDA compute processes appeared in 43 observations at
ten-second intervals. This does not guarantee continuous GPU exclusivity or
reserve CPU resources. Workers use fresh processes, one BLAS/OMP/MKL thread and
separate initially empty compilation caches. Whole timing includes startup,
decoding, compilation, calibration, covariance, controls and saved validation.

Audit the source/result/observer bindings without a GPU or the input bag:

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_gyro_deskew
```

Reproducing the rejected prototype requires a clean checkout of measured commit
`6d5db689ade25af5b6f69de9f960f684449ce7eb`, the public input identified in the
archive protocol and the frozen observer/launcher. The active runtime does not
contain this prototype. The previous comparison source is
`9cf33f996aaaffaf40ab053fc6b3ad1d5f94f46d`.

Further optimization should target complete odometry and fit costs together.
Small isolated kernel improvements cannot provide a large pipeline reduction
when substantial work remains in decoding, setup and unchanged fitting.
