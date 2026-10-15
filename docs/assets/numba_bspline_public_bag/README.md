# Complete public GLIM bag comparison with compiled CPU planes

Three alternating fresh-process pairs compare NumPy and `numba_cpu` fitting
on the first six seconds of the original public AIST GLIM Livox recording.
Both variants use the same frozen runtime revision and the native CPU
odometry path. The compiled plane evaluator uses up to four independent CPU
row slices. Every compiled worker starts with an empty private Numba cache;
the first-use module import/JIT cost remains inside the measurement.

| Scope | NumPy median | Compiled median | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete fresh worker | 86.076 s | 60.834 s | 1.415x |
| Four fits, including covariance and first-use JIT | 24.771 s | 14.554 s | 1.702x |

| Pair | Variant | Complete worker seconds | Fit seconds |
| --- | --- | ---: | ---: |
| 0 | prior | 57.740046 | 17.346838 |
| 0 | current | 60.834188 | 14.553727 |
| 1 | current | 60.704879 | 13.466180 |
| 1 | prior | 87.605110 | 24.770903 |
| 2 | prior | 86.076258 | 29.968840 |
| 2 | current | 108.814551 | 24.412813 |

The complete worker includes imports, raw bag decoding, initial angular-rate
calibration and odometry, gyro-deskewed odometry, four map construction and
association rounds, all fits/covariances, noise updates, held-out evaluation,
32 signed controls, input hashes, result saving and verified frozen imports.
Source freezing in the parent is outside the worker scope. These are
observational timings under variable host conditions. An earlier complete cohort gave the opposite whole-worker ratio (0.745x). These outcomes do not establish reliable whole-pipeline acceleration. CPU execution time is also reported
when the protocol declares it; it can still vary with CPU frequency. Sampling,
iteration budgets and held-out checks are unchanged; geometry and returned
motions are not implicitly quantized. BLAS/OMP/MKL use one thread. Host CPU
resources are unreserved, and ratios describe this six-second workload.

All three complete evidence comparisons pass. Policy, block/control,
separability and convergence decisions agree. All twelve paired full fit
records preserve status, iterations, dimensions and calibration parameter
order and pass the fixed `rtol=1e-8`, `atol=1e-9` checks for trajectories,
all six calibration blocks, covariance, costs and residual variance. These
are numerical-agreement checks, rather than independent accuracy evidence.

Both methods retain an **INCONCLUSIVE** verdict on this capture. The map is
estimated from the trajectory being fitted, and formal covariance holds it
fixed. The supplied GT is not consumed. These results do not establish
absolute extrinsic accuracy or a general speedup on other recordings.

Measured source: `db4566c4a667c89bb9e1277557073658ee37cb00`. Original bag SHA-256:
`f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643`. `/livox/points` uses the per-return `t` offset;
`/livox/imu` reports acceleration in g. The raw recording is external and
is not redistributed. Each worker retains the complete schema-valid
calibration result and every per-round fitted array. Source manifests and
tar archives bind the actual imported bytes; frozen comparison helpers and
the publication index bind all reported decisions and ratios.

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/numba_bspline_public_bag
python -m tools.benchmark_bspline_lidar_imu \
  --baseline-commit db4566c4a667c89bb9e1277557073658ee37cb00 \
  --current-fit-backend numba_cpu --fit-workers 4 \
  --max-seconds 6 --max-rounds 4 --repeats 3 \
  --output outputs/compiled-plane-public-bag
```

The archive audit runs with CPU dependencies and does not require the
recording, Numba or CUDA. It replays the frozen physical and per-fit quality
checks and recomputes all medians from the six complete workers. See
[implementation and scope](../../benchmarks/numba_bspline_planes.md).
