# Structured CPU spline normal systems

Three alternating fresh-process pairs use the same frozen runtime v0.3 source
(`8801fa74f52560d80b077041e4de92d7a9dd7f6b`) and the same captured public GLIM-derived fitting problem.
Every worker performs a complete first fit and then a complete warm fit.
All residuals, LM iterations, sparse solves and final marginal covariance are
included. NumPy remains the default; `numba_cpu` selects the compiled path.

| Complete fit scope | NumPy median | Structured compiled median | Observed ratio |
| --- | ---: | ---: | ---: |
| First use, empty JIT cache | 18.550 s | 20.963 s | 0.885x |
| After the worker's own first fit | 33.047 s | 3.180 s | 10.391x |

All six complete fitted-output comparisons pass the predeclared `rtol=1e-8`,
`atol=1e-9` gates. Status, 22 iterations, dimensions and parameter order agree
exactly. Every trajectory, calibration block, covariance, cost and residual
variance is retained. Equality is numerical, rather than byte identity.

Plane normal products are accumulated in seven control blocks, using the
existing four-worker pool. IMU products use a serial compiled kernel and
preserve control/calibration cross terms. Aligned IMU samples share states,
control derivatives and clock probes within each linearization. The final
covariance uses the complete assembled Jacobian and original Schur calculation.
FP64 arithmetic, `fastmath=false`, samples and iteration budgets are unchanged.

The fixed problem has 28,563 plane returns and 1,550 rows per IMU family.
Its extraction helper was unversioned. Input hashes identify the exact consumed
NPZ/metadata; the protocol names the original public bag hash. The raw arrays
are external. This cohort measures fixed fitting, excluding bag decoding,
odometry, mapping, association and evidence controls. It provides neither
whole-bag acceleration nor independent extrinsic accuracy evidence.

Workers verify imported source bytes and use one BLAS/OMP/MKL thread.
Each compiled worker starts with an empty private Numba cache. Host resources
were unreserved and timings vary substantially between pairs. The archive
retains every attempt, including that variation. These observations must not
be multiplied by ratios from earlier cohorts to infer an incremental gain.

```bash
python -m tools.audit_bspline_planes docs/assets/numba_bspline_banded
```

The CPU-only audit verifies schemas, byte/source bindings, backend versions,
complete decisions/numeric arrays and every median/ratio.
See [usage and limits](../../benchmarks/numba_bspline_planes.md).
