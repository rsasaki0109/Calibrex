# Fused CPU B-spline plane evaluation

Three alternating fresh-process NumPy/compiled pairs evaluate the same fixed
public GLIM-derived problem twice per worker. Each full fit includes all LM
iterations, sparse solves and marginal covariance. The first fit includes JIT
import and compilation; the second is the warm measurement.

| Scope | NumPy median | Compiled median | Observed speedup |
| --- | ---: | ---: | ---: |
| Full fit, first use with empty JIT cache | 27.068 s | 28.776 s | 0.941x |
| Full fit after that worker's own prime | 31.166 s | 11.176 s | 2.789x |

All six complete fitted-output comparisons pass the predeclared `rtol=1e-8`,
`atol=1e-9` gates. Status, iterations, residual/parameter counts and calibration
parameter order agree exactly. All fits converge in 22 iterations. The complete
trajectory, six calibration blocks, costs, covariance and residual variance are
retained. This is floating-point agreement, not byte identity.

Measured source: `55ea7f105d500d8ad0254bb5a7f2c1dc5a001ecb`. Both variants use this same frozen source tree and
different explicit backend options. Workers verify every imported Calibrex
source file against the archived manifest. BLAS, OMP and MKL use one thread;
the compiled plane evaluator uses four independent nogil row slices. Resources
were unreserved, so these ratios are observations for this workload.

The fixed input has 28,563 plane returns and 1,550 samples per IMU family. It was
captured locally from a public AIST GLIM Livox run; its extraction helper was
unversioned. Input NPZ and metadata hashes bind the consumed problem, and the
protocol identifies the original bag hash. Extraction and the full bag pipeline
are not reproduced by this cohort. Raw sensor inputs are not redistributed.
This archive provides no independent calibration truth or whole-bag speedup.

The small SO(3) workspaces are allocated once per row slice, rather than once
per return. Plane residuals and their 24 local Jacobian columns share the
rotation evaluation. IMU factors, sparse solving and covariance retain their
native implementations; sampling, iterations and calibration budgets are
unchanged. The default backend remains NumPy.

```bash
python -m tools.audit_bspline_planes docs/assets/numba_bspline_planes
```

The audit validates schemas and hashes, checks executed source bindings and
recomputes every quality comparison, median and reported ratio. See
[usage and reproduction](../../benchmarks/numba_bspline_planes.md).
