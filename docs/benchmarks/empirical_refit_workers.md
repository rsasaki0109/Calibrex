# Faster empirical Camera–LiDAR uncertainty

Empirical uncertainty repeatedly refits the same calibration on seeded temporal
block samples. `--workers` runs independent refits in separate CPU processes:

```bash
calibrex camera-lidar empirical-uncertainty correspondence.json problem.json \
  --result-dir outputs/refits --output outputs/uncertainty.yaml \
  --block-length 3 --resample-count 20 --workers 4 --stability-only
```

The default remains one process. The Python API accepts `workers=4` through
`run_empirical_se3_uncertainty`. Scripts calling that API with multiple workers
must protect their entry point with `if __name__ == "__main__":` because workers
use Python's spawn start method on every platform. The CLI handles this already.
Each process receives the input snapshot once. Concurrency is capped at the
number of refits, and at most twice that concurrency is queued. Choose a count
appropriate for available CPU and memory; each process holds its own input and
solver state. Small workloads can be faster with the default serial execution.

Workers inherit the caller's BLAS thread settings. To prevent nested BLAS
oversubscription, the comparison sets `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`
and `OMP_NUM_THREADS` to `1` before starting Python. The library leaves those
environment settings to the caller.

## Preserved calibration behavior

The same iteration index determines each refit's RNG seed, block split and
correspondence bootstrap regardless of completion order. Results are collected
in their original order. Every refit retains its full trace, selected and
candidate poses, holdout, acceptance and failure status. Failed refits retain
their artifacts; worker exceptions propagate without a hidden serial retry.
The parent captures source provenance once and binds every saved child output
to the final uncertainty artifact with SHA-256.

The probabilistic pose evaluator also calculates radial-distortion powers with
NumPy's float64 `float_power` loop, removing per-element Python dispatch. This
retains the platform `pow` operation used by Python float exponentiation.
Extreme finite radii use the original scalar powers to preserve overflow
exceptions. Projection formulas, reductions, confidence gates, evaluation
budgets, covariance, interval construction and coverage controls are unchanged.
The scalar evaluator and small-input fallback remain available. Tests compare
power bytes across wide, strided, read-only and nonfinite inputs, and complete
pose evaluations across camera models and uncertainty ablations.

This is CPU acceleration using existing dependencies. The public runtime
schemas and their versions are unchanged; the worker option is recorded in
command provenance.

## Complete-process comparison

Three alternating pairs use the existing execution control from nine public
KITTI raw scans in `2011_09_30_drive_0018_sync`. It retains 1,024 points per
scan, or 9,216 points in total. The camera correspondences are generated from
declared pinhole, MEI and double-sphere projections with fixed noise. This
control measures execution and numerical parity. It does not establish
real-image calibration accuracy or empirical coverage. `--stability-only`
ignores the injected reference, and all six policies remain inconclusive.

Both variants perform twenty seeded refits over three-frame blocks, with the
same 400-evaluation limit for each refit. Timings include fresh Python startup,
imports, input validation, process initialization, all solver evaluations and
diagnostics, twenty YAML outputs, final uncertainty saving and pool shutdown.
Source checks and output auditing run outside that timing. No timing warmup or
failed worker is removed. The host was unreserved.

| Complete empirical uncertainty CLI | Median seconds |
| --- | ---: |
| Previous implementation, one process | 29.935852 |
| Current implementation, four processes | 12.643885 |
| Observed ratio | **2.367615x** |

| Repeat | First variant | Previous seconds | Current seconds |
| --- | --- | ---: | ---: |
| 0 | previous | 28.047527 | 14.412418 |
| 1 | current | 29.935852 | 11.293290 |
| 2 | previous | 31.793272 | 12.643885 |

Every run retains exactly **6,224 trace entries** across twenty refits. All
120 complete refit results and six final uncertainty results match exactly.
Only timestamps, source commit metadata, declared output paths, the worker
strategy flag and already verified child-output hashes are normalized.
No numeric result, diagnostic, interval or solver status is normalized.

The measured baseline is `dd29d3cc554786a333bf9da5a09e012d5cf1ba45` and the
candidate is `ed83117aab745514c21b5d2e4dd049a9a79020e7`. The environment uses
Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1, Pydantic 2.12.5 and PyYAML 6.0.3.
The ratio combines array powers and process refits; it does not isolate either
optimization or establish a portable speed guarantee. Separate exploratory
single-pair probes are not pooled with these six fresh-process measurements.

The [indexed evidence archive](../assets/empirical_refit_workers/README.md)
retains every result, timing, input identity, source capsule and parent/child
import receipt. The initial Git compatibility failure occurred before any timed
worker. A later summary error occurred after all six successful timed workers;
the archive retains that error and the corrected postprocessing without reruns
or changes to the measurements.

## Reproduce and audit

Obtain the nine original scans identified by the archived fixture protocol
under KITTI's upstream terms. The builder uses scans present in the supplied
directory, so use those nine input identities for this workload. The original
builder is retained as `fixture-builder.py.txt`; the current tool keeps the same
fixture construction. Rebuilt input hashes can differ with output paths,
creation timestamps and source provenance.

```bash
python -m tools.benchmark_probabilistic_refinement \
  --build-kitti-fixture path/to/2011_09_30_drive_0018_sync \
  --points-per-frame 1024 --output-dir outputs/projected-control
python -m tools.benchmark_empirical_refits \
  --baseline-commit dd29d3cc554786a333bf9da5a09e012d5cf1ba45 \
  --fixture-dir outputs/projected-control --workers 4 --repetitions 3 \
  --output-dir outputs/empirical-refit-comparison
python -m tools.benchmark_empirical_refits \
  --audit docs/assets/empirical_refit_workers
```

Use fresh output directories and the recorded candidate runtime to reproduce
the comparison. The frozen launcher in the archive records the original run;
the current tool also handles an omitted optional reference during summary
generation. Auditing the published archive needs only ordinary CPU dependencies,
without the raw scans, generated inputs or optional accelerator packages.
