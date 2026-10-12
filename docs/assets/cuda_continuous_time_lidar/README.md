# CUDA continuous-time LiDAR finite differences

The optional backend computes all thirteen finite-difference evaluations on CUDA.
Standalone residuals, trajectory/world coordinates, associations and ordered normal equations
stay on CPU. Native selected timestamp/point rows are reused; small capture budgets retain
CPU execution without importing CuPy. FP64 arithmetic retains exact complete results.

## Final complete-process comparisons

Five fresh paired subprocesses per workload, alternating order. Initially empty independent
compiler caches, first runs, outliers and slower cases are all included. Wall covers imports,
initialization, compilation, replay, transfers, refinement, diagnostics, hashes and saves.

| Public GLIM workload | Train / holdout targets | CPU median | Candidate median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|---:|
| 120 frames, up to 400 points/frame | 38,400 / 9,600 | 14.034531 s | 12.313007 s | 1.140x | 1.140x |
| Default sampling caps (CPU dispatch) | 6,000 / 3,200 | 8.817494 s | 8.487755 s | 1.039x | 1.009x |
| 120 frames, up to 8,000 points/frame | 768,000 / 192,000 | 169.696051 s | 97.910917 s | 1.733x | 1.470x |

Small budgets execute on CPU and add no GPU work. The standard captured population
has limited whole-process gains. Kernel-only timing is not an end-to-end speedup.

| Workload | API medians (CPU -> candidate) | Solver medians | All-thread CPU medians |
|---|---:|---:|---:|
| 120 frames, up to 400 points/frame | 11.211649 -> 9.176215 s (1.222x) | 8.642711 -> 5.915977 s (1.461x) | 21.109375 -> 16.375000 s (1.289x) |
| Default sampling caps (CPU dispatch) | 5.215400 -> 5.138731 s (1.015x) | 3.704360 -> 3.589994 s (1.032x) | 10.062500 -> 10.062500 s (1.000x) |
| 120 frames, up to 8,000 points/frame | 166.174938 -> 95.117275 s (1.747x) | 130.850360 -> 65.553647 s (1.996x) | 273.812500 -> 147.656250 s (1.854x) |

### 120 frames, up to 400 points/frame

Status `max_iterations`, rank 6; retained train/holdout correspondences 29,181/3,451. These statuses are preserved.

| Pair | Order | CPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | CPU -> candidate | 31.715202 s | 20.811603 s | 1.524x |
| 2 | candidate -> CPU | 14.034531 s | 12.313007 s | 1.140x |
| 3 | CPU -> candidate | 14.477672 s | 13.031414 s | 1.111x |
| 4 | candidate -> CPU | 13.482300 s | 10.800248 s | 1.248x |
| 5 | CPU -> candidate | 12.968461 s | 11.998809 s | 1.081x |

[full_capture definition](full_capture.definition.json), [aggregate](full_capture.benchmark.json), [complete candidate measurement](full_capture.sparse.repeat-01.measurement.json), [saved result](full_capture.sparse.repeat-01.result.yaml).

### Default sampling caps (CPU dispatch)

Status `converged`, rank 6; retained train/holdout correspondences 4,773/2,526. These statuses are preserved.

| Pair | Order | CPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | CPU -> candidate | 20.169513 s | 22.290304 s | 0.905x |
| 2 | candidate -> CPU | 7.609956 s | 10.618622 s | 0.717x |
| 3 | CPU -> candidate | 10.065022 s | 7.677320 s | 1.311x |
| 4 | candidate -> CPU | 8.817494 s | 8.205857 s | 1.075x |
| 5 | CPU -> candidate | 8.561621 s | 8.487755 s | 1.009x |

[default_budget definition](default_budget.definition.json), [aggregate](default_budget.benchmark.json), [complete candidate measurement](default_budget.sparse.repeat-01.measurement.json), [saved result](default_budget.sparse.repeat-01.result.yaml).

### 120 frames, up to 8,000 points/frame

Status `max_iterations`, rank 6; retained train/holdout correspondences 582,119/67,602. These statuses are preserved.

| Pair | Order | CPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | CPU -> candidate | 169.696051 s | 93.553613 s | 1.814x |
| 2 | candidate -> CPU | 143.887811 s | 97.910917 s | 1.470x |
| 3 | CPU -> candidate | 180.100372 s | 91.711999 s | 1.964x |
| 4 | candidate -> CPU | 136.738592 s | 107.858087 s | 1.268x |
| 5 | CPU -> candidate | 189.933579 s | 161.983742 s | 1.173x |

[dense_capture definition](dense_capture.definition.json), [aggregate](dense_capture.benchmark.json), [complete candidate measurement](dense_capture.sparse.repeat-01.measurement.json), [saved result](dense_capture.sparse.repeat-01.result.yaml).

## Actual GPU work

The separate guarded diagnostic records 129 GPU linearizations over 3,782,178 correspondence rows, each retaining thirteen evaluations. It records 207 CPU standalone residual calls and 129 ordered CPU normal calls. There are 26 observation uploads (103,523,608 bytes), with no CPU linearization fallback.

The diagnostic is instrumented, excluded from formal timings, and matches the complete
formal result. Source/input/HEAD guards pass.

[Diagnostic](diagnostic.json), [measurement](diagnostic.measurement.json), [launcher](diagnostic-launcher.py.txt).

## Preserved preceding attempts

Initial GPU standalone residuals produced only 1.052x whole-process gain for the standard
population and regressed default caps to 0.807x. Sharing compiled kernels and keeping small
budgets on CPU removed that default regression. Single GPU residual evaluations remained
more expensive than the fused CPU path. The final policy transfers only retained MAD batches.

- [Initial candidate](attempts/v1/publication-index.json): 20 complete measurements plus diagnostic.
- [Shared-kernel candidate](attempts/v2/publication-index.json): 20 complete measurements, diagnostic,
  four completed dense measurements and an interrupted third CPU attempt.
- [Prepared timestamps](attempts/v3/full_capture.definition.json): 20 complete standard/default
  measurements, one complete dense CPU run and an interrupted GPU attempt.

Stopped dense cohorts are partial research evidence. Complete walls in stop records retain
six-decimal stdout precision; interrupted attempts retain elapsed/CPU lower bounds and have
no completed results. Preceding attempts are not pooled with the final comparison. The v2
archived index retains its original initial/ namespace; identical files are in attempts/v1/.

## Reproduction and provenance

Measured source `fb265b33cf0752e4ff9829f2c798b416da5a2fd0`; baseline `eb50c99878706a45eb0e584f3573507313cdc172`. Publication also adds docs/evidence.

```bash
pip install -e ".[dev,cuda,numba]"
python tools/benchmark_adaptive_plane_search.py \
  --baseline-scope compiled \
  --baseline-commit eb50c99878706a45eb0e584f3573507313cdc172 \
  --candidate-query-backend cuda \
  --config examples/public_datasets/glim_versatile/online_continuous_time_compiled_dense_capture_config.yaml \
  --repetitions 5 \
  --output-dir outputs/public-glim-cuda-continuous-dense-reproduction
```

Use the public-dataset tutorial to download/checksum-verify the GLIM bag/ground-truth trajectory
and perform the declared restamping/copied-topic conversion. This identity control supplies
execution parity, without independent two-sensor extrinsic/clock accuracy evidence.
Captures, temporal split, robust weights and profile/solver budgets are fixed within each pair.

The sole candidate execution override is recorded with the shared physical configuration.
Every saved/API calibration value, count, status, trace and provenance field is retained.
Comparison normalizes path separators and excludes only the exact execution note.
Portable measurements change only loaded-module locations; definitions also change command
and source locations. Original digests and changed fields are recorded in the manifest.

[Index](publication-index.json), [conversion lineage](conversion-provenance.json),
[publication launcher](publication-launcher.py.txt). Source archives retain exact checkout bytes;
committed-code checks normalize source line endings only.
