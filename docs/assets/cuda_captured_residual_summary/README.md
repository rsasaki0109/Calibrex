# Captured CUDA residual summaries: experimental, disabled by default

Moving already-captured residual summaries to CUDA did not establish a
reliable whole-pipeline gain. The dense ratio of medians is 0.914x
(about 9% slower), while its median paired ratio is 1.130x.
Keep the existing ordered CPU summary route by default. The direct Python
API permits explicit experiments; configuration and result schemas stay unchanged.

```python
from calibrex.solvers.continuous_time_lidar_pair_solver import ContinuousTimeLidarPairOptions

options = ContinuousTimeLidarPairOptions(
    query_backend="cuda",
    cuda_captured_summary=True,  # Experimental; benchmark the complete workload.
)
```

## Computation and ownership

Only native factors with at least 8,192 observations can use the CUDA
summary route, and only after finite differences have captured that exact
immutable observation payload. The summary never captures or uploads new
observations. An exact transform cache match borrows read-only host residuals;
otherwise the existing FP64, no-FMA residual kernel evaluates the captured
device arrays. Ordered CPU summation retains CPython 3.10/3.11 plain addition
and CPython 3.12 compensation, followed by Python division and square root.
Unsupported runtimes, uncaptured data and initial/holdout summaries retain
the existing CPU route. Small batches do not import CUDA for this operation.

Fresh summaries still upload a 56-byte transform and download residual
values. This avoids extra observation uploads, not all transfers.
Owned streams and locks retain caller stream state and serialize shared
contexts. Cached finite differences are not evicted by summary requests.
Forks remain empty; public residual and linearization outputs remain owned.
Public hooks, robust loss, diagnostics and optimizer histories are preserved.

## All whole-process observations

Three alternating pairs each for dense and full capture, plus five for
default sampling, use independent initially empty persistent Numba/CuPy
caches per workload and method. Every first compilation, cold observation
and slower trial is retained. Whole wall includes startup, imports, replay,
all candidates/refinements, transfers, diagnostics, hashes and saves.
Parent audits are excluded; OS caches persist. No current CPU-only cohort
was run. These cohorts are not pooled or multiplied by earlier ratios.

| Workload | Baseline whole median | Experimental whole median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|
| 960,000 captured points | 35.314477 s | 38.644438 s | 0.914x | 1.130x |
| 48,000 captured points | 19.251234 s | 17.538761 s | 1.098x | 1.098x |
| Default sampling | 7.490853 s | 6.837300 s | 1.096x | 0.982x |

Default sampling never runs the new CUDA summary. Its 1.096x ratio of
medians and 0.982x median paired ratio do not establish a consistent gain
or resolve the preceding small-batch timing variation.

| Workload | Pair | Order | Baseline whole wall | Experimental whole wall |
|---|---:|---|---:|---:|
| dense | 1 | baseline, experimental | 49.512418 s | 43.797816 s |
| dense | 2 | experimental, baseline | 34.254949 s | 29.544137 s |
| dense | 3 | baseline, experimental | 35.314477 s | 38.644438 s |
| full | 1 | baseline, experimental | 19.251234 s | 17.538761 s |
| full | 2 | experimental, baseline | 8.675902 s | 7.592925 s |
| full | 3 | baseline, experimental | 22.847606 s | 22.517647 s |
| default | 1 | baseline, experimental | 19.181053 s | 16.474535 s |
| default | 2 | experimental, baseline | 8.585596 s | 10.154833 s |
| default | 3 | baseline, experimental | 7.490853 s | 6.458327 s |
| default | 4 | experimental, baseline | 6.638537 s | 6.760881 s |
| default | 5 | baseline, experimental | 6.700437 s | 6.837300 s |

All 22 complete formal results match after path separator normalization
and exclusion of strictly validated execution identity notes. Full API
results, unchanged saved YAML bytes, histories and diagnostics are retained.
Dense sampling uses 768,000 train and 192,000 holdout targets; full capture
uses 38,400 and 9,600. Both finish at rank 6 with `max_iterations`.
Default sampling uses 6,000 and 3,200 and finishes at rank 6 with `converged`.

## Actual work outside formal timing

The independent dense diagnostic records 102 additional GPU residual
kernel launches over 59,724,347 rows. Its GPU summary cache hits are zero;
cache borrowing is covered separately by actual-device tests.
CPU full-observation summaries decrease from 154 to 52 calls, retaining
16,950,268 CPU rows; GPU values are reduced in order on CPU.
The diagnostic exactly matches the first formal experimental result.

| Execution counter | Count |
|---|---:|
| `gpu_linearization_calls` | 56 |
| `gpu_linearization_requests` | 129 |
| `initial_rank_calls` | 26 |
| `initial_rank_compiled` | 26 |
| `gpu_linearization_cache_hits` | 73 |
| `max_per_context_cache_bytes` | 33,398,792 |
| `gpu_linearization_rows` | 32,768,988 |
| `gpu_residual_calls` | 0 |
| `gpu_residual_rows` | 0 |
| `gpu_payload_uploads` | 26 |
| `gpu_payload_bytes` | 2,066,730,664 |
| `cpu_linearization_calls` | 0 |
| `cpu_residual_calls` | 53 |
| `cpu_summary_calls` | 52 |
| `gpu_summary_calls` | 102 |
| `gpu_summary_rows` | 59,724,347 |
| `gpu_summary_requests` | 102 |
| `gpu_summary_cache_hits` | 0 |
| `cpu_summary_rows` | 16,950,268 |
| `discarded_residual_lists_avoided` | 102 |
| `discarded_residual_rows_avoided` | 59,724,347 |
| `cpu_normal_calls` | 129 |
| `cpu_uniform_query_calls` | 0 |
| `cpu_adaptive_query_calls` | 0 |
| `cpu_adaptive_repaired_rows` | 0 |
| `compiled_packing_calls` | 52 |
| `compiled_packing_rows` | 24,960,000 |
| `cpu_distinct_centroid_tie_rows` | 307,670 |
| `compiled_replay_calls` | 120 |
| `compiled_replay_rows` | 960,000 |
| `gpu_plane_query_calls` | 52 |
| `gpu_uniform_query_rows` | 24,960,000 |
| `gpu_adaptive_query_rows` | 24,960,000 |
| `gpu_map_upload_bytes` | 105,635 |
| `gpu_query_upload_bytes` | 1,198,080,000 |

`gpu_residual_calls=0` counts the public standalone residual API;
`gpu_summary_calls=102` counts the private captured summary kernels.
Input payload uploads, finite differences, query work and CPU ordered
normal equations retain their work. Extra GPU coverage does not imply
a whole-process gain.

| Instrumented phase | Pre-change call sum | Experimental call sum |
|---|---:|---:|
| `CudaPlaneQueries.query` | 11.103964 s | 6.203211 s |
| `compiled_packing` | 5.690168 s | 2.649909 s |
| `_PackedObservationBuilder.build` | 17.558993 s | 9.290624 s |
| `cpu_residual_calls` | 5.282785 s | 2.956381 s |
| `cpu_summary` | 18.527361 s | 3.002949 s |
| `gpu_linearization` | 12.036049 s | 10.468887 s |
| `cpu_normal_calls` | 3.694416 s | 2.786667 s |
| `gpu_captured_summary` | — | 2.425454 s |
| `cpu_captured_summary_reduction` | — | 0.194925 s |

Call sums nest and overlap across concurrent candidates and include
instrumentation. They are neither critical-path percentages nor formal
phase speedups. Timings vary substantially; no portable confidence estimate
or peak memory measurement is claimed.

The retained pre-change pose probe finds only 35,328 adjacent repeated
poses among 15,196,549 observation rows (about 0.23%). A consecutive-pose
grouping optimization was therefore not implemented.

## Provenance, final policy and reproduction

Measured auto-enabled implementation: `4c3204637f94ec7f573df21f8fdec3535b13b362`.
Previous GPU control: `f59c2b84c93f4fb06ea6b069d139b8adf2936d40`.
Final default-disabled API policy: `18ee25541d32bf66399a3f3521121d30471cf4e4`.

Formal timings describe the measured auto-enabled implementation.
The final policy revision only adds the explicit boolean option, its
dispatch guard and validation. Its default-disabled behavior was tested
with complete synthetic CPU/CUDA clock profiles; this revision was not
used for a new formal timing cohort. Enabling the option selects the
measured summary path. The final source and test bytes are archived separately.

Targeted validation: 270 tests passed, Ruff passed, mypy passed over 356
source files, and all 126 generated schemas are byte-identical to the base.
Tests cover actual streams, shared-context concurrency, empty/forked and
replaced payloads, cache ownership, both summation policies, robust losses,
default-disabled dispatch, explicit opt-in, histories and logical counters.

Use an isolated checkout of the measured commit to reproduce the frozen
`run-captured-summary-v1.py.txt` launcher; it expects the verified public
inputs and writes the original three output cohort directories.
`diagnose-captured-summary-v1.py.txt` reproduces the instrumented dense
control on that measured checkout after the formal cohorts.
The launchers are archived unchanged and do not enable the final option
when run on a later checkout. Hashes bind 20 measured sources, 14 preceding
GPU modules, configs, launchers and every result. Measured checkout line
endings are preserved; only execution-location metadata is made portable.

The public GLIM sensor is copied into two topics as an execution parity
control. This does not establish independent two-sensor calibration accuracy.
No raw sensor data or hardware model identifier is published here.
