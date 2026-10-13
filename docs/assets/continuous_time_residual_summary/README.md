# Fuse ordered LiDAR residual error summaries

Native compiled factors with at least 8,192 observations compute residuals,
squares and their ordered sum in one optional CPU Numba pass. No per-row
residual array or Python float list is created for optimizer/holdout RMSE.
Fastmath stays disabled and the kernel releases the GIL. CPython 3.10/3.11
retain plain ordered addition; CPython 3.12 retains Neumaier compensation,
including native underflow and overflow handling. Python computes the final
division and square root. Other runtimes use the original residual/list path.

The summation policy follows the
[CPython 3.12 implementation](https://github.com/python/cpython/blob/v3.12.10/Python/bltinmodule.c#L2464).
Bitwise tests compare the running Python reduction, general corrections,
weights, signed zero, extreme dynamic range, underflow and overflow.

The native optimizer also requests normal equations without converting their
discarded residual vector to a Python list. Direct callers still receive
complete residuals by default. Custom compiled-system hooks keep their existing
arguments and return contract. Public residual/correction hooks, replaced
reductions, small factors and unsupported runtimes keep the original path.
Public residuals and final observability diagnostics remain complete.

## Complete fresh-process comparisons

Each workload has three alternating pairs and independent initially empty
persistent Numba/CuPy caches. Every initial compilation and slower trial is
retained. Whole wall includes startup, imports, replay, all clock candidates
and refinements, transfers, diagnostics, hashes, YAML and measurement saves.
Parent audits are excluded; OS caches persist. Separate cohorts are neither
pooled nor multiplied by earlier speedup ratios.

| Comparison | Baseline median | Candidate median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|
| Previous GPU, 960,000 captured points | 69.805556 s | 50.653481 s | 1.378x | 1.380x |
| Previous GPU, 48,000 captured points | 13.870382 s | 13.226075 s | 1.049x | 1.034x |
| Previous GPU, default sampling | 7.703580 s | 8.765065 s | 0.879x | 0.965x |
| Packed CPU, separate dense control | 196.076865 s | 39.026530 s | 5.024x | 5.024x |

Default sampling regresses by about 14% in the ratio of medians;
its median paired ratio is 0.965x. The cause of timing variation remains
unresolved. Default factor sizes retain the original residual/RMSE route,
without dispatching the new summary kernel. No consistent small-batch gain
is claimed. Cold dense CPU wall of 289.691684 s is retained in full.

| Workload | Pair | Order | Baseline whole wall | Candidate whole wall |
|---|---:|---|---:|---:|
| dense | 1 | baseline, candidate | 60.541125 s | 57.133829 s |
| dense | 2 | candidate, baseline | 69.805556 s | 35.290130 s |
| dense | 3 | baseline, candidate | 69.896860 s | 50.653481 s |
| full | 1 | baseline, candidate | 25.900847 s | 25.043427 s |
| full | 2 | candidate, baseline | 13.870382 s | 10.238290 s |
| full | 3 | baseline, candidate | 13.174195 s | 13.226075 s |
| default | 1 | baseline, candidate | 15.982607 s | 16.568270 s |
| default | 2 | candidate, baseline | 6.441744 s | 8.765065 s |
| default | 3 | baseline, candidate | 7.703580 s | 6.228300 s |
| cpu-dense | 1 | baseline, candidate | 289.691684 s | 50.459870 s |
| cpu-dense | 2 | candidate, baseline | 149.829356 s | 35.999593 s |
| cpu-dense | 3 | baseline, candidate | 196.076865 s | 39.026530 s |

All 24 formal results match after normalizing path separators and excluding
only strictly validated execution identity notes. Full API results, saved YAML
bytes, histories, diagnostics and logical counters are retained. Dense controls
use 768,000 train and 192,000 holdout targets, rank 6 and `max_iterations`.

The preceding GPU comparison archives fourteen modules. The separate CPU
control archives ten CPU modules and shares the packed builder, with neither
GPU position queries nor the compiled packing callback supplied by that
archived CPU solver. Settings, samples, splits and writers are unchanged.

## Actual work outside formal timing

| Current execution | Count |
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
| `cpu_summary_calls` | 154 |
| `cpu_summary_rows` | 76,674,615 |
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

| Instrumented phase | Pre-change call sum | Current call sum |
|---|---:|---:|
| `CudaPlaneQueries.query` | 5.719896 s | 4.749705 s |
| `compiled_packing` | 2.759657 s | 2.238720 s |
| `_PackedObservationBuilder.build` | 8.894645 s | 7.411736 s |
| `cpu_residual_kernel` | 9.421330 s | — |
| `cpu_residual_calls` | 12.476100 s | 2.508685 s |
| `fixed_rmse` | 7.442820 s | — |
| `gpu_linearization` | 12.107272 s | 5.372967 s |
| `cpu_normal_calls` | 2.437897 s | 2.064291 s |
| `pair_rmse` | 0.119885 s | — |
| `cpu_summary` | — | 10.066799 s |

Call sums nest and overlap across candidates, and instrumentation differs;
they are not critical-path percentages or formal phase speedups. GPU queries,
finite differences, cache hits and ordered normal equations keep their work.
CPU residual computation is retained; the new summary avoids object conversion
and Python iteration. GPU workload coverage does not increase.

## Provenance and limits

The complete pre-change profile is retained with source bytes verified against
the preceding commit. Its historical comparison excludes only git commit fields;
full saved provenance is retained. The current diagnostic matches the complete
first formal candidate result and checks source/input bytes before and after.

The public GLIM sensor is copied into two topics as an execution parity control.
This does not establish independent two-sensor calibration accuracy. Three local
pairs are descriptive observations, without portable confidence estimates.
Peak memory, other interpreters and Python 3.13+ acceleration are not quantified.

Measured implementation: `a514397f5219b1423f3aedbc8b79be9b581f928b`.
The publication index binds every measured source, baseline module, config,
launcher, full result and portable location metadata. Source archives preserve
measured checkout bytes. API values and saved YAML bytes remain unchanged.
