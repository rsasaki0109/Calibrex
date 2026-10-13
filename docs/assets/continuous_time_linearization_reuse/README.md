# Reuse exact continuous-time LiDAR finite differences

An owned CUDA linearizer retains only its most recent residual/Jacobian arrays.
Keys contain the exact FP64 bits of all thirteen transforms and six difference steps.
Changing the captured immutable observations invalidates the entry. Each fork owns its
lock, stream, payload and cache. Cached host arrays are read-only; public results are
independent writable copies. Entries exceeding 64 MiB bypass caching.

The fixed-trajectory optimizer computes only the unweighted initial rank for factors with at least 8,192 residuals when its native
compiled factor supports it. Public evaluation hooks and unsupported factors keep full
evaluation. Final residual, observability, range and holdout diagnostics remain complete.

## Comparisons with the preceding GPU implementation

Three dense/standard pairs and five default pairs alternate order. Both methods have independent
initially empty persistent Numba/CuPy caches. The first run, compilation and every slow
trial are retained. Whole wall includes startup, imports, replay, all clock candidates,
refinements, transfers, diagnostics, hashes and saves. Parent audits are outside the timer.
CUDA-full scope archives all fourteen preceding geometry, optimizer, query and replay modules.

| Workload | Train / holdout targets | Previous GPU median | Candidate median | Ratio of medians | Median paired ratio |
|---|---:|---:|---:|---:|---:|
| 120 frames, up to 8,000 points/frame | 768,000 / 192,000 | 81.842589 s | 58.798766 s | 1.392x | 1.392x |
| 120 frames, up to 400 points/frame | 38,400 / 9,600 | 17.541021 s | 13.136499 s | 1.335x | 0.974x |
| Default sampling caps, CPU dispatch | 6,000 / 3,200 | 7.633336 s | 8.759261 s | 0.871x | 0.911x |

These local descriptive ratios do not establish portable performance. Small budgets
continue to use CPU dispatch. Ratios from separate cohorts are never multiplied.
Standard paired timing does not establish a gain, and final default timing regresses.
The cause of variability remains unresolved; no small-workload speedup is claimed.

| Workload | API medians | Solver medians | All-thread CPU medians |
|---|---:|---:|---:|
| 120 frames, up to 8,000 points/frame | 78.120409 -> 54.990135 s (1.421x) | 62.428247 -> 42.114281 s (1.482x) | 105.421875 -> 82.171875 s (1.283x) |
| 120 frames, up to 400 points/frame | 12.341297 -> 9.432027 s (1.308x) | 7.003360 -> 6.027572 s (1.162x) | 19.062500 -> 15.203125 s (1.254x) |
| Default sampling caps, CPU dispatch | 4.319437 -> 5.300716 s (0.815x) | 3.016924 -> 3.841161 s (0.785x) | 8.937500 -> 10.562500 s (0.846x) |

### 120 frames, up to 8,000 points/frame

Retained status `max_iterations`, rank 6, train/holdout correspondences 582,119/67,602.

| Pair | Order | Previous GPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | baseline -> candidate | 86.768432 s | 70.049194 s | 1.239x |
| 2 | candidate -> baseline | 81.842589 s | 58.798766 s | 1.392x |
| 3 | baseline -> candidate | 64.312364 s | 43.094047 s | 1.492x |

[dense definition](dense.definition.json), [aggregate](dense.benchmark.json), [complete saved result](dense.sparse.repeat-01.result.yaml).

### 120 frames, up to 400 points/frame

Retained status `max_iterations`, rank 6, train/holdout correspondences 29,181/3,451.

| Pair | Order | Previous GPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | baseline -> candidate | 19.402509 s | 57.206190 s | 0.339x |
| 2 | candidate -> baseline | 17.541021 s | 13.136499 s | 1.335x |
| 3 | baseline -> candidate | 12.272086 s | 12.606016 s | 0.974x |

[full definition](full.definition.json), [aggregate](full.benchmark.json), [complete saved result](full.sparse.repeat-01.result.yaml).

### Default sampling caps, CPU dispatch

Retained status `converged`, rank 6, train/holdout correspondences 4,773/2,526.

| Pair | Order | Previous GPU wall | Candidate wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | baseline -> candidate | 20.496180 s | 18.635960 s | 1.100x |
| 2 | candidate -> baseline | 9.138590 s | 6.403741 s | 1.427x |
| 3 | baseline -> candidate | 7.633336 s | 8.759261 s | 0.871x |
| 4 | candidate -> baseline | 5.465509 s | 9.206229 s | 0.594x |
| 5 | baseline -> candidate | 5.338449 s | 5.861534 s | 0.911x |

[default definition](default.definition.json), [aggregate](default.benchmark.json), [complete saved result](default.sparse.repeat-01.result.yaml).

## Separate dense CPU/hybrid comparison

Three fresh alternating pairs compare the earlier packed/parallel CPU version with the
current hybrid. This is a separate cohort; its first compilation and all slower runs remain.
Compiled scope archives ten CPU modules. The current packed builder is shared with its
GPU callback disabled for CPU, preserving the existing numeric selection.

Whole-process medians 149.580531 -> 52.385271 s (2.855x); median paired ratio 3.231x.

| Pair | Order | CPU wall | Hybrid wall | Paired ratio |
|---:|---|---:|---:|---:|
| 1 | CPU -> hybrid | 148.058603 s | 52.385271 s | 2.826x |
| 2 | hybrid -> CPU | 149.580531 s | 46.301385 s | 3.231x |
| 3 | CPU -> hybrid | 256.507218 s | 62.293333 s | 4.118x |

[CPU/hybrid definition](cpu_control/dense.definition.json), [aggregate](cpu_control/dense.benchmark.json).

## Actual-device diagnostics outside formal timing

Of 129 requested finite differences, 56 launch GPU work and 73 reuse exact host results. Actual launched rows: 32,768,988.
Largest retained per-context entry: 33,398,792 bytes, below the 67,108,864-byte limit. Initial compiled rank calls: 26 of 26.
GPU query batches: 52; uniform/adaptive query rows: 24,960,000/24,960,000; payload uploads: 26.
CPU standalone residual calls: 207; ordered normal reductions: 129; GPU standalone residual calls: 0; CPU finite-difference fallback calls: 0.
Compiled CPU replay: 120 batches, 960,000 points. Query boundary/tie repair stays on CPU.

The complete diagnostic result matches the first final dense candidate artifact. Instrumented
call sums overlap across concurrent/nested calls and are not critical-path fractions.
[Actual counts](diagnostic.json), [diagnostic launcher](diagnostic-launcher.py.txt).

The separate preceding-version profile found 73 identical repeats in 129 GPU requests.
It binds the exact source and input hashes. Its historical comparison excludes only the
provenance Git commit and explicitly bound execution notes. Full original results remain.
[Preceding profile](attempts/v2/profiles/baseline/profile.json), [profile result](attempts/v2/profiles/baseline/result.yaml).

## Stopped first comparison

The initial twelve-module comparison stopped after the first baseline worker completed.
The older parent binding checks rejected the preceding GPU association/replay imports.
No speedup is reported for this incomplete attempt. Its complete worker result, exact source
snapshots, launcher and path-portable failure logs remain under [attempts/v1](attempts/v2/attempts/v1/source-index.json).
The final cohort uses a new output directory and new empty caches after the binding fix.

## Retained preceding candidate and aborted secondary cohort

Before the 8,192-residual crossover, a complete five-pair cohort showed a dense median
ratio of 1.290x (paired median 1.163x), standard 1.006x (paired 1.056x), and a default
regression: 6.101217 -> 11.804868 seconds, 0.517x (paired 0.815x). Its full 30 trials
are preserved separately. Timing variation does not identify the cause of the slower
default runs. The final dispatch leaves small factors on native complete evaluation.
The first secondary CPU cohort was explicitly stopped before its first worker completed
to apply this dispatch guard. Its logs, archived baseline modules, termination record
and source/input bindings remain under attempts/v2/cpu_stopped.
[Preceding index](attempts/v2/publication-index.json), [abort record](attempts/v2/cpu_stopped/abort.json).

## Result and lineage checks

Every complete final API/saved artifact matches all numeric fields, iterations, status,
correspondence counts, holdout results and logical interpolation counters. Comparison only
normalizes path separators and excludes strictly validated execution identity notes.
No sampling, physical settings, robust loss, solver or holdout budget is reduced.
The AIST GLIM source recording is copied into two topics as an execution parity control;
it provides no independent two-sensor calibration accuracy evidence.
The stopped trial and both instrumented diagnostics are separate from the 28 final formal trials.
Complete result YAML stays byte-identical. Only location metadata is made portable; original
hashes and changed fields are recorded in the index. Measured sources retain actual bytes.

[Publication index](publication-index.json), [benchmark launcher](benchmark-launcher.py.txt), [CPU launcher](cpu-control-launcher.py.txt).

Preceding GPU source `9f68d3d165385fa6a3f154d5e14fe31676b40a85`; measured candidate `1bf71e2668e3ff790a427d9c5d6ea66df90d22e3`.
CPU source `eb50c99878706a45eb0e584f3573507313cdc172`. Schema-valid complete measurements, definitions and aggregates are archived.
