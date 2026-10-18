# Rejected LiDAR execution candidates

Five further scheduling/search candidates did not establish a substantial
improvement. None is adopted. `src/calibrex` remains byte-identical to the
runtime in `917748b10d5eff9bb077b332a018eafdb33e9fcd`.

| Candidate | Prior median | Candidate median | Prior / candidate | Scope |
| --- | ---: | ---: | ---: | --- |
| Parallel CPU samples, ordered native reductions | 17.694 s | 15.641 s | 1.131x | Complete map-evaluation worker |
| Reused pinned system buffers | 0.4201 s | 0.4093 s | 1.026x | Eight target constructions and registrations |
| Reductions directly into mapped host memory | 0.4201 s | 0.4239 s | 0.991x | Same eight-target scope |
| Fused normal-neighbor search and PCA | 0.3978 s | 0.4269 s | 0.932x | Separate eight-target cohort |
| Interval-selected FP32 search with FP64 refinement/fallback | 0.4211 s | 0.5110 s | 0.824x | Separate eight-target cohort |

The 13% CPU improvement is too small to justify adding the proposed executor
lifecycle and batch interface. The two transfer variants provide little benefit,
and both search variants have slower median times. This is a post hoc engineering
decision on these observations, not a pre-registered performance claim or proof
of behavior on other workloads. Cohorts are kept separate; ratios are neither
pooled nor multiplied with the [complete-bag measurements](cpu_map_bag_validation.md).

## CPU sample scheduling

Two fresh-process pairs rotate query-parallel/sample-parallel execution order
over the same spent public KITTI copied-sensor map control as the earlier
[CPU map comparison](cpu_map_normal_cache.md). Both have an eight-worker budget
and one BLAS thread. The candidate runs independent sample contributions in a
persistent pool, uses one native query worker per sample and accumulates each
Hessian, gradient and residual sequence in the original sample order. Held-out
scores use parallel sample queries with the original scalar accumulation order.

| Pair | Query parallelism | Sample parallelism |
| --- | ---: | ---: |
| 1, queries first | 17.850 s | 15.944 s |
| 2, samples first | 17.538 s | 15.337 s |

Every worker includes imports, reads, all five complete fits, covariance,
held-out scoring, six axis controls, source/input verification and saves. Bag
decoding and reference odometry are outside this scope. All four workers and
20 full fit records are retained. Complete numerical artifacts, evaluation
fields, prepared-input hashes and fit arrays preserve every float bit after
normalizing only artifact creation time and explicitly measured durations.

## GPU system transfer and normal search

Three separate warm component screens use the same eight fixed registration
captures from the already spent public GLIM Livox bag. They retain all 288
timed registrations and seven first-case warmups. Transfer variants use four
cyclic repeats; each search variant uses six alternating repeats. Times include
target construction and registration, including candidate setup and fallbacks.
First-case compilation/warmup is outside those component times and remains in
the archive. None measures cold startup, full bag replay or calibration speed.
All three original protocols record a false cache-directory emptiness flag;
there is no empty-cache claim. CPU/GPU activity was unreserved and BLAS used
one thread. These subsecond observations are exploratory.

Pinned buffers reuse small device/host system arrays. The mapped variant places
the reduction output in device-addressable pinned host memory and waits for an
event before reading it. Neither changes the reduction or native 6x6 solve.

Normal fusion joins the original tree traversal and FP64 PCA in one kernel.
The interval-search candidate ranks outward-rounded FP32 squared-distance
intervals, recomputes selected distances with the original FP64 arithmetic,
and repeats the original FP64 search when an unselected interval overlaps
the cutoff. It retains native repairs and all neighbors. This prototype was
checked on eight fixed inputs; it is not a production API or a general boundary
correctness validation.

All stored registration transforms, convergence decisions, iterations,
correspondence counts, RMSE, Hessian eigenvalues and existing logical query/work
counters agree bit for bit, including the retained warmups. Counters do not
measure speculative internal interval operations. Experimental code is bound
by launcher/generated-code hashes; inherited `execution_backend` strings
describe the unchanged base implementation rather than the added kernels.

## Replay and limits

The [byte-bound archive](../assets/lidar_execution_screen/README.md) includes
original protocols, every measurement/result, raw logs, warmups, generated code,
launchers, helper snapshots and coordinate-capture metadata. Shared source
capsules reconstruct all 396 runtime modules. Raw points, compiler caches and
binary profiles are not redistributed. Copied sensors and captured registrations
provide execution agreement, without independent calibration accuracy or SOTA
evidence.

```bash
python -m tools.audit_lidar_execution_screen docs/assets/lidar_execution_screen --check
```

The CPU-only auditor checks source/code/input bindings, complete fit/evidence
bits, original order and timing receipts, all registration fields/work, and
separate medians. It rejects one-ULP fit/quality changes, rebound timings and
source-manifest substitutions. It checks saved coordinate identities, without
rerunning geometry or executing archived prototype text.

These results do not support further small scheduling changes as a substantial
whole-pipeline gain. A larger improvement needs a change to the combined
preparation/registration work or an explicitly validated arithmetic policy.
