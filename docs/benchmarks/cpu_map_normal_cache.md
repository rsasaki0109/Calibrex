# CPU map normals and parallel queries

LiDAR–LiDAR map registration can compute native map normals when a fit first
reaches a point and retain them for later iterations, jackknife fits, held-out
scoring and controls. `MapSampleOptions(query_workers=8)` also runs independent
SciPy query rows on eight CPU workers. Both correspondence searches and normal
neighborhoods use this budget. The default remains one worker.

Two separate development cohorts compare the existing one-worker CPU path
with reached-point normals and eight query workers. Complete fresh-process
ratios are **1.987x** and **2.138x**. All six paired eight-worker processes are
faster, but these unreserved-host observations do not establish a portable
minimum of 2x. Bag decoding and reference odometry are outside this workload.
A later [whole-bag GLIM comparison](cpu_map_bag_validation.md) includes those
stages and observes 1.249x at 0.2 m map voxels and 1.687x at 0.03 m map voxels.

## Complete measurements

Each worker reads all original points from nine spent public KITTI scans,
constructs six local-map samples, runs the training fit and four block
jackknife fits, computes covariance and held-out evidence, runs all six axis
controls, validates and saves the complete artifact, and verifies input and
executed-source hashes. Every process has one BLAS thread. Startup, imports,
reads, preparation, all evaluation work, verification and saves enter wall time.

| First cohort | Previous CPU, 1 worker (s) | Reached normals, 1 worker (s) | Reached normals, 8 workers (s) |
| --- | ---: | ---: | ---: |
| Triad 1 | 29.589 | 22.552 | 13.914 |
| Triad 2 | 28.931 | 24.062 | 16.403 |
| Triad 3 | 30.331 | 27.784 | 14.889 |
| Median | **29.589** | **24.062** | **14.889** |

The one-worker ratio is 1.230x; the eight-worker ratio is 1.987x. The initial
2x advancement gate is missed. Its original comparator also flagged different
artifact creation times. Original records, errors and the negative advancement
decision remain unchanged. The independent audit normalizes that one additional
run-metadata field and verifies that all physical and evidence bits match.

| Follow-up cohort | Previous CPU, 1 worker (s) | Reached normals, 8 workers (s) | Query-certificate candidate, 8 workers (s) |
| --- | ---: | ---: | ---: |
| Triad 1 | 41.999 | 19.387 | 23.074 |
| Triad 2 | 41.458 | 19.978 | 23.417 |
| Triad 3 | 34.002 | 15.395 | 23.653 |
| Median | **41.458** | **19.387** | **23.417** |

The retained middle variant is 2.138x faster than the previous CPU path in
this cohort. The certificate experiment tries to retain nearest hits while
triangle bounds exclude every competing point. It reduces searches but adds
enough memory traffic and checks to make complete execution **20.8% slower**
than reached normals alone (0.828x). It fails its original gates and is excluded
from the implementation. No unfavorable observation is discarded or replaced.

Keeping reached normals and explicit query budgets is a **post-hoc engineering
choice**, informed by both cohorts and bit parity. It is not a claim that either
frozen screen passed every advancement gate. The two cohorts have different
instrumentation and host conditions; their times and ratios are not pooled or
multiplied. No complete two-LiDAR bag speedup is established by these records.

## Geometry, scope and retained work

The workload uses KITTI `2011_09_30_drive_0018_sync`, frames 114–116, 229–231
and 344–346. Six samples reuse three raw windows under generated reference
poses. Targets copy central scans with fixed 0.01 m Gaussian noise, seed
20261012. Four sample blocks train and two hold out. These overlapping copied
samples are an execution control, with no independent dual-LiDAR accuracy or
temporal-validation evidence. Raw windows are not motion compensated.

Normal computations fall from 2,286,890 to 1,247,324 rows, **45.5% fewer**.
The retained change still searches every correspondence query: 9,782,926 rows
per complete evaluation. It retains neighborhood size, native tie ordering,
covariance arithmetic, NumPy eigenspaces, planarity without an eigenvalue floor,
distance gates, Huber loss, all iteration limits and every evidence call.

All 18 workers retain all 90 complete fits. The independent audit compares 60
paired fits, full evaluation records, complete covariance and saved artifacts
bit for bit after only declared run metadata and query-budget options are
normalized. Decisions, counts, iterations, policies and all controls match.
Prepared array and split hashes also match. These are execution-agreement
checks, not independent accuracy or SOTA evidence.

Each sample owns its normal cache. A normal is computed once, with one extra
completion byte per map point. Explicit `MapSample.prepare()` still materializes
all normals, and caller-prepared normals retain their captured policy. No CUDA
or other optional dependency is imported by the default CPU path. The final
numerical implementation is the measured `09f27193c9418ec9b6584caea459661e047fe875`;
a follow-up only records nondefault query budgets in bag-artifact options.
The rejected certificate runtime is retained as source evidence only.

## Use and audit

```bash
calibrex lidar-lidar ros2 path/to/bag \
  --reference-topic /reference/points --target-topic /target/points \
  --initial initial.yaml --registration-query-workers 8 \
  --dataset-family your_dataset --dataset-license "your dataset terms" \
  --output outputs/lidar-pair.yaml

python -m tools.audit_cpu_map_query_cache docs/assets/cpu_map_query_cache --check
```

The CLI records the explicit budget in command provenance; the bag API records
nondefault budgets in artifact options. Select a budget for the available CPU;
small queries can spend more time starting threads. This option changes
execution scheduling and introduces no result-schema change.

The [bound archive](../assets/cpu_map_query_cache/README.md) retains both cohorts,
all original errors and decisions, all complete worker artifacts and fits,
launchers, input/source bindings, source deltas and the CPU-only audit. The
shared prior source capsule is already published in the preceding resident-scan
validation archive. No raw scans or fitted coordinate arrays are redistributed.
