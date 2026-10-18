# Further ICP GPU acceleration: rejected after whole-process evaluation

The additional changes did not meet the requested whole-process improvement.
The final GPU/GPU comparison reached **1.233x**, while the CPU comparison was
approximately **1.022x**. The 768-point CUDA control was slower. Restore all
production solver, CUDA adapter and spatial-search code to the preceding
merged implementation, `4ef988ee543d6c1f9303f5a6758c89960b857015`.
Retain the reusable historical-source comparison tool, schema and complete
evidence. No new GPU execution path is enabled by this publication.

The data is the same public KITTI Raw sequence and actual-frame fixtures as
the [preceding ICP evaluation](../cuda_point_to_point_icp/actual_frames/README.md).
The primary uses frames 115→116, roughly 125k points, the default 50-iteration
budget, spatial holdout, rematching and all six initial-value diagnostics.
Every primary solve converges. Breadth and small controls use the predefined
20-iteration cap. Navigation supplies a seed/consistency reference; these
same-sensor registrations do not establish independent calibration accuracy.

## Complete-process results

Each primary phase uses three alternating prior/candidate pairs. CPU primary
also uses three pairs; breadth and small controls use one smoke pair each.
All subprocess startup, input/source guards, initial compilation, preparation,
solve, diagnostics, extra initial-holdout evaluation and saving are included
in whole time. Solver time excludes startup/preparation and that extra evaluation.
Workers use one verified BLAS thread and automatic bounded diagnostic workers.
Every CUDA disk cache begins empty. OS/driver caches persist. All cold, slow
and completed formal observations are retained.

| Phase | Prior whole median | Experimental whole median | Whole ratio |
|---|---:|---:|---:|
| Two-candidate host ordering | 29.931710 s | 25.493214 s | 1.174x |
| Direct GPU distances and moving-world search | 26.268393 s | 20.033780 s | 1.311x |
| Separate fixed-frame traversal | 30.087811 s | 23.373413 s | 1.287x |
| Final distance-gated GPU traversal | 21.220727 s | 17.203994 s | 1.233x |
| Final CPU control | 48.041180 s | 47.016009 s | 1.022x |
| Frames 230→231, final GPU smoke | 16.501400 s | 14.098342 s | 1.170x |
| Frames 345→346, final GPU smoke | 16.877622 s | 13.367437 s | 1.263x |
| 768 points, final GPU smoke | 3.600471 s | 3.675084 s | 0.980x |

Phases have different ambient timings and are separate comparisons. Ratios
must not be pooled or multiplied. The final experimental source is
`7e81e0a`; every protocol binds its full revision and exact source bytes.

All **36 formal complete calibration results** match byte for byte within
their cohort: transform, all iteration histories, spatial IDs, holdout,
rematching, six diagnostic starts and symmetry flags. Complete models are
deduplicated by SHA-256, with a reference from every worker. Original worker
measurements, logs, protocols, attempt journals, fixtures and both sets of
source snapshots are retained.

## Investigated work

The host prototype replaces a per-row two-element lexicographic sort with
distance/index comparisons, preserving signed zeros, stable ties and NaN
ordering. Its microbenchmark improved substantially, but whole-process gain
was only 1.174x.

A fixed two-proposal CUDA kernel reduced temporary arrays and sized DFS
scratch to the captured tree depth. Actual-frame kernel measurements improved
by only about 3%; preserve every probe, including first compilations. This
kernel was rejected before a full-process claim.

The subsequent prototype computes nearest FP64 squared distances directly on
CUDA. Moving-source queries reuse raw-source topology but evaluate the actual
CPU-generated moving-world coordinates. Conservative pruning uses singular
value margins, measured orthogonality defect and downward rounding; unsafe
coordinates/matrices retain native fallback. Separate static traversal and
distance-gated fitting queries remove more intermediate work. Native holdout
remains unbounded. Distance bits, ties, layout variants, extreme-coordinate
fallback, public substitutions and complete native results were tested.

The [CUDA intrinsic documentation](https://docs.nvidia.com/cuda/archive/12.6.2/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html)
specifies downward rounding for the bound arithmetic. This arithmetic property
does not establish a whole-process speedup; measured results above determine
the rejection.

## Exploratory observations and corrections

The first host-ordering pilot workers accidentally overlapped. Their timings
are excluded from formal comparisons; both complete models and an explicit
policy record are retained. A subsequent serial pilot ran while separate CPU
benchmark workers were active. Those processes were left alone; the ambient
note and both complete models are retained, separately from the formal cohort.

The initial phase profiler pointed its parity check at the wrong cohort's
model. Its unmodified report and launcher are retained together with a
correction binding the report SHA-256 and confirming the actual primary model
digest. Nested/concurrent phase sums are not critical-path percentages.

Exploratory native-tool smoke models, direct-GPU pilot, host microbenchmark and
rejected pair-kernel trials are retained. These observations are not added to
formal medians. Compiler cache binaries are omitted; source/launcher bytes and
cache policy remain recorded.

## Reproduction and audit

The benchmark activates historical modules before importing calibration code,
checks their actual file paths and hashes, and proves CPU workers import no
CuPy. Its protocols, attempts and summaries have a dedicated tool schema;
exported core schemas are unchanged.

```shell
python -m tools.benchmark_icp_neighbor_order --baseline-commit 4ef988e --backend cuda --repeats 3 --fixture outputs/kitti-icp-pairs-default-v1/0000000115-0000000116.json --data-root data/public/kitti_raw_icp/2011_09_30_drive_0018_sync --output outputs/neighbor-comparison
python tools/audit_icp_neighbor_experiments.py
```

Check out the measured experimental revision to reproduce its implementation;
the publication revision restores production code. The preceding page explains
public input acquisition and fixture preparation. The CPU-only archive auditor
checks all file digests, frozen import bindings, complete models, source guards,
BLAS policy, schema validity and recomputed aggregate metrics.
