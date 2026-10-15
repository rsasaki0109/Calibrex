# CUDA map normals computed on demand

The optional CUDA LiDAR–LiDAR map backend computes a normal when registration
first reaches that map point, then reuses it across optimization, jackknife,
holdout and perturbation controls. A single FP64 kernel gathers the exact
neighborhood, forms its covariance and obtains the smallest eigenvector.
The native NumPy/SciPy path repairs uncertain geometry and neighborhood ties.

This reduces work but is an **incremental improvement**. Complete fresh-process
medians improved by approximately 1.11–1.13x on this public workload. A warm
shared-process pilot suggested a larger gain; that is not the complete-process
result and is not presented as a large end-to-end speedup.

| Compiler-cache policy | Previous whole process | Current whole process | Ratio | Previous API | Current API | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Empty for each worker, 3 pairs | 18.813 s | 16.998 s | 1.107x | 13.667 s | 11.648 s | 1.173x |
| Retained per method, 4 pairs | 10.814 s | 9.595 s | 1.127x | 5.175 s | 3.946 s | 1.311x |

Every process is included, including first initialization and compilation.
Process order alternates by pair. The first retained-cache current process
was slower: 21.732 s versus 20.221 s for the previous implementation. Its
observation remains in the cohort. The separate cold cohort has three positive
pairs. These descriptive measurements use an unreserved host; they do not
establish portable speed guarantees. Ratios from different scopes are not multiplied.

## Frozen workload and result checks

Both variants use every original point in nine public KITTI raw scans from
`2011_09_30_drive_0018_sync`: frames 114–116, 229–231 and 344–346. Each three-scan
window forms two samples with generated reference poses. Targets copy the three
central scans with fixed 0.01 m Gaussian noise, seed 20261012. There are six map
samples, 2,286,890 map rows and 762,118 target rows; four blocks train and two
hold out. The complete API includes the training fit, four block jackknife fits,
covariance, held-out scoring and all six known-bad controls.

These overlapping copied-sensor samples are an execution control, with no
independent two-LiDAR extrinsic truth or independent temporal validation.
Raw map windows are not motion compensated. Bag decoding and reference odometry
are outside this map-registration workload.

All 14 results preserve every evaluation field under the existing tolerances:
absolute 1e-9 / relative 1e-6 for floating-point fields, absolute 1e-13 / relative
1e-5 for covariance. Counts, status, iterations, policy, calibrated degrees of
freedom and control decisions match exactly. Prepared arrays, split identities,
raw input hashes and complete saved schema-valid artifacts are also checked.
Both CUDA execution identities are independently bound to frozen kernel bytes.
This is numerical parity rather than bitwise equality of eigenvectors.

The previous revision is `3340116c2b0c1a7c0cd1aa3b29468a44d4fc330f` and the
measured current revision is `8d7a6477343d2a727ae02c86704fe8c76f894c5c`.
The pipeline settings, correspondence gates, normal neighborhood, planarity
criterion, loss, iteration limits and evaluation policy are unchanged.

## Work avoided and bounds

Normal computations fall from 2,286,890 to 1,247,324 map rows: **45.5% fewer**.
Actual GPU query rows fall from 12,069,816 to 11,030,250. Their difference is
exactly the omitted normal neighborhoods; registration query work is retained.
The new implementation records 66 normal-kernel launches and 7,691 native normal
repairs. Query launches increase from 113 to 143 because normal preparation is
requested in smaller batches. These counters remain visible in every record.

The cache is owned by the existing solve-scoped immutable snapshot. Each normal
is computed once; it adds one byte per map row for completion flags. Retained
payload increases from 181,676,658 to 183,963,548 bytes on this workload and
remains subject to the existing 1 GiB limit. Scratch and allocator pools are
additional. A pre-existing native normal cache retains its captured policy.
The map criterion remains `low <= ratio * mid`, with no eigenvalue floor.
The separate scan-odometry criterion is not substituted for it.

CPU tree construction, topology capture, startup, compilation and native repairs
remain material costs. CPU is still the default backend. This change introduces
no dependency, CLI option or result-schema change.

## Reproduction and audit

Obtain the public scans under the [official KITTI terms](https://www.cvlibs.net/datasets/kitti/raw_data.php),
install `calibrex[cuda]`, check out the measured current revision, then run:

```bash
python -m tools.benchmark_cuda_map_preparation \
  --sequence path/to/2011_09_30_drive_0018_sync \
  --baseline-commit 3340116c2b0c1a7c0cd1aa3b29468a44d4fc330f \
  --half-window 1 --repetitions 3 --output-dir outputs/map-normal-cold
```

Use a new output directory. For the separate retained-cache cohort, use
`--compiler-cache retained --repetitions 4`. Both methods start with empty,
independent caches; retained mode preserves each method's cache across its
subsequent fresh processes. Every worker uses one BLAS thread and verifies its
actual imports against the frozen complete runtime manifest. Complete wall time
includes startup, raw reads, preparation, GPU initialization/compilation, all
evaluation work, result saves and worker source/hash verification. Parent guards
and aggregation are outside wall time.

The byte-bound archives contain complete observations and artifacts, both source
capsules/manifests, cache-file hashes, exact launchers and protocol schema.
Original operator paths remain in provenance. Raw scans and compiler binaries
are not redistributed. Recompute all quality checks, kernel identities, work
counts, pair ratios and medians without a GPU or raw scans:

```bash
python -m tools.audit_cuda_map_preparation docs/assets/cuda_map_normal_cache/cold
python -m tools.audit_cuda_map_preparation docs/assets/cuda_map_normal_cache/retained
```

See the complete [cold-cache records](../assets/cuda_map_normal_cache/cold/README.md)
and [retained-cache records](../assets/cuda_map_normal_cache/retained/README.md).

Local validation includes 64 map/device/benchmark/archive tests, full-repository
Ruff, strict mypy over 376 source files and a strict documentation build.
All 130 exported schema/ledger files remain byte-identical to the baseline.
Archive mutation tests reject changed transforms, kernel identities, work counts,
imports, timings and cache scope even after outer file hashes are rebound.
