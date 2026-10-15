# Fused CUDA map registration

The CUDA LiDAR–LiDAR map backend combines point projection and exact two-neighbor
search in one kernel, gathers source-ordered point-to-plane residuals in another,
and forms weighted analytic Jacobians in a third. Both variants retain the
previously adopted normal cache. This comparison isolates the registration
dispatch change and its compilation cost.

| Compiler-cache policy | Previous whole (s) | Current whole (s) | Ratio | Previous API (s) | Current API (s) | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Empty per worker, 3 pairs | 19.202 | 13.253 | 1.449x | 12.901 | 7.183 | 1.796x |
| Retained per method, 4 pairs | 10.049 | 11.393 | 0.882x | 4.249 | 3.921 | 1.084x |

Cold mode gives every process a separate initially empty compiler cache.
Retained mode gives each method an independent empty cache, then retains it for
that method's subsequent fresh processes. Every process enters the medians,
including the first compilation; there is no excluded warmup. Pair order
alternates. The initial-compilation and retained-cache results have different
scopes and must not be combined or multiplied with earlier normal-cache ratios.

The final cold-cache cohort has three positive pair ratios (1.510x, 1.449x and
1.298x). The retained-cache cohort's complete-process median is **13.4% slower**,
with pair ratios 1.576x, 0.759x, 1.118x and 1.049x. Its API median improves by
1.084x, but the complete fresh-process observations establish **no end-to-end
improvement with an existing cache**. Startup/source verification and unchanged
CPU work remain in those timings; shared-host variation is not isolated from
implementation effects. Every observation remains available.

The generated compiler-cache program count falls from 47 to 21 on this workload.
The cold-start improvement comes mainly from compiling and launching fewer small
GPU operations. Once the compilation cache exists, the gain is substantially
smaller. These measurements describe an unreserved host and do not establish
portable speed guarantees or steady-state throughput for other map sizes.

## Unchanged calibration work

Nine original KITTI raw scans from `2011_09_30_drive_0018_sync`, frames 114–116,
229–231 and 344–346, form three-scan maps. The central scans provide copied
targets with fixed 0.01 m Gaussian noise, seed 20261012. Six generated pose
samples contain 2,286,890 map rows and 762,118 target rows; four blocks train and
two hold out. Every original point, the training fit, four block jackknife fits,
covariance, held-out scoring and all six known-bad controls remain in the API.

This copied-sensor execution control has no independent two-LiDAR extrinsic
truth or independent temporal validation. The raw map windows are not motion
compensated. Bag decoding and reference odometry are outside this workload.

All complete evaluation fields and saved schema-valid artifacts are compared
under the existing map tolerances: absolute 1e-9 / relative 1e-6 for floating-point
fields and absolute 1e-13 / relative 1e-5 for covariance. Counts, iterations,
statuses, policy, estimated degrees of freedom and control decisions match
exactly. Prepared arrays, raw scan hashes and train/holdout identities match.
Both CUDA identities are independently checked against their frozen query,
normal and registration kernel bytes. This is numerical tolerance parity,
with no claim of bitwise floating-point equality.

All previously recorded work counters are identical in each pair. Both variants
query 11,030,250 rows and compute 1,247,324 normals, with 7,691 native normal
repairs and 78,474 native query repairs. The current implementation additionally
reports 77 projection/query launches over 9,782,926 rows, 77 residual-gather
launches over 5,678,656 rows and 59 weighted-Jacobian launches over 4,372,180 rows.
The auditor checks these counters against the complete original query work.

## Arithmetic, layout and memory

Projection, nearest-neighbor search, residuals and Jacobian arithmetic remain
FP64, with fused multiply-add disabled in these explicit kernels. Existing GPU
matrix products reduce the same weighted Jacobians into the normal equations.
Six-dimensional updates, covariance and evidence policy retain their native
definitions. Correspondence gates, tie guards, planarity, Huber loss and
convergence limits are unchanged. Ambiguous ties and gate boundaries retain
the original native queries and transforms.

One byte flag per source point reports validity and uncertainty. The host
compacts these flags in increasing source order, avoiding a separate dynamic
GPU nonzero/count operation. Normal selection reuses the existing bounded cache.
Explicit FP64 row-major matrix snapshots support ordinary, column-major and
strided caller arrays, including float32 input matrices. No new input-layout
restriction is introduced.

Retained device input bytes remain 183,963,548 on this workload, under the existing
1 GiB bound. Scratch and allocator pools are additional. No new dependency,
public calibration option, result schema, point cap or approximation is added.
CPU remains the default backend.

## Reproduction and independent audit

The baseline is `22c19ebc75ec61715c83a2e77e9c3e8eea34dffc`.
The measured current revision is `8fcc8d63052908e8fb06c0153a2e975acfb0dc9f`.
Obtain the original scans under the [official KITTI terms](https://www.cvlibs.net/datasets/kitti/raw_data.php),
install `calibrex[cuda]`, check out that current revision and run:

```bash
python -m tools.benchmark_cuda_map_preparation \
  --sequence path/to/2011_09_30_drive_0018_sync \
  --baseline-commit 22c19ebc75ec61715c83a2e77e9c3e8eea34dffc \
  --half-window 1 --repetitions 3 --output-dir outputs/map-registration-cold
```

Use a new output directory. For retained-cache measurements, use
`--compiler-cache retained --repetitions 4` and a separate output directory.
All workers use one BLAS thread and verify their actual imports against frozen
complete runtime manifests. Complete wall time includes startup, imports,
input reads/preparation, GPU initialization/compilation, all evaluation work,
full result saves, hashes and worker source verification. Parent guards and
aggregation are outside the timed process.

The archives retain complete observations/results, both source capsules,
manifests, exact launchers/schema, cache-file hashes and publication byte
bindings. Original operator paths remain in provenance. Raw scans and compiled
binaries are external. Recompute the evidence without a GPU or raw scans:

```bash
python -m tools.audit_cuda_map_preparation docs/assets/cuda_map_registration_fusion/cold
python -m tools.audit_cuda_map_preparation docs/assets/cuda_map_registration_fusion/retained
```

See the full [cold records](../assets/cuda_map_registration_fusion/cold/README.md)
and [retained-cache records](../assets/cuda_map_registration_fusion/retained/README.md).

Final local validation: 83 relevant unit/device/benchmark/archive tests pass,
including FP32/FP64 matrices in C, column-major and strided layouts, native
weighted systems, exact tie/gate selection and empty targets. Full-repository
Ruff, strict mypy over 376 source files and strict MkDocs pass. All 130 schema
and ledger exports are byte-identical to the measured baseline. CPU-only mutation
tests reject changed fused kernel identities, projection/residual/Jacobian work
and original query counters even after outer file hashes are rebound.
