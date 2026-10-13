# Complete public Livox runtime acceleration

Incremental comparisons against Draft PR #156, source 6b30cdf92ce710b0810396deeee45d651fd308d5,
using measured source 726ea13bc77ba3739a159109a9ee21d69b25992a. Each protocol has five fresh-process pairs,
alternating order, no warmup and no discarded timing. The same config and raw bytes
are used by both methods within each protocol. These are offline PCD runs.

## All finite target points

The added native_full_points_config.yaml changes project naming/description/output
path and sets max_target_points: 0, the existing option that removes the target
reader cap. Both methods use all 23,666 finite target points and the full source,
yielding 16,884 correspondences and nine optimizer iterations.

| Seconds, ratio of medians | Baseline | Accelerated | Ratio | Baseline range | Accelerated range |
|---|---:|---:|---:|---:|---:|
| subprocess_wall_seconds | 11.248637 | 5.450934 | 2.06x | 10.847388–12.688656 | 5.122952–5.656675 |
| pipeline_seconds | 8.584811 | 3.551832 | 2.42x | 8.198224–9.329689 | 3.174448–3.790449 |
| optimizer_seconds | 3.918037 | 1.471051 | 2.66x | 3.863537–4.024803 | 1.258962–1.557260 |
| livox_evidence_seconds | 1.432184 | 0.841829 | 1.70x | 1.332660–1.767523 | 0.779626–0.893283 |
| worker_import_seconds | 2.226639 | 1.405307 | 1.58x | 2.099863–2.841722 | 1.376731–1.464249 |
| capture_decode_seconds | 0.788348 | 0.106910 | 7.37x | 0.706385–0.790263 | 0.092221–0.111037 |

| Pair | Execution order | Baseline whole process (s) | Accelerated whole process (s) |
|---|---|---:|---:|
| 1 | baseline → batched | 12.688656 | 5.450934 |
| 2 | batched → baseline | 11.248535 | 5.443949 |
| 3 | baseline → batched | 10.847388 | 5.122952 |
| 4 | batched → baseline | 11.248637 | 5.521191 |
| 5 | baseline → batched | 11.406334 | 5.656675 |

## Unchanged 4,000-point configuration

The existing online_accumulation_config.yaml retains 2,732 correspondences and nine
iterations in both methods. Despite its filename, the measured invocation is
offline calibration. This preceding workload is retained separately; ratios from
the two configurations are not mixed or multiplied.

| Seconds, ratio of medians | Baseline | Accelerated | Ratio | Baseline range | Accelerated range |
|---|---:|---:|---:|---:|---:|
| subprocess_wall_seconds | 7.208035 | 3.969547 | 1.82x | 6.824238–7.358250 | 3.805070–5.128493 |
| pipeline_seconds | 4.387420 | 2.052225 | 2.14x | 4.026760–4.522228 | 1.971759–2.795940 |
| optimizer_seconds | 0.745160 | 0.214198 | 3.48x | 0.575445–0.758798 | 0.193761–0.230729 |
| livox_evidence_seconds | 1.411606 | 0.837807 | 1.68x | 1.347568–1.522706 | 0.811112–1.060923 |
| worker_import_seconds | 2.280053 | 1.446511 | 1.58x | 2.247225–2.316778 | 1.371403–1.668134 |
| capture_decode_seconds | 0.864762 | 0.108102 | 8.00x | 0.626431–0.875154 | 0.100585–0.209898 |

| Pair | Execution order | Baseline whole process (s) | Accelerated whole process (s) |
|---|---|---:|---:|
| 1 | baseline → batched | 7.022649 | 3.805070 |
| 2 | batched → baseline | 6.824238 | 4.008343 |
| 3 | baseline → batched | 7.208035 | 3.969547 |
| 4 | batched → baseline | 7.358250 | 3.946075 |
| 5 | baseline → batched | 7.314292 | 5.128493 |

## Numerical contract

- Share invocation-scoped captured records with quality bins and native input
  loading: reader calls fall from 8 to 2 in every measured run. Cache bins by
  exact edges/sample limit, return copies, and observe replaced inputs next run.
  Scalar square roots, interval order and last-bin inclusivity are preserved.
- Decode checked strided float32 PCD views, retaining first-16-byte interpretation,
  finite-XYZ filtering, intensity bits, normalized normals, declared/available
  counts and small/unsupported-layout scalar fallbacks. Unaligned padding,
  truncated tails, optional normals and signaling NaN payloads are tested.
- Preserve first-occurrence extrema and sequential intensity additions. Reuse
  per-file scalar range/angle values for total geometry, without approximating
  trigonometry or changing addition order.
- Defer scipy.spatial until a valid reference actually needs its tree. Exact tree
  searches and exhaustive fallbacks are unchanged. The accelerated fixed-rig
  workers never load it; every baseline does.
- Run the same 24 controls with at most four CPU workers for larger clouds.
  Candidate-prepared captured data/geometry are shared read-only; each probe owns
  its transform cache. Ordered executor results retain case order. The Python
  probe_workers=1 option selects serial execution. Every accelerated run observes
  one four-worker pool; every baseline is sequential.
- Batch static uniform-plane association with the exact occupied-neighbor lookup.
  Preserve 27-offset ties, normal-less-nearest rejection, gates and observation
  order. Adaptive/moving cases and unsupported extents retain existing paths.
- Compose bit-identical shared poses once per residual evaluation. Retain Hamilton
  products and normalization counts. Weighted/unweighted normal equations use
  bounded 4,096-row chunks with zero seeds and carried sequential additions.
  Product order, Huber weights, finite differences, Hessian/gradient values and
  optimizer decisions match the original; nonfinite cases retain scalar handling.

All ten complete canonical results match within each protocol, including poses,
history, diagnostics, 24 controls, grades and provenance. The two protocols have
different payloads and are not compared for equality with each other. All saved
results validate and match API payloads; every saved artifact hash was audited.
Quality stays warn, observability stays 3/6, and solver status stays converged.
More points and faster execution do not establish full observability or accuracy.

## Scope, inputs and reproduction

Whole-process wall includes interpreter/import startup, config, all input reads,
inspection, candidate/controls, association, full solve, diagnostics, hashing,
results, reports, artifacts and worker measurement writes. It excludes parent
audits and CLI dispatch. API time starts at run_calibration. Optimizer time includes
its initial/final diagnostics; the adapter's later diagnostics are outside it.
Reader, evidence and worker-import timers measure subsets of the same run without
extra calls. Worker-import time excludes interpreter/tool startup before entry.
Only child OMP_NUM_THREADS=1 is set. No other local performance tests ran concurrently.
OS caches and unrelated user work persist; medians are descriptive, not portable
speedup guarantees or independent accuracy trials.

Eight named runtime modules are substituted for baseline; other code and writers
are shared. Source/config/input bytes and loaded module paths are bound and checked.
Both methods report the same checkout Git revision; archives identify the baseline
substitution independently. Working-byte and committed-byte hashes are separate.
Publication changes private workspace prefixes in commands/metadata only; complete
canonical result strings remain unchanged. Only volatile run identity, timestamps,
destinations and Windows separators are normalized in those result strings.

Official source: [Livox calibration sample](https://github.com/Livox-SDK/Livox_automatic_calibration).
Inputs are not redistributed here; use upstream sample terms.

- base_horizon_100432.pcd: 769,568 bytes, 23,921 finite points; SHA-256
  19b3864cbee76b785d207fcb3fe48f6f506d142d8233f842d0a87fe4330fe407.
- target_horizon_100538.pcd: 761,408 bytes, 23,666 finite points; SHA-256
  61a74eb8b8ff4eeba4ca0ff824a4046ae220e947e8156f27d93da452538f5b6f.
  The same 120 nonfinite header points are excluded in both methods.

From the measured source checkout with the public pair downloaded:

    python tools/benchmark_livox_evaluation.py --workflow fixed-rig --baseline-scope runtime --baseline-commit 6b30cdf92ce710b0810396deeee45d651fd308d5 --config examples/public_datasets/livox_horizon_horizon_pcd_sample/native_full_points_config.yaml --repetitions 5 --omp-threads 1 --output-dir outputs/livox-runtime-full-rerun
    python tools/benchmark_livox_evaluation.py --workflow fixed-rig --baseline-scope runtime --baseline-commit 6b30cdf92ce710b0810396deeee45d651fd308d5 --repetitions 5 --omp-threads 1 --output-dir outputs/livox-runtime-bounded-rerun

The all-point [definition](pipeline.definition.json) and [aggregate](pipeline.benchmark.json),
and bounded [definition](bounded/pipeline.definition.json) and
[aggregate](bounded/pipeline.benchmark.json), retain every timing, canonical result,
input/config/output/artifact hash and exact source binding. Archives are shared.
The [publication launcher](publication-launcher.py.txt) checks actual saved results
and artifacts before publishing. Raw outputs remain in the local workspace under
outputs/public-livox-runtime-full-points-v1 and outputs/public-livox-runtime-bounded-v1.
