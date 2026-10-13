# Exact batched fixed-rig point-to-plane factor

This is an incremental comparison against Draft PR #155, on the existing
`examples/public_datasets/livox_horizon_horizon_pcd_sample/online_accumulation_config.yaml`.
It runs the normal offline calibration API on the official public Livox PCD pair.
The config selects `native_lidar_point_to_plane`, limits target points to 4,000,
and permits 40 iterations. Both methods use the same 2,732 fixed correspondences
and complete nine optimizer iterations (one accepted, seven rejected, then
convergence). No configuration, point budget, finite differences, correspondence,
probe, save or quality threshold is removed or relaxed.

Despite the config filename, these are **offline PCD runs**, not online rosbag
replay. This workload is different from the earlier ICP/GICP/NDT comparison;
its timings and ratios must not be combined with those cohorts.

## Complete process and optimizer subset

Five fresh-process pairs, alternating order, no warmup and no outlier exclusion.
All ten complete canonical results match, including estimated transforms,
finite-difference diagnostics, optimizer history, evidence, grades and provenance.

| Metric (seconds) | Baseline median | Batched median | Ratio of medians | Baseline range | Batched range |
|---|---:|---:|---:|---:|---:|
| subprocess_wall_seconds | 12.383584 | 5.723871 | 2.16x | 11.470166–14.037344 | 5.029327–6.752677 |
| pipeline_seconds | 10.011195 | 3.421480 | 2.93x | 9.233168–11.354264 | 2.890714–4.214353 |
| optimizer_seconds | 6.369221 | 0.524065 | 12.15x | 6.007813–7.586272 | 0.426406–0.658440 |

| Pair | Execution order | Baseline whole process (s) | Batched whole process (s) |
|---|---|---:|---:|
| 1 | baseline → batched | 14.037344 | 5.466299 |
| 2 | batched → baseline | 11.470166 | 5.029327 |
| 3 | baseline → batched | 12.732146 | 6.752677 |
| 4 | batched → baseline | 12.191389 | 5.723871 |
| 5 | baseline → batched | 12.383584 | 5.822800 |

Whole-process wall includes interpreter/import startup, config, all input reads,
inspection, candidate plus all 24 known-bad controls, correspondence construction,
native refinement, diagnostics, input hashing, reports, results, artifacts and
worker measurement JSON writes. It excludes parent audits and CLI dispatch.
The API timer is a subset starting at `run_calibration`. The optimizer timer wraps
the existing `FixedTrajectorySe3ExtrinsicSolver.solve`, including its initial and
final factor diagnostics; subsequent adapter diagnostics are outside that subset.
No additional solve is performed to obtain the timer. The evidence-call subset
is also retained in the definition, with unchanged evaluation code.

The baseline loads only the archived factor module from
`d017dfa03c6275924c8da44f55b2d3707ca58311`. Other source files, solvers, inspection,
correspondence builders, evidence and writers are shared with measured commit
`9cdf1ef43d3dd46a0ae36ef8a68b74967cb56612`. Both methods report the same run Git revision because they execute
in this shared checkout; module bindings and exact source archives identify
the substituted factor independently.

## Numerical contract and quality

At 64 or more observations, cache read-only observation arrays within the factor
and compute residuals in chunks of at most 4,096 observations. Replay the original
Hamilton products and **all four composed-rotation normalizations**, including
the two in point rotation. Preserve the translation composition, ordered dot
products, square-root weights, residual order and numerical finite differences.
Hessian/gradient accumulation, robust weighting and optimizer logic are unchanged.
Small populations and nonfinite input/transform values retain the scalar path.
Replacing the observation tuple refreshes the cache; changing the initial transform
is observed on the next call. Retained arrays cost 17 float64 values per observation
(136 bytes), plus array objects; temporaries are bounded by the chunk size.

Tests compare residual bits for static and per-point moving poses, chunk boundaries,
weights, signed zeros, nonfinite and overflow cases; they compare complete numerical
linearization, observability and solver history against the original scalar formula.

The optimizer still reports `converged`, but observability rank remains **3/6**
and overall quality remains **warn**. Faster execution does not establish full
six-DoF observability or absolute calibration accuracy. This single public capture
is performance evidence; five repeats are not independent accuracy trials.

## Inputs and reproducibility

Official source: [Livox automatic calibration sample](https://github.com/Livox-SDK/Livox_automatic_calibration).
Use the upstream public sample terms. Input files are not redistributed here.

- `base_horizon_100432.pcd`: 769,568 bytes, 23,921 finite points,
  SHA-256 `19b3864cbee76b785d207fcb3fe48f6f506d142d8233f842d0a87fe4330fe407`.
- `target_horizon_100538.pcd`: 761,408 bytes, 23,666 finite points,
  SHA-256 `61a74eb8b8ff4eeba4ca0ff824a4046ae220e947e8156f27d93da452538f5b6f`.
  The same 120 nonfinite header points are excluded by the reader in both methods.

From the measured source checkout, with the public pair downloaded:

```bash
python tools/download_public_dataset.py livox_horizon_horizon_pcd_sample
python tools/benchmark_livox_evaluation.py --workflow fixed-rig \
  --baseline-commit d017dfa03c6275924c8da44f55b2d3707ca58311 \
  --repetitions 5 --omp-threads 1 --output-dir outputs/livox-fixed-rig-rerun
```

Only child `OMP_NUM_THREADS=1` is set; unrelated user activity and OS caches persist.
No other Codex performance tests ran concurrently during the formal cohort.
Host variability and small-sample medians prevent a portable speedup guarantee.

The schema-valid [definition](pipeline.definition.json) retains every trial, its
complete canonical result, input/config/output/artifact hashes and module bindings.
The [aggregate](pipeline.benchmark.json) is recomputed from that definition.
Canonical results normalize only run IDs, timestamps, destination paths and Windows
separators. Publication replaces private workspace prefixes only in commands and
metadata; complete result strings remain byte-identical to the measurements.
`baseline.*.txt` and `measured.*.txt` retain exact Git source bytes, with separate
committed and working-byte hashes. The [publication launcher](publication-launcher.py.txt)
checks each saved result schema, API/save equality and every saved artifact hash
before publishing. Raw per-worker output files remain under
`outputs/public-livox-fixed-rig-batched-v1` in the local workspace.
