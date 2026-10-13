# Packed and parallel continuous-time LiDAR preparation

The measured candidate is `7f31335c2ca265a46cd8b02110ad5b8b4e0ac7bb`;
the baseline is `700a88d6827e9b33242d1a8fef3bb72c8f0c4090`, the preceding
compiled CPU implementation. This change accelerates CPU observation preparation
and independent clock candidates. It adds no GPU execution.

Native target points, captured plane geometry and normalized pose rows stay in
owned read-only arrays across correspondence assembly, MAD filtering and numerical
linearization. The public factor still exposes a tuple of native observation objects
on access. Replacing those observations invalidates prepared arrays. Unsupported
points/maps, changed constructors or custom query methods retain the existing path.
The nearest normal-less uniform voxel continues to reject its candidate, and hybrid
selection keeps the original three-product residual and scalar centroid-distance tie.

Each row still evaluates all thirteen central-difference transforms with
`fastmath=false`. Large factors fuse those evaluations per row, retaining the
single-threaded numeric kernel. For at least 16,384 raw targets and an unlimited
sampling cap or a cap of at least 8,192 per split, the independent clock candidates
use up to three owned worker threads. Results are consumed in the original candidate
order. Initial evaluation and final correspondence refinement remain serial.
There are no nested Numba worker pools or caller thread-mask changes. Custom factor,
solver, query or trajectory methods and small sample caps keep the serial profile.
Hessian and gradient additions remain sequential in their original order.
Factor diagnostics consume numeric
linearization directly, avoiding conversion of a large unused Jacobian to Python lists.
The public linearization fields and the custom-method fallback remain unchanged.

Row-only pose batches share the existing 128 MiB LRU payload budget. Their keys
contain full timestamp tuples and exact base-transform bytes. Changed extrinsics
and points always recompute world coordinates. Every logical interpolation and
clamp is counted, including hits. Stricter extrapolation tolerances keep the first
failed prefix; diagnostic overrides keep individual callbacks. For native diagnostics,
the prefix maximum can be recorded in one call.

Adaptive KD distances that are zero or below `1e-150` use a bounded ball of
contenders and exact scalar `math.dist`, preserving inclusive gates and captured
insertion-order ties even when squared distances underflow. This avoids scanning
the broad occupied-voxel neighborhood for every coincident point. Unsupported
queries/maps keep the existing scalar behavior.

The Python default, sampling, holdout, robust weights, damping, acceptance and
iteration budgets are unchanged. The optional `continuous_time_query_backend:
numba_cpu` execution note identifies adapter v0.6 and the Numba/cache/arithmetic policy.
No result schema or additional runtime configuration is introduced.

## Public-data measurements

The [AIST GLIM Livox recording](https://zenodo.org/records/6864654) uses its supplied
trajectory and the tutorial's existing conversion. The second point-cloud topic
copies the first. This is an execution parity control, without an independent
two-sensor extrinsic or clock-accuracy reference.

There are five paired fresh processes per workload, with reversed order every
other pair and a separate initially empty compiler cache for each method. Initial
compilation, every outlier and every slower candidate run remain included. Whole
wall includes startup, imports, replay, optimization, diagnostics, input hashes,
YAML output and measurement saving. Parent verification/aggregation is outside
that wall. CPU diagnostics sum worker threads and exclude startup and measurement
JSON serialization. These are descriptive local timings; the paired ratios and
full ranges show substantial variation.

## Final candidate

| Workload / metric | Baseline median (s) | Candidate median (s) | Ratio of medians |
|---|---:|---:|---:|
| 120 frames, all captured targets: Complete process | 27.095958 | 12.722702 | 2.13× |
| 120 frames, all captured targets: Evaluation API | 24.417539 | 10.244870 | 2.38× |
| 120 frames, all captured targets: Clock and extrinsic refinement | 21.895268 | 7.773029 | 2.82× |
| 120 frames, all captured targets: Worker CPU, all threads | 27.906250 | 18.734375 | 1.49× |
| Default sampling budgets: Complete process | 8.402265 | 6.582169 | 1.28× |
| Default sampling budgets: Evaluation API | 5.652135 | 3.909736 | 1.45× |
| Default sampling budgets: Clock and extrinsic refinement | 4.722846 | 2.980531 | 1.58× |
| Default sampling budgets: Worker CPU, all threads | 9.468750 | 7.703125 | 1.23× |

### 120 frames: all paired complete-process times

| Pair | Order | Baseline (s) | Candidate (s) | Paired ratio |
|---|---|---:|---:|---:|
| 1 | baseline → candidate | 27.885351 | 20.469651 | 1.36× |
| 2 | candidate → baseline | 27.095958 | 12.856092 | 2.11× |
| 3 | baseline → candidate | 23.505286 | 11.067644 | 2.12× |
| 4 | candidate → baseline | 23.168776 | 10.892669 | 2.13× |
| 5 | baseline → candidate | 27.125936 | 12.722702 | 2.13× |

Median of paired ratios: **2.12×**. This differs from the ratio of independently calculated medians above.
Baseline range: 23.168776–27.885351 s; candidate range: 10.892669–20.469651 s.

### Default budgets: all paired complete-process times

| Pair | Order | Baseline (s) | Candidate (s) | Paired ratio |
|---|---|---:|---:|---:|
| 1 | baseline → candidate | 16.589649 | 20.066642 | 0.83× |
| 2 | candidate → baseline | 8.166339 | 6.335960 | 1.29× |
| 3 | baseline → candidate | 8.504036 | 6.582169 | 1.29× |
| 4 | candidate → baseline | 8.339911 | 6.440510 | 1.29× |
| 5 | baseline → candidate | 8.402265 | 7.505207 | 1.12× |

Median of paired ratios: **1.29×**. This differs from the ratio of independently calculated medians above.
Baseline range: 8.166339–16.589649 s; candidate range: 6.335960–20.066642 s.

All 20 final saved/API results match exactly after path-separator normalization
and exclusion of only the declared optional execution identity note. Numeric
values, statuses, traces, counts and all other provenance are retained. The full
capture uses 38,400 training targets and 9,600 temporal holdout targets, with
29,181 and 3,451 final correspondences and rank six. Its status is `max_iterations`;
the configured budget remains unchanged. Default budgets use 6,000 and 3,200
targets, with 4,773 and 2,526 final correspondences, rank six and `converged` status.

## Actual preparation diagnostic

The separate [instrumented diagnostic](diagnostic.json) processed all 52 batches
through the packed path, with 1,248,000 input rows and 1,027,299 correspondences.
It materialized one final batch of 29,181 observations for range diagnostics,
and no preparation batch fell back. All 24 profile jobs used the three owned
workers. Its complete result matches the formal run.
Diagnostic instrumentation and its compiler cache are separate from formal
timings; its runtime is not pooled or used for a speed claim.

## Preserved preceding candidate

The preceding packed-only candidate was measured at
`3e43fc2b6243caf30ff343c20d5da556e7f2600a`. Its complete results, source
bytes, raw timing ranges and diagnostic remain in [the prior archive](prior/publication-index.json).
Its full-capture ratio of medians was 2.51×, while the median paired ratio was
only 1.62×. That difference motivated the further parallelization. The two cohorts
are separate and are not pooled.

The separate [row-parallel intermediate](intermediate.definition.json) was measured
at `e6bef54ee2ac7e57fb9de4f89c66dd8ac287559e` with ten matching full-capture results.
It delivered 21.962 → 13.339 s whole-process medians, but worker CPU medians
increased from 22.828 → 30.844 s. The subsequent
[nested-worker variant](threaded.full_capture.definition.json), measured at
`d631e86e4938fc752f5924df44c570113f4ac617`, used row workers inside candidate workers:
23.750 → 11.244 s whole-process medians and 24.750 → 38.734 s worker CPU medians.
Its complete default-budget cohort and all 20 matching results are retained too.
Those higher CPU costs motivated removing the inner pools. Historical publication
names in `threaded.publication-index.json` use the `threaded.` filename prefix in
the current index; shared preceding evidence stays in `prior/`.

## Preceding candidate: all retained measurements

| Workload / metric | Baseline median (s) | Candidate median (s) | Ratio of medians |
|---|---:|---:|---:|
| 120 frames, all captured targets: Complete process | 35.214933 | 14.044282 | 2.51× |
| 120 frames, all captured targets: Evaluation API | 32.608844 | 11.685373 | 2.79× |
| 120 frames, all captured targets: Clock and extrinsic refinement | 30.363044 | 9.492106 | 3.20× |
| 120 frames, all captured targets: Worker CPU, all threads | 35.343750 | 15.015625 | 2.35× |
| Default sampling budgets: Complete process | 17.214709 | 11.440757 | 1.50× |
| Default sampling budgets: Evaluation API | 10.732289 | 6.025165 | 1.78× |
| Default sampling budgets: Clock and extrinsic refinement | 8.739809 | 4.656662 | 1.88× |
| Default sampling budgets: Worker CPU, all threads | 16.671875 | 12.265625 | 1.36× |

### 120 frames: all paired complete-process times

| Pair | Order | Baseline (s) | Candidate (s) | Paired ratio |
|---|---|---:|---:|---:|
| 1 | baseline → candidate | 35.214933 | 24.327710 | 1.45× |
| 2 | candidate → baseline | 22.197032 | 13.734447 | 1.62× |
| 3 | baseline → candidate | 21.758166 | 13.627698 | 1.60× |
| 4 | candidate → baseline | 42.812154 | 14.044282 | 3.05× |
| 5 | baseline → candidate | 38.670354 | 20.974921 | 1.84× |

Median of paired ratios: **1.62×**. This differs from the ratio of independently calculated medians above.
Baseline range: 21.758166–42.812154 s; candidate range: 13.627698–24.327710 s.

### Default budgets: all paired complete-process times

| Pair | Order | Baseline (s) | Candidate (s) | Paired ratio |
|---|---|---:|---:|---:|
| 1 | baseline → candidate | 22.594933 | 20.059047 | 1.13× |
| 2 | candidate → baseline | 7.790854 | 5.807560 | 1.34× |
| 3 | baseline → candidate | 10.768910 | 11.440757 | 0.94× |
| 4 | candidate → baseline | 18.181636 | 24.211891 | 0.75× |
| 5 | baseline → candidate | 17.214709 | 7.443218 | 2.31× |

Median of paired ratios: **1.13×**. This differs from the ratio of independently calculated medians above.
Baseline range: 7.790854–22.594933 s; candidate range: 5.807560–24.211891 s.

## Evidence and reproduction

[The publication index](publication-index.json) binds measured source snapshots,
ten archived baseline modules, exact saved YAML results, portable measurement
records, complete schema-valid definitions/aggregates, configuration snapshots
and conversion lineage. Machine-local command/module locations are replaced by
portable placeholders. Complete calibration payloads and all timing values stay
unchanged; original raw measurement/definition hashes are retained.
Source snapshots retain the actual measured checkout bytes, including their line endings.

Install the optional `calibrex[numba]` dependency, verify/convert the public
inputs as described in the [tutorial](../../tutorials/gpu_acceleration.md), and
check out the measured candidate. Use new output directories for each run:

```bash
python tools/benchmark_adaptive_plane_search.py \
  --baseline-scope compiled \
  --baseline-commit 700a88d6827e9b33242d1a8fef3bb72c8f0c4090 \
  --config examples/public_datasets/glim_versatile/online_continuous_time_compiled_full_capture_config.yaml \
  --repetitions 5 --output-dir outputs/packed-full-rerun
```

Repeat with `online_continuous_time_compiled_queries_config.yaml` and a second new
output directory for the default budgets. The archived publication and diagnostic
launchers preserve the published cohort directory names; choose fresh directories
when reproducing them. All source, input and HEAD guards must pass.
