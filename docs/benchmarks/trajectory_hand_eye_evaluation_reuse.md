# Reusing trajectory hand-eye evaluations

Complete prepared-input INS-LiDAR evidence evaluation, including all block
jackknife fits and held-out controls, changes from **101.600171 to
28.628948 s (3.548862x)**. The prepared evaluation
phase changes from **98.410585 to 24.926606 s
(3.948014x)**. All six complete results, every residual
evaluation and every least-squares result match exactly. This is automatic
CPU reuse with existing dependencies.

## Workload and scope

The reference is all 2,762 original OXTS poses of KITTI
`2011_09_30_drive_0018_sync`. The 2,742 sensor poses at indices 10-2751 are
generated from that reference with a declared extrinsic, 0.012 s clock offset,
1.02 reference scale and seeded rotation/translation noise. The existing
default motion builder creates all 788 block-contained motions at steps
5, 10 and 20. Five-second blocking yields 38 training blocks and 19 holdout
blocks, with the original one-in-three holdout policy.

**Sensor motions are generated execution controls.** This measures prepared
trajectory calibration/evaluation and numerical parity, not independent
calibration accuracy, raw LiDAR odometry speed or a SOTA result. No locked
accuracy cohort is opened. Original OXTS bytes, original download provenance
and the exact generator are archived, and the audit regenerates every pose
and motion. The declared reference transform enters evaluation only.

Both variants run the main multi-start calibration, all 38 leave-one-block-out
fits, analytic/jackknife uncertainty, every held-out known-bad control and
policy classification. All yaw/axis-alignment starts, five IRLS passes,
clock/scale bounds, finite-difference steps, Huber/noise model, priors,
observability thresholds and stopping behavior remain unchanged.

## Reused work

Each solve prepares its fixed start/end times, sensor-motion matrices and
noise sigmas. A sixteen-state cache reuses reference motions for identical
binary clock values. A 128-state cache reuses normalized data residuals for
identical base transforms and full parameter vectors across optimizer and
IRLS evaluations. Root weights, robust spreads, priors and optimizer anchors
are applied afresh. No parameter is rounded and no candidate is skipped.

Prepared evaluation retains the original SLERP/interpolation calls, matrix
layout/products, scale multiplication, rotation-vector conversion and
normalization order. Returned cached arrays are read-only; binary keys retain
signed zeros. State belongs to one solve and never crosses a jackknife fit.
A conservative 64 MiB allowance at 12 KiB per motion bounds inputs, both
caches and work arrays. Larger inputs retain the original evaluation path.
Public APIs, schemas and defaults are unchanged.

| Observed work per complete evaluation | Prior | Reused |
| --- | ---: | ---: |
| Calibration solves, including jackknife | 39 | 39 |
| Least-squares calls across all IRLS passes | 410 | 410 |
| Requested normalized residual evaluations | 17,446 | 17,446 |
| Batched reference-pose interpolation calls | 35,000 | 6,430 |
| Motion-noise sigma construction calls | 17,460 | 53 |

Every residual request records its fit, motion count, full base/parameter
values and complete residual-vector digest. Every least-squares call records
its parameters, residual/Jacobian dimensions and digests, termination status,
evaluation counters, cost and optimality. The audit independently replays
both cache capacities and validates observed work against the full history.

## Complete fresh-process comparison

Three alternating pairs use frozen complete core sources, identical inputs
and dependencies, actual Git identities and one BLAS thread. The host is
unreserved; no successful run is dropped or relabeled warmup.
Whole-process timing includes imports, actual Git prechecks, prepared problem
loading, construction, complete evaluation and history instrumentation,
public result building/saving, history serialization/compression and
imported-source hashing. The prepared evaluation phase covers problem loading
through result saving, including instrumentation. Data generation, freezing,
raw-input compression and external audits are untimed.

| Pair (execution order) | Prior whole s | Reused whole s | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 (prior, reused) | 144.558286 | 28.628948 | 5.049x |
| 2 (reused, prior) | 99.656693 | 31.291424 | 3.185x |
| 3 (prior, reused) | 101.600171 | 27.744277 | 3.662x |

The **3.548862x** headline is the ratio of three-run medians
for this full-drive prepared control. Baseline:
`823ae70a9edabaeac93a2461ed2dd8361f94fb40`. Measured source:
`d6f1b6dd51c0584f58941618634d07e25965e71d`. Only the trajectory hand-eye solver changes
in the complete core manifests.

An initial incomplete cohort is preserved under `interrupted-v1`. Its prior
worker completed before the comparison was stopped to fix audit handling of
source IDs and omitted optional control fields. It contributes no final
timing. Both variants' complete core manifests and input bytes are identical
between the interrupted and final cohorts. Only measurement/audit handling
changed, with regression coverage; missing results are never invented.

## Validation and provenance

The existing solver/evaluation suite passes 16 cases, prepared reuse adds nine,
and the measurement/replay suite adds twelve. Coverage includes exact residual
bits, enabled/fixed clock and scale, signed zeros, cache eviction, distinct
starts, fresh input/options, empty/budget fallback, complete block evaluation,
optimizer Jacobians, actual abbreviated/full source identities and omitted
optional public fields. Strict mypy checks all 378 core files. Final archive,
registry, Ruff, strict MkDocs and all 129-public-schema-plus-ledger receipts
are retained in the archive.

The [portable archive](../assets/trajectory_hand_eye_evaluation_reuse/README.md)
retains every input/source/helper/schema pin, original OXTS provenance, all six
complete results/histories/timings, validation receipts and interrupted files.
Result comparison excludes only `provenance.created_at` and
`provenance.git_commit`. History comparison excludes its independently
verified full and short Git IDs; protocol/input identities remain compared.
Actual imported bytes and real source identities are checked in every run.
The offline raw-OXTS reconstruction allows at most `2e-8` m translation and
`1e-13` rotation-matrix element differences (`rtol=0`) across math libraries.
Windows versus glibc Mercator reconstruction differed by `9.313225746e-10` m
in a direct full-drive check. Timestamps, motion IDs, input/source hashes,
and all archived result/residual/Jacobian comparisons remain exact. This
portability bound affects only regeneration of geometry during an offline audit.
KITTI-derived data retains CC BY-NC-SA 3.0 attribution separately from
Apache-2.0 code. The complete publication has a byte index.

## Reproduce

Audit without calibration or raw downloads:

```bash
python -m tools.benchmark_trajectory_hand_eye_reuse \
  --audit-only docs/assets/trajectory_hand_eye_evaluation_reuse
```

For remeasurement, extract only regular files of `raw-oxts.tar.gz` into a
fresh directory, preserving member names, and rename
`download-provenance.json` to `provenance.json`. Then run:

```bash
python -m tools.benchmark_trajectory_hand_eye_reuse \
  --baseline-commit 823ae70a9edabaeac93a2461ed2dd8361f94fb40 \
  --problem docs/assets/trajectory_hand_eye_evaluation_reuse/problem.json \
  --control-generator docs/assets/trajectory_hand_eye_evaluation_reuse/control-generator.py.txt \
  --oxts-dir outputs/hand-eye-remeasurement-oxts \
  --output-dir outputs/hand-eye-remeasurement --repetitions 3
```
