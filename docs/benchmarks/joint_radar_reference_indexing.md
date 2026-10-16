# Indexed reference lookup for joint Radar calibration

The complete prepared-input seven-DoF Radar calibration workflow changes from
**49.293604 to 11.866233 s (4.154107x)**.
The solver API changes from **45.601982 to 9.320259 s
(4.892780x)**. All numerical output, five optimizer iterations,
eligibility decisions, rank diagnostics and fourteen holdout probes match
exactly across all six runs. This is automatic CPU acceleration through the
existing joint solver and RMSE APIs.

## Workload and scope

The input is byte-identical to the preceding [profiled-Radar control](radar_reference_indexing.md):
all 2,762 OXTS reference samples of public KITTI raw drive
`2011_09_30_drive_0018_sync`, with 2,758 generated Radar velocities, deterministic
0.0001 m/s sinusoidal noise and unequal weights. Generation uses a declared
17-degree yaw, `[0.42, -0.18, 0.27]` m lever arm and 0.023 s offset.
**The Radar velocities are generated execution controls.** These measurements
establish performance and parity, not independent Radar accuracy, covariance
coverage, LiDAR pipeline speed or a SOTA result. No locked accuracy cohort is
opened or updated.

The joint solver starts from an identity SE(3) and zero offset, and fits all
six extrinsic components plus time. Its original 3 m/s initial-residual gate
retains 2,099 measurements and rejects the same 659 IDs in every run. The
unchanged default settings retain an 80-iteration limit, bounded clock,
finite-difference Jacobians, Huber LM, grouped train/holdout split, complete
observability diagnostics and fourteen signed calibration probes. Each run
converges after five iterations and makes 171,502 interpolation calls.
No measurement budget or numerical stopping rule is reduced.

## Change

Each residual/Jacobian evaluation previously searched the reference from the
first timestamp. The solver now prepares an immutable, finite, strictly
increasing reference tuple and uses binary search to select the same interval.
Lookup per evaluation changes from O(N) to O(log N), with O(N) preparation
work beyond the existing sort. Existing interpolation arithmetic evaluates
the selected pair, including the left interval at exact knots.

Both the solve and public RMSE path use the index. Duplicate or non-finite
reference clocks retain the original scan semantics in the RMSE path; solve
input validation keeps its existing rejection rules. The index is owned by
one call, with no cross-solve cache. Public signatures, numerical operations,
factor weighting, results and public schemas retain their existing contracts.

## Complete fresh-process measurements

Three alternating pairs use frozen source trees, real Git identities, the
same interpreter/dependencies and one BLAS thread. The host was unreserved;
all timing variation is retained. Whole-process time includes startup, JSON
parsing, construction, initial-residual gating, the complete optimizer,
interpolation counting, all diagnostics, serialization and saving. Input
generation, downloading, freezing and external auditing are untimed.

| Pair (execution order) | Prior whole s | Indexed whole s | Paired ratio |
| --- | ---: | ---: | ---: |
| 1 (prior, indexed) | 45.545767 | 11.563787 | 3.939x |
| 2 (indexed, prior) | 53.370424 | 11.866233 | 4.498x |
| 3 (prior, indexed) | 49.293604 | 16.945178 | 2.909x |

The reported **4.154107x** is the ratio of the two three-run
medians and applies to this workload and reference length. No slow run is
discarded or relabeled as a warmup. The baseline is
`2204c39ffc6f661e44972189ec55007f9e5ab8f0` and measured runtime is
`c90d226b30a02c840876f9d897d522633e75653a`; only the joint Radar solver changes in the
complete runtime manifests.

## Parity, validation and provenance

Every run saves its complete result, optimizer history, weighting contracts,
accepted/rejected steps, initial rejected IDs, all rank/condition and family
diagnostics, holdout values and probes. Numerical values receive no tolerance
or normalization. Source commits are recorded separately in the provenance
wrapper. Interpolation call counts also match in all six workers.

Regression coverage includes exact floating-point interpolation bytes at
knots/nextafter neighbors, endpoints, epoch-sized times, empty/single-sample
references, duplicate/non-finite clock fallback, complete weighted optimizer
history and initial-rejection parity, and logarithmic reference access.
The related suite passes 91 cases; fourteen dedicated joint-archive checks
exercise complete parity and tamper rejection. Strict core typing, Ruff,
documentation and public schema/ledger export parity are checked separately.
The original profiled-Radar archive keeps its identical audit result.

The [complete archive](../assets/joint_radar_reference_indexing/README.md)
contains original input/generator identities, source capsules/manifests, all
six workers, full results, timings and validation receipts. The input generator
and measured launcher are separately bound so the previous control can be
reused byte-for-byte. Derived numeric data retains KITTI attribution and
CC BY-NC-SA 3.0 terms separately from Apache-2.0 source code.

## Reproduce

Audit without calibration, a raw download or a GPU:

```bash
python -m tools.benchmark_radar_reference_indexing \
  --audit-only docs/assets/joint_radar_reference_indexing
```

Remeasure using the exact committed control and its original generator:

```bash
python -m tools.benchmark_radar_reference_indexing --solver joint \
  --baseline-commit 2204c39ffc6f661e44972189ec55007f9e5ab8f0 \
  --control docs/assets/joint_radar_reference_indexing/control.json \
  --control-generator docs/assets/joint_radar_reference_indexing/control-generator.py.txt \
  --output-dir outputs/joint-radar-new --repetitions 3
```
