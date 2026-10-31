# Image-only gated depth sampling improves bias accuracy, with regressions

Frozen at `8a9babf4` before evaluation on 2026-10-30. See
[protocol](edge_sampling_protocol.md), [plan](../assets/edge_sampling_20261030/plan.json),
[all 63 results](../assets/edge_sampling_20261030/evaluation.json) and
[schema](../assets/edge_sampling_20261030/evaluation.schema.json).

The new sampler uses only depth-image values and projected pixel coordinates.
It prevents interpolation across a >.2 m jump from the nearest pixel's depth.
Solver function definitions are AST-identical to the seven-parameter bilinear
baseline; only the sampler import changes. No scalar rescue is included.
Baseline records are archived, source/hash pinned, not rerun in this experiment.

| Sphere controls | Bilinear baseline | Image-only gated sampler |
| --- | ---: | ---: |
| Perturbed pose+bias recovery | 12/36 | 32/36 |
| Truth-start pose+bias recovery | 6/18 | 10/18 |
| Perturbed pose-only recovery | 36/36 | 36/36 |
| Truth-start pose-only recovery | 18/18 | 18/18 |
| Median absolute bias error, all 54 | .055900 m | .002071 m |
| Numerical convergence, all 54 | 54/54 | 54/54 |

Across sphere controls there are 26 gains and **two regressions**. Both regressions
are staggered-layout negative-bias truth starts (visible and all-return), now
converging with bias errors about .248 m. Numerical success and local rank do not
certify accurate correction. Total joint recovery by layout: original 12/18,
near_foreground 14/18, staggered 16/18. Zero-bias controls recover jointly in 18/18;
positive and negative bias each recover 12/18.

All 12 remaining sphere failures fail the bias gate, while meeting pose gates:

- Four +.25 m positive-seed fits: original and near_foreground, both populations;
  bias errors .165–.179 m.
- Two original +.25 m truth starts: .05090/.05103 m error, still failures under
  the unchanged .05 m threshold.
- Six -.25 m truth starts: .2478–.2495 m error, including the two regressions.

All nine parallel-plane controls remain flagged as locally degenerate and are
not accepted jointly. Actual optimizer calls are 8–201 across all 63 cases,
within the fixed 800 budget. Runtime including diagnostics is 18.89 seconds;
this is not a controlled runtime comparison. All cases remain in the report.

Ten focused tests pass; Ruff and strict documentation build pass. Validation
also checked report schema, frozen hashes, full case count, solver AST equivalence
and call bounds. The tests cover depth-step separation, additive bias equivariance,
smooth-patch equivalence, range-to-z conversion and invalid-neighbor penalties.

This is substantial synthetic evidence that boundary mixing explains much of
the bias offset without requiring a scene-aware sampler. It is not real-data or
SOTA evidence. Hard selection still creates discontinuities and the two
regressions block unqualified adoption; core/defaults are unchanged. The .2 m
threshold is fixed, not tuned after the evaluation, and may not suit noisy depth.

Next test the already frozen residual-triggered scalar rescue on this image-only
sampler in a separate comparison. Keep the six negative-bias truth starts, four
positive-seed failures, two near-threshold failures and all plane controls. Count
both stages under the same budget, retain baseline fallback and report staged
termination rather than claiming joint stationarity. Do not tune the sampler or
relax gates while testing rescue; real-data validation remains required later.
