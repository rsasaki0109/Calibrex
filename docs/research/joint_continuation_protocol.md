# Baseline-preserving continuation protocol

Follow the failed [equal-split control](joint_multistart_results.md) on the same
frozen synthetic scene, five conditions, two perturbations and separate truth-start
controls. This is an adaptive experiment informed by prior spent synthetic results,
not independent validation. No real data or core/default changes.

Total cap: 800 MI evaluations INCLUDING the original baseline. Run the unchanged
single-start solver first. If fewer than 160 calls remain, return that result.
Otherwise allocate at most 40 new calls to each of the four existing tetrahedral
starts, without changing offsets, objective, histogram ranges or global bounds.
Choose the highest MI explorer (earlier index on ties) for all remaining budget,
even if its current MI is below baseline. Preserve its best point, step sizes and
trace. Stop/pause only between complete coordinate sweeps: never replay a partially
evaluated sweep or count replayed work as free. Unused sub-sweep budgets remain unused.

Return the highest-MI **converged** candidate only if it exceeds baseline MI by
1e-9; otherwise return the exact baseline. No reference-based selection, no new
convergence criterion and no relaxed pose gate. A higher-MI converged candidate
may still worsen physical pose; retain that outcome, especially on truth-start
controls. Selection safety here means preserving a converged objective baseline,
not proving calibration accuracy.

Report all 15 baseline results, all exploratory states and traces, refined and
selected indices (null selected means baseline fallback), total evaluation count,
baseline and additional wall times, and every final pose. Count 10 perturbed
recoveries separately from 5 truth-start controls. Verify recomputed baseline
matches previous results. Unit tests verify resumed/uninterrupted trace equality,
no replay when budget cannot fit a sweep, global bounds and evaluation accounting.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 /workspace/work/p0-venv/bin/python \
  tools/evaluate_joint_continuation.py \
  docs/assets/joint_continuation_20261029/plan.json \
  docs/assets/joint_continuation_20261029/evaluation.json
```
