# Fixed point-priority translation refinement

## Frozen development comparison (2026-10-29)

This follows the [post-hoc sampling diagnostic](camera_lidar_stereo_translation.md).
Use the exact eight spent-data bundles and D2D initializers of the previous run.
Change only the selection of at most 2,000 eligible pairs per frame: rank the
immutable LiDAR point indices by SplitMix64, take the lowest priorities, then
restore pixel order. Priorities have no pose, iteration or reference dependence.
The hash is deterministic without a tunable seed. Eligibility and pixel identity
can still change, so this does not promise fixed correspondences.

Keep 15 iterations, 0.1 mm convergence, all geometric/noise settings, bounds,
rank and support checks, and fallback behavior unchanged. Default remains uniform.
Freeze source and plan before fitting. Run every case once, score references only
after the fitting subprocess exits, retain failures and raw rejected candidates.
No new evaluation data is used; this is not a SOTA audit.

Compare accepted-stage and returned-pipeline recovery, translation differences
against D2D and the archived uniform candidates, and last-step sizes. Replay the
new saved trajectories using the same pair retention definitions as before;
report the last five recorded updates (or all if fewer), with update counts.
A change in sampling also changes the chosen geometry: improvement cannot be
attributed exclusively to membership stability. No post-result tuning or default
promotion is planned.

Plan and settings: [plan](../assets/stereo_priority_20261029/plan.json),
[options](../assets/stereo_priority_20261029/options.json).
Original uniform artifacts remain immutable. To replay those with their required
source hashes, check out `cbd0288b`; the current diagnostic rejects mismatched
source rather than silently interpreting historical evidence with changed code.


## Result

Source and plan were frozen at `c003dae8`. All eight executions completed.
One case converged at iteration 15; seven reached the unchanged iteration limit
and returned their D2D initializers exactly. Rotation was unchanged in all cases.

- Accepted refinements: **1/8**, versus uniform's 0/8.
- Accepted-stage recovery: **1/8**; returned-pipeline recovery: **4/8**, unchanged.
- The accepted case (0005 negative) **worsens** translation relative to the dataset
  reference from 0.1288 m to 0.1612 m (+0.0324 m); it already met the recovery gates
  before refinement. It creates no newly recovered case.
- Raw candidates improve translation relative to D2D in 6/8, worsen in 2/8.
  Against the archived uniform candidates they also improve in 6/8, worsen in 2/8.
  Seven candidates remain rejected; these are not seven usable calibrations.

Translation columns below are metres of difference from the dataset reference,
not ground truth. Retention and eligible-but-unsampled fractions are arithmetic
means over the last five saved updates, with all cases having 15 updates.

| Recording / seed | Status | D2D | Uniform raw | Priority raw | Selected pairs retained | Still eligible but unsampled |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0005 positive | max_iterations | 0.2773 | 0.1556 | 0.1643 | 94.6% | 0.08% |
| 0005 negative | converged | 0.1288 | 0.1647 | 0.1612 | 97.3% | 0.03% |
| 0002 positive | max_iterations | 0.1064 | 0.1282 | 0.1055 | 66.8% | 0.06% |
| 0002 negative | max_iterations | 0.1821 | 0.0886 | 0.0627 | 49.1% | 0.17% |
| 0009 positive | max_iterations | 0.2569 | 0.1383 | 0.1460 | 90.0% | 0.06% |
| 0009 negative | max_iterations | 0.2915 | 0.1747 | 0.1706 | 94.3% | 0.05% |
| 0011 positive | max_iterations | 0.1358 | 0.3397 | 0.2670 | 96.3% | 0.01% |
| 0011 negative | max_iterations | 0.2751 | 0.1478 | 0.1224 | 61.9% | 0.22% |

Selected-pair retention rises from the previous 6.8–16.2% range to 49.1–97.3%.
Previously selected exact pairs dropped solely by resampling fall from
40.1–75.5% to below 0.3%. Thus the changed rule addresses the measured sampling
turnover. Nevertheless seven cases still fail convergence, and the one accepted
case worsens accuracy. Sampling stabilization is **insufficient** for a useful
translation correction under this protocol. This is not proof that sampling is
irrelevant: changed membership also changes surface coverage and local geometry.

The diagnostic replays every update and verifies fixed-set costs and matcher
outputs against the saved fits. No iteration cap, gate, uncertainty model or
reference-dependent selection was changed after seeing the results. Do not
promote point-priority sampling or claim calibration improvement. The next
substantive investigation is surface/depth consistency and sensitivity to frozen
rotation; simply extending the iteration budget is not supported by this result.

Artifacts: [evaluation](../assets/stereo_priority_20261029/evaluation.json),
[replay](../assets/stereo_priority_20261029/replay.json), and per-case typed outputs
with input/source hashes, candidates and logs. Reproduce in a checkout of the
frozen source with the existing spent-data bundles:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  python tools/evaluate_stereo_translation_refinement.py \
  docs/assets/stereo_priority_20261029/plan.json /tmp/stereo-priority-new
```

For trajectory replay, place the frozen `plan.json` alongside the generated case
directories and run `tools/diagnose_stereo_translation.py OUTPUT_DIR REPLAY_JSON`.
The original uniform experiment requires its original checkout; frozen-source
hash checks intentionally reject use of changed source for historical replay.
