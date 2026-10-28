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
