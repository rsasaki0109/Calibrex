# Bounded coordinate search on the fixed disparity objective

Pre-registered 2026-10-30. Run all 171 cases against archived direct-disparity TRF,
using identical depth generation/hashes, original seeds and zero bias. Keep all
clean/noisy/occlusion/bias/reference/plane controls. No reference paths enter search.

Replace only optimization with deterministic best-improvement coordinate polling.
Bounds rotation ±2 degrees, translation ±1 m, bias ±.5 m unchanged. Initial steps:
.25 degrees, .1 m translation, .05 m bias. Minimum .01 degrees and .005 m for
translation/bias. Poll axes 0..6, directions - then + at current iterate; clip to
bounds and skip identical candidates. Select the largest measured cost decrease
exceeding 1e-9 over the complete poll. Without improvement halve steps, floored
at minima. Success requires a complete minimum-step poll without improvement.
This is resolution-based termination, not equivalence to TRF stationarity.

Count initial evaluation and every trial under hard 800 actual residual calls.
An incomplete poll can return its best observed point but cannot report success.
If budget expires before termination, keep failure status even if geometry passes.
Use the identical residuals/soft-L1 cost and support/rank gates. Support and local
sensitivity diagnostics are separate from optimizer accounting as before.

Compare joint recovery (.5 degree/.20 m/.05 m), pose-only recovery, convergence,
bias error, gains/regressions, both noise seeds, truth starts separately, all nine
plane rejections and call counts. Baseline archived, not rerun; verify all paired
depth hashes. Preserve negative cases; no post-result step/cap tuning or adoption.
Tests verify actual-call counts, monotonic accepted costs, known smooth optimum
and incomplete-poll rejection. Freeze code/protocol/archive hashes before running.
