# Camera–LiDAR: does a stronger baseline search recover pitch detection?

**Result: reject this proposal for adoption.** On all fourteen reacquired spent
inputs, an additional training-only baseline search increases detected controls
from 19/42 to 26/42, but the worst active-axis reference disagreement among the
saved folds of detected controls increases from 0.350 to 8.165 deg. The targeted
0002 pitch +3 control still fails. These are control and training-fold diagnostics,
not native calibration verdicts or a new SOTA audit.

## Hypothesis and fixed ablation

The prior training-only initialization experiment missed pitch +3 on KITTI raw
2011_09_28_0002. Holding nuisance axes fixed detects its pitch control at
`t=3.38157`; allowing roll/yaw to refit reduces it to `t=0.77397`. A post hoc
inspection found that two folds preferred wrong-pitch poses on held-out frames,
while one fold even had a worse *training* baseline than its refitted controls.

Test whether that baseline search deficit explains the miss. For each fold,
first compute the existing baseline and both signed, nuisance-refitted controls.
Release the tested axis of each control and run the existing rotation search on
training frames only. Keep the best of the original baseline and the two refined
poses, retaining the original on ties. Then recompute both controls around the
selected baseline. This is one bounded pass, not repeated until a desired result.
Seeds are clipped to the existing rotation bounds before baseline refinement.
Held-out frames score only the five final poses; they never select a restart.

Source `f1ff144f` was frozen before the batch in an isolated worktree. The only
ablation is `--control-seeded-baseline`, on top of `--training-only-start
--multi-start`. The scalar objective, settings, six folds, 2 deg controls and
2.5 t threshold remain unchanged. Both arms run on every baseline and +1/+3 deg
axis injection for the same two spent drives. The original arm reproduces the
prior numerical evidence within absolute `1e-9` on all fourteen inputs.
No fresh recording was read; capture progress remains 2/55.

## Results

The [complete report](../assets/control_seeded_baseline_20261028/report.json) and
[summary](../assets/control_seeded_baseline_20261028/summary.json) retain all arms,
folds, input digests, timings and the focused pitch score replay. Every input
completed; none was excluded.

| Diagnostic | Original training-only starts | Additional baseline search |
| --- | ---: | ---: |
| Detected controls, 0005 | 19/21 | 20/21 |
| Detected controls, 0002 | 0/21 | 6/21 |
| Total detected controls | 19/42 | 26/42 |
| Worst active-axis fold reference disagreement among detected controls | 0.350 deg | 8.165 deg |
| Target pitch +3 control t | 0.77397 | 0.84766 |
| Target pitch +3 control detected | No | No |
| Total profile runtime | 138.1 s | 186.8 s |

The strongest counterexample is **0002 yaw +3, pitch-axis control**. It passes
at `t=2.71277`, but one training-fold pose disagrees with the reference in pitch
by -8.165 deg. More detected controls therefore do not establish better calibration.
The forty-two correlated controls are not independent recordings. Runtime is one
sequential run per input, not a repeated performance benchmark.

For the targeted pitch +3 input, the extra search eliminates the observed
training-score deficit: all six selected baselines beat both wrong-pitch controls
on their training frames. Yet on held-out frames, only three folds beat both
controls, versus four in the original arm. Its maximum active-axis fold reference
disagreement stays about 0.15 deg. A small reference error at a fitted pose does
not make its displacement detectable under nuisance ambiguity.

## Decision

This experiment rules out the tested baseline-search deficit as a sufficient
explanation or remedy for the pitch miss. It does not prove global optimality or
identify the physical cause of every ambiguity. The result is consistent with
spurious edge alignments scoring well under the scalar objective, across different
training/held-out scene portions. Dataset calibration remains reference evidence,
not ground truth.

Keep the option research-only and off by default. Do not tune thresholds to
accept the extra detections. The next hypothesis should address correspondence
quality: actual LiDAR boundary geometry versus image-edge direction, rather than
more search or the previously unsuccessful projected-azimuth orientation proxy.
Test it on the same full denominator with reference disagreement beside detection.

Validation: 24 focused tests passed, including escape from a separated basin,
unchanged tie behavior, bounded seeds, held-out-only scoring, and archived evidence
compatibility. All experiment reports were schema-validated and publication files
are digest-bound. No production estimator default or historical audit changes.
