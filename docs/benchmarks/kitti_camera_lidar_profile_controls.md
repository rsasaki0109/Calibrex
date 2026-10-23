# KITTI nuisance-profiled rotation controls

**Post hoc development, not a new audit or promoted estimator.** This extends
the [conditional-axis investigation](kitti_camera_lidar_v3_development.md#conditional-axis-development-probe)
on the same two spent drives. No new sensor data was acquired. Camera-LiDAR's
v2 refutation, thresholds, standings and existing default remain unchanged.

## Method

`tools/kitti_camera_lidar_profile_probe.py` compares two controls about the
same leave-one-block-out fit. For each axis and fold it starts from the prior
conditional correction and refits all three rotation coordinates on training
frames. It then shifts the tested coordinate by the existing plus/minus 2 deg
control amount. The direct control holds the other coordinates fixed; the
profiled control refits the other two rotation coordinates on **training frames
only**, keeping the tested value and translation fixed. The held-out block
scores these five fixed poses: baseline, two direct shifts, two profiled shifts.
No held-out objective value chooses an optimizer step.

The initialization is inherited from the full-input conditional probe. As with
the existing local jackknife workflow, these are **not nested validation folds**.
The nuisance searches are local and can miss distant optima. Their training
gain does not guarantee an improvement on held-out data. Profile controls can
even have higher held-out t statistics than direct controls. Shifted tested
coordinates may lie beyond the fit bound, as ordinary controls do; a fixed-axis
objective adapter lets nuisance coordinates continue to optimize inside their
own bounds rather than silently skipping all moves.

```bash
python -m tools.kitti_camera_lidar_profile_probe SAVED_SNAPSHOT \
  --conditional-report PREVIOUS_CONDITIONAL_REPORT.json --output NEW_DIRECTORY
```

The tool verifies the previous receipt against the snapshot identity and full
NPZ digest, refuses to overwrite results, emits a JSON schema, and retains
partial results on estimation failure. It uses no reference calibration in
optimization. The fixed batch used source committed at `7a6ea534` before the
experiment. Exact physical source bytes and input/report hashes are recorded.

## Fixed two-drive results

All seven inputs from spent 0005 and 0002 completed: **14 inputs, 42 axis-control
comparisons**, each over six folds. There were no failures. The
[batch report](../assets/kitti_camera_lidar_v3/profiled_controls/report.json)
and [manifest](../assets/kitti_camera_lidar_v3/profiled_controls/manifest.json)
bind all numerical fold poses, drops, schemas and source capsules, about
544 KB excluding the manifest. Images, points, bags and NPZ arrays remain local.
Source bytes and Git HEAD were checked unchanged through the experiment.

| Drive | Direct controls detected | Profiled controls detected |
| --- | ---: | ---: |
| 2011_09_26_0005 | 20/21 | 20/21 |
| 2011_09_28_0002 | 5/21 | 1/21 |
| Total | 25/42 | 21/42 |

These are **control outcomes**, not observed-axis coverage, calibration
accuracy or native injected-turn detection rates. No jackknife-std gate or
native verdict is applied to these counts. The direct comparison is calculated
at these refitted fold poses, not the earlier conditional-only poses.

The four changes on 0002 are pitch controls in baseline, roll +1, pitch +1 and
pitch +3. For pitch +3 the profiled t statistic is about 0.391, below the existing
2.5 control threshold; baseline pitch is about 0.446. The conditional probe's
small pitch uncertainty therefore does not survive this nuisance-aware test.
This confirms a limitation, but does not recover useful detection by itself.

There is also a retained counterexample: on 0002 roll +3, the roll control
passes both versions, with profiled t about 3.158. Its recorded baseline-fold
rotations have a maximum absolute roll error of about **1.5 deg** relative to
the capture reference. That error is computed post hoc from the saved fold
corrections composed with the known injection; it is not a final calibration
result. A detected profiled control therefore does not establish accurate
calibration or resolve the wrong-basin problem.

Synthetic tests distinguish an identifiable axis from a coupled ridge: both
pass direct controls, but only the identifiable axis passes profiled controls.
Other tests check held-out score isolation, agreement with existing direct
control statistics, and nuisance refitting when the tested shift exceeds the
fit bound. Published-evidence tests verify every digest, input binding and
the real-data counterexample.

## Decision

Do not promote this control or tune a new threshold from these two drives.
The next development question is whether broader nuisance searches expose
the remaining wrong basin while preserving useful axis coverage and turn
detection. Any alternative must also address the earlier 0056 false optimum
and the remaining spent-data comparisons. Capture progress stays **2/55**;
53 drives, exposure auditing and a separate frozen preregistration remain.
