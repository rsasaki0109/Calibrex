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

## Multiple nuisance starts

`--multi-start` retains the original local nuisance fit and additionally
profiles each of the other rotation coordinates over the existing configured
range and step. It refines starts at each individual profile maximum and their
combination, removing duplicate starts. Only a strictly higher **training**
objective replaces the original local result; ties retain it. These finite
starts do not establish a global optimum. The unshifted baseline and direct
controls retain their original searches.

Synthetic tests demonstrate escape from a separated nuisance peak and retention
of the local solution on a flat tie. The held-out isolation test runs under
both modes, including controls whose tested coordinate exceeds the fit bound.
The default remains local; the additional starts are a research opt-in.

The follow-up fixed source at `24543be7` before re-running **all 14 cached
inputs**. Every run completed and verified parity with the previous baseline
fold poses, direct shifts and direct-control evidence within `1e-9`. The
[paired batch](../assets/kitti_camera_lidar_v3/profiled_controls_multistart/report.json)
binds each previous local report by SHA-256. The
[publication manifest](../assets/kitti_camera_lidar_v3/profiled_controls_multistart/manifest.json)
binds all new receipts and source bytes, about 550 KB excluding the manifest;
no sensor payload is republished. Source bytes and Git HEAD stayed fixed.

| Drive | Direct controls detected | Local profiled | Multiple-start profiled |
| --- | ---: | ---: | ---: |
| 2011_09_26_0005 | 20/21 | 20/21 | 20/21 |
| 2011_09_28_0002 | 5/21 | 1/21 | 0/21 |
| Total | 25/42 | 21/42 | 20/42 |

The **only changed detection** between local and multiple-start profiling is
the known 0002 roll +3 false control. Its t statistic falls from
`3.1576336192396854` to `1.7323722509750514`, below the unchanged 2.5 threshold.
Its baseline-fold poses and approximately 1.5 deg maximum absolute roll
reference error are unchanged: the new search exposes ambiguity rather than
correcting the fit. The other 41 control detection flags are unchanged,
including every flag on 0005.

This is progress on the selected false-confidence case, **not a demonstrated
overall calibration improvement**. No control passes on 0002, and pitch +3
remains undetected. These are control outcomes at refitted fold poses, not
native verdicts or observed-axis coverage. Neither method is promoted from
this deliberately selected two-drive development set, and no threshold was
retuned after the comparison. Tests bind every paired input, assert unchanged
baseline/direct evidence, and retain both the removed false control and the
still-missed pitch control.

## Decision

The [projected-azimuth orientation probe](kitti_camera_lidar_orientation.md)
adds a signed-safe image-gradient compatibility penalty. On six preselected
cached inputs it loses all eight previously detected scalar controls and does
not recover pitch on 0002. The proxy is not adopted; the separate default-mode
replay retains the previous full scalar numerical evidence within `1e-9`.

The follow-up [fixed spatial-weight experiment](kitti_camera_lidar_spatial_weights.md)
keeps frame totals and margin-point weights while balancing candidate-inside
2x2 tiles. It does not recover 0002 pitch detection and reduces detected 0005
controls from 20/21 to 12/21. The initial margin-support implementation failure
and complete retry are retained; this weighting proposal is not adopted.

Do not promote this control or tune a new threshold from these two drives.
Broader nuisance searches expose the selected wrong basin but do not restore
useful pitch detection on 0002. The next development question is how to change
the evidence or search model while retaining turn sensitivity, rather than
treating additional abstention as accurate calibration. Any alternative must
also address the earlier 0056 false optimum
and the remaining spent-data comparisons. Capture progress stays **2/55**;
53 drives, exposure auditing and a separate frozen preregistration remain.
