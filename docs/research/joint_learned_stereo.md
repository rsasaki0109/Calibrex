# Three-frame joint learned stereo calibration

## Frozen protocol

Follow the [single-frame experiment](learned_stereo_comparison.md) with a matched
three-frame joint D2D test. Use all three already archived frames in each of six
spent recordings, both original seeds, and the unchanged BorerSixDofSolver.
No new recordings, confidence thresholds, parameter tuning or production changes.

[Plan](../assets/joint_learned_stereo_20261029/plan.json) pins code, seeds, frame
configuration and input metadata. Generate both SGBM and UniMatch depths anew
for all 18 image pairs, including the six previously used central frames. Share
image-only depth between seed directions. Each fit receives exactly three
observations, each with its own corresponding cloud; no averaging images/clouds
and no motion registration across frames. Each pair is already temporally aligned
by the archived input preparation, with the same residual timing limitations.

Keep the pinned SceneFlow checkpoint, CPU float32/four threads, native resolution,
300-second per-frame inference budget, rectification and depth conversion from
the preceding study. RGB UniMatch versus grayscale SGBM is disclosed. Reference
extrinsics enter only scoring after optimization. Record provenance, all 24
solver outputs, per-frame depth coverage and inference times. Failure stops with
logs retained, never shrinks the denominator or triggers a smaller-model retry.

Success means convergence plus rotation <0.5 deg and translation <0.20 m. Report
paired full-pose recovery and separate rotation/translation, including worsening
and converged inaccurate fits. These are six dependent development recordings,
not a fresh SOTA evaluation. Compare methods within this run. Historical joint
SGBM 5/12 used different rectification; do not treat that score as matched control.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 /workspace/work/p0-venv/bin/python \
  tools/compare_joint_learned_stereo.py \
  docs/assets/joint_learned_stereo_20261029/plan.json \
  /workspace/work/joint-learned-stereo-20261029 \
  docs/assets/joint_learned_stereo_20261029/evaluation.json
```

## Results

Protocol frozen at `1948a499`. Completed 18 stereo pairs and 24 joint fits.

| Case | SGBM deg | SGBM m | Pass | UniMatch deg | UniMatch m | Pass |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 2011_09_26_0005-positive | 0.8773 | 0.2577 | False | 0.8089 | 0.2525 | False |
| 2011_09_26_0005-negative | 0.3149 | 0.1803 | True | 0.3392 | 0.0952 | True |
| 2011_09_28_0002-positive | 0.1835 | 0.0400 | True | 0.6993 | 0.1625 | False |
| 2011_09_28_0002-negative | 0.5553 | 0.3536 | False | 0.3034 | 0.0664 | True |
| 2011_09_26_0009-positive | 0.9009 | 0.2658 | False | 0.9240 | 0.2440 | False |
| 2011_09_26_0009-negative | 0.1985 | 0.1250 | True | 0.3976 | 0.2043 | False |
| 2011_09_26_0011-positive | 0.1371 | 0.0468 | True | 0.2318 | 0.0954 | True |
| 2011_09_26_0011-negative | 0.2782 | 0.3197 | False | 0.3180 | 0.2610 | False |
| 2011_09_26_0013-positive | 0.6802 | 0.3085 | False | 1.0978 | 0.3113 | False |
| 2011_09_26_0013-negative | 0.8442 | 0.4727 | False | 0.9745 | 0.2500 | False |
| 2011_09_26_0015-positive | 0.3044 | 0.1281 | True | 0.3227 | 0.1031 | True |
| 2011_09_26_0015-negative | 0.1214 | 0.0904 | True | 0.3358 | 0.0710 | True |

### Interpretation and decision

Recovery is **SGBM 6/12, UniMatch 5/12**. All 24 fits converged; six SGBM and
seven UniMatch outputs nevertheless fail the accuracy gates. UniMatch improves
translation in 8/12 (median 0.2190 → 0.1834 m), but rotation in only 2/12
(median 0.3096 → 0.3684 deg). It gains 0002-negative but loses 0002-positive and
0009-negative. The latter translation is 0.2043 m: retain the original 0.20 m
gate rather than relabeling this near miss.

Both methods improve their own recovery count over the single-frame experiment
(2/12 each), but the learned model does not beat the matched joint SGBM control.
All six central-frame depth outputs reproduce the prior experiment byte-for-byte
for both methods, checking that the additional observations, rather than a
changed central frontend, distinguish these experiments. This does not isolate
which extra frame contributed or constitute independent validation.

UniMatch provides more valid depth in all 18 frames. Forward inference takes
12.66–15.30 seconds per image pair on the configured CPU; no failures/retries.
Greater coverage and better median translation are insufficient for full-pose
accuracy. No production defaults, confidence thresholds or acceptance gates change.

[Full evaluation](../assets/joint_learned_stereo_20261029/evaluation.json) and
[report schema](../assets/joint_learned_stereo_20261029/evaluation.schema.json)
retain all cases, three frame identifiers per fit, source hashes, output hashes,
solver results and runtime records. Relevant eight unit tests, repository Ruff,
strict docs build and report validation pass.

### Next substantive question

Close the sequence of stereo substitution/aggregation trials. Before another
model or threshold variant, test whether the fixed D2D solver can recover all
six pose components with independently generated, known-geometry camera depth
and LiDAR, including controlled occlusion and non-planar surfaces. Separate
exact-depth and perturbed-depth conditions with predeclared seeds and gates.
This must exercise the same three-observation path and distinguish optimizer/
geometry limits from stereo measurement error. It is a controlled diagnostic,
not a replacement for real-data validation or proof that stereo is the sole cause.
Only then choose the next algorithmic change; no further spent-data matcher tuning.
