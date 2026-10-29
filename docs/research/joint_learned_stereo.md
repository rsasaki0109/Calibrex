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
