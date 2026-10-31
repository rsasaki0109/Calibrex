# Wrong surfaces can have low and ambiguous photometric costs

Frozen `1812d061`, evaluated2026-10-30. See
[protocol](stereo_ambiguity_protocol.md),
[plan](../assets/stereo_ambiguity_20261030/plan.json),
[all18 diagnoses and depth strata](../assets/stereo_ambiguity_20261030/evaluation.json)
and [schema](../assets/stereo_ambiguity_20261030/evaluation.schema.json).

Wrong-normal regions more often have ambiguous image appearance, but the fixed
photometric probes do not cleanly separate correct and incorrect surfaces. In
11/18 frames, the wrong-normal group's median base photometric cost is LOWER
than the agreeing group's. Small left/right image difference cannot be used as
a correctness certificate for the reconstructed surface.

| Per-frame pooled-depth statistic, median across18 frames | Normal agrees | Normal disagrees |
| --- | ---: | ---: |
| Competing disparity within .5gray units of base | 32.48% | 44.56% |
| Alternative better by >1gray unit | 2.27% | 3.95% |
| Patch texture standard deviation <=5 | 45.29% | 48.38% |

Wrong-group ambiguity exceeds agreeing-group ambiguity in13/18 frames; the
alternative-better rate exceeds it in15/18, and low-texture rate in10/18. These
associations do not establish a calibrated confidence threshold or causal effect.
Depth distributions, surface type and visibility differ between labeled groups;
per-depth-band counts/statistics are retained rather than treating pooled medians
as depth-controlled evidence. Sparse far-depth strata limit inference there.

## The three persistent failures

| Frame | Agreeing median image cost | Wrong-normal median cost | Agreeing ambiguity | Wrong-normal ambiguity |
| --- | ---: | ---: | ---: | ---: |
| 0005/frame38 | 2.6720 | 2.2096 | 29.79% | 40.61% |
| 0005/frame114 | 3.2288 | .9456 | 28.61% | 53.29% |
| 0009/frame111 | 3.4496 | 1.4720 | 34.16% | 52.69% |

Values are zero-mean5x5 mean absolute gray differences, NOT SGBM's regularized
cost. In frame114, wrong-normal patches have median texture std3.03 versus6.58
for agreeing patches. At frame111 it is5.43 versus7.23. Matching a nearly uniform
region can yield low residual under multiple disparities. These local probes
cannot identify which disparity is physically correct or whether stereo, LiDAR
normals, reference/time assumptions or occlusion cause each wrong surface label.

## Scope and verification

Exactly121636 frozen selected-grid samples were considered;120010 have common
valid photometric support across all nine offset probes. Base disparity>8px and
remap/border constraints deliberately remove some far/border samples. No claim
covers excluded points. Original raw SGBM correspondences, not smoothed-plane
disparity, are probed. The source depth bands use original stereo optical z;
normal labels come from frozen offline LiDAR/reference maps only after selection.

All18 recomputed restored SGBM arrays reproduce the archive exactly. Four focused
tests, repository Ruff and strict documentation build pass. Checks validate schema,
source/input hashes,18-case completion, positive/common-candidate support and
explicit empty groups. Tests cover an exact known image shift, flat-patch ambiguity
and the map population denominators. No cost-selected depth, new mask, optimizer,
production default or calibration verdict changes. No SOTA claim.

## Development decision

Close the current stereo-surface confidence/filter branch. Its successive local,
reverse and temporal consistency tests cannot certify all18 frames, and the new
appearance probes show that low cost itself is not a sufficient replacement.
Do not tune SGBM or choose gray/texture thresholds using these labeled outcomes.
Keep the existing full-cloud MI6/12 as the real-data baseline.

The next materially different direction is a review of direct cross-modal RGB–LiDAR
correspondence providers beyond the already rejected SuperGlue/I2PNet exports.
Before execution compare architecture, reproducible checkpoints, license/isolation,
training overlap and CPU/GPU availability, then select one feasible, explicitly
frozen spent-data pilot with geometric-support and pose-recovery accounting.
Do not consume fresh holdouts, repeat the rejected provider's threshold sweep or
claim a new provider works merely because runtime is available. If no compatible
candidate is feasible, state the resource/evidence prerequisite instead of
continuing these local confidence variants.
