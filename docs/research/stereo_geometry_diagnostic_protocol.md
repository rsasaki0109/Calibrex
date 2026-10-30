# Offline reference-pose geometry overlap diagnostic

Pre-registered 2026-10-30. Reuse all18 prepared stereo/LiDAR pairs after the
108-fit ICP pilot. No new registration fits, data or thresholds fitted to outcomes.
Transform source into camera coordinates using reference only for offline diagnosis.
Record nearest-target distances for all source points and for positive-depth points
inside the original image rectangle. Quantiles .1,.5,.9,.99 and fractions within
.2,.5,1 m; retain both denominators. Record mutual nearest pairs within 1 m and
reverse target-to-source distance quantiles. Pixel-valid depth support is not the
same as field-of-view membership; do not conflate the two.

Estimate both clouds' local normals with the existing Chen-Medioni implementation:
12 neighbors, normalized eigengap .02. For camera-FOV source points within 1 m,
where both normals are valid, record sign-invariant angle acos(abs(dot)), quantiles
.1,.5,.9, fraction <=15 degrees and explicit normal-pair count. Local planes are
not ground-truth surfaces; agreement is not correspondence correctness, covariance
or global observability. Unmatched/weak neighborhoods remain explicit.

Keep all18 frames and freeze code/input/reference hashes before evaluation. The
same distance gate as ICP is a diagnostic threshold, not a newly selected acceptance
rule. Report per-frame and across-frame ranges/medians, avoiding point-level
pseudo-independent evidence. No claim that this proves a unique failure cause.
No reference-filtered solver population, iteration extension or SOTA claim.
