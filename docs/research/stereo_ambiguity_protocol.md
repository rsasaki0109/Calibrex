# Fixed stereo appearance ambiguity diagnosis

Pre-registered2026-10-30 after reviewing earlier disparity-band/cycle experiments.
No matcher, depth, plane, mask or motion changes. All18 original real frames.
Recompute exact pinned restored SGBM arrays before probing. Rectify raw grayscale
left/right images identically; use frozen forward disparity at nearest rectified
pixel corresponding to a source-camera pixel. Do NOT substitute smoothed plane
depth for the original stereo correspondence.

Sample the frozen predictive surface domain at every4pixels (top-left crop phase0).
Probe5x5 patches at disparity offsets[-8,-4,-2,-1,0,1,2,4,8]. Mean-center each patch
and use mean absolute grayscale difference (zero-mean SAD), bilinear sampling in
right image. Require positive disparities for ALL candidates (base>8px) and image/
remap-valid support at every candidate; retain sampled versus valid counts. This
common-candidate population excludes far disparity<=8 and border samples; no claims
about their ambiguity. These local costs are NOT SGBM's regularized matching cost.

Report texture std quantiles, base cost quantiles, alternatives>=2px away within
.5 gray units of base (ambiguity), alternatives better by>1gray unit, and texture
std<=5 fraction. Thresholds are fixed descriptive diagnostics, not confidence gates.
Group all, reference-normal-agrees(<=15deg supported), disagrees, distance>1m;
reference labels are scoring only. Stratify original depth(0,10],(10,20],(20,40],
(40,80], plus pooled(0,80]. Do not fit or select a gate from labeled groups.

All18 frames, same tests and no exclusions. Check known synthetic shift and flat
patch ambiguity, archived raw-depth reproduction and hashes. Compare frame-level
statistics, not independent pixels; no causal claim from mere association or
posthoc pooled medians. No refitting, ICP, default change or SOTA claim. If simple
appearance cannot separate failed surfaces, report that rather than retune it.
