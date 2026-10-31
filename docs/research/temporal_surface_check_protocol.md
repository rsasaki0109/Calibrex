# Temporal verification of predictive surfaces

Pre-registered2026-10-30. First validate reprojection/visibility on18 exact render
controls: three known sphere layouts, camera translations ±.1m along each axis,
source grid every4pixels within10px border. Render target independently by raycast.
Known true visible source points must be >=95% temporally consistent at the fixed
1px disparity difference threshold; all18 must pass before real diagnosis.
This validates projection/frame direction and smooth sampling, not estimated
motion noise, all occlusion boundaries or full temporal accuracy.

Real: all18 centers, two adjacent neighbors each, all36 previously estimated motions.
Transform measured predictive surface XYZ into neighbor camera. Bilinearly sample
raw stereo neighbor depth, requiring all four finite positive samples. Compare
fB/z_target_projection to fB/z_neighbor, fixed absolute difference<=1px. Record
invalid projection, out-of-view, missing depth, consistent, nearer observed surface,
farther observed surface and unsupported motion. Nearer surface is an occlusion
candidate, not certified bad geometry; do not call all unverified points outliers.

Only points consistent in BOTH neighbors are temporally verified. If either motion
pair failed its frozen internal gate, keep unsupported status and no verified points
for that center. No truth-based fallback or pair exclusion. Never use reference
extrinsics to transform temporal points. Surface selection/motion share stereo
measurements: temporal consistency is not independent ground truth and can miss
coherently wrong depths. Keep common camera-frame/stereo assumptions explicit.

Score AFTER temporal classification with frozen reference-only distance/normal maps.
Report baseline, verified and unverified populations, domain retention, cells,
normal pair counts, distance and normal failure rates, with separate denominators.
Quality continuation requires every center >=10% domain,8cells,1000normal pairs,
within1m>=80%, normal agreement>=80%; empty/unsupported centers fail, precision null.
Compare all18 centers including three earlier surface failures and0013/107 with
both unsupported motions. No post-result threshold tuning, new images, ICP or SOTA.
Pin source/input/depth/motion hashes before execution; preserve control failures.
