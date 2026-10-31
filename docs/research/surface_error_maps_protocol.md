# Fixed spatial maps of transferred surface errors

Pre-registered2026-10-30. Reuse the predictive21 selection for all18 spent frames,
unchanged camera/stereo/noise and reference-only evaluation. No fits, new fitting
masks or thresholds. Reproduce aggregate selected/normal-pair counts and failure
fractions from the preceding report. Save dense maps and artifact hashes.

Define measurement boundary pixels by adjacent surrogate-disparity jump>1px or
invalid depth, dilated by a21x21 square (10px radius), then crop the same interior.
This is a fixed diagnostic partition, not an assumed physical object boundary.
Report all selected, near-boundary and interior populations separately. Distance
failure means nearest-LiDAR distance>1m; its denominator is selected pixels.
Normal failure means unsigned disagreement>15deg among pairs within1m and with
valid LiDAR normals. Missing normals are not normal-angle failures. Also report
all16 fixed image cells per frame. Preserve empty region denominators as null.

Produce all18 images with common scales: selected support/boundary, distance0–2m,
normal angle0–90deg, and original image for qualitative orientation. Include both
passed and failed frames; the three failures identified by the earlier report
are emphasized in prose only. Plot colors clipped for display; numeric maps
remain unclipped. Source images are pinned inputs and are not used for selection.

Compare boundary and interior rates without treating pixels as independent
recordings. Concentration at boundaries does not prove motion or a unique sensor
error; nearest pairs/normals are proxies. Wide interior errors may implicate broader
shape mismatch but do not identify timing, stereo or reference error uniquely.
Do not exclude frames or turn reference maps into production masks. Freeze sources,
inputs and rendering code before execution. Preserve schema/provenance; no SOTA.
