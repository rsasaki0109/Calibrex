# Bounded short-baseline image preparation

Pre-registered2026-10-30. Only six already-spent drives, the existing18 centers,
and their exact ±1 capture neighbors:36 stereo pairs/72 PNG files. No new drive,
LiDAR scan, fresh holdout or threshold selection. Existing timestamp files and
center images remain pinned. No claim that these new frame contents were already
examined; they are additional development material from spent recordings.

Stage1 reads only HEAD and ZIP central-directory metadata, verifies prior archive
size/ETag and exact member presence. Payload request cap32MiB per drive and128MiB
aggregate; bounds include header allowance4096 per member. Central-directory/HEAD
traffic and bounded retries are separate metadata/transport overhead. Freeze exact
member offsets, compressed sizes, CRC32, uncompressed sizes and request ranges in
index.json BEFORE fetching any neighboring image payload. Metadata-only inventory
may be read before the payload plan. No full ZIP downloads or automatic widening.

Stage2 source/index hash pins and commit required. HEAD/If-Match guard archive
identity on reads; exact range lengths and ZIP CRC/size are verified. Four independent
member fetch workers, fixed helper retry behavior, partial outputs/logs preserved
on failure, no shrinking denominator or alternate-host retry. Retain SHA256/path
for every image; workspace outputs must not overwrite prior inputs. TLS and managed
proxy preserved. Check disk capacity before fetching.

Then run the SAME image-support algorithm as temporal_surface_audit on36 adjacent
pairs, earlier→center and center→later, source image_02 only; retain right images
for a future metric-motion study. Verify center raw grayscale pixels equal the
previous audit's center images before comparing input intervals. Same ORB2000,
RANSAC1px/.999, >=30 cheirality inliers and6cells screen, no retuning. Treat support
as feasibility, not accurate metric motion or true overlap. Report camera timestamp
intervals and every failure; no camera/LiDAR reference enters image motion.

No temporal depth mask, ICP, scale from reference extrinsics or SOTA claim. Metric
camera motion remains a separate validation requirement. Any stereoVO shares
stereo-depth errors and must disclose that coupling.
