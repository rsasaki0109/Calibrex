# Short-baseline audit input reconciliation

Initial audit11f93d00 aborted before producing a case because raw-center grayscale
was compared directly with a histogram-equalized archived image. The source builder
kitti_aligned_stereo_bundle.py explicitly writes cv2.equalizeHist(left_image).
The assertion correctly exposed an input preprocessing mismatch. Initial plan and
traceback are retained, and no failed comparison is counted as a motion result.

Revised audit verifies equalizeHist(raw_center) equals archived_center pixel-for-
pixel, then applies the SAME histogram equalization to newly acquired neighbors.
ORB, RANSAC, support thresholds, frame identities and timestamps remain unchanged.
Use the already archived equalized center, save equalized neighbors in a separate
workspace with hashes, and run all36 pairs. Do not relax the equality assertion.
This restores like-for-like photometric preprocessing with the long-gap audit;
it does not tune preprocessing based on feature or recovery outcomes. Acquisition
and original image bytes remain unchanged. Freeze revised source hashes first.
