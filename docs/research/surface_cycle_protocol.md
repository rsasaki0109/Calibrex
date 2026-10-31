# Conditional cycle diagnostic after predictive surface selection

Pre-registered2026-10-30. Review found the prior1px cycle mask insufficient in
stereo translation refinement (stereo_cycle_filter_comparison.md); do not rerun
that optimizer or tune its thresholds. New question: does the exact same reverse
consistency test distinguish wrong surfaces inside the NEW predictive21 selected
population? This is conditional measurement diagnosis, not a calibration trial.

Reconstruct forward/reverse SGBM with frozen settings/OpenCV and re-rectification
from joint_learned_stereo18 frames. Restore masked depths identically to the old
cycle code; only archived saved-depth suffix changes from rerectified to sgbm.
Require exact raw restored-depth equality for all18 and exact old filtered-depth
equality on the12 previously tested frames. The6 other frames are already-spent
0013/0015, not fresh data. Pin original images/calibration/metadata and hashes.

Keep fixed <=1px finite round-trip consistency. Cross it with predictive21 masks
from the frozen error maps, without refitting a plane on the masked depth. Record
all-selected,cycle-pass,rejected-or-missing counts, domain retention, distance
failure rate (>1m among selected), normal failure rate (>15deg among supported
pairs), with original reference maps scoring-only. No new masks enter a solver.
Rejected and reverse-missing are grouped; do not call them all wrong matches.

Retain all18 frames and emphasize prior3 failures without selecting favorable
frames. Compare paired rates/counts and explicit denominators; dense pixels are
not independent recordings. A consistent match is not ground truth. If the three
failed frames remain below80% normal agreement, do not present cycle masking as
a remedy or run ICP with it. No gate/threshold adjustment, fresh-data use, core
change, calibration verdict change or SOTA claim. Freeze code/input hashes first.
