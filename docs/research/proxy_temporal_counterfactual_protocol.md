# Evaluation-only proxy-motion temporal counterfactual

Frozen after the separate OXTS disagreement audit, before this counterfactual run,
2026-10-30. Replace ONLY the temporal check's camera motion with the nominal
camera-timestamp-interpolated OXTS/calibration proxy. Use its inverse for a center
at the second endpoint. Keep all18 surface masks, depths, disparity threshold,
visibility categories,36 pairs and original internal motion-support flags. Failed
0013 motion pairs remain unsupported, not magically validated by reference input.

This deliberately reference-assisted counterfactual is offline error attribution,
not a measurement-only selector, new production gate, calibration candidate or
independent ground truth. All rows labeled reference_assisted. It asks whether
motion replacement alone materially changes the remaining bad-surface outcome.
Keep original population and all failures; do not choose a favorable time offset.
Reuse exact render controls and hash checks. Compare all18 counts/failure rates
against the archived estimated-motion temporal diagnosis; no refits or SOTA claim.
