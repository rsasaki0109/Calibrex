# Solid-state LiDAR physical ground-truth evaluation

- Schema: `slac.solid_state_metrology_evaluation/v0.1`
- Evaluation: `solid-state-metrology-v0.1-template`
- Status: **PLANNED**
- Decision: `collect`

This report is PASS-capable only when independent spatial/temporal references, repeat remount sessions, and the declared downstream holdout metric are all present.

## Reference

- Extrinsic method: `unknown`
- Clock method: `unknown`
- Independent of solver: `False`

## Runs

| Estimate | Session | Remount | Rotation error (deg) | Translation error (m) | Clock error (ms) | Gate |
|---|---|---|---:|---:|---:|---|
| — | — | — | — | — | — | NOT RUN |

## Evidence integrity

- Checked: `False`
- Passed: `False`
- Sources verified: `0/0`

## Aggregate

- Usable sessions: `0`
- Remounts: `0`
- Passing runs: `0/0`
- Downstream metric gate: `None`

## Reasons

- reference_not_declared_independent_of_solver
- extrinsic_reference_method_does_not_match_protocol
- clock_reference_method_does_not_match_protocol
- missing_reference_transform
- missing_reference_time_offset
- missing_reference_source_provenance
- missing_reference_rotation_uncertainty
- missing_reference_translation_uncertainty
- missing_reference_time_uncertainty
- insufficient_usable_sessions
- insufficient_remounts
- missing_downstream_metric
- missing_estimates

The synthetic benchmark remains a separate solver-mechanics gate; this artifact is the path for physical metrology evidence.
