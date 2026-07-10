# Radar Calibration Evidence

Calibrex evaluates a supplied radar extrinsic before attempting native radar
calibration. The current velocity-consistency path compares measured Doppler
against the line-of-sight projection of ego velocity from an external pose
track. Dataset decoding remains an adapter concern; the residual API is typed,
ROS-independent, and does not depend on nuScenes types.

## Evidence Contract

- Split support by radar frame, never by individual return.
- Reuse one deterministic train/holdout manifest for the candidate and every
  known-bad control.
- Evaluate mandatory yaw controls at -10, -5, +5, and +10 degrees.
- Compute detectability only on holdout support.
- Report `INCONCLUSIVE`, rather than `FAIL`, when holdout support or motion
  excitation cannot falsify a bad candidate.
- Treat translation as unobservable from the current constant-velocity Doppler
  residual unless angular motion supplies lever-arm information.

Current diagnostics record the fraction of range-eligible returns classified
as static, including frames with zero static returns in the denominator. Yaw
observability uses the residual Jacobian
`j = u^T R^T (z × v)` in metres per second per radian. This combines ego speed,
velocity direction, line-of-sight geometry, and the candidate rotation in one
direct sensitivity measure. A one-degree-of-freedom yaw condition number would
always be one, so Calibrex reports the physical sensitivity instead.

The provisional yaw policy is deliberately fail-closed. It checks holdout
support, per-sensor yaw sensitivity, and all four known-bad controls before it
interprets the candidate residual. Missing support or falsification power is
`INCONCLUSIVE`; only a sufficiently supported candidate above a declared
residual limit is `FAIL`. Residual limits default to unset until a real-data
protocol freezes them, so the development policy cannot emit an accidental
`PASS`. In result metrics these statuses map to `pass`, `fail`, and `warn`
grades respectively, while the three-valued status remains explicit in
provenance.

The report materializer publishes this contract as
`radar_lidar_seeded_holdout_doppler_yaw/v0.1`: three evidence summaries,
four signed yaw-control cases, and the embedded policy decision are written to
the standard evidence, protocol, assessment, and policy sidecars. The core
assessment preserves the native three-valued decision as one
`radar_policy_decision` rule so a subordinate residual failure cannot override
missing observability and turn an `INCONCLUSIVE` result into `FAIL`.

For multi-Radar rigs, every configured sensor receives its own frame split,
static-return summary, yaw sensitivity, four controls, residual summary, and
three-valued assessment. Evidence is never pooled to make a weak sensor appear
supported. The aggregate rule is:

```text
any conclusive sensor FAIL → FAIL
otherwise any required sensor INCONCLUSIVE → INCONCLUSIVE
otherwise all required sensors PASS → PASS
```

A conclusive failure from one sensor is not hidden by another sensor's missing
support. Conversely, quorum passing is not allowed: every required Radar must
pass before the rig receives a yaw-only `PASS`.

The fixed split and mandatory controls are Calibrex evaluation safeguards. They are
not claims that the calibration literature conventionally uses a machine
learning-style train/holdout protocol.

## Paper-to-Design Mapping

| Paper | Design consequence for Calibrex |
| --- | --- |
| [Wise et al., continuous-time radar-camera calibration](https://arxiv.org/abs/2103.07505) | Use signed line-of-sight Doppler residuals, evaluate motion excitation, and eventually sample trajectory velocity at each radar measurement time. |
| [Wise, Cheng, and Kelly, spatiotemporal calibration](https://arxiv.org/abs/2211.01871) | Keep temporal convention and spatial extrinsic evidence connected through one trajectory; diagnose time/extrinsic compensation instead of accepting a low residual alone. |
| [Cheng, Wise, and Kelly, radar-pair calibration](https://arxiv.org/abs/2302.00660) | Report identifiable components and motion requirements. Yaw probes are justified; unconditional translation probes are not. |
| [RAVE radar ego-velocity estimation](https://arxiv.org/abs/2406.18850) | Add robust static-return selection, inlier ratio, direction coverage, and uncertainty before promoting the metric to a policy gate. |
| [Joint radar-camera-LiDAR calibration](https://doi.org/10.1109/TIV.2021.3065208) | Keep target-based tools behind adapters and use loop-closure residuals as independent validation evidence. |
| [LiRaCo](https://arxiv.org/abs/2503.17002) | Consider cylindrical occupancy as a separate geometric evidence family for cases where Doppler excitation is weak. |

## Planned Promotion to a Policy Gate

1. Record train and holdout residual summaries and the shared frame manifest.
2. Require all four yaw controls to have sufficient holdout support before
   computing a detectable fraction.
3. Validate the implemented static-return ratio and yaw-Jacobian sensitivity
   thresholds on real data; add angular-rate excitation before evaluating a
   translation lever arm.
4. Define `PASS`, `FAIL`, and `INCONCLUSIVE` thresholds in a schema-validated
   radar evaluation config.
5. Validate the unchanged policy on nuScenes mini, including a known-bad
   candidate.
6. Add capture-time convention and time-offset probes before claiming absolute
   temporal accuracy.

External implementations remain adapters unless their licenses and dependency
boundaries are compatible. No ROS or license-uncertain calibration code should
be copied into `src/calibrex`.
