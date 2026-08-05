# Solid-state LiDAR physical collection runbook

This runbook turns the physical ground-truth gate into a repeatable data
collection task. Public recordings are useful for solver smoke tests and
holdout behavior, but they normally do not contain an independently surveyed
extrinsic and clock reference. They cannot, by themselves, support a physical
accuracy claim.

The recommended first collection has four captures over two mechanical
remounts: two captures before remounting and two after remounting. The packet
keeps the reference for each session separate because the true transform can
change when the sensor is removed and attached again.

## 1. Generate the collection packet

From the repository root, generate a schema-valid packet and report:

```bash
python tools/run_solid_state_metrology_evaluation.py \
  --prepare-collection-plan \
  --plan-sessions 4 \
  --plan-remounts 2 \
  --output outputs/solid-state-metrology/plan.yaml \
  --markdown-output outputs/solid-state-metrology/plan.md
```

The generated packet is intentionally `INCONCLUSIVE`/`review`: it contains
placeholders, not measurements. Validate it before editing:

```bash
calibrex validate outputs/solid-state-metrology/plan.yaml \
  --kind solid-state-metrology-evaluation
```

The session/remount layout is:

| Session | Remount | Capture | Independent reference | Solver estimate |
|---|---|---|---|---|
| `session-00` | `remount-1` | before remount | required | required |
| `session-01` | `remount-2` | after remount | required | required |
| `session-02` | `remount-1` | repeat | required | required |
| `session-03` | `remount-2` | repeat | required | required |

The minimum gate is three usable sessions across two remounts. Four sessions
make the first collection easier to diagnose and leave one repeat per mount.

## 2. Collect the captures

Use the same static scene and sensor configuration for all four sessions. The
scene should contain geometry across the overlapping field of view, depth, and
orientation; a flat wall alone is a poor six-DoF calibration target. Move the
rig through several poses while preserving enough overlap for the chosen
solver. Record the exact sensor serials, firmware, temperature, frame names,
recording start/stop times, and any dropped or rejected windows.

Run capture readiness on each candidate window before spending time on the
solver. It checks motion excitation, geometry diversity, and a six-DoF
observability proxy:

```bash
calibrex calibration-readiness path/to/config.yaml \
  --output outputs/solid-state-metrology/readiness-session-00.yaml
```

Keep a `RECAPTURE` result as evidence of a failed collection attempt; do not
silently select a favorable window after inspecting calibration error.

## 3. Measure an independent reference for every session

For each session, fill `sessions[*].reference` with both parts of the physical
reference:

- `transform`: the measured source-to-target extrinsic, using one declared
  frame convention;
- `time_offset_sec`: the measured clock relationship used by the capture.

Use a method that does not consume the solver estimate. Supported method names
are `surveyed_rig`, `optical_tracker`, `robot_arm`, `calibration_target`, and
`mechanical_cad`; supported clock methods are `hardware_trigger`, `common_pps`,
`ptp`, `external_timebase`, and `timestamp_injection`.

Record the uncertainty for rotation, translation, and time. Also set
`independent_of_solver: true`, set the transform provenance level to
`independently_measured`, and attach the survey/measurement file in
`source_paths`. The global `reference` record should describe the common
protocol and frame/time convention; the session reference is the value used
for that particular capture.

Do not copy one global transform into all sessions. A remount is precisely the
condition that tests whether the calibration remains valid after mechanical
reinstallation.

## 4. Preserve the evidence tree

Keep paths relative to `plan.yaml` so the packet can be moved and reviewed as
a unit. A practical layout is:

```text
outputs/solid-state-metrology/
  plan.yaml
  plan.md
  captures/
    session-00.mcap
    session-01.mcap
    session-02.mcap
    session-03.mcap
  metrology/
    reference-protocol.yaml
    session-00-reference.yaml
    session-01-reference.yaml
    session-02-reference.yaml
    session-03-reference.yaml
  estimates/
    session-00.yaml
    session-01.yaml
    session-02.yaml
    session-03.yaml
  holdout/
    downstream-metric.yaml
```

Fill `sessions[*].capture_sha256`, each session reference's
`source_sha256[path]`, and each estimate's `source_sha256` only after the
files are finalized. For example:

```bash
sha256sum captures/session-00.mcap
sha256sum metrology/session-00-reference.yaml
sha256sum estimates/session-00.yaml
```

On PowerShell, use `Get-FileHash -Algorithm SHA256 path\to\file`.

## 5. Run the solver and keep a held-out check

Run the selected calibration method on the capture and write its transform and
time estimate into the matching `estimates/session-XX.yaml`. The estimate
source must be linked to exactly one session. Do not use the downstream metric,
the independent survey values, or the held-out recording to tune the solver.

The downstream metric must be declared before looking at the final result. It
can be a task-level camera/LiDAR alignment score, held-out map consistency, or
another physical-use metric, but it must have an explicit unit, threshold, and
`held_out: true` / `independent_of_solver: true` evidence.

## 6. Enforce the packet

After all references, estimates, and digests are filled, evaluate the packet:

```bash
python tools/run_solid_state_metrology_evaluation.py \
  --input outputs/solid-state-metrology/plan.yaml \
  --output outputs/solid-state-metrology/final.yaml \
  --markdown-output outputs/solid-state-metrology/final.md \
  --enforce

calibrex validate outputs/solid-state-metrology/final.yaml \
  --kind solid-state-metrology-evaluation
```

`--enforce` verifies that every declared reference, capture, and estimate
source exists relative to the input packet and has the declared SHA-256. It
also checks unique IDs and estimate-to-session links. A missing, mismatched,
or structurally inconsistent source blocks a physical `PASS`; missing physical
measurements remain `planned` or `inconclusive`.

Before publishing, review `final.md` and the saved `evidence_integrity` block.
The result is only a physical claim when the numerical thresholds, uncertainty
limits, remount coverage, and held-out metric all pass together.
