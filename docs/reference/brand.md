# Brand and naming

Calibrex is a robotics sensor calibration framework for people who need to
estimate transforms, check a deployed rig, and compare its calibration across
recordings. Its distinguishing output is per-axis evidence: uncertainty,
observability, evaluation controls, and input provenance alongside the estimate.

## Shared description

Use this short description in package metadata, the documentation, and CLI help:

> Sensor calibration with per-axis evidence for robotics.

Use this tagline for the workflow:

> Estimate. Check. Track drift.

For an introduction, use:

> Calibrex estimates sensor calibration, checks deployed transforms, and tracks
> drift across recordings. It reports uncertainty, observability, and provenance
> for LiDAR, IMU, camera, GNSS, and vehicle calibration, including which axes the
> data could not judge.

The primary audience is robotics engineers integrating or maintaining sensor
rigs. Researchers can inspect the same schema-valid results, solver adapters,
and public-data evaluations. Lead with the calibration workflow, then link to
the relevant method, dataset, or acceleration measurements.

## Three workflow verbs

| Verb | Command | Promise |
|---|---|---|
| Estimate | `calibrex estimate` | Estimate calibration from a recording and export supported transforms; mark axes the recording did not measure. |
| Check | `calibrex check` | Compare deployed transforms with a recording and report verdicts only for the axes the data can judge. |
| Track drift | `calibrex drift` | Compare recordings of the same rig and report changes, stable axes, and inconclusive axes. |

ROS 2 and ROS 1 bag support is an input capability. The core remains
ROS-independent; ecosystem tools enter through adapters. Browser pages expose
the workflows they implement: bag check and estimate, or IMU-to-trajectory
calibration. Drift across recordings is currently a CLI workflow.

## Names and compatibility

| Context | Form |
|---|---|
| Product, prose, headings | **Calibrex** |
| Command, Python distribution, import | `calibrex` |
| Repository | `rsasaki0109/Calibrex` |
| Technical concept | **SLAC**, simultaneous localization and calibration |
| Existing serialized contracts | Their registered identifiers, including `slac.*` |

SLAC describes an architecture or an algorithm, including Open3D SLAC; it is
not the product name. Historical ADRs and release notes retain the names that
were used at the time.

Existing `slac.*` schema IDs, `slac_version` fields, `slac_native` producer
values, and versioned protocol or policy IDs remain compatibility contracts.
Keep them exact in code, examples, schemas, and stored evidence. Their presence
in an artifact does not imply a second product. See
[ADR 0009](../adr/0009-restore-calibrex-name.md) for the rename and migration
policy, and the [schema reference](schemas.md) for registered contracts.

## Visual identity

Reuse the coordinate-frame mark in
[logo.svg](../assets/brand/logo.svg) and
[favicon.svg](../assets/brand/favicon.svg). Keep its proportions and use a
dark background so all three axes remain visible. A separate text label should
spell **Calibrex** when the mark identifies the product.

| Role | Value | Use |
|---|---|---|
| Base | Slate `#0f172a` | Header and mark background |
| Accent | Teal `#0d9488` on light, `#2dd4bf` on dark | Primary actions and coordinate axis |
| Data accent | Orange `#fb923c` | Coordinate axis and data illustrations |
| Text | Inter with system sans-serif fallback | Interface and documentation |
| Code | JetBrains Mono with monospace fallback | Commands, numbers, and artifacts |

The documentation and browser pages already use these assets and colors.
Keep verdict colors tied to the actual result, with a text label alongside
color. Product decoration must not imply a `pass` verdict.

## Claims backed by evidence

Write concrete capabilities and attach the relevant limits:

- A `pass` covers the axes a recording could judge. Name the partial coverage,
  and keep `inconclusive` distinct from success.
- Describe calibration coverage per sensor pair. For example, the bag-level
  camera–LiDAR workflow judges rotation only; it does not judge translation.
- A state-of-the-art claim belongs to a specific method, dataset, and frozen
  audit. Link the [leaderboard](../benchmarks/sota_leaderboard.md), including
  refuted and incomplete claims, rather than describing the whole product as SOTA.
- State performance gains with the measured workload and baseline. Distinguish
  a kernel or fitting stage from a complete bag run, and first-use compilation
  from subsequent calls. Link the measurements and provenance.
- Label synthetic demonstrations and development-data results as such. Preserve
  the original source-bound artifacts when revising the surrounding copy.

Calibrex remains alpha research software. Introductions should make its
workflow clear while keeping the [current limitations](../development_roadmap.md)
and evaluation evidence easy to find.
