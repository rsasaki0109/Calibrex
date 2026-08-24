# Raw-capture replay and field replacement

`calibrex replay` is a digest-bound, read-only orchestration contract for
replaying a calibration against an existing capture manifest.  A definition
freezes the source/config/candidate or provider, protocol ID, deterministic
seed, stage order, timeouts, holdout and known-bad controls, expected metric
and transform budgets, tool digest, and environment/container digests.

The supported flow is:

```text
calibrex replay plan <definition.yaml> --output plan.json --json
calibrex replay run <definition.yaml> --output-dir replay-out --json
calibrex replay verify replay-out/replay-result.json \
  --definition <definition.yaml> --json
calibrex replay compare left/replay-result.json right/replay-result.json \
  --output comparison.json --json
```

`precomputed` mode is intended for costly or external providers.  The
candidate bytes must match the declared digest and its result must declare the
exact protocol ID.  `in_process` uses the existing calibration API; `argv`
uses a timeout-bounded adapter command and never evaluates shell text.

Every run emits one self-digested `replay-result.json` plus a digest for each
stage artifact.  Input drift, an unavailable READY capture, a missing holdout
or known-bad control, weak observability, malformed results, and budget
violations fail closed.  A PASS/ADOPT result is a recommendation only: CI does
not modify an Autoware package.  When requested, the Autoware artifact is a
read-only promotion plan.

## Field replacement pilot

Add a `registry` block to a definition and use:

```text
calibrex replay field-replacement <definition.yaml> \
  --output-dir field-pilot-out --json
```

The pilot records the vehicle, sensor kit, calibration edge, baseline sensor,
replacement serial/mount, and lifecycle evaluation in the append-only
registry.  A changed serial or mount is recorded as baseline removal plus a
new replacement sensor identity; an identity mismatch is blocked.  Rollback
is explicitly simulated and reports `package_mutated: false`.  Promotion and
package application remain separate operator actions requiring their own
promotion and smoke artifacts.

The repository fixture at
`examples/raw_replay/synthetic/synthetic-definition.yaml` is deliberately
synthetic and only proves plumbing.  It is not physical LiDAR evidence and
has no ground-truth accuracy claim.  A real-data replay should be added only
when the capture license, source provenance, sensor identity, and independent
ground truth are documented; a physical replacement/remount and Autoware
smoke run remain hardware gaps for this pilot.
