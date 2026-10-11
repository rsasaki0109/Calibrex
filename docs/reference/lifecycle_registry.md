# Calibration lifecycle registry

`calibrex lifecycle` provides a ROS-independent, filesystem-first record for
sensor replacement and calibration adoption. It is additive to the synthetic
`slac.calibration_lifecycle/v0.1` replay: the v0.2 registry keeps physical
identity, exact input digests, operator provenance, and an append-only event
chain.

## Field-replacement workflow

Create a registry in a durable directory (the directory is the trust boundary):

```text
calibrex lifecycle init --registry-root .calibrex/lifecycle --registry-id fleet-07
calibrex lifecycle register-sensor --registry-root .calibrex/lifecycle \
  --sensor-id lidar-front-2026-08 --vehicle-id car-07 \
  --sensor-kit-id kit-2026 --serial VLP-123 --model VLP-16 \
  --firmware 3.1.0 --mount roof-front --install
```

The first sensor registration materialises the vehicle and sensor-kit IDs if
they do not already exist. A later remount must use the same serial, model,
firmware, and mount identity; a mismatch is blocked as a likely replacement
mix-up. Use the typed `install_sensor`/`remove_sensor` APIs for explicit
remount/remove events.

Register each calibration edge, then evaluate a candidate from a verified
capture manifest:

```text
calibrex lifecycle register-edge --registry-root .calibrex/lifecycle \
  --edge-id lidar-front-camera --vehicle-id car-07 --sensor-kit-id kit-2026 \
  --parent-frame base_link --child-frame lidar_front
calibrex lifecycle evaluate --registry-root .calibrex/lifecycle \
  --edge-id lidar-front-camera --capture-manifest capture.yaml \
  --candidate-result candidate.yaml --evidence evidence.yaml \
  --assessment assessment.yaml --promotion promotion.yaml --smoke smoke.yaml \
  --output evaluation.yaml
calibrex lifecycle promote --registry-root .calibrex/lifecycle \
  --edge-id lidar-front-camera --evaluation evaluation.yaml
calibrex lifecycle verify --registry-root .calibrex/lifecycle --json
```

Evaluation is fail closed. Missing source files, changed digests, a non-READY
capture, weak or `INCONCLUSIVE` evidence, and anything other than
`PASS`/`ADOPT` leave the incumbent untouched. Promotion updates only the edge
IDs declared in that command. Rollback appends a new event that references the
prior promotion and never erases history:

```text
calibrex lifecycle rollback --registry-root .calibrex/lifecycle \
  --edge-id lidar-front-camera
```

Native `slac.result/v0.1` candidates, `slac.autoware_promotion/v0.1` plans,
and `slac.autoware_smoke/v0.1` evidence are parsed and self-digest checked at
this boundary. Provider-specific evidence remains an adapter contract, but it
must still expose a schema version and an explicit status. New registry
commands also retain the projection sequence they read; a concurrent operator
therefore gets a stale-write error instead of silently replacing a newer
incumbent.

`events.jsonl` is immutable by convention and every line carries
`sequence`, `previous_event_sha256`, and `event_sha256`. `head.json` binds the
last event and the replayed state projection. Appends use an exclusive lock
file plus a compare-and-swap head check, so concurrent operators cannot both
claim the same sequence.

## From Calibration CI

`calibrex lifecycle record-ci` turns a `calibrex ci` decision into registry
events for one edge:

```text
calibrex ci calibration/candidate.yaml --policy calibration/policy.json \
  --output-dir calibrex-ci
calibrex lifecycle record-ci calibrex-ci/calibration-ci.json \
  --registry-root .calibrex/lifecycle --edge-id lidar-front-camera \
  --capture-manifest capture.yaml --promotion promotion.yaml --smoke smoke.yaml \
  --promote --output lifecycle-record.json --json
```

The decision, the candidate result, the CI assessment and the CI policy are
copied into `ci/<sha256 of calibration-ci.json>/` inside the registry, once;
a snapshot file with different bytes is refused, so later verification never
depends on a CI workspace. The candidate is then evaluated with the decision
as evidence and the CI assessment as assessment:

| CI decision | Registry events | Incumbent |
| --- | --- | --- |
| PASS, promotion and smoke PASS, `--promote` | `evaluate`, `promote` | the candidate |
| PASS without `--promote` | `evaluate` (ADOPT) | unchanged |
| INCONCLUSIVE or WARN | `hold` | unchanged |
| FAIL | `reject` | unchanged |
| FAIL about the promoted incumbent, `--rollback-on-fail` | `rollback` (decision as evidence) | the incumbent before that promotion |

A candidate whose bytes changed after the decision is refused, a decision
cannot be recorded twice, and the command verifies the registry afterwards
(exit 1 if verification fails). The record is a schema-valid
`slac.calibration_ci_lifecycle/v0.1` artifact. A GitHub workflow that promotes
on merge and re-checks the incumbent weekly is in
[`examples/ci/calibration-ci-lifecycle.yml`](https://github.com/rsasaki0109/Calibrex/blob/main/examples/ci/calibration-ci-lifecycle.yml).

## Limitations

The registry does not itself decode ROS bags, run a calibration solver, or
apply Autoware files. Those are adapter boundaries. The current optional
rollback callback is intentionally guarded by the registry; deployments must
provide a downstream Autoware adapter and its own smoke evidence. A lock file
left by a process killed during an append requires operator cleanup after
checking that no writer remains. For high-availability or multi-host use,
replace the filesystem backend with a transactional service while preserving
the event and artifact contracts.
