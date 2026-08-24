# Koide real execution handoff

`koide-real-plan`, `koide-real-verify`, and `koide-real-finalize` form a
fail-closed boundary around the official Koide executable. None of these
commands starts Docker, ROS, `colcon`, or Koide. A plan is a request, not an
execution result; a verification is not ready until every digest and evidence
gate is supplied; finalization cannot make an official claim from synthetic or
test-labelled evidence.

## Pinned official facts

The request records the following facts and their provenance:

| Item | Binding |
|---|---|
| Koide source | [`koide3/direct_visual_lidar_calibration`](https://github.com/koide3/direct_visual_lidar_calibration), MIT, commit `02a0dc039f5509708f384be4ff3228e0ae09352d` |
| Container | [`koide3/direct_visual_lidar_calibration` on Docker Hub](https://hub.docker.com/r/koide3/direct_visual_lidar_calibration/tags), `koide3/direct_visual_lidar_calibration@sha256:f7363cc3deadc2419f3527a7b9ec7bdabaa3031fd8e1b9c6cc46666ae1301bbb` |
| Native workflow | [`Koide example commands`](https://koide3.github.io/direct_visual_lidar_calibration/example/), recorded as argv-only preprocess → `initial_guess_manual` → calibrate stages |
| Native output | [`Koide program documentation`](https://koide3.github.io/direct_visual_lidar_calibration/programs/), `results.T_lidar_camera` in `x,y,z,qx,qy,qz,qw` order |
| Dataset record | [`KITTI raw data`](https://www.cvlibs.net/datasets/kitti/raw_data.php), sequence `2011_09_26_drive_0005_sync`, archive filename `2011_09_26_drive_0005_sync.zip` |
| License | [`KITTI terms`](https://www.cvlibs.net/datasets/kitti/), CC BY-NC-SA 3.0; raw data are not redistributed by Calibrex |

The official KITTI raw page requires registration/login and does not publish an
authoritative per-archive byte size or SHA-256 in the accessible record. The
plan therefore uses `archive_metadata_status=not_published_by_official_record`
and leaves archive URL, size, and checksum unknown. An operator must observe
the downloaded archive, record its exact filename/size/SHA-256, set
`archive_metadata_status=operator_observed`, and preserve the immutable copy
before official verification. Values copied from an unauthorised mirror are
not official evidence. This is an evidence-boundary decision, not a claim that
the archive lacks a checksum elsewhere.

The source repository's documentation also warns that the optional SuperGlue
dependency is not permitted for commercial use. The commercial lock therefore
requires manual initialization and rejects SuperGlue tokens in stages and
external commands. See the [official Docker workflow](https://koide3.github.io/direct_visual_lidar_calibration/docker/)
for the upstream container context; the exact image digest remains the
Calibrex execution pin.

## Commands and states

Create a non-executing request:

```bash
calibrex camera-lidar koide-real-plan \
  --camera-frame camera0 --lidar-frame lidar0 \
  --output koide-real/request.yaml --json
```

It always writes `status: BLOCKED`, `official_result_claim: false`, and
`executes_external_process: false` in the CLI response. The request contains
the exact Docker argv, `network=none`, read-only input/root filesystem,
writable output mount, manual-stage `DISPLAY` requirement, ROS 2 bag contract,
fitting/holdout frame IDs and digests, twelve signed known-bad injections, and
the frozen KPI thresholds.

Verify supplied evidence without running it:

```bash
calibrex camera-lidar koide-real-verify koide-real/request.yaml \
  --dataset /path/to/2011_09_26_drive_0005_sync \
  --archive /path/to/2011_09_26_drive_0005_sync.zip \
  --input-manifest /path/to/kitti-input.yaml \
  --readiness /path/to/koide-readiness.yaml \
  --config /path/to/frozen-config.yaml \
  --candidate /path/to/calib.json \
  --external-run /path/to/external-run.yaml \
  --execution-log /path/to/stages.log \
  --environment /path/to/environment.txt \
  --output koide-real/verification.yaml --json
```

The verifier checks source commit, image digest, native output bytes/digest,
input inventory, fitting/holdout disjointness and overlap, readiness, command
and environment digests, stage log evidence, training isolation, and the
external-run output binding. Missing output, missing archive metadata, source
or container mismatch, holdout overlap, stale manifest, failed run, and
SuperGlue references remain `BLOCKED`.

Only after `READY_FOR_PILOT` may the existing independent
`camera-lidar benchmark-koide-pilot` artifact be finalized:

```bash
calibrex camera-lidar koide-real-finalize \
  koide-real/request.yaml koide-real/verification.yaml \
  --pilot koide-real/pilot.yaml \
  --output koide-real/finalization.yaml --json
```

An official `PASS` requires an executed, digest-bound run, all holdout KPI
thresholds, all twelve known-bad controls, and pilot `PASS/ADOPT`. A
test-labelled fixture can at most produce `TEST_ONLY`; it can never set
`official_result_claim: true`.

## Evidence matrix

| Gate | Required evidence | Failure state |
|---|---|---|
| Official input | KITTI sequence, operator-observed archive size/SHA-256, locked raw inventory | `BLOCKED` |
| Fitting isolation | Exact fitting and holdout IDs/digests; fitting-only ROS 2 bags and external inventory | `BLOCKED` |
| Runtime identity | Locked source commit, MIT boundary, image digest, engine, command, network/mount policy | `BLOCKED` |
| Execution proof | Successful container run, stage stdout/stderr hashes, environment snapshot/hash, native `calib.json` path/size/SHA-256 | `BLOCKED` |
| Independent acceptance | Readiness `ready`, frozen holdout metrics, twelve signed controls, pilot `PASS/ADOPT` | `FAIL` or `BLOCKED` |

In this checkout's 2026-08-24 environment audit, `docker`, `podman`, `ros2`,
and `colcon` were not available. That is an operator/runtime blocker, not a
successful or failed Koide result. No multi-GB KITTI archive was downloaded.

## Synthetic handoff

[`examples/koide_real_pilot/synthetic/koide_real_pilot_handoff.yaml`](../../examples/koide_real_pilot/synthetic/koide_real_pilot_handoff.yaml)
is a compact schema-valid request example. It intentionally contains no raw
data, native output, logs, environment snapshot, or external-run artifact and
must load as `BLOCKED`. The checked-in fake evidence used by unit tests is
labelled `test` and is useful only for exercising schema/CLI wiring; it is not
an official Koide execution.
