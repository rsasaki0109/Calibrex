# Koide pilot acceptance

`calibrex camera-lidar benchmark-koide-pilot` is the frozen acceptance boundary
for an external `direct_visual_lidar_calibration` candidate. It consumes either
the official native `calib.json` or a validated `external-run` artifact; it
does not run a Calibrex optimizer or fit the holdout.

The pilot binds the candidate and native output digests, KITTI input inventory,
Calibrex config, Koide readiness artifact, source/tool identity, command, and
execution environment in `pilot.json`. On the same deterministic
timestamp-ordered frame split it records train and untouched holdout:

- projection ratio;
- image edge alignment; and
- depth-edge alignment.

It also evaluates twelve mandatory signed controls (± roll, pitch, yaw, x, y,
and z) without refitting. The candidate-to-dataset-reference SE(3) delta is
recorded when a reference calibration is available.

The default holdout gates are deliberately non-zero (projection ratio ≥ 0.50,
edge and depth-edge alignment ≥ 0.20), and each known-bad perturbation must
change a holdout metric by at least 0.01. The external-run artifact must bind
the fitting input inventory and declare matching train/holdout ID digests. A
native `calib.json` import has no execution or isolation evidence by itself and
therefore can never be adopted.

For an external run, list the fitting camera/LiDAR files (and any calibration
inputs) explicitly in `input_paths`; binding the whole KITTI sequence directory
would include holdout files and is rejected as non-independent.
The train/holdout ID digest is SHA-256 over the sorted, newline-separated
canonical IDs (`camera:<index>:lidar:<index>`).

For the maintained commercial handoff, first validate
`examples/official/koide_execution_lock.yaml` and run
`calibrex camera-lidar koide-handoff`. The lock pins the official source/image,
uses `initial_guess_manual`, and excludes SuperGlue. Its `base_image_digest`
is intentionally unresolved because the upstream Dockerfile publishes only a
mutable base tag; an operator must resolve that digest before rebuilding.

```bash
calibrex camera-lidar benchmark-koide-pilot \
  /path/to/2011_09_26_drive_0005_sync \
  --candidate /path/to/calib.json \
  --config examples/public_datasets/kitti_raw_2011_09_26_drive_0005/config.yaml \
  --readiness artifacts/koide-readiness.yaml \
  --output-dir outputs/koide-pilot \
  --official-command docker run --rm --network none \
    <immutable-koide-image> calibrate
```

The machine-readable status is one of `PASS`, `WARN`, `FAIL`,
`INCONCLUSIVE`, or `BLOCKED`. Only `PASS` with `adoption_decision: ADOPT` is
admissible. Missing/stale readiness or candidate artifacts, a failed native
parse, incomplete holdout evidence, or an unavailable official environment
cannot become an adoption decision; `execution-manifest.yaml` retains the
exact reason and command. Recording an `--official-command` is intent/provenance
only; the pilot never invokes Docker or Podman. `execution.attempted` is true
only when the supplied external-run artifact records an actual process attempt.

Autoware export is intentionally a separate gated operation:

```bash
calibrex camera-lidar export-koide-pilot outputs/koide-pilot/pilot.json \
  --base-frame camera0 --output sensor_kit_calibration.yaml
```

It rejects every non-admissible pilot status. The general result exporter is
not an assertion that an external Koide candidate has passed this pilot.

The small repository KITTI-shaped fixture and the A2D2 sample are plumbing and
diagnostic data, not official Koide executions or claims. A2D2 is used only for
the real-data readiness diagnostic; it is not a KITTI projection protocol and
does not support an official Koide score. An official Koide-on-public-data
result requires a user-supplied native output plus its immutable environment,
actual execution evidence, and training-data isolation evidence.
