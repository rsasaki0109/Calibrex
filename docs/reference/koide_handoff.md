# Koide official handoff

The checked-in [Koide execution lock](https://github.com/rsasaki0109/Calibrex/blob/main/examples/official/koide_execution_lock.yaml)
is a commercial-safe, digest-bound handoff contract. It is an execution plan,
not evidence that Koide ran. Validate it and print the operator checklist with:

```bash
calibrex validate examples/official/koide_execution_lock.yaml \
  --kind koide-execution-lock
calibrex camera-lidar koide-handoff \
  --lock examples/official/koide_execution_lock.yaml --json
```

The lock pins the upstream repository at commit
[`02a0dc039f5509708f384be4ff3228e0ae09352d`](https://github.com/koide3/direct_visual_lidar_calibration/commit/02a0dc039f5509708f384be4ff3228e0ae09352d),
declares its MIT license, and pins the published ROS 2 Humble image to
`koide3/direct_visual_lidar_calibration@sha256:f7363cc3deadc2419f3527a7b9ec7bdabaa3031fd8e1b9c6cc46666ae1301bbb`.
The image digest is recorded from the official
[Docker Hub repository](https://hub.docker.com/r/koide3/direct_visual_lidar_calibration)
and the official tag metadata endpoint
[`/tags/humble`](https://hub.docker.com/v2/repositories/koide3/direct_visual_lidar_calibration/tags/humble).
The upstream Dockerfile names `koide3/gtsam_docker:humble` as a mutable base
tag; the lock leaves its digest null and requires an operator to resolve and
record that digest before rebuilding. No invented base digest is accepted.

## What the upstream documentation establishes

The [installation guide](https://koide3.github.io/direct_visual_lidar_calibration/installation/)
lists ROS 1/2, PCL, OpenCV, GTSAM, Ceres, and Iridescence as dependencies and
documents the optional SuperGlue installation. The [Docker guide](https://koide3.github.io/direct_visual_lidar_calibration/docker/)
publishes `noetic`, `humble`, and `jazzy` images, and gives the preprocess,
manual initial-guess, and calibration commands. The [calibration example](https://koide3.github.io/direct_visual_lidar_calibration/example/)
shows that the native output is `calib.json` with
`results.T_lidar_camera = [x, y, z, qx, qy, qz, qw]`.

The commercial lock uses `initial_guess_manual` and excludes SuperGlue. The
upstream example warns that SuperGlue is not allowed for commercial use, and
the Docker guide says the published images omit it because of its strict
license. No SuperGlue checkpoint, source, or GPL code is present in Calibrex.

The [data-collection guide](https://koide3.github.io/direct_visual_lidar_calibration/collection/)
requires pre-calibrated camera intrinsics and a rigid camera/LiDAR mount. It
asks operators to record Image and PointCloud2 messages (CameraInfo is
recommended), keep the sensor at rest at the beginning of each bag, and use
sensor-appropriate motion. These requirements are copied into the lock as an
operator contract; they are not a calibration-quality claim.

## A2D2 diagnostic artifact

The ignored `outputs/koide_pilot_a2d2_blocked` directory is a local readiness
diagnostic, not a checked-in Koide result. It can be regenerated with the
public CLI (the expected pilot exit code is `1` because the status is
`BLOCKED`):

```bash
calibrex doctor --workflow koide \
  --config examples/public_datasets/a2d2_pandey_mutual_information/config.yaml \
  --output outputs/koide_pilot_a2d2_blocked/readiness.yaml
calibrex camera-lidar benchmark-koide-pilot \
  data/public/a2d2_pandey_mutual_information \
  --config examples/public_datasets/a2d2_pandey_mutual_information/config.yaml \
  --readiness outputs/koide_pilot_a2d2_blocked/readiness.yaml \
  --output-dir outputs/koide_pilot_a2d2_blocked \
  --official-command 'docker run --rm --network none -v <capture-dir>:/tmp/input_bags:ro -v <preprocessed-dir>:/tmp/preprocessed:rw koide3/direct_visual_lidar_calibration@sha256:f7363cc3deadc2419f3527a7b9ec7bdabaa3031fd8e1b9c6cc46666ae1301bbb ros2 run direct_visual_lidar_calibration preprocess -a /tmp/input_bags /tmp/preprocessed' \
  --tool-source-commit 02a0dc039f5509708f384be4ff3228e0ae09352d \
  --container-digest koide3/direct_visual_lidar_calibration@sha256:f7363cc3deadc2419f3527a7b9ec7bdabaa3031fd8e1b9c6cc46666ae1301bbb
```

The output must remain `BLOCKED`, must contain the explicit A2D2 diagnostic
blocker, and must pass `calibrex validate ... --kind koide-pilot`. Do not pass
this A2D2 artifact to the official Koide stages or interpret it as a score.

## Exact handoff sequence

1. Provide a user-owned ROS 2 capture and a Calibrex config. A2D2 is not a
   supported Koide/KITTI pilot dataset.
2. Run the Koide readiness diagnostic and stop on `blocked` or `warn`:

   ```bash
   calibrex doctor --workflow koide \
     --config <config.yaml> --output <readiness.yaml>
   calibrex validate <readiness.yaml> --kind koide-readiness
   ```

3. Pull the exact image digest from the lock with Docker or Podman. The host
   must provide the container runtime and a display for the manual GUI stage.
4. Run the three lock stages in order: `preprocess`, `initial_guess_manual`,
   then `calibrate`. Keep input mounts read-only, the output mount writable,
   and networking disabled.
5. Import the resulting native `calib.json`, supplying the exact fitting
   inputs, source commit, MIT license, and training-isolation evidence:

   ```bash
   calibrex external-run import-koide <preprocessed>/calib.json \
     --output <external-run.yaml> --lidar-frame <lidar-frame> \
     --camera-frame <camera-frame> --source-commit \
     02a0dc039f5509708f384be4ff3228e0ae09352d \
     --input-artifact <fitting-input> \
     --training-isolation-declared \
     --training-isolation-evidence "<user-owned evidence>"
   ```

6. Run the frozen pilot on a supported KITTI raw sequence with a frame-
   disjoint holdout. A2D2 readiness output is explicitly a diagnostic and
   cannot become an official Koide score or adoption decision.
7. Export to Autoware only when `pilot.json` is `PASS` and
   `adoption_decision: ADOPT`.

The local audit machine has no Docker, Podman, ROS, or colcon, so it can
validate this lock and generate blocked readiness diagnostics but cannot claim
an official execution. The lock's self-digest and all generated pilot
self-digests must be verified before handoff.
