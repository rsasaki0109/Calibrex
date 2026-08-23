# External Calibration Adapters

Calibrex keeps external calibration implementations behind a ROS-independent,
subprocess/import boundary. It does not import UniCalib, Koide's direct
visual-LiDAR calibration implementation, model weights, ROS, CUDA, or GPL code
into `src/calibrex`.

The maintained commercial Koide handoff is pinned in
[`examples/official/koide_execution_lock.yaml`](https://github.com/rsasaki0109/Calibrex/blob/main/examples/official/koide_execution_lock.yaml).
Validate it and print the operator checklist with
`calibrex camera-lidar koide-handoff --lock examples/official/koide_execution_lock.yaml`.
The lock records the official source commit and published image digest, uses
the upstream manual-initialization command, excludes SuperGlue, and leaves the
upstream mutable base-image tag unresolved for any rebuild.

Every Koide or UniCalib adapter result includes an
`external_calibration_run` mapping. Set `external_run_output` to materialize
the same mapping as a standalone `slac.external_calibration_run/v0.1`
YAML/JSON artifact:

```yaml
pipeline:
  factors:
    unicalib_lidar_camera:
      enabled: true
      options:
        result_path: outputs/unicalib/transform.yaml
        input_paths:
          - inputs/fixed_split_manifest.yaml
        expected_result_sha256: "<sha256>"
        external_run_output: outputs/unicalib/external_run.yaml
        tool_version: "wacv-2026"
        source_repository: https://github.com/han-15/UniCalib
        source_commit: "<full commit>"
        license_spdx: MIT
        training_isolation_declared: true
        training_isolation_evidence: "checkpoint training split excludes evaluation sequence"
```

Commands may be a string or argv list. Placeholders are substituted per argv
element and no shell is enabled:

```yaml
command:
  - /opt/unicalib/run
  - --dataset
  - "{dataset_path}"
  - --output
  - "{result_path}"
execute: true
timeout_sec: 600
```

Supported placeholders are `{dataset_path}`, `{result_path}`,
`{camera_streams}`, and `{lidar_streams}`. Container execution should declare
`execution_mode: container` and a content-addressed `container_digest`.

The imported transform format is deliberately adapter-neutral:

```yaml
transforms:
  T_camera0_lidar0:
    convention: T_parent_child
    parent: camera0
    child: lidar0
    translation_m: [0.0, 0.0, 0.0]
    rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
```

### Koide native `calib.json`

`direct_visual_lidar_calibration` writes a native JSON result with
`results.T_lidar_camera = [x, y, z, qx, qy, qz, qw]`.  The vector maps camera
coordinates into LiDAR coordinates.  The adapter requires an explicit
`lidar_frame` and `camera_frame` (or a unique frame binding from configured
streams), stores the native estimate as `T_<lidar>_<camera>`, and emits the
strict inverse `T_<camera>_<lidar>`.  No generic `camera0`/`lidar0` convention is
guessed.  Missing fields, wrong vector length, non-finite/non-numeric values,
and a zero quaternion become `invalid_output` with a concrete parse error.

For example:

```yaml
pipeline:
  factors:
    koide_lidar_camera:
      enabled: true
      options:
        result_path: outputs/external/calib.json
        lidar_frame: lidar0
        camera_frame: camera0
```

The result-level and typed external-run provenance retains the native format,
source path and SHA-256, declared `T_lidar_camera` convention, bound frame
names, and both generated transform names.  The importer is available as
`calibrex.importers.koide.import_koide_result`; the strict lower-level parser
raises `ValueError` so external subprocess adapters cannot turn malformed
output into a usable transform.

The same boundary is available from the CLI:

```bash
calibrex external-run import-koide calib.json \
  --output external-run.json \
  --lidar-frame lidar0 \
  --camera-frame camera0 \
  --input-artifact recording.mcap \
  --tool-version <koide-version> \
  --source-commit <koide-commit> \
  --training-isolation-declared \
  --training-data-ids-sha256 <train-id-digest> \
  --holdout-data-ids-sha256 <holdout-id-digest>
```

For the Koide pilot, repeat `--input-artifact` only for files used during
fitting. Do not import an artifact whose input inventory contains the frozen
holdout; the pilot verifies the combined input digest and rejects overlapping
files or whole-sequence directory mounts.

External objective values can be recorded under `external_metrics`, but the
schema labels them non-comparable with Calibrex holdout evidence by default.
The normal calibration pipeline recomputes its camera-LiDAR evidence after the
adapter transform is applied; an external training loss never substitutes for
those gates.

### Typed Koide workflow runner

For a maintained execution path, use the typed Koide runner instead of a
single shell command.  Each stage is an argv list and is short-circuited on
failure; the external-run artifact stores status, return code, duration, log
tails, and stdout/stderr digests for every stage:

```yaml
pipeline:
  factors:
    koide_lidar_camera:
      enabled: true
      options:
        execution_mode: subprocess
        profile: commercial       # the default
        dataset_path: captures/drive.mcap
        result_path: outputs/koide/calib.json
        lidar_frame: lidar_front
        camera_frame: camera_front
        stages:
          - name: preprocess
            argv: [python, tools/koide_preprocess.py, "{dataset_path}"]
          - name: initial_guess
            argv: [python, tools/manual_initial_guess.py, "{initial_guess_path}"]
          - name: calibrate
            argv: [python, tools/koide_calibrate.py, "{result_path}"]
        initial_guess_mode: manual
```

The runner never invokes a shell.  Legacy `command` options remain supported,
but new integrations should use `stages` so preprocess, initial guess, and
calibration are independently reproducible.  A zero exit code is not enough
for success: an unreadable or frame-inconsistent native `calib.json` becomes
`invalid_output`.

Commercial mode rejects automatic initial guess and SuperGlue.  A research
workflow must explicitly set `profile: research-noncommercial` and record the
provider, provider license, checkpoint path, and checkpoint SHA-256.  Manual
or precomputed initial guesses are the supported fallback for commercial
deployments.

For ROS 2 Humble/Jazzy, keep ROS and Koide outside Calibrex and use an
immutable image digest.  The container runner emits Docker/Podman argv with a
read-only input bind, output-only writable bind, `--network none`, dropped
capabilities, and optional CPU/memory limits:

```yaml
        execution_mode: container
        container:
          engine: docker
          image_digest: ghcr.io/example/koide-ros2@sha256:<64-hex-digest>
          input_dir: captures
          output_dir: outputs/koide
          network_mode: none
          cpus: 4
          memory: 8g
```

Run a configured workflow with one command:

```bash
calibrex external-run run-koide calibrex-koide.yaml \
  --output outputs/koide/external-run.yaml
calibrex validate outputs/koide/external-run.yaml --kind external-run
```

The artifact records tool/source/container/config/input/output digests and
the declared frame/time conventions.  Docker/Podman and the ROS 2 Humble or
Jazzy image remain optional runtime requirements; fake process runners cover
the contract tests in environments without those tools.

## License and provenance gate

Release-ready runs pin tool version, source commit, SPDX identifier, adapter
version, input/output SHA-256 digests, frame/time conventions, execution mode,
and training-data isolation. Missing fields remain schema-valid but produce a
WARN provenance metric. Digest mismatch makes the external-run artifact
`failed`.

All allowed execution modes are out-of-process execution or result import.
There is no in-process external-code mode. GPL tools therefore remain separate
executables or containers and are never copied into the Calibrex core.

### Frozen Camera-LiDAR baseline readiness audit

Before spending a frozen evaluation split, audit the Koide and UniCalib
boundaries against the exact Camera-LiDAR problem and benchmark protocol:

```bash
calibrex camera-lidar audit-external-baselines \
  problem.yaml six-dof-protocol.yaml \
  --output-dir external-baseline-audit \
  --koide-source-commit <40-hex-commit> \
  --unicalib-source-commit <40-hex-commit>
```

The command verifies the problem/protocol digest and frame coverage, pins both
official repositories and their MIT licenses, and emits two
`slac.external_calibration_run/v0.1` records. It intentionally reports
`not_executed` and exits with status 2: a repository identity is not a
benchmark result. A comparable result still requires a pinned command or
container, input/checkpoint lineage, a digest-bound transform, and independent
Calibrex evaluation under the same frozen protocol. Upstream objective names
or reported runtime values are retained as non-comparable metadata and cannot
open the release gate.

## KITTI-family probabilistic correspondences

Learned correspondence providers for KITTI raw and KITTI-360 can export one
`.npz` per selected capture frame and a small YAML/JSON manifest. The manifest
is validated with `camera_lidar_correspondence_export.schema.json` and converted
to the provider-neutral `probabilistic_correspondence` artifact:

```bash
calibrex camera-lidar build-probabilistic-correspondence \
  /path/to/kitti-correspondence-export.yaml \
  --output /path/to/correspondence.yaml
```

Each NPZ must contain `point_lidar_m` (`N×3`) and `image_mean_px` (`N×2`);
`image_covariance_px2` (`N×4`), `outlier_probability` (`N`),
`reliability` (`N`), and `correspondence_id` are optional and receive explicit
deterministic defaults when omitted. The manifest records the KITTI-family
sequence, dataset SPDX declaration, camera model/intrinsics, provider commit,
provider license, checkpoint lineage, and SHA-256 plus byte size for every
export. The resulting artifact repeats the provider identity and records the
manifest/export digests in its provenance.

The provider must use the same `dataset_id`, frame IDs, camera/LiDAR frame names,
and camera model as the corresponding `build-kitti-problem` or
`build-kitti360-problem` output. KITTI-360 fisheye exports declare the official
MEI parameters and radial-tangential coefficients; the core applies the
declared projection but does not import the KITTI-360 or provider runtime.

Development confidence calibrations from alternative provider/adaptor versions
can be compared without opening an evaluation sequence:

```bash
calibrex camera-lidar compare-probabilistic-provider-support \
  confidence-provider-a.yaml confidence-provider-b.yaml \
  --output provider-support-comparison.yaml
```

Every input must be a geometric v0.2 confidence calibration for the exact same
development problem, threshold grid, split seeds, support gates, and excluded
evaluation datasets. The output ranks worst-split and frame-level geometric
support first, then uses non-geometric frame/accepted support to break exact
geometry ties. It also records Pareto dominance, but a diagnostic leader is
not automatically usable at runtime. If every source calibration is rejected,
the comparison also returns `rejected`, writes `selected_candidate_id: null`,
and exits with status 2.

### I2PNet development adapter

`tools/run_i2pnet_correspondence_provider.py` keeps the I2PNet repository,
checkpoint, PyTorch, CUDA, and native extensions outside `src/calibrex`. The
adapter pins the official repository commit, MIT license, checkpoint digest,
adapter digest, optional checkpoint-archive digest, portability-patch digest,
camera/LiDAR input digests, and complete command. It requires a separate
schema-valid initial `TransformResult` whose role is `initial`; an input with
`execution_mode: dataset_reference` is rejected, and the Camera--LiDAR problem
or reference transform is not an accepted argument.

On Windows, use the default `--neighbor-backend torch`. It reproduces I2PNet's
local-grid KNN selection, including cylindrical width wrapping and nearest
padding, while avoiding the unstable fused-neighbor execution path. The
digest-pinned patch in
`tools/provider_patches/i2pnet_windows_int64_indices.patch` only changes the
external extension's Windows index ABI from `long` to `int64_t`; no I2PNet or
GPL source is copied into the Calibrex package. The external PointNet2
extension remains required by the official model.

The optional `--pose-output` writes a typed aggregate `TransformResult` using
the official convention
`T_camera_lidar_output = T_correction @ T_camera_lidar_initial`. Per-frame
translations are averaged, rotations use a quaternion eigenvector mean, and
the quality standard deviations are explicitly labeled frame-to-frame spread,
not calibrated model uncertainty. The coarse cost-volume correspondence export
is diagnostic: its manifest records a rejection warning whenever mean
reliability is below the fixed `0.01` development gate. A rejected diagnostic
artifact must not be passed to PnP or refinement merely because it is
schema-valid.

`tools/run_i2pnet_pose_benchmark.py` executes a
`slac.camera_lidar_pose_initializer_protocol/v0.1` matrix without passing the
Camera--LiDAR problem or its reference transform to the provider subprocess.
The runner derives one explicit initial transform per frozen perturbation,
retains provider failures, verifies the returned manifest's dataset, frame,
adapter, checkpoint, backend, and initial-transform bindings, and emits the
generic schema-valid benchmark definition and aggregate benchmark artifacts.
Evaluation protocols are locked unless the caller explicitly supplies
`--allow-evaluation` after provider and threshold selection are frozen.

The first fixed development matrix uses three KITTI raw 0018 frames disjoint
from the two-frame exploratory diagnostic and four left-camera-frame SE(3)
perturbations. I2PNet completes 4/4 trials but reaches 0/4 strict hits at
`<0.5 deg` and `<0.2 m`. Mean rotation error changes from `8.1046` to
`6.7043 deg`; mean translation error changes from `0.40395` to `0.37994 m`.
This is a completed provider execution, not a successful initializer gate.
The available independent KITTI-360 development/evaluation material uses the
`image_03` MEI fisheye stream, which is incompatible with this rectified
pinhole KITTI-large checkpoint, so no evaluation partition was unlocked.

`tools/analyze_i2pnet_pose_benchmark.py` performs the next read-only diagnostic
step. It verifies the protocol/problem/definition lineage, every generated
initial/manifest/pose digest, and every frame NPZ digest before comparing
`T_reference @ inverse(T_initial)` with
`T_output @ inverse(T_initial)`. The resulting
`slac.camera_lidar_pose_initializer_failure_analysis/v0.1` artifact records
per-frame pose errors, signed correction gain along the required direction,
orthogonal correction leakage, and explicitly non-causal findings. It does not
retune a provider, threshold, or perturbation.
