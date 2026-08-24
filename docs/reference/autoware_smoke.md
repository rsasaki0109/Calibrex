# Autoware downstream smoke evidence

`calibrex.autoware_smoke` is the ROS-independent adapter boundary for the
last-mile checks that belong in an Autoware workspace.  A smoke artifact is
bound to one promotion plan self-digest, candidate digest, and package
baseline manifest digest.  It records each stage's argv, exit/timeout state,
stdout/stderr digests and tails, execution policy, and provenance.

The canonical stages are:

1. `xacro_urdf` — URDF/Xacro expansion or structural description validation.
2. `build_test` — `colcon build/test` or an equivalent package build command.
3. `tf_static` — static TF/frame graph validation.
4. `sensor_launch` — sensor launch plus topic/frame smoke.
5. `perception` — optional downstream perception smoke.

Stages are supplied as argv vectors.  Calibrex always invokes subprocesses
with `shell=False`; it does not interpret shell text.  Container mode adds
`--network=none`, a read-only workspace mount, and requires an immutable
`sha256:<digest>` image for a production artifact.  The local repository does
not contain ROS, xacro, ament, or a container runtime, so unit tests use
controlled helper commands and precomputed artifacts.  Those tests are
simulated adapter evidence, not a claim that a real ROS workspace passed.

## Production flow

```text
calibrex autoware promotion plan ... --profile production --output plan.yaml
calibrex autoware smoke run plan.yaml --profile production \
  --stage-command 'xacro_urdf=["xacro","..."]' \
  --stage-command 'build_test=["colcon","build","--event-handlers","console_direct+"]' \
  --stage-command 'tf_static=["python","check_tf.py"]' \
  --stage-command 'sensor_launch=["ros2","launch","..."]' \
  --output smoke.yaml
calibrex autoware smoke verify smoke.yaml --plan plan.yaml
calibrex autoware promotion apply plan.yaml --smoke-artifact smoke.yaml
```

`apply` re-verifies the smoke self-digest and exact plan binding, re-reads the
candidate and package baseline immediately before mutation, and refuses stale
or mismatched evidence.  It stores backup inventory, output file digests, and
the smoke digest in the updated promotion artifact.  Rollback preflights every
target and backup digest before restoring any file, so an intervening edit
cannot result in a partial rollback.

`--profile developer` is an explicit local opt-out.  A passing developer smoke
artifact is labelled `developer-only` and can never authorize a production or
commercial promotion.  A production plan without a PASS,
`production-admissible` smoke artifact is blocked.

## Precomputed handoff

On a ROS-enabled CI/vehicle machine, save a smoke artifact and transfer it with
the promotion plan.  On a ROS-free signing/apply machine:

```text
calibrex autoware smoke import smoke.yaml --plan plan.yaml --output bound-smoke.yaml
calibrex autoware smoke verify bound-smoke.yaml --plan plan.yaml
calibrex autoware promotion apply plan.yaml --smoke-artifact bound-smoke.yaml
```

Import validates the artifact's self-digest and all plan bindings.  It does not
upgrade simulated or developer evidence to production evidence.
