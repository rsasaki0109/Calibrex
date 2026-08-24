# Autoware sensor-kit promotion

`calibrex autoware promotion` is a ROS-independent package adapter for a
calibration candidate. It creates a digest-bound, read-only plan before any
vehicle package is changed:

```text
calibrex autoware promotion plan CANDIDATE \
  --workspace-root /path/to/autoware \
  --package-root /path/to/autoware/src \
  --individual-params-root /path/to/autoware/src/individual_params \
  --sensor-kit-description-root /path/to/autoware/src/sensor_kit/my_sensor_kit_description \
  --vehicle-id my_vehicle --sensor-kit-id my_sensor_kit \
  --profile production \
  --output promotion.yaml

calibrex autoware promotion verify promotion.yaml --output promotion.verified.yaml
calibrex autoware smoke run promotion.verified.yaml --profile production \
  --stage-command 'xacro_urdf=["xacro","..."]' \
  --stage-command 'build_test=["colcon","build"]' \
  --stage-command 'tf_static=["python","check_tf.py"]' \
  --stage-command 'sensor_launch=["ros2","launch","..."]' \
  --output smoke.yaml
calibrex autoware smoke verify smoke.yaml --plan promotion.verified.yaml
calibrex autoware promotion apply promotion.verified.yaml --smoke-artifact smoke.yaml
calibrex autoware promotion rollback promotion.verified.yaml
```

The adapter understands the package layout described by the [Autoware sensor
model guide](https://autowarefoundation.github.io/autoware-documentation/main/tutorials/integrating-autoware/creating-vehicle-and-sensor-model/creating-sensor-model/):

* `individual_params/**/sensor_kit_calibration.yaml`;
* the vehicle/sensor-kit `sensors_calibration.yaml`;
* sensor-kit URDF/Xacro files; and
* YAML sensor and topic parameters.

The importer rejects duplicate YAML keys, conflicting duplicate child frames,
cycles, disconnected calibration edges, a missing `base_link` to
`sensor_kit_base_link` edge, non-unit/undeclared quaternion conventions, and
declarative topic/frame references that are not in the package graph. The
official guide specifies Euler `[x, y, z, roll, pitch, yaw]` values in meters
and radians and says that the mirrored `individual_params` calibration file
must be kept synchronized. Identical mirrored declarations are therefore
accepted; conflicting declarations are not.

Only `PASS`/`ADOPT` plans can be applied. Production plans additionally require
a `PASS`, `production-admissible` smoke artifact bound to the exact promotion
self-digest, candidate digest, and current package baseline. A developer plan
must explicitly use `--profile developer`; its application is labelled
`developer-only` and is not production/admissible. A Koide candidate additionally must
be the frozen pilot artifact with `status: PASS`, `adoption_decision: ADOPT`,
and a valid self-digest. Generic result candidates must carry an explicit
`quality.grade: pass` under the default policy. Applying rechecks the baseline
manifest and every patch input digest, backs up exact files inside the
explicit workspace root, then atomically replaces each target. Rollback
refuses to overwrite an intervening edit.

The core boundary does not import ROS, `ros2`, `xacro`, `colcon`, or a
perception implementation. `calibrex autoware smoke` is a downstream adapter:
it records argv-only subprocess/container/precomputed evidence, timeout and
resource/network/read-only policies, output digests/tails, immutable container
digests, and provenance. In this repository ROS is not installed, so unit tests
use controlled helper commands and precomputed artifacts; they are simulated
adapter tests, not evidence that a real ROS workspace passed.

Primary references:

* [Creating a sensor model](https://autowarefoundation.github.io/autoware-documentation/1.8.0/tutorials/integrating-autoware/creating-vehicle-and-sensor-model/creating-sensor-model/)
* [Sensor calibration tools](https://autowarefoundation.github.io/autoware-documentation/main/tutorials/integrating-autoware/creating-vehicle-and-sensor-model/calibrating-sensors/calibration-tools/)
* [Autoware launch overview](https://autowarefoundation.github.io/autoware-documentation/main/tutorials/integrating-autoware/overview/)
