# Measured browser demos

These browser panels use real recorded observations, not generated sensor
streams. The [gallery](../../app/check.html) now has eleven selectable demos.
Data loads only when a panel is selected. Drag a 3D plot, zoom, or use the
observation slider to inspect its numerical coordinates.

## Camera–LiDAR method comparison

The ACFR `poses.csv` contains 40 real checkerboard captures. Four native methods
use the same 32 fit and 8 held-out captures: plane-only, point+plane, Horn points,
and line+plane. The plot connects measured camera centres with transformed LiDAR
centres. A common held-out centre RMSE is displayed alongside the method's own
result; it does not replace each method's different objective or observability
checks. There is no accepted independent extrinsic truth.

The line+plane boundary extractor resolves corner symmetry using the **fit-only
plane rotation**; it never uses holdout corner residuals to choose the rotation.
Its shared split is checked by the producer.

Pandey mutual-information and Levinson–Thrun edge results are shown separately
from existing real A2D2 reports. They retain their **FAIL / WARN** grades and
pre-registration caveats. Different datasets and objectives are not ranked
against board-centre RMSE. A2D2's camera-registered points do not establish
independent extrinsic accuracy.

## Robot hand–eye

Eight native `AX=XB` methods use the same ETHZ robot-arm motion split. Relative
motions are constructed from disjoint absolute-pose blocks before fitting or
evaluation. The two point sets are the origins of `A X` and `X B` for each held-out
motion equation; they are **not two global trajectories**. Rotation/translation
closure measures consistency, not absolute accuracy. Source poses and aligned
timestamps are preserved in the original archive.

## GNSS tracks and candidate lever geometry

Actual QCOM fixes, u-blox fixes and fused camera poses from the same comma2k19
minute are transformed into a local East/North tangent plane. Height is excluded
because altitude conventions may differ. Fusion incorporates GNSS, so the fused
trajectory is **not independent truth**. Receiver sample rates and timestamps
differ; equal plotted indices are not treated as simultaneous correspondences.

The forward/right lever sliders draw a **candidate**, geometrically shifted
body path using recorded orientations. They do not estimate a lever arm or
assign calibration accuracy. Original GNSS and pose data stay unchanged.

## Wheels / INS / vehicle axes

All four wheel-speed channels come from actual CAN wheel encoders. They are
compared with the recorded fused camera forward velocity interpolated onto
wheel timestamps. A separate view displays forward/right/down velocity axes.
Differences can combine clock, wheel scale, turns and camera-to-vehicle alignment;
this plot does not identify these effects separately.

Upstream `utils/camera.py` defines the recorded Hamilton `wxyz` quaternion as
**ECEF-from-device**. Its transpose maps ECEF velocities into camera FRD; the
producer follows that implementation. The prose README describes the direction
inconsistently, so the upstream helper and an independent axis regression are
preserved as evidence. Camera FRD is not asserted to equal the vehicle frame.

## Radar tracks

The four snapshots contain real CAN-reported forward range, left distance and
relative speed. Colour represents that relative-speed channel. The yaw slider
rotates the target coordinates as a manual extrinsic experiment. These are
processed Radar tracks, **not raw Doppler samples**, a native ego-velocity fit,
or independently calibrated Radar extrinsics.

## Live calibration

The browser runs an actual six-parameter point least-squares fit on ACFR measured
centres. The starting transform is the native fit-only Horn result with a
deliberate rotation/translation perturbation. Finite-difference Gauss–Newton
updates, damping, a loss-reducing line search and termination checks run locally;
the UI displays each actual iteration and residual history.

Only the 32 fit observations enter the updates. The 8 holdout observations are
evaluated separately. Tests perturb held-out targets and confirm that the fitted
parameters stay unchanged. Run, single-step, stop and reset are explicit actions;
switching panels stops the loop. No interpolated replay is labelled as an
optimizer iteration. This small browser example is separate from the full native
calibration pipeline; closure is not an independent accuracy claim.

## Sources, attribution and reproducibility

* **ACFR:** [pinned measured VLP observations](https://github.com/acfr/cam_lidar_calibration/blob/ecda574ed902913fbc2ae50f92ad3356f318431c/data/vlp/poses.csv),
  Apache-2.0. Original CSV SHA-256:
  `024bc6ed9009652761e9c0df49b106d325a10c88d41a80e9b36ea55fd567e110`.
* **ETHZ ASL:** [pinned real robot-arm archive](https://github.com/ethz-asl/hand_eye_calibration/blob/966cd92518f24aa7dfdacc8ba9c5fa4a441270cd/datasets/robot_arm_w_color_camera_real.zip),
  DOI `10.3929/ethz-c-000788527`, In Copyright – Non-Commercial Use Permitted.
  Upstream sample code is BSD-3-Clause; that does not relicense dataset rights.
* **comma.ai:** [comma2k19 example segment](https://github.com/commaai/comma2k19/tree/4c7f1a6e1957745beadc1def0e7225f559b09a2a/Example_1),
  Harald Schafer, Eder Santana, Andrew Haden and Riccardo Biasini (2018).
  Original repository MIT notice is preserved in `source/comma/LICENSE`.
  `source/comma/receipt.json` binds every NumPy file and helper to its pinned URL
  and SHA-256. No pickled arrays are loaded.
* **A2D2:** displayed targetless report metrics retain their existing dataset
  conditions and source reports. Raw A2D2 images or point clouds are not copied
  into this bundle; see the existing report provenance.

Rebuild with `PYTHONPATH=src:tools python tools/build_evidence_demos.py`.
[evidence_demo_index.json](../../app/samples/evidence_demo_index.json) binds all
original observations, producer, native methods and display assets to SHA-256.
The source archive uses no generated Doppler or prepared synthetic GNSS paths.
