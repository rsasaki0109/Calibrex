# CUDA spline plane fitting

The optional CUDA plane backend removes the largest point-dependent work from
the joint B-spline fitter: per-return pose evaluation, point-to-plane residuals,
analytic control derivatives and the control normal matrix/gradient. It fuses
these operations and directly reduces the seven-block control band, avoiding
the large plane Jacobian during LM iterations. The final Jacobian is assembled
for the existing marginal covariance computation.

Each normal entry has one owner, with a fixed row reduction order and no
floating-point atomic additions. Kernels use FP64 and disable fused multiply-add.
Referenced control-edge logarithms keep the native CPU near-pi guard. Immutable
measurements and their time schedule stay on device for one fit; changing the
trajectory invalidates residual/derivative reuse. A conservative 1 GiB payload
check precedes device uploads. This is not a process RSS or CUDA allocator bound.

IMU residuals/derivatives, clock-offset central differences and their segment
selection, LM damping/acceptance/stopping, the solve, marginal covariance,
holdout and signed controls keep their CPU implementation. Return budgets,
measurement weights and numerical convergence thresholds are unchanged.
NumPy remains the default and does not import CuPy.

## Complete public-data comparison

| Scope | NumPy planes | CUDA planes | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 65.884 s | 61.146 s | 1.077x |
| Four fits, including covariance | 10.239 s | 3.506 s | 2.921x |

Two alternating fresh-process pairs replay six seconds of the public GLIM Livox
bag with four map rounds and one BLAS thread. Each worker has a separate empty
CuPy compiler cache; first initialization/compilation are included. Both sides
use identical CUDA scan odometry and the same dense odometry settings. Host
background load is unreserved. These are observations for this workload, with
different scopes for fit and whole-worker ratios.

The [bound archive](../assets/cuda_spline_planes/README.md) retains every full
result, full fit, measurement/log, source capsule/manifest, launcher/schema and
source-bound correctness run. The CPU-only audit checks the actual adapter
source identity and work against each round's plane count. Fit status,
iterations and dimensions must match exactly; all calibration and trajectory
fields and covariance are compared at explicit roundoff tolerances. The
existing physical/evidence checks apply to the complete pipeline. This path
does not claim bit equality. The clip has no independent extrinsic truth and
the final policy verdict remains INCONCLUSIVE.

## Usage

Install the optional CUDA dependency and explicitly select the plane backend:

```bash
pip install 'calibrex[cuda]'
calibrex imu-lidar joint path/to/public.bag \
  --lidar-topic /livox/points --imu-topic /livox/imu --acceleration-unit g \
  --dataset-family glim_versatile --dataset-license 'upstream AIST GLIM dataset terms' \
  --odometry-backend cuda --fit-plane-backend cuda --output calibration.json
```

The Python fitter accepts
`BSplineLidarImuFitOptions(plane_backend="cuda")`. Its result reports the
executed kernel identity and actual work. Bag artifacts also retain every
round's identity/counters in `options.fit_execution` when CUDA planes are
requested. Existing artifact schemas are unchanged.

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cuda-cuda \
  --baseline-commit c3845d1eb42326e6da11a235b5e6239c33448160 \
  --current-fit-plane-backend cuda --max-seconds 6 --max-rounds 4 --repeats 2 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-cuda-spline-planes-pairs
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/cuda_spline_planes
```

Use committed runtime sources and a new output directory. Keep source, HEAD,
comparison launcher and schema fixed until all workers finish.
