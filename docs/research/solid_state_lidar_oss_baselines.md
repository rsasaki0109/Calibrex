# Solid-state LiDAR OSS baseline plan

This page records the external implementations that are relevant to
solid-state LiDAR calibration. It is a comparison plan, not a claim that every
tool has already been run on the public Calibrex captures.

The comparison boundary is deliberate: the Calibrex core remains
ROS-independent, and external code is executed or imported through an adapter
with a pinned commit, SPDX license, input/output digests, and an explicit
train/holdout declaration.

## Candidate matrix

| Candidate | Sensor problem | Why it matters for solid-state LiDAR | Planned boundary |
| --- | --- | --- | --- |
| [Livox automatic calibration](https://github.com/Livox-SDK/Livox_automatic_calibration) / TFAC-Livox | LiDAR–LiDAR extrinsic | Directly targets multi-LiDAR geometry and describes map registration with a Livox reference sensor. The vendor notes require synchronized captures, a usable initial transform, and shared scene coverage. | External command/container first; pin the repository commit and license metadata before scoring. |
| [iKalibr](https://github.com/Unsigned-Long/iKalibr) | Targetless multi-sensor spatial + temporal calibration | Continuous-time estimation, LiDAR/IMU support, and a published example containing Livox Avia plus Velodyne VLP-32C make it the closest independent research baseline. It expects an integrated inertial sensor suite and sufficiently excited motion. | External container/import adapter; compare only after converting its output to the common Calibrex artifact and recomputing Calibrex holdout evidence. |
| [IA-HeLiC](https://cslinzhang.github.io/IA-HeLiC/) | IMU-assisted heterogeneous LiDAR extrinsic | Its public description specifically pairs a Livox Avia solid-state LiDAR with an RS-LiDAR-16 and uses continuous-time optimization. | Research candidate; first verify source commit, license, build, and public-data access. |
| [Open3D GICP](https://www.open3d.org/docs/latest/tutorial/pipelines/generalized_iterative_closest_point.html) | Generic point-cloud registration | A low-friction, permissively licensed registration baseline. It is useful as a sanity check, but it does not by itself solve temporal offset or LiDAR rolling/non-repetitive acquisition effects. | Optional import adapter; report Open3D metrics only as raw provenance and use Calibrex spatial holdout metrics for comparison. |
| [Koide direct visual-LiDAR calibration](https://github.com/koide3/direct_visual_lidar_calibration) | Camera–LiDAR extrinsic | MIT-licensed targetless camera–LiDAR calibration supports non-repetitive scan LiDAR and is a natural camera×LiDAR gallery/comparison track. It is not a LiDAR–LiDAR baseline. | Separate camera–LiDAR adapter track; do not mix its objective with the current pair benchmark. |
| [UniCalib](https://github.com/han-15/UniCalib) | Learned targetless camera–LiDAR extrinsic | A recent camera–LiDAR method with unified depth representations and uncertainty-aware correspondences. It needs model/checkpoint provenance and likely GPU/runtime isolation. | External-only candidate; require checkpoint hash, training/evaluation split evidence, and container digest before any claim. |

`livox_camera_calib` is intentionally not a core dependency: its repository is
GPL-2.0. It may be evaluated in a separately distributed process if a future
experiment needs it, but GPL source must not be copied into `src/calibrex`.

## Common evaluation contract

Every external run should produce an `slac.external_calibration_run/v0.1`
artifact containing:

1. the exact tool repository and full commit;
2. SPDX license, adapter version, execution mode, and container digest when
   applicable;
3. SHA-256 digests for the public input manifest, converted input, result, and
   any model/checkpoint;
4. frame and timestamp conventions, including whether the tool estimated a
   time offset;
5. a declaration that holdout frames were not used for initialization,
   tuning, or candidate selection.

The imported transform is then scored by the same Calibrex evaluator used for
native variants. The primary comparison remains spatial holdout RMSE and
support/observability gates; a tool's native loss, fitness, or training error
is retained as provenance but is not treated as cross-tool evidence.

## Executed OSS fixture: Open3D GICP

The optional Open3D dependency was installed at the pinned version `0.19.0`
and run on the public Livox Horizon PCD pair declared by
`examples/public_datasets/livox_horizon_horizon_pcd_sample/icp_comparison_config.yaml`.
The existing integration test
`tests/integration/test_livox_registration_comparison.py` fixes the expected
provenance and metrics:

| Evidence | Result |
| --- | ---: |
| Open3D train RMSE | 0.4863 m |
| Open3D holdout RMSE | 2.9039 m (**FAIL**) |
| Common Calibrex holdout RMSE | 5.2634 m (**FAIL**) |
| Inlier fraction | 0.2771 |
| Minimum correspondence Jaccard | 0.6873 |
| Weak directions | 3 (**WARN**) |
| Native-vs-GICP rotation delta | 18.7127° (**WARN**) |
| Native-vs-GICP translation delta | 0.8550 m (**WARN**) |

The comparison status is `inconclusive`, even though the external backend is
available and the artifact records `tool_version=0.19.0`, `license_spdx=MIT`,
`external_code_vendored=false`, input hashes, and adapter options. This is the
intended result: generic GICP is a useful OSS registration baseline, but this
fixture does not establish a successful solid-state calibration result and it
does not estimate capture-time offset.

## Execution order

1. Finish the native v0.4 3×3×3 public matrix with the train-only MAD 2.5
   candidate.
2. Add a Livox/TFAC adapter contract and test it with a fixture result before
   attempting a full run.
3. Prepare an iKalibr adapter because it is the strongest temporal baseline;
   start with its published Livox/Velodyne example rather than claiming that
   the current public bags are directly compatible.
4. Add IA-HeLiC as an independent research comparison if its source and data
   can be pinned reproducibly.
5. Keep camera–LiDAR methods (Koide and UniCalib) in a separate benchmark and
   use the README GIF gallery only for visual evidence, never as calibration
   accuracy evidence.

## References

- Chen et al., [iKalibr: Unified Targetless Spatiotemporal Calibration for
  Resilient Integrated Inertial Systems](https://arxiv.org/abs/2407.11420).
- Yan et al., [IA-HeLiC: IMU-Assisted Target-Free Extrinsic Calibration of
  Heterogeneous LiDARs Based on Continuous-Time Optimization](https://cslinzhang.github.io/IA-HeLiC/).
- Koide et al., [General, Single-shot, Target-less, and Automatic LiDAR-Camera
  Extrinsic Calibration Toolbox](https://arxiv.org/abs/2302.05094).
- Han et al., [UniCalib: Targetless LiDAR-camera Calibration via Probabilistic
  Flow on Unified Depth Representations](https://openaccess.thecvf.com/content/WACV2026/html/Han_UniCalib_Targetless_LiDAR-camera_Calibration_via_Probabilistic_Flow_on_Unified_Depth_WACV_2026_paper.html).
