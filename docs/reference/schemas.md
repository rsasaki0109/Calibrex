# Schemas Reference

Calibrex ships JSON Schema definitions for every config, result, and report
sidecar artifact. Schemas live in `schemas/` at the repository root and are
regenerated with:

```bash
calibrex schema all --output-dir schemas
```

| Schema | Description |
|---|---|
| [`assessment.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/assessment.schema.json) | `AssessmentArtifact` — machine-readable falsification assessment (`assessment.json`). |
| [`comparison.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/comparison.schema.json) | `ResultComparison` — machine-readable comparison between two Calibrex results (`comparison.json`). |
| [`benchmark_definition.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/benchmark_definition.schema.json) | `BenchmarkDefinition` — shared method × split trial matrix before aggregation. |
| [`benchmark.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/benchmark.schema.json) | `BenchmarkArtifact` — N-way failure rates, distributions, rankings, paired bootstrap comparisons, and provenance. |
| [`calibration_ci.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/calibration_ci.schema.json) | `CalibrationCIArtifact` — digest-bound CI inputs, falsification and compatibility gates, artifact paths, and execution provenance. |
| [`config.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/config.schema.json) | `CalibrationConfig` — schema for `calibrex calibrate` configuration files. |
| [`dataset_manifest.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/dataset_manifest.schema.json) | `DatasetManifest` — portable dataset manifest independent of ROS message classes. |
| [`doctor.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/doctor.schema.json) | `DoctorArtifact` — environment, dataset readiness, workflow suggestions, and generation provenance from `calibrex doctor`. |
| [`depth_provider.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/depth_provider.schema.json) | `DepthProviderArtifact` — model/environment identity and digest-pinned depth-map observations. |
| [`continuous_time_camera_lidar_problem.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/continuous_time_camera_lidar_problem.schema.json) | `ContinuousTimeCameraLidarProblemArtifact` — dataset license, per-point firing times, exposure/readout times, probabilistic correspondences, initial and optional reference extrinsic/clock state, and either supplied body twists or a digest-pinned piecewise SE(3) trajectory. |
| [`continuous_time_camera_lidar_result.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/continuous_time_camera_lidar_result.schema.json) | `ContinuousTimeCameraLidarResultArtifact` — joint extrinsic/clock estimate, optional reference errors, selected trajectory model, complete candidate trace, observability, disjoint holdout evidence, and source digest. |
| [`probabilistic_correspondence.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/probabilistic_correspondence.schema.json) | `ProbabilisticCorrespondenceArtifact` — separate dataset/provider license lineage plus 2D–3D means, covariance, outlier probability, and reliability. |
| [`probabilistic_pnp_result.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/probabilistic_pnp_result.schema.json) | `ProbabilisticPnpResultArtifact` — robust pose output, uncertainty-aware diagnostics, tool identity, and immutable input provenance. |
| [`probabilistic_refinement_result.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/probabilistic_refinement_result.schema.json) | `ProbabilisticRefinementResultArtifact` — explicitly sourced problem-pose or D2D-trace initialization, shared multi-frame pose, reference errors, uncertainty ablations, disjoint holdout, full trace, and all input digests. |
| [`camera_lidar_problem.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/camera_lidar_problem.schema.json) | `CameraLidarCalibrationProblem` — fixed depth/LiDAR observations, transforms, bounds, splits, and digests. |
| [`camera_lidar_sota_audit_protocol.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/camera_lidar_sota_audit_protocol.schema.json) | `CameraLidarSotaAuditProtocol` — immutable category, numerical gates, coverage requirements, artifact locators, and digests. |
| [`camera_lidar_sota_audit_result.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/camera_lidar_sota_audit_result.schema.json) | `CameraLidarSotaAuditResult` — requirement-by-requirement achieved/contradicted/missing evidence and supported/refuted/incomplete claim verdict. |
| [`camera_lidar_benchmark_protocol.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/camera_lidar_benchmark_protocol.schema.json) | `CameraLidarBenchmarkProtocol` — immutable frame sampling, perturbations, hit rule, isolation, and metrics. |
| [`calibration_candidate_trace.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/calibration_candidate_trace.schema.json) | `CalibrationCandidateTrace` — every objective evaluation, accepted update, stopping reason, output, and independent hit decision. |
| [`bullseye_plot.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/bullseye_plot.schema.json) | `BullseyePlotArtifact` — digest-pinned three-plane rotation recovery visualization and all plotted trial points. |
| [`kitti_benchmark_input.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/kitti_benchmark_input.schema.json) | `KITTIBenchmarkInputManifest` — digest-locked official KITTI raw frame, timestamp, and calibration inputs. |
| [`evidence_bundle.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/evidence_bundle.schema.json) | `EvidenceBundleManifest` — digest manifest for report and evidence artifacts (`bundle.json`). |
| [`evidence_bundle_verification.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/evidence_bundle_verification.schema.json) | `EvidenceBundleVerification` — machine-readable verification result for an evidence bundle (`verification.json`). |
| [`external_run.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/external_run.schema.json) | `ExternalCalibrationRunArtifact` — digest-bound external execution/import identity, conventions, isolation, status, parsed outputs, and provenance (`external-run.json`). |
| [`kitti_falsification.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/kitti_falsification.schema.json) | `KITTIFalsificationBenchmarkArtifact` — pinned KITTI frames, source metadata, input/output digests, frozen protocol, reference/known-bad trial decisions, and provenance (`benchmark.json`). |
| [`online_timeline.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/online_timeline.schema.json) | `OnlineCalibrationTimelineArtifact` — per-batch evidence timeline for an online/streaming calibration session (`timeline.json`). |
| [`policy.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/policy.schema.json) | `PolicyArtifact` — machine-readable assessment policy declarations for one run (`policy.json`). |
| [`protocol.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/protocol.schema.json) | `ProtocolArtifact` — machine-readable evidence protocol declarations for one run (`protocol.json`). |
| [`report_comparison.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/report_comparison.schema.json) | `ReportComparison` — machine-readable N-way comparison across labeled Calibrex results (`report_comparison.json`). |
| [`report_degeneracy.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/report_degeneracy.schema.json) | `ReportDegeneracyArtifact` — schema for `degeneracy.json` report sidecars. |
| [`report_evidence.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/report_evidence.schema.json) | `ReportEvidenceArtifact` — schema for `evidence.json` report sidecars. |
| [`report_metrics.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/report_metrics.schema.json) | `ReportMetricsArtifact` — schema for `metrics.json` report sidecars. |
| [`report_observability.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/report_observability.schema.json) | `ReportObservabilityArtifact` — schema for `observability.json` report sidecars. |
| [`report_summary.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/report_summary.schema.json) | `ReportSummaryArtifact` — schema for `summary.json` report sidecars. |
| [`result.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/result.schema.json) | `CalibrationResult` — schema for `result.yaml`, the primary calibration output. |
| [`transforms.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/transforms.schema.json) | `TransformArtifact` — machine-readable transform estimates for one calibration run (`transforms.json`). |
