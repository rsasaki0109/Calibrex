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
| [`config.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/config.schema.json) | `CalibrationConfig` — schema for `calibrex calibrate` configuration files. |
| [`dataset_manifest.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/dataset_manifest.schema.json) | `DatasetManifest` — portable dataset manifest independent of ROS message classes. |
| [`evidence_bundle.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/evidence_bundle.schema.json) | `EvidenceBundleManifest` — digest manifest for report and evidence artifacts (`bundle.json`). |
| [`evidence_bundle_verification.schema.json`](https://github.com/rsasaki0109/Calibrex/blob/main/schemas/evidence_bundle_verification.schema.json) | `EvidenceBundleVerification` — machine-readable verification result for an evidence bundle (`verification.json`). |
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
