"""Consistency evidence for reusing a LiDAR calibration across capture windows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path

from calibrex import __version__
from calibrex.core.dynamic_window import (
    DynamicWindowCaptureWindow,
    DynamicWindowConsistencyArtifact,
    DynamicWindowConsistencyInput,
    DynamicWindowConsistencyPair,
    DynamicWindowConsistencyProvenance,
    DynamicWindowConsistencyThresholds,
)
from calibrex.core.provenance import sha256_path
from calibrex.core.result import CalibrationResult, Grade
from calibrex.evaluation.report_compare import compare_reports


def evaluate_dynamic_window_consistency(
    labeled_results: Sequence[tuple[str, CalibrationResult]],
    *,
    paths: Mapping[str, str | Path],
    transform_id: str,
    thresholds: DynamicWindowConsistencyThresholds,
    reference_label: str | None = None,
    tool_name: str = "calibrex",
    tool_version: str = __version__,
) -> DynamicWindowConsistencyArtifact:
    """Evaluate whether one transform is stable across all labeled windows.

    The gate deliberately compares every unordered pair. A result can pass its
    own holdout and trajectory checks while still landing in a different local
    solution, so single-result quality is retained as a warning and transform
    dispersion is the blocking criterion here.
    """

    if len(labeled_results) < 2:
        raise ValueError("dynamic-window consistency requires at least two results")
    labels = [label for label, _result in labeled_results]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate dynamic-window labels: {labels}")
    if reference_label is not None and reference_label not in labels:
        raise ValueError(f"reference label not found among entries: {reference_label}")
    missing_paths = sorted(set(labels) - set(paths))
    if missing_paths:
        raise ValueError(
            "dynamic-window consistency requires result paths for all labels: "
            + ", ".join(missing_paths)
        )

    ordered_results = sorted(labeled_results, key=lambda item: item[0])
    results_by_label = dict(ordered_results)
    report = compare_reports(
        ordered_results,
        paths={label: paths[label] for label, _result in ordered_results},
    )
    input_records = [
        _input_record(
            label,
            results_by_label[label],
            Path(paths[label]),
            transform_id=transform_id,
        )
        for label in sorted(results_by_label)
    ]

    pairs: list[DynamicWindowConsistencyPair] = []
    for left_label, right_label in combinations(sorted(results_by_label), 2):
        pairs.append(
            _compare_transform_pair(
                left_label,
                results_by_label[left_label],
                right_label,
                results_by_label[right_label],
                transform_id=transform_id,
                thresholds=thresholds,
            )
        )

    blocking_failures = [
        f"{pair.left_label} vs {pair.right_label}: {pair.reason}"
        for pair in pairs
        if pair.grade == "fail"
    ]
    warnings: list[str] = []
    if report.summary.protocol_compatibility_status != "compatible":
        reasons = sorted(
            {
                reason
                for comparison_pair in report.pairwise.values()
                for reason in comparison_pair.comparison.protocol_compatibility.reasons
            }
        )
        detail = "; ".join(reasons) if reasons else "protocol compatibility was not established"
        warnings.append(
            "evidence protocol status "
            f"{report.summary.protocol_compatibility_status}: {detail}"
        )
    quality_warning_labels = [
        record.label for record in input_records if record.quality_grade != "pass"
    ]
    if quality_warning_labels:
        warnings.append(
            "input result quality is not PASS for: " + ", ".join(quality_warning_labels)
        )

    grade: Grade
    if blocking_failures:
        grade = "fail"
    elif warnings:
        grade = "warn"
    else:
        grade = "pass"

    source_paths = [str(paths[label]) for label in sorted(results_by_label)]
    source_sha256 = {
        label: _required_sha256(Path(paths[label]))
        for label in sorted(results_by_label)
    }
    return DynamicWindowConsistencyArtifact(
        transform_id=transform_id,
        reference_label=reference_label,
        thresholds=thresholds,
        inputs=input_records,
        pairs=pairs,
        protocol_compatibility_status=report.summary.protocol_compatibility_status,
        grade=grade,
        blocking_failures=blocking_failures,
        warnings=warnings,
        provenance=DynamicWindowConsistencyProvenance(
            source_paths=source_paths,
            source_sha256=source_sha256,
            tool_name=tool_name,
            tool_version=tool_version,
            notes=[
                "all unordered result pairs are evaluated against the declared "
                "transform-delta thresholds",
                "single-result holdout/trajectory quality is retained as a warning, "
                "not a substitute for cross-window consistency",
            ],
        ),
    )


def _input_record(
    label: str,
    result: CalibrationResult,
    path: Path,
    *,
    transform_id: str,
) -> DynamicWindowConsistencyInput:
    provenance = result.run.provenance
    return DynamicWindowConsistencyInput(
        label=label,
        result_path=str(path),
        result_sha256=_required_sha256(path),
        run_id=result.run.id,
        quality_grade=result.quality.grade,
        transform_present=transform_id in result.transforms,
        capture_window_start_timestamp_ns=_int_or_none(
            provenance.get("rosbag2_capture_window_start_timestamp_ns")
            or provenance.get("rosbag1_capture_window_start_timestamp_ns")
        ),
        capture_window_end_timestamp_ns=_int_or_none(
            provenance.get("rosbag2_capture_window_end_timestamp_ns")
            or provenance.get("rosbag1_capture_window_end_timestamp_ns")
        ),
        capture_windows=_capture_windows(provenance),
    )


def _compare_transform_pair(
    left_label: str,
    left: CalibrationResult,
    right_label: str,
    right: CalibrationResult,
    *,
    transform_id: str,
    thresholds: DynamicWindowConsistencyThresholds,
) -> DynamicWindowConsistencyPair:
    left_transform = left.transforms.get(transform_id)
    right_transform = right.transforms.get(transform_id)
    if left_transform is None or right_transform is None:
        missing = []
        if left_transform is None:
            missing.append(left_label)
        if right_transform is None:
            missing.append(right_label)
        return DynamicWindowConsistencyPair(
            left_label=left_label,
            right_label=right_label,
            grade="fail",
            reason=f"transform {transform_id!r} is missing from: {', '.join(missing)}",
        )

    left_se3 = left_transform.as_se3()
    right_se3 = right_transform.as_se3()
    translation_delta = math.dist(left_se3.translation_m, right_se3.translation_m)
    dot = abs(
        sum(
            left_component * right_component
            for left_component, right_component in zip(
                left_se3.rotation_quat_xyzw,
                right_se3.rotation_quat_xyzw,
                strict=True,
            )
        )
    )
    rotation_delta_deg = math.degrees(2.0 * math.acos(min(1.0, max(0.0, dot))))
    within_translation = translation_delta <= thresholds.max_translation_delta_m
    within_rotation = rotation_delta_deg <= thresholds.max_rotation_delta_deg
    grade: Grade = "pass" if within_translation and within_rotation else "fail"
    reason = (
        f"translation delta {translation_delta:.6f} m <= "
        f"{thresholds.max_translation_delta_m:.6f} m; rotation delta "
        f"{rotation_delta_deg:.6f} deg <= {thresholds.max_rotation_delta_deg:.6f} deg"
    )
    if grade == "fail":
        exceeded: list[str] = []
        if not within_translation:
            exceeded.append("translation threshold exceeded")
        if not within_rotation:
            exceeded.append("rotation threshold exceeded")
        reason += " (" + ", ".join(exceeded) + ")"
    return DynamicWindowConsistencyPair(
        left_label=left_label,
        right_label=right_label,
        translation_delta_m=translation_delta,
        rotation_delta_deg=rotation_delta_deg,
        grade=grade,
        reason=reason,
    )


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _capture_windows(provenance: Mapping[str, object]) -> list[DynamicWindowCaptureWindow]:
    """Copy multi-window selection provenance into consistency evidence."""

    raw_windows = provenance.get("rosbag2_capture_windows")
    if not isinstance(raw_windows, list):
        raw_windows = provenance.get("rosbag1_capture_windows")
    if isinstance(raw_windows, list):
        windows: list[DynamicWindowCaptureWindow] = []
        for raw_window in raw_windows:
            if not isinstance(raw_window, Mapping):
                continue
            windows.append(
                DynamicWindowCaptureWindow(
                    window_index=_int_or_none(raw_window.get("window_index")),
                    start_timestamp_ns=_int_or_none(
                        raw_window.get("start_timestamp_ns")
                    ),
                    end_timestamp_ns=_int_or_none(raw_window.get("end_timestamp_ns")),
                    selected_source_message_count=_int_or_none(
                        raw_window.get("selected_source_message_count")
                    ),
                    selected_source_point_count=_int_or_none(
                        raw_window.get("selected_source_point_count")
                    ),
                    selected_target_message_count=_int_or_none(
                        raw_window.get("selected_target_message_count")
                    ),
                    selected_target_point_count=_int_or_none(
                        raw_window.get("selected_target_point_count")
                    ),
                )
            )
        if windows:
            return windows

    start_timestamp_ns = _int_or_none(
        provenance.get("rosbag2_capture_window_start_timestamp_ns")
        or provenance.get("rosbag1_capture_window_start_timestamp_ns")
    )
    end_timestamp_ns = _int_or_none(
        provenance.get("rosbag2_capture_window_end_timestamp_ns")
        or provenance.get("rosbag1_capture_window_end_timestamp_ns")
    )
    if start_timestamp_ns is None and end_timestamp_ns is None:
        return []
    return [
        DynamicWindowCaptureWindow(
            start_timestamp_ns=start_timestamp_ns,
            end_timestamp_ns=end_timestamp_ns,
        )
    ]


def _required_sha256(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"dynamic-window consistency input does not exist: {path}")
    return digest
