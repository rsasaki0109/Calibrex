"""Independent holdout evaluation of imported Camera--IMU calibrations.

The evaluator checks the Kalibr relation ``omega_cam = R_cam_imu omega_imu``
with gyro samples looked up at ``t_imu = t_cam + timeshift_cam_imu``. Only a
constant gyro bias is fitted, and only on the training intervals; the imported
rotation and time shift are never adjusted. Signed rotation and time controls
must raise the holdout residual, otherwise the recording cannot falsify the
candidate and the evidence is not admissible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex.core.external_camera_imu_evidence import (
    CameraImuCandidate,
    CameraImuControl,
    CameraImuEvidenceThresholds,
    CameraImuExcitation,
    CameraImuMotionRecording,
    CameraImuMotionRecordingProvenance,
    CameraImuTemporalSplit,
    CameraOrientationSample,
    ExternalCameraImuEvidenceArtifact,
    ExternalCameraImuEvidenceProvenance,
    ExternalCameraImuEvidenceStatus,
    ExternalRunReference,
    ImuGyroSample,
    RecordingReference,
    load_camera_imu_motion_recording,
)
from calibrex.core.external_run import ExternalCalibrationRunArtifact, load_external_run
from calibrex.core.io import write_mapping
from calibrex.core.provenance import git_commit, sha256_path

FloatArray: TypeAlias = NDArray[np.float64]

_GYRO_SUBSAMPLES = 5
_AXES: tuple[tuple[str, int], ...] = (("x", 0), ("y", 1), ("z", 2))


@dataclass(frozen=True)
class _Interval:
    start_sec: float
    end_sec: float
    omega_cam_rad_s: FloatArray

    @property
    def mid_sec(self) -> float:
        return 0.5 * (self.start_sec + self.end_sec)


@dataclass(frozen=True)
class _Fit:
    bias_rad_s: FloatArray
    train_rmse_rad_s: float
    holdout_rmse_rad_s: float


def evaluate_external_camera_imu(
    external_run_path: str | Path,
    recording_path: str | Path,
    *,
    camera_name: str = "cam0",
    thresholds: CameraImuEvidenceThresholds | None = None,
    evaluation_id: str | None = None,
    command: Sequence[str] = (),
) -> ExternalCameraImuEvidenceArtifact:
    """Evaluate an imported ``T_cam_imu``/time shift on a separate recording."""

    gates = thresholds or CameraImuEvidenceThresholds()
    run_path = Path(external_run_path)
    rec_path = Path(recording_path)
    run_sha = sha256_path(run_path)
    rec_sha = sha256_path(rec_path)
    input_sha = {
        key: value
        for key, value in (("external_run", run_sha), ("recording", rec_sha))
        if value is not None
    }
    provenance = ExternalCameraImuEvidenceProvenance(
        command=list(command),
        input_sha256=input_sha,
    )
    run_ref = ExternalRunReference(path=str(run_path), sha256=run_sha)
    rec_ref = RecordingReference(path=str(rec_path), sha256=rec_sha)
    identifier = evaluation_id or f"external-camera-imu:{run_path.stem}:{camera_name}"

    def blocked(reason: str, warnings: Sequence[str] = ()) -> ExternalCameraImuEvidenceArtifact:
        return ExternalCameraImuEvidenceArtifact(
            evaluation_id=identifier,
            camera_name=camera_name,
            external_run=run_ref,
            recording=rec_ref,
            thresholds=gates,
            status="blocked",
            reasons=[reason],
            warnings=list(warnings),
            provenance=provenance,
        )

    if run_sha is None:
        return blocked(f"external-run artifact does not exist: {run_path}")
    if rec_sha is None:
        return blocked(f"evaluation recording does not exist: {rec_path}")
    try:
        run = load_external_run(run_path)
    except (OSError, ValueError) as exc:
        return blocked(f"external-run artifact is not schema-valid: {exc}")
    run_ref = _run_reference(run_path, run_sha, run)
    try:
        recording = load_camera_imu_motion_recording(rec_path)
    except (OSError, ValueError) as exc:
        return blocked(f"evaluation recording is not schema-valid: {exc}")
    rec_ref = RecordingReference(
        path=str(rec_path),
        sha256=rec_sha,
        recording_id=recording.recording_id,
        camera_orientation_count=len(recording.camera_orientations),
        imu_sample_count=len(recording.imu_gyro),
        synthetic=recording.provenance.synthetic,
    )

    if run.status != "success":
        return blocked(f"external run status is {run.status!r}, not 'success'")
    if rec_sha in run_ref.fitting_input_sha256:
        return blocked(
            "evaluation recording bytes match a declared external fitting input; "
            "holdout evidence would not be independent"
        )
    if recording.camera_name != camera_name:
        return blocked(
            f"recording camera {recording.camera_name!r} does not match "
            f"evaluated camera {camera_name!r}"
        )
    transform_name = f"T_{camera_name}_{recording.imu_name}"
    transform = run.parsed_outputs.transforms.get(transform_name)
    if transform is None:
        return blocked(f"external run has no {transform_name} transform")
    if transform.parent != camera_name or transform.child != recording.imu_name:
        return blocked(
            f"{transform_name} is {transform.parent}<-{transform.child}, "
            f"expected {camera_name}<-{recording.imu_name}"
        )

    warnings: list[str] = []
    time_offset_name = f"dt_{recording.imu_name}_minus_{camera_name}"
    time_offset_declared = time_offset_name in run.parsed_outputs.time_offsets_seconds
    timeshift = float(run.parsed_outputs.time_offsets_seconds.get(time_offset_name, 0.0))
    if not time_offset_declared:
        warnings.append(f"external run declares no {time_offset_name}; evaluated with 0 s")
    if not run.train_data_isolation.declared:
        warnings.append("external training-data isolation is not declared")
    if not run_ref.fitting_input_sha256:
        warnings.append("external run declares no digest-bound fitting input")
    candidate = CameraImuCandidate(
        transform_name=transform_name,
        rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        translation_m=list(transform.translation_m),
        time_offset_name=time_offset_name if time_offset_declared else None,
        timeshift_cam_imu_sec=timeshift,
        time_offset_declared=time_offset_declared,
    )
    rotation = Rotation.from_quat(transform.rotation_quat_xyzw).as_matrix()

    intervals = _camera_intervals(recording)
    gyro_t: FloatArray = np.asarray(
        [item.timestamp_sec for item in recording.imu_gyro], dtype=float
    )
    gyro_w: FloatArray = np.asarray([item.omega_rad_s for item in recording.imu_gyro], dtype=float)
    split, train, holdout = _split(intervals, gates, timeshift, gyro_t)

    def result(
        status: ExternalCameraImuEvidenceStatus,
        reasons: list[str],
        **fields: object,
    ) -> ExternalCameraImuEvidenceArtifact:
        return ExternalCameraImuEvidenceArtifact.model_validate(
            {
                "evaluation_id": identifier,
                "camera_name": camera_name,
                "external_run": run_ref,
                "recording": rec_ref,
                "candidate": candidate,
                "thresholds": gates,
                "split": split,
                "status": status,
                "reasons": reasons,
                "warnings": warnings,
                "provenance": provenance,
                **fields,
            }
        )

    if len(train) < gates.min_train_intervals or len(holdout) < gates.min_holdout_intervals:
        return result(
            "inconclusive",
            [
                f"{len(train)} train / {len(holdout)} holdout intervals are below the "
                f"{gates.min_train_intervals} / {gates.min_holdout_intervals} minimum"
            ],
        )

    fit = _fit(train, holdout, rotation, timeshift, gyro_t, gyro_w)
    excitation = _excitation(holdout, timeshift, gyro_t, gyro_w)
    controls = _controls(train, holdout, rotation, timeshift, gyro_t, gyro_w, fit, gates)
    detection = sum(control.detected for control in controls) / len(controls)
    fields: dict[str, object] = {
        "estimated_gyro_bias_rad_s": [float(value) for value in fit.bias_rad_s],
        "train_rate_rmse_rad_s": fit.train_rmse_rad_s,
        "holdout_rate_rmse_rad_s": fit.holdout_rmse_rad_s,
        "excitation": excitation,
        "controls": controls,
        "control_detection_fraction": detection,
    }

    inconclusive: list[str] = []
    if excitation.min_rate_excitation_rad_s < gates.min_rate_excitation_rad_s:
        inconclusive.append(
            f"holdout rate excitation {excitation.min_rate_excitation_rad_s:.3f} rad/s "
            f"is below {gates.min_rate_excitation_rad_s:.3f} rad/s on its weakest axis; "
            "rotation about that axis is not observable"
        )
    if excitation.rms_angular_acceleration_rad_s2 < gates.min_angular_acceleration_rad_s2:
        inconclusive.append(
            f"holdout angular acceleration {excitation.rms_angular_acceleration_rad_s2:.3f} "
            f"rad/s^2 is below {gates.min_angular_acceleration_rad_s2:.3f}; "
            "time shift is not observable"
        )
    if inconclusive:
        return result("inconclusive", inconclusive, **fields)

    failures: list[str] = []
    if fit.holdout_rmse_rad_s > gates.max_holdout_rate_rmse_rad_s:
        failures.append(
            f"holdout angular-rate RMSE {fit.holdout_rmse_rad_s:.4f} rad/s exceeds "
            f"{gates.max_holdout_rate_rmse_rad_s:.4f} rad/s"
        )
    if detection < gates.min_control_detection_fraction:
        missed = ", ".join(control.control_id for control in controls if not control.detected)
        failures.append(
            f"known-bad detection fraction {detection:.2f} is below "
            f"{gates.min_control_detection_fraction:.2f} (missed: {missed})"
        )
    if failures:
        return result("fail", failures, **fields)

    passed = (
        f"holdout angular-rate RMSE {fit.holdout_rmse_rad_s:.4f} rad/s is within budget "
        f"and {detection:.0%} of signed controls raise it"
    )
    if warnings:
        return result("warn", [passed, "admissible only after review of warnings"], **fields)
    return result("pass", [passed], **fields)


def _run_reference(
    path: Path,
    sha256: str,
    run: ExternalCalibrationRunArtifact,
) -> ExternalRunReference:
    return ExternalRunReference(
        path=str(path),
        sha256=sha256,
        run_id=run.run_id,
        adapter_name=run.adapter_name,
        tool_name=run.tool.name,
        tool_version=run.tool.version,
        source_commit=run.tool.source_commit,
        license_spdx=run.tool.license_spdx,
        status=run.status,
        training_isolation_declared=run.train_data_isolation.declared,
        fitting_input_sha256=sorted(
            {artifact.sha256 for artifact in run.artifacts if artifact.role == "input"}
        ),
    )


def _camera_intervals(recording: CameraImuMotionRecording) -> list[_Interval]:
    stamps = [sample.timestamp_sec for sample in recording.camera_orientations]
    rotations = Rotation.from_quat(
        [sample.rotation_quat_xyzw for sample in recording.camera_orientations]
    )
    intervals: list[_Interval] = []
    for index in range(len(stamps) - 1):
        dt = stamps[index + 1] - stamps[index]
        delta = (rotations[index].inv() * rotations[index + 1]).as_rotvec()
        intervals.append(
            _Interval(
                start_sec=stamps[index],
                end_sec=stamps[index + 1],
                omega_cam_rad_s=np.asarray(delta, dtype=float) / dt,
            )
        )
    return intervals


def _split(
    intervals: list[_Interval],
    gates: CameraImuEvidenceThresholds,
    timeshift: float,
    gyro_t: FloatArray,
) -> tuple[CameraImuTemporalSplit, list[_Interval], list[_Interval]]:
    # Gyro coverage is checked at the widest control shift so every candidate
    # and control is scored on the same interval set.
    margin = abs(timeshift) + gates.time_control_sec
    covered = [
        interval
        for interval in intervals
        if interval.start_sec - margin >= gyro_t[0] and interval.end_sec + margin <= gyro_t[-1]
    ]
    if not covered:
        split = CameraImuTemporalSplit(
            holdout_start_sec=0.0,
            holdout_end_sec=0.0,
            guard_sec=0.0,
            train_interval_count=0,
            holdout_interval_count=0,
            excluded_interval_count=len(intervals),
        )
        return split, [], []
    first = covered[0].start_sec
    span = covered[-1].end_sec - first
    start = first + gates.holdout_fraction_start * span
    end = first + gates.holdout_fraction_end * span
    guard = 2.0 * max(interval.end_sec - interval.start_sec for interval in covered)
    train: list[_Interval] = []
    holdout: list[_Interval] = []
    for interval in covered:
        if interval.start_sec >= start and interval.end_sec <= end:
            holdout.append(interval)
        elif interval.end_sec <= start - guard or interval.start_sec >= end + guard:
            train.append(interval)
    split = CameraImuTemporalSplit(
        holdout_start_sec=start,
        holdout_end_sec=end,
        guard_sec=guard,
        train_interval_count=len(train),
        holdout_interval_count=len(holdout),
        excluded_interval_count=len(intervals) - len(train) - len(holdout),
    )
    return split, train, holdout


def _mean_gyro(
    intervals: Sequence[_Interval],
    timeshift: float,
    gyro_t: FloatArray,
    gyro_w: FloatArray,
) -> FloatArray:
    rows = []
    for interval in intervals:
        query = np.linspace(interval.start_sec, interval.end_sec, _GYRO_SUBSAMPLES) + timeshift
        rows.append(
            [float(np.mean(np.interp(query, gyro_t, gyro_w[:, axis]))) for axis in range(3)]
        )
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def _fit(
    train: Sequence[_Interval],
    holdout: Sequence[_Interval],
    rotation: FloatArray,
    timeshift: float,
    gyro_t: FloatArray,
    gyro_w: FloatArray,
) -> _Fit:
    train_cam = np.asarray([interval.omega_cam_rad_s for interval in train], dtype=float)
    holdout_cam = np.asarray([interval.omega_cam_rad_s for interval in holdout], dtype=float)
    train_imu = _mean_gyro(train, timeshift, gyro_t, gyro_w)
    holdout_imu = _mean_gyro(holdout, timeshift, gyro_t, gyro_w)
    # omega_cam = R (omega_imu + b)  =>  b = mean(R^T omega_cam - omega_imu)
    bias = np.mean(train_cam @ rotation - train_imu, axis=0)
    return _Fit(
        bias_rad_s=bias,
        train_rmse_rad_s=_rmse(train_cam - (train_imu + bias) @ rotation.T),
        holdout_rmse_rad_s=_rmse(holdout_cam - (holdout_imu + bias) @ rotation.T),
    )


def _rmse(residuals: FloatArray) -> float:
    return float(math.sqrt(float(np.mean(np.sum(residuals * residuals, axis=1)))))


def _excitation(
    holdout: Sequence[_Interval],
    timeshift: float,
    gyro_t: FloatArray,
    gyro_w: FloatArray,
) -> CameraImuExcitation:
    omega = _mean_gyro(holdout, timeshift, gyro_t, gyro_w)
    centered = omega - np.mean(omega, axis=0)
    eigen = np.linalg.eigvalsh(centered.T @ centered / len(omega))
    rate_eigen = [float(math.sqrt(max(value, 0.0))) for value in eigen]
    mids: FloatArray = np.asarray([interval.mid_sec for interval in holdout], dtype=float)
    cam = np.asarray([interval.omega_cam_rad_s for interval in holdout], dtype=float)
    acceleration = np.diff(cam, axis=0) / np.diff(mids)[:, None]
    return CameraImuExcitation(
        min_rate_excitation_rad_s=rate_eigen[0],
        rate_excitation_eigen_rad_s=rate_eigen,
        rms_angular_acceleration_rad_s2=_rmse(acceleration) if len(acceleration) else 0.0,
    )


def _controls(
    train: Sequence[_Interval],
    holdout: Sequence[_Interval],
    rotation: FloatArray,
    timeshift: float,
    gyro_t: FloatArray,
    gyro_w: FloatArray,
    baseline: _Fit,
    gates: CameraImuEvidenceThresholds,
) -> list[CameraImuControl]:
    controls: list[CameraImuControl] = []
    for sign in (1.0, -1.0):
        for axis_name, axis in _AXES:
            rotvec = np.zeros(3)
            rotvec[axis] = sign * math.radians(gates.rotation_control_deg)
            perturbed = rotation @ Rotation.from_rotvec(rotvec).as_matrix()
            fit = _fit(train, holdout, perturbed, timeshift, gyro_t, gyro_w)
            controls.append(
                _control(
                    f"rotation_{'+' if sign > 0 else '-'}{axis_name}",
                    "rotation",
                    axis_name,
                    sign * gates.rotation_control_deg,
                    "deg",
                    fit,
                    baseline,
                    gates,
                )
            )
    for sign in (1.0, -1.0):
        shift = sign * gates.time_control_sec
        fit = _fit(train, holdout, rotation, timeshift + shift, gyro_t, gyro_w)
        controls.append(
            _control(
                f"time_{'+' if sign > 0 else '-'}",
                "time",
                None,
                shift,
                "s",
                fit,
                baseline,
                gates,
            )
        )
    return controls


def _control(
    control_id: str,
    kind: str,
    axis: str | None,
    magnitude: float,
    unit: str,
    fit: _Fit,
    baseline: _Fit,
    gates: CameraImuEvidenceThresholds,
) -> CameraImuControl:
    increase = fit.holdout_rmse_rad_s - baseline.holdout_rmse_rad_s
    detected = (
        increase >= gates.control_min_rmse_increase_rad_s
        and fit.holdout_rmse_rad_s >= gates.control_min_rmse_ratio * baseline.holdout_rmse_rad_s
    )
    return CameraImuControl.model_validate(
        {
            "control_id": control_id,
            "kind": kind,
            "axis": axis,
            "signed_magnitude": magnitude,
            "unit": unit,
            "holdout_rate_rmse_rad_s": fit.holdout_rmse_rad_s,
            "rmse_increase_rad_s": increase,
            "detected": detected,
        }
    )


# Synthetic fixture ---------------------------------------------------------

_TRUE_ROTVEC_CAM_IMU = (1.2, -1.2, 1.2)
_TRUE_TRANSLATION_CAM_IMU = (0.05, -0.02, 0.01)
_TRUE_TIMESHIFT_CAM_IMU = 0.004


@dataclass(frozen=True)
class SyntheticCameraImuFixture:
    """Paths of one synthetic Kalibr Camera--IMU fixture."""

    camchain_path: Path
    fitting_recording_path: Path
    evaluation_recording_path: Path


def write_synthetic_camera_imu_fixture(
    output_dir: str | Path,
    *,
    seed: int = 20260927,
    rotation_error_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    timeshift_error_sec: float = 0.0,
    motion: str = "full",
    duration_sec: float = 12.0,
) -> SyntheticCameraImuFixture:
    """Write a Kalibr camchain plus separate fitting and evaluation recordings.

    The camchain carries the synthetic truth composed with the requested
    error, so a test can check that Calibrex accepts the truth and rejects a
    wrong candidate without Kalibr being installed. The data are nonphysical
    and make no accuracy claim about Kalibr.
    """

    if motion not in {"full", "yaw_only"}:
        msg = f"unsupported synthetic motion {motion!r}"
        raise ValueError(msg)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    truth = Rotation.from_rotvec(_TRUE_ROTVEC_CAM_IMU)
    error = Rotation.from_rotvec(np.radians(np.asarray(rotation_error_deg, dtype=float)))
    candidate = (truth * error).as_matrix()
    camchain = directory / "camchain-imucam.yaml"
    matrix = np.eye(4)
    matrix[:3, :3] = candidate
    matrix[:3, 3] = _TRUE_TRANSLATION_CAM_IMU
    write_mapping(
        camchain,
        {
            "cam0": {
                "camera_model": "pinhole",
                "intrinsics": [460.0, 460.0, 376.0, 240.0],
                "distortion_model": "radtan",
                "distortion_coeffs": [0.0, 0.0, 0.0, 0.0],
                "resolution": [752, 480],
                "rostopic": "/cam0/image_raw",
                "T_cam_imu": [[float(value) for value in row] for row in matrix],
                "timeshift_cam_imu": _TRUE_TIMESHIFT_CAM_IMU + timeshift_error_sec,
            }
        },
    )
    fitting = directory / "fitting-recording.json"
    evaluation = directory / "evaluation-recording.json"
    _synthetic_recording(seed, motion, duration_sec, "fitting").save(fitting)
    _synthetic_recording(seed + 1, motion, duration_sec, "evaluation").save(evaluation)
    return SyntheticCameraImuFixture(camchain, fitting, evaluation)


def _synthetic_recording(
    seed: int,
    motion: str,
    duration_sec: float,
    role: str,
) -> CameraImuMotionRecording:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * math.pi, 3)
    imu_from_cam = Rotation.from_rotvec(_TRUE_ROTVEC_CAM_IMU).inv()

    def world_imu(time_sec: float) -> Rotation:
        if motion == "yaw_only":
            return Rotation.from_rotvec([0.0, 0.0, 0.8 * math.sin(1.3 * time_sec + phases[2])])
        return Rotation.from_rotvec(
            [
                0.6 * math.sin(1.7 * time_sec + phases[0]),
                0.5 * math.sin(2.3 * time_sec + phases[1]),
                0.8 * math.sin(1.1 * time_sec + phases[2]),
            ]
        )

    epsilon = 1.0e-4
    gyro: list[ImuGyroSample] = []
    for true_time in np.arange(0.0, duration_sec, 1.0 / 200.0):
        rate = (
            world_imu(float(true_time)).inv() * world_imu(float(true_time) + epsilon)
        ).as_rotvec() / epsilon
        gyro.append(
            ImuGyroSample(
                timestamp_sec=float(true_time) + _TRUE_TIMESHIFT_CAM_IMU,
                omega_rad_s=[float(value) for value in rate + rng.normal(0.0, 0.004, 3)],
            )
        )
    cameras: list[CameraOrientationSample] = []
    for true_time in np.arange(0.05, duration_sec - 0.05, 1.0 / 20.0):
        noise = Rotation.from_rotvec(rng.normal(0.0, 0.0002, 3))
        world_cam = world_imu(float(true_time)) * imu_from_cam * noise
        cameras.append(
            CameraOrientationSample(
                timestamp_sec=float(true_time),
                rotation_quat_xyzw=[float(value) for value in world_cam.as_quat()],
            )
        )
    return CameraImuMotionRecording(
        recording_id=f"synthetic-camera-imu-{role}-{motion}-{seed}",
        camera_name="cam0",
        camera_orientations=cameras,
        imu_gyro=gyro,
        provenance=CameraImuMotionRecordingProvenance(
            producer=__name__,
            producer_version="0.1",
            source="deterministic synthetic Camera--IMU motion; nonphysical",
            git_commit=git_commit(),
            seed=seed,
            synthetic=True,
        ),
    )
