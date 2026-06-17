"""Compile Calibrex configs into backend-neutral graph problems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from calibrex.core.config import CalibrationConfig, FactorConfig
from calibrex.core.frames import FrameGraph
from calibrex.data.inspect import DatasetInspection
from calibrex.graph.factors import FactorDescriptor, get_factor_plugin
from calibrex.graph.gauge import GaugeConstraint
from calibrex.graph.observability import ObservabilityDiagnostic
from calibrex.graph.priors import PriorDescriptor
from calibrex.graph.variables import VariableDescriptor


@dataclass(frozen=True)
class CalibrationProblemSpec:
    """A backend-neutral calibration problem."""

    name: str
    pipeline: str
    variables: list[VariableDescriptor] = field(default_factory=list)
    factors: list[FactorDescriptor] = field(default_factory=list)
    priors: list[PriorDescriptor] = field(default_factory=list)
    gauges: list[GaugeConstraint] = field(default_factory=list)
    observability: ObservabilityDiagnostic | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable problem representation."""

        return {
            "name": self.name,
            "pipeline": self.pipeline,
            "variables": [variable.as_dict() for variable in self.variables],
            "factors": [_factor_as_dict(factor) for factor in self.factors],
            "priors": [prior.as_dict() for prior in self.priors],
            "gauges": [gauge.as_dict() for gauge in self.gauges],
            "observability": self.observability.as_dict() if self.observability else None,
            "warnings": list(self.warnings),
        }

    def summary(self) -> dict[str, int | str]:
        """Return a compact problem summary."""

        return {
            "name": self.name,
            "pipeline": self.pipeline,
            "variables": len(self.variables),
            "estimated_variables": sum(1 for variable in self.variables if variable.estimate),
            "factors": len(self.factors),
            "priors": len(self.priors),
            "gauges": len(self.gauges),
        }


def build_problem(
    config: CalibrationConfig,
    frame_graph: FrameGraph,
    inspection: DatasetInspection,
) -> CalibrationProblemSpec:
    """Compile a config into a backend-neutral problem spec."""

    del inspection
    variables = _build_variables(config, frame_graph)
    factors, warnings = _build_factors(config, frame_graph)
    priors = _build_priors(config, frame_graph)
    gauges = _build_gauges(frame_graph)
    observability = _diagnose_observability(variables, factors, priors, gauges)
    return CalibrationProblemSpec(
        name=config.project.name,
        pipeline=config.pipeline.type,
        variables=variables,
        factors=factors,
        priors=priors,
        gauges=gauges,
        observability=observability,
        warnings=warnings,
    )


def _build_variables(
    config: CalibrationConfig,
    frame_graph: FrameGraph,
) -> list[VariableDescriptor]:
    variables: list[VariableDescriptor] = [
        VariableDescriptor(
            name="trajectory",
            kind="trajectory",
            dimension=6,
            estimate=_uses_motion_variables(config),
            frame=frame_graph.root,
            initial={"representation": "pose_spline"},
        )
    ]
    for name, node in sorted(frame_graph.nodes.items()):
        if node.parent is None:
            continue
        variables.append(
            VariableDescriptor(
                name=f"T_{node.parent}_{name}",
                kind="extrinsic",
                dimension=6,
                estimate=node.estimate,
                frame=name,
                sensor=name if name in config.sensors else None,
                initial={
                    "translation_m": list(node.transform_to_parent.translation_m),
                    "rotation_quat_xyzw": list(node.transform_to_parent.rotation_quat_xyzw),
                },
            )
        )

    for sensor_name, sensor in sorted(config.sensors.items()):
        if sensor.type == "camera" and sensor.intrinsics is not None:
            variables.append(
                VariableDescriptor(
                    name=f"intrinsics_{sensor_name}",
                    kind="intrinsic",
                    dimension=4 + len(sensor.intrinsics.distortion),
                    estimate=sensor.intrinsics.estimate,
                    sensor=sensor_name,
                    initial=sensor.intrinsics.model_dump(mode="json"),
                )
            )
        if sensor.type == "imu" and _factor_enabled(config, "imu_preintegration"):
            variables.append(
                VariableDescriptor(
                    name=f"imu_bias_{sensor_name}",
                    kind="imu_bias",
                    dimension=6,
                    estimate=True,
                    sensor=sensor_name,
                    initial={"gyro_bias": [0.0, 0.0, 0.0], "accel_bias": [0.0, 0.0, 0.0]},
                )
            )

    for sensor_name, offset in sorted(config.time_offsets.items()):
        variables.append(
            VariableDescriptor(
                name=f"dt_{sensor_name}",
                kind="time_offset",
                dimension=1,
                estimate=offset.estimate,
                sensor=sensor_name,
                initial={"seconds": offset.initial_sec},
            )
        )

    if _factor_enabled(config, "lidar_point_to_surfel"):
        variables.append(VariableDescriptor(name="surfel_map", kind="map", dimension=3))
    if _factor_enabled(config, "lidar_rig_point_to_plane"):
        variables.append(VariableDescriptor(name="lidar_plane_map", kind="map", dimension=3))
    if _factor_enabled(config, "radar_doppler_motion"):
        variables.append(VariableDescriptor(name="body_velocity", kind="velocity", dimension=3))
    if config.pipeline.type == "rgbd_open3d_slac":
        variables.append(
            VariableDescriptor(
                name="open3d_control_grid",
                kind="control_grid",
                dimension=3,
                initial={"backend": "open3d_slac"},
            )
        )
    return variables


def _build_factors(
    config: CalibrationConfig,
    frame_graph: FrameGraph,
) -> tuple[list[FactorDescriptor], list[str]]:
    factors: list[FactorDescriptor] = []
    warnings: list[str] = []
    for factor_name, factor_config in sorted(config.pipeline.factors.items()):
        if not factor_config.enabled:
            continue
        contexts = _factor_contexts(config, frame_graph.root, factor_name)
        if not contexts:
            factors.append(_fallback_factor(factor_name, factor_config))
            warnings.append(
                f"factor '{factor_name}' has no typed builder; using fallback descriptor"
            )
            continue
        try:
            plugin_type = get_factor_plugin(factor_name)
        except KeyError:
            for context in contexts:
                factors.append(_special_factor(factor_name, context, factor_config))
            continue
        for context in contexts:
            context["options"] = factor_config.options
            factors.extend(plugin_type().build(context))
    return factors, warnings


def _build_priors(
    config: CalibrationConfig,
    frame_graph: FrameGraph,
) -> list[PriorDescriptor]:
    priors: list[PriorDescriptor] = []
    for name, frame in sorted(config.frames.items()):
        if frame.root or frame.parent is None or frame.transform is None:
            continue
        variable = f"T_{frame.parent}_{name}"
        if frame.transform.prior_sigma is not None:
            sigma: dict[str, float] = {}
            if frame.transform.prior_sigma.translation_m is not None:
                sigma["translation_m"] = frame.transform.prior_sigma.translation_m
            if frame.transform.prior_sigma.rotation_deg is not None:
                sigma["rotation_deg"] = frame.transform.prior_sigma.rotation_deg
            priors.append(PriorDescriptor(f"prior_{variable}", variable, "se3_prior", sigma=sigma))
    for sensor_name, offset in sorted(config.time_offsets.items()):
        if offset.prior_sigma_sec is not None:
            priors.append(
                PriorDescriptor(
                    name=f"prior_dt_{sensor_name}",
                    variable=f"dt_{sensor_name}",
                    prior_type="time_offset_prior",
                    sigma={"seconds": offset.prior_sigma_sec},
                    value={"seconds": offset.initial_sec},
                )
            )
    if _uses_motion_variables(config):
        priors.append(
            PriorDescriptor(
                name=f"prior_pose_{frame_graph.root}",
                variable="trajectory",
                prior_type="root_pose_prior",
                sigma={"translation_m": 0.0, "rotation_deg": 0.0},
            )
        )
    return priors


def _build_gauges(frame_graph: FrameGraph) -> list[GaugeConstraint]:
    return [
        GaugeConstraint(
            name=f"fix_{frame_graph.root}",
            variable="trajectory",
            constraint_type="fix_root_pose",
            reason="remove global pose gauge freedom",
        )
    ]


def _diagnose_observability(
    variables: list[VariableDescriptor],
    factors: list[FactorDescriptor],
    priors: list[PriorDescriptor],
    gauges: list[GaugeConstraint],
) -> ObservabilityDiagnostic:
    estimated_dimension = sum(variable.dimension for variable in variables if variable.estimate)
    fixed_dimension = sum(variable.dimension for variable in variables if not variable.estimate)
    weak_directions: list[str] = []
    if not any(variable.kind == "trajectory" and variable.estimate for variable in variables):
        weak_directions.append("trajectory_fixed_or_missing")
    estimated_extrinsics = [
        variable for variable in variables if variable.kind == "extrinsic" and variable.estimate
    ]
    if estimated_extrinsics and not factors:
        weak_directions.append("estimated_extrinsics_without_factors")
    if not priors:
        weak_directions.append("no_priors")
    return ObservabilityDiagnostic(
        estimated_dimension=estimated_dimension,
        fixed_dimension=fixed_dimension,
        factor_count=len(factors),
        prior_count=len(priors),
        gauge_count=len(gauges),
        weak_directions=weak_directions,
    )


def _factor_contexts(
    config: CalibrationConfig,
    root: str,
    factor_name: str,
) -> list[dict[str, Any]]:
    if factor_name == "camera_reprojection":
        return [
            {"root": root, "sensor": name}
            for name, sensor in sorted(config.sensors.items())
            if sensor.type == "camera"
        ]
    if factor_name == "lidar_point_to_surfel":
        return [
            {"root": root, "sensor": name}
            for name, sensor in sorted(config.sensors.items())
            if sensor.type == "lidar"
        ]
    if factor_name == "lidar_rig_point_to_plane":
        return [
            {"root": root, "sensor": name}
            for name, sensor in sorted(config.sensors.items())
            if sensor.type == "lidar"
        ]
    if factor_name == "radar_doppler_motion":
        return [
            {"root": root, "sensor": name}
            for name, sensor in sorted(config.sensors.items())
            if sensor.type == "radar"
        ]
    if factor_name == "fixed_lidar_mount_prior":
        return [
            {"root": root, "sensor": name}
            for name, sensor in sorted(config.sensors.items())
            if sensor.type == "lidar"
        ]
    if factor_name in {
        "lidar_camera_mutual_information",
        "koide_lidar_camera",
        "direct_visual_lidar_calibration",
        "lidar_camera_targetless_baseline",
    }:
        cameras = [
            name for name, sensor in sorted(config.sensors.items()) if sensor.type == "camera"
        ]
        lidars = [
            name for name, sensor in sorted(config.sensors.items()) if sensor.type == "lidar"
        ]
        return [
            {"root": root, "camera": camera, "lidar": lidar}
            for camera in cameras
            for lidar in lidars
        ]
    if factor_name in {"imu_preintegration", "open3d_slac"}:
        if factor_name == "imu_preintegration":
            return [
                {"root": root, "sensor": name}
                for name, sensor in sorted(config.sensors.items())
                if sensor.type == "imu"
            ]
        return [{"root": root}]
    return []


def _special_factor(
    factor_name: str,
    context: dict[str, Any],
    factor_config: FactorConfig,
) -> FactorDescriptor:
    root = str(context.get("root", "base"))
    if factor_name == "imu_preintegration":
        return FactorDescriptor(
            name=factor_name,
            variables=["trajectory", *_imu_bias_variables(context)],
            residual="imu_preintegration_residual",
            sensor_streams=["imu.measurements"],
            options=factor_config.options,
        )
    if factor_name == "open3d_slac":
        return FactorDescriptor(
            name=factor_name,
            variables=["trajectory", "open3d_control_grid"],
            residual="open3d_slac_residual",
            sensor_streams=["rgbd.fragments", "pose_graph"],
            options=factor_config.options,
        )
    return FactorDescriptor(
        name=factor_name,
        variables=[f"T_{root}_{factor_name}"],
        residual=f"{factor_name}_residual",
        options=factor_config.options,
    )


def _fallback_factor(factor_name: str, factor_config: FactorConfig) -> FactorDescriptor:
    return FactorDescriptor(
        name=factor_name,
        variables=[],
        residual=f"{factor_name}_residual",
        options=factor_config.options,
    )


def _factor_enabled(config: CalibrationConfig, factor_name: str) -> bool:
    factor = config.pipeline.factors.get(factor_name)
    return bool(factor and factor.enabled)


def _uses_motion_variables(config: CalibrationConfig) -> bool:
    motion_factors = {
        "imu_preintegration",
        "lidar_point_to_surfel",
        "lidar_rig_point_to_plane",
        "open3d_slac",
    }
    return config.pipeline.type.endswith("slac") or any(
        _factor_enabled(config, factor_name) for factor_name in motion_factors
    )


def _imu_bias_variables(context: dict[str, Any]) -> list[str]:
    sensor = context.get("sensor")
    if isinstance(sensor, str):
        return [f"imu_bias_{sensor}"]
    return []


def _factor_as_dict(factor: FactorDescriptor) -> dict[str, Any]:
    return {
        "name": factor.name,
        "variables": list(factor.variables),
        "residual": factor.residual,
        "sensor_streams": list(factor.sensor_streams),
        "options": dict(factor.options),
    }
