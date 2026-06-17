"""Factor plugin registry.

Plugins create backend-neutral factor descriptors. Solver adapters are
responsible for translating those descriptors into GTSAM, Ceres, scipy, Open3D,
or another backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar


@dataclass(frozen=True)
class FactorDescriptor:
    """Backend-neutral factor descriptor."""

    name: str
    variables: list[str]
    residual: str
    sensor_streams: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)


class FactorPlugin(ABC):
    """Base class for user-defined calibration factors."""

    required_streams: ClassVar[list[str]] = []

    @abstractmethod
    def build(self, context: dict[str, Any]) -> list[FactorDescriptor]:
        """Build backend-neutral factor descriptors."""


_FACTOR_PLUGINS: dict[str, type[FactorPlugin]] = {}
PluginType = TypeVar("PluginType", bound=type[FactorPlugin])


def register_factor(name: str) -> Callable[[PluginType], PluginType]:
    """Register a factor plugin class."""

    def decorator(plugin: PluginType) -> PluginType:
        if name in _FACTOR_PLUGINS:
            msg = f"factor plugin already registered: {name}"
            raise ValueError(msg)
        _FACTOR_PLUGINS[name] = plugin
        return plugin

    return decorator


def get_factor_plugin(name: str) -> type[FactorPlugin]:
    """Return a registered factor plugin by name."""

    try:
        return _FACTOR_PLUGINS[name]
    except KeyError as exc:
        msg = f"unknown factor plugin: {name}"
        raise KeyError(msg) from exc


def list_factor_plugins() -> list[str]:
    """List registered factor names."""

    return sorted(_FACTOR_PLUGINS)


@register_factor("camera_reprojection")
class CameraReprojectionFactor(FactorPlugin):
    """Built-in descriptor for camera reprojection factors."""

    required_streams: ClassVar[list[str]] = ["camera.image"]

    def build(self, context: dict[str, Any]) -> list[FactorDescriptor]:
        sensor = str(context.get("sensor", "camera0"))
        root = str(context.get("root", "base"))
        return [
            FactorDescriptor(
                name="camera_reprojection",
                variables=[f"T_{root}_{sensor}", f"intrinsics_{sensor}"],
                residual="reprojection_error_px",
                sensor_streams=self.required_streams,
            )
        ]


@register_factor("lidar_point_to_surfel")
class LidarPointToSurfelFactor(FactorPlugin):
    """Built-in descriptor for LiDAR surfel alignment factors."""

    required_streams: ClassVar[list[str]] = ["lidar.points"]

    def build(self, context: dict[str, Any]) -> list[FactorDescriptor]:
        sensor = str(context.get("sensor", "lidar0"))
        root = str(context.get("root", "base"))
        return [
            FactorDescriptor(
                name="lidar_point_to_surfel",
                variables=[f"T_{root}_{sensor}", "surfel_map"],
                residual="point_to_surfel_distance_m",
                sensor_streams=self.required_streams,
            )
        ]


@register_factor("lidar_rig_point_to_plane")
class LidarRigPointToPlaneDescriptorFactor(FactorPlugin):
    """Descriptor for native fixed-rig LiDAR point-to-plane factors."""

    required_streams: ClassVar[list[str]] = ["lidar.points", "ego.pose"]

    def build(self, context: dict[str, Any]) -> list[FactorDescriptor]:
        sensor = str(context.get("sensor", "lidar0"))
        root = str(context.get("root", "base"))
        options = {
            "trajectory_source": "dataset_ego_pose_or_prior",
            "correction_order": ["x", "y", "z", "roll", "pitch", "yaw"],
            "perturbation": "left",
            **dict(context.get("options", {})),
        }
        return [
            FactorDescriptor(
                name="lidar_rig_point_to_plane",
                variables=[f"T_{root}_{sensor}", "trajectory", "lidar_plane_map"],
                residual="point_to_plane_distance_m",
                sensor_streams=self.required_streams,
                options=options,
            )
        ]


@register_factor("radar_doppler_motion")
class RadarDopplerMotionFactor(FactorPlugin):
    """Built-in descriptor for future autonomous-driving Radar factors."""

    required_streams: ClassVar[list[str]] = ["radar.detections"]

    def build(self, context: dict[str, Any]) -> list[FactorDescriptor]:
        sensor = str(context.get("sensor", "radar0"))
        root = str(context.get("root", "base"))
        return [
            FactorDescriptor(
                name="radar_doppler_motion",
                variables=[f"T_{root}_{sensor}", "body_velocity"],
                residual="doppler_velocity_residual_mps",
                sensor_streams=self.required_streams,
            )
        ]


@register_factor("lidar_camera_mutual_information")
class LidarCameraMutualInformationFactor(FactorPlugin):
    """Built-in descriptor for targetless LiDAR-camera alignment factors."""

    required_streams: ClassVar[list[str]] = ["camera.image", "lidar.points"]

    def build(self, context: dict[str, Any]) -> list[FactorDescriptor]:
        root = str(context.get("root", "base"))
        camera = str(context.get("camera", "camera0"))
        lidar = str(context.get("lidar", "lidar0"))
        return [
            FactorDescriptor(
                name="lidar_camera_mutual_information",
                variables=[f"T_{root}_{camera}", f"T_{root}_{lidar}"],
                residual="mutual_information_alignment",
                sensor_streams=self.required_streams,
            )
        ]


@register_factor("fixed_lidar_mount_prior")
class FixedLidarMountPriorFactor(FactorPlugin):
    """Descriptor for fixed-mounted LiDAR extrinsic stability."""

    required_streams: ClassVar[list[str]] = ["lidar.points"]

    def build(self, context: dict[str, Any]) -> list[FactorDescriptor]:
        root = str(context.get("root", "base"))
        sensor = str(context.get("sensor", "lidar0"))
        options = {
            "assumption": "single rigid transform for fixed-mounted lidar",
            **dict(context.get("options", {})),
        }
        return [
            FactorDescriptor(
                name="fixed_lidar_mount_prior",
                variables=[f"T_{root}_{sensor}"],
                residual="fixed_lidar_extrinsic_stability",
                sensor_streams=self.required_streams,
                options=options,
            )
        ]


@register_factor("koide_lidar_camera")
class KoideLidarCameraFactor(FactorPlugin):
    """Descriptor for an external targetless LiDAR-camera baseline adapter."""

    required_streams: ClassVar[list[str]] = ["camera.image", "lidar.points"]

    def build(self, context: dict[str, Any]) -> list[FactorDescriptor]:
        root = str(context.get("root", "base"))
        camera = str(context.get("camera", "camera0"))
        lidar = str(context.get("lidar", "lidar0"))
        options = {
            "boundary": "external_adapter",
            **dict(context.get("options", {})),
        }
        return [
            FactorDescriptor(
                name="koide_lidar_camera",
                variables=[f"T_{root}_{camera}", f"T_{root}_{lidar}"],
                residual="external_targetless_lidar_camera_calibration",
                sensor_streams=self.required_streams,
                options=options,
            )
        ]


@register_factor("direct_visual_lidar_calibration")
class DirectVisualLidarCalibrationFactor(KoideLidarCameraFactor):
    """Alias for Koide's direct visual LiDAR-camera calibration adapter."""


@register_factor("lidar_camera_targetless_baseline")
class LidarCameraTargetlessBaselineFactor(KoideLidarCameraFactor):
    """Generic alias for targetless LiDAR-camera baseline adapters."""
