"""Sensor declarations used by the problem compiler."""

from __future__ import annotations

from dataclasses import dataclass

from calibrex.core.config import CalibrationConfig, SensorType


@dataclass(frozen=True)
class SensorDefinition:
    """A normalized sensor declaration independent of ROS message types."""

    name: str
    type: SensorType
    model: str
    topic: str | None
    frame: str


def build_sensor_registry(config: CalibrationConfig) -> dict[str, SensorDefinition]:
    """Build a normalized sensor registry from config."""

    registry: dict[str, SensorDefinition] = {}
    for name, sensor in sorted(config.sensors.items()):
        registry[name] = SensorDefinition(
            name=name,
            type=sensor.type,
            model=sensor.model,
            topic=sensor.topic,
            frame=sensor.frame_id or name,
        )
    return registry
