"""Built-in Livox profile fields without array or bag-reader dependencies."""

from typing import Literal, TypeAlias

_ProfileFields: TypeAlias = tuple[
    str, str, str | None, Literal["offset_s", "absolute_ns", "absolute_s"], Literal["mps2", "g"]
]
LIVOX_PROFILE_FIELDS: dict[str, _ProfileFields] = {
    "rtk-slam": ("/livox/points", "/livox/imu", "offset_time", "offset_s", "mps2"),
    "livox-ros-driver2": ("/livox/lidar", "/livox/imu", "timestamp", "absolute_ns", "g"),
}
