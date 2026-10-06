"""Decode raw Velodyne packet scans (``velodyne_msgs/msg/VelodyneScan``) into points.

Autoware and other ROS 2 drivers record the raw 1206-byte UDP data packets of a
Velodyne sensor next to (or instead of) the converted cloud. The packets carry
the per-firing time, which the transformed ``PointCloud2`` of a pipeline such as
Autoware's concatenated cloud does not, and they are stamped in the sensor's own
frame rather than in ``base_link``. This module converts a scan into a
:class:`~calibrex.data.ros_messages.PointCloud2Message` with per-point time
offsets, ROS-independent and written from the public VLP-16 and VLP-32C user
manuals (angles, units and firing sequences below are the manuals' values).

Supported: VLP-16 (product id 0x21 or 0x22) and VLP-32C (0x28). Other models raise
``DatasetError``. Point positions are in the sensor frame, in metres; no intrinsic
calibration file is read (the VLP-32C per-laser azimuth offsets are the manual's
constants).

Axes: the manual's convention is x right, y forward (azimuth clockwise from +y).
Autoware's drivers publish the sensor frame with x forward, y left; that is the
default here (``x_forward=True``) because it matches the ``velodyne_*`` frames of an
Autoware ``/tf_static``. Pass ``x_forward=False`` for the manual's axes.

Verification (Autoware ``all-sensors-bag1``, see the check tutorial): decoded
VLP-16 front points transformed with the bag's ``/tf_static`` reproduce the
concatenated ``base_link`` cloud of the same scan to a median of under 1 mm; for the
two VLP-32C scans the per-laser azimuth offsets and elevations fitted to that cloud
equal the constants here to 0.06 deg (a common bias, not per laser). Not verified:
dual-return packets (the return-mode byte is ignored, so both returns are emitted as
if one) and any model other than the two above; treat those as experimental.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr import CdrReader
from calibrex.data.ros_messages import MAX_ROS_STRING_BYTES, PointCloud2Message, PointField

VELODYNE_SCAN_TYPE = "velodyne_msgs/msg/VelodyneScan"
PACKET_SIZE = 1206
_BLOCK_SIZE = 100
_BLOCKS = 12
_MAX_PACKETS = 20000

_PRODUCT_VLP16 = {0x21, 0x22, 0x24}
_PRODUCT_VLP32C = {0x28}

_VLP16_ELEVATION_DEG = np.array(
    [-15.0, 1.0, -13.0, 3.0, -11.0, 5.0, -9.0, 7.0, -7.0, 9.0, -5.0, 11.0, -3.0, 13.0, -1.0, 15.0]
)
_VLP32C_ELEVATION_DEG = np.array(
    [
        -25.0, -1.0, -1.667, -15.639, -11.31, 0.0, -0.667, -8.843,
        -7.254, 0.333, -0.333, -6.148, -5.333, 1.333, 0.667, -4.0,
        -4.667, 1.667, 1.0, -3.667, -3.333, 3.333, 2.333, -2.667,
        -3.0, 7.0, 4.667, -2.333, -2.0, 15.0, 10.333, -1.333,
    ]
)  # fmt: skip
_VLP32C_AZIMUTH_OFFSET_DEG = np.array(
    [
        1.4, -4.2, 1.4, -1.4, 1.4, -1.4, 4.2, -1.4,
        1.4, -4.2, 1.4, -1.4, 4.2, -1.4, 4.2, -1.4,
        1.4, -4.2, 1.4, -4.2, 4.2, -1.4, 1.4, -1.4,
        1.4, -1.4, 1.4, -4.2, 4.2, -1.4, 1.4, -1.4,
    ]
)  # fmt: skip

POINT_FIELDS: tuple[PointField, ...] = (
    PointField(name="x", offset=0, datatype=7, count=1),
    PointField(name="y", offset=4, datatype=7, count=1),
    PointField(name="z", offset=8, datatype=7, count=1),
    PointField(name="intensity", offset=12, datatype=7, count=1),
    PointField(name="time", offset=16, datatype=7, count=1),
)


@dataclass(frozen=True)
class VelodynePacketScan:
    """The packets of one scan: header stamp, frame id and the raw packet payloads."""

    timestamp_ns: int
    frame_id: str
    packet_stamps_ns: tuple[int, ...]
    packets: tuple[bytes, ...]


def decode_velodyne_scan_packets(data: bytes) -> VelodynePacketScan:
    """Parse a CDR ``velodyne_msgs/msg/VelodyneScan`` into its raw packets."""

    reader = CdrReader(data)
    sec = reader.read_int32()
    nsec = reader.read_uint32()
    frame_id = reader.read_string(max_length=MAX_ROS_STRING_BYTES)
    count = reader.read_uint32()
    if count > _MAX_PACKETS:
        raise DatasetError(f"VelodyneScan claims {count} packets")
    little = reader.little_endian
    position = reader.offset
    stamps: list[int] = []
    packets: list[bytes] = []
    for _ in range(count):
        position += (-(position - 4)) % 4
        if position + 8 + PACKET_SIZE > len(data):
            raise DatasetError("truncated VelodyneScan payload")
        p_sec = int.from_bytes(
            data[position : position + 4], "little" if little else "big", signed=True
        )
        p_nsec = int.from_bytes(data[position + 4 : position + 8], "little" if little else "big")
        stamps.append(p_sec * 1_000_000_000 + p_nsec)
        position += 8
        packets.append(bytes(data[position : position + PACKET_SIZE]))
        position += PACKET_SIZE
    return VelodynePacketScan(
        timestamp_ns=sec * 1_000_000_000 + nsec,
        frame_id=frame_id,
        packet_stamps_ns=tuple(stamps),
        packets=tuple(packets),
    )


def velodyne_model_of_packet(packet: bytes) -> str:
    """Return ``"vlp16"`` or ``"vlp32c"`` from the packet's product-id byte."""

    if len(packet) != PACKET_SIZE:
        raise DatasetError(f"a Velodyne data packet is {PACKET_SIZE} bytes, got {len(packet)}")
    product = packet[PACKET_SIZE - 1]
    if product in _PRODUCT_VLP16:
        return "vlp16"
    if product in _PRODUCT_VLP32C:
        return "vlp32c"
    raise DatasetError(f"unsupported Velodyne product id 0x{product:02x}")


def decode_velodyne_scan(
    topic: str,
    timestamp_ns: int,
    data: bytes,
    *,
    min_range_m: float = 0.4,
    max_range_m: float = 200.0,
    x_forward: bool = True,
) -> PointCloud2Message:
    """Decode a ``VelodyneScan`` into a cloud in the sensor frame with per-point time.

    The ``time`` offsets (seconds) are relative to the scan's first firing; the
    header stamp is the scan's first packet stamp, so the offset is the time
    since the header. Returns an empty cloud for a scan with no packets.
    """

    scan = decode_velodyne_scan_packets(data)
    xyz_parts: list[np.ndarray] = []
    intensity_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    first_stamp = scan.packet_stamps_ns[0] if scan.packet_stamps_ns else scan.timestamp_ns
    for stamp, packet in zip(scan.packet_stamps_ns, scan.packets, strict=True):
        model = velodyne_model_of_packet(packet)
        xyz, intensity, offset_s = (
            _decode_vlp16_packet(packet) if model == "vlp16" else _decode_vlp32c_packet(packet)
        )
        valid = (intensity[:, 1] >= min_range_m) & (intensity[:, 1] <= max_range_m)
        xyz_parts.append(xyz[valid])
        intensity_parts.append(intensity[valid, 0])
        time_parts.append(offset_s[valid] + (stamp - first_stamp) * 1.0e-9)
    if xyz_parts:
        xyz_all = np.concatenate(xyz_parts)
        if x_forward:  # manual axes (x right, y forward) -> x forward, y left
            xyz_all = np.stack([xyz_all[:, 1], -xyz_all[:, 0], xyz_all[:, 2]], axis=1)
        intensity_all: np.ndarray | None = np.concatenate(intensity_parts)
        time_all: np.ndarray | None = np.concatenate(time_parts)
    else:
        xyz_all, intensity_all, time_all = np.zeros((0, 3)), None, None
    return PointCloud2Message(
        topic=topic,
        timestamp_ns=first_stamp if scan.packet_stamps_ns else timestamp_ns,
        frame_id=scan.frame_id,
        width=int(xyz_all.shape[0]),
        height=1,
        point_step=20,
        fields=POINT_FIELDS,
        xyz=xyz_all.astype(np.float64),
        intensity=None if intensity_all is None else intensity_all.astype(np.float64),
        point_time_offsets_s=time_all,
        raw_point_count=int(xyz_all.shape[0]),
    )


def _spherical_to_xyz(
    range_m: np.ndarray, azimuth_deg: np.ndarray, elevation_deg: np.ndarray
) -> np.ndarray:
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    horizontal = range_m * np.cos(elevation)
    return np.stack(
        [horizontal * np.sin(azimuth), horizontal * np.cos(azimuth), range_m * np.sin(elevation)],
        axis=1,
    )


def _blocks(packet: bytes) -> np.ndarray:
    raw = np.frombuffer(packet, dtype=np.uint8, count=_BLOCKS * _BLOCK_SIZE).reshape(
        _BLOCKS, _BLOCK_SIZE
    )
    if not np.all(raw[:, 0] == 0xFF) or not np.all(raw[:, 1] == 0xEE):
        raise DatasetError("Velodyne packet block flag is not 0xEEFF")
    return raw


def _block_azimuth_deg(raw: np.ndarray) -> np.ndarray:
    return (raw[:, 2].astype(np.float64) + 256.0 * raw[:, 3]) * 0.01


def _decode_vlp16_packet(packet: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VLP-16: 12 blocks x 2 firing sequences x 16 channels, 2 mm units, 55.296 us per sequence."""

    raw = _blocks(packet)
    azimuth = _block_azimuth_deg(raw)
    next_azimuth = np.empty_like(azimuth)
    next_azimuth[:-1] = azimuth[1:]
    step = (next_azimuth[:-1] - azimuth[:-1]) % 360.0
    next_azimuth[-1] = azimuth[-1] + step[-1]
    block_step = (next_azimuth - azimuth) % 360.0
    channels = raw[:, 4:].reshape(_BLOCKS, 2, 16, 3).astype(np.float64)
    range_m = (channels[..., 0] + 256.0 * channels[..., 1]) * 0.002
    reflectivity = channels[..., 2]
    sequence_fraction = np.array([0.0, 0.5])[None, :, None]
    azimuth_deg = (azimuth[:, None, None] + block_step[:, None, None] * sequence_fraction) + (
        np.zeros((1, 1, 16))
    )
    # Within a sequence the head turns by 16 firings of 2.304 us of a 55.296 us cycle.
    azimuth_deg = azimuth_deg + (
        block_step[:, None, None] * 0.5 * (np.arange(16)[None, None, :] * 2.304 / 55.296)
    )
    elevation = np.broadcast_to(_VLP16_ELEVATION_DEG[None, None, :], range_m.shape)
    time_us = (
        np.arange(_BLOCKS)[:, None, None] * 110.592
        + np.arange(2)[None, :, None] * 55.296
        + np.arange(16)[None, None, :] * 2.304
    )
    flat_range = range_m.reshape(-1)
    xyz = _spherical_to_xyz(flat_range, azimuth_deg.reshape(-1), elevation.reshape(-1))
    info = np.stack([reflectivity.reshape(-1), flat_range], axis=1)
    return xyz, info, (time_us * 1.0e-6).reshape(-1)


def _decode_vlp32c_packet(packet: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VLP-32C: 12 blocks x 32 channels, 4 mm units, 55.296 us per block."""

    raw = _blocks(packet)
    azimuth = _block_azimuth_deg(raw)
    channels = raw[:, 4:].reshape(_BLOCKS, 32, 3).astype(np.float64)
    range_m = (channels[..., 0] + 256.0 * channels[..., 1]) * 0.004
    reflectivity = channels[..., 2]
    azimuth_deg = azimuth[:, None] + _VLP32C_AZIMUTH_OFFSET_DEG[None, :]
    elevation = np.broadcast_to(_VLP32C_ELEVATION_DEG[None, :], range_m.shape)
    time_us = np.arange(_BLOCKS)[:, None] * 55.296 + (np.arange(32)[None, :] // 2) * 2.304
    flat_range = range_m.reshape(-1)
    xyz = _spherical_to_xyz(flat_range, azimuth_deg.reshape(-1), elevation.reshape(-1))
    info = np.stack([reflectivity.reshape(-1), flat_range], axis=1)
    return xyz, info, (time_us * 1.0e-6).reshape(-1)
