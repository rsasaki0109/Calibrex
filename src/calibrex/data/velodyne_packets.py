"""Decode raw Velodyne packet scans (``velodyne_msgs/msg/VelodyneScan``) into points.

Autoware and other ROS 2 drivers record the raw 1206-byte UDP data packets of a
Velodyne sensor next to (or instead of) the converted cloud. The packets carry
the per-firing time, which the transformed ``PointCloud2`` of a pipeline such as
Autoware's concatenated cloud does not, and they are stamped in the sensor's own
frame rather than in ``base_link``. This module converts a scan into a
:class:`~calibrex.data.ros_messages.PointCloud2Message` with per-point time
offsets, ROS-independent and written from the public VLP-16, VLP-32C, HDL-64E
and VLS-128 user manuals (angles, units, firing sequences and return-mode layouts
below are the manuals' values).

Supported, each verified against a reference cloud (see below):

* VLP-16 (product id 0x22), single (strongest/last) and dual return;
* VLP-32C (product id 0x28), single return;
* HDL-64E (no product id; recognised by its alternating upper/lower blocks),
  single and dual return, with the sensor's own calibration
  (:func:`load_velodyne_calibration`: Velodyne ``db.xml`` or ROS
  ``velodyne_pointcloud`` YAML);
* VLS-128 / Alpha Prime (product id 0xA1), single (strongest/last) return, with the
  sensor's 128-laser calibration in the same formats.

Every other packet format raises ``DatasetError`` naming it: HDL-32E (0x21),
VLP-32A/B (0x23), VLP-16 Hi-Res (0x24), Velarray (0x31), VLP-32C and VLS-128 dual
return, an HDL-64E or VLS-128 without its calibration, and unknown ids. Earlier versions
decoded 0x21 and 0x24 as a VLP-16; their elevations differ, so they are refused.

Point positions are in the sensor frame, in metres. In dual-return mode both
returns are emitted, except a second return identical to the first (same raw
distance and intensity), which is dropped. HDL-64E positions follow the
correction model of Velodyne's LidarView (rotational, vertical, distance,
horizontal and vertical offset corrections); the manual's two-point distance
interpolation (``distCorrectionX/Y``) that the ROS driver applies is not, and
differs by a few centimetres.

Axes: the manual's convention is x right, y forward (azimuth clockwise from +y).
Autoware's drivers publish the sensor frame with x forward, y left; that is the
default here (``x_forward=True``) because it matches the ``velodyne_*`` frames of an
Autoware ``/tf_static``. Pass ``x_forward=False`` for the manual's axes.

Verification:

* Autoware ``all-sensors-bag1`` (see the check tutorial): decoded VLP-16 front
  points transformed with the bag's ``/tf_static`` reproduce the concatenated
  ``base_link`` cloud of the same scan to a median of under 1 mm; for the two
  VLP-32C scans the per-laser azimuth offsets and elevations fitted to that cloud
  equal the constants here to 0.06 deg.
* LidarView's published test recordings (Apache-2.0; pcap plus the clouds
  LidarView decoded from them), compared point to point with
  ``tools/validate_velodyne_decode.py --pcap``; the medians are in
  ``docs/tutorials/calibrex_check.md``.
"""

from __future__ import annotations

import hashlib
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from calibrex.core.exceptions import DatasetError
from calibrex.data.ros_cdr import CdrReader
from calibrex.data.ros_messages import MAX_ROS_STRING_BYTES, PointCloud2Message, PointField

VELODYNE_SCAN_TYPE = "velodyne_msgs/msg/VelodyneScan"
PACKET_SIZE = 1206
_BLOCK_SIZE = 100
_BLOCKS = 12
_MAX_PACKETS = 20000
_UPPER_BLOCK = 0xEEFF
_LOWER_BLOCK = 0xDDFF
# VLS-128: the four blocks of a firing sequence carry lasers 0-31, 32-63, 64-95, 96-127.
_VLS128_BANKS = {0xEEFF: 0, 0xDDFF: 32, 0xCCFF: 64, 0xBBFF: 96}

VelodyneModel = Literal["vlp16", "vlp32c", "hdl64e", "vls128"]
VelodyneReturnMode = Literal["strongest", "last", "single", "dual", "unknown"]

_PRODUCT_NAMES = {
    0x21: "HDL-32E",
    0x22: "VLP-16",
    0x23: "VLP-32A/B",
    0x24: "VLP-16 Hi-Res",
    0x28: "VLP-32C",
    0x31: "Velarray",
    0xA1: "VLS-128",
}
_RETURN_MODES: dict[int, VelodyneReturnMode] = {0x37: "strongest", 0x38: "last", 0x39: "dual"}
_SUPPORTED: dict[tuple[str, str], VelodyneModel] = {
    ("VLP-16", "strongest"): "vlp16",
    ("VLP-16", "last"): "vlp16",
    ("VLP-16", "dual"): "vlp16",
    ("VLP-32C", "strongest"): "vlp32c",
    ("VLP-32C", "last"): "vlp32c",
    ("HDL-64E", "single"): "hdl64e",
    ("HDL-64E", "dual"): "hdl64e",
    ("VLS-128", "strongest"): "vls128",
    ("VLS-128", "last"): "vls128",
}
# Models whose lasers are not fixed constants here: the unit's calibration is required.
_CALIBRATED_LASERS: dict[VelodyneModel, int] = {"hdl64e": 64, "vls128": 128}

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
class VelodynePacketFormat:
    """What a data packet is: sensor, return mode, and whether it is decoded here."""

    sensor: str
    return_mode: VelodyneReturnMode
    product_id: int | None
    model: VelodyneModel | None

    @property
    def supported(self) -> bool:
        return self.model is not None

    @property
    def description(self) -> str:
        product = "" if self.product_id is None else f", product id 0x{self.product_id:02x}"
        return f"{self.sensor} {self.return_mode} return{product}"

    @property
    def reason(self) -> str | None:
        """Why the format is not decoded, or ``None`` when it is."""

        if self.supported:
            return None
        return (
            f"unsupported Velodyne packet format ({self.description}): not verified "
            "against a reference cloud"
        )


@dataclass(frozen=True)
class VelodyneLaserCorrection:
    """One laser's calibration, in the units of the decoder (degrees and metres)."""

    laser_id: int
    rot_correction_deg: float
    vert_correction_deg: float
    dist_correction_m: float
    vert_offset_m: float
    horiz_offset_m: float


@dataclass(frozen=True)
class VelodyneCalibration:
    """A per-unit Velodyne calibration (HDL-64E) and where it came from."""

    lasers: tuple[VelodyneLaserCorrection, ...]
    distance_resolution_m: float
    source: str
    sha256: str

    def laser_table(self) -> dict[str, np.ndarray]:
        ordered = sorted(self.lasers, key=lambda item: item.laser_id)
        return {
            name: np.asarray([getattr(item, name) for item in ordered], dtype=np.float64)
            for name in (
                "rot_correction_deg",
                "vert_correction_deg",
                "dist_correction_m",
                "vert_offset_m",
                "horiz_offset_m",
            )
        }


@dataclass(frozen=True)
class VelodynePacketPoints:
    """Every return of one data packet, in the manual's axes (x right, y forward).

    ``time_s`` is the firing time relative to the packet's first firing;
    ``laser`` is the laser (ring) index; ``second_return`` marks the second
    return of a dual-return firing.
    """

    xyz: np.ndarray
    reflectivity: np.ndarray
    range_m: np.ndarray
    time_s: np.ndarray
    laser: np.ndarray
    second_return: np.ndarray


def load_velodyne_calibration(path: str | Path) -> VelodyneCalibration:
    """Read a Velodyne ``db.xml`` (as shipped with the sensor or LidarView) or ROS YAML.

    ``db.xml`` holds degrees and centimetres (``distLSB_`` in centimetres); the
    ROS ``velodyne_pointcloud`` YAML holds radians and metres. Only the fields
    the decoder uses are read.
    """

    source = Path(path)
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    try:
        if source.suffix.lower() == ".xml":
            lasers, resolution = _db_xml_lasers(data)
        else:
            lasers, resolution = _ros_yaml_lasers(yaml.safe_load(data))
    except (ET.ParseError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        raise DatasetError(f"cannot read the Velodyne calibration {source}: {error}") from error
    ids = sorted(item.laser_id for item in lasers)
    if ids != list(range(len(ids))) or len(ids) not in {16, 32, 64, 128}:
        raise DatasetError(
            f"{source}: expected laser ids 0..N-1 for 16, 32, 64 or 128 lasers, got {len(ids)}"
        )
    if not 0.0 < resolution < 0.1:
        raise DatasetError(f"{source}: implausible distance resolution {resolution} m")
    return VelodyneCalibration(tuple(lasers), resolution, str(source), digest)


def _db_xml_lasers(data: bytes) -> tuple[list[VelodyneLaserCorrection], float]:
    # LidarView's Alpha_Prime.xml repeats the XML declaration after its DOCTYPE, which
    # boost reads and ElementTree refuses; the declarations carry nothing needed here.
    root = ET.fromstring(re.sub(rb"<\?xml[^>]*\?>", b"", data))
    lsb = root.find(".//distLSB_")
    resolution = 0.002 if lsb is None or lsb.text is None else float(lsb.text) / 100.0
    lasers: list[VelodyneLaserCorrection] = []
    for px in root.iter("px"):
        values = {child.tag.rstrip("_"): float(child.text or "nan") for child in px}
        lasers.append(
            VelodyneLaserCorrection(
                laser_id=int(values["id"]),
                rot_correction_deg=values["rotCorrection"],
                vert_correction_deg=values["vertCorrection"],
                dist_correction_m=values["distCorrection"] / 100.0,
                vert_offset_m=values.get("vertOffsetCorrection", 0.0) / 100.0,
                # LidarView's VLS-128 file (Alpha_Prime.xml) spells it "Corection".
                horiz_offset_m=values.get(
                    "horizOffsetCorrection", values.get("horizOffsetCorection", 0.0)
                )
                / 100.0,
            )
        )
    return lasers, resolution


def _ros_yaml_lasers(payload: Any) -> tuple[list[VelodyneLaserCorrection], float]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("lasers"), list):
        raise ValueError("expected a mapping with a 'lasers' list")
    lasers = [
        VelodyneLaserCorrection(
            laser_id=int(item["laser_id"]),
            rot_correction_deg=math.degrees(float(item["rot_correction"])),
            vert_correction_deg=math.degrees(float(item["vert_correction"])),
            dist_correction_m=float(item.get("dist_correction", 0.0)),
            vert_offset_m=float(item.get("vert_offset_correction", 0.0)),
            horiz_offset_m=float(item.get("horiz_offset_correction", 0.0)),
        )
        for item in payload["lasers"]
    ]
    return lasers, float(payload.get("distance_resolution", 0.002))


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


def velodyne_packet_format(packet: bytes) -> VelodynePacketFormat:
    """Identify a data packet's sensor and return mode, decoded here or not."""

    if len(packet) != PACKET_SIZE:
        raise DatasetError(f"a Velodyne data packet is {PACKET_SIZE} bytes, got {len(packet)}")
    flags = [
        packet[index * _BLOCK_SIZE] | packet[index * _BLOCK_SIZE + 1] << 8 for index in range(3)
    ]
    if flags[1] == _LOWER_BLOCK and flags[2] == _UPPER_BLOCK:
        dual = packet[2:4] == packet[2 * _BLOCK_SIZE + 2 : 2 * _BLOCK_SIZE + 4]
        # The HDL-64E reports strongest/last only in its status cycle; "single" here.
        mode: VelodyneReturnMode = "dual" if dual else "single"
        return VelodynePacketFormat("HDL-64E", mode, None, _SUPPORTED.get(("HDL-64E", mode)))
    product = packet[PACKET_SIZE - 1]
    name = _PRODUCT_NAMES.get(product, "unknown Velodyne sensor")
    mode = _RETURN_MODES.get(packet[PACKET_SIZE - 2], "unknown")
    return VelodynePacketFormat(name, mode, product, _SUPPORTED.get((name, mode)))


def velodyne_model_of_packet(packet: bytes) -> VelodyneModel:
    """Return the decoder model of a packet; raise ``DatasetError`` for other formats."""

    packet_format = velodyne_packet_format(packet)
    if packet_format.model is None:
        raise DatasetError(packet_format.reason or "unsupported Velodyne packet")
    return packet_format.model


def decode_velodyne_packet(
    packet: bytes, calibration: VelodyneCalibration | None = None
) -> VelodynePacketPoints:
    """Decode every non-empty return of one data packet (manual axes)."""

    packet_format = velodyne_packet_format(packet)
    if packet_format.model is None:
        raise DatasetError(packet_format.reason or "unsupported Velodyne packet")
    dual = packet_format.return_mode == "dual"
    if packet_format.model == "vlp16":
        return _decode_vlp16_packet(packet, dual=dual)
    if packet_format.model == "vlp32c":
        return _decode_vlp32c_packet(packet)
    name = packet_format.sensor
    if calibration is None:
        raise DatasetError(
            f"{name} packets need the sensor's calibration (db.xml or ROS YAML); "
            "pass it with --calibration"
        )
    lasers = _CALIBRATED_LASERS[packet_format.model]
    if len(calibration.lasers) != lasers:
        raise DatasetError(
            f"a {name} needs a {lasers}-laser calibration, got {len(calibration.lasers)}"
        )
    if packet_format.model == "vls128":
        return _decode_vls128_packet(packet, calibration)
    return _decode_hdl64e_packet(packet, calibration, dual=dual)


def decode_velodyne_scan(
    topic: str,
    timestamp_ns: int,
    data: bytes,
    *,
    min_range_m: float = 0.4,
    max_range_m: float = 200.0,
    x_forward: bool = True,
    calibration: VelodyneCalibration | None = None,
) -> PointCloud2Message:
    """Decode a ``VelodyneScan`` into a cloud in the sensor frame with per-point time.

    The ``time`` offsets (seconds) are relative to the scan's first firing; the
    header stamp is the scan's first packet stamp, so the offset is the time
    since the header. Returns an empty cloud for a scan with no packets.
    ``calibration`` is required for HDL-64E packets and ignored otherwise.
    """

    scan = decode_velodyne_scan_packets(data)
    xyz_parts: list[np.ndarray] = []
    intensity_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    first_stamp = scan.packet_stamps_ns[0] if scan.packet_stamps_ns else scan.timestamp_ns
    for stamp, packet in zip(scan.packet_stamps_ns, scan.packets, strict=True):
        points = decode_velodyne_packet(packet, calibration)
        valid = (points.range_m >= min_range_m) & (points.range_m <= max_range_m)
        xyz_parts.append(points.xyz[valid])
        intensity_parts.append(points.reflectivity[valid])
        time_parts.append(points.time_s[valid] + (stamp - first_stamp) * 1.0e-9)
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


def _corrected_points(
    azimuth_centideg: np.ndarray,
    laser: np.ndarray,
    distance_raw: np.ndarray,
    calibration: VelodyneCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """LidarView's correction model: per-laser rotation, elevation, distance and offsets.

    Returns the points (manual axes) and the corrected range, zero where nothing returned.
    """

    table = calibration.laser_table()
    rotation = np.deg2rad(azimuth_centideg * 0.01 - table["rot_correction_deg"][laser])
    vertical = np.deg2rad(table["vert_correction_deg"][laser])
    distance = distance_raw * calibration.distance_resolution_m + table["dist_correction_m"][laser]
    vert_offset = table["vert_offset_m"][laser]
    horiz_offset = table["horiz_offset_m"][laser]
    planar = distance * np.cos(vertical) - vert_offset * np.sin(vertical)
    xyz = np.stack(
        [
            planar * np.sin(rotation) - horiz_offset * np.cos(rotation),
            planar * np.cos(rotation) + horiz_offset * np.sin(rotation),
            distance * np.sin(vertical) + vert_offset,
        ],
        axis=-1,
    ).reshape(-1, 3)
    return xyz, np.where(distance_raw > 0, distance, 0.0)


# VLS-128 firing timing (manual rev. 4, firmware 5.2.3.0, as LidarView applies it): a
# firing sequence is 20 cycles of 2.89 us; eight lasers fire together per cycle, lasers
# 64-127 two cycles later (maintenance cycles), all 7 us before the block's stamp.
_VLS128_CYCLE_US = 2.89
_VLS128_OFFSET_US = 7.0


def _decode_vls128_packet(
    packet: bytes, calibration: VelodyneCalibration, *, cycle_us: float = _VLS128_CYCLE_US
) -> VelodynePacketPoints:
    """VLS-128 single return: 3 firing sequences of 4 blocks (32 lasers each), 4 mm units.

    ``cycle_us`` is the firing cycle: 2.89 us from firmware 5.2.3.0 (manual rev. 4, the
    default), 2.66666 us before it (rev. 3). It sets the per-point time only: the
    azimuth advance is a fraction of the sequence, which the cycle cancels from.

    Each laser's azimuth is advanced by the fraction of its firing sequence elapsed when
    it fires, at the azimuth step to the same bank of the next sequence (the last
    sequence of a packet reuses the step before it; a step over 1 deg, a field-of-view
    gap, reuses the previous one), rounded to 0.01 deg as LidarView does.
    """

    raw = _blocks(packet, tuple(_VLS128_BANKS))
    flag = raw[:, 0].astype(np.int64) | (raw[:, 1].astype(np.int64) << 8)
    bank = np.asarray([_VLS128_BANKS[int(value)] for value in flag])
    laser = bank[:, None] + np.arange(32)[None, :]
    azimuth = raw[:, 2].astype(np.int64) + 256 * raw[:, 3].astype(np.int64)
    channels = raw[:, 4:].reshape(_BLOCKS, 32, 3).astype(np.int64)
    distance_raw = channels[..., 0] + 256 * channels[..., 1]
    reflectivity = channels[..., 2].astype(np.float64)

    steps = np.empty(_BLOCKS, dtype=np.int64)
    previous = 0
    for block in range(_BLOCKS):
        start = min(block, _BLOCKS - 5)
        step = int((azimuth[start + 4] - azimuth[start]) % 36000)
        if step > 100:
            step = previous
        previous = step
        steps[block] = step
    blocks = np.arange(_BLOCKS)
    channel = (np.arange(32)[None, :] + 32 * (blocks % 4)[:, None]).astype(np.int64)
    sequence_us = 20 * cycle_us
    sequence_start_us = sequence_us * (blocks // 4) - _VLS128_OFFSET_US
    time_us = (
        sequence_us * (blocks // 4)[:, None]
        + (channel // 8 + (channel // 64) * 2) * cycle_us
        - _VLS128_OFFSET_US
    )
    fraction = (time_us - sequence_start_us[:, None]) / sequence_us
    adjusted = (azimuth[:, None] + np.floor(steps[:, None] * fraction + 0.5)) % 36000
    xyz, range_m = _corrected_points(adjusted, laser, distance_raw, calibration)
    partner = np.full(range_m.shape, -1, dtype=np.int64)
    return _packet_points(xyz, reflectivity, range_m, time_us, laser, partner)


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


def _blocks(packet: bytes, flags: tuple[int, ...] = (_UPPER_BLOCK,)) -> np.ndarray:
    raw = np.frombuffer(packet, dtype=np.uint8, count=_BLOCKS * _BLOCK_SIZE).reshape(
        _BLOCKS, _BLOCK_SIZE
    )
    flag = raw[:, 0].astype(np.int64) | (raw[:, 1].astype(np.int64) << 8)
    if not np.all(np.isin(flag, flags)):
        raise DatasetError("Velodyne packet block flag is not 0xEEFF")
    return raw


def _block_azimuth_deg(raw: np.ndarray) -> np.ndarray:
    return (raw[:, 2].astype(np.float64) + 256.0 * raw[:, 3]) * 0.01


def _packet_points(
    xyz: np.ndarray,
    reflectivity: np.ndarray,
    range_m: np.ndarray,
    time_us: np.ndarray,
    laser: np.ndarray,
    partner: np.ndarray,
) -> VelodynePacketPoints:
    """Drop zero-range returns, and second returns equal to their first return.

    ``partner`` holds, for each second return, the flat index of the first return
    of the same firing and laser, and -1 elsewhere.
    """

    flat_range = range_m.reshape(-1).astype(np.float64)
    flat_reflectivity = reflectivity.reshape(-1).astype(np.float64)
    flat_partner = partner.reshape(-1)
    keep = flat_range > 0.0
    second = np.flatnonzero(flat_partner >= 0)
    first = flat_partner[second]
    duplicate = (flat_range[second] == flat_range[first]) & (
        flat_reflectivity[second] == flat_reflectivity[first]
    )
    keep[second[duplicate]] = False
    return VelodynePacketPoints(
        xyz=xyz[keep],
        reflectivity=flat_reflectivity[keep],
        range_m=flat_range[keep],
        time_s=(time_us.reshape(-1) * 1.0e-6)[keep],
        laser=laser.reshape(-1).astype(np.int64)[keep],
        second_return=(flat_partner >= 0)[keep],
    )


def _decode_vlp16_packet(packet: bytes, *, dual: bool = False) -> VelodynePacketPoints:
    """VLP-16: 12 blocks x 2 firing sequences x 16 channels, 2 mm units, 55.296 us per sequence.

    In dual-return mode blocks come in pairs with one azimuth: the even block holds
    the last return and the odd block the strongest (or second-strongest) one.
    """

    raw = _blocks(packet)
    azimuth = _block_azimuth_deg(raw)
    stride = 2 if dual else 1
    firing_azimuth = azimuth[::stride]
    firing_step = np.empty_like(firing_azimuth)
    firing_step[:-1] = (firing_azimuth[1:] - firing_azimuth[:-1]) % 360.0
    firing_step[-1] = firing_step[-2]
    block_step = np.repeat(firing_step, stride)
    firing = np.arange(_BLOCKS) // stride
    channels = raw[:, 4:].reshape(_BLOCKS, 2, 16, 3).astype(np.float64)
    range_m = (channels[..., 0] + 256.0 * channels[..., 1]) * 0.002
    reflectivity = channels[..., 2]
    sequence_fraction = np.array([0.0, 0.5])[None, :, None]
    # Within a sequence the head turns by 16 firings of 2.304 us of a 55.296 us cycle.
    azimuth_deg = (
        azimuth[:, None, None]
        + block_step[:, None, None] * sequence_fraction
        + block_step[:, None, None] * 0.5 * (np.arange(16)[None, None, :] * 2.304 / 55.296)
    )
    elevation = np.broadcast_to(_VLP16_ELEVATION_DEG[None, None, :], range_m.shape)
    time_us = (
        firing[:, None, None] * 110.592
        + np.arange(2)[None, :, None] * 55.296
        + np.arange(16)[None, None, :] * 2.304
    )
    flat_index = np.arange(range_m.size).reshape(range_m.shape)
    partner = np.full(range_m.shape, -1, dtype=np.int64)
    if dual:
        partner[1::2] = flat_index[1::2] - 32
    xyz = _spherical_to_xyz(range_m.reshape(-1), azimuth_deg.reshape(-1), elevation.reshape(-1))
    laser = np.broadcast_to(np.arange(16)[None, None, :], range_m.shape)
    return _packet_points(xyz, reflectivity, range_m, time_us, laser, partner)


def _decode_vlp32c_packet(packet: bytes) -> VelodynePacketPoints:
    """VLP-32C single return: 12 blocks x 32 channels, 4 mm units, 55.296 us per block."""

    raw = _blocks(packet)
    azimuth = _block_azimuth_deg(raw)
    block_step = np.empty_like(azimuth)
    block_step[:-1] = (azimuth[1:] - azimuth[:-1]) % 360.0
    block_step[-1] = block_step[-2]
    channels = raw[:, 4:].reshape(_BLOCKS, 32, 3).astype(np.float64)
    range_m = (channels[..., 0] + 256.0 * channels[..., 1]) * 0.004
    reflectivity = channels[..., 2]
    # Lasers fire in pairs every 2.304 us of the 55.296 us block: the head has
    # turned by that fraction of the block's azimuth step when each pair fires.
    fraction = (np.arange(32) // 2) * 2.304 / 55.296
    azimuth_deg = (
        azimuth[:, None]
        + block_step[:, None] * fraction[None, :]
        + _VLP32C_AZIMUTH_OFFSET_DEG[None, :]
    )
    elevation = np.broadcast_to(_VLP32C_ELEVATION_DEG[None, :], range_m.shape)
    time_us = np.arange(_BLOCKS)[:, None] * 55.296 + (np.arange(32)[None, :] // 2) * 2.304
    xyz = _spherical_to_xyz(range_m.reshape(-1), azimuth_deg.reshape(-1), elevation.reshape(-1))
    laser = np.broadcast_to(np.arange(32)[None, :], range_m.shape)
    partner = np.full(range_m.shape, -1, dtype=np.int64)
    return _packet_points(xyz, reflectivity, range_m, time_us, laser, partner)


# HDL-64E firing timing (manual): an upper (lasers 0-31) and a lower (32-63) block
# fire together; within a firing the 32 lasers fire in groups of four.
_HDL64E_GROUP_US = {False: (2.34, 3.54, 4.74, 6.0), True: (3.5, 4.7, 5.9, 7.2)}
_HDL64E_FIRING_US = {False: 48.0, True: 57.6}


def _decode_hdl64e_packet(
    packet: bytes, calibration: VelodyneCalibration, *, dual: bool
) -> VelodynePacketPoints:
    """HDL-64E: 6 (single) or 3 (dual) firings of an upper and a lower block.

    In dual-return mode each firing spans four blocks: upper and lower first
    returns, then upper and lower second returns, all with one azimuth.
    """

    raw = _blocks(packet, (_UPPER_BLOCK, _LOWER_BLOCK))
    flag = raw[:, 0].astype(np.int64) | (raw[:, 1].astype(np.int64) << 8)
    base = np.where(flag == _UPPER_BLOCK, 0, 32)
    laser = base[:, None] + np.arange(32)[None, :]
    azimuth = raw[:, 2].astype(np.int64) + 256 * raw[:, 3].astype(np.int64)
    channels = raw[:, 4:].reshape(_BLOCKS, 32, 3).astype(np.int64)
    distance_raw = channels[..., 0] + 256 * channels[..., 1]
    reflectivity = channels[..., 2].astype(np.float64)

    # Each laser fires a little after its firing's azimuth stamp: advance the
    # azimuth by the fraction of the firing period elapsed, at the packet's
    # largest block-to-block step (as LidarView does), in 0.01 deg units.
    diffs = np.sort((azimuth[1:] - azimuth[:-1] + 54000) % 36000 - 18000)
    azimuth_step = int(diffs[-1])
    if not 0 <= azimuth_step <= 100:
        azimuth_step = 0
    group_us = _HDL64E_GROUP_US[dual]
    period_us = _HDL64E_FIRING_US[dual]
    reversed_laser = 31 - np.arange(32)
    within_us = np.asarray(group_us)[reversed_laser % 4] + (reversed_laser // 4) * group_us[3]
    firings_per_group = 4 if dual else 2
    firing = (_BLOCKS - 1 - np.arange(_BLOCKS)) // firings_per_group
    before_stamp_us = firing[:, None] * period_us + within_us[None, :]
    time_us = before_stamp_us[0, 0] - before_stamp_us
    fraction = (within_us[0] - within_us) / period_us
    adjusted = (azimuth[:, None] + np.floor(azimuth_step * fraction[None, :] + 0.5)) % 36000
    xyz, range_m = _corrected_points(adjusted, laser, distance_raw, calibration)
    flat_index = np.arange(range_m.size).reshape(range_m.shape)
    partner = np.full(range_m.shape, -1, dtype=np.int64)
    if dual:
        second = (np.arange(_BLOCKS) % 4) >= 2
        partner[second] = flat_index[second] - 64
    return _packet_points(xyz, reflectivity, range_m, time_us, laser, partner)
