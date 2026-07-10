"""nuScenes radar PCD reader (binary PCD v0.7, no nuScenes SDK)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibrex.core.exceptions import DatasetError

_NUSCENES_RADAR_FIELDS = (
    "x",
    "y",
    "z",
    "dyn_prop",
    "id",
    "rcs",
    "vx",
    "vy",
    "vx_comp",
    "vy_comp",
    "is_quality_valid",
    "ambig_state",
    "x_rms",
    "y_rms",
    "invalid_state",
    "pdh0",
    "vx_rms",
    "vy_rms",
)
_NUSCENES_RADAR_SIZES = (4, 4, 4, 1, 2, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1)
_NUSCENES_RADAR_TYPES = (
    "F",
    "F",
    "F",
    "I",
    "I",
    "F",
    "F",
    "F",
    "F",
    "F",
    "I",
    "I",
    "I",
    "I",
    "I",
    "I",
    "I",
    "I",
)


@dataclass(frozen=True)
class NuScenesRadarPCD:
    """Decoded nuScenes radar point cloud."""

    fields: dict[str, Any]
    point_count: int

    def array(self, name: str) -> Any:
        """Return a field array; raises KeyError when missing."""

        return self.fields[name]


def read_nuscenes_radar_pcd(path: str | Path) -> NuScenesRadarPCD:
    """Read a nuScenes binary radar PCD file."""

    data = Path(path).read_bytes()
    marker = b"DATA binary"
    marker_index = data.find(marker)
    if marker_index < 0:
        msg = f"radar PCD header missing DATA binary record: {path}"
        raise DatasetError(msg)
    header_text = data[: marker_index + len(marker)].decode("utf-8", errors="replace")
    line_end = data.find(b"\n", marker_index)
    if line_end < 0:
        msg = "radar PCD header is missing a newline after DATA binary"
        raise DatasetError(msg)
    binary = data[line_end + 1 :]
    header = _parse_pcd_header(header_text)
    data_mode = str(header.get("data", ""))
    if data_mode != "binary":
        msg = f"unsupported radar PCD DATA mode {data_mode!r}; expected binary"
        raise DatasetError(msg)
    fields = _decode_binary_fields(header, binary)
    point_count = _int_value(header.get("points")) or _int_value(header.get("width"))
    return NuScenesRadarPCD(fields=fields, point_count=point_count)


def write_nuscenes_radar_pcd(path: str | Path, fields: dict[str, Any]) -> None:
    """Write a nuScenes-style binary radar PCD for tests and fixtures."""

    np_mod = _require_numpy()
    names = list(_NUSCENES_RADAR_FIELDS)
    arrays = []
    point_count = 0
    for name in names:
        values = fields.get(name)
        if values is None:
            values = np_mod.zeros(point_count, dtype=_field_dtype(name))
        array = np_mod.asarray(values)
        if point_count == 0 and array.shape[0] == 0:
            pass
        elif point_count == 0:
            point_count = int(array.shape[0])
        elif int(array.shape[0]) != point_count:
            msg = f"field {name} length mismatch for radar PCD writer"
            raise ValueError(msg)
        arrays.append(array.astype(_field_dtype(name), copy=False))
    header = _format_nuscenes_radar_header(point_count)
    payload = _encode_binary_fields(arrays)
    Path(path).write_bytes(header.encode("ascii") + payload)


def _parse_pcd_header(header_text: str) -> dict[str, str | list[str]]:
    lines = [line.strip() for line in header_text.splitlines() if line.strip()]
    parsed: dict[str, str | list[str]] = {}
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        if key in {"FIELDS", "SIZE", "TYPE", "COUNT"}:
            parsed[key.lower()] = parts[1:]
            continue
        parsed[key.lower()] = parts[1] if len(parts) == 2 else " ".join(parts[1:])
    return parsed



def _decode_binary_fields(header: dict[str, str | list[str]], binary: bytes) -> dict[str, Any]:
    np_mod = _require_numpy()
    names = _string_list(header.get("fields"))
    sizes = [_int_value(item) for item in _string_list(header.get("size"))]
    types = _string_list(header.get("type"))
    counts = [_int_value(item) for item in _string_list(header.get("count"))]
    if not names or len(names) != len(sizes) or len(names) != len(types):
        msg = "radar PCD header fields/size/type are inconsistent"
        raise DatasetError(msg)
    if counts and any(count != 1 for count in counts):
        msg = "radar PCD COUNT != 1 is not supported"
        raise DatasetError(msg)
    point_count = _int_value(header.get("points")) or _int_value(header.get("width"))
    if point_count == 0:
        names = _string_list(header.get("fields"))
        np_mod = _require_numpy()
        return {name: np_mod.asarray([], dtype=np_mod.float64) for name in names}
    stride = sum(sizes)
    expected = point_count * stride
    if len(binary) < expected:
        msg = f"radar PCD binary payload too short: expected {expected} bytes, got {len(binary)}"
        raise DatasetError(msg)

    unpackers = [
        _struct_code(type_code, size) for type_code, size in zip(types, sizes, strict=True)
    ]
    columns: dict[str, list[float]] = {name: [] for name in names}
    offset = 0
    for _ in range(point_count):
        for name, size, code in zip(names, sizes, unpackers, strict=True):
            end = offset + size
            (value,) = struct.unpack(code, binary[offset:end])
            columns[name].append(float(value))
            offset = end

    first = [columns[name][0] for name in names if columns[name]]
    if first and any(np_mod.isnan(value) for value in first):
        return {name: np_mod.asarray([], dtype=np_mod.float64) for name in names}

    output: dict[str, Any] = {}
    for name in names:
        output[name] = np_mod.asarray(columns[name], dtype=np_mod.float64)
    return output


def _encode_binary_fields(arrays: list[Any]) -> bytes:
    if not arrays:
        return b""
    point_count = len(arrays[0])
    codes = [
        _struct_code(_NUSCENES_RADAR_TYPES[index], _NUSCENES_RADAR_SIZES[index])
        for index in range(len(arrays))
    ]
    chunks: list[bytes] = []
    for point_index in range(point_count):
        for array, code in zip(arrays, codes, strict=True):
            value = array[point_index]
            if "f" in code or "d" in code:
                chunks.append(struct.pack(code, float(value)))
            else:
                chunks.append(struct.pack(code, int(value)))
    return b"".join(chunks)


def _format_nuscenes_radar_header(point_count: int) -> str:
    fields = " ".join(_NUSCENES_RADAR_FIELDS)
    sizes = " ".join(str(size) for size in _NUSCENES_RADAR_SIZES)
    types = " ".join(_NUSCENES_RADAR_TYPES)
    counts = " ".join("1" for _ in _NUSCENES_RADAR_FIELDS)
    return (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {sizes}\n"
        f"TYPE {types}\n"
        f"COUNT {counts}\n"
        f"WIDTH {point_count}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {point_count}\n"
        "DATA binary\n"
    )


def _field_dtype(name: str) -> Any:
    np_mod = _require_numpy()
    if name in {
        "dyn_prop",
        "is_quality_valid",
        "ambig_state",
        "x_rms",
        "y_rms",
        "invalid_state",
        "pdh0",
        "vx_rms",
        "vy_rms",
    }:
        return np_mod.int8
    if name == "id":
        return np_mod.int16
    return np_mod.float32


def _struct_code(type_code: str, size: int) -> str:
    lut = {
        "F": {4: "<f", 8: "<d"},
        "I": {1: "<b", 2: "<h", 4: "<i", 8: "<q"},
        "U": {1: "<B", 2: "<H", 4: "<I", 8: "<Q"},
    }
    try:
        return lut[type_code][size]
    except KeyError as exc:
        msg = f"unsupported radar PCD field type {type_code} size {size}"
        raise DatasetError(msg) from exc


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return value.split()
    return []


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _require_numpy() -> Any:
    try:
        import numpy as numpy_module
    except ImportError as exc:  # pragma: no cover - exercised via error path test
        msg = "nuScenes radar PCD support requires numpy"
        raise DatasetError(msg) from exc
    return numpy_module
