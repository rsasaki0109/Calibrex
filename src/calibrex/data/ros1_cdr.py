"""Transcode ROS 1 serialized messages into ROS 2 CDR (XCDR1, little-endian).

A ROS 1 bag stores every message in the ROS 1 wire format and carries the full
``message_definition`` text in each connection record. This module parses that
definition and rewrites a payload into the byte layout the ROS 2 CDR decoders in
:mod:`calibrex.data.ros_cdr` already read, so every estimator input reader works
on a ``.bag`` without a second set of decoders.

The differences handled here:

* ROS 1 packs fields without alignment; CDR aligns every primitive to its size
  relative to the first byte after the 4-byte encapsulation header.
* A ROS 1 string is ``uint32 length`` plus the characters; CDR counts the
  terminating NUL in the length and stores it.
* ``std_msgs/Header`` carries a leading ``uint32 seq`` in ROS 1 that ROS 2 dropped.
* ``time`` / ``duration`` become two 4-byte integers (``builtin_interfaces``).

Arrays of fixed-size records (a Livox ``CustomPoint[]``, say) are rewritten with
numpy when it is available, so a 24 000-point message costs milliseconds.

The module is pure Python / numpy and imports nothing from ROS.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field

from calibrex.core.exceptions import DatasetError

_CDR_LITTLE_ENDIAN_HEADER = b"\x00\x01\x00\x00"
_PRIMITIVE_SIZES: dict[str, int] = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "byte": 1,
    "char": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float32": 4,
    "float64": 8,
}
_VECTORIZE_MIN_ELEMENTS = 16
_MAX_ALIGNMENT = 8

_FIELD_RE = re.compile(r"^([A-Za-z_][\w/]*)(\[(\d*)\])?\s+([A-Za-z_]\w*)\s*(?:#.*)?$")
_SEPARATOR_RE = re.compile(r"^={20,}\s*$")


@dataclass(frozen=True)
class _Field:
    name: str
    type_name: str
    array: int | None  # None: scalar, -1: variable-length sequence, >= 0: fixed array


@dataclass
class _Struct:
    name: str
    fields: list[_Field] = field(default_factory=list)
    drop_leading_seq: bool = False
    flat_sizes: list[int] | None = None  # primitive sizes when the record is fixed-size


def ros2_message_type(ros1_type: str) -> str:
    """Map a ROS 1 type name (``sensor_msgs/Imu``) to its ROS 2 spelling."""

    if ros1_type == "tf/tfMessage":
        return "tf2_msgs/msg/TFMessage"
    package, separator, name = ros1_type.partition("/")
    if not separator or "/msg/" in ros1_type:
        return ros1_type
    return f"{package}/msg/{name}"


class Ros1ToCdrTranscoder:
    """Rewrite payloads of one ROS 1 message type into CDR."""

    def __init__(self, ros1_type: str, definition: str) -> None:
        self._structs: dict[str, _Struct] = {}
        self._parse_definition(ros1_type, definition)
        self._root = self._qualify(ros1_type, ros1_type)
        if self._root not in self._structs:
            msg = f"ROS 1 message definition does not define {ros1_type!r}"
            raise DatasetError(msg)
        if not self._structs[self._root].fields:
            msg = f"ROS 1 message definition of {ros1_type!r} declares no fields"
            raise DatasetError(msg)
        for struct_ in self._structs.values():
            struct_.flat_sizes = self._flat_sizes(struct_.name, set())

    # -- definition parsing -------------------------------------------------

    def _parse_definition(self, root_type: str, definition: str) -> None:
        section_type = root_type
        lines: list[str] = []
        sections: list[tuple[str, list[str]]] = []
        for raw in definition.splitlines():
            line = raw.strip()
            if _SEPARATOR_RE.match(line):
                sections.append((section_type, lines))
                lines = []
                section_type = ""
                continue
            if line.startswith("MSG:") and not section_type:
                section_type = line[4:].strip()
                continue
            lines.append(line)
        sections.append((section_type, lines))
        for type_name, body in sections:
            qualified = self._qualify(type_name, type_name)
            struct_ = _Struct(name=qualified)
            for line in body:
                parsed = _parse_field_line(line)
                if parsed is not None:
                    struct_.fields.append(parsed)
            if qualified == "std_msgs/Header":
                struct_.drop_leading_seq = True
            self._structs[qualified] = struct_

    @staticmethod
    def _qualify(type_name: str, package_hint: str) -> str:
        if type_name == "Header":
            return "std_msgs/Header"
        if "/" in type_name:
            return type_name
        package = package_hint.partition("/")[0]
        return f"{package}/{type_name}"

    def _resolve(self, field_: _Field, owner: _Struct) -> str | None:
        """Return the qualified struct name of a field, or ``None`` for primitives."""

        name = field_.type_name
        if name in _PRIMITIVE_SIZES or name in ("string", "time", "duration"):
            return None
        qualified = self._qualify(name, owner.name)
        if qualified not in self._structs:
            msg = f"ROS 1 message definition lacks the nested type {name!r}"
            raise DatasetError(msg)
        return qualified

    def _flat_sizes(self, name: str, seen: set[str]) -> list[int] | None:
        struct_ = self._structs[name]
        if struct_.drop_leading_seq or name in seen:
            return None
        seen = seen | {name}
        sizes: list[int] = []
        for field_ in struct_.fields:
            if field_.array is not None:
                return None
            if field_.type_name == "string":
                return None
            if field_.type_name in ("time", "duration"):
                sizes.extend((4, 4))
            elif field_.type_name in _PRIMITIVE_SIZES:
                sizes.append(_PRIMITIVE_SIZES[field_.type_name])
            else:
                nested = self._resolve(field_, struct_)
                assert nested is not None
                inner = self._flat_sizes(nested, seen)
                if inner is None:
                    return None
                sizes.extend(inner)
        return sizes

    # -- transcoding --------------------------------------------------------

    def transcode(self, data: bytes) -> bytes:
        """Return the CDR encoding of one ROS 1 serialized message."""

        out = bytearray(_CDR_LITTLE_ENDIAN_HEADER)
        try:
            end = self._write_struct(self._root, data, 0, out)
        except (struct.error, IndexError) as exc:
            msg = "truncated or malformed ROS 1 message payload"
            raise DatasetError(msg) from exc
        if end != len(data):
            msg = "ROS 1 message payload has trailing bytes after the definition"
            raise DatasetError(msg)
        return bytes(out)

    def _write_struct(self, name: str, src: bytes, pos: int, out: bytearray) -> int:
        struct_ = self._structs[name]
        for index, field_ in enumerate(struct_.fields):
            if index == 0 and struct_.drop_leading_seq:
                pos += 4
                continue
            pos = self._write_field(field_, struct_, src, pos, out)
        return pos

    def _write_field(
        self, field_: _Field, owner: _Struct, src: bytes, pos: int, out: bytearray
    ) -> int:
        if field_.array is None:
            return self._write_value(field_, owner, src, pos, out)
        count = field_.array
        if count < 0:
            (count,) = struct.unpack_from("<I", src, pos)
            pos += 4
            _align(out, 4)
            out += struct.pack("<I", count)
        if count == 0:
            return pos
        type_name = field_.type_name
        size = _PRIMITIVE_SIZES.get(type_name)
        if size is not None:
            end = pos + count * size
            if end > len(src):
                msg = "ROS 1 array runs past the end of the message"
                raise DatasetError(msg)
            _align(out, size)
            out += src[pos:end]
            return end
        nested = self._resolve(field_, owner)
        if nested is not None:
            flat = self._structs[nested].flat_sizes
            if flat is not None and count >= _VECTORIZE_MIN_ELEMENTS:
                vectorized = _write_flat_array(flat, count, src, pos, out)
                if vectorized is not None:
                    return vectorized
        for _ in range(count):
            pos = self._write_value(field_, owner, src, pos, out)
        return pos

    def _write_value(
        self, field_: _Field, owner: _Struct, src: bytes, pos: int, out: bytearray
    ) -> int:
        type_name = field_.type_name
        size = _PRIMITIVE_SIZES.get(type_name)
        if size is not None:
            end = pos + size
            if end > len(src):
                msg = "ROS 1 message ends inside a primitive field"
                raise DatasetError(msg)
            _align(out, size)
            out += src[pos:end]
            return end
        if type_name == "string":
            length: int = struct.unpack_from("<I", src, pos)[0]
            pos += 4
            end = pos + length
            if end > len(src):
                msg = "ROS 1 string runs past the end of the message"
                raise DatasetError(msg)
            _align(out, 4)
            out += struct.pack("<I", length + 1)
            out += src[pos:end]
            out += b"\x00"
            return end
        if type_name in ("time", "duration"):
            if pos + 8 > len(src):
                msg = "ROS 1 message ends inside a time field"
                raise DatasetError(msg)
            _align(out, 4)
            out += src[pos : pos + 8]
            return pos + 8
        nested = self._resolve(field_, owner)
        assert nested is not None
        return self._write_struct(nested, src, pos, out)


def _align(out: bytearray, alignment: int) -> None:
    if alignment > 1:
        padding = (-(len(out) - 4)) % alignment
        if padding:
            out += bytes(padding)


def _parse_field_line(line: str) -> _Field | None:
    stripped = line.split("#", 1)[0].strip()
    if not stripped or "=" in stripped:  # blank, comment, or constant
        return None
    match = _FIELD_RE.match(stripped)
    if match is None:
        msg = f"cannot parse ROS 1 message definition line: {line!r}"
        raise DatasetError(msg)
    type_name, bracket, length, name = match.groups()
    array: int | None = None
    if bracket:
        array = int(length) if length else -1
    return _Field(name=name, type_name=type_name, array=array)


def _write_flat_array(
    sizes: list[int], count: int, src: bytes, pos: int, out: bytearray
) -> int | None:
    """Rewrite ``count`` fixed-size records with numpy; ``None`` when numpy is missing."""

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a core dependency of the readers
        return None
    packed = sum(sizes)
    end = pos + count * packed
    if end > len(src):
        msg = "ROS 1 record array runs past the end of the message"
        raise DatasetError(msg)
    # Layout of one record for each of the eight possible start phases (CDR offsets
    # are relative to the byte after the header; alignment is at most 8).
    field_offsets = np.zeros((_MAX_ALIGNMENT, len(sizes)), dtype=np.int64)
    ends = [0] * _MAX_ALIGNMENT
    for phase in range(_MAX_ALIGNMENT):
        cursor = phase
        for index, size in enumerate(sizes):
            cursor += (-cursor) % size
            field_offsets[phase, index] = cursor - phase
            cursor += size
        ends[phase] = cursor - phase
    start = len(out) - 4
    starts = np.empty(count, dtype=np.int64)
    cursor = start
    for index in range(count):
        starts[index] = cursor
        cursor += ends[cursor & 7]
    block = np.zeros(cursor - start, dtype=np.uint8)
    records = np.frombuffer(src, dtype=np.uint8, count=count * packed, offset=pos).reshape(
        count, packed
    )
    base = starts - start
    table = field_offsets[starts & 7]
    source_offset = 0
    for index, size in enumerate(sizes):
        destination = base + table[:, index]
        for byte in range(size):
            block[destination + byte] = records[:, source_offset + byte]
        source_offset += size
    out += block.tobytes()
    return end
