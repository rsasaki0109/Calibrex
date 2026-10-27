"""ROS 1 chunk records must fit inside the decompressed buffer."""
import bz2
import struct

import pytest

from calibrex.core.exceptions import DatasetError
from calibrex.data import rosbag1


def fields(**values):
    chunks = [name.encode('ascii') + b'=' + value for name, value in values.items()]
    return b''.join(struct.pack('<I', len(chunk)) + chunk for chunk in chunks)


def record(header, data):
    return struct.pack('<I', len(header)) + header + struct.pack('<I', len(data)) + data


def test_every_incomplete_record_prefix_is_rejected():
    header = fields(op=b'\xff')
    complete = record(header, b'payload')
    assert list(rosbag1._iter_inner_records(complete)) == [({'op': b'\xff'}, b'payload')]
    for end in range(1, len(complete)):
        with pytest.raises(DatasetError, match='truncated ROS bag inner record'):
            list(rosbag1._iter_inner_records(complete[:end]))


@pytest.mark.parametrize('length', [100, 2**32 - 1])
def test_declared_lengths_cannot_exceed_chunk(length):
    for content in (struct.pack('<I', length) + b'x',
                    struct.pack('<II', 0, length) + b'x'):
        with pytest.raises(DatasetError, match='truncated ROS bag inner record'):
            list(rosbag1._iter_inner_records(content))


def test_empty_and_multiple_records_preserve_boundaries():
    empty = record(b'', b'')
    full = record(fields(op=b'\xff'), b'abc')
    assert list(rosbag1._iter_inner_records(b'')) == []
    assert list(rosbag1._iter_inner_records(empty + full + empty)) == [
        ({}, b''), ({'op': b'\xff'}, b'abc'), ({}, b'')]
    for tail in (b'\x00', b'\x00' * 2, b'\x00' * 3):
        with pytest.raises(DatasetError, match='header length'):
            list(rosbag1._iter_inner_records(full + tail))


@pytest.mark.parametrize('compression', ['none', 'bz2', 'lz4'])
def test_public_reader_rejects_truncated_inner_message_in_complete_chunk(tmp_path, compression):
    def compress(data):
        return data

    if compression == 'bz2':
        compress = bz2.compress
    elif compression == 'lz4':
        compress = pytest.importorskip('lz4.frame').compress
    conn = struct.pack('<I', 1)
    connection = record(fields(op=bytes([rosbag1.OP_CONNECTION]), conn=conn, topic=b'/imu'),
                        fields(type=b'sensor_msgs/Imu'))
    message = record(fields(op=bytes([rosbag1.OP_MSG_DATA]), conn=conn,
                            time=struct.pack('<II', 2, 3)), b'payload')
    bag = tmp_path / 'records.bag'

    def write_chunk(content):
        header = fields(op=bytes([rosbag1.OP_CHUNK]), compression=compression.encode('ascii'),
                        size=struct.pack('<I', len(content)))
        bag.write_bytes(rosbag1.BAG_MAGIC + connection + record(header, compress(content)))

    write_chunk(message)
    messages = list(rosbag1.iter_messages(bag))
    assert len(messages) == 1
    assert messages[0][0].topic == '/imu'
    assert messages[0][1:] == (2_000_000_003, b'payload')
    for end in range(1, len(message)):
        # Outer record framing, compression and declared expanded size remain valid.
        write_chunk(message[:end])
        with pytest.raises(DatasetError, match='truncated ROS bag inner record'):
            list(rosbag1.iter_messages(bag))


@pytest.mark.parametrize('case', ['concatenated', 'junk', 'truncated', 'corrupt', 'empty'])
def test_public_reader_validates_whole_bz2_chunk(tmp_path, case):
    conn = struct.pack('<I', 1)
    connection = record(fields(op=bytes([rosbag1.OP_CONNECTION]), conn=conn, topic=b'/imu'),
                        fields(type=b'sensor_msgs/Imu'))
    message = record(fields(op=bytes([rosbag1.OP_MSG_DATA]), conn=conn,
                            time=struct.pack('<II', 2, 3)), b'payload')
    first = bz2.compress(message)
    second = bz2.compress(message)
    corrupt = bytearray(second)
    corrupt[10] ^= 1
    encoded = {'concatenated': first + second, 'junk': first + b'junk',
               'truncated': first + second[:-1], 'corrupt': first + bytes(corrupt),
               'empty': b''}[case]
    size = len(message) * 2 if case == 'concatenated' else len(message)
    if case == 'empty':
        size = 0
    header = fields(op=bytes([rosbag1.OP_CHUNK]), compression=b'bz2',
                    size=struct.pack('<I', size))
    bag = tmp_path / 'compressed.bag'
    bag.write_bytes(rosbag1.BAG_MAGIC + connection + record(header, encoded))
    if case == 'concatenated':
        messages = list(rosbag1.iter_messages(bag))
        assert len(messages) == 2
        assert all(item[0].topic == '/imu' and item[1:] == (2_000_000_003, b'payload')
                   for item in messages)
    else:
        with pytest.raises(DatasetError, match='bz2'):
            list(rosbag1.iter_messages(bag))


@pytest.mark.parametrize('reader,width', [
    (rosbag1._uint32, 4), (rosbag1._uint64, 8), (rosbag1._time_field_to_ns, 8),
])
def test_numeric_header_fields_require_exact_width_when_present(reader, width):
    assert reader(None) is None
    assert reader(b'\x00' * width) == 0
    for length in (*range(width), width + 1, width * 2):
        with pytest.raises(DatasetError, match='header field length'):
            reader(b'\x00' * length)
    assert rosbag1._uint32(struct.pack('<I', 2**32 - 1)) == 2**32 - 1
    assert rosbag1._uint64(struct.pack('<Q', 2**64 - 1)) == 2**64 - 1
    assert rosbag1._time_field_to_ns(struct.pack('<II', 2, 3)) == 2_000_000_003


@pytest.mark.parametrize('field,width', [
    ('connection_id', 4), ('message_connection_id', 4), ('time', 8),
    ('index_pos', 8), ('chunk_count', 4), ('op', 1),
])
@pytest.mark.parametrize('delta', [-1, 1])
def test_public_reader_rejects_wrong_header_field_width(tmp_path, field, width, delta):
    value = b'\x00' * (width + delta)
    if field == 'connection_id':
        header = fields(op=bytes([rosbag1.OP_CONNECTION]), conn=value, topic=b'/imu')
        encoded = record(header, fields(type=b'sensor_msgs/Imu'))
    elif field in ('message_connection_id', 'time'):
        conn = value if field == 'message_connection_id' else struct.pack('<I', 1)
        stamp = value if field == 'time' else struct.pack('<II', 2, 3)
        inner = record(fields(op=bytes([rosbag1.OP_MSG_DATA]), conn=conn, time=stamp), b'payload')
        encoded = record(fields(op=bytes([rosbag1.OP_CHUNK]), compression=b'none',
                                size=struct.pack('<I', len(inner))), inner)
    elif field == 'op':
        encoded = record(fields(op=value), b'')
    else:
        header = {'op': bytes([rosbag1.OP_BAG_HEADER]), field: value}
        encoded = record(fields(**header), b'')
    bag = tmp_path / 'header.bag'
    bag.write_bytes(rosbag1.BAG_MAGIC + encoded)
    with pytest.raises(DatasetError, match=r'header.*(length|missing)'):
        list(rosbag1.iter_messages(bag))
