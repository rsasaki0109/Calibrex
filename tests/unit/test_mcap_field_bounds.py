"""MCAP field readers must respect enclosing record bounds."""
import io
import struct

import pytest

from calibrex.core.exceptions import DatasetError
from calibrex.data import rosbag2
from calibrex.data.ros_cdr_writer import encode_imu


def chunk(payload=b'abc', declared_length=None):
    length = len(payload) if declared_length is None else declared_length
    return (struct.pack('<QQQI', 0, 0, len(payload), 0) + struct.pack('<I', 0)
            + struct.pack('<Q', length) + payload)


def test_every_incomplete_chunk_prefix_is_rejected():
    content = chunk()
    assert rosbag2._parse_mcap_chunk(content) == ('', b'abc')
    for end in range(len(content)):
        with pytest.raises(DatasetError, match='truncated MCAP record field'):
            rosbag2._parse_mcap_chunk(content[:end])


@pytest.mark.parametrize('length', [4, 2**64 - 1])
def test_declared_encoded_length_cannot_exceed_record(length):
    with pytest.raises(DatasetError, match='truncated MCAP record field'):
        rosbag2._parse_mcap_chunk(chunk(declared_length=length))


@pytest.mark.parametrize('read,width', [(rosbag2._read_mcap_uint16, 2),
                                       (rosbag2._read_mcap_uint32, 4),
                                       (rosbag2._read_mcap_uint64, 8)])
def test_integer_fields_enforce_bounds_and_offsets(read, width):
    buffer = b'X' + b'\x01' + b'\x00' * (width - 1)
    assert read(buffer, 1) == (1, width + 1)
    for offset in (-1, 2, len(buffer), 10**100):
        with pytest.raises(DatasetError, match='truncated MCAP record field'):
            read(buffer, offset)


def test_strings_and_maps_cannot_silently_shorten_or_skip_past_record():
    assert rosbag2._read_mcap_string(struct.pack('<I', 0), 0) == ('', 4)
    assert rosbag2._read_mcap_string(struct.pack('<I', 2) + b'ok', 0) == ('ok', 6)
    assert rosbag2._skip_mcap_string_map(struct.pack('<I', 0), 0) == 4
    for buffer in (b'', b'\x00' * 3, struct.pack('<I', 10) + b'x',
                   struct.pack('<I', 2**32 - 1)):
        for read in (rosbag2._read_mcap_string, rosbag2._skip_mcap_string_map):
            with pytest.raises(DatasetError, match='truncated MCAP record field'):
                read(buffer, 0)


def test_empty_byte_field_can_end_at_record_boundary():
    assert rosbag2._read_mcap_length_prefixed_bytes(struct.pack('<Q', 0), 0) == (b'', 8)
    assert rosbag2._parse_mcap_chunk(chunk(b'')) == ('', b'')


def test_channel_metadata_and_message_headers_use_checked_fields():
    channel = struct.pack('<HHII', 1, 1, 0, 0) + struct.pack('<I', 10)
    with pytest.raises(DatasetError, match='truncated MCAP record field'):
        rosbag2._parse_mcap_channel(channel, {})
    for end in range(22):  # Channel ID, sequence and two uint64 timestamps.
        with pytest.raises(DatasetError, match='truncated MCAP record field'):
            rosbag2._parse_mcap_message(b'\x00' * end)


def test_public_reader_rejects_length_mismatch_inside_complete_outer_record(tmp_path):
    content = chunk(declared_length=999)
    bag = tmp_path / 'malformed.mcap'
    bag.write_bytes(rosbag2.MCAP_MAGIC + b'\x06' + struct.pack('<Q', len(content)) + content)
    with pytest.raises(DatasetError, match='truncated MCAP record field'):
        list(rosbag2.iter_messages(bag))


@pytest.mark.parametrize('length', [2**32, 2**63, 2**64 - 1])
def test_stream_reader_bounds_read_requests_for_untrusted_record_length(length):
    class BoundedStream(io.BytesIO):
        def read(self, size=-1):
            assert 0 <= size <= 1024 * 1024
            return super().read(size)

    source = BoundedStream(b'\xff' + struct.pack('<Q', length) + b'short')
    with pytest.raises(DatasetError, match='truncated MCAP record content'):
        list(rosbag2._iter_mcap_records(source))


def test_stream_reader_preserves_large_records_empty_records_and_boundaries():
    payload = b'x' * (2 * 1024 * 1024 + 1)
    encoded = b'\xff' + struct.pack('<Q', len(payload)) + payload
    empty = b'\xfe' + struct.pack('<Q', 0)
    assert list(rosbag2._iter_mcap_records(io.BytesIO(empty + encoded + empty))) == [
        (0xfe, b''), (0xff, payload), (0xfe, b'')]
    with pytest.raises(DatasetError, match='truncated MCAP record content'):
        list(rosbag2._iter_mcap_records(io.BytesIO(encoded[:-1])))


def test_stream_reader_rejects_every_incomplete_outer_record_prefix():
    encoded = b'\xff' + struct.pack('<Q', 7) + b'payload'
    for end in range(1, len(encoded)):
        with pytest.raises(DatasetError, match='truncated MCAP record'):
            list(rosbag2._iter_mcap_records(io.BytesIO(encoded[:end])))


@pytest.mark.parametrize('compressed', [False, True])
@pytest.mark.parametrize('length', [2**32, 2**63, 2**64 - 1])
def test_public_readers_reject_impossible_outer_length(tmp_path, compressed, length):
    encoded = rosbag2.MCAP_MAGIC + b'\xff' + struct.pack('<Q', length) + b'short'
    suffix = '.mcap'
    if compressed:
        encoded = pytest.importorskip('zstandard').ZstdCompressor().compress(encoded)
        suffix += '.zstd'
    bag = tmp_path / ('oversized' + suffix)
    bag.write_bytes(encoded)
    if compressed:
        (tmp_path / 'metadata.yaml').write_text(
            'rosbag2_bagfile_information:\n'
            '  version: 5\n'
            '  storage_identifier: mcap\n'
            f'  relative_file_paths: [{bag.name}]\n'
            '  compression_format: zstd\n'
            '  compression_mode: file\n')
        bag = tmp_path
    with pytest.raises(DatasetError, match='truncated MCAP record content'):
        list(rosbag2.iter_messages(bag))
    with pytest.raises(DatasetError, match='truncated MCAP record content'):
        rosbag2.list_rosbag2_connections(bag)


def test_public_message_reader_checks_chunk_start_before_unpacking(tmp_path):
    bag = tmp_path / 'short-chunk.mcap'
    for length in range(8):
        bag.write_bytes(rosbag2.MCAP_MAGIC + bytes([rosbag2.OP_CHUNK])
                        + struct.pack('<Q', length) + b'\x00' * length)
        with pytest.raises(DatasetError, match='truncated MCAP record field'):
            list(rosbag2.iter_messages(bag))


@pytest.mark.parametrize('encoded', [b'\xff', b'\x80', b'\xc0\xaf', b'\xe2\x82',
                                    b'\xed\xa0\x80', b'\xf4\x90\x80\x80'])
def test_string_fields_reject_invalid_utf8(encoded):
    content = b'prefix' + struct.pack('<I', len(encoded)) + encoded
    with pytest.raises(DatasetError, match='invalid UTF-8 in MCAP string field') as error:
        rosbag2._read_mcap_string(content, 6)
    assert isinstance(error.value.__cause__, UnicodeDecodeError)


@pytest.mark.parametrize('text', ['', '/imu', '/センサー/位置', 'sensor/🛰', '\ufffd'])
def test_string_fields_preserve_valid_unicode_and_byte_offsets(text):
    encoded = text.encode('utf-8')
    content = b'X' + struct.pack('<I', len(encoded)) + encoded + b'tail'
    assert rosbag2._read_mcap_string(content, 1) == (text, 5 + len(encoded))


def mcap_string(encoded):
    return struct.pack('<I', len(encoded)) + encoded


def mcap_record(opcode, content):
    return bytes([opcode]) + struct.pack('<Q', len(content)) + content


@pytest.mark.parametrize('field', ['schema_name', 'topic', 'encoding', 'compression'])
def test_public_readers_reject_invalid_utf8_fields(tmp_path, field):
    invalid = mcap_string(b'\xff')
    if field == 'schema_name':
        content = struct.pack('<H', 1) + invalid + mcap_string(b'ros2msg') + struct.pack('<I', 0)
        opcode = rosbag2.OP_SCHEMA
    elif field in ('topic', 'encoding'):
        topic = invalid if field == 'topic' else mcap_string(b'/imu')
        encoding = invalid if field == 'encoding' else mcap_string(b'cdr')
        content = struct.pack('<HH', 1, 1) + topic + encoding + struct.pack('<I', 0)
        opcode = rosbag2.OP_CHANNEL
    else:
        content = struct.pack('<QQQI', 0, 0, 0, 0) + invalid + struct.pack('<Q', 0)
        opcode = rosbag2.OP_CHUNK
    bag = tmp_path / 'invalid-string.mcap'
    bag.write_bytes(rosbag2.MCAP_MAGIC + mcap_record(opcode, content))
    with pytest.raises(DatasetError, match='invalid UTF-8 in MCAP string field'):
        list(rosbag2.iter_messages(bag))
    with pytest.raises(DatasetError, match='invalid UTF-8 in MCAP string field'):
        rosbag2.list_rosbag2_connections(bag)


def test_public_reader_preserves_unicode_names_and_binary_payload(tmp_path):
    topic = '/センサー/位置'
    message_type = '独自/msg/位置'
    schema = (struct.pack('<H', 1) + mcap_string(message_type.encode('utf-8'))
              + mcap_string(b'ros2msg') + struct.pack('<I', 0))
    channel = (struct.pack('<HH', 1, 1) + mcap_string(topic.encode('utf-8'))
               + mcap_string(b'cdr') + struct.pack('<I', 0))
    payload = b'\xff\x80\x00'
    message = struct.pack('<HIQQ', 1, 0, 10, 10) + payload
    bag = tmp_path / 'unicode.mcap'
    bag.write_bytes(rosbag2.MCAP_MAGIC + mcap_record(rosbag2.OP_SCHEMA, schema)
                    + mcap_record(rosbag2.OP_CHANNEL, channel)
                    + mcap_record(rosbag2.OP_MESSAGE, message))
    messages = list(rosbag2.iter_messages(bag, topics={topic}))
    assert len(messages) == 1
    assert messages[0][0].topic == topic
    assert messages[0][0].message_type == message_type
    assert messages[0][1:] == (10, payload)
    assert rosbag2.list_rosbag2_connections(bag)[0][0] == messages[0][0]


@pytest.mark.parametrize('encoding', ['cdr', 'json', 'protobuf', '', 'CDR'])
def test_mcap_encoding_is_preserved_and_checked_before_sensor_decoding(tmp_path, encoding):
    schema = (struct.pack('<H', 1) + mcap_string(rosbag2.IMU_TYPE.encode())
              + mcap_string(b'ros2msg') + struct.pack('<I', 0))
    channel = (struct.pack('<HH', 1, 1) + mcap_string(b'/imu')
               + mcap_string(encoding.encode()) + struct.pack('<I', 0))
    # Even CDR-shaped bytes must not override an incompatible declared encoding.
    payload = encode_imu(frame_id='imu', timestamp_ns=10,
                         angular_velocity=(1.0, 2.0, 3.0),
                         linear_acceleration=(0.0, 0.0, 9.8))
    message = struct.pack('<HIQQ', 1, 0, 10, 10) + payload
    bag = tmp_path / 'encoding.mcap'
    bag.write_bytes(rosbag2.MCAP_MAGIC + mcap_record(rosbag2.OP_SCHEMA, schema)
                    + mcap_record(rosbag2.OP_CHANNEL, channel)
                    + mcap_record(rosbag2.OP_MESSAGE, message))
    raw = list(rosbag2.iter_messages(bag))
    assert raw[0][0].serialization_format == encoding
    assert raw[0][2] == payload
    assert rosbag2.list_rosbag2_connections(bag)[0][0].serialization_format == encoding
    if encoding == 'cdr':
        decoded = list(rosbag2.read_imu_messages(bag))
        assert len(decoded) == 1
        assert decoded[0].angular_velocity == (1.0, 2.0, 3.0)
        assert rosbag2.summarize_rosbag2(bag).status != 'malformed'
    else:
        with pytest.raises(DatasetError, match='unsupported serialization format'):
            list(rosbag2.read_imu_messages(bag))
        summary = rosbag2.summarize_rosbag2(bag)
        assert summary.status == 'malformed'
        assert 'unsupported serialization format' in summary.reason
        assert '/imu' in summary.reason


@pytest.mark.parametrize('reader,message_type', [
    (rosbag2.read_pointcloud2_messages, rosbag2.POINTCLOUD2_TYPE),
    (rosbag2.read_lidar_messages, rosbag2.LIVOX_CUSTOMMSG_TYPE),
    (rosbag2.read_image_messages, rosbag2.IMAGE_TYPE),
    (rosbag2.read_camera_info_messages, rosbag2.CAMERA_INFO_TYPE),
    (rosbag2.read_radar_scan_messages, rosbag2.RADAR_SCAN_TYPE),
    (rosbag2.read_radar_messages, rosbag2.RADAR_SCAN_TYPE),
    (rosbag2.read_imu_messages, rosbag2.IMU_TYPE),
])
def test_typed_readers_check_format_after_message_type_filter(monkeypatch, reader, message_type):
    selected = rosbag2.Rosbag2Connection(1, '/sensor', message_type, 'json')
    unrelated = rosbag2.Rosbag2Connection(2, '/other', 'custom/msg/Other', 'json')
    monkeypatch.setattr(rosbag2, 'iter_messages', lambda *a, **kw: iter([(unrelated, 1, b'')]))
    assert list(reader('unused')) == []
    monkeypatch.setattr(rosbag2, 'iter_messages', lambda *a, **kw: iter([(selected, 1, b'')]))
    with pytest.raises(DatasetError, match="unsupported serialization format 'json'"):
        list(reader('unused'))


def test_generic_decoder_checks_explicit_format_and_preserves_cdr_default():
    payload = encode_imu(frame_id='imu', timestamp_ns=10,
                         angular_velocity=(1.0, 2.0, 3.0),
                         linear_acceleration=(0.0, 0.0, 9.8))
    for kwargs in ({}, {'serialization_format': 'cdr'}):
        message = rosbag2.decode_rosbag2_message('/imu', rosbag2.IMU_TYPE, 10, payload, **kwargs)
        assert message.angular_velocity == (1.0, 2.0, 3.0)
    for encoding in ('json', 'protobuf', '', 'CDR'):
        with pytest.raises(DatasetError, match='unsupported serialization format'):
            rosbag2.decode_rosbag2_message('/imu', rosbag2.IMU_TYPE, 10, payload,
                                           serialization_format=encoding)


def schema_content(data=b'\xff\x80\x00', encoding=b'ros2msg', length=None):
    return (struct.pack('<H', 1) + mcap_string(b'example/Message') + mcap_string(encoding)
            + struct.pack('<I', len(data) if length is None else length) + data)


@pytest.mark.parametrize('nested', [False, True])
def test_public_readers_reject_every_incomplete_schema_prefix(tmp_path, nested):
    complete = schema_content()
    assert rosbag2._parse_mcap_schema(complete) == (1, 'example/Message')
    bag = tmp_path / 'schema.mcap'
    for end in range(len(complete)):
        encoded = mcap_record(rosbag2.OP_SCHEMA, complete[:end])
        if nested:
            encoded = mcap_record(rosbag2.OP_CHUNK, chunk(encoded))
        # Outer record framing and chunk size remain valid at every cut.
        bag.write_bytes(rosbag2.MCAP_MAGIC + encoded)
        with pytest.raises(DatasetError, match='truncated MCAP record field'):
            list(rosbag2.iter_messages(bag))
        with pytest.raises(DatasetError, match='truncated MCAP record field'):
            rosbag2.list_rosbag2_connections(bag)


@pytest.mark.parametrize('length', [4, 2**32 - 1])
def test_schema_data_length_cannot_exceed_record(length):
    with pytest.raises(DatasetError, match='truncated MCAP record field'):
        rosbag2._parse_mcap_schema(schema_content(length=length))


def test_schema_encoding_requires_utf8_but_definition_is_binary(tmp_path):
    for data in (b'', b'\xff\x80\x00'):
        assert rosbag2._parse_mcap_schema(schema_content(data)) == (1, 'example/Message')
    bag = tmp_path / 'invalid-encoding.mcap'
    bag.write_bytes(rosbag2.MCAP_MAGIC
                    + mcap_record(rosbag2.OP_SCHEMA, schema_content(encoding=b'\xff')))
    with pytest.raises(DatasetError, match='invalid UTF-8'):
        list(rosbag2.iter_messages(bag))
    with pytest.raises(DatasetError, match='invalid UTF-8'):
        rosbag2.list_rosbag2_connections(bag)
