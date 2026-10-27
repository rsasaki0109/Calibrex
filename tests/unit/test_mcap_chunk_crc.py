"""Validate MCAP chunk CRCs over expanded records, with zero meaning omitted."""
import struct
import zlib

import pytest

from calibrex.core.exceptions import DatasetError
from calibrex.data import rosbag2


@pytest.mark.parametrize('compression', ['', 'none', 'lz4', 'zstd'])
@pytest.mark.parametrize('payload', [b'', b'123456789'])
def test_chunk_crc_checks_expanded_data(compression, payload):
    encoded = payload
    if compression == 'lz4':
        encoded = pytest.importorskip('lz4.frame').compress(payload)
    elif compression == 'zstd':
        encoded = pytest.importorskip('zstandard').ZstdCompressor().compress(payload)

    def chunk(crc):
        name = compression.encode('ascii')
        return (struct.pack('<QQQI', 0, 0, len(payload), crc)
                + struct.pack('<I', len(name)) + name
                + struct.pack('<Q', len(encoded)) + encoded)

    # Standard CRC-32 test vector, independent of the implementation's calculation.
    expected_crc = 0xCBF43926 if payload else 0
    assert rosbag2._parse_mcap_chunk(chunk(expected_crc)) == (compression, payload)
    assert rosbag2._parse_mcap_chunk(chunk(0)) == (compression, payload)
    with pytest.raises(DatasetError, match='MCAP chunk CRC mismatch'):
        rosbag2._parse_mcap_chunk(chunk(expected_crc ^ 1))


def test_same_size_payload_corruption_is_rejected():
    original = b'payload'
    changed = b'payloae'
    content = (struct.pack('<QQQI', 0, 0, len(changed), zlib.crc32(original))
               + struct.pack('<IQ', 0, len(changed)) + changed)
    with pytest.raises(DatasetError, match='MCAP chunk CRC mismatch'):
        rosbag2._parse_mcap_chunk(content)


@pytest.mark.parametrize('compression', ['NONE', 'LZ4', 'ZSTD'])
@pytest.mark.parametrize('enable_crcs', [False, True])
def test_public_readers_validate_writer_chunk_crc(tmp_path, compression, enable_crcs):
    writer_module = pytest.importorskip('mcap.writer')
    if compression == 'LZ4':
        pytest.importorskip('lz4.frame')
    elif compression == 'ZSTD':
        pytest.importorskip('zstandard')
    bag = tmp_path / 'crc.mcap'
    with bag.open('wb') as stream:
        kind = getattr(writer_module.CompressionType, compression)
        writer = writer_module.Writer(stream,
                                      compression=kind,
                                      enable_crcs=enable_crcs)
        writer.start()
        schema = writer.register_schema('sensor_msgs/msg/Imu', 'ros2msg', b'')
        channel = writer.register_channel('/imu', 'cdr', schema)
        writer.add_message(channel, log_time=10, publish_time=10, data=b'payload')
        writer.finish()
    messages = list(rosbag2.iter_messages(bag))
    assert len(messages) == 1
    assert messages[0][1:] == (10, b'payload')
    assert rosbag2.list_rosbag2_connections(bag)[0][0].topic == '/imu'

    content = bytearray(bag.read_bytes())
    offset = len(rosbag2.MCAP_MAGIC)
    while content[offset] != rosbag2.OP_CHUNK:
        offset += 9 + struct.unpack_from('<Q', content, offset + 1)[0]
    crc_offset = offset + 9 + 24
    original_crc = struct.unpack_from('<I', content, crc_offset)[0]
    assert bool(original_crc) == enable_crcs
    # Keep all framing and codec bytes intact; only invalidate the chunk CRC.
    struct.pack_into('<I', content, crc_offset, (original_crc ^ 1) or 2)
    bag.write_bytes(content)
    with pytest.raises(DatasetError, match='MCAP chunk CRC mismatch'):
        list(rosbag2.iter_messages(bag))
    with pytest.raises(DatasetError, match='MCAP chunk CRC mismatch'):
        rosbag2.list_rosbag2_connections(bag)
