"""MCAP field readers must respect enclosing record bounds."""
import io
import struct

import pytest

from calibrex.core.exceptions import DatasetError
from calibrex.data import rosbag2


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
