"""Damaged compressed frames must never become successful partial bag reads."""
import bz2
import sqlite3
import struct

import pytest

from calibrex.core.exceptions import DatasetError
from calibrex.data import rosbag1, rosbag2


@pytest.mark.parametrize('content_size', [False, True])
@pytest.mark.parametrize('checksum', [False, True])
def test_zstd_requires_complete_frames_with_known_or_unknown_size(content_size, checksum):
    zstd = pytest.importorskip('zstandard')
    payload = b'calibrex payload' * 20
    frame = zstd.ZstdCompressor(write_content_size=content_size,
                               write_checksum=checksum).compress(payload)
    assert rosbag2._zstd_decompress_message(frame) == payload
    assert rosbag2._zstd_decompress(frame, len(payload)) == payload
    assert rosbag2._zstd_decompress(frame, 0) == payload
    for end in range(len(frame)):
        with pytest.raises(DatasetError, match='zstd'):
            rosbag2._zstd_decompress_message(frame[:end])
        with pytest.raises(DatasetError, match='zstd'):
            rosbag2._zstd_decompress(frame[:end], len(payload))
    with pytest.raises(DatasetError, match='unexpected size'):
        rosbag2._zstd_decompress(frame, len(payload) + 1)


def test_zstd_invalid_checksum_and_empty_complete_frame():
    zstd = pytest.importorskip('zstandard')
    frame = zstd.ZstdCompressor(write_checksum=True).compress(b'payload')
    corrupt = frame[:-1] + bytes([frame[-1] ^ 1])
    for data in (b'invalid frame', corrupt):
        with pytest.raises(DatasetError, match='invalid zstd') as error:
            rosbag2._zstd_decompress_message(data)
        assert isinstance(error.value.__cause__, zstd.ZstdError)
    assert rosbag2._zstd_decompress_message(zstd.ZstdCompressor().compress(b'')) == b''


def test_lz4_frame_corruption_is_a_dataset_error_for_each_reader():
    lz4 = pytest.importorskip('lz4.frame')
    payload = b'calibrex payload' * 20
    frame = lz4.compress(payload)
    readers = (lambda data: rosbag1._lz4_decompress(data, len(payload)),
               lambda data: rosbag2._lz4_decompress(data, len(payload)),
               rosbag2._lz4_decompress_message)
    for read in readers:
        assert read(frame) == payload
        for data in (b'invalid frame', frame[:-1]):
            with pytest.raises(DatasetError, match='lz4') as error:
                read(data)
            assert error.value.__cause__ is not None


def test_lz4_raw_block_corruption_is_a_dataset_error():
    lz4 = pytest.importorskip('lz4.block')
    payload = b'calibrex payload' * 20
    block = lz4.compress(payload, store_size=False)
    assert rosbag2._lz4_decompress(block, len(payload)) == payload
    with pytest.raises(DatasetError, match='lz4') as error:
        rosbag2._lz4_decompress(block[:-1], len(payload))
    assert isinstance(error.value.__cause__, lz4.LZ4BlockError)


def test_ros1_bz2_corruption_is_a_dataset_error():
    payload = b'calibrex payload' * 20
    frame = bz2.compress(payload)
    assert rosbag1._decompress_chunk('bz2', len(payload), frame) == payload
    for data in (b'invalid frame', frame[:-1]):
        with pytest.raises(DatasetError, match='bz2') as error:
            rosbag1._decompress_chunk('bz2', len(payload), data)
        assert error.value.__cause__ is not None


@pytest.mark.parametrize('case', ['concatenated', 'checksum', 'junk', 'second_frame',
                                 'second_checksum'])
@pytest.mark.parametrize('codec', ['zstd', 'lz4'])
def test_public_message_reader_validates_whole_compressed_payload(tmp_path, case, codec):
    bag = tmp_path / 'bad-bag'
    bag.mkdir()
    if codec == 'zstd':
        frame = pytest.importorskip('zstandard').ZstdCompressor(write_checksum=True).compress(
            b'complete message payload')
    else:
        frame = pytest.importorskip('lz4.frame').compress(
            b'complete message payload', content_checksum=True)
    encoded = {
        'concatenated': frame + frame,
        'checksum': frame[:-1],
        'junk': frame + b'junk',
        'second_frame': frame + frame[:-1],
        'second_checksum': frame + frame[:-1] + bytes([frame[-1] ^ 1]),
    }[case]
    with sqlite3.connect(bag / 'bag.db3') as db:
        db.execute('CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, '
                   'serialization_format TEXT, offered_qos_profiles TEXT)')
        db.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, '
                   'timestamp INTEGER, data BLOB)')
        db.execute("INSERT INTO topics VALUES (1, '/imu', 'sensor_msgs/msg/Imu', 'cdr', '')")
        db.execute('INSERT INTO messages VALUES (1, 1, 1, ?)', (encoded,))
    (bag / 'metadata.yaml').write_text(f'''rosbag2_bagfile_information:
  version: 5
  storage_identifier: sqlite3
  relative_file_paths: [bag.db3]
  compression_format: {codec}
  compression_mode: message
''')
    if case == 'concatenated':
        messages = list(rosbag2.iter_messages(bag))
        assert len(messages) == 1
        assert messages[0][2] == b'complete message payload' * 2
    else:
        with pytest.raises(DatasetError, match=codec):
            list(rosbag2.iter_messages(bag))


@pytest.mark.parametrize('content_size', [False, True])
def test_lz4_consumes_and_validates_every_frame(content_size):
    lz4 = pytest.importorskip('lz4.frame')
    first = lz4.compress(b'first', store_size=content_size, content_checksum=True)
    second = lz4.compress(b'second', store_size=content_size, content_checksum=True)
    empty = lz4.compress(b'')
    readers = (rosbag2._lz4_decompress_message,
               lambda data: rosbag1._lz4_decompress(data, 0),
               lambda data: rosbag2._lz4_decompress(data, 0))
    for magic in range(0x184D2A50, 0x184D2A60):
        skip = struct.pack('<II', magic, 3) + b'abc'
        for read in readers:
            for encoded in (first + second, empty + first + empty + second + empty,
                            skip + first + skip + second + skip):
                assert read(encoded) == b'firstsecond'
            assert read(skip + empty + skip) == b''
    for read in readers:
        for tail in (second, skip):
            for end in range(1, len(tail)):
                with pytest.raises(DatasetError, match='lz4'):
                    read(first + tail[:end])
        for tail in (b'junk', second[:-1] + bytes([second[-1] ^ 1])):
            with pytest.raises(DatasetError, match='lz4') as error:
                read(first + tail)
            assert error.value.__cause__ is not None
    for read_chunk in (rosbag1._lz4_decompress, rosbag2._lz4_decompress):
        assert read_chunk(first + second, 11) == b'firstsecond'
        with pytest.raises(DatasetError, match='unexpected size'):
            read_chunk(first + second, 5)


@pytest.mark.parametrize('content_size', [False, True])
def test_zstd_consumes_and_validates_every_frame(content_size):
    zstd = pytest.importorskip('zstandard')
    compressor = zstd.ZstdCompressor(write_content_size=content_size, write_checksum=True)
    first = compressor.compress(b'first')
    second = compressor.compress(b'second')
    empty = compressor.compress(b'')
    # Zstandard permits skippable frames as well as concatenated data frames.
    skip = struct.pack('<II', 0x184D2A50, 3) + b'abc'
    for encoded in (first + second, empty + first + empty + second + empty,
                    skip + first + skip + second + skip):
        assert rosbag2._zstd_decompress_message(encoded) == b'firstsecond'
        assert rosbag2._zstd_decompress(encoded, 11) == b'firstsecond'
        with pytest.raises(DatasetError, match='unexpected size'):
            rosbag2._zstd_decompress(encoded, 5)
    assert rosbag2._zstd_decompress_message(skip + empty + skip) == b''
    for tail in (second, skip):
        for end in range(1, len(tail)):
            with pytest.raises(DatasetError, match='zstd'):
                rosbag2._zstd_decompress_message(first + tail[:end])
            with pytest.raises(DatasetError, match='zstd'):
                rosbag2._zstd_decompress(first + tail[:end], 5)
    for tail in (b'junk', second[:-1] + bytes([second[-1] ^ 1])):
        with pytest.raises(DatasetError, match='invalid zstd') as error:
            rosbag2._zstd_decompress_message(first + tail)
        assert isinstance(error.value.__cause__, zstd.ZstdError)


@pytest.mark.parametrize('compression', ['', 'none', 'lz4', 'zstd'])
@pytest.mark.parametrize('payload', [b'', b'chunk records'])
def test_mcap_declared_chunk_size_is_checked_for_all_formats(compression, payload):
    encoded = payload
    if compression == 'lz4':
        encoded = pytest.importorskip('lz4.frame').compress(payload)
    elif compression == 'zstd':
        encoded = pytest.importorskip('zstandard').ZstdCompressor().compress(payload)

    def chunk(size):
        name = compression.encode('ascii')
        return (struct.pack('<QQQI', 0, 0, size, 0) + struct.pack('<I', len(name)) + name
                + struct.pack('<Q', len(encoded)) + encoded)

    assert rosbag2._parse_mcap_chunk(chunk(len(payload))) == (compression, payload)
    for size in {0, len(payload) + 1, max(0, len(payload) - 1)} - {len(payload)}:
        with pytest.raises(DatasetError, match='unexpected size'):
            rosbag2._parse_mcap_chunk(chunk(size))


@pytest.mark.parametrize('compression', ['none', 'bz2', 'lz4'])
@pytest.mark.parametrize('empty', [False, True])
def test_ros1_declared_chunk_size_is_checked_before_record_parsing(compression, empty):
    # A well-formed unknown record is ignored by the message iterator.
    header = struct.pack('<I', 4) + b'op=\xff'
    payload = b'' if empty else struct.pack('<I', len(header)) + header + struct.pack('<I', 0)
    encoded = payload
    if compression == 'bz2':
        encoded = bz2.compress(payload)
    elif compression == 'lz4':
        encoded = pytest.importorskip('lz4.frame').compress(payload)

    def records(size):
        fields = {'compression': compression.encode('ascii')}
        if size is not None:
            fields['size'] = struct.pack('<I', size)
        return list(rosbag1._iter_chunk_messages(fields, encoded, {}, None))

    assert records(len(payload)) == []
    assert records(None) == []  # Preserve the existing missing-size compatibility path.
    for size in {0, len(payload) + 1, max(0, len(payload) - 1)} - {len(payload)}:
        with pytest.raises(DatasetError, match='unexpected size'):
            records(size)


@pytest.mark.parametrize('size', [b'', b'\x01', b'\x00' * 5])
def test_malformed_ros1_size_fields_are_not_treated_as_absent(size):
    with pytest.raises(DatasetError, match='size field'):
        list(rosbag1._iter_chunk_messages({'compression': b'none', 'size': size}, b'', {}, None))
