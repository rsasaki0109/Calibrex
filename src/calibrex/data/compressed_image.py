"""Gray pixels from ``sensor_msgs/CompressedImage`` payloads (JPEG, PNG), without ROS.

image_transport's ``compressed`` plugin writes ``format`` as
``"<encoding>; <jpeg|png> compressed <encoding>"``; older drivers write just ``"jpeg"``
or ``"png"``. Whatever encoding the string names, the payload is a standard JPEG or PNG
file whose colours are in the usual order (OpenCV's encoder takes BGR and writes it
correctly). The pixels are therefore decoded from the file itself, and the format string
is used only to refuse payloads that are not pictures: ``compressedDepth`` (depth maps)
is rejected.

Pillow (``calibrex[images]``) decodes; OpenCV (``calibrex[opencv]``) is used when Pillow
is not installed. Gray is computed with the weights of
:func:`calibrex.data.ros2_camera_imu.to_gray`, and 16-bit images keep their high byte as
``mono16`` does there, so a lossless PNG of a raw image decodes to the same gray image as
the raw ``sensor_msgs/Image``. A JPEG is lossy: its gray image differs from the raw one by
the compression error, and decoders of different libjpeg builds may differ by a level or
two. The decoder and its version are reported by :func:`compressed_image_decoder` so a
check can record them.
"""

from __future__ import annotations

import importlib
from io import BytesIO
from typing import Any

import numpy as np

from calibrex.data.ros2_camera_imu import GrayImage, color_to_gray
from calibrex.data.ros_messages import MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH

INSTALL_HINT = (
    "decoding sensor_msgs/CompressedImage needs Pillow: pip install 'calibrex[images]' "
    "(or OpenCV: pip install 'calibrex[opencv]')"
)


def _pillow() -> Any | None:
    try:
        import PIL
        from PIL import Image
    except ImportError:
        return None
    return Image, PIL.__version__


def _opencv() -> Any | None:
    try:
        return importlib.import_module("cv2")
    except ImportError:
        return None


def compressed_image_decoder() -> str | None:
    """The decoder this process would use, as ``"pillow 12.3.0"``, or ``None``."""

    pillow = _pillow()
    if pillow is not None:
        return f"pillow {pillow[1]}"
    cv2 = _opencv()
    if cv2 is not None:
        return f"opencv {cv2.__version__}"
    return None


def require_compressed_image_decoder() -> str:
    """Return :func:`compressed_image_decoder` or raise ``ImportError`` with the install hint."""

    decoder = compressed_image_decoder()
    if decoder is None:
        raise ImportError(INSTALL_HINT)
    return decoder


def is_depth_format(image_format: str) -> bool:
    """Whether a ``format`` string names image_transport's ``compressedDepth`` payload."""

    return "compresseddepth" in image_format.lower().replace(" ", "")


def decode_compressed_gray(data: bytes, image_format: str = "") -> GrayImage:
    """Decode a JPEG or PNG payload to an 8-bit gray image."""

    if is_depth_format(image_format):
        raise ValueError(f"{image_format!r} is a compressed depth map, not a camera image")
    pillow = _pillow()
    if pillow is not None:
        return _decode_with_pillow(pillow[0], data, image_format)
    cv2 = _opencv()
    if cv2 is not None:
        return _decode_with_opencv(cv2, data, image_format)
    raise ImportError(INSTALL_HINT)


def _check_size(width: int, height: int, image_format: str) -> None:
    if not (0 < width <= MAX_IMAGE_WIDTH and 0 < height <= MAX_IMAGE_HEIGHT):
        raise ValueError(
            f"compressed image ({image_format!r}) is {width}x{height}, outside the supported "
            f"size {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}"
        )


def _decode_with_pillow(image_module: Any, data: bytes, image_format: str) -> GrayImage:
    try:
        image = image_module.open(BytesIO(data))
        _check_size(int(image.width), int(image.height), image_format)
        image.load()
    except ValueError:
        raise
    except Exception as exc:  # Pillow raises OSError, UnidentifiedImageError, SyntaxError
        raise ValueError(f"cannot decode compressed image ({image_format!r}): {exc}") from exc
    mode = image.mode
    if mode == "L":
        return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if mode in {"I;16", "I;16L", "I;16B", "I"}:
        values = np.asarray(image).astype(np.int64)
        return np.asarray(np.clip(values, 0, 65535) >> 8, dtype=np.uint8)
    if mode in {"1", "LA", "La"}:
        return np.ascontiguousarray(np.asarray(image.convert("L"), dtype=np.uint8))
    rgb = image if mode == "RGB" else image.convert("RGB")
    return color_to_gray(np.asarray(rgb, dtype=np.uint8), bgr=False)


def _decode_with_opencv(cv2: Any, data: bytes, image_format: str) -> GrayImage:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode compressed image ({image_format!r})")
    _check_size(int(image.shape[1]), int(image.shape[0]), image_format)
    if image.dtype == np.uint16:
        image = np.asarray(image >> 8, dtype=np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"compressed image ({image_format!r}) has unsupported depth {image.dtype}")
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    if image.shape[2] == 1:
        return np.ascontiguousarray(image[:, :, 0])
    if image.shape[2] == 2:  # gray and alpha
        return np.ascontiguousarray(image[:, :, 0])
    return color_to_gray(image[:, :, :3], bgr=True)
