"""Image dimension padding and cropping utilities."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def resize_to_square(image_bytes: bytes) -> tuple[bytes | None, dict[str, int]]:
    """Resize an image to a square (1024x1024) to preserve all content without cropping.

    This squashes the image so OpenAI sees all original content without any padding
    that could confuse its composition generation.
    Returns (resized_image_bytes, original_dimensions)
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning("resize_to_square: failed to decode image")
        return None, {}

    # Convert to RGBA if not already
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    h, w = img.shape[:2]
    original_dims = {"w": w, "h": h}

    if h == w:
        return image_bytes, original_dims

    # Squash to 1024x1024 mapping the entire original image
    resized = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)

    success, encoded = cv2.imencode(".png", resized)
    if not success:
        logger.warning("resize_to_square: imencode failed")
        return None, original_dims

    return bytes(encoded), original_dims


def resize_to_original(
    image_bytes: bytes, original_dims: dict[str, int]
) -> bytes | None:
    """Stretch a squared image from OpenAI back to its original aspect ratio."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning("resize_to_original: failed to decode image")
        return None

    orig_w = original_dims.get("w")
    orig_h = original_dims.get("h")

    if not orig_w or not orig_h:
        return image_bytes

    # Stretch back to original dimensions
    resized = cv2.resize(img, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    success, encoded = cv2.imencode(".png", resized)
    if not success:
        logger.warning("resize_to_original: imencode failed")
        return None

    return bytes(encoded)
