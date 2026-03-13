"""Face detection utilities for poster face inpainting pipeline."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FACE_CASCADE: cv2.CascadeClassifier | None = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE  # noqa: PLW0603
    if _FACE_CASCADE is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        _FACE_CASCADE = cv2.CascadeClassifier(path)
    return _FACE_CASCADE


def _decode(image_bytes: bytes) -> Any:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img if img is not None else None


def _detect_largest_face(gray: Any) -> tuple[int, int, int, int] | None:
    faces = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    return int(x), int(y), int(w), int(h)


def crop_face(image_bytes: bytes, padding: float = 0.25) -> bytes | None:
    """Detect the largest face in image_bytes and return a cropped JPEG.

    padding: fractional expansion around the detected bbox (0.25 = 25%).
    Returns None if the image cannot be decoded or no face is found.
    """
    img = _decode(image_bytes)
    if img is None:
        logger.warning("crop_face: failed to decode image")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = _detect_largest_face(gray)
    if face is None:
        logger.warning("crop_face: no face detected")
        return None

    x, y, w, h = face
    ph, pw = img.shape[:2]
    pad_x, pad_y = int(w * padding), int(h * padding)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(pw, x + w + pad_x), min(ph, y + h + pad_y)

    success, encoded = cv2.imencode(".jpg", img[y1:y2, x1:x2])
    if not success:
        logger.warning("crop_face: imencode failed")
        return None
    return bytes(encoded)


def create_face_mask_rgba(image_bytes: bytes, padding: float = 0.15) -> bytes | None:
    """Return an RGBA PNG of image_bytes where the face bbox is transparent (alpha=0).

    Transparent pixels tell OpenAI images.edit() to regenerate that region.
    padding: fractional expansion of the face bbox for the transparent region.
    Returns None if the image cannot be decoded or no face is found.
    """
    img = _decode(image_bytes)
    if img is None:
        logger.warning("create_face_mask_rgba: failed to decode image")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = _detect_largest_face(gray)
    if face is None:
        logger.warning("create_face_mask_rgba: no face detected in generated poster")
        return None

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = 255  # fully opaque

    x, y, w, h = face
    ph, pw = img.shape[:2]
    pad_x, pad_y = int(w * padding), int(h * padding)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(pw, x + w + pad_x), min(ph, y + h + pad_y)

    rgba[y1:y2, x1:x2, 3] = 0  # transparent → OpenAI will inpaint here

    success, encoded = cv2.imencode(".png", rgba)
    if not success:
        logger.warning("create_face_mask_rgba: imencode failed")
        return None
    return bytes(encoded)
