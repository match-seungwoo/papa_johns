from __future__ import annotations

import io
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
_face_cascade = cv2.CascadeClassifier(_CASCADE_PATH)


def crop_face(image_bytes: bytes, padding: float = 0.1) -> bytes:
    """Detect the largest face and return a padded crop as JPEG bytes.

    Uses OpenCV Haar cascade frontal face detector. Falls back to the
    upper-center portrait crop if no face is detected.

    Args:
        image_bytes: Source image bytes (any PIL-readable format).
        padding:     Fractional padding added around the detected bounding box.

    Returns:
        JPEG bytes of the cropped face region.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    if not len(faces):
        logger.warning(
            "face_crop: no face detected — using upper-center fallback crop"
        )
        fallback = _fallback_portrait_crop(image)
        return _to_jpeg_bytes(fallback)

    # Pick the largest face by area
    fx, fy, fw, fh = max(faces, key=lambda f: int(f[2]) * int(f[3]))

    x1 = max(0, int(fx - padding * fw))
    y1 = max(0, int(fy - padding * fh))
    x2 = min(w, int(fx + fw * (1 + padding)))
    y2 = min(h, int(fy + fh * (1 + padding)))

    cropped = image.crop((x1, y1, x2, y2))

    logger.info(
        "face_crop: detected face (%dx%d) → crop bbox=(%d,%d,%d,%d) %dx%d",
        fw, fh, x1, y1, x2, y2, x2 - x1, y2 - y1,
    )
    return _to_jpeg_bytes(cropped)


def _to_jpeg_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def _fallback_portrait_crop(image: Image.Image) -> Image.Image:
    """Return an upper-center square crop to avoid clothing-heavy full image fallback."""
    w, h = image.size
    crop_size = max(128, int(min(w, h) * 0.65))
    crop_size = min(crop_size, w, h)

    x1 = max(0, (w - crop_size) // 2)
    top_anchor = int(h * 0.35) - (crop_size // 2)
    y1 = max(0, min(h - crop_size, top_anchor))
    return image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
