from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.adapters.face_crop import crop_face


def _make_jpeg(w: int = 300, h: int = 300) -> bytes:
    img = Image.new("RGB", (w, h), color=(128, 100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_no_face_detected_returns_fallback_crop() -> None:
    original = _make_jpeg()
    with patch("app.adapters.face_crop._face_cascade") as mock_cascade:
        mock_cascade.detectMultiScale.return_value = ()
        result = crop_face(original)
    assert result != original
    img = Image.open(io.BytesIO(result))
    assert img.size[0] == img.size[1]


def test_face_detected_returns_cropped_jpeg() -> None:
    original = _make_jpeg(300, 300)
    # face at (60, 60), size 90x90
    faces = np.array([[60, 60, 90, 90]])
    with patch("app.adapters.face_crop._face_cascade") as mock_cascade:
        mock_cascade.detectMultiScale.return_value = faces
        result = crop_face(original, padding=0.0)

    img = Image.open(io.BytesIO(result))
    assert img.size == (90, 90)


def test_padding_expands_crop() -> None:
    original = _make_jpeg(300, 300)
    faces = np.array([[60, 60, 90, 90]])

    with patch("app.adapters.face_crop._face_cascade") as mock_cascade:
        mock_cascade.detectMultiScale.return_value = faces
        result_no_pad = crop_face(original, padding=0.0)

    with patch("app.adapters.face_crop._face_cascade") as mock_cascade:
        mock_cascade.detectMultiScale.return_value = faces
        result_pad = crop_face(original, padding=0.4)

    no_pad_size = Image.open(io.BytesIO(result_no_pad)).size
    pad_size = Image.open(io.BytesIO(result_pad)).size
    assert pad_size[0] > no_pad_size[0]
    assert pad_size[1] > no_pad_size[1]


def test_largest_face_selected() -> None:
    """When multiple faces are detected, the largest one is used."""
    original = _make_jpeg(400, 400)
    # small face: 40x40, large face: 160x160
    faces = np.array([[20, 20, 40, 40], [160, 160, 160, 160]])

    with patch("app.adapters.face_crop._face_cascade") as mock_cascade:
        mock_cascade.detectMultiScale.return_value = faces
        result = crop_face(original, padding=0.0)

    img = Image.open(io.BytesIO(result))
    assert img.size == (160, 160)


def test_crop_clamped_to_image_bounds() -> None:
    """Padded crop must not exceed image dimensions."""
    original = _make_jpeg(200, 200)
    # face near edge: x=150, size=40 → with padding would exceed 200
    faces = np.array([[150, 150, 40, 40]])

    with patch("app.adapters.face_crop._face_cascade") as mock_cascade:
        mock_cascade.detectMultiScale.return_value = faces
        result = crop_face(original, padding=0.4)

    img = Image.open(io.BytesIO(result))
    assert img.size[0] <= 200
    assert img.size[1] <= 200
