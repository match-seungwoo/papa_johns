from __future__ import annotations

import logging
import os
from typing import Any

import fal_client
import httpx

from app.adapters.exceptions import (
    AdapterAPIError,
    AdapterConfigError,
    AdapterResponseError,
    AdapterTimeoutError,
)

logger = logging.getLogger(__name__)


class FALFaceSwapAdapter:
    """FAL AI face-swap adapter for post-processing generated poster images.

    Two-step pipeline:
      1. Upload both images to FAL CDN
      2. Call fal-ai/face-swap to replace the face in the generated poster
         with the user's actual face
    """

    def __init__(self, api_key: str, model: str = "fal-ai/face-swap") -> None:
        if not api_key:
            raise AdapterConfigError(
                "FAL API key is missing. Set the FAL_KEY environment variable."
            )
        os.environ["FAL_KEY"] = api_key
        self._model = model

    async def swap(
        self, generated_image_bytes: bytes, user_face_bytes: bytes
    ) -> bytes:
        """Replace the face in generated_image with the face from user_face.

        Args:
            generated_image_bytes: Poster image produced by the generation step.
            user_face_bytes: Original user photo providing the face identity.

        Returns:
            Final poster image bytes with the user's face swapped in.
        """
        logger.info(
            "FAL faceswap — uploading images: generated=%dB user_face=%dB",
            len(generated_image_bytes),
            len(user_face_bytes),
        )

        try:
            generated_url, face_url = await _upload_pair(
                generated_image_bytes, user_face_bytes
            )
        except Exception as exc:
            raise AdapterAPIError(f"FAL image upload failed: {exc}") from exc

        logger.info(
            "FAL faceswap — submitting to %s source=%s swap=%s",
            self._model,
            generated_url,
            face_url,
        )

        try:
            result: dict[str, Any] = await fal_client.subscribe_async(
                self._model,
                arguments={
                    "source_image_url": generated_url,
                    "swap_image_url": face_url,
                },
            )
        except TimeoutError as exc:
            raise AdapterTimeoutError(f"FAL faceswap timed out: {exc}") from exc
        except Exception as exc:
            raise AdapterAPIError(f"FAL faceswap API error: {exc}") from exc

        image_url: str | None = (result.get("image") or {}).get("url")
        if not image_url:
            raise AdapterResponseError(
                f"FAL faceswap response missing image URL: {result}"
            )

        logger.info("FAL faceswap — downloading result from %s", image_url)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(image_url)
        except Exception as exc:
            raise AdapterAPIError(
                f"FAL faceswap result download failed: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise AdapterResponseError(
                f"FAL faceswap download HTTP {resp.status_code}"
            )

        logger.info(
            "FAL faceswap — complete, result=%dB", len(resp.content)
        )
        return resp.content


async def _upload_pair(
    generated_bytes: bytes, face_bytes: bytes
) -> tuple[str, str]:
    """Upload both images to FAL CDN concurrently and return their URLs."""
    import asyncio

    generated_url, face_url = await asyncio.gather(
        fal_client.upload_async(generated_bytes, "image/png"),
        fal_client.upload_async(face_bytes, "image/jpeg"),
    )
    return str(generated_url), str(face_url)
