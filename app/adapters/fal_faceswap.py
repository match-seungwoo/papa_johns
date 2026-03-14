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
    """FAL AI face post-processing adapter.

    Supports two models:
      - fal-ai/face-swap : classic face-swap (source + swap image)
      - fal-ai/pulid     : identity-preserving generation (face ref + init image)
    """

    def __init__(self, api_key: str, model: str = "fal-ai/pulid") -> None:
        if not api_key:
            raise AdapterConfigError(
                "FAL API key is missing. Set the FAL_KEY environment variable."
            )
        os.environ["FAL_KEY"] = api_key
        self._model = model
        self._is_pulid = "pulid" in model

    async def swap(
        self, generated_image_bytes: bytes, user_face_bytes: bytes
    ) -> bytes:
        """Apply face identity from user_face onto the generated poster.

        Args:
            generated_image_bytes: Poster image from the generation step.
            user_face_bytes: Cropped user face providing identity.

        Returns:
            Final poster bytes with the user's face applied.
        """
        logger.info(
            "FAL [%s] — uploading images: generated=%dB user_face=%dB",
            self._model,
            len(generated_image_bytes),
            len(user_face_bytes),
        )

        try:
            generated_url, face_url = await _upload_pair(
                generated_image_bytes, user_face_bytes
            )
        except Exception as exc:
            raise AdapterAPIError(f"FAL image upload failed: {exc}") from exc

        arguments = self._build_arguments(generated_url, face_url)
        logger.info(
            "FAL [%s] — submitting, args=%s", self._model, list(arguments.keys())
        )

        try:
            result: dict[str, Any] = await fal_client.subscribe_async(
                self._model,
                arguments=arguments,
            )
        except TimeoutError as exc:
            raise AdapterTimeoutError(f"FAL [{self._model}] timed out: {exc}") from exc
        except Exception as exc:
            raise AdapterAPIError(f"FAL [{self._model}] API error: {exc}") from exc

        image_url = self._extract_result_url(result)
        if not image_url:
            raise AdapterResponseError(
                f"FAL [{self._model}] response missing image URL: {result}"
            )

        logger.info("FAL [%s] — downloading result from %s", self._model, image_url)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(image_url)
        except Exception as exc:
            raise AdapterAPIError(
                f"FAL [{self._model}] result download failed: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise AdapterResponseError(
                f"FAL [{self._model}] download HTTP {resp.status_code}"
            )

        logger.info(
            "FAL [%s] — complete, result=%dB", self._model, len(resp.content)
        )
        return resp.content

    def _build_arguments(
        self, generated_url: str, face_url: str
    ) -> dict[str, Any]:
        if self._is_pulid:
            return {
                "face_image_url": face_url,      # identity reference
                "image_url": generated_url,       # poster as init image (img2img)
                "prompt": "high quality portrait photo",
                "guidance_scale": 1.5,
                "num_inference_steps": 20,
            }
        # fal-ai/face-swap
        return {
            "source_image_url": generated_url,
            "swap_image_url": face_url,
        }

    def _extract_result_url(self, result: dict[str, Any]) -> str | None:
        if self._is_pulid:
            images = result.get("images") or []
            return images[0].get("url") if images else None
        # fal-ai/face-swap
        return (result.get("image") or {}).get("url")


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
