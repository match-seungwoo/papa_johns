from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.adapters.exceptions import (
    AdapterAPIError,
    AdapterConfigError,
    AdapterResponseError,
    AdapterTimeoutError,
)
from app.adapters.storage import StorageAdapter

logger = logging.getLogger(__name__)

_BASE_URL_V3 = "https://openapi.akool.com/api/open/v3"
_BASE_URL_V4 = "https://openapi.akool.com/api/open/v4"


class AkoolFaceSwapAdapter:
    """Akool high-quality image face-swap adapter.

    Uploads both images to S3 (presigned URLs) then calls the Akool
    specifyimage endpoint.  Polls for completion when the initial response
    does not contain a result URL.
    """

    def __init__(
        self,
        api_key: str,
        storage: StorageAdapter,
        face_enhance: int = 1,
        poll_interval: float = 2.0,
        poll_max_attempts: int = 30,
    ) -> None:
        if not api_key:
            raise AdapterConfigError(
                "Akool API key is missing. Set the AKOOL_API_KEY environment variable."
            )
        self._api_key = api_key
        self._storage = storage
        self._face_enhance = face_enhance
        self._poll_interval = poll_interval
        self._poll_max_attempts = poll_max_attempts

    async def swap(
        self,
        generated_image_bytes: bytes,
        user_face_bytes: bytes,
        job_id: str = "",
    ) -> bytes:
        """Apply the user face onto the generated poster via Akool faceswap.

        Args:
            generated_image_bytes: Poster image from the generation step.
            user_face_bytes: Cropped user face providing identity.
            job_id: Used as a namespace for temp S3 objects.

        Returns:
            Final poster bytes with the user's face applied.
        """
        logger.info(
            "Akool faceswap — uploading images: generated=%dB user_face=%dB",
            len(generated_image_bytes),
            len(user_face_bytes),
        )

        poster_key, face_key = await asyncio.gather(
            self._storage.upload_temp(
                job_id, generated_image_bytes, "akool_generated.png", "image/png"
            ),
            self._storage.upload_temp(
                job_id, user_face_bytes, "akool_face.jpg", "image/jpeg"
            ),
        )
        poster_url = self._storage.generate_presigned_url(poster_key)
        face_url = self._storage.generate_presigned_url(face_key)

        logger.info("Akool faceswap — submitting request")
        result_url = await self._submit_and_poll(poster_url, face_url)

        logger.info("Akool faceswap — downloading result from %s", result_url)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(result_url)
        except Exception as exc:
            raise AdapterAPIError(
                f"Akool result download failed: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise AdapterResponseError(
                f"Akool download HTTP {resp.status_code}"
            )

        logger.info("Akool faceswap — complete, result=%dB", len(resp.content))
        return resp.content

    async def _submit_and_poll(self, poster_url: str, face_url: str) -> str:
        payload: dict[str, Any] = {
            # V4 Face Swap Pro: single-face path, no opts required.
            "targetImage": [{"path": poster_url}],
            "sourceImage": [{"path": face_url}],
            "model_name": "akool_faceswap_image_hq",
            "webhookUrl": "",
            "face_enhance": bool(self._face_enhance),
        }
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    f"{_BASE_URL_V4}/faceswap/faceswapByImage",
                    json=payload,
                    headers=headers,
                )
            except Exception as exc:
                raise AdapterAPIError(f"Akool API request failed: {exc}") from exc

            if resp.status_code != 200:
                raise AdapterAPIError(
                    f"Akool API HTTP {resp.status_code}: {resp.text[:200]}"
                )

            body: dict[str, Any] = resp.json()
            if body.get("code") != 1000:
                raise AdapterAPIError(
                    f"Akool API error code {body.get('code')}: {body.get('msg')}"
                )

            data: dict[str, Any] = body.get("data") or {}
            result_url: str = data.get("url") or ""
            request_id: str = data.get("_id") or ""

        if result_url:
            return result_url

        if not request_id:
            raise AdapterResponseError(
                f"Akool returned neither a result URL nor a request ID: {body}"
            )

        logger.info(
            "Akool faceswap — async mode, polling _id=%s (max %d attempts)",
            request_id,
            self._poll_max_attempts,
        )
        return await self._poll(client=None, request_id=request_id, headers=headers)

    async def _poll(
        self,
        client: httpx.AsyncClient | None,
        request_id: str,
        headers: dict[str, str],
    ) -> str:
        for attempt in range(1, self._poll_max_attempts + 1):
            await asyncio.sleep(self._poll_interval)
            try:
                async with httpx.AsyncClient(timeout=30.0) as poll_client:
                    resp = await poll_client.get(
                        f"{_BASE_URL_V3}/faceswap/result/listbyids",
                        params={"_ids": request_id},
                        headers=headers,
                    )
            except Exception as exc:
                raise AdapterAPIError(
                    f"Akool poll request failed: {exc}"
                ) from exc

            if resp.status_code != 200:
                raise AdapterAPIError(
                    f"Akool poll HTTP {resp.status_code}: {resp.text[:200]}"
                )

            body: dict[str, Any] = resp.json()
            if body.get("code") != 1000:
                raise AdapterAPIError(
                    f"Akool poll error code {body.get('code')}: {body.get('msg')}"
                )

            data: dict[str, Any] = body.get("data") or {}
            results: list[dict[str, Any]] = data.get("result") or []
            if not results:
                logger.debug(
                    "Akool faceswap — poll attempt %d/%d: empty result list",
                    attempt,
                    self._poll_max_attempts,
                )
                continue

            item = results[0]
            status = int(item.get("faceswap_status") or 0)
            result_url: str = item.get("url") or ""
            if status == 4:
                raise AdapterAPIError(
                    f"Akool faceswap failed for request_id={request_id}: {item}"
                )

            if status == 3 and result_url:
                logger.info(
                    "Akool faceswap — poll complete after %d attempt(s)", attempt
                )
                return result_url

            logger.debug(
                "Akool faceswap — poll attempt %d/%d: status=%s not ready",
                attempt,
                self._poll_max_attempts,
                status,
            )

        raise AdapterTimeoutError(
            f"Akool faceswap timed out after {self._poll_max_attempts} poll attempts "
            f"(request_id={request_id})"
        )
