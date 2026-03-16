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
_DETECT_URL = "https://openapi.akool.com/interface/detect-api/detect_faces"


def _image_content_type(data: bytes) -> tuple[str, str]:
    """Return (content_type, extension) from image magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", "png"
    return "image/jpeg", "jpg"


class AkoolFaceSwapAdapter:
    """Akool high-quality image face-swap adapter.

    Pipeline per swap call:
      1. Upload poster + user images to S3 → presigned URLs
      2. Detect face landmarks on both images in parallel (D)
      3. Submit faceswap with opts (D) → poll for completion
      4. Download and return result bytes
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
        user_image_bytes: bytes,
        job_id: str = "",
        swap_all_faces: bool = False,
    ) -> bytes:
        """Apply the user face onto the generated poster via Akool faceswap.

        Args:
            generated_image_bytes: Poster image from the generation step.
            user_image_bytes: Original user photo (full image, not pre-cropped).
            job_id: Used as a namespace for temp S3 objects.
            swap_all_faces: If True, replace every detected face in the poster.
                            If False (default), replace only the largest face.

        Returns:
            Final poster bytes with the user's face applied.
        """
        logger.info(
            "Akool faceswap — uploading images: generated=%dB user=%dB",
            len(generated_image_bytes),
            len(user_image_bytes),
        )

        user_ct, user_ext = _image_content_type(user_image_bytes)

        poster_key, face_key = await asyncio.gather(
            self._storage.upload_temp(
                job_id, generated_image_bytes, "akool_generated.png", "image/png"
            ),
            self._storage.upload_temp(
                job_id, user_image_bytes, f"akool_source.{user_ext}", user_ct
            ),
        )
        poster_url = self._storage.generate_presigned_url(poster_key)
        face_url = self._storage.generate_presigned_url(face_key)

        # D: detect landmarks on both images in parallel; failures are non-fatal
        poster_landmarks, user_opts = await self._detect_landmarks_pair(
            poster_url, face_url, swap_all_faces=swap_all_faces
        )

        face_count = len(poster_landmarks) if poster_landmarks else 1
        logger.info(
            "Akool faceswap — swap_all=%s detected_faces=%d user_opts=%s",
            swap_all_faces,
            face_count,
            bool(user_opts),
        )

        # Single V3 call: pass all target faces + poster as modifyImage at once.
        # V4 faceswapByImage ignores opts and always picks the dominant face,
        # so V3 specifyimage (which accepts modifyImage + targetImage[]) is used.
        result_url = await self._submit_and_poll(
            poster_url, face_url, poster_landmarks, user_opts
        )
        result_bytes = await self._download(result_url)
        logger.info(
            "Akool faceswap — complete result_url=%s result=%dB",
            result_url,
            len(result_bytes),
        )
        return result_bytes

    # ------------------------------------------------------------------
    # Download helper
    # ------------------------------------------------------------------

    async def _download(self, url: str) -> bytes:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(url)
        except Exception as exc:
            raise AdapterAPIError(f"Akool result download failed: {exc}") from exc

        if resp.status_code != 200:
            raise AdapterResponseError(f"Akool download HTTP {resp.status_code}")

        return resp.content

    # ------------------------------------------------------------------
    # D: Face detect helpers
    # ------------------------------------------------------------------

    async def _detect_landmarks_pair(
        self,
        poster_url: str,
        face_url: str,
        swap_all_faces: bool = False,
    ) -> tuple[list[str], str]:
        """Detect faces on both images concurrently.

        Returns:
            poster_landmarks: list of landmarks_str, one per detected face.
                              Empty list means detect failed → swap without opts.
            user_opts:        landmarks_str for the user's (single) face, or "".
        """
        results = await asyncio.gather(
            self._detect_faces(poster_url, single_face=not swap_all_faces),
            self._detect_faces(face_url, single_face=True),
            return_exceptions=True,
        )

        if not isinstance(results[0], list):
            logger.warning("Akool face detect (poster) failed: %s", results[0])
            poster_landmarks: list[str] = []
        else:
            poster_landmarks = results[0]

        if not isinstance(results[1], list):
            logger.warning("Akool face detect (source) failed: %s", results[1])
            user_opts = ""
        else:
            user_opts = results[1][0] if results[1] else ""

        return poster_landmarks, user_opts

    async def _detect_faces(self, image_url: str, single_face: bool) -> list[str]:
        """Call Akool Face Detect API. Returns list of landmarks_str (one per face)."""
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"single_face": single_face, "url": image_url}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(_DETECT_URL, json=payload, headers=headers)

        if resp.status_code != 200:
            raise AdapterAPIError(
                f"Akool face detect HTTP {resp.status_code}: {resp.text[:200]}"
            )

        body: dict[str, Any] = resp.json()
        error_code = body.get("error_code", -1)
        if error_code != 0:
            raise AdapterAPIError(
                f"Akool face detect error {error_code}: {body.get('error_msg')}"
            )

        faces_obj: dict[str, Any] = body.get("faces_obj") or {}
        landmarks: list[str] = []
        for key in sorted(faces_obj.keys(), key=lambda k: int(k)):
            frame: dict[str, Any] = faces_obj[key]
            ls: list[str] = frame.get("landmarks_str") or []
            regions: list[Any] = frame.get("region") or []
            for i, lm in enumerate(ls):
                region = regions[i] if i < len(regions) else None
                landmarks.append(lm)
                logger.info(
                    "Akool face detect — face[%s][%d] region=%s landmarks_prefix=%s",
                    key,
                    i,
                    region,
                    lm[:40],
                )

        logger.info(
            "Akool face detect — total=%d single_face=%s url=...%s",
            len(landmarks),
            single_face,
            image_url[-40:],
        )

        if not landmarks:
            raise AdapterAPIError("Akool face detect returned no landmarks")

        return landmarks

    # ------------------------------------------------------------------
    # Submit + poll
    # ------------------------------------------------------------------

    async def _submit_and_poll(
        self,
        poster_url: str,
        face_url: str,
        poster_landmarks: list[str],
        user_opts: str = "",
    ) -> str:
        # V3 specifyimage: modifyImage is the poster to edit;
        # targetImage lists which faces to replace (one entry per face with opts);
        # sourceImage lists the replacement face repeated to match.
        if poster_landmarks:
            target_items: list[dict[str, Any]] = [
                {"path": poster_url, "opts": opts} for opts in poster_landmarks
            ]
        else:
            target_items = [{"path": poster_url}]

        source_base: dict[str, Any] = {"path": face_url}
        if user_opts:
            source_base["opts"] = user_opts
        source_items: list[dict[str, Any]] = [
            source_base for _ in target_items
        ]

        payload: dict[str, Any] = {
            "modifyImage": poster_url,
            "targetImage": target_items,
            "sourceImage": source_items,
            "face_enhance": self._face_enhance,
        }
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        logger.info(
            "Akool faceswap — V3 specifyimage faces=%d", len(target_items)
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    f"{_BASE_URL_V3}/faceswap/highquality/specifyimage",
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
            request_id: str = data.get("_id") or ""

        # V3 specifyimage always returns a pre-allocated url before processing starts.
        # Always poll by _id to wait for actual completion.
        if not request_id:
            raise AdapterResponseError(
                f"Akool returned no request ID: {body}"
            )

        logger.info(
            "Akool faceswap — async mode, polling _id=%s (max %d attempts)",
            request_id,
            self._poll_max_attempts,
        )
        return await self._poll(request_id=request_id, headers=headers)

    async def _poll(self, request_id: str, headers: dict[str, str]) -> str:
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
                raise AdapterAPIError(f"Akool poll request failed: {exc}") from exc

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
