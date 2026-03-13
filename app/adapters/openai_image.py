from __future__ import annotations

import asyncio
import base64
import io
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any

import openai

from app.adapters.exceptions import (
    AdapterAPIError,
    AdapterConfigError,
    AdapterRequestError,
    AdapterResponseError,
    AdapterTimeoutError,
)
from app.adapters.models import (
    FetchResult,
    GenerationStatus,
    ImageGenerationRequest,
    PollResult,
    SubmissionResult,
)
from app.utils.face import create_face_mask_rgba, crop_face


class ImageGenerationAdapter(ABC):
    @abstractmethod
    async def submit(self, request: ImageGenerationRequest) -> SubmissionResult:
        """Submit image generation request.

        Returns SubmissionResult with tracking metadata.
        """
        ...

    @abstractmethod
    async def poll(self, external_job_id: str) -> PollResult:
        """Poll vendor job status. Returns normalized PollResult."""
        ...

    @abstractmethod
    async def fetch_result(self, external_job_id: str) -> FetchResult:
        """Fetch generated image. Returns normalized FetchResult with bytes."""
        ...


logger = logging.getLogger(__name__)


class OpenAIImageAdapter(ImageGenerationAdapter):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-image-1",
        timeout: float = 120.0,
        max_retries: int = 2,
        org_id: str = "",
    ) -> None:
        if not api_key:
            raise AdapterConfigError(
                "OpenAI API key is missing. "
                "Set the OPENAI_API_KEY environment variable."
            )
        self._api_key = api_key
        self._org_id = org_id or None
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        # In-memory result cache: external_job_id -> FetchResult.
        # OpenAI images.edit() is synchronous; results are stored here after submit()
        # so that fetch_result() can return them without a second API call.
        self._results: dict[str, FetchResult] = {}

    def _build_client(self) -> openai.OpenAI:
        """Build and return a configured OpenAI client."""
        return openai.OpenAI(
            api_key=self._api_key,
            organization=self._org_id,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    def _call_edit(
        self,
        images: list[tuple[str, io.BytesIO, str]],
        prompt: str,
        size: str,
        quality: str | None,
    ) -> bytes:
        """Synchronous wrapper around OpenAI images.edit(). Returns raw image bytes."""
        client = self._build_client()
        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "image": images,
            "prompt": prompt,
            "n": 1,
            "size": size,
        }
        if quality:
            call_kwargs["quality"] = quality

        try:
            response = client.images.edit(**call_kwargs)
        except openai.AuthenticationError as exc:
            raise AdapterConfigError(f"OpenAI authentication failed: {exc}") from exc
        except openai.BadRequestError as exc:
            raise AdapterRequestError(f"OpenAI rejected the request: {exc}") from exc
        except openai.APITimeoutError as exc:
            raise AdapterTimeoutError(f"OpenAI request timed out: {exc}") from exc
        except openai.APIError as exc:
            raise AdapterAPIError(f"OpenAI API error: {exc}") from exc

        if not response.data:
            raise AdapterResponseError("OpenAI returned an empty data list")

        image_item = response.data[0]
        b64_data: str | None = getattr(image_item, "b64_json", None)
        url: str | None = getattr(image_item, "url", None)

        if b64_data:
            try:
                return base64.b64decode(b64_data)
            except Exception as exc:
                raise AdapterResponseError(
                    f"Failed to base64-decode OpenAI image response: {exc}"
                ) from exc
        if url:
            import urllib.request

            with urllib.request.urlopen(url) as resp:  # noqa: S310
                return resp.read()  # type: ignore[no-any-return]
        raise AdapterResponseError(
            "OpenAI response missing both b64_json and url fields."
        )

    def _step1_generate(self, request: ImageGenerationRequest) -> bytes:
        """Step 1: generate poster character from the template ad image."""
        size = request.size or "1024x1024"
        logger.info(
            "Step1 generate — model=%s poster=%dB size=%s prompt=%r",
            self._model,
            len(request.poster_image_bytes),
            size,
            request.prompt[:80],
        )
        return self._call_edit(
            images=[
                ("poster.png", io.BytesIO(request.poster_image_bytes), "image/png"),
            ],
            prompt=request.prompt,
            size=size,
            quality=request.quality,
        )

    def _step2_inpaint(
        self,
        base_poster_bytes: bytes,
        request: ImageGenerationRequest,
    ) -> bytes | None:
        """Step 2: replace the face region in the generated poster with the user's face.

        Returns inpainted image bytes, or None if face detection fails (caller should
        fall back to the Step 1 result).
        """
        assert request.face_inpaint_prompt  # caller must check before calling

        masked = create_face_mask_rgba(base_poster_bytes)
        if masked is None:
            logger.warning("Step2 skipped: no face detected in generated poster")
            return None

        face_crop = crop_face(request.user_image_bytes)
        if face_crop is None:
            logger.warning("Step2 skipped: no face detected in user image")
            return None

        size = request.size or "1024x1024"
        logger.info(
            "Step2 inpaint — masked=%dB face_crop=%dB size=%s prompt=%r",
            len(masked),
            len(face_crop),
            size,
            request.face_inpaint_prompt[:80],
        )
        return self._call_edit(
            images=[
                ("masked_poster.png", io.BytesIO(masked), "image/png"),
                ("user_face.jpg", io.BytesIO(face_crop), "image/jpeg"),
            ],
            prompt=request.face_inpaint_prompt,
            size=size,
            quality=request.quality,
        )

    def _call_openai(self, request: ImageGenerationRequest) -> tuple[bytes, str]:
        """
        Orchestrate the 2-step face-preserving pipeline.

        Step 1 — generate base poster from the template ad image.
        Step 2 — inpaint the face region using the user's face (only when
                  face_inpaint_prompt is set and face detection succeeds).
        Falls back to the Step 1 result when Step 2 is skipped.
        """
        if not request.prompt:
            raise AdapterRequestError("prompt must not be empty")
        if not request.poster_image_bytes:
            raise AdapterRequestError("poster_image_bytes must not be empty")
        if not request.user_image_bytes:
            raise AdapterRequestError("user_image_bytes must not be empty")

        base_poster = self._step1_generate(request)

        if request.face_inpaint_prompt:
            result = self._step2_inpaint(base_poster, request)
            if result is not None:
                logger.info("Step2 inpaint succeeded")
                return result, "image/png"
            logger.warning("Step2 failed — falling back to Step 1 result")

        mime_type = "image/jpeg" if request.output_format == "jpeg" else "image/png"
        return base_poster, mime_type

    async def submit(self, request: ImageGenerationRequest) -> SubmissionResult:
        """
        Submit an image generation request to OpenAI.

        OpenAI images.edit() is synchronous — the full API call executes here
        and the result is cached internally for retrieval via fetch_result().
        The returned SubmissionResult carries an external_job_id for subsequent calls.
        """
        vendor_job_id = str(uuid.uuid4())
        raw_metadata: dict[str, Any] = {
            "model": self._model,
            "vendor_job_id": vendor_job_id,
        }

        try:
            image_bytes, mime_type = await asyncio.to_thread(self._call_openai, request)
        except (
            AdapterConfigError,
            AdapterRequestError,
            AdapterTimeoutError,
            AdapterAPIError,
            AdapterResponseError,
        ):
            raise
        except Exception as exc:
            raise AdapterAPIError(f"Unexpected error calling OpenAI: {exc}") from exc

        self._results[vendor_job_id] = FetchResult(
            external_job_id=vendor_job_id,
            status=GenerationStatus.SUCCEEDED,
            result_bytes=image_bytes,
            mime_type=mime_type,
            raw_metadata=raw_metadata,
        )
        return SubmissionResult(
            external_job_id=vendor_job_id,
            status=GenerationStatus.SUCCEEDED,
            raw_metadata=raw_metadata,
        )

    async def poll(self, external_job_id: str) -> PollResult:
        """
        Return the status of a previously submitted job.

        Since OpenAI images.edit() is synchronous, the status is already final after
        submit() returns. This method exists to preserve the adapter contract for
        consistency with asynchronous vendors (e.g., BFL).
        """
        cached = self._results.get(external_job_id)
        if cached is None:
            return PollResult(
                external_job_id=external_job_id,
                status=GenerationStatus.FAILED,
                raw_metadata={"error": "unknown job id"},
            )
        return PollResult(
            external_job_id=external_job_id,
            status=cached.status,
            raw_metadata=cached.raw_metadata,
        )

    async def fetch_result(self, external_job_id: str) -> FetchResult:
        """Retrieve the generated image for a completed job."""
        cached = self._results.get(external_job_id)
        if cached is None:
            raise AdapterResponseError(
                f"No result found for job '{external_job_id}'. "
                "Call submit() before fetch_result()."
            )
        return cached
