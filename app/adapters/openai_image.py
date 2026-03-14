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

    def _call_openai(self, request: ImageGenerationRequest) -> tuple[bytes, str]:
        """
        Synchronously call OpenAI images.edit() with gpt-image-1.
        poster image + user image are passed as a list of file-like objects.
        Intended to run inside asyncio.to_thread.

        Returns (image_bytes, mime_type).
        Converts OpenAI SDK exceptions to internal AdapterError subtypes.
        """
        if not request.prompt:
            raise AdapterRequestError("prompt must not be empty")
        if not request.poster_image_bytes:
            raise AdapterRequestError("poster_image_bytes must not be empty")
        if not request.user_image_bytes:
            raise AdapterRequestError("user_image_bytes must not be empty")

        client = self._build_client()

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "image": [
                ("poster.png", io.BytesIO(request.poster_image_bytes), "image/png"),
                ("user.jpg", io.BytesIO(request.user_image_bytes), "image/jpeg"),
            ],
            "prompt": request.prompt,
            "n": 1,
        }
        if request.size:
            call_kwargs["size"] = request.size
        if request.quality:
            call_kwargs["quality"] = request.quality

        logger.info(
            "OpenAI images.edit — model=%s poster=%dB user=%dB size=%s",
            self._model,
            len(request.poster_image_bytes),
            len(request.user_image_bytes),
            call_kwargs.get("size"),
        )
        try:
            response = client.images.edit(**call_kwargs)
            logger.info(
                "OpenAI images.edit — response received, data count=%d",
                len(response.data or []),
            )
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
                image_bytes = base64.b64decode(b64_data)
            except Exception as exc:
                raise AdapterResponseError(
                    f"Failed to base64-decode OpenAI image response: {exc}"
                ) from exc
        elif url:
            import urllib.request
            with urllib.request.urlopen(url) as resp:  # noqa: S310
                image_bytes = resp.read()
        else:
            raise AdapterResponseError(
                "OpenAI response missing both b64_json and url fields."
            )

        mime_type = "image/jpeg" if request.output_format == "jpeg" else "image/png"
        return image_bytes, mime_type

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
