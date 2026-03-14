from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import httpx

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
from app.adapters.openai_image import ImageGenerationAdapter

logger = logging.getLogger(__name__)

_DEFAULT_BFL_BASE_URL = "https://api.bfl.ai/v1"


def _map_bfl_status(bfl_status: str) -> GenerationStatus:
    normalized = bfl_status.lower()
    if normalized == "ready":
        return GenerationStatus.SUCCEEDED
    if normalized in ("error", "request moderated", "content moderated"):
        return GenerationStatus.FAILED
    return GenerationStatus.RUNNING


class BFLImageAdapter(ImageGenerationAdapter):
    """BFL FLUX adapter with IP-Adapter face conditioning.

    Calls flux.generate(template_style, ip_adapter_face=user_face) semantics:
    - ``prompt`` encodes template style intent
    - ``image_prompt`` (IP-Adapter) carries the user face for identity preservation
    """

    def __init__(
        self,
        api_key: str,
        model: str = "flux-pro-1.1-ultra",
        base_url: str = _DEFAULT_BFL_BASE_URL,
        image_prompt_strength: float = 0.15,
        submit_timeout_seconds: float = 60.0,
        submit_max_retries: int = 2,
        poll_interval: float = 2.0,
        poll_max_attempts: int = 60,
    ) -> None:
        if not api_key:
            raise AdapterConfigError(
                "BFL API key is missing. Set the BFL_API_KEY environment variable."
            )
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._image_prompt_strength = image_prompt_strength
        self._submit_timeout_seconds = submit_timeout_seconds
        self._submit_max_retries = submit_max_retries
        self._poll_interval = poll_interval
        self._poll_max_attempts = poll_max_attempts

    def _headers(self) -> dict[str, str]:
        return {
            "x-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def submit(self, request: ImageGenerationRequest) -> SubmissionResult:
        """Submit to BFL FLUX with IP-Adapter face conditioning.

        ``request.user_image_bytes`` is passed as ``image_prompt`` (IP-Adapter)
        so the model conditions on the user's facial identity during generation.
        ``request.poster_image_bytes`` provides the template style context via
        the text prompt; the recipe prompt already encodes the visual style intent.
        """
        if not request.prompt:
            raise AdapterRequestError("prompt must not be empty")
        if not request.user_image_bytes:
            raise AdapterRequestError("user_image_bytes must not be empty")

        # IP-Adapter face conditioning: encode user face as base64
        face_b64 = base64.b64encode(request.user_image_bytes).decode()

        body: dict[str, Any] = {
            "prompt": request.prompt,
            "image_prompt": face_b64,
            "image_prompt_strength": self._image_prompt_strength,
            "output_format": request.output_format,
        }

        if request.size:
            try:
                w_str, h_str = request.size.split("x")
                body["width"] = int(w_str)
                body["height"] = int(h_str)
            except ValueError:
                pass  # omit if malformed; BFL uses its own default

        url = f"{self._base_url}/{self._model}"
        logger.info(
            "BFL submit — model=%s ip_adapter_face=%dB prompt=%r",
            self._model,
            len(request.user_image_bytes),
            request.prompt[:80],
        )

        timeout_error: httpx.TimeoutException | None = None
        for attempt in range(1, self._submit_max_retries + 2):
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        connect=self._submit_timeout_seconds,
                        read=self._submit_timeout_seconds,
                        write=self._submit_timeout_seconds,
                        pool=5.0,
                    )
                ) as client:
                    response = await client.post(url, json=body, headers=self._headers())
                break
            except httpx.TimeoutException as exc:
                timeout_error = exc
                if attempt > self._submit_max_retries:
                    raise AdapterTimeoutError(
                        "BFL submit request timed out after "
                        f"{attempt} attempt(s) to {url}: {exc!r}"
                    ) from exc
                backoff_seconds = float(attempt)
                logger.warning(
                    "BFL submit timeout on attempt %d/%d for %s; retrying in %.1fs",
                    attempt,
                    self._submit_max_retries + 1,
                    url,
                    backoff_seconds,
                )
                await asyncio.sleep(backoff_seconds)
            except httpx.HTTPError as exc:
                raise AdapterAPIError(f"BFL submit HTTP error: {exc}") from exc
        else:
            # Defensive fallback; loop exits via break or raise.
            raise AdapterTimeoutError(
                f"BFL submit request timed out for {url}: {timeout_error!r}"
            )

        if response.status_code != 200:
            raise AdapterAPIError(
                f"BFL submit failed: HTTP {response.status_code} — {response.text}"
            )

        data: dict[str, Any] = response.json()
        external_job_id: str | None = data.get("id")
        if not external_job_id:
            raise AdapterResponseError(f"BFL submit response missing 'id': {data}")

        logger.info("BFL submit accepted — external_job_id=%s", external_job_id)
        return SubmissionResult(
            external_job_id=external_job_id,
            status=GenerationStatus.QUEUED,
            raw_metadata=data,
        )

    async def poll(self, external_job_id: str) -> PollResult:
        """Check BFL job status via GET /get_result."""
        url = f"{self._base_url}/get_result"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    url,
                    params={"id": external_job_id},
                    headers=self._headers(),
                )
        except httpx.TimeoutException as exc:
            raise AdapterTimeoutError(f"BFL poll request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise AdapterAPIError(f"BFL poll HTTP error: {exc}") from exc

        if response.status_code == 404:
            data_404: dict[str, Any] = response.json()
            status_text = str(data_404.get("status", "")).lower()
            if "task not found" in status_text:
                # BFL can briefly return 404 right after submit before the task
                # is fully visible to the result endpoint.
                return PollResult(
                    external_job_id=external_job_id,
                    status=GenerationStatus.RUNNING,
                    raw_metadata=data_404,
                )

        if response.status_code != 200:
            raise AdapterAPIError(
                f"BFL poll failed: HTTP {response.status_code} — {response.text}"
            )

        data: dict[str, Any] = response.json()
        bfl_status: str = data.get("status", "")
        return PollResult(
            external_job_id=external_job_id,
            status=_map_bfl_status(bfl_status),
            raw_metadata=data,
        )

    async def fetch_result(self, external_job_id: str) -> FetchResult:
        """Poll until the job is done, then download and return image bytes."""
        for attempt in range(1, self._poll_max_attempts + 1):
            poll = await self.poll(external_job_id)
            logger.info(
                "BFL poll [%d/%d] — id=%s status=%s",
                attempt,
                self._poll_max_attempts,
                external_job_id,
                poll.status,
            )

            if poll.status == GenerationStatus.FAILED:
                raise AdapterAPIError(
                    f"BFL job {external_job_id} failed: {poll.raw_metadata}"
                )

            if poll.status == GenerationStatus.SUCCEEDED:
                result_data: dict[str, Any] = poll.raw_metadata.get("result") or {}
                sample_url: str | None = result_data.get("sample")
                if not sample_url:
                    raise AdapterResponseError(
                        f"BFL result missing 'sample' URL: {poll.raw_metadata}"
                    )

                logger.info("BFL fetch — downloading result from %s", sample_url)
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        img_response = await client.get(sample_url)
                except httpx.HTTPError as exc:
                    raise AdapterAPIError(
                        f"Failed to download BFL result image: {exc}"
                    ) from exc

                if img_response.status_code != 200:
                    raise AdapterResponseError(
                        f"BFL result download failed: HTTP {img_response.status_code}"
                    )

                mime_type = (
                    "image/png" if sample_url.endswith(".png") else "image/jpeg"
                )
                return FetchResult(
                    external_job_id=external_job_id,
                    status=GenerationStatus.SUCCEEDED,
                    result_bytes=img_response.content,
                    mime_type=mime_type,
                    raw_metadata=poll.raw_metadata,
                )

            await asyncio.sleep(self._poll_interval)

        raise AdapterTimeoutError(
            f"BFL job {external_job_id} did not complete after "
            f"{self._poll_max_attempts} poll attempts "
            f"({self._poll_max_attempts * self._poll_interval:.0f}s)."
        )
