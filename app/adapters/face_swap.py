from __future__ import annotations

import asyncio
import logging
import uuid

from app.adapters.models import (
    FetchResult,
    GenerationStatus,
    ImageGenerationRequest,
    PollResult,
    SubmissionResult,
)
from app.adapters.openai_image import ImageGenerationAdapter

logger = logging.getLogger(__name__)


class MockFaceSwapAdapter(ImageGenerationAdapter):
    """
    A mock adapter for Face Swap (Stage 2 of the pipeline).
    In a real implementation, this would call InsightFace or Replicate.
    """

    def __init__(self, delay_seconds: float = 2.0) -> None:
        self._delay = delay_seconds
        self._results: dict[str, FetchResult] = {}

    async def submit(self, request: ImageGenerationRequest) -> SubmissionResult:
        logger.info("MockFaceSwap: Submitting face swap job...")
        vendor_job_id = f"fs_{uuid.uuid4().hex[:8]}"

        await asyncio.sleep(self._delay)

        # Mock: just pass through the poster_image_bytes
        # In reality, this merges request.poster_image_bytes and
        # request.user_image_bytes
        result_bytes = request.poster_image_bytes

        self._results[vendor_job_id] = FetchResult(
            external_job_id=vendor_job_id,
            status=GenerationStatus.SUCCEEDED,
            result_bytes=result_bytes,
            mime_type="image/jpeg" if request.output_format == "jpeg" else "image/png",
            raw_metadata={"mock": True, "stage": "face_swap"},
        )
        return SubmissionResult(
            external_job_id=vendor_job_id,
            status=GenerationStatus.SUCCEEDED,
            raw_metadata={"mock": True},
        )

    async def poll(self, external_job_id: str) -> PollResult:
        cached = self._results.get(external_job_id)
        if cached is None:
            return PollResult(
                external_job_id=external_job_id,
                status=GenerationStatus.FAILED,
                raw_metadata={"error": "not found"},
            )
        return PollResult(
            external_job_id=external_job_id,
            status=cached.status,
            raw_metadata=cached.raw_metadata,
        )

    async def fetch_result(self, external_job_id: str) -> FetchResult:
        cached = self._results.get(external_job_id)
        if cached is None:
            raise ValueError(f"Job not found: {external_job_id}")
        return cached


class MockHarmonizationAdapter(ImageGenerationAdapter):
    """
    A mock adapter for Harmonization (Stage 3 of the pipeline).
    In a real implementation, this would do an img2img pass with low denoising.
    """

    def __init__(self, delay_seconds: float = 1.0) -> None:
        self._delay = delay_seconds
        self._results: dict[str, FetchResult] = {}

    async def submit(self, request: ImageGenerationRequest) -> SubmissionResult:
        logger.info("MockHarmonization: Submitting harmonization job...")
        vendor_job_id = f"harm_{uuid.uuid4().hex[:8]}"

        await asyncio.sleep(self._delay)

        # Mock: pass through the poster_image_bytes
        result_bytes = request.poster_image_bytes

        self._results[vendor_job_id] = FetchResult(
            external_job_id=vendor_job_id,
            status=GenerationStatus.SUCCEEDED,
            result_bytes=result_bytes,
            mime_type="image/jpeg" if request.output_format == "jpeg" else "image/png",
            raw_metadata={"mock": True, "stage": "harmonization"},
        )
        return SubmissionResult(
            external_job_id=vendor_job_id,
            status=GenerationStatus.SUCCEEDED,
            raw_metadata={"mock": True},
        )

    async def poll(self, external_job_id: str) -> PollResult:
        cached = self._results.get(external_job_id)
        if cached is None:
            return PollResult(
                external_job_id=external_job_id,
                status=GenerationStatus.FAILED,
                raw_metadata={"error": "not found"},
            )
        return PollResult(
            external_job_id=external_job_id,
            status=cached.status,
            raw_metadata=cached.raw_metadata,
        )

    async def fetch_result(self, external_job_id: str) -> FetchResult:
        cached = self._results.get(external_job_id)
        if cached is None:
            raise ValueError(f"Job not found: {external_job_id}")
        return cached
