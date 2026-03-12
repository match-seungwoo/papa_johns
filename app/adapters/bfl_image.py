from __future__ import annotations

from app.adapters.models import (
    FetchResult,
    ImageGenerationRequest,
    PollResult,
    SubmissionResult,
)
from app.adapters.openai_image import ImageGenerationAdapter


class BFLImageAdapter(ImageGenerationAdapter):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def submit(self, request: ImageGenerationRequest) -> SubmissionResult:
        raise NotImplementedError("BFL adapter not implemented")

    async def poll(self, external_job_id: str) -> PollResult:
        raise NotImplementedError("BFL adapter not implemented")

    async def fetch_result(self, external_job_id: str) -> FetchResult:
        raise NotImplementedError("BFL adapter not implemented")
