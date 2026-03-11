from __future__ import annotations

from app.adapters.openai_image import ImageGenerationAdapter
from app.domain.models import Job


class BFLImageAdapter(ImageGenerationAdapter):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def submit(self, job: Job, prompt: str, style_image: bytes) -> str:
        raise NotImplementedError("BFL adapter not implemented")

    async def poll(self, job: Job, vendor_job_id: str) -> str:
        raise NotImplementedError("BFL adapter not implemented")

    async def fetch_result(self, job: Job, vendor_job_id: str) -> bytes:
        raise NotImplementedError("BFL adapter not implemented")
