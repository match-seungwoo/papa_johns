from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models import Job


class ImageGenerationAdapter(ABC):
    @abstractmethod
    async def submit(self, job: Job, prompt: str, style_image: bytes) -> str:
        """Submit style transfer request. Returns vendor job ID."""
        ...

    @abstractmethod
    async def poll(self, job: Job, vendor_job_id: str) -> str:
        """Poll vendor job status. Returns vendor status string."""
        ...

    @abstractmethod
    async def fetch_result(self, job: Job, vendor_job_id: str) -> bytes:
        """Fetch generated image bytes from vendor."""
        ...


class OpenAIImageAdapter(ImageGenerationAdapter):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def submit(self, job: Job, prompt: str, style_image: bytes) -> str:
        raise NotImplementedError("OpenAI adapter not implemented")

    async def poll(self, job: Job, vendor_job_id: str) -> str:
        raise NotImplementedError("OpenAI adapter not implemented")

    async def fetch_result(self, job: Job, vendor_job_id: str) -> bytes:
        raise NotImplementedError("OpenAI adapter not implemented")
