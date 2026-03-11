from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import yaml

from app.adapters.openai_image import ImageGenerationAdapter
from app.adapters.queue import QueueAdapter
from app.adapters.storage import StorageAdapter
from app.domain.models import Job, JobStatus
from app.services.job_service import JobService

logger = logging.getLogger(__name__)

RECIPES_DIR = Path(__file__).parent.parent.parent / "configs" / "recipes"
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_recipe(template_id: str) -> dict[str, Any]:
    path = RECIPES_DIR / f"{template_id}.yaml"
    with open(path) as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


class Worker:
    def __init__(
        self,
        queue: QueueAdapter,
        storage: StorageAdapter,
        job_service: JobService,
        adapters: dict[str, ImageGenerationAdapter],
    ) -> None:
        self._queue = queue
        self._storage = storage
        self._job_service = job_service
        self._adapters = adapters

    async def process_once(self) -> bool:
        job = await self._queue.receive_job()
        if job is None:
            return False

        logger.info("Processing job %s (template=%s)", job.job_id, job.template_id)
        await self._job_service.update_job_status(job.job_id, JobStatus.RUNNING)
        try:
            await self._process_job(job)
        except Exception:
            logger.exception("Job %s failed", job.job_id)
            await self._job_service.update_job_status(job.job_id, JobStatus.FAILED)

        return True

    async def _run_vendor(
        self,
        job: Job,
        vendor: str,
        prompt: str,
        style_image: bytes,
    ) -> tuple[str, str]:
        """Run a single vendor adapter. Returns (vendor, result_url)."""
        adapter = self._adapters.get(vendor)
        if adapter is None:
            raise ValueError(f"No adapter registered for vendor: {vendor}")

        vendor_job_id = await adapter.submit(job, prompt, style_image)
        result_bytes = await adapter.fetch_result(job, vendor_job_id)
        result_key = await self._storage.upload_result(job.job_id, result_bytes, vendor)
        return vendor, self._storage.get_url(result_key)

    async def _process_job(self, job: Job) -> None:
        recipe = load_recipe(job.template_id)

        vendors: list[str] = recipe.get("vendors") or [recipe.get("vendor", "openai")]

        ad_image_path = PROJECT_ROOT / recipe["ad_image"]
        style_image = ad_image_path.read_bytes()

        prompt_template: str = recipe.get("prompt_template", "")
        prompt = prompt_template.format(subject_category=job.subject_category.value)

        tasks = [
            self._run_vendor(job, vendor, prompt, style_image)
            for vendor in vendors
        ]
        results = await asyncio.gather(*tasks)
        result_urls = dict(results)

        await self._job_service.update_job_status(
            job.job_id, JobStatus.SUCCEEDED, result_urls=result_urls
        )
        logger.info("Job %s completed: %s", job.job_id, result_urls)

    async def run(self) -> None:
        logger.info("Worker started, polling SQS...")
        while True:
            processed = await self.process_once()
            if not processed:
                await asyncio.sleep(1)
