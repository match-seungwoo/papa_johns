from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import yaml

from app.adapters.models import ImageGenerationRequest
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

        logger.info("[%s] Dequeued — template=%s subject=%s", job.job_id, job.template_id, job.subject_category)
        await self._job_service.update_job_status(job.job_id, JobStatus.RUNNING)
        logger.info("[%s] Status → RUNNING", job.job_id)
        try:
            await self._process_job(job)
        except Exception:
            logger.exception("[%s] Unhandled exception — marking FAILED", job.job_id)
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
        logger.info("[%s][%s] Starting vendor", job.job_id, vendor)
        adapter = self._adapters.get(vendor)
        if adapter is None:
            raise ValueError(f"No adapter registered for vendor: {vendor}")

        logger.info("[%s][%s] Downloading user image from S3 key=%s", job.job_id, vendor, job.input_s3_key)
        user_image = await self._storage.download(job.input_s3_key)
        logger.info("[%s][%s] User image downloaded (%d bytes)", job.job_id, vendor, len(user_image))

        request = ImageGenerationRequest(
            prompt=prompt,
            poster_image_bytes=style_image,
            user_image_bytes=user_image,
            template_id=job.template_id,
            subject_category=job.subject_category.value,
        )

        logger.info("[%s][%s] Submitting to adapter — prompt=%r", job.job_id, vendor, prompt[:80])
        submission = await adapter.submit(request)
        logger.info("[%s][%s] Submission complete — external_job_id=%s", job.job_id, vendor, submission.external_job_id)

        external_job_id = submission.external_job_id or ""
        fetch = await adapter.fetch_result(external_job_id)
        logger.info("[%s][%s] Fetch complete — status=%s", job.job_id, vendor, fetch.status)

        result_bytes = fetch.result_bytes
        if result_bytes is None:
            raise RuntimeError(
                f"Adapter '{vendor}' returned no image bytes for job {job.job_id}"
            )

        logger.info("[%s][%s] Uploading result (%d bytes) to S3", job.job_id, vendor, len(result_bytes))
        result_key = await self._storage.upload_result(job.job_id, result_bytes, vendor)
        result_url = self._storage.get_url(result_key)
        logger.info("[%s][%s] Upload complete — url=%s", job.job_id, vendor, result_url)
        return vendor, result_url

    async def _process_job(self, job: Job) -> None:
        logger.info("[%s] Loading recipe for template=%s", job.job_id, job.template_id)
        recipe = load_recipe(job.template_id)

        vendors: list[str] = recipe.get("vendors") or [recipe.get("vendor", "openai")]
        logger.info("[%s] Vendors: %s", job.job_id, vendors)

        ad_image_path = PROJECT_ROOT / recipe["ad_image"]
        style_image = ad_image_path.read_bytes()
        logger.info("[%s] Style image loaded (%d bytes) from %s", job.job_id, len(style_image), ad_image_path)

        prompt_template: str = recipe.get("prompt_template", "")
        prompt = prompt_template.format(subject_category=job.subject_category.value)
        logger.info("[%s] Prompt: %r", job.job_id, prompt[:120])

        tasks = [
            self._run_vendor(job, vendor, prompt, style_image)
            for vendor in vendors
        ]
        results = await asyncio.gather(*tasks)
        result_urls = dict(results)

        await self._job_service.update_job_status(
            job.job_id, JobStatus.SUCCEEDED, result_urls=result_urls
        )
        logger.info("[%s] Status → SUCCEEDED — urls=%s", job.job_id, result_urls)

    async def run(self) -> None:
        logger.info("Worker started, polling SQS...")
        while True:
            processed = await self.process_once()
            if not processed:
                await asyncio.sleep(1)
