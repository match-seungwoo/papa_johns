from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import yaml

from app.adapters.face_crop import crop_face
from app.adapters.fal_faceswap import FALFaceSwapAdapter
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
        faceswap: FALFaceSwapAdapter | None = None,
    ) -> None:
        self._queue = queue
        self._storage = storage
        self._job_service = job_service
        self._adapters = adapters
        self._faceswap = faceswap

    async def process_once(self) -> bool:
        job = await self._queue.receive_job()
        if job is None:
            return False

        logger.info(
            "[%s] Dequeued — template=%s subject=%s",
            job.job_id,
            job.template_id,
            job.subject_category,
        )
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
        quality: str | None = None,
        size: str | None = None,
        apply_faceswap: bool = False,
    ) -> tuple[str, str]:
        """Run a single vendor adapter. Returns (vendor, result_url)."""
        logger.info("[%s][%s] Starting vendor", job.job_id, vendor)
        adapter = self._adapters.get(vendor)
        if adapter is None:
            raise ValueError(f"No adapter registered for vendor: {vendor}")

        logger.info(
            "[%s][%s] Downloading user image from S3 key=%s",
            job.job_id,
            vendor,
            job.input_s3_key,
        )
        user_image = await self._storage.download(job.input_s3_key)
        # Use separate crops for generation and faceswap.
        # Generation gets a tighter crop to minimize clothing leakage.
        user_face_for_gen = crop_face(user_image, padding=0.08)
        # Faceswap gets slightly wider context for stability.
        user_face_for_swap = crop_face(user_image, padding=0.15)
        logger.info(
            "[%s][%s] Downloaded %dB → gen_face %dB swap_face %dB",
            job.job_id,
            vendor,
            len(user_image),
            len(user_face_for_gen),
            len(user_face_for_swap),
        )

        request = ImageGenerationRequest(
            prompt=prompt,
            poster_image_bytes=style_image,
            user_image_bytes=user_face_for_gen,
            template_id=job.template_id,
            subject_category=job.subject_category.value,
            quality=quality,
            size=size,
        )

        logger.info(
            "[%s][%s] Submitting to adapter — prompt=%r",
            job.job_id,
            vendor,
            prompt[:80],
        )
        submission = await adapter.submit(request)
        logger.info(
            "[%s][%s] Submission complete — external_job_id=%s",
            job.job_id,
            vendor,
            submission.external_job_id,
        )

        external_job_id = submission.external_job_id or ""
        fetch = await adapter.fetch_result(external_job_id)
        logger.info(
            "[%s][%s] Fetch complete — status=%s", job.job_id, vendor, fetch.status
        )

        result_bytes = fetch.result_bytes
        if result_bytes is None:
            raise RuntimeError(
                f"Adapter '{vendor}' returned no image bytes for job {job.job_id}"
            )

        if apply_faceswap and self._faceswap is not None:
            logger.info(
                "[%s][%s] Applying faceswap — generated=%dB user_face=%dB",
                job.job_id,
                vendor,
                len(result_bytes),
                len(user_face_for_swap),
            )
            result_bytes = await self._faceswap.swap(result_bytes, user_face_for_swap)
            logger.info(
                "[%s][%s] Faceswap complete — result=%dB",
                job.job_id,
                vendor,
                len(result_bytes),
            )

        logger.info(
            "[%s][%s] Uploading result (%d bytes) to S3",
            job.job_id,
            vendor,
            len(result_bytes),
        )
        result_key = await self._storage.upload_result(job.job_id, result_bytes, vendor)
        result_url = self._storage.get_url(result_key)
        logger.info("[%s][%s] Upload complete — url=%s", job.job_id, vendor, result_url)
        return vendor, result_url

    async def _process_job(self, job: Job) -> None:
        logger.info("[%s] Loading recipe for template=%s", job.job_id, job.template_id)
        recipe = load_recipe(job.template_id)

        vendors: list[str] = recipe.get("vendors") or [recipe.get("vendor", "openai")]
        post_process: list[str] = recipe.get("post_process") or []
        apply_faceswap = "faceswap" in post_process
        logger.info(
            "[%s] Vendors: %s post_process: %s", job.job_id, vendors, post_process
        )

        ad_image_path = PROJECT_ROOT / recipe["ad_image"]
        style_image = ad_image_path.read_bytes()
        logger.info(
            "[%s] Style image loaded (%d bytes) from %s",
            job.job_id,
            len(style_image),
            ad_image_path,
        )

        prompt_template: str = recipe.get("prompt_template", "")
        prompt = prompt_template.format(subject_category=job.subject_category.value)
        quality: str | None = recipe.get("quality")
        size: str | None = recipe.get("size")
        logger.info(
            "[%s] Prompt: %r quality=%s size=%s",
            job.job_id, prompt[:120], quality, size,
        )

        tasks = [
            self._run_vendor(
                job, vendor, prompt, style_image, quality, size, apply_faceswap
            )
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
