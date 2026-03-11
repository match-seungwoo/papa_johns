from __future__ import annotations

import uuid
from datetime import UTC, datetime

from app.adapters.queue import QueueAdapter
from app.adapters.storage import StorageAdapter
from app.domain.models import Job, JobStatus, SpeciesHint, SubjectCategory


class JobService:
    def __init__(self, storage: StorageAdapter, queue: QueueAdapter) -> None:
        self._storage = storage
        self._queue = queue
        self._jobs: dict[str, Job] = {}

    async def create_job(
        self,
        template_id: str,
        subject_category: SubjectCategory,
        image_bytes: bytes,
        content_type: str,
        species_hint: SpeciesHint | None = None,
    ) -> Job:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        input_s3_key = await self._storage.upload_input(
            job_id, image_bytes, content_type
        )

        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            template_id=template_id,
            subject_category=subject_category,
            species_hint=species_hint,
            input_s3_key=input_s3_key,
            created_at=datetime.now(UTC),
        )
        self._jobs[job_id] = job
        await self._queue.enqueue_job(job)
        return job

    async def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        result_url: str | None = None,
        result_urls: dict[str, str] | None = None,
    ) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        updated = job.model_copy(
            update={"status": status, "result_url": result_url, "result_urls": result_urls}
        )
        self._jobs[job_id] = updated
        return updated
