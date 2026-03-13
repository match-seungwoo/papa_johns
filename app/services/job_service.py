from __future__ import annotations

import uuid
from datetime import UTC, datetime

from app.adapters.job_store import JobStoreAdapter
from app.adapters.queue import QueueAdapter
from app.adapters.storage import StorageAdapter
from app.domain.models import Job, JobStatus, SpeciesHint, SubjectCategory


class JobService:
    def __init__(
        self,
        storage: StorageAdapter,
        queue: QueueAdapter,
        job_store: JobStoreAdapter,
    ) -> None:
        self._storage = storage
        self._queue = queue
        self._job_store = job_store

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
        await self._job_store.save(job)
        await self._queue.enqueue_job(job)
        return job

    async def get_job(self, job_id: str) -> Job | None:
        return await self._job_store.get(job_id)

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        result_url: str | None = None,
        result_urls: dict[str, str] | None = None,
    ) -> Job | None:
        job = await self._job_store.get(job_id)
        if job is None:
            return None

        now = datetime.now(UTC)
        updates: dict[str, object] = {"status": status}

        if status == JobStatus.RUNNING:
            updates["started_at"] = now
        elif status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            updates["completed_at"] = now

        if result_url is not None:
            updates["result_url"] = result_url
        if result_urls is not None:
            updates["result_urls"] = result_urls

        updated = job.model_copy(update=updates)
        await self._job_store.save(updated)
        return updated
