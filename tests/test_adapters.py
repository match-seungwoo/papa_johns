from __future__ import annotations

from unittest.mock import AsyncMock

from app.domain.models import JobStatus, SubjectCategory
from app.services.job_service import JobService


def _make_service(
    storage: AsyncMock | None = None,
    queue: AsyncMock | None = None,
    job_store: AsyncMock | None = None,
) -> JobService:
    if storage is None:
        storage = AsyncMock()
        storage.upload_input.return_value = "uploads/test/input.jpg"
    if queue is None:
        queue = AsyncMock()
    if job_store is None:
        job_store = AsyncMock()
        job_store.get.return_value = None
    return JobService(storage=storage, queue=queue, job_store=job_store)


async def test_job_service_create_job() -> None:
    storage = AsyncMock()
    storage.upload_input.return_value = "uploads/test/input.jpg"
    queue = AsyncMock()
    job_store = AsyncMock()

    service = JobService(storage=storage, queue=queue, job_store=job_store)
    job = await service.create_job(
        template_id="poster_01",
        subject_category=SubjectCategory.MALE,
        image_bytes=b"fake-image-data",
        content_type="image/jpeg",
    )

    assert job.status == JobStatus.QUEUED
    assert job.job_id.startswith("job_")
    assert job.template_id == "poster_01"
    assert job.input_s3_key == "uploads/test/input.jpg"
    storage.upload_input.assert_called_once_with(
        job.job_id, b"fake-image-data", "image/jpeg"
    )
    queue.enqueue_job.assert_called_once_with(job)
    job_store.save.assert_called_once_with(job)


async def test_job_service_get_job_not_found() -> None:
    service = _make_service()
    result = await service.get_job("nonexistent_job_id")
    assert result is None


async def test_job_service_get_job_found() -> None:
    storage = AsyncMock()
    storage.upload_input.return_value = "uploads/test/input.jpg"
    job_store = AsyncMock()

    service = JobService(storage=storage, queue=AsyncMock(), job_store=job_store)
    job = await service.create_job(
        template_id="poster_02",
        subject_category=SubjectCategory.FEMALE,
        image_bytes=b"data",
        content_type="image/jpeg",
    )
    job_store.get.return_value = job

    found = await service.get_job(job.job_id)
    assert found is not None
    assert found.job_id == job.job_id


async def test_job_service_update_status_running() -> None:
    storage = AsyncMock()
    storage.upload_input.return_value = "uploads/test/input.jpg"
    job_store = AsyncMock()

    service = JobService(storage=storage, queue=AsyncMock(), job_store=job_store)
    job = await service.create_job(
        template_id="poster_01",
        subject_category=SubjectCategory.BOY,
        image_bytes=b"data",
        content_type="image/jpeg",
    )
    job_store.get.return_value = job

    updated = await service.update_job_status(job.job_id, JobStatus.RUNNING)
    assert updated is not None
    assert updated.status == JobStatus.RUNNING
    assert updated.result_url is None


async def test_job_service_update_status_succeeded() -> None:
    storage = AsyncMock()
    storage.upload_input.return_value = "uploads/test/input.jpg"
    job_store = AsyncMock()

    service = JobService(storage=storage, queue=AsyncMock(), job_store=job_store)
    job = await service.create_job(
        template_id="poster_01",
        subject_category=SubjectCategory.GIRL,
        image_bytes=b"data",
        content_type="image/jpeg",
    )
    job_store.get.return_value = job

    result_url = "https://bucket.s3.amazonaws.com/generated/poster.png"
    updated = await service.update_job_status(
        job.job_id, JobStatus.SUCCEEDED, result_url=result_url
    )
    assert updated is not None
    assert updated.status == JobStatus.SUCCEEDED
    assert updated.result_url == result_url


async def test_job_service_update_status_not_found() -> None:
    service = _make_service()
    result = await service.update_job_status("nonexistent", JobStatus.FAILED)
    assert result is None
