from __future__ import annotations

from unittest.mock import AsyncMock

from app.adapters.job_store import JobStoreAdapter
from app.domain.models import Job, JobStatus, SubjectCategory
from app.services.job_service import JobService


class FakeJobStore(JobStoreAdapter):
    """In-memory job store for unit tests."""

    def __init__(self) -> None:
        self._store: dict[str, Job] = {}

    async def save(self, job: Job) -> None:
        self._store[job.job_id] = job

    async def get(self, job_id: str) -> Job | None:
        return self._store.get(job_id)


def _make_service(**overrides: object) -> JobService:
    storage = AsyncMock()
    storage.upload_input.return_value = "uploads/test/input.jpg"
    kwargs: dict[str, object] = {
        "storage": storage,
        "queue": AsyncMock(),
        "job_store": FakeJobStore(),
    }
    kwargs.update(overrides)
    return JobService(**kwargs)  # type: ignore[arg-type]


async def test_job_service_create_job() -> None:
    queue = AsyncMock()
    service = _make_service(queue=queue)

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
    queue.enqueue_job.assert_called_once_with(job)


async def test_job_service_get_job_not_found() -> None:
    service = _make_service()
    result = await service.get_job("nonexistent_job_id")
    assert result is None


async def test_job_service_get_job_found() -> None:
    service = _make_service()

    job = await service.create_job(
        template_id="poster_02",
        subject_category=SubjectCategory.FEMALE,
        image_bytes=b"data",
        content_type="image/jpeg",
    )
    found = await service.get_job(job.job_id)
    assert found is not None
    assert found.job_id == job.job_id


async def test_job_service_update_status_running() -> None:
    service = _make_service()

    job = await service.create_job(
        template_id="poster_01",
        subject_category=SubjectCategory.BOY,
        image_bytes=b"data",
        content_type="image/jpeg",
    )
    updated = await service.update_job_status(job.job_id, JobStatus.RUNNING)
    assert updated is not None
    assert updated.status == JobStatus.RUNNING
    assert updated.result_url is None


async def test_job_service_update_status_succeeded() -> None:
    service = _make_service()

    job = await service.create_job(
        template_id="poster_01",
        subject_category=SubjectCategory.GIRL,
        image_bytes=b"data",
        content_type="image/jpeg",
    )
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
