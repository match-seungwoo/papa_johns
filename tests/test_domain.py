from __future__ import annotations

from datetime import UTC, datetime

from app.domain.models import Job, JobStatus, SpeciesHint, SubjectCategory


def test_job_status_values() -> None:
    assert JobStatus.QUEUED.value == "queued"
    assert JobStatus.RUNNING.value == "running"
    assert JobStatus.SUCCEEDED.value == "succeeded"
    assert JobStatus.FAILED.value == "failed"


def test_subject_category_values() -> None:
    assert SubjectCategory.MALE.value == "male"
    assert SubjectCategory.FEMALE.value == "female"
    assert SubjectCategory.BOY.value == "boy"
    assert SubjectCategory.GIRL.value == "girl"
    assert SubjectCategory.ANIMAL.value == "animal"


def test_species_hint_values() -> None:
    assert SpeciesHint.DOG.value == "dog"
    assert SpeciesHint.CAT.value == "cat"
    assert SpeciesHint.OTHER.value == "other"


def test_job_creation_defaults() -> None:
    job = Job(
        job_id="job_test001",
        status=JobStatus.QUEUED,
        template_id="poster_01",
        subject_category=SubjectCategory.MALE,
        input_s3_key="uploads/job_test001/input.jpg",
        created_at=datetime.now(UTC),
    )
    assert job.job_id == "job_test001"
    assert job.status == JobStatus.QUEUED
    assert job.result_url is None
    assert job.species_hint is None


def test_job_with_all_fields() -> None:
    now = datetime.now(UTC)
    job = Job(
        job_id="job_test002",
        status=JobStatus.SUCCEEDED,
        template_id="poster_03",
        subject_category=SubjectCategory.ANIMAL,
        species_hint=SpeciesHint.DOG,
        input_s3_key="uploads/job_test002/input.jpg",
        result_url="https://bucket.s3.amazonaws.com/generated/job_test002/poster.png",
        created_at=now,
    )
    assert job.species_hint == SpeciesHint.DOG
    assert job.result_url is not None
    assert job.created_at == now


def test_job_serialization() -> None:
    job = Job(
        job_id="job_serial",
        status=JobStatus.RUNNING,
        template_id="poster_01",
        subject_category=SubjectCategory.GIRL,
        input_s3_key="uploads/job_serial/input.jpg",
        created_at=datetime.now(UTC),
    )
    data = job.model_dump()
    assert data["status"] == "running"
    assert data["subject_category"] == "girl"

    restored = Job.model_validate(data)
    assert restored.job_id == job.job_id
    assert restored.status == JobStatus.RUNNING
