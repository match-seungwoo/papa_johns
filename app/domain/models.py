from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SubjectCategory(StrEnum):
    MALE = "male"
    FEMALE = "female"
    BOY = "boy"
    GIRL = "girl"
    ANIMAL = "animal"


class SpeciesHint(StrEnum):
    DOG = "dog"
    CAT = "cat"
    OTHER = "other"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    template_id: str
    subject_category: SubjectCategory
    species_hint: SpeciesHint | None = None
    input_s3_key: str
    result_url: str | None = None
    result_urls: dict[str, str] | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
