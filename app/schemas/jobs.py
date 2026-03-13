from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.domain.models import JobStatus


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result_url: str | None = None
    result_urls: dict[str, str] | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
