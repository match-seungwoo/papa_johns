from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_job_service
from app.domain.models import SpeciesHint, SubjectCategory
from app.schemas.jobs import CreateJobResponse, JobStatusResponse
from app.services.job_service import JobService

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.post("", response_model=CreateJobResponse, status_code=202)
async def create_job(
    template_id: str = Form(...),
    subject_category: SubjectCategory = Form(...),
    image_file: UploadFile = File(...),
    species_hint: SpeciesHint | None = Form(None),
    service: JobService = Depends(get_job_service),
) -> CreateJobResponse:
    image_bytes = await image_file.read()
    content_type = image_file.content_type or "image/jpeg"

    job = await service.create_job(
        template_id=template_id,
        subject_category=subject_category,
        image_bytes=image_bytes,
        content_type=content_type,
        species_hint=species_hint,
    )
    return CreateJobResponse(job_id=job.job_id, status=job.status)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> JobStatusResponse:
    job = await service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result_url=job.result_url,
        created_at=job.created_at,
    )
