from fastapi import Request

from app.services.job_service import JobService


def get_job_service(request: Request) -> JobService:
    return request.app.state.job_service  # type: ignore[no-any-return]
