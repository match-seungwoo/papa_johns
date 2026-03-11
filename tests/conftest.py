from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_job_service
from app.main import app
from app.services.job_service import JobService


@pytest.fixture
def mock_storage() -> AsyncMock:
    storage = AsyncMock()
    storage.upload_input.return_value = "uploads/test_job/input.jpg"
    storage.upload_result.return_value = "generated/test_job/poster.png"
    storage.get_url.return_value = "https://bucket.s3.amazonaws.com/generated/test_job/poster.png"
    return storage


@pytest.fixture
def mock_queue() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def job_service(mock_storage: AsyncMock, mock_queue: AsyncMock) -> JobService:
    return JobService(storage=mock_storage, queue=mock_queue)


@pytest.fixture
def client(job_service: JobService) -> Generator[TestClient]:
    app.dependency_overrides[get_job_service] = lambda: job_service
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
