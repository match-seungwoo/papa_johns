from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_job_service
from app.main import app
from app.services.job_service import JobService
from tests.test_adapters import FakeJobStore


@pytest.fixture
def mock_storage() -> AsyncMock:
    storage = AsyncMock()
    storage.upload_input.return_value = "uploads/test_job/input.jpg"
    storage.upload_result.return_value = "generated/test_job/poster.png"
    storage.get_url.return_value = (
        "https://bucket.s3.amazonaws.com/generated/test_job/poster.png"
    )
    return storage


@pytest.fixture
def mock_queue() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def job_service(mock_storage: AsyncMock, mock_queue: AsyncMock) -> JobService:
    return JobService(
        storage=mock_storage, queue=mock_queue, job_store=FakeJobStore()
    )


@asynccontextmanager
async def _null_lifespan(app: object) -> AsyncGenerator[None]:
    yield


@pytest.fixture
def client(job_service: JobService) -> Generator[TestClient]:
    app.dependency_overrides[get_job_service] = lambda: job_service
    with patch.object(app.router, "lifespan_context", _null_lifespan):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()
