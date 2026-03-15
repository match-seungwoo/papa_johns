from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.domain.models import Job, JobStatus, SubjectCategory
from app.workers.worker import Worker


@pytest.mark.asyncio
async def test_faceswap_recipe_requires_akool_adapter() -> None:
    worker = Worker(
        queue=AsyncMock(),
        storage=AsyncMock(),
        job_service=AsyncMock(),
        adapters={},
        faceswap=None,
    )
    job = Job(
        job_id="job_test_faceswap_missing_adapter",
        status=JobStatus.QUEUED,
        template_id="poster_01",
        subject_category=SubjectCategory.MALE,
        input_s3_key="uploads/job_test_faceswap_missing_adapter/input.jpg",
        created_at=datetime.now(UTC),
    )

    recipe = {
        "template_id": "poster_01",
        "ad_image": "images/1.png",
        "prompt_template": "test prompt {subject_category}",
        "vendors": ["openai"],
        "post_process": ["faceswap"],
    }
    with patch("app.workers.worker.load_recipe", return_value=recipe):
        with pytest.raises(RuntimeError, match="Recipe requires faceswap"):
            await worker._process_job(job)
