from __future__ import annotations

import asyncio
import logging

from app.adapters.bfl_image import BFLImageAdapter
from app.adapters.fal_faceswap import FALFaceSwapAdapter
from app.adapters.job_store import DynamoDBJobStore
from app.adapters.openai_image import OpenAIImageAdapter
from app.adapters.queue import SQSQueue
from app.adapters.storage import S3Storage
from app.config import get_settings
from app.services.job_service import JobService
from app.workers.worker import Worker

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    settings = get_settings()

    storage = S3Storage(
        bucket=settings.s3_bucket,
        region=settings.aws_region,
        profile=settings.aws_profile,
    )
    queue = SQSQueue(
        queue_url=settings.sqs_queue_url,
        region=settings.aws_region,
        profile=settings.aws_profile,
    )
    job_store = DynamoDBJobStore(
        table_name=settings.dynamodb_table,
        region=settings.aws_region,
        profile=settings.aws_profile,
    )
    job_service = JobService(storage=storage, queue=queue, job_store=job_store)

    adapters = {
        "openai": OpenAIImageAdapter(
            api_key=settings.openai_api_key,
            model=settings.openai_image_model,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        ),
        "bfl": BFLImageAdapter(
            api_key=settings.bfl_api_key,
            model=settings.bfl_flux_model,
            base_url=settings.bfl_base_url,
            image_prompt_strength=settings.bfl_image_prompt_strength,
            submit_timeout_seconds=settings.bfl_submit_timeout_seconds,
            submit_max_retries=settings.bfl_submit_max_retries,
            poll_interval=settings.bfl_poll_interval_seconds,
            poll_max_attempts=settings.bfl_poll_max_attempts,
        ),
    }

    faceswap = FALFaceSwapAdapter(
        api_key=settings.fal_key,
        model=settings.fal_faceswap_model,
    ) if settings.fal_key else None

    worker = Worker(
        queue=queue,
        storage=storage,
        job_service=job_service,
        adapters=adapters,
        faceswap=faceswap,
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
