from __future__ import annotations

import asyncio
import logging

from app.adapters.bfl_image import BFLImageAdapter
from app.adapters.openai_image import OpenAIImageAdapter
from app.adapters.queue import SQSQueue
from app.adapters.storage import S3Storage
from app.config import get_settings
from app.services.job_service import JobService
from app.workers.worker import Worker

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    settings = get_settings()

    storage = S3Storage(bucket=settings.s3_bucket, region=settings.aws_region)
    queue = SQSQueue(queue_url=settings.sqs_queue_url, region=settings.aws_region)
    job_service = JobService(storage=storage, queue=queue)

    adapters = {
        "openai": OpenAIImageAdapter(api_key=settings.openai_api_key),
        "bfl": BFLImageAdapter(api_key=settings.bfl_api_key),
    }

    worker = Worker(
        queue=queue,
        storage=storage,
        job_service=job_service,
        adapters=adapters,
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
