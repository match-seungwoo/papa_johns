from __future__ import annotations

import asyncio
import logging
import logging.config
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.adapters.bfl_image import BFLImageAdapter
from app.adapters.job_store import DynamoDBJobStore
from app.adapters.openai_image import OpenAIImageAdapter
from app.adapters.queue import SQSQueue
from app.adapters.storage import S3Storage
from app.api.routes import health, jobs
from app.config import get_settings
from app.services.job_service import JobService
from app.workers.worker import Worker

_LOG_DIR = Path(__file__).parent.parent / "logs"


def _configure_logging() -> None:
    _LOG_DIR.mkdir(exist_ok=True)
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)s %(name)s — %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(_LOG_DIR / "app.log"),
                "maxBytes": 10 * 1024 * 1024,  # 10 MB
                "backupCount": 5,
                "formatter": "default",
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"],
        },
    })


_configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
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
            org_id=settings.openai_org_id,
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

    worker = Worker(
        queue=queue,
        storage=storage,
        job_service=job_service,
        adapters=adapters,
    )

    app.state.job_service = job_service
    task = asyncio.create_task(worker.run())
    logger.info("Worker background task started")

    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info("Worker background task stopped")


app = FastAPI(title="AI Poster Engine", version="0.1.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(jobs.router)
