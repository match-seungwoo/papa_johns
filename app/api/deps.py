from functools import lru_cache

from app.adapters.queue import SQSQueue
from app.adapters.storage import S3Storage
from app.config import get_settings
from app.services.job_service import JobService


@lru_cache
def get_job_service() -> JobService:
    settings = get_settings()
    storage = S3Storage(bucket=settings.s3_bucket, region=settings.aws_region)
    queue = SQSQueue(queue_url=settings.sqs_queue_url, region=settings.aws_region)
    return JobService(storage=storage, queue=queue)
