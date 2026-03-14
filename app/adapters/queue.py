from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any

from app.domain.models import Job


class QueueAdapter(ABC):
    @abstractmethod
    async def enqueue_job(self, job: Job) -> None:
        """Send job to the queue."""
        ...

    @abstractmethod
    async def receive_job(self) -> Job | None:
        """Receive and acknowledge one job. Returns None if queue is empty."""
        ...


class SQSQueue(QueueAdapter):
    def __init__(
        self, queue_url: str, region: str = "us-east-1", profile: str = ""
    ) -> None:
        import boto3  # deferred to avoid import cost at module load

        self._queue_url = queue_url
        self._region = region
        session = boto3.Session(profile_name=profile or None, region_name=region)
        self._client: Any = session.client("sqs")

    async def enqueue_job(self, job: Job) -> None:
        await asyncio.to_thread(
            self._client.send_message,
            QueueUrl=self._queue_url,
            MessageBody=job.model_dump_json(),
        )

    async def receive_job(self) -> Job | None:
        response: dict[str, Any] = await asyncio.to_thread(
            self._client.receive_message,
            QueueUrl=self._queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
        )
        messages = response.get("Messages", [])
        if not messages:
            return None

        msg = messages[0]
        data: dict[str, Any] = json.loads(msg["Body"])

        await asyncio.to_thread(
            self._client.delete_message,
            QueueUrl=self._queue_url,
            ReceiptHandle=msg["ReceiptHandle"],
        )

        return Job.model_validate(data)
