from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from app.domain.models import Job


class JobStoreAdapter(ABC):
    @abstractmethod
    async def save(self, job: Job) -> None:
        """Persist a job (insert or overwrite)."""
        ...

    @abstractmethod
    async def get(self, job_id: str) -> Job | None:
        """Retrieve a job by ID. Returns None if not found."""
        ...


class DynamoDBJobStore(JobStoreAdapter):
    def __init__(self, table_name: str, region: str = "us-east-1") -> None:
        import boto3  # deferred to avoid import cost at module load

        self._table: Any = boto3.resource("dynamodb", region_name=region).Table(
            table_name
        )

    async def save(self, job: Job) -> None:
        item = {k: v for k, v in job.model_dump(mode="json").items() if v is not None}
        await asyncio.to_thread(self._table.put_item, Item=item)

    async def get(self, job_id: str) -> Job | None:
        response: dict[str, Any] = await asyncio.to_thread(
            self._table.get_item, Key={"job_id": job_id}
        )
        item = response.get("Item")
        if item is None:
            return None
        return Job.model_validate(item)
