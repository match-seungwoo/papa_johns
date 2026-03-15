from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class StorageAdapter(ABC):
    @abstractmethod
    async def upload_input(self, job_id: str, data: bytes, content_type: str) -> str:
        """Upload input image. Returns S3 key."""
        ...

    @abstractmethod
    async def upload_result(self, job_id: str, data: bytes, vendor: str) -> str:
        """Upload generated poster. Returns S3 key."""
        ...

    @abstractmethod
    def get_url(self, key: str) -> str:
        """Return public URL for the given S3 key."""
        ...

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Download object by S3 key. Returns raw bytes."""
        ...

    @abstractmethod
    async def upload_temp(
        self, job_id: str, data: bytes, suffix: str, content_type: str
    ) -> str:
        """Upload a temporary file. Returns S3 key."""
        ...

    @abstractmethod
    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Return a presigned URL for the given S3 key."""
        ...


class S3Storage(StorageAdapter):
    def __init__(
        self, bucket: str, region: str = "us-east-1", profile: str = ""
    ) -> None:
        import boto3  # deferred to avoid import cost at module load

        self._bucket = bucket
        self._region = region
        session = boto3.Session(profile_name=profile or None, region_name=region)
        self._client: Any = session.client("s3")

    async def upload_input(self, job_id: str, data: bytes, content_type: str) -> str:
        key = f"uploads/{job_id}/input.jpg"
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return key

    async def upload_result(self, job_id: str, data: bytes, vendor: str) -> str:
        key = f"generated/{job_id}/poster_{vendor}.png"
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType="image/png",
        )
        return key

    def get_url(self, key: str) -> str:
        return f"https://{self._bucket}.s3.{self._region}.amazonaws.com/{key}"

    async def download(self, key: str) -> bytes:
        def _do() -> bytes:
            resp: dict[str, Any] = self._client.get_object(Bucket=self._bucket, Key=key)
            data: bytes = resp["Body"].read()
            return data

        return await asyncio.to_thread(_do)

    async def upload_temp(
        self, job_id: str, data: bytes, suffix: str, content_type: str
    ) -> str:
        key = f"temp/{job_id}/{suffix}"
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return key

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        url: str = self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )
        return url
