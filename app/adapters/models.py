from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class GenerationStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ImageGenerationRequest:
    prompt: str
    poster_image_bytes: bytes
    user_image_bytes: bytes
    output_format: Literal["png", "jpeg"] = "png"
    size: str | None = None
    quality: str | None = None
    template_id: str | None = None
    subject_category: str | None = None
    face_inpaint_prompt: str | None = None


@dataclass(frozen=True)
class SubmissionResult:
    external_job_id: str | None
    status: GenerationStatus
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PollResult:
    external_job_id: str | None
    status: GenerationStatus
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FetchResult:
    external_job_id: str | None
    status: GenerationStatus
    result_bytes: bytes | None
    mime_type: str | None
    raw_metadata: dict[str, Any] = field(default_factory=dict)
