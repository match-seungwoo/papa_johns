from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.adapters.exceptions import (
    AdapterAPIError,
    AdapterConfigError,
    AdapterRequestError,
    AdapterResponseError,
)
from app.adapters.models import (
    FetchResult,
    GenerationStatus,
    ImageGenerationRequest,
    SubmissionResult,
)
from app.adapters.openai_image import OpenAIImageAdapter
from app.config import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(**kwargs: Any) -> ImageGenerationRequest:
    defaults: dict[str, Any] = {
        "prompt": "style transfer: classic portrait",
        "poster_image_bytes": b"poster-bytes",
        "user_image_bytes": b"user-bytes",
    }
    defaults.update(kwargs)
    return ImageGenerationRequest(**defaults)


def _fake_b64(data: bytes = b"generated-image-bytes") -> str:
    return base64.b64encode(data).decode()


def _mock_openai_response(b64: str) -> MagicMock:
    image_item = MagicMock()
    image_item.b64_json = b64
    response = MagicMock()
    response.data = [image_item]
    return response


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

def test_adapter_raises_config_error_for_empty_api_key() -> None:
    with pytest.raises(AdapterConfigError, match="API key"):
        OpenAIImageAdapter(api_key="")


def test_adapter_loads_api_key_from_settings() -> None:
    settings = Settings(openai_api_key="test-key-from-settings")
    adapter = OpenAIImageAdapter(
        api_key=settings.openai_api_key,
        model=settings.openai_image_model,
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )
    assert adapter._api_key == "test-key-from-settings"
    assert adapter._model == settings.openai_image_model
    assert adapter._timeout == settings.openai_timeout_seconds
    assert adapter._max_retries == settings.openai_max_retries


# ---------------------------------------------------------------------------
# submit() — success path
# ---------------------------------------------------------------------------

async def test_submit_returns_submission_result() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key")
    mock_client = MagicMock()
    mock_client.images.edit.return_value = _mock_openai_response(_fake_b64())

    with patch.object(adapter, "_build_client", return_value=mock_client):
        result = await adapter.submit(_make_request())

    assert isinstance(result, SubmissionResult)
    assert result.status == GenerationStatus.SUCCEEDED
    assert result.external_job_id is not None
    assert result.raw_metadata["model"] == "gpt-image-1"


async def test_submit_builds_openai_request_correctly() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key", model="gpt-image-1.5")
    mock_client = MagicMock()
    mock_client.images.edit.return_value = _mock_openai_response(_fake_b64())

    request = _make_request(prompt="my custom prompt")

    with patch.object(adapter, "_build_client", return_value=mock_client):
        await adapter.submit(request)

    call_kwargs = mock_client.images.edit.call_args.kwargs
    assert call_kwargs["model"] == "gpt-image-1.5"
    assert call_kwargs["prompt"] == "my custom prompt"
    assert call_kwargs["n"] == 1
    assert isinstance(call_kwargs["image"], list)
    assert len(call_kwargs["image"]) == 2


async def test_submit_passes_size_and_quality_when_set() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key")
    mock_client = MagicMock()
    mock_client.images.edit.return_value = _mock_openai_response(_fake_b64())

    request = _make_request(size="1024x1024", quality="high")

    with patch.object(adapter, "_build_client", return_value=mock_client):
        await adapter.submit(request)

    call_kwargs = mock_client.images.edit.call_args.kwargs
    assert call_kwargs["size"] == "1024x1024"
    assert call_kwargs["quality"] == "high"


async def test_submit_omits_optional_params_when_not_set() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key")
    mock_client = MagicMock()
    mock_client.images.edit.return_value = _mock_openai_response(_fake_b64())

    with patch.object(adapter, "_build_client", return_value=mock_client):
        await adapter.submit(_make_request())

    call_kwargs = mock_client.images.edit.call_args.kwargs
    assert "size" not in call_kwargs
    assert "quality" not in call_kwargs


# ---------------------------------------------------------------------------
# fetch_result()
# ---------------------------------------------------------------------------

async def test_fetch_result_returns_normalized_result() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key")
    image_data = b"real-image-bytes"
    mock_client = MagicMock()
    mock_client.images.edit.return_value = _mock_openai_response(_fake_b64(image_data))

    with patch.object(adapter, "_build_client", return_value=mock_client):
        submission = await adapter.submit(_make_request())

    fetch = await adapter.fetch_result(submission.external_job_id or "")

    assert isinstance(fetch, FetchResult)
    assert fetch.result_bytes == image_data
    assert fetch.mime_type == "image/png"
    assert fetch.status == GenerationStatus.SUCCEEDED
    assert fetch.external_job_id == submission.external_job_id


async def test_fetch_result_respects_jpeg_output_format() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key")
    mock_client = MagicMock()
    mock_client.images.edit.return_value = _mock_openai_response(_fake_b64())

    request = _make_request(output_format="jpeg")
    with patch.object(adapter, "_build_client", return_value=mock_client):
        submission = await adapter.submit(request)

    fetch = await adapter.fetch_result(submission.external_job_id or "")
    assert fetch.mime_type == "image/jpeg"


async def test_fetch_result_raises_for_unknown_job_id() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key")
    with pytest.raises(AdapterResponseError, match="No result found"):
        await adapter.fetch_result("nonexistent-job-id")


# ---------------------------------------------------------------------------
# poll()
# ---------------------------------------------------------------------------

async def test_poll_returns_succeeded_for_known_job() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key")
    mock_client = MagicMock()
    mock_client.images.edit.return_value = _mock_openai_response(_fake_b64())

    with patch.object(adapter, "_build_client", return_value=mock_client):
        submission = await adapter.submit(_make_request())

    poll = await adapter.poll(submission.external_job_id or "")
    assert poll.status == GenerationStatus.SUCCEEDED
    assert poll.external_job_id == submission.external_job_id


async def test_poll_returns_failed_for_unknown_job() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key")
    poll = await adapter.poll("unknown-job-id")
    assert poll.status == GenerationStatus.FAILED


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

async def test_api_failure_raises_adapter_api_error() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key")

    with patch.object(
        adapter, "_call_openai", side_effect=AdapterAPIError("service unavailable")
    ):
        with pytest.raises(AdapterAPIError, match="service unavailable"):
            await adapter.submit(_make_request())


async def test_malformed_response_raises_response_error() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key")

    with patch.object(
        adapter, "_call_openai", side_effect=AdapterResponseError("missing b64 content")
    ):
        with pytest.raises(AdapterResponseError, match="missing b64 content"):
            await adapter.submit(_make_request())


async def test_empty_data_list_raises_response_error() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key")
    mock_client = MagicMock()
    mock_client.images.edit.return_value = MagicMock(data=[])

    with patch.object(adapter, "_build_client", return_value=mock_client):
        with pytest.raises(AdapterResponseError, match="empty data list"):
            await adapter.submit(_make_request())


async def test_missing_b64_json_raises_response_error() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key")
    image_item = MagicMock()
    image_item.b64_json = None
    image_item.url = None
    mock_client = MagicMock()
    mock_client.images.edit.return_value = MagicMock(data=[image_item])

    with patch.object(adapter, "_build_client", return_value=mock_client):
        with pytest.raises(AdapterResponseError, match="missing both b64_json and url"):
            await adapter.submit(_make_request())


async def test_empty_prompt_raises_request_error() -> None:
    adapter = OpenAIImageAdapter(api_key="test-key")
    mock_client = MagicMock()

    with patch.object(adapter, "_build_client", return_value=mock_client):
        with pytest.raises(AdapterRequestError, match="prompt must not be empty"):
            await adapter.submit(_make_request(prompt=""))


# ---------------------------------------------------------------------------
# Secret safety
# ---------------------------------------------------------------------------

async def test_api_key_does_not_appear_in_exception_message() -> None:
    secret = "sk-supersecret-api-key-abc123"
    adapter = OpenAIImageAdapter(api_key=secret)

    with patch.object(
        adapter, "_call_openai", side_effect=AdapterAPIError("quota exceeded")
    ):
        try:
            await adapter.submit(_make_request())
        except AdapterAPIError as exc:
            assert secret not in str(exc)
            assert secret not in repr(exc)
