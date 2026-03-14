from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.adapters.bfl_image import BFLImageAdapter, _map_bfl_status
from app.adapters.exceptions import (
    AdapterAPIError,
    AdapterConfigError,
    AdapterRequestError,
    AdapterResponseError,
    AdapterTimeoutError,
)
from app.adapters.models import GenerationStatus, ImageGenerationRequest

# ---------------------------------------------------------------------------
# _map_bfl_status
# ---------------------------------------------------------------------------


def test_map_bfl_status_ready() -> None:
    assert _map_bfl_status("Ready") == GenerationStatus.SUCCEEDED


def test_map_bfl_status_error() -> None:
    assert _map_bfl_status("Error") == GenerationStatus.FAILED


def test_map_bfl_status_moderated() -> None:
    assert _map_bfl_status("Request Moderated") == GenerationStatus.FAILED
    assert _map_bfl_status("Content Moderated") == GenerationStatus.FAILED


def test_map_bfl_status_pending() -> None:
    assert _map_bfl_status("Pending") == GenerationStatus.RUNNING


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_init_raises_on_missing_api_key() -> None:
    with pytest.raises(AdapterConfigError):
        BFLImageAdapter(api_key="")


def test_init_stores_params() -> None:
    adapter = BFLImageAdapter(
        api_key="key",
        model="flux-pro-1.1-ultra",
        image_prompt_strength=0.2,
        submit_timeout_seconds=45.0,
        submit_max_retries=3,
        poll_interval=3.0,
        poll_max_attempts=10,
    )
    assert adapter._model == "flux-pro-1.1-ultra"
    assert adapter._image_prompt_strength == 0.2
    assert adapter._submit_timeout_seconds == 45.0
    assert adapter._submit_max_retries == 3
    assert adapter._poll_interval == 3.0
    assert adapter._poll_max_attempts == 10


# ---------------------------------------------------------------------------
# submit()
# ---------------------------------------------------------------------------


def _make_request(**kwargs: object) -> ImageGenerationRequest:
    defaults: dict[str, object] = {
        "prompt": "a stylish poster",
        "poster_image_bytes": b"poster-bytes",
        "user_image_bytes": b"face-bytes",
    }
    defaults.update(kwargs)
    return ImageGenerationRequest(**defaults)  # type: ignore[arg-type]


async def test_submit_raises_on_empty_prompt() -> None:
    adapter = BFLImageAdapter(api_key="key")
    with pytest.raises(AdapterRequestError):
        await adapter.submit(_make_request(prompt=""))


async def test_submit_raises_on_empty_user_image() -> None:
    adapter = BFLImageAdapter(api_key="key")
    with pytest.raises(AdapterRequestError):
        await adapter.submit(_make_request(user_image_bytes=b""))


async def test_submit_sends_ip_adapter_face() -> None:
    """user_image_bytes must be sent as image_prompt (IP-Adapter face)."""
    import base64

    adapter = BFLImageAdapter(api_key="test-key", image_prompt_strength=0.15)
    request = _make_request(user_image_bytes=b"face-image-data")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "bfl-job-123", "status": "pending"}

    with patch("app.adapters.bfl_image.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await adapter.submit(request)

    assert result.external_job_id == "bfl-job-123"
    assert result.status == GenerationStatus.QUEUED

    call_kwargs = mock_client.post.call_args
    body = call_kwargs.kwargs["json"]
    assert body["image_prompt"] == base64.b64encode(b"face-image-data").decode()
    assert body["image_prompt_strength"] == 0.15


async def test_submit_raises_on_http_error_status() -> None:
    adapter = BFLImageAdapter(api_key="key")

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"

    with patch("app.adapters.bfl_image.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with pytest.raises(AdapterAPIError, match="HTTP 401"):
            await adapter.submit(_make_request())


async def test_submit_retries_on_timeout_then_succeeds() -> None:
    adapter = BFLImageAdapter(api_key="key", submit_max_retries=2)
    request = _make_request()

    timeout_exc = httpx.ConnectTimeout("connect timeout")
    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {"id": "bfl-job-456"}

    with (
        patch("app.adapters.bfl_image.httpx.AsyncClient") as mock_client_cls,
        patch("app.adapters.bfl_image.asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=[timeout_exc, success_response])
        mock_client_cls.return_value = mock_client

        result = await adapter.submit(request)

    assert result.external_job_id == "bfl-job-456"
    assert mock_client.post.call_count == 2
    mock_sleep.assert_awaited_once_with(1.0)


async def test_submit_raises_timeout_after_exhausting_retries() -> None:
    adapter = BFLImageAdapter(api_key="key", submit_max_retries=1)
    request = _make_request()

    timeout_exc = httpx.ConnectTimeout("connect timeout")

    with (
        patch("app.adapters.bfl_image.httpx.AsyncClient") as mock_client_cls,
        patch("app.adapters.bfl_image.asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=timeout_exc)
        mock_client_cls.return_value = mock_client

        with pytest.raises(AdapterTimeoutError, match="after 2 attempt"):
            await adapter.submit(request)

    assert mock_client.post.call_count == 2
    mock_sleep.assert_awaited_once_with(1.0)


# ---------------------------------------------------------------------------
# poll()
# ---------------------------------------------------------------------------


async def test_poll_returns_running_for_pending() -> None:
    adapter = BFLImageAdapter(api_key="key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "job-1", "status": "Pending"}

    with patch("app.adapters.bfl_image.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await adapter.poll("job-1")

    assert result.status == GenerationStatus.RUNNING


async def test_poll_returns_succeeded_for_ready() -> None:
    adapter = BFLImageAdapter(api_key="key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "job-1",
        "status": "Ready",
        "result": {"sample": "https://cdn.example.com/result.png"},
    }

    with patch("app.adapters.bfl_image.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await adapter.poll("job-1")

    assert result.status == GenerationStatus.SUCCEEDED


async def test_poll_treats_task_not_found_404_as_running() -> None:
    adapter = BFLImageAdapter(api_key="key")

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = '{"status":"Task not found"}'
    mock_response.json.return_value = {
        "id": "job-1",
        "status": "Task not found",
        "result": None,
    }

    with patch("app.adapters.bfl_image.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await adapter.poll("job-1")

    assert result.status == GenerationStatus.RUNNING


# ---------------------------------------------------------------------------
# fetch_result()
# ---------------------------------------------------------------------------


async def test_fetch_result_success() -> None:
    adapter = BFLImageAdapter(api_key="key", poll_interval=0.0, poll_max_attempts=3)

    poll_response = MagicMock()
    poll_response.status_code = 200
    poll_response.json.return_value = {
        "id": "job-1",
        "status": "Ready",
        "result": {"sample": "https://cdn.example.com/result.png"},
    }

    img_response = MagicMock()
    img_response.status_code = 200
    img_response.content = b"final-image-bytes"

    call_count = 0

    async def fake_get(url: str, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if "get_result" in url:
            return poll_response
        return img_response

    with patch("app.adapters.bfl_image.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = fake_get
        mock_client_cls.return_value = mock_client

        result = await adapter.fetch_result("job-1")

    assert result.status == GenerationStatus.SUCCEEDED
    assert result.result_bytes == b"final-image-bytes"
    assert result.mime_type == "image/png"


async def test_fetch_result_raises_on_failure() -> None:
    adapter = BFLImageAdapter(api_key="key", poll_interval=0.0, poll_max_attempts=3)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "job-1", "status": "Error"}

    with patch("app.adapters.bfl_image.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with pytest.raises(AdapterAPIError):
            await adapter.fetch_result("job-1")


async def test_fetch_result_raises_timeout_when_stuck_pending() -> None:
    adapter = BFLImageAdapter(api_key="key", poll_interval=0.0, poll_max_attempts=2)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "job-1", "status": "Pending"}

    with patch("app.adapters.bfl_image.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with pytest.raises(AdapterTimeoutError):
            await adapter.fetch_result("job-1")


async def test_fetch_result_raises_when_sample_url_missing() -> None:
    adapter = BFLImageAdapter(api_key="key", poll_interval=0.0, poll_max_attempts=3)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "job-1",
        "status": "Ready",
        "result": {},  # no 'sample' key
    }

    with patch("app.adapters.bfl_image.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with pytest.raises(AdapterResponseError):
            await adapter.fetch_result("job-1")
