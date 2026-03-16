from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.adapters.akool_faceswap import AkoolFaceSwapAdapter


class _DummyStorage:
    async def upload_temp(
        self, job_id: str, data: bytes, suffix: str, content_type: str
    ) -> str:
        return f"temp/{job_id}/{suffix}"

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return f"https://example.com/{key}"


@pytest.mark.asyncio
async def test_submit_payload_uses_v4_faceswap_by_image_shape() -> None:
    adapter = AkoolFaceSwapAdapter(api_key="akool-test-key", storage=_DummyStorage())
    adapter._poll = AsyncMock(return_value="https://example.com/result.png")  # type: ignore[method-assign]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"code": 1000, "data": {"_id": "req_123"}}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client_ctx = AsyncMock()
    mock_client_ctx.__aenter__.return_value = mock_client
    mock_client_ctx.__aexit__.return_value = False

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("app.adapters.akool_faceswap.httpx.AsyncClient", lambda **_: mock_client_ctx)
        await adapter._submit_and_poll(
            poster_url="https://example.com/poster.png",
            face_url="https://example.com/face.jpg",
            poster_landmarks=[],
        )

    args, kwargs = mock_client.post.call_args
    assert args[0].endswith("/api/open/v3/faceswap/highquality/specifyimage")
    payload = kwargs["json"]
    assert payload["modifyImage"] == "https://example.com/poster.png"
    assert payload["targetImage"] == [{"path": "https://example.com/poster.png"}]
    assert payload["sourceImage"] == [{"path": "https://example.com/face.jpg"}]


@pytest.mark.asyncio
async def test_poll_uses_listbyids_and_returns_url_on_status_3() -> None:
    adapter = AkoolFaceSwapAdapter(
        api_key="akool-test-key",
        storage=_DummyStorage(),
        poll_interval=0.0,
        poll_max_attempts=1,
    )

    poll_response = MagicMock()
    poll_response.status_code = 200
    poll_response.json.return_value = {
        "code": 1000,
        "data": {
            "result": [
                {
                    "_id": "req_123",
                    "faceswap_status": 3,
                    "url": "https://example.com/swapped.png",
                }
            ]
        },
    }

    poll_client = AsyncMock()
    poll_client.get.return_value = poll_response
    poll_ctx = AsyncMock()
    poll_ctx.__aenter__.return_value = poll_client
    poll_ctx.__aexit__.return_value = False

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "app.adapters.akool_faceswap.httpx.AsyncClient",
            lambda **_: poll_ctx,
        )
        url = await adapter._poll(
            request_id="req_123",
            headers={"x-api-key": "akool-test-key"},
        )

    args, kwargs = poll_client.get.call_args
    assert args[0].endswith("/api/open/v3/faceswap/result/listbyids")
    assert kwargs["params"] == {"_ids": "req_123"}
    assert url == "https://example.com/swapped.png"
