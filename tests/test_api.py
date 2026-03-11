from __future__ import annotations

import io

from fastapi.testclient import TestClient


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_job(client: TestClient) -> None:
    response = client.post(
        "/v1/jobs",
        data={"template_id": "poster_01", "subject_category": "male"},
        files={"image_file": ("photo.jpg", io.BytesIO(b"fake-image"), "image/jpeg")},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["job_id"].startswith("job_")


def test_create_job_with_species_hint(client: TestClient) -> None:
    response = client.post(
        "/v1/jobs",
        data={
            "template_id": "poster_01",
            "subject_category": "animal",
            "species_hint": "dog",
        },
        files={"image_file": ("pet.jpg", io.BytesIO(b"fake-image"), "image/jpeg")},
    )
    assert response.status_code == 202
    assert response.json()["status"] == "queued"


def test_create_job_invalid_subject_category(client: TestClient) -> None:
    response = client.post(
        "/v1/jobs",
        data={"template_id": "poster_01", "subject_category": "unknown"},
        files={"image_file": ("photo.jpg", io.BytesIO(b"fake-image"), "image/jpeg")},
    )
    assert response.status_code == 422


def test_get_job_not_found(client: TestClient) -> None:
    response = client.get("/v1/jobs/nonexistent")
    assert response.status_code == 404


def test_get_job_returns_created_job(client: TestClient) -> None:
    create_resp = client.post(
        "/v1/jobs",
        data={"template_id": "poster_02", "subject_category": "female"},
        files={"image_file": ("photo.jpg", io.BytesIO(b"fake-image"), "image/jpeg")},
    )
    job_id = create_resp.json()["job_id"]

    get_resp = client.get(f"/v1/jobs/{job_id}")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["job_id"] == job_id
    assert body["status"] == "queued"
    assert body["result_url"] is None
    assert "created_at" in body
