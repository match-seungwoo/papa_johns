# API Contract

Base path:

/v1

---

# POST /v1/jobs

Create a poster generation job.

## Request

multipart/form-data

Fields:

| field            | type   | required |
| ---------------- | ------ | -------- |
| template_id      | string | yes      |
| subject_category | string | yes      |
| image_file       | file   | yes      |
| species_hint     | string | no       |

Allowed values:

subject_category:
male | female | boy | girl | animal

Example:

POST /v1/jobs
Content-Type: multipart/form-data

---

## Response

{
"job_id": "job_12345",
"status": "queued"
}

---

# GET /v1/jobs/{job_id}

Retrieve job status.

## Response

{
"job_id": "job_12345",
"status": "running",
"result_url": null,
"created_at": "2026-03-10T12:00:00Z"
}

When completed:

{
"job_id": "job_12345",
"status": "succeeded",
"result_urls": {
  "openai": "https://s3.amazonaws.com/.../openai.png"
}
}

---

# Job Status

Allowed states:

queued
running
succeeded
failed

---

# Health Check

GET /healthz

Response:

{ "status": "ok" }
