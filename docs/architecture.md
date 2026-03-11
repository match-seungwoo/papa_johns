# System Architecture

## Overview

The POC engine uses an asynchronous job pipeline.

Architecture:

Client
|
v
FastAPI API Service
|
v
SQS Job Queue
|
v
Worker Service
|
+---- OpenAI Image API
|
+---- BFL FLUX API
|
v
S3 Storage

---

## Components

### API Service

Responsibilities:

- receive upload
- validate request
- store image to S3
- enqueue job to SQS
- return job_id

The API **must not perform image generation directly**.

---

### Worker Service

Responsibilities:

- consume SQS messages
- load template recipe
- call vendor adapters
- upload result to S3
- update job status

---

### Storage

Artifacts stored in S3:

uploads/
generated/

Example:

uploads/{job_id}/input.jpg
generated/{job_id}/poster.png

---

### Vendor Adapter Layer

External APIs must be wrapped behind adapters:

adapters/openai_image.py
adapters/bfl_image.py

Interface example:

submit(job)
poll(job)
fetch_result(job)

---

### Configuration

Poster recipes live in:

configs/recipes/

Example:

poster_01.yaml
poster_02.yaml

---

## Deployment Target

Containerized services running on:

AWS ECS Fargate

Components:

- API service
- worker service
- SQS queue
- S3 bucket
