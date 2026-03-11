# Claude Code Instructions

Before writing code read these files:

- docs/poc_scope.md
- docs/api_contract.md
- docs/architecture.md

---

## Project Rules

This repository builds **only the backend engine**.

Do not implement:

- frontend
- UI
- authentication
- admin tools

---

## Technical Stack

Language:

Python 3.13

Framework:

FastAPI

Validation:

Pydantic v2

Infrastructure:

AWS ECS + SQS + S3

---

## Coding Constraints

- Public API must follow docs/api_contract.md
- All image generation must run in worker service
- API service must not call vendor APIs directly
- Vendor calls must be implemented via adapter classes

---

## Storage Rules

Input images must be stored in S3 immediately.

Local disk is allowed only for temporary files.

---

## Security Rules

Never:

- print secrets
- run deployment commands
- modify infrastructure code

---

## Testing

Before finishing work run:

ruff
mypy
pytest
