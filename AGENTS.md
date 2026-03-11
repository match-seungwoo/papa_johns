# Repository Agent Rules

Agents must read the following first:

- docs/poc_scope.md
- docs/api_contract.md
- docs/architecture.md

---

## Development Guidelines

1. Implement the system incrementally
2. Keep API contract stable
3. Prefer small commits
4. Write tests for domain logic

---

## Forbidden Actions

Agents must not execute:

terraform apply
aws ecs update-service
kubectl apply
rm -rf

---

## Testing Policy

Unit tests must not call real vendor APIs.

Vendor calls must be mocked.

---

## Architecture Constraints

Vendor integrations must stay behind adapter interfaces.

Never mix vendor logic with business logic.

Allowed location:

app/adapters/

---

## Job Execution Model

All generation work must run in the worker service.

API service responsibilities:

- validate request
- enqueue job
- return job_id
