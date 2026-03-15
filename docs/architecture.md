# System Architecture

## Overview

비동기 잡 파이프라인 기반의 AI 포스터 생성 백엔드 엔진.

```
Client
  │
  ▼
FastAPI API Service
  │  (이미지 → S3, Job → DynamoDB, 메시지 → SQS)
  ▼
SQS Job Queue
  │
  ▼
Worker Service
  ├─── OpenAI Image API  (포스터 스타일 합성)
  ├─── BFL FLUX API      (IP-Adapter 이미지 생성)
  └─── Akool Face-Swap   (얼굴 후처리)
  │
  ▼
S3 Storage (결과 이미지)
  │
  ▼
DynamoDB (Job 상태 업데이트)
```

---

## Components

### API Service (`app/main.py`, `app/api/`)

책임:
- 유저 이미지 수신 및 S3 즉시 저장
- 요청 유효성 검사
- DynamoDB에 Job 레코드 생성 (status: PENDING)
- SQS에 Job 메시지 발행
- job_id 반환

**API 서버는 벤더 API를 직접 호출하지 않는다.**

---

### Worker Service (`app/workers/`)

두 가지 실행 방식:
- FastAPI lifespan 내 백그라운드 태스크 (개발/단일 프로세스)
- `python -m app.workers` 독립 프로세스 (운영)

책임:
- SQS 폴링 (1초 간격)
- recipe YAML 로드 (template_id 기반)
- 스타일 이미지 로드 (`images/*.png`)
- 벤더 어댑터 병렬 실행 (`asyncio.gather`)
- 후처리 실행 (`post_process` 항목 기반)
- 결과 이미지 S3 업로드
- Job 상태 업데이트 (RUNNING → SUCCEEDED / FAILED)

---

### Vendor Adapter Layer (`app/adapters/`)

모든 벤더 어댑터는 `ImageGenerationAdapter` ABC를 구현한다.

```python
class ImageGenerationAdapter(ABC):
    async def submit(request) -> SubmissionResult
    async def poll(external_job_id) -> PollResult
    async def fetch_result(external_job_id) -> FetchResult
```

| 파일 | 벤더 | 방식 |
|------|------|------|
| `openai_image.py` | OpenAI `gpt-image-1` | `images.edit()` 동기 호출 → asyncio.to_thread |
| `bfl_image.py` | BFL `flux-pro-1.1-ultra` | submit → poll 루프 → fetch |
| `akool_faceswap.py` | Akool `highquality/specifyimage` | 후처리 전용 (ABC 미구현) |

**AkoolFaceSwapAdapter**는 생성 어댑터가 아닌 후처리 어댑터로, Worker에 별도 주입된다.

---

### Storage (`app/adapters/storage.py`)

S3 기반. AWS Profile(`mc`)을 통해 인증.

```
uploads/{job_id}/input.{ext}        ← 유저 업로드 이미지
generated/{job_id}/{vendor}.png     ← 생성 결과 이미지
```

---

### Job Store (`app/adapters/job_store.py`)

DynamoDB 기반 Job 상태 영속화.

상태 전이: `PENDING → RUNNING → SUCCEEDED | FAILED`

`result_urls`는 벤더별 S3 URL dict로 저장:
```json
{ "openai": "https://..." }
```

---

### Queue (`app/adapters/queue.py`)

SQS 기반. 메시지 페이로드: `job_id`, `template_id`, `subject_category`, `input_s3_key`.

---

### Recipe Configuration (`configs/recipes/`)

워커가 읽는 YAML 파일. 코드 변경 없이 파이프라인 동작을 제어.

```yaml
template_id: poster_01
ad_image: images/1.png
prompt_template: "..."
quality: high          # OpenAI quality 파라미터
vendors:
  - openai             # 병렬 실행할 벤더 목록
post_process:
  - faceswap           # FAL faceswap 후처리 활성화
```

지원 필드:

| 필드 | 설명 |
|------|------|
| `ad_image` | 포스터 스타일 참조 이미지 경로 |
| `prompt_template` | `{subject_category}` 치환 가능 |
| `quality` | OpenAI quality (`standard` / `high`) |
| `vendors` | 실행할 벤더 리스트 (병렬) |
| `post_process` | 후처리 목록 (`faceswap` 지원) |

---

## AWS 인프라

| 서비스 | 리소스 |
|--------|--------|
| S3 | `papa-poster-bucket` (`ap-northeast-2`) |
| SQS | `papa-poster-jobs` |
| DynamoDB | `papa-poster-jobs` |
| ECS Fargate | API 서비스 + Worker 서비스 |

AWS 인증: `boto3.Session(profile_name="mc")` — `.env`의 `AWS_PROFILE` 환경변수로 제어.

---

## 환경변수 (`config.py`)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `AWS_PROFILE` | `""` | boto3 프로파일 |
| `AWS_REGION` | `ap-northeast-2` | |
| `S3_BUCKET` | `papa-poster-bucket` | |
| `SQS_QUEUE_URL` | — | SQS URL |
| `DYNAMODB_TABLE` | `papa-poster-jobs` | |
| `OPENAI_API_KEY` | — | |
| `OPENAI_IMAGE_MODEL` | `gpt-image-1` | |
| `AKOOL_API_KEY` | — | Akool API 키 |
| `AKOOL_FACE_ENHANCE` | `1` | 0=Classic, 1=HD Optimized |
| `BFL_API_KEY` | — | |
| `BFL_FLUX_MODEL` | `flux-pro-1.1-ultra` | |
| `BFL_BASE_URL` | `https://api.bfl.ai/v1` | |
