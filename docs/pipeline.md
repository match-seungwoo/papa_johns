# Image Generation Pipeline

## 파이프라인 설계 결정

### 검토한 방식

| 방식 | 설명 | 결과 |
|------|------|------|
| A. OpenAI only | `images.edit()`에 포스터 + 유저 이미지 동시 입력 | 스타일 반영 우수, 얼굴 identity 약함 |
| B. BFL FLUX only | `image_prompt`에 유저 얼굴 conditioning | 스타일 이미지 반영 불가 (슬롯 1개 제한) |
| **A+C. OpenAI → FAL faceswap** | OpenAI로 스타일 생성 후 FAL로 얼굴 교체 | **채택** |

### 채택 근거

BFL `flux-pro-1.1-ultra`는 `image_prompt` 슬롯이 하나뿐이어서 포스터 스타일 이미지와 유저 얼굴 이미지를 동시에 conditioning할 수 없다. OpenAI `gpt-image-1`은 `images.edit()`에 여러 이미지를 전달할 수 있어 스타일 반영이 우수하나, 유저 얼굴 정확도가 낮다. FAL AI `fal-ai/face-swap`으로 생성 후 얼굴을 교체하면 두 장점을 모두 취할 수 있다.

---

## Worker 처리 흐름

```
SQS 메시지 수신
  │
  ▼
Job 상태 → RUNNING
  │
  ▼
recipe YAML 로드
  ├─ ad_image 로드 (포스터 스타일 이미지)
  ├─ prompt 구성 ({subject_category} 치환)
  └─ vendors, post_process 파싱
  │
  ▼
벤더별 _run_vendor() 병렬 실행 (asyncio.gather)
  │
  ├─ S3에서 유저 이미지 다운로드
  ├─ ImageGenerationRequest 생성
  ├─ adapter.submit() → adapter.fetch_result()
  └─ [faceswap 활성 시] FALFaceSwapAdapter.swap()
       ├─ 생성 이미지 + 유저 이미지 FAL CDN 업로드
       └─ fal-ai/face-swap 호출 → 결과 다운로드
  │
  ▼
결과 이미지 S3 업로드
  result_key: generated/{job_id}/{vendor}.png
  │
  ▼
Job 상태 → SUCCEEDED
  result_urls: { vendor: s3_url }
```

---

## 어댑터 인터페이스

### ImageGenerationAdapter (생성 어댑터 공통)

```python
class ImageGenerationAdapter(ABC):
    async def submit(request: ImageGenerationRequest) -> SubmissionResult
    async def poll(external_job_id: str) -> PollResult
    async def fetch_result(external_job_id: str) -> FetchResult
```

OpenAI는 동기 API이므로 `submit()` 내에서 `asyncio.to_thread`로 실행하고 결과를 메모리 캐시에 저장. `fetch_result()`는 캐시에서 반환.

BFL은 비동기 API이므로 `submit()`은 잡 ID만 반환하고, `fetch_result()`에서 poll 루프를 실행.

### FALFaceSwapAdapter (후처리 어댑터)

```python
class FALFaceSwapAdapter:
    async def swap(generated_image_bytes: bytes, user_face_bytes: bytes) -> bytes
```

두 이미지를 FAL CDN에 병렬 업로드한 뒤 `fal-ai/face-swap`을 호출하고 결과를 bytes로 반환.

---

## Recipe 필드 레퍼런스

```yaml
template_id: poster_01           # 템플릿 식별자
name: "Classic Portrait"         # 사람이 읽는 이름
ad_image: images/1.png           # 스타일 참조 이미지 (프로젝트 루트 기준)
prompt_template: "..."           # {subject_category} 치환 지원
quality: high                    # OpenAI quality 파라미터 (standard | high)
vendors:                         # 실행할 벤더 (병렬)
  - openai
post_process:                    # 후처리 (순서대로 적용)
  - faceswap
```

`vendors`가 여러 개면 병렬로 실행되고 결과는 각각 S3에 저장된다.
`post_process: [faceswap]`이 있으면 각 벤더 결과에 faceswap이 적용된다.

---

## 환경변수 의존성

```
FAL_KEY              → FALFaceSwapAdapter 활성화 조건
                       (없으면 faceswap 어댑터 미생성, recipe의 faceswap 무시됨)
OPENAI_API_KEY       → OpenAIImageAdapter
BFL_API_KEY          → BFLImageAdapter
AWS_PROFILE          → boto3 인증 프로파일 (mc)
```
