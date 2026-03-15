# AI Poster Engine POC Scope

## Objective

선택된 포스터 템플릿과 유저 사진을 기반으로 개인화된 포스터 이미지를 생성하는 백엔드 엔진.

POC 범위: **생성 파이프라인 아키텍처 검증**

제외 항목:
- frontend / UI
- 인증
- 어드민 툴

---

## 지원 Subject Category

호출자가 직접 지정해야 한다. 엔진은 이미지로부터 추론하지 않는다.

| 값 | 설명 |
|----|------|
| `male` | 성인 남성 |
| `female` | 성인 여성 |
| `boy` | 남아 |
| `girl` | 여아 |
| `animal` | 동물 (선택적으로 `species_hint` 제공 가능) |

---

## 포스터 템플릿

6개 템플릿 (`poster_01` ~ `poster_06`).

각 템플릿은 `configs/recipes/{template_id}.yaml`로 정의:
- 스타일 참조 이미지 (`ad_image`)
- 프롬프트 템플릿
- 사용 벤더 (`vendors`)
- 후처리 목록 (`post_process`)
- 생성 품질 (`quality`)

---

## 지원 벤더

| 벤더 | 역할 | 상태 |
|------|------|------|
| OpenAI `gpt-image-1` | 포스터 스타일 + 유저 이미지 합성 | 활성 |
| BFL `flux-pro-1.1-ultra` | IP-Adapter 기반 이미지 생성 | 구현 완료 |
| Akool `highquality/specifyimage` | 얼굴 후처리 (생성 후 적용) | 활성 |

---

## 현재 활성 파이프라인 (A+C 조합)

poster_01 기준:

```
유저 이미지 업로드
  → OpenAI images.edit()
      입력: 포스터 스타일 이미지 + 유저 이미지
      출력: 스타일이 적용된 포스터
  → Akool faceswap (highquality/specifyimage)
      modifyImage: 생성된 포스터
      sourceImage: 유저 얼굴
      출력: 유저 얼굴이 정확히 합성된 최종 포스터
  → S3 저장 → result_url 반환
```

**선택 이유**: BFL FLUX는 `image_prompt` 슬롯이 하나뿐이라 스타일 이미지와 유저 얼굴을 동시에 conditioning할 수 없음. OpenAI로 스타일을 살리고 FAL faceswap으로 얼굴 identity를 정확히 보존하는 두 단계 방식을 채택.

---

## 출력

- 최종 포스터 이미지 1장 (S3 저장)
- API 응답: `job_id`, `status`, `result_urls` (벤더별 URL dict)

---

## Non-Goals

- frontend
- 사용자 인증
- 어드민 패널
- 분석/통계
- A/B 테스트
- CDN 연동
- 결제
