# Wan 2.1 Video Studio

Wan 2.1 FLF2V(First-Last-Frame-to-Video) 모델 기반의 웹 비디오 생성 스튜디오입니다.
이미지 쌍을 입력하면 두 이미지 사이의 전환 영상을 생성하고, 자동 후처리까지 수행합니다.

---

## 주요 기능

- **다중 이미지 업로드** + 드래그앤드롭 순서 변경
- **유저 친화적 파라미터** -- 영상 길이 / 품질 / 변환 강도를 슬라이더로 조절
- **한글 프롬프트 지원** -- 한국어 입력 시 자동 영어 번역 (googletrans)
- **실시간 진행률 표시** -- WebSocket을 통한 단계별 상태 업데이트
- **배치 처리** -- 여러 이미지 쌍을 순차 생성 후 하나의 영상으로 결합
- **자동/수동 후처리 옵션** -- 얼굴 복원, 업스케일링, 프레임 보간을 개별 선택 가능
- **GPU 모니터링** -- `/monitor`에서 VRAM/RAM 사용량 실시간 확인

---

## 실행 방법

### 방법 A: Google Colab (무료/저비용)

빠른 테스트에 적합합니다. GPU 환경 설정이 자동으로 처리됩니다.

**요구사항**
- Google Colab T4 GPU (무료 티어) 이상
- **High-RAM 런타임** 필수 (14B 모델 = ~28GB CPU RAM)
  - 런타임 → 런타임 유형 변경 → High-RAM 선택

**실행 절차**
1. `colab_launch.ipynb`를 Google Colab에서 엽니다
2. 셀을 순서대로 실행합니다 (코드 셀 3개)
3. 출력된 ngrok URL을 클릭하여 접속합니다

**T4 제약사항**
| 항목 | 제한 |
|------|------|
| 해상도 | 480P (832x480) |
| 최대 프레임 | 33 |
| 최대 스텝 | 30 |
| VRAM | 16GB |

**선택사항: `.env` 파일**

Colab 작업 디렉토리에 `.env` 파일을 업로드하면 자동 로드됩니다:
```
HF_TOKEN=hf_your_token_here
NGROK_AUTH_TOKEN=your_ngrok_token
```

---

### 방법 B: Docker Compose (GPU 클라우드)

RunPod, Vast.ai, GCE 등에서 사용합니다. VSCode SSH 원격 개발을 지원합니다.

**요구사항**
- NVIDIA GPU 16GB+ VRAM (T4, L4, A10G, A100)
- Docker + NVIDIA Container Toolkit
- 디스크 ~50GB (모델 + 코드)

**실행 절차**

```bash
git clone https://github.com/hyunlord/image-video-studio.git
cd image-video-studio
cp .env.sample .env    # HF_TOKEN 설정
cd docker
docker compose up
```

첫 실행 시 모델 다운로드 (~28GB, 10-15분 소요).
이후 실행부터는 Docker 볼륨에 캐시되어 즉시 시작됩니다.

브라우저에서 `http://localhost:8000`으로 접속합니다.

**VSCode 원격 개발**
1. VSCode에서 "Remote - SSH" 확장 설치
2. GPU 클라우드 인스턴스에 SSH 접속
3. `image-video-studio` 디렉토리 열기
4. `localhost:8000`으로 접속 (필요시 포트 포워딩)

**Docker 모델 관리**

모델은 Docker 볼륨(`models`)에 영구 저장됩니다:
```bash
docker compose down -v           # 전체 볼륨 삭제 (모델 포함)
docker volume rm docker_models   # 모델만 삭제 (업로드/출력 유지)
```

---

### 방법 C: 직접 실행 (로컬 GPU)

로컬 GPU 환경에서 직접 실행하려면:

```bash
git clone https://github.com/hyunlord/image-video-studio.git
cd image-video-studio
pip install -r requirements.txt
bash scripts/setup.sh --base-dir /path/to/models
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

`setup.sh`가 Wan 2.1, 모델 다운로드, 후처리 도구를 자동 설치합니다.

---

## GPU별 성능 가이드

| GPU | VRAM | 최대 해상도 | 최대 프레임 | Offload | 비고 |
|-----|------|-----------|------------|---------|------|
| T4 | 16GB | 480P | 33 | Yes | Colab 무료 티어 |
| L4 | 24GB | 720P | 49 | Yes | |
| A10G | 24GB | 720P | 81 | Yes | |
| A100 40GB | 40GB | 720P | 81 | No | |
| A100 80GB | 80GB | 720P | 81 | No | |

---

## 환경 변수

`.env.sample`을 참고하여 `.env` 파일을 생성합니다. `setup.sh`가 자동으로 설정하지만, 직접 지정도 가능합니다.

| 변수 | 설명 | Colab 기본값 | Docker 기본값 |
|------|------|-------------|--------------|
| `HF_TOKEN` | Hugging Face 토큰 (모델 다운로드) | - | - |
| `NGROK_AUTH_TOKEN` | ngrok 인증 토큰 (Colab용) | - | - |
| `WAN21_DIR` | Wan 2.1 설치 경로 | `/content/Wan2.1` | `/models/Wan2.1` |
| `CODEFORMER_DIR` | CodeFormer 설치 경로 | `/content/CodeFormer` | `/models/CodeFormer` |
| `RIFE_DIR` | RIFE 설치 경로 | `/content/RIFE` | `/models/RIFE` |
| `MODEL_CACHE_DIR` | 모델 체크포인트 경로 | `/content/Wan2.1/ckpts/FLF2V-14B-720P` | `/models/Wan2.1/ckpts/FLF2V-14B-720P` |

---

## 프로젝트 구조

```
image-video-studio/
├── backend/
│   ├── app.py                  # FastAPI 메인 애플리케이션
│   ├── config.py               # 경로, GPU 프로필, 생성 기본값 설정
│   ├── models.py               # Pydantic 요청/응답 모델
│   ├── smart_params.py         # 스마트 파라미터 매핑
│   ├── ws_manager.py           # WebSocket 연결 관리
│   ├── job_queue.py            # 비동기 작업 큐
│   ├── analyzers/
│   │   ├── image_analyzer.py   # CLIP 기반 이미지 분석
│   │   └── prompt_analyzer.py  # 프롬프트 분석 + 한영 번역
│   ├── pipeline/
│   │   ├── preprocessor.py     # 이미지 전처리 (리사이즈, 정규화)
│   │   ├── generator.py        # Wan 2.1 FLF2V 영상 생성
│   │   ├── face_restore.py     # CodeFormer 얼굴 복원
│   │   ├── upscaler.py         # Real-ESRGAN 업스케일링
│   │   ├── interpolator.py     # RIFE 프레임 보간
│   │   └── concatenator.py     # ffmpeg 영상 결합
│   └── utils/
│       ├── gpu.py              # GPU 감지 및 모니터링
│       ├── files.py            # 업로드/출력 파일 관리
│       └── progress.py         # 진행률 추적 및 보고
├── frontend/
│   ├── index.html              # React 18 SPA (CDN 로드)
│   └── monitor.html            # GPU 모니터링 대시보드
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
├── scripts/
│   └── setup.sh                # 공통 설치 스크립트 (Colab/Docker 공용)
├── static/
│   ├── uploads/                # 업로드 이미지 저장
│   └── outputs/                # 생성 영상 출력
├── colab_launch.ipynb          # Google Colab 실행 노트북
├── .env.sample                 # 환경변수 템플릿
└── requirements.txt            # Python 의존성
```

---

## 사용된 기술

| 기술 | 용도 |
|------|------|
| [Wan 2.1](https://github.com/Wan-Video/Wan2.1) (Alibaba) | Flow Matching + DiT 기반 영상 생성 |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | 이미지 유사도 측정 및 분석 |
| [CodeFormer](https://github.com/sczhou/CodeFormer) | 얼굴 복원 |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 영상 업스케일링 |
| [RIFE](https://github.com/hzwer/ECCV2022-RIFE) | 프레임 보간 (24fps -> 48fps) |
| [FastAPI](https://fastapi.tiangolo.com/) + WebSocket | 비동기 백엔드 서버 |
| [React 18](https://react.dev/) + [Tailwind CSS](https://tailwindcss.com/) | 프론트엔드 UI |

---

## 라이선스

MIT License
