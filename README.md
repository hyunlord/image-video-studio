# Wan 2.1 Video Studio

Wan 2.1 FLF2V(First-Last-Frame-to-Video) 모델 기반의 웹 비디오 생성 스튜디오입니다.
이미지 쌍을 입력하면 두 이미지 사이의 전환 영상을 생성하고, 자동 후처리까지 수행합니다.

---

## 프로젝트 소개

- **Wan 2.1 FLF2V-14B** 기반 이미지-to-비디오 생성 웹 애플리케이션
- **FastAPI** 비동기 백엔드 + **React 18** 프론트엔드 (CDN 로드, 별도 빌드 불필요)
- CLIP 이미지 분석 + 프롬프트 분석을 통한 **자동 파라미터 최적화**
- 후처리 파이프라인: **CodeFormer** 얼굴 복원, **Real-ESRGAN** 업스케일링, **RIFE** 프레임 보간

---

## 주요 기능

- **다중 이미지 업로드** + 드래그앤드롭 순서 변경
- **유저 친화적 파라미터** -- 영상 길이 / 품질 / 변환 강도를 슬라이더로 조절
- **한글 프롬프트 지원** -- 한국어 입력 시 자동 영어 번역 (googletrans)
- **실시간 진행률 표시** -- WebSocket을 통한 단계별 상태 업데이트
- **배치 처리** -- 여러 이미지 쌍을 순차 생성 후 하나의 영상으로 결합
- **자동/수동 후처리 옵션** -- 얼굴 복원, 업스케일링, 프레임 보간을 개별 선택 가능

---

## 실행 환경

### Google Colab (권장)

가장 간편한 실행 방법입니다. GPU 환경 설정이 자동으로 처리됩니다.

1. `colab_launch.ipynb` 노트북을 Google Colab에서 엽니다.
2. **런타임 > 런타임 유형 변경**에서 GPU(T4 이상)를 선택합니다.
3. 셀을 순서대로 실행합니다.
4. 출력된 ngrok URL로 접속합니다.

### Docker (로컬/클라우드 GPU)

NVIDIA Container Toolkit이 설치된 환경에서 Docker로 실행할 수 있습니다.

```bash
git clone https://github.com/hyunlord/image-video-studio.git
cd image-video-studio
docker compose -f docker/docker-compose.yml up --build
```

브라우저에서 `http://localhost:8000`으로 접속합니다.

### 직접 실행 (로컬 GPU)

로컬 GPU 환경에서 직접 실행하려면 아래 절차를 따릅니다. 사전 요구사항 섹션의 모든 항목이 설치되어야 합니다.

```bash
git clone https://github.com/hyunlord/image-video-studio.git
cd image-video-studio
pip install -r requirements.txt
# Wan 2.1 + 후처리 도구 설치가 필요합니다. 아래 "사전 요구사항" 참고.
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000`으로 접속합니다.

---

## 사전 요구사항 (직접 실행 시)

직접 실행 환경에서는 아래 항목을 수동으로 설치해야 합니다.

| 항목 | 설명 |
|------|------|
| **Python** | 3.10 이상 |
| **CUDA 지원 GPU** | 최소 14GB VRAM (T4 이상) |
| **ffmpeg** | 영상 인코딩/결합에 사용 |
| **git** | 외부 저장소 클론에 사용 |

### 외부 모델 및 도구

**Wan 2.1**

```bash
git clone https://github.com/Wan-Video/Wan2.1.git
```

**FLF2V-14B 모델 체크포인트**

`huggingface_hub`를 통해 다운로드합니다. 상세 절차는 Wan 2.1 공식 저장소를 참고하세요.

**CodeFormer** (얼굴 복원)

```bash
git clone https://github.com/sczhou/CodeFormer.git
```

**RIFE** (프레임 보간)

```bash
git clone https://github.com/hzwer/ECCV2022-RIFE.git
```

**Real-ESRGAN** (업스케일링)

```bash
pip install realesrgan
```

---

## 환경 변수

외부 도구 경로를 환경 변수로 지정할 수 있습니다. 설정하지 않으면 기본값이 사용됩니다.

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `WAN21_DIR` | Wan 2.1 설치 경로 | `/content/Wan2.1` |
| `CODEFORMER_DIR` | CodeFormer 설치 경로 | `/content/CodeFormer` |
| `RIFE_DIR` | RIFE 설치 경로 | `/content/RIFE` |
| `MODEL_CACHE_DIR` | 모델 체크포인트 경로 | `{WAN21_DIR}/ckpts/FLF2V-14B-720P` |

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
│       ├── gpu.py              # GPU 감지 및 프로필 매칭
│       ├── files.py            # 업로드/출력 파일 관리
│       └── progress.py         # 진행률 추적 및 보고
├── frontend/
│   └── index.html              # React 18 SPA (CDN 로드)
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
├── static/
│   ├── uploads/                # 업로드 이미지 저장
│   └── outputs/                # 생성 영상 출력
├── colab_launch.ipynb          # Google Colab 실행 노트북
└── requirements.txt            # Python 의존성
```

---

## GPU별 제한사항

GPU VRAM에 따라 생성 가능한 최대 프레임 수와 해상도가 달라집니다.

| GPU | VRAM | 최대 프레임 | 최대 해상도 | FP8 |
|-----|------|------------|-----------|-----|
| T4 | 16GB | 49 | 480P | 필수 |
| L4 | 24GB | 49 | 720P | 권장 |
| A100 40GB | 40GB | 81 | 720P | 불필요 |
| A100 80GB | 80GB | 81 | 720P | 불필요 |

> FP8 양자화를 사용하면 VRAM 사용량이 줄어들지만 화질이 소폭 저하될 수 있습니다.

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
