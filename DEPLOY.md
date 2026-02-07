# Deployment Guide

Wan 2.1 Video Studio supports two deployment paths.

## Path A: Google Colab (Free / Low-cost)

Best for quick testing. Requires a GPU runtime.

### Requirements
- Google Colab with **T4 GPU** (free tier) or better
- **High-RAM runtime** required for the 14B model (~28GB CPU RAM)
  - Runtime > Change runtime type > High-RAM

### Steps
1. Open `colab_launch.ipynb` in Google Colab
2. Run each cell in order (3 cells total)
3. Click the ngrok URL to access the studio

### T4 Limitations
| Setting | Limit |
|---------|-------|
| Resolution | 480P (832x480) |
| Max frames | 33 |
| Max steps | 30 |
| VRAM | 16GB |

### Optional: `.env` file
Upload a `.env` file to your Colab working directory for persistent config:
```
HF_TOKEN=hf_your_token_here
NGROK_AUTH_TOKEN=your_ngrok_token
```

---

## Path B: GPU Cloud (RunPod / Vast.ai / GCE)

Best for production use. Full Docker Compose support with VSCode SSH.

### Requirements
- NVIDIA GPU with 16GB+ VRAM (T4, L4, A10G, A100)
- Docker + NVIDIA Container Toolkit
- ~50GB disk space (models + code)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/hyunlord/image-video-studio.git
cd image-video-studio
```

2. Create `.env` from sample:
```bash
cp .env.sample .env
# Edit .env and set HF_TOKEN
```

3. Start with Docker Compose:
```bash
cd docker
docker compose up
```

First run downloads models (~28GB) and may take 10-15 minutes.
Subsequent runs start in seconds (models are cached in a Docker volume).

4. Access at `http://localhost:8000`

### VSCode Remote Development

Connect VSCode to your GPU cloud instance via SSH:
1. Install "Remote - SSH" extension in VSCode
2. Connect to your cloud instance
3. Open the `image-video-studio` directory
4. Server runs at `localhost:8000` (port-forward if needed)

### GPU Performance Guide

| GPU | VRAM | Max Res | Max Frames | Offload |
|-----|------|---------|------------|---------|
| T4 | 16GB | 480P | 33 | Yes |
| L4 | 24GB | 720P | 49 | Yes |
| A10G | 24GB | 720P | 81 | Yes |
| A100 40GB | 40GB | 720P | 81 | No |
| A100 80GB | 80GB | 720P | 81 | No |

### Managing Models

Models persist in a Docker volume (`models`). To clear cached models:
```bash
docker compose down -v  # removes all volumes including models
```

To keep uploads/outputs but re-download models:
```bash
docker volume rm docker_models
```
