#!/bin/bash
# ============================================================================
# FramePack Video Studio - Common Setup Script
# Usage:
#   bash scripts/setup.sh --base-dir /content          # Colab
#   source scripts/setup.sh --base-dir /models          # Docker (source to export env vars)
# ============================================================================
set -e

# ── Parse arguments ──────────────────────────────────────────────────────────

BASE_DIR="/content"
SKIP_MODEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        --skip-model) SKIP_MODEL=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

FRAMEPACK_DIR="${BASE_DIR}/FramePack"
CODEFORMER_DIR="${BASE_DIR}/CodeFormer"
RIFE_DIR="${BASE_DIR}/RIFE"

echo "============================================"
echo "  FramePack Video Studio - Setup"
echo "  Base directory: ${BASE_DIR}"
echo "============================================"

# ── 1. FramePack ──────────────────────────────────────────────────────────────

if [ ! -d "${FRAMEPACK_DIR}/.git" ]; then
    echo "[1/5] Cloning FramePack..."
    git clone --depth 1 https://github.com/lllyasviel/FramePack.git "${FRAMEPACK_DIR}"
    pip install --no-cache-dir -r "${FRAMEPACK_DIR}/requirements_fp.txt"
    echo "  -> FramePack installed"
else
    echo "[1/5] FramePack already cached"
fi

# ── 2. FlashAttention compatibility check ────────────────────────────────────

echo "[2/5] Checking FlashAttention compatibility..."
python3 -c "
import torch
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    if cc[0] < 8:
        print(f'GPU compute capability {cc[0]}.{cc[1]} < 8.0 (non-Ampere)')
        print('Removing flash-attn (incompatible with this GPU)...')
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'flash-attn'],
                       capture_output=True)
        print('  -> flash-attn removed')
    else:
        print(f'GPU compute capability {cc[0]}.{cc[1]} >= 8.0 - FlashAttention OK')
else:
    print('No CUDA GPU detected - skipping FlashAttention check')
"

# ── 3. FramePack model pre-download ──────────────────────────────────────────

if [ "$SKIP_MODEL" = true ]; then
    echo "[3/5] Skipping model download (--skip-model)"
else
    echo "[3/5] Pre-downloading FramePack models (~20GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN') or None

# Main transformer
snapshot_download(
    repo_id='lllyasviel/FramePackI2V_HY',
    token=token,
)
print('  -> FramePack transformer downloaded')

# HunyuanVideo base (text encoders, VAE, tokenizers)
snapshot_download(
    repo_id='hunyuanvideo-community/HunyuanVideo',
    allow_patterns=['text_encoder/*', 'text_encoder_2/*', 'tokenizer/*', 'tokenizer_2/*', 'vae/*'],
    token=token,
)
print('  -> HunyuanVideo text encoders & VAE downloaded')

# Vision encoder (flux redux)
snapshot_download(
    repo_id='lllyasviel/flux_redux_bfl',
    allow_patterns=['feature_extractor/*', 'image_encoder/*'],
    token=token,
)
print('  -> Vision encoder downloaded')
"
fi

# ── 4. CodeFormer ────────────────────────────────────────────────────────────

if [ ! -d "${CODEFORMER_DIR}/.git" ]; then
    echo "[4/5] Cloning CodeFormer..."
    git clone --depth 1 https://github.com/sczhou/CodeFormer.git "${CODEFORMER_DIR}"
    pip install --no-cache-dir -r "${CODEFORMER_DIR}/requirements.txt"
    (cd "${CODEFORMER_DIR}" && python3 scripts/download_pretrained_models.py CodeFormer)
    echo "  -> CodeFormer installed"
else
    echo "[4/5] CodeFormer already cached"
fi

# Real-ESRGAN (needed by CodeFormer and standalone upscaling)
python3 -c "import realesrgan" 2>/dev/null || pip install --no-cache-dir realesrgan -q

# ── 5. RIFE ──────────────────────────────────────────────────────────────────

if [ ! -d "${RIFE_DIR}/.git" ]; then
    echo "[5/5] Cloning RIFE..."
    git clone --depth 1 https://github.com/hzwer/ECCV2022-RIFE.git "${RIFE_DIR}"
    echo "  -> RIFE installed"
else
    echo "[5/5] RIFE already cached"
fi

# ── Export environment variables ─────────────────────────────────────────────

export FRAMEPACK_DIR="${FRAMEPACK_DIR}"
export CODEFORMER_DIR="${CODEFORMER_DIR}"
export RIFE_DIR="${RIFE_DIR}"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  FRAMEPACK_DIR=${FRAMEPACK_DIR}"
echo "  CODEFORMER_DIR=${CODEFORMER_DIR}"
echo "  RIFE_DIR=${RIFE_DIR}"
echo "============================================"
