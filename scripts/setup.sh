#!/bin/bash
# ============================================================================
# Wan 2.1 Video Studio - Common Setup Script
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

WAN21_DIR="${BASE_DIR}/Wan2.1"
CODEFORMER_DIR="${BASE_DIR}/CodeFormer"
RIFE_DIR="${BASE_DIR}/RIFE"
CKPT_DIR="${WAN21_DIR}/ckpts/FLF2V-14B-720P"

echo "============================================"
echo "  Wan 2.1 Video Studio - Setup"
echo "  Base directory: ${BASE_DIR}"
echo "============================================"

# ── 1. Wan 2.1 ───────────────────────────────────────────────────────────────

if [ ! -d "${WAN21_DIR}/.git" ]; then
    echo "[1/5] Cloning Wan 2.1..."
    git clone --depth 1 https://github.com/Wan-Video/Wan2.1.git "${WAN21_DIR}"
    pip install --no-cache-dir -r "${WAN21_DIR}/requirements.txt"
    # Wan 2.1 requirements downgrades numpy; fix compatibility
    pip install --no-cache-dir "numpy>=2.0" --force-reinstall -q
    echo "  -> Wan 2.1 installed"
else
    echo "[1/5] Wan 2.1 already cached"
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

# ── 3. FLF2V-14B model download ─────────────────────────────────────────────

if [ "$SKIP_MODEL" = true ]; then
    echo "[3/5] Skipping model download (--skip-model)"
elif [ -d "${CKPT_DIR}" ] && [ -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ]; then
    echo "[3/5] FLF2V-14B model already cached"
else
    echo "[3/5] Downloading FLF2V-14B model (~28GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
import os
token = os.environ.get('HF_TOKEN') or None
snapshot_download(
    repo_id='Wan-AI/Wan2.1-FLF2V-14B-720P',
    local_dir='${CKPT_DIR}',
    token=token,
)
print('  -> Model download complete')
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

export WAN21_DIR="${WAN21_DIR}"
export CODEFORMER_DIR="${CODEFORMER_DIR}"
export RIFE_DIR="${RIFE_DIR}"
export MODEL_CACHE_DIR="${CKPT_DIR}"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  WAN21_DIR=${WAN21_DIR}"
echo "  MODEL_CACHE_DIR=${CKPT_DIR}"
echo "  CODEFORMER_DIR=${CODEFORMER_DIR}"
echo "  RIFE_DIR=${RIFE_DIR}"
echo "============================================"
