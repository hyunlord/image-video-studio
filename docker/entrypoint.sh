#!/bin/bash
set -e

MODEL_DIR="${MODEL_CACHE_DIR:-/models}"
WAN21="${MODEL_DIR}/Wan2.1"
CODEFORMER="${MODEL_DIR}/CodeFormer"
RIFE="${MODEL_DIR}/RIFE"

echo "============================================"
echo "  Wan 2.1 Video Studio - Docker Entrypoint"
echo "============================================"

# ── Clone external repos if not cached ───────────────────────────────────────

if [ ! -d "${WAN21}/.git" ]; then
    echo "[1/4] Cloning Wan 2.1..."
    git clone --depth 1 https://github.com/Wan-Video/Wan2.1.git "${WAN21}"
    pip install --no-cache-dir -r "${WAN21}/requirements.txt"
else
    echo "[1/4] Wan 2.1 already cached"
fi

if [ ! -d "${CODEFORMER}/.git" ]; then
    echo "[2/4] Cloning CodeFormer..."
    git clone --depth 1 https://github.com/sczhou/CodeFormer.git "${CODEFORMER}"
    pip install --no-cache-dir -r "${CODEFORMER}/requirements.txt"
    cd "${CODEFORMER}" && python scripts/download_pretrained_models.py CodeFormer && cd /app
else
    echo "[2/4] CodeFormer already cached"
fi

if [ ! -d "${RIFE}/.git" ]; then
    echo "[3/4] Cloning RIFE..."
    git clone --depth 1 https://github.com/hzwer/ECCV2022-RIFE.git "${RIFE}"
else
    echo "[3/4] RIFE already cached"
fi

# ── Download Wan 2.1 FLF2V model if not cached ──────────────────────────────

CKPT_DIR="${WAN21}/ckpts/FLF2V-14B-720P"
if [ ! -d "${CKPT_DIR}" ] || [ -z "$(ls -A ${CKPT_DIR} 2>/dev/null)" ]; then
    echo "[4/4] Downloading FLF2V-14B model (~28GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Wan-AI/Wan2.1-FLF2V-14B-720P',
    local_dir='${CKPT_DIR}',
)
print('Model download complete')
"
else
    echo "[4/4] FLF2V-14B model already cached"
fi

# ── Export paths for the app ─────────────────────────────────────────────────

export WAN21_DIR="${WAN21}"
export CODEFORMER_DIR="${CODEFORMER}"
export RIFE_DIR="${RIFE}"
export MODEL_CACHE_DIR="${CKPT_DIR}"

echo ""
echo "All models ready. Starting server..."
echo "Access at: http://0.0.0.0:8000"
echo "============================================"

# ── Start the server ─────────────────────────────────────────────────────────

exec uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 1
