#!/bin/bash
set -e

MODEL_DIR="${FRAMEPACK_DIR:-/models/FramePack}"
MODEL_DIR="$(dirname "${MODEL_DIR}")"

echo "============================================"
echo "  FramePack Video Studio - Docker Entrypoint"
echo "============================================"

# ── Run common setup (clone repos, download models, check FlashAttention) ────

source /app/scripts/setup.sh --base-dir "${MODEL_DIR}"

echo ""
echo "Starting server..."
echo "Access at: http://0.0.0.0:8000"
echo "============================================"

# ── Start the server ─────────────────────────────────────────────────────────

exec uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 1
