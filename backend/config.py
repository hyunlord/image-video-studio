import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
OUTPUT_DIR = STATIC_DIR / "outputs"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── External tool paths (overridable via env) ────────────────────────────────
WAN21_DIR = Path(os.getenv("WAN21_DIR", "/content/Wan2.1"))
CODEFORMER_DIR = Path(os.getenv("CODEFORMER_DIR", "/content/CodeFormer"))
RIFE_DIR = Path(os.getenv("RIFE_DIR", "/content/RIFE"))
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/content/Wan2.1/ckpts/FLF2V-14B-720P"))

# ── GPU Profiles ─────────────────────────────────────────────────────────────
GPU_PROFILES = {
    "T4": {
        "offload": True,
        "max_frames": 33,
        "safe_res": "480P",
        "vram_gb": 16,
        "sample_steps_cap": 30,
    },
    "L4": {
        "offload": True,
        "max_frames": 49,
        "safe_res": "720P",
        "vram_gb": 24,
    },
    "A10G": {
        "offload": True,
        "max_frames": 81,
        "safe_res": "720P",
        "vram_gb": 24,
    },
    "A100-40GB": {
        "offload": False,
        "max_frames": 81,
        "safe_res": "720P",
        "vram_gb": 40,
    },
    "A100-80GB": {
        "offload": False,
        "max_frames": 81,
        "safe_res": "720P",
        "vram_gb": 80,
    },
}

# Fallback for unknown GPUs
DEFAULT_GPU_PROFILE = {
    "offload": True,
    "max_frames": 33,
    "safe_res": "480P",
    "vram_gb": 8,
}

# ── Generation defaults ──────────────────────────────────────────────────────
DEFAULT_SEED = 42
DEFAULT_SAMPLE_STEPS = 50
DEFAULT_FPS = 24
INTERPOLATED_FPS = 48

# ── Resolution map ───────────────────────────────────────────────────────────
RESOLUTION_MAP = {
    "480P": {"width": 832, "height": 480},
    "720P": {"width": 1280, "height": 720},
}

# ── Upload limits ────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 20
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
