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
FRAMEPACK_DIR = Path(os.getenv("FRAMEPACK_DIR", "/content/FramePack"))
FRAMEPACK_MODEL_ID = os.getenv("FRAMEPACK_MODEL_ID", "lllyasviel/FramePackI2V_HY")
CODEFORMER_DIR = Path(os.getenv("CODEFORMER_DIR", "/content/CodeFormer"))
RIFE_DIR = Path(os.getenv("RIFE_DIR", "/content/RIFE"))

# ── GPU Profiles (FramePack: VRAM-independent frame count) ───────────────────
GPU_PROFILES = {
    "T4": {
        "offload": True,
        "max_frames": 129,
        "safe_res": "720P",
        "vram_gb": 16,
    },
    "L4": {
        "offload": False,
        "max_frames": 129,
        "safe_res": "720P",
        "vram_gb": 24,
    },
    "A10G": {
        "offload": False,
        "max_frames": 129,
        "safe_res": "720P",
        "vram_gb": 24,
    },
    "A100-40GB": {
        "offload": False,
        "max_frames": 129,
        "safe_res": "720P",
        "vram_gb": 40,
    },
    "A100-80GB": {
        "offload": False,
        "max_frames": 129,
        "safe_res": "720P",
        "vram_gb": 80,
    },
}

# Fallback for unknown GPUs
DEFAULT_GPU_PROFILE = {
    "offload": True,
    "max_frames": 65,
    "safe_res": "480P",
    "vram_gb": 8,
}

# ── Generation defaults ──────────────────────────────────────────────────────
DEFAULT_SEED = 42
DEFAULT_SAMPLE_STEPS = 25
DEFAULT_FPS = 30
INTERPOLATED_FPS = 60

# ── Resolution map ───────────────────────────────────────────────────────────
RESOLUTION_MAP = {
    "480P": {"width": 832, "height": 480},
    "720P": {"width": 1280, "height": 720},
}

# ── Upload limits ────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 20
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
