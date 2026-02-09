from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── User-facing enums (friendly names) ───────────────────────────────────────

class VideoLength(str, Enum):
    SHORT = "short"    # ~1s, 33 frames
    MEDIUM = "medium"  # ~2s, 49 frames
    LONG = "long"      # ~3s, 81 frames


class VideoQuality(str, Enum):
    STANDARD = "standard"  # 480P
    HD = "hd"              # 720P


class TransitionIntensity(str, Enum):
    GENTLE = "gentle"    # guidance_scale ~3.5
    NORMAL = "normal"    # guidance_scale ~5.0
    DYNAMIC = "dynamic"  # guidance_scale ~7.0


class PostProcessMode(str, Enum):
    AUTO = "auto"      # backend decides based on analysis
    MANUAL = "manual"  # user controls each toggle
    OFF = "off"        # skip all post-processing


# ── Job status ───────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    GENERATING = "generating"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── Request / Response models ────────────────────────────────────────────────

class PostProcessOptions(BaseModel):
    face_restore: Optional[bool] = None   # None = auto
    upscale: Optional[bool] = None        # None = auto
    interpolate: Optional[bool] = None    # None = auto


class JobRequest(BaseModel):
    image_ids: list[str] = Field(..., min_length=2)
    prompt: str = Field(..., min_length=1, max_length=500)
    video_length: VideoLength = VideoLength.MEDIUM
    video_quality: VideoQuality = VideoQuality.STANDARD
    transition_intensity: TransitionIntensity = TransitionIntensity.NORMAL
    post_process: PostProcessMode = PostProcessMode.AUTO
    post_process_options: PostProcessOptions = PostProcessOptions()
    seed: Optional[int] = None


class JobProgress(BaseModel):
    job_id: str
    status: JobStatus
    stage: str = ""
    progress: float = 0.0          # 0-100
    current_pair: int = 0
    total_pairs: int = 0
    eta_seconds: Optional[float] = None
    message: str = ""


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    progress: float = 0.0
    video_url: Optional[str] = None
    error: Optional[str] = None


class ImageInfo(BaseModel):
    id: str
    filename: str
    url: str
    width: int
    height: int
    has_faces: bool = False


class SystemConfig(BaseModel):
    gpu_name: str
    gpu_tier: str
    vram_gb: float
    max_frames: int
    safe_resolution: str
    available_lengths: list[str]
    available_qualities: list[str]


class SystemMonitor(BaseModel):
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    vram_free_gb: float = 0.0
    vram_usage_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    torch_allocated_gb: float = 0.0
    torch_reserved_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_usage_percent: float = 0.0
    active_job_id: Optional[str] = None
    active_job_stage: str = ""
    active_job_progress: float = 0.0


# ── Internal technical params (not exposed to frontend) ──────────────────────

class TechnicalParams(BaseModel):
    frame_num: int = 65
    resolution: str = "720P"
    width: int = 1280
    height: int = 720
    guidance_scale: float = 10.0
    sample_steps: int = 25
    seed: int = 42
    offload_model: bool = True
    # post-processing flags
    apply_codeformer: bool = False
    codeformer_fidelity: float = 0.7
    apply_upscale: bool = False
    upscale_factor: int = 2
    apply_interpolation: bool = False


class ImageAnalysis(BaseModel):
    has_faces: bool = False
    face_ratio: float = 0.0         # face area / image area
    avg_resolution: float = 0.0
    min_width: int = 0
    min_height: int = 0
    clip_similarity: float = 0.0    # between image pairs


class PromptAnalysis(BaseModel):
    translated_prompt: str = ""
    is_korean: bool = False
    motion_intensity: float = 0.0   # -1 (slow) to +1 (fast)
    suggest_codeformer: bool = False
    guidance_delta: float = 0.0
    steps_delta: int = 0
