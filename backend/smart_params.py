from __future__ import annotations

import logging
from pathlib import Path

from backend.config import DEFAULT_SAMPLE_STEPS, DEFAULT_SEED, RESOLUTION_MAP
from backend.models import (
    ImageAnalysis,
    JobRequest,
    PostProcessMode,
    PromptAnalysis,
    TechnicalParams,
    TransitionIntensity,
    VideoLength,
    VideoQuality,
)
from backend.utils.gpu import detect_gpu_tier

logger = logging.getLogger(__name__)

# ── Base mappings ────────────────────────────────────────────────────────────

LENGTH_TO_FRAMES = {
    VideoLength.SHORT: 33,
    VideoLength.MEDIUM: 65,
    VideoLength.LONG: 129,
}

QUALITY_TO_RES = {
    VideoQuality.STANDARD: "480P",
    VideoQuality.HD: "720P",
}

INTENSITY_TO_SCALE = {
    TransitionIntensity.GENTLE: 5.0,
    TransitionIntensity.NORMAL: 10.0,
    TransitionIntensity.DYNAMIC: 15.0,
}


def map_parameters(
    request: JobRequest,
    image_analysis: ImageAnalysis,
    prompt_analysis: PromptAnalysis,
) -> TechnicalParams:
    """Convert user-friendly request + analysis into technical parameters."""
    tier_name, gpu_profile = detect_gpu_tier()

    # ── Base from user selections ────────────────────────────────────────
    frame_num = LENGTH_TO_FRAMES[request.video_length]
    resolution = QUALITY_TO_RES[request.video_quality]
    guidance_scale = INTENSITY_TO_SCALE[request.transition_intensity]
    sample_steps = DEFAULT_SAMPLE_STEPS
    seed = request.seed if request.seed is not None else DEFAULT_SEED

    # ── GPU constraints ──────────────────────────────────────────────────
    max_frames = gpu_profile["max_frames"]
    if frame_num > max_frames:
        logger.info("Clamping frames %d → %d for %s", frame_num, max_frames, tier_name)
        frame_num = max_frames

    offload = gpu_profile["offload"]

    # ── Image analysis adjustments ───────────────────────────────────────
    # Low similarity → more steps for smoother transition
    if image_analysis.clip_similarity < 0.5:
        steps_boost = int((0.5 - image_analysis.clip_similarity) * 50)
        sample_steps = min(sample_steps + steps_boost, 50)
        logger.info("Low similarity (%.2f) → steps boosted to %d",
                     image_analysis.clip_similarity, sample_steps)
    elif image_analysis.clip_similarity > 0.8:
        sample_steps = max(sample_steps - 5, 15)
        logger.info("High similarity (%.2f) → steps reduced to %d",
                     image_analysis.clip_similarity, sample_steps)

    # ── Prompt analysis adjustments ──────────────────────────────────────
    guidance_scale += prompt_analysis.guidance_delta
    guidance_scale = max(1.0, min(20.0, guidance_scale))

    sample_steps += prompt_analysis.steps_delta
    sample_steps = max(10, min(50, sample_steps))

    # GPU memory cap on steps
    steps_cap = gpu_profile.get("sample_steps_cap")
    if steps_cap and sample_steps > steps_cap:
        logger.info("Capping steps %d → %d for %s", sample_steps, steps_cap, tier_name)
        sample_steps = steps_cap

    # ── Resolution dimensions ────────────────────────────────────────────
    res_info = RESOLUTION_MAP[resolution]

    # ── Post-processing decisions ────────────────────────────────────────
    if request.post_process == PostProcessMode.OFF:
        apply_codeformer = False
        apply_upscale = False
        apply_interpolation = False
    elif request.post_process == PostProcessMode.MANUAL:
        apply_codeformer = request.post_process_options.face_restore or False
        apply_upscale = request.post_process_options.upscale or False
        apply_interpolation = request.post_process_options.interpolate or False
    else:
        # AUTO mode: decide based on analysis
        apply_codeformer = _should_codeformer(image_analysis, prompt_analysis)
        apply_upscale = _should_upscale(request, image_analysis)
        apply_interpolation = _should_interpolate(request)

    # CodeFormer fidelity based on face size
    fidelity = 0.7
    if image_analysis.face_ratio > 0.15:
        fidelity = 0.8  # close-up: preserve more
    elif image_analysis.face_ratio > 0:
        fidelity = 0.6  # distant: restore more aggressively

    return TechnicalParams(
        frame_num=frame_num,
        resolution=resolution,
        width=res_info["width"],
        height=res_info["height"],
        guidance_scale=round(guidance_scale, 1),
        sample_steps=sample_steps,
        seed=seed,
        offload_model=offload,
        apply_codeformer=apply_codeformer,
        codeformer_fidelity=fidelity,
        apply_upscale=apply_upscale,
        upscale_factor=2,
        apply_interpolation=apply_interpolation,
    )


def _should_codeformer(analysis: ImageAnalysis, prompt: PromptAnalysis) -> bool:
    return analysis.has_faces or prompt.suggest_codeformer


def _should_upscale(request: JobRequest, analysis: ImageAnalysis) -> bool:
    if request.video_quality == VideoQuality.HD:
        return True
    return analysis.min_width < 720 or analysis.min_height < 480


def _should_interpolate(request: JobRequest) -> bool:
    return request.video_length in (VideoLength.MEDIUM, VideoLength.LONG)
