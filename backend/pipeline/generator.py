"""
FramePack FLF2V video generation wrapper.

In-process Python API with lazy model loading, real-time progress tracking,
and cancellation support.  Uses reverse-generation to anchor both the first
and last frames of the output video.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import sys
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np
import torch

from backend.config import DEFAULT_SAMPLE_STEPS, FRAMEPACK_DIR, FRAMEPACK_MODEL_ID
from backend.models import TechnicalParams

logger = logging.getLogger(__name__)

# ── Exceptions ────────────────────────────────────────────────────────────────


class GenerationError(Exception):
    """Raised when video generation fails."""
    pass


class GenerationCancelled(Exception):
    """Raised when generation is cancelled by user."""
    pass


# ── Lazy model cache ─────────────────────────────────────────────────────────

_models: dict | None = None


def _ensure_framepack_importable():
    """Add FramePack repo to sys.path so we can import its helpers."""
    fp_str = str(FRAMEPACK_DIR)
    if fp_str not in sys.path:
        sys.path.insert(0, fp_str)


def _load_models() -> dict:
    """Load all FramePack models once, return cached dict."""
    global _models
    if _models is not None:
        return _models

    _ensure_framepack_importable()

    logger.info("Loading FramePack models (first run — may take a few minutes) ...")

    from diffusers import AutoencoderKLHunyuanVideo
    from transformers import (
        CLIPTextModel,
        CLIPTokenizer,
        LlamaModel,
        LlamaTokenizerFast,
        SiglipImageProcessor,
        SiglipVisionModel,
    )
    from diffusers_helper.models.hunyuan_video_packed import (
        HunyuanVideoTransformer3DModelPacked,
    )

    gpu = torch.device("cuda")
    free_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    high_vram = free_mem_gb > 20

    # Text encoders
    hf_base = "hunyuanvideo-community/HunyuanVideo"
    text_encoder = LlamaModel.from_pretrained(
        hf_base, subfolder="text_encoder", torch_dtype=torch.float16
    ).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained(
        hf_base, subfolder="text_encoder_2", torch_dtype=torch.float16
    ).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained(hf_base, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(hf_base, subfolder="tokenizer_2")

    # VAE
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        hf_base, subfolder="vae", torch_dtype=torch.float16
    ).cpu()

    # Vision encoder
    redux_repo = "lllyasviel/flux_redux_bfl"
    feature_extractor = SiglipImageProcessor.from_pretrained(
        redux_repo, subfolder="feature_extractor"
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        redux_repo, subfolder="image_encoder", torch_dtype=torch.float16
    ).cpu()

    # Main transformer
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        FRAMEPACK_MODEL_ID, torch_dtype=torch.bfloat16
    ).cpu()

    # Install dynamic swap for low-VRAM GPUs
    if not high_vram:
        from diffusers_helper.memory import DynamicSwapInstaller
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(vae, device=gpu)
    else:
        transformer.to(gpu)
        vae.to(gpu)

    vae.eval()
    transformer.eval()

    _models = dict(
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        transformer=transformer,
        gpu=gpu,
        high_vram=high_vram,
    )

    logger.info("FramePack models loaded (high_vram=%s)", high_vram)
    return _models


def unload_model():
    """Release all FramePack models from GPU/CPU memory."""
    global _models
    if _models is None:
        return

    for key in list(_models.keys()):
        obj = _models[key]
        if hasattr(obj, "cpu"):
            obj.cpu()
        del _models[key]
    _models = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("FramePack models unloaded")


# ── Main generation function ─────────────────────────────────────────────────


async def generate_video(
    first_frame: Path,
    last_frame: Path,
    prompt: str,
    output_path: Path,
    params: TechnicalParams,
    progress_callback: Optional[Callable[[float], Awaitable[None]]] = None,
    cancel_event: Optional[asyncio.Event] = None,
) -> Path:
    """
    Generate video using FramePack (FLF2V mode).

    Uses the last frame as the input image (FramePack reverse generation)
    and the first frame as an anchor, then reverses the output so the video
    flows naturally from first_frame -> last_frame.

    The function signature is intentionally identical to the previous Wan 2.1
    generator so that job_queue.py requires zero changes.
    """
    if not first_frame.exists():
        raise FileNotFoundError(f"First frame not found: {first_frame}")
    if not last_frame.exists():
        raise FileNotFoundError(f"Last frame not found: {last_frame}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            None,
            _generate_sync,
            first_frame,
            last_frame,
            prompt,
            output_path,
            params,
            progress_callback,
            cancel_event,
            loop,
        )
    except GenerationCancelled:
        logger.info("Generation cancelled")
        raise
    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"Unexpected error during generation: {e}") from e

    if progress_callback:
        await progress_callback(1.0)

    logger.info("Generation complete: %s", output_path)
    return result


def _generate_sync(
    first_frame: Path,
    last_frame: Path,
    prompt: str,
    output_path: Path,
    params: TechnicalParams,
    progress_callback: Optional[Callable] = None,
    cancel_event: Optional[asyncio.Event] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> Path:
    """Blocking helper that runs on a thread-pool executor."""

    _ensure_framepack_importable()

    from PIL import Image
    from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
    from diffusers_helper.utils import (
        crop_or_pad_yield_mask,
        resize_and_center_crop,
        save_bcthw_as_mp4,
        soft_append_bcthw,
    )
    from diffusers_helper.clip_vision import hf_clip_vision_encode
    from diffusers_helper.bucket_tools import find_nearest_bucket

    m = _load_models()
    gpu = m["gpu"]
    high_vram = m["high_vram"]

    # ── Load & preprocess images ─────────────────────────────────────────
    # FramePack generates forward from input_image. We use last_frame as
    # input, generate backward, then reverse the result.
    input_img_np = np.array(Image.open(last_frame).convert("RGB"))
    first_img_np = np.array(Image.open(first_frame).convert("RGB"))

    H, W, _ = input_img_np.shape
    height, width = find_nearest_bucket(H, W, resolution=640)
    input_img_np = resize_and_center_crop(input_img_np, width, height)
    first_img_np = resize_and_center_crop(first_img_np, width, height)

    # ── Encode prompt ────────────────────────────────────────────────────
    if not high_vram:
        m["text_encoder"].to(gpu)
        m["text_encoder_2"].to(gpu)

    llama_vec, clip_l_pooler = encode_prompt_conds(
        prompt,
        m["text_encoder"], m["text_encoder_2"],
        m["tokenizer"], m["tokenizer_2"],
    )
    llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
        "",
        m["text_encoder"], m["text_encoder_2"],
        m["tokenizer"], m["tokenizer_2"],
    )

    if not high_vram:
        m["text_encoder"].cpu()
        m["text_encoder_2"].cpu()
        torch.cuda.empty_cache()

    llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
        llama_vec_n, length=512
    )

    # ── VAE-encode input image ───────────────────────────────────────────
    input_pt = torch.from_numpy(input_img_np).float() / 127.5 - 1
    input_pt = input_pt.permute(2, 0, 1)[None, :, None]
    start_latent = vae_encode(input_pt, m["vae"])

    # VAE-encode first frame (used as end-anchor after reversal)
    first_pt = torch.from_numpy(first_img_np).float() / 127.5 - 1
    first_pt = first_pt.permute(2, 0, 1)[None, :, None]
    end_latent = vae_encode(first_pt, m["vae"])

    # ── Vision encoder ───────────────────────────────────────────────────
    if not high_vram:
        m["image_encoder"].to(gpu)

    image_encoder_output = hf_clip_vision_encode(
        input_img_np, m["feature_extractor"], m["image_encoder"]
    )
    image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

    if not high_vram:
        m["image_encoder"].cpu()
        torch.cuda.empty_cache()

    # ── Generation parameters ────────────────────────────────────────────
    latent_window_size = 9
    section_frames = latent_window_size * 4 - 3  # 33 pixel frames per section
    total_latent_sections = max(1, (params.frame_num - 1) // (section_frames - 1))
    steps = params.sample_steps
    gs = params.guidance_scale
    seed = params.seed

    rng = torch.Generator(device="cpu").manual_seed(seed)

    # ── Section-by-section generation ────────────────────────────────────
    history_latents = start_latent
    history_pixels = None

    for section_idx in range(total_latent_sections):
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled("Video generation was cancelled by user")

        is_last_section = section_idx == total_latent_sections - 1

        # Indices for this section's latent frames
        indices = list(range(latent_window_size + 2))

        # Clean latents: start_latent always at position 0
        clean_list = [start_latent.to(gpu)]
        clean_idx = [0]

        # Context from previous sections
        if section_idx > 0:
            ctx_n = min(3, history_latents.shape[2] - 1)
            ctx = history_latents[:, :, -(ctx_n + 1):-1].to(gpu)
            for i in range(ctx.shape[2]):
                clean_list.append(ctx[:, :, i : i + 1])
                clean_idx.append(i + 1)

        # End anchor on last section
        if is_last_section and total_latent_sections > 1:
            clean_list.append(end_latent.to(gpu))
            clean_idx.append(len(indices) - 1)

        clean_latents = torch.cat(clean_list, dim=2)

        # Progress callback wrapper
        def step_cb(d, _si=section_idx):
            if not progress_callback or not loop:
                return
            cur = d.get("i", 0) + 1
            overall = (_si + cur / steps) / total_latent_sections
            asyncio.run_coroutine_threadsafe(
                progress_callback(min(overall, 0.95)), loop
            )

        generated = sample_hunyuan(
            transformer=m["transformer"],
            sampler="unipc",
            width=width,
            height=height,
            frames=section_frames,
            real_guidance_scale=1.0,
            distilled_guidance_scale=gs,
            guidance_rescale=0.0,
            num_inference_steps=steps,
            generator=rng,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=gpu,
            dtype=torch.bfloat16,
            image_embeddings=image_encoder_last_hidden_state,
            latent_indices=indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_idx,
            callback=step_cb,
        )

        # Prepend start_latent for first section
        generated = torch.cat(
            [start_latent.to(generated), generated], dim=2
        )

        # Decode to pixels
        section_pixels = vae_decode(generated, m["vae"]).cpu()

        if history_pixels is None:
            history_pixels = section_pixels
        else:
            history_pixels = soft_append_bcthw(section_pixels, history_pixels, 3)

        history_latents = torch.cat(
            [history_latents, generated[:, :, 1:].cpu()], dim=2
        )

        torch.cuda.empty_cache()

    # ── Reverse video (last→first becomes first→last) ────────────────────
    if history_pixels is not None and history_pixels.shape[2] > 1:
        history_pixels = history_pixels[:, :, torch.arange(
            history_pixels.shape[2] - 1, -1, -1
        )]

    # ── Save MP4 ─────────────────────────────────────────────────────────
    save_bcthw_as_mp4(history_pixels, str(output_path), fps=30, crf=16)

    return output_path


# ── Time estimation ──────────────────────────────────────────────────────────


async def estimate_generation_time(params: TechnicalParams) -> float:
    """Estimate generation time in seconds based on parameters."""
    base_time_per_frame = 5.0 if params.offload_model else 2.5
    resolution_multiplier = {"480P": 0.7, "720P": 1.0}.get(params.resolution, 1.0)
    steps_multiplier = params.sample_steps / DEFAULT_SAMPLE_STEPS
    return params.frame_num * base_time_per_frame * resolution_multiplier * steps_multiplier
