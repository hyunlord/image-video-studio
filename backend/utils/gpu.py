from __future__ import annotations

import logging

from backend.config import DEFAULT_GPU_PROFILE, GPU_PROFILES

logger = logging.getLogger(__name__)

_cached_profile: dict | None = None


def detect_gpu_tier() -> tuple[str, dict]:
    """Detect GPU and return (tier_name, profile_dict)."""
    global _cached_profile
    if _cached_profile is not None:
        return _cached_profile

    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU fallback profile")
            result = ("CPU", DEFAULT_GPU_PROFILE)
            _cached_profile = result
            return result

        name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)

        # Match known profiles
        for tier, profile in GPU_PROFILES.items():
            if tier.lower().replace("-", "") in name.lower().replace("-", "").replace(" ", ""):
                logger.info("Detected GPU: %s → tier %s (%.1f GB)", name, tier, vram_gb)
                result = (tier, {**profile, "vram_gb": round(vram_gb, 1)})
                _cached_profile = result
                return result

        # Fallback: estimate from VRAM
        if vram_gb >= 70:
            tier = "A100-80GB"
        elif vram_gb >= 35:
            tier = "A100-40GB"
        elif vram_gb >= 20:
            tier = "L4"
        elif vram_gb >= 14:
            tier = "T4"
        else:
            tier = "unknown"

        profile = GPU_PROFILES.get(tier, DEFAULT_GPU_PROFILE)
        profile = {**profile, "vram_gb": round(vram_gb, 1)}
        logger.info("GPU %s (%.1f GB) → estimated tier %s", name, vram_gb, tier)
        result = (tier, profile)
        _cached_profile = result
        return result

    except Exception as e:
        logger.error("GPU detection failed: %s", e)
        result = ("CPU", DEFAULT_GPU_PROFILE)
        _cached_profile = result
        return result


def get_vram_free_gb() -> float:
    """Return free VRAM in GB. Returns 0 if CUDA unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info(0)
        return free / (1024 ** 3)
    except Exception:
        return 0.0
