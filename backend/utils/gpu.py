from __future__ import annotations

import logging
import subprocess

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


def get_gpu_monitor_data() -> dict:
    """Collect real-time GPU metrics."""
    data = {
        "vram_total_gb": 0.0,
        "vram_used_gb": 0.0,
        "vram_free_gb": 0.0,
        "vram_usage_percent": 0.0,
        "gpu_utilization_percent": 0.0,
        "torch_allocated_gb": 0.0,
        "torch_reserved_gb": 0.0,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return data

        free, total = torch.cuda.mem_get_info(0)
        total_gb = total / (1024 ** 3)
        free_gb = free / (1024 ** 3)
        used_gb = total_gb - free_gb

        data["vram_total_gb"] = round(total_gb, 2)
        data["vram_free_gb"] = round(free_gb, 2)
        data["vram_used_gb"] = round(used_gb, 2)
        data["vram_usage_percent"] = round((used_gb / total_gb) * 100, 1) if total_gb > 0 else 0.0
        data["torch_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
        data["torch_reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2)
    except Exception:
        pass

    data["gpu_utilization_percent"] = _get_gpu_utilization()
    return data


def _get_gpu_utilization() -> float:
    """Query GPU utilization % via nvidia-smi. Returns 0 on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return 0.0


def get_system_ram() -> dict:
    """Return system RAM metrics."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "ram_total_gb": round(mem.total / (1024 ** 3), 2),
            "ram_used_gb": round(mem.used / (1024 ** 3), 2),
            "ram_usage_percent": round(mem.percent, 1),
        }
    except ImportError:
        return {"ram_total_gb": 0.0, "ram_used_gb": 0.0, "ram_usage_percent": 0.0}
