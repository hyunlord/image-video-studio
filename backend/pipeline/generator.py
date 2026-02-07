"""
Wan 2.1 FLF2V video generation wrapper.

Async interface to the Wan 2.1 generate.py CLI with real-time progress tracking
and cancellation support.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Awaitable, Callable, Optional

from backend.config import DEFAULT_SAMPLE_STEPS, WAN21_DIR, MODEL_CACHE_DIR
from backend.models import TechnicalParams

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Raised when video generation fails."""
    pass


class GenerationCancelled(Exception):
    """Raised when generation is cancelled by user."""
    pass


async def generate_video(
    first_frame: Path,
    last_frame: Path,
    prompt: str,
    output_path: Path,
    params: TechnicalParams,
    progress_callback: Optional[Callable[[float], Awaitable[None]]] = None,
    cancel_event: Optional[asyncio.Event] = None
) -> Path:
    """
    Generate video using Wan 2.1 FLF2V model.

    Executes the Wan 2.1 generate.py CLI asynchronously, parsing stdout for
    progress updates and supporting cancellation.

    Args:
        first_frame: Path to preprocessed first frame image
        last_frame: Path to preprocessed last frame image
        prompt: Text prompt describing desired motion/transition
        output_path: Path to save generated video file
        params: Technical parameters for generation
        progress_callback: Optional async callback for progress updates (0.0-1.0)
        cancel_event: Optional event to signal cancellation

    Returns:
        Path to generated video file

    Raises:
        GenerationError: If generation fails or CLI returns error
        GenerationCancelled: If cancel_event is set during generation
        FileNotFoundError: If generate.py or checkpoint not found
    """
    # Validate inputs
    if not first_frame.exists():
        raise FileNotFoundError(f"First frame not found: {first_frame}")
    if not last_frame.exists():
        raise FileNotFoundError(f"Last frame not found: {last_frame}")

    # Locate generate.py script
    generate_script = WAN21_DIR / "generate.py"
    if not generate_script.exists():
        raise FileNotFoundError(
            f"Wan 2.1 generate.py not found at {generate_script}. "
            f"Check WAN21_DIR configuration."
        )

    # Locate model checkpoint
    ckpt_dir = MODEL_CACHE_DIR
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {ckpt_dir}. "
            f"Check MODEL_CACHE_DIR configuration."
        )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build command arguments
    cmd_args = [
        "python",
        str(generate_script),
        "--task", "flf2v-14B",
        "--ckpt_dir", str(ckpt_dir),
        "--first_frame", str(first_frame),
        "--last_frame", str(last_frame),
        "--prompt", prompt,
        "--frame_num", str(params.frame_num),
        "--sample_steps", str(params.sample_steps),
        "--sample_guide_scale", str(params.guidance_scale),
        "--base_seed", str(params.seed),
        "--size", f"{params.width}*{params.height}",
        "--offload_model", str(params.offload_model),
        "--save_file", str(output_path),
    ]

    # Memory optimization: keep T5 encoder on CPU to free VRAM
    if params.offload_model:
        cmd_args.append("--t5_cpu")

    logger.info(f"Starting generation: {' '.join(cmd_args)}")

    # Subprocess environment: reduce CUDA memory fragmentation
    sub_env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    # Start subprocess
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=WAN21_DIR,
            env=sub_env,
        )
    except Exception as e:
        raise GenerationError(f"Failed to start generation process: {e}") from e

    # Progress tracking patterns
    # Expected patterns: "step 25/50", "25/50", "50%", etc.
    progress_patterns = [
        re.compile(r'step\s+(\d+)\s*/\s*(\d+)', re.IGNORECASE),
        re.compile(r'(\d+)\s*/\s*(\d+)'),
        re.compile(r'(\d+(?:\.\d+)?)\s*%'),
    ]

    stderr_lines = []

    async def read_stdout():
        """Read and parse stdout for progress updates."""
        nonlocal progress_patterns

        if process.stdout is None:
            return

        try:
            async for line_bytes in process.stdout:
                line = line_bytes.decode('utf-8', errors='replace').strip()
                if line:
                    logger.debug(f"[stdout] {line}")

                    # Try to extract progress
                    for pattern in progress_patterns:
                        match = pattern.search(line)
                        if match:
                            if len(match.groups()) == 2:
                                # Pattern: "step X/Y" or "X/Y"
                                current = int(match.group(1))
                                total = int(match.group(2))
                                if total > 0:
                                    progress = current / total
                                    if progress_callback:
                                        await progress_callback(progress)
                            elif len(match.groups()) == 1:
                                # Pattern: "X%"
                                percent = float(match.group(1))
                                progress = percent / 100.0
                                if progress_callback:
                                    await progress_callback(progress)
                            break
        except asyncio.CancelledError:
            logger.info("Stdout reading cancelled")
            raise

    async def read_stderr():
        """Read stderr for error messages."""
        nonlocal stderr_lines

        if process.stderr is None:
            return

        try:
            async for line_bytes in process.stderr:
                line = line_bytes.decode('utf-8', errors='replace').strip()
                if line:
                    logger.warning(f"[stderr] {line}")
                    stderr_lines.append(line)
        except asyncio.CancelledError:
            logger.info("Stderr reading cancelled")
            raise

    async def check_cancellation():
        """Monitor cancel event and terminate process if triggered."""
        if cancel_event is None:
            return

        try:
            await cancel_event.wait()
            logger.info("Cancellation requested, terminating process")
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Process didn't terminate, killing it")
                process.kill()
        except asyncio.CancelledError:
            pass

    # Run all monitoring tasks
    try:
        await asyncio.gather(
            read_stdout(),
            read_stderr(),
            check_cancellation(),
            return_exceptions=True
        )

        # Wait for process to complete
        returncode = await process.wait()

        # Check if cancelled
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled("Video generation was cancelled by user")

        # Check return code
        if returncode != 0:
            error_msg = "\n".join(stderr_lines[-10:]) if stderr_lines else "Unknown error"
            raise GenerationError(
                f"Generation failed with exit code {returncode}. "
                f"Last stderr lines:\n{error_msg}"
            )

        # Verify output file exists
        if not output_path.exists():
            raise GenerationError(
                f"Generation completed but output file not found: {output_path}"
            )

        # Final progress callback
        if progress_callback:
            await progress_callback(1.0)

        logger.info(f"Generation complete: {output_path}")
        return output_path

    except GenerationCancelled:
        logger.info("Generation cancelled")
        raise
    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"Unexpected error during generation: {e}") from e
    finally:
        # Ensure process is cleaned up
        if process.returncode is None:
            process.kill()
            await process.wait()


async def estimate_generation_time(params: TechnicalParams) -> float:
    """
    Estimate generation time in seconds based on parameters.

    This is a rough heuristic based on typical GPU performance.

    Args:
        params: Technical parameters for generation

    Returns:
        Estimated time in seconds
    """
    # Base time per frame (seconds)
    base_time_per_frame = 1.5 if params.offload_model else 0.8

    # Adjust for resolution
    resolution_multiplier = {
        "480P": 1.0,
        "720P": 1.8,
    }.get(params.resolution, 1.0)

    # Adjust for sample steps
    steps_multiplier = params.sample_steps / DEFAULT_SAMPLE_STEPS

    # Calculate total estimate
    estimated_seconds = (
        params.frame_num *
        base_time_per_frame *
        resolution_multiplier *
        steps_multiplier
    )

    return estimated_seconds
