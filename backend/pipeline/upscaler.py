"""Real-ESRGAN video upscaling module.

This module provides async video upscaling using Real-ESRGAN with frame extraction,
batch processing, and progress tracking.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from backend.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


# Optional Real-ESRGAN import with graceful degradation
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    REALESRGAN_AVAILABLE = True
except ImportError:
    logger.warning(
        "Real-ESRGAN not installed. Upscaling will be skipped. "
        "Install with: pip install realesrgan basicsr"
    )
    REALESRGAN_AVAILABLE = False
    RRDBNet = None
    RealESRGANer = None


class UpscalerError(RuntimeError):
    """Raised when upscaling operations fail."""

    pass


def _extract_frames(video_path: Path, output_dir: Path) -> tuple[list[Path], float]:
    """Extract all frames from video using OpenCV.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames

    Returns:
        Tuple of (list of frame paths, fps)

    Raises:
        UpscalerError: If video cannot be read or frames cannot be extracted
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise UpscalerError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0  # fallback to default

    frame_paths = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = output_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            frame_idx += 1

        if not frame_paths:
            raise UpscalerError("No frames extracted from video")

        logger.info(f"Extracted {len(frame_paths)} frames at {fps} fps")
        return frame_paths, fps

    finally:
        cap.release()


def _reassemble_video(
    frame_paths: list[Path], output_path: Path, fps: float, original_video: Path
) -> None:
    """Reassemble frames into video using OpenCV with audio preservation.

    Args:
        frame_paths: List of frame image paths in order
        output_path: Path for output video file
        fps: Frame rate for output video
        original_video: Original video path for audio extraction

    Raises:
        UpscalerError: If video cannot be written or audio cannot be merged
    """
    if not frame_paths:
        raise UpscalerError("No frames to reassemble")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise UpscalerError(f"Cannot read first frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]

    # Create temporary video without audio
    temp_video = output_path.with_suffix(".temp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Cannot read frame: {frame_path}, skipping")
                continue
            out.write(frame)

        out.release()

        # Merge audio from original video using ffmpeg
        _merge_audio_sync(original_video, temp_video, output_path)

        logger.info(f"Reassembled {len(frame_paths)} frames into {output_path}")

    finally:
        out.release()
        if temp_video.exists():
            temp_video.unlink()


def _merge_audio_sync(
    source_video: Path, video_no_audio: Path, output_video: Path
) -> None:
    """Merge audio from source video into video without audio (synchronous helper).

    Args:
        source_video: Original video with audio
        video_no_audio: Video without audio
        output_video: Final output video path

    Raises:
        UpscalerError: If ffmpeg command fails
    """
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-i",
        str(video_no_audio),
        "-i",
        str(source_video),
        "-c:v",
        "copy",  # copy video stream
        "-c:a",
        "aac",  # encode audio to aac
        "-map",
        "0:v:0",  # video from first input
        "-map",
        "1:a:0?",  # audio from second input (optional)
        "-shortest",  # match shortest stream
        str(output_video),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        logger.debug(f"Audio merge completed: {result.stdout}")
    except subprocess.CalledProcessError as e:
        # If no audio stream, just copy video
        if "does not contain any stream" in e.stderr or "No such stream" in e.stderr:
            logger.info("No audio stream found, copying video only")
            shutil.copy(video_no_audio, output_video)
        else:
            raise UpscalerError(f"Audio merge failed: {e.stderr}") from e
    except subprocess.TimeoutExpired:
        raise UpscalerError("Audio merge timed out after 5 minutes")


async def _merge_audio_async(
    source_video: Path, video_no_audio: Path, output_video: Path
) -> None:
    """Async wrapper for audio merging."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, _merge_audio_sync, source_video, video_no_audio, output_video
    )


def _upscale_frame_batch(
    upsampler: "RealESRGANer",
    frame_paths: list[Path],
    output_dir: Path,
    scale: int,
) -> list[Path]:
    """Upscale a batch of frames using Real-ESRGAN.

    Args:
        upsampler: Initialized RealESRGANer instance
        frame_paths: List of frame paths to upscale
        output_dir: Directory to save upscaled frames
        scale: Upscale factor

    Returns:
        List of upscaled frame paths

    Raises:
        UpscalerError: If upscaling fails
    """
    upscaled_paths = []

    for frame_path in frame_paths:
        img_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise UpscalerError(f"Cannot read frame: {frame_path}")

        try:
            output, _ = upsampler.enhance(img_bgr, outscale=scale)
        except Exception as e:
            raise UpscalerError(f"Upscaling failed for {frame_path}: {e}") from e

        output_path = output_dir / frame_path.name
        cv2.imwrite(str(output_path), output)
        upscaled_paths.append(output_path)

    return upscaled_paths


async def upscale_video(
    video_path: Path,
    output_path: Path,
    scale: int = 2,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """Upscale video using Real-ESRGAN with frame extraction and reassembly.

    Process:
    1. Extract all frames from video
    2. Upscale frames in batches using Real-ESRGAN
    3. Reassemble frames into video with original audio

    Args:
        video_path: Path to input video file
        output_path: Path for output video file
        scale: Upscale factor (default 2x)
        progress_callback: Optional callback(progress_percent, message)

    Returns:
        Path to upscaled video file

    Raises:
        UpscalerError: If upscaling fails or Real-ESRGAN is not available
    """
    if not REALESRGAN_AVAILABLE:
        logger.warning("Real-ESRGAN not available, copying original video")
        shutil.copy(video_path, output_path)
        return output_path

    if not video_path.exists():
        raise UpscalerError(f"Video file not found: {video_path}")

    logger.info(f"Starting video upscale: {video_path} -> {output_path} (scale={scale})")

    # Create temporary directories for frame processing
    temp_dir = Path(tempfile.mkdtemp(prefix="upscaler_"))
    frames_dir = temp_dir / "frames"
    upscaled_dir = temp_dir / "upscaled"
    frames_dir.mkdir(parents=True)
    upscaled_dir.mkdir(parents=True)

    try:
        # Step 1: Extract frames
        if progress_callback:
            progress_callback(10.0, "Extracting video frames")

        frame_paths, fps = await asyncio.get_event_loop().run_in_executor(
            None, _extract_frames, video_path, frames_dir
        )

        total_frames = len(frame_paths)
        logger.info(f"Extracted {total_frames} frames for upscaling")

        # Step 2: Initialize Real-ESRGAN upsampler
        if progress_callback:
            progress_callback(15.0, "Initializing Real-ESRGAN model")

        def _init_upsampler() -> "RealESRGANer":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=scale,
            )
            return RealESRGANer(
                scale=scale,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                model=model,
                half=True,  # FP16 for faster inference
                device="cuda",
            )

        upsampler = await asyncio.get_event_loop().run_in_executor(None, _init_upsampler)

        # Step 3: Upscale frames in batches
        batch_size = 10  # Process 10 frames at a time
        upscaled_paths = []

        for batch_idx in range(0, total_frames, batch_size):
            batch = frame_paths[batch_idx : batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            total_batches = (total_frames + batch_size - 1) // batch_size

            if progress_callback:
                progress = 15.0 + (70.0 * batch_idx / total_frames)
                progress_callback(
                    progress, f"Upscaling frames (batch {batch_num}/{total_batches})"
                )

            batch_upscaled = await asyncio.get_event_loop().run_in_executor(
                None, _upscale_frame_batch, upsampler, batch, upscaled_dir, scale
            )
            upscaled_paths.extend(batch_upscaled)

            logger.info(f"Upscaled batch {batch_num}/{total_batches}")

        # Step 4: Reassemble video
        if progress_callback:
            progress_callback(85.0, "Reassembling video with audio")

        await asyncio.get_event_loop().run_in_executor(
            None, _reassemble_video, upscaled_paths, output_path, fps, video_path
        )

        if progress_callback:
            progress_callback(100.0, "Upscaling complete")

        logger.info(f"Video upscaling completed: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Upscaling failed: {e}")
        raise UpscalerError(f"Video upscaling failed: {e}") from e

    finally:
        # Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
