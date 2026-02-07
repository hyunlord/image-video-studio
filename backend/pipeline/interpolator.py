"""RIFE frame interpolation module.

This module provides async frame interpolation using RIFE with frame extraction,
CLI execution, and video reassembly.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional

import cv2

from backend.config import INTERPOLATED_FPS, RIFE_DIR

logger = logging.getLogger(__name__)


class InterpolatorError(RuntimeError):
    """Raised when interpolation operations fail."""

    pass


def _check_rife_installation() -> bool:
    """Check if RIFE is installed and available.

    Returns:
        True if RIFE is installed, False otherwise
    """
    if not RIFE_DIR.exists():
        logger.warning(f"RIFE directory not found: {RIFE_DIR}")
        return False

    inference_script = RIFE_DIR / "inference_video.py"
    if not inference_script.exists():
        logger.warning(f"RIFE inference script not found: {inference_script}")
        return False

    return True


def _extract_frames_opencv(video_path: Path, output_dir: Path) -> tuple[int, float]:
    """Extract all frames from video using OpenCV.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames

    Returns:
        Tuple of (frame count, fps)

    Raises:
        InterpolatorError: If video cannot be read or frames cannot be extracted
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise InterpolatorError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0  # fallback

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = output_dir / f"{frame_idx:08d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_idx += 1

        if frame_idx == 0:
            raise InterpolatorError("No frames extracted from video")

        logger.info(f"Extracted {frame_idx} frames at {fps} fps")
        return frame_idx, fps

    finally:
        cap.release()


async def _run_rife_inference(
    frames_dir: Path,
    output_dir: Path,
    exp: int,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> None:
    """Run RIFE inference CLI to interpolate frames.

    Args:
        frames_dir: Directory containing input frames
        output_dir: Directory for interpolated output frames
        exp: Interpolation exponent (2^exp times more frames)
        progress_callback: Optional callback(progress_percent, message)

    Raises:
        InterpolatorError: If RIFE inference fails
    """
    inference_script = RIFE_DIR / "inference_video.py"

    cmd = [
        "python",
        str(inference_script),
        "--img",
        str(frames_dir),
        "--exp",
        str(exp),
        "--output",
        str(output_dir),
    ]

    logger.info(f"Running RIFE inference: {' '.join(cmd)}")

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(RIFE_DIR),
        )

        # Monitor process output
        async def monitor_output():
            if process.stdout:
                async for line in process.stdout:
                    line_str = line.decode().strip()
                    logger.debug(f"RIFE stdout: {line_str}")

                    # Parse progress if available
                    if "frame" in line_str.lower() and progress_callback:
                        # Estimate progress (RIFE doesn't provide detailed progress)
                        progress_callback(50.0, "Interpolating frames")

        async def monitor_errors():
            stderr_lines = []
            if process.stderr:
                async for line in process.stderr:
                    line_str = line.decode().strip()
                    stderr_lines.append(line_str)
                    logger.debug(f"RIFE stderr: {line_str}")
            return stderr_lines

        # Wait for process to complete
        stdout_task = asyncio.create_task(monitor_output())
        stderr_task = asyncio.create_task(monitor_errors())

        await process.wait()
        stderr_lines = await stderr_task
        await stdout_task

        if process.returncode != 0:
            error_msg = "\n".join(stderr_lines) if stderr_lines else "Unknown error"
            raise InterpolatorError(
                f"RIFE inference failed with return code {process.returncode}: {error_msg}"
            )

        logger.info("RIFE inference completed successfully")

    except asyncio.CancelledError:
        if process:
            process.kill()
            await process.wait()
        raise
    except Exception as e:
        raise InterpolatorError(f"RIFE inference execution failed: {e}") from e


def _reassemble_video_sync(
    frames_dir: Path, output_path: Path, fps: float, original_video: Path
) -> None:
    """Reassemble interpolated frames into video with audio.

    Args:
        frames_dir: Directory containing interpolated frames
        output_path: Path for output video file
        fps: Frame rate for output video
        original_video: Original video path for audio extraction

    Raises:
        InterpolatorError: If reassembly fails
    """
    import subprocess

    # Get sorted list of frame files
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        raise InterpolatorError(f"No frames found in {frames_dir}")

    logger.info(f"Reassembling {len(frame_files)} interpolated frames at {fps} fps")

    # Create temporary video without audio using ffmpeg
    temp_video = output_path.with_suffix(".temp.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        str(frames_dir / "*.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(temp_video),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        logger.debug(f"Video assembly completed: {result.stdout}")

    except subprocess.CalledProcessError as e:
        raise InterpolatorError(f"Video assembly failed: {e.stderr}") from e
    except subprocess.TimeoutExpired:
        raise InterpolatorError("Video assembly timed out after 10 minutes")

    # Merge audio from original video
    _merge_audio_sync(original_video, temp_video, output_path)

    # Clean up temp file
    if temp_video.exists():
        temp_video.unlink()


def _merge_audio_sync(
    source_video: Path, video_no_audio: Path, output_video: Path
) -> None:
    """Merge audio from source video into video without audio.

    Args:
        source_video: Original video with audio
        video_no_audio: Video without audio
        output_video: Final output video path

    Raises:
        InterpolatorError: If audio merge fails
    """
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_no_audio),
        "-i",
        str(source_video),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-shortest",
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
            raise InterpolatorError(f"Audio merge failed: {e.stderr}") from e
    except subprocess.TimeoutExpired:
        raise InterpolatorError("Audio merge timed out after 5 minutes")


async def interpolate_video(
    video_path: Path,
    output_path: Path,
    exp: int = 1,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """Interpolate video frames using RIFE to increase frame rate.

    Process:
    1. Extract frames from input video
    2. Run RIFE inference to interpolate frames (2^exp times more frames)
    3. Reassemble interpolated frames into video with original audio

    Args:
        video_path: Path to input video file
        output_path: Path for output video file
        exp: Interpolation exponent (exp=1 doubles frames, exp=2 quadruples, etc.)
        progress_callback: Optional callback(progress_percent, message)

    Returns:
        Path to interpolated video file

    Raises:
        InterpolatorError: If interpolation fails or RIFE is not available
    """
    if not _check_rife_installation():
        logger.warning("RIFE not available, copying original video")
        shutil.copy(video_path, output_path)
        return output_path

    if not video_path.exists():
        raise InterpolatorError(f"Video file not found: {video_path}")

    logger.info(
        f"Starting frame interpolation: {video_path} -> {output_path} (exp={exp})"
    )

    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp(prefix="interpolator_"))
    input_frames_dir = temp_dir / "input_frames"
    output_frames_dir = temp_dir / "output_frames"
    input_frames_dir.mkdir(parents=True)
    output_frames_dir.mkdir(parents=True)

    try:
        # Step 1: Extract frames
        if progress_callback:
            progress_callback(10.0, "Extracting video frames")

        frame_count, original_fps = await asyncio.get_event_loop().run_in_executor(
            None, _extract_frames_opencv, video_path, input_frames_dir
        )

        logger.info(
            f"Extracted {frame_count} frames at {original_fps} fps for interpolation"
        )

        # Step 2: Run RIFE interpolation
        if progress_callback:
            progress_callback(20.0, f"Running RIFE interpolation (exp={exp})")

        await _run_rife_inference(
            input_frames_dir, output_frames_dir, exp, progress_callback
        )

        # Calculate target FPS (original * 2^exp, capped at INTERPOLATED_FPS)
        target_fps = min(original_fps * (2**exp), INTERPOLATED_FPS)
        logger.info(f"Target FPS after interpolation: {target_fps}")

        # Step 3: Reassemble video
        if progress_callback:
            progress_callback(80.0, "Reassembling interpolated video")

        await asyncio.get_event_loop().run_in_executor(
            None,
            _reassemble_video_sync,
            output_frames_dir,
            output_path,
            target_fps,
            video_path,
        )

        if progress_callback:
            progress_callback(100.0, "Interpolation complete")

        logger.info(f"Frame interpolation completed: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        raise InterpolatorError(f"Video interpolation failed: {e}") from e

    finally:
        # Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
