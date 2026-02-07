"""
CodeFormer face restoration wrapper.

Async interface to CodeFormer CLI for face restoration and enhancement in
generated videos.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Awaitable, Callable, Optional

import cv2
import imageio.v3 as iio
import numpy as np

from backend.config import CODEFORMER_DIR, DEFAULT_FPS

logger = logging.getLogger(__name__)


class FaceRestoreError(Exception):
    """Raised when face restoration fails."""
    pass


class FaceRestoreCancelled(Exception):
    """Raised when face restoration is cancelled by user."""
    pass


async def restore_faces(
    video_path: Path,
    output_path: Path,
    fidelity: float = 0.7,
    progress_callback: Optional[Callable[[float], Awaitable[None]]] = None,
    cancel_event: Optional[asyncio.Event] = None
) -> Path:
    """
    Restore and enhance faces in video using CodeFormer.

    Extracts video frames, runs CodeFormer CLI for face restoration with
    background upsampling, then reassembles frames back into video.

    Args:
        video_path: Path to input video file
        output_path: Path to save restored video file
        fidelity: CodeFormer fidelity weight (0-1, higher = more identity preservation)
        progress_callback: Optional async callback for progress updates (0.0-1.0)
        cancel_event: Optional event to signal cancellation

    Returns:
        Path to restored video file

    Raises:
        FaceRestoreError: If restoration fails at any stage
        FaceRestoreCancelled: If cancel_event is set during processing
        FileNotFoundError: If video or CodeFormer not found
    """
    # Validate inputs
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not 0.0 <= fidelity <= 1.0:
        raise ValueError(f"Fidelity must be between 0 and 1, got {fidelity}")

    # Locate CodeFormer script
    codeformer_script = CODEFORMER_DIR / "inference_codeformer.py"
    if not codeformer_script.exists():
        raise FileNotFoundError(
            f"CodeFormer inference_codeformer.py not found at {codeformer_script}. "
            f"Check CODEFORMER_DIR configuration."
        )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary directories for frame extraction and processing
    temp_dir = Path(tempfile.mkdtemp(prefix="face_restore_"))
    frames_dir = temp_dir / "frames"
    output_frames_dir = temp_dir / "output"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting face restoration: {video_path}")
    logger.info(f"Temporary directory: {temp_dir}")

    try:
        # Check for cancellation
        if cancel_event and cancel_event.is_set():
            raise FaceRestoreCancelled("Face restoration cancelled before starting")

        # Step 1: Extract frames (20% of progress)
        logger.info("Extracting video frames")
        fps = await _extract_frames(video_path, frames_dir)
        if progress_callback:
            await progress_callback(0.2)

        # Check for cancellation
        if cancel_event and cancel_event.is_set():
            raise FaceRestoreCancelled("Face restoration cancelled after frame extraction")

        # Step 2: Run CodeFormer (60% of progress)
        logger.info("Running CodeFormer face restoration")
        await _run_codeformer(
            frames_dir,
            output_frames_dir,
            fidelity,
            progress_callback=lambda p: progress_callback(0.2 + p * 0.6) if progress_callback else None,
            cancel_event=cancel_event
        )
        if progress_callback:
            await progress_callback(0.8)

        # Check for cancellation
        if cancel_event and cancel_event.is_set():
            raise FaceRestoreCancelled("Face restoration cancelled after CodeFormer processing")

        # Step 3: Reassemble video (20% of progress)
        logger.info("Reassembling video from restored frames")
        result_frames_dir = output_frames_dir / "final_results"
        if not result_frames_dir.exists():
            raise FaceRestoreError(
                f"CodeFormer output directory not found: {result_frames_dir}"
            )

        await _reassemble_video(result_frames_dir, output_path, fps)
        if progress_callback:
            await progress_callback(1.0)

        # Verify output
        if not output_path.exists():
            raise FaceRestoreError(
                f"Restoration completed but output file not found: {output_path}"
            )

        logger.info(f"Face restoration complete: {output_path}")
        return output_path

    except FaceRestoreCancelled:
        logger.info("Face restoration cancelled")
        raise
    except FaceRestoreError:
        raise
    except Exception as e:
        raise FaceRestoreError(f"Unexpected error during face restoration: {e}") from e
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")


async def _extract_frames(video_path: Path, frames_dir: Path) -> float:
    """
    Extract frames from video to directory.

    Args:
        video_path: Path to input video
        frames_dir: Directory to save extracted frames

    Returns:
        Video FPS (frames per second)

    Raises:
        FaceRestoreError: If frame extraction fails
    """
    loop = asyncio.get_event_loop()

    def _extract() -> float:
        try:
            # Open video with cv2 to get FPS
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise FaceRestoreError(f"Failed to open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                logger.warning(f"Could not detect FPS from video, using default {DEFAULT_FPS}")
                fps = DEFAULT_FPS

            cap.release()

            # Extract frames with imageio
            frames = iio.imread(str(video_path), plugin="pyav")

            # Save frames as images
            for i, frame in enumerate(frames):
                frame_path = frames_dir / f"frame_{i:06d}.png"
                iio.imwrite(frame_path, frame)

            logger.info(f"Extracted {len(frames)} frames at {fps} FPS")
            return fps

        except Exception as e:
            raise FaceRestoreError(f"Failed to extract frames: {e}") from e

    return await loop.run_in_executor(None, _extract)


async def _run_codeformer(
    input_dir: Path,
    output_dir: Path,
    fidelity: float,
    progress_callback: Optional[Callable[[float], Awaitable[None]]] = None,
    cancel_event: Optional[asyncio.Event] = None
) -> None:
    """
    Run CodeFormer CLI on extracted frames.

    Args:
        input_dir: Directory containing input frames
        output_dir: Directory to save restored frames
        fidelity: CodeFormer fidelity weight (0-1)
        progress_callback: Optional callback for progress updates
        cancel_event: Optional event to signal cancellation

    Raises:
        FaceRestoreError: If CodeFormer execution fails
        FaceRestoreCancelled: If cancel_event is set
    """
    codeformer_script = CODEFORMER_DIR / "inference_codeformer.py"

    # Build command arguments
    cmd_args = [
        "python",
        str(codeformer_script),
        "-w", str(fidelity),
        "--input_path", str(input_dir),
        "--output_path", str(output_dir),
        "--bg_upsampler", "realesrgan",
        "--face_upsample",
    ]

    logger.info(f"Running CodeFormer: {' '.join(cmd_args)}")

    # Start subprocess
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=CODEFORMER_DIR
        )
    except Exception as e:
        raise FaceRestoreError(f"Failed to start CodeFormer process: {e}") from e

    stderr_lines = []

    async def read_output():
        """Read stdout and stderr."""
        if process.stdout:
            async for line_bytes in process.stdout:
                line = line_bytes.decode('utf-8', errors='replace').strip()
                if line:
                    logger.debug(f"[CodeFormer stdout] {line}")

        if process.stderr:
            async for line_bytes in process.stderr:
                line = line_bytes.decode('utf-8', errors='replace').strip()
                if line:
                    logger.warning(f"[CodeFormer stderr] {line}")
                    stderr_lines.append(line)

    async def check_cancellation():
        """Monitor cancel event and terminate process if triggered."""
        if cancel_event is None:
            return

        try:
            await cancel_event.wait()
            logger.info("Cancellation requested, terminating CodeFormer process")
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("CodeFormer process didn't terminate, killing it")
                process.kill()
        except asyncio.CancelledError:
            pass

    # Run monitoring tasks
    try:
        await asyncio.gather(
            read_output(),
            check_cancellation(),
            return_exceptions=True
        )

        # Wait for process completion
        returncode = await process.wait()

        # Check if cancelled
        if cancel_event and cancel_event.is_set():
            raise FaceRestoreCancelled("CodeFormer processing cancelled by user")

        # Check return code
        if returncode != 0:
            error_msg = "\n".join(stderr_lines[-10:]) if stderr_lines else "Unknown error"
            raise FaceRestoreError(
                f"CodeFormer failed with exit code {returncode}. "
                f"Last stderr lines:\n{error_msg}"
            )

        logger.info("CodeFormer processing complete")

    except FaceRestoreCancelled:
        raise
    except FaceRestoreError:
        raise
    except Exception as e:
        raise FaceRestoreError(f"Unexpected error during CodeFormer processing: {e}") from e
    finally:
        # Ensure process is cleaned up
        if process.returncode is None:
            process.kill()
            await process.wait()


async def _reassemble_video(
    frames_dir: Path,
    output_path: Path,
    fps: float
) -> None:
    """
    Reassemble video from processed frames.

    Args:
        frames_dir: Directory containing processed frames
        output_path: Path to save output video
        fps: Frames per second for output video

    Raises:
        FaceRestoreError: If video reassembly fails
    """
    loop = asyncio.get_event_loop()

    def _reassemble() -> None:
        try:
            # Get sorted list of frame files
            frame_files = sorted(frames_dir.glob("*.png"))
            if not frame_files:
                raise FaceRestoreError(f"No frames found in {frames_dir}")

            logger.info(f"Reassembling {len(frame_files)} frames at {fps} FPS")

            # Read all frames
            frames = []
            for frame_path in frame_files:
                frame = iio.imread(frame_path)
                frames.append(frame)

            # Write video
            iio.imwrite(
                str(output_path),
                np.array(frames),
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                output_params=["-preset", "medium", "-crf", "23"]
            )

            logger.info(f"Video reassembled: {output_path}")

        except Exception as e:
            raise FaceRestoreError(f"Failed to reassemble video: {e}") from e

    await loop.run_in_executor(None, _reassemble)


async def has_faces(image_path: Path) -> bool:
    """
    Quick check if image contains detectable faces.

    Uses OpenCV Haar Cascade for fast face detection.

    Args:
        image_path: Path to image file

    Returns:
        True if faces detected, False otherwise
    """
    loop = asyncio.get_event_loop()

    def _detect() -> bool:
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Could not load image for face detection: {image_path}")
                return False

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Load Haar Cascade classifier
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            return len(faces) > 0

        except Exception as e:
            logger.warning(f"Face detection failed for {image_path}: {e}")
            return False

    return await loop.run_in_executor(None, _detect)
