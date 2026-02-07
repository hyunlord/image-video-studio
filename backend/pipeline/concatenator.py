"""Video concatenation module.

This module provides async video concatenation with optional crossfade transitions
using ffmpeg concat demuxer and xfade filter.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ConcatenatorError(RuntimeError):
    """Raised when concatenation operations fail."""

    pass


async def _get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds

    Raises:
        ConcatenatorError: If duration cannot be determined
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise ConcatenatorError(
                f"ffprobe failed: {stderr.decode() if stderr else 'unknown error'}"
            )

        duration_str = stdout.decode().strip()
        return float(duration_str)

    except ValueError as e:
        raise ConcatenatorError(f"Invalid duration format: {duration_str}") from e
    except Exception as e:
        raise ConcatenatorError(f"Failed to get video duration: {e}") from e


async def _concat_simple(video_paths: list[Path], output_path: Path) -> None:
    """Concatenate videos without crossfade using concat demuxer.

    Args:
        video_paths: List of video paths to concatenate
        output_path: Path for output video file

    Raises:
        ConcatenatorError: If concatenation fails
    """
    # Create temporary file list for concat demuxer
    temp_dir = Path(tempfile.mkdtemp(prefix="concat_"))
    file_list_path = temp_dir / "file_list.txt"

    try:
        # Write file list
        with open(file_list_path, "w") as f:
            for video_path in video_paths:
                # Use absolute paths and escape single quotes
                escaped_path = str(video_path.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        logger.info(f"Created file list: {file_list_path}")

        # Run ffmpeg concat
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(file_list_path),
            "-c",
            "copy",  # stream copy for speed
            str(output_path),
        ]

        logger.info(f"Running concat: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise ConcatenatorError(f"ffmpeg concat failed: {error_msg}")

        logger.info(f"Simple concatenation completed: {output_path}")

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


async def _concat_with_crossfade(
    video_paths: list[Path], output_path: Path, crossfade_duration: float
) -> None:
    """Concatenate videos with crossfade transitions using xfade filter.

    Args:
        video_paths: List of video paths to concatenate
        output_path: Path for output video file
        crossfade_duration: Duration of crossfade in seconds

    Raises:
        ConcatenatorError: If concatenation fails
    """
    if len(video_paths) < 2:
        raise ConcatenatorError("Need at least 2 videos for crossfade")

    logger.info(
        f"Concatenating {len(video_paths)} videos with {crossfade_duration}s crossfade"
    )

    # Get durations of all videos
    durations = []
    for video_path in video_paths:
        duration = await _get_video_duration(video_path)
        durations.append(duration)
        logger.debug(f"Video {video_path.name}: {duration:.2f}s")

    # Build complex filter for crossfade
    filter_parts = []
    input_args = []

    # Add all input files
    for i, video_path in enumerate(video_paths):
        input_args.extend(["-i", str(video_path)])

    # Build xfade filter chain
    # For n videos, we need n-1 crossfades
    current_label = "[0:v]"
    offset = 0.0

    for i in range(len(video_paths) - 1):
        next_input = f"[{i+1}:v]"
        output_label = f"[v{i}]" if i < len(video_paths) - 2 else "[vout]"

        # Calculate offset (cumulative duration minus crossfade overlap)
        if i > 0:
            offset += durations[i] - crossfade_duration

        xfade_filter = (
            f"{current_label}{next_input}xfade="
            f"transition=fade:"
            f"duration={crossfade_duration}:"
            f"offset={offset}"
            f"{output_label}"
        )
        filter_parts.append(xfade_filter)
        current_label = output_label

    # Build audio mixing
    # Concatenate audio with acrossfade
    audio_filter_parts = []
    current_audio = "[0:a]"

    for i in range(len(video_paths) - 1):
        next_audio = f"[{i+1}:a]"
        output_audio = f"[a{i}]" if i < len(video_paths) - 2 else "[aout]"

        acrossfade_filter = (
            f"{current_audio}{next_audio}acrossfade="
            f"d={crossfade_duration}"
            f"{output_audio}"
        )
        audio_filter_parts.append(acrossfade_filter)
        current_audio = output_audio

    # Combine filters
    filter_complex = ";".join(filter_parts + audio_filter_parts)

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        *input_args,
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]

    logger.info(f"Running crossfade concat: {' '.join(cmd[:10])}...")
    logger.debug(f"Filter complex: {filter_complex}")

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Monitor stderr for progress
        async def monitor_stderr():
            stderr_lines = []
            if process.stderr:
                async for line in process.stderr:
                    line_str = line.decode().strip()
                    stderr_lines.append(line_str)
                    if "time=" in line_str:
                        logger.debug(f"Crossfade progress: {line_str}")
            return stderr_lines

        stderr_task = asyncio.create_task(monitor_stderr())
        await process.wait()
        stderr_lines = await stderr_task

        if process.returncode != 0:
            error_msg = "\n".join(stderr_lines) if stderr_lines else "Unknown error"
            raise ConcatenatorError(f"ffmpeg crossfade failed: {error_msg}")

        logger.info(f"Crossfade concatenation completed: {output_path}")

    except asyncio.CancelledError:
        if process:
            process.kill()
            await process.wait()
        raise
    except Exception as e:
        raise ConcatenatorError(f"Crossfade concatenation failed: {e}") from e


async def concatenate_videos(
    video_paths: list[Path],
    output_path: Path,
    crossfade_duration: float = 0.0,
) -> Path:
    """Concatenate multiple video segments into one final video.

    Args:
        video_paths: List of video file paths to concatenate in order
        output_path: Path for output concatenated video
        crossfade_duration: Duration of crossfade transition in seconds (default 0.0)
                           If 0, uses simple concat without re-encoding

    Returns:
        Path to concatenated video file

    Raises:
        ConcatenatorError: If concatenation fails or invalid inputs
    """
    if not video_paths:
        raise ConcatenatorError("No video paths provided for concatenation")

    # Validate all input videos exist
    for video_path in video_paths:
        if not video_path.exists():
            raise ConcatenatorError(f"Video file not found: {video_path}")

    # Single video case - just copy
    if len(video_paths) == 1:
        logger.info(f"Single video, copying to output: {output_path}")
        shutil.copy(video_paths[0], output_path)
        return output_path

    logger.info(
        f"Concatenating {len(video_paths)} videos "
        f"(crossfade={crossfade_duration}s) -> {output_path}"
    )

    try:
        if crossfade_duration > 0:
            await _concat_with_crossfade(video_paths, output_path, crossfade_duration)
        else:
            await _concat_simple(video_paths, output_path)

        if not output_path.exists():
            raise ConcatenatorError("Output video was not created")

        logger.info(f"Video concatenation completed: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Concatenation failed: {e}")
        raise ConcatenatorError(f"Video concatenation failed: {e}") from e
