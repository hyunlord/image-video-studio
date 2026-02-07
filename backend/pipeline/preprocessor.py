"""
Image preprocessing module for Wan 2.1 FLF2V pipeline.

Provides letterbox resizing with aspect ratio preservation and black padding
to prepare image pairs for video generation.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Tuple

from PIL import Image

from backend.config import RESOLUTION_MAP
from backend.models import TechnicalParams

logger = logging.getLogger(__name__)


class PreprocessError(Exception):
    """Raised when image preprocessing fails."""
    pass


def _letterbox_resize(
    img: Image.Image,
    target_width: int,
    target_height: int
) -> Image.Image:
    """
    Resize image maintaining aspect ratio with black letterbox padding.

    Args:
        img: Input PIL Image in RGB mode
        target_width: Target canvas width in pixels
        target_height: Target canvas height in pixels

    Returns:
        Resized image on black canvas at target dimensions
    """
    original_width, original_height = img.size

    # Calculate scale to fit within target dimensions
    scale = min(target_width / original_width, target_height / original_height)

    # Calculate new dimensions maintaining aspect ratio
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize with high-quality Lanczos filter
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create black canvas at target dimensions
    canvas = Image.new('RGB', (target_width, target_height), (0, 0, 0))

    # Calculate paste position to center the image
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # Paste resized image onto canvas
    canvas.paste(resized_img, (paste_x, paste_y))

    return canvas


async def preprocess_image(
    image_path: Path,
    output_path: Path,
    target_width: int,
    target_height: int
) -> None:
    """
    Preprocess a single image asynchronously.

    Args:
        image_path: Path to input image file
        output_path: Path to save processed image
        target_width: Target width in pixels
        target_height: Target height in pixels

    Raises:
        PreprocessError: If image loading or processing fails
    """
    try:
        # Run CPU-intensive PIL operations in thread pool
        loop = asyncio.get_event_loop()

        def _process() -> None:
            img = Image.open(image_path).convert('RGB')
            processed_img = _letterbox_resize(img, target_width, target_height)
            processed_img.save(output_path, format='PNG', optimize=True)

        await loop.run_in_executor(None, _process)
        logger.info(f"Preprocessed image: {image_path} -> {output_path}")

    except FileNotFoundError as e:
        raise PreprocessError(f"Image not found: {image_path}") from e
    except OSError as e:
        raise PreprocessError(f"Failed to read image {image_path}: {e}") from e
    except Exception as e:
        raise PreprocessError(f"Preprocessing failed for {image_path}: {e}") from e


async def preprocess_pair(
    img1_path: Path,
    img2_path: Path,
    output_dir: Path,
    params: TechnicalParams
) -> Tuple[Path, Path]:
    """
    Preprocess an image pair for video generation.

    Both images are letterboxed to the target resolution specified in params,
    maintaining aspect ratio with black padding. Output files are saved as
    'first_frame.png' and 'last_frame.png'.

    Args:
        img1_path: Path to first image
        img2_path: Path to last image
        output_dir: Directory to save preprocessed images
        params: Technical parameters containing resolution settings

    Returns:
        Tuple of (first_frame_path, last_frame_path)

    Raises:
        PreprocessError: If preprocessing fails for either image
        ValueError: If resolution is not supported
    """
    # Validate resolution
    if params.resolution not in RESOLUTION_MAP:
        raise ValueError(
            f"Unsupported resolution: {params.resolution}. "
            f"Must be one of {list(RESOLUTION_MAP.keys())}"
        )

    # Get target dimensions
    target_width = params.width
    target_height = params.height

    logger.info(
        f"Preprocessing pair: {img1_path.name} + {img2_path.name} "
        f"-> {target_width}x{target_height}"
    )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    first_frame_path = output_dir / "first_frame.png"
    last_frame_path = output_dir / "last_frame.png"

    # Process both images in parallel
    try:
        await asyncio.gather(
            preprocess_image(img1_path, first_frame_path, target_width, target_height),
            preprocess_image(img2_path, last_frame_path, target_width, target_height)
        )
    except PreprocessError as e:
        logger.error(f"Preprocessing pair failed: {e}")
        raise

    logger.info(f"Preprocessing complete: {first_frame_path}, {last_frame_path}")
    return first_frame_path, last_frame_path


async def validate_images(img1_path: Path, img2_path: Path) -> Tuple[bool, str]:
    """
    Validate image pair before preprocessing.

    Args:
        img1_path: Path to first image
        img2_path: Path to last image

    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []

    # Check file existence
    if not img1_path.exists():
        errors.append(f"First image not found: {img1_path}")
    if not img2_path.exists():
        errors.append(f"Last image not found: {img2_path}")

    if errors:
        return False, "; ".join(errors)

    # Validate images can be opened
    loop = asyncio.get_event_loop()

    def _validate() -> Tuple[bool, str]:
        try:
            with Image.open(img1_path) as img1:
                img1.verify()
            with Image.open(img2_path) as img2:
                img2.verify()
            return True, ""
        except Exception as e:
            return False, f"Invalid image file: {e}"

    return await loop.run_in_executor(None, _validate)
