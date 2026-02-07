from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from PIL import Image

from backend.config import ALLOWED_IMAGE_EXTENSIONS, UPLOAD_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


def generate_id() -> str:
    return uuid.uuid4().hex[:12]


def save_upload(data: bytes, original_filename: str) -> tuple[str, Path]:
    """Save uploaded file, return (image_id, path)."""
    ext = Path(original_filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported format: {ext}")

    image_id = generate_id()
    dest = UPLOAD_DIR / f"{image_id}{ext}"
    dest.write_bytes(data)
    logger.info("Saved upload %s → %s", original_filename, dest)
    return image_id, dest


def get_upload_path(image_id: str) -> Path | None:
    """Find uploaded image by ID (any extension)."""
    for ext in ALLOWED_IMAGE_EXTENSIONS:
        path = UPLOAD_DIR / f"{image_id}{ext}"
        if path.exists():
            return path
    return None


def get_image_info(image_id: str) -> dict | None:
    """Return basic info about an uploaded image."""
    path = get_upload_path(image_id)
    if path is None:
        return None
    img = Image.open(path)
    return {
        "id": image_id,
        "filename": path.name,
        "url": f"/static/uploads/{path.name}",
        "width": img.width,
        "height": img.height,
    }


def delete_upload(image_id: str) -> bool:
    path = get_upload_path(image_id)
    if path and path.exists():
        path.unlink()
        return True
    return False


def list_uploads() -> list[dict]:
    """List all uploaded images with info."""
    results = []
    for path in sorted(UPLOAD_DIR.iterdir()):
        if path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
            image_id = path.stem
            img = Image.open(path)
            results.append({
                "id": image_id,
                "filename": path.name,
                "url": f"/static/uploads/{path.name}",
                "width": img.width,
                "height": img.height,
            })
    return results


def create_job_dir(job_id: str) -> Path:
    """Create and return a working directory for a job."""
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "frames").mkdir(exist_ok=True)
    (job_dir / "restored").mkdir(exist_ok=True)
    (job_dir / "upscaled").mkdir(exist_ok=True)
    (job_dir / "interpolated").mkdir(exist_ok=True)
    return job_dir


def cleanup_job_dir(job_id: str, keep_final: bool = True):
    """Remove intermediate files, optionally keep final video."""
    job_dir = OUTPUT_DIR / job_id
    if not job_dir.exists():
        return

    for subdir in ["frames", "restored", "upscaled", "interpolated"]:
        d = job_dir / subdir
        if d.exists():
            shutil.rmtree(d)

    if not keep_final:
        shutil.rmtree(job_dir)


def get_video_path(job_id: str) -> Path | None:
    """Find the final video for a job."""
    job_dir = OUTPUT_DIR / job_id
    for name in ["final.mp4", "output.mp4"]:
        path = job_dir / name
        if path.exists():
            return path
    return None
