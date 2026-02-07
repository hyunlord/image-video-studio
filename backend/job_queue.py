from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from backend.analyzers.image_analyzer import analyze_image_pair
from backend.analyzers.prompt_analyzer import analyze_prompt
from backend.config import OUTPUT_DIR
from backend.models import (
    JobProgress,
    JobRequest,
    JobResponse,
    JobStatus,
    TechnicalParams,
)
from backend.smart_params import map_parameters
from backend.utils.files import create_job_dir, cleanup_job_dir, generate_id, get_upload_path
from backend.utils.progress import ProgressTracker
from backend.ws_manager import ws_manager

logger = logging.getLogger(__name__)


class _JobInfo:
    def __init__(self, job_id: str, request: JobRequest):
        self.job_id = job_id
        self.request = request
        self.status = JobStatus.QUEUED
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.progress = 0.0
        self.video_path: Optional[Path] = None
        self.error: Optional[str] = None
        self.cancel_event = asyncio.Event()

    def to_response(self) -> JobResponse:
        video_url = None
        if self.video_path and self.video_path.exists():
            video_url = f"/static/outputs/{self.job_id}/{self.video_path.name}"
        return JobResponse(
            job_id=self.job_id,
            status=self.status,
            created_at=self.created_at,
            progress=self.progress,
            video_url=video_url,
            error=self.error,
        )


class JobQueue:
    """Single-GPU sequential job queue with async processing."""

    def __init__(self):
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: Dict[str, _JobInfo] = {}
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("Job queue worker started")

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            self._worker_task = None

    async def submit(self, request: JobRequest) -> str:
        job_id = generate_id()
        info = _JobInfo(job_id, request)
        self._jobs[job_id] = info
        await self._queue.put(job_id)
        logger.info("Job %s queued (queue size=%d)", job_id, self._queue.qsize())
        return job_id

    def get_job(self, job_id: str) -> Optional[JobResponse]:
        info = self._jobs.get(job_id)
        return info.to_response() if info else None

    def list_jobs(self) -> list[JobResponse]:
        return [info.to_response() for info in self._jobs.values()]

    def get_active_job_info(self) -> dict | None:
        """Return info about the currently running job, if any."""
        for info in self._jobs.values():
            if info.status in (JobStatus.PREPROCESSING, JobStatus.GENERATING, JobStatus.POSTPROCESSING):
                return {
                    "job_id": info.job_id,
                    "status": info.status.value,
                    "progress": info.progress,
                }
        return None

    def cancel_job(self, job_id: str) -> bool:
        info = self._jobs.get(job_id)
        if not info:
            return False
        if info.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        info.cancel_event.set()
        info.status = JobStatus.CANCELLED
        return True

    # ── Worker loop ──────────────────────────────────────────────────────

    async def _worker(self):
        while True:
            job_id = await self._queue.get()
            info = self._jobs.get(job_id)
            if not info or info.cancel_event.is_set():
                self._queue.task_done()
                continue

            try:
                await self._execute(info)
            except asyncio.CancelledError:
                info.status = JobStatus.CANCELLED
                logger.info("Job %s cancelled", job_id)
            except Exception as e:
                info.status = JobStatus.FAILED
                info.error = str(e)
                logger.exception("Job %s failed: %s", job_id, e)
                await self._broadcast_status(info)
            finally:
                self._queue.task_done()

    async def _execute(self, info: _JobInfo):
        from backend.pipeline.preprocessor import preprocess_pair
        from backend.pipeline.generator import generate_video
        from backend.pipeline.face_restore import restore_faces
        from backend.pipeline.upscaler import upscale_video
        from backend.pipeline.interpolator import interpolate_video
        from backend.pipeline.concatenator import concatenate_videos

        job_id = info.job_id
        request = info.request
        job_dir = create_job_dir(job_id)

        # Build consecutive image pairs from ordered image_ids
        image_ids = request.image_ids
        pairs = [(image_ids[i], image_ids[i + 1]) for i in range(len(image_ids) - 1)]
        total_pairs = len(pairs)

        # Analyze first pair for smart params (representative)
        first_img = get_upload_path(pairs[0][0])
        last_img = get_upload_path(pairs[0][1])
        if not first_img or not last_img:
            raise FileNotFoundError("Upload images not found")

        image_analysis = analyze_image_pair(first_img, last_img)
        prompt_analysis = analyze_prompt(request.prompt)
        tech_params = map_parameters(request, image_analysis, prompt_analysis)
        prompt_text = prompt_analysis.translated_prompt or request.prompt

        # Determine active post-processing stages
        active_stages = ["preprocessing", "generating"]
        if tech_params.apply_codeformer:
            active_stages.append("face_restore")
        if tech_params.apply_upscale:
            active_stages.append("upscaling")
        if tech_params.apply_interpolation:
            active_stages.append("interpolation")
        if total_pairs > 1:
            active_stages.append("concatenation")

        tracker = ProgressTracker(
            job_id=job_id,
            total_pairs=total_pairs,
            callback=lambda p: self._on_progress(info, p),
            active_stages=active_stages,
        )

        segment_videos: list[Path] = []

        for pair_idx, (first_id, last_id) in enumerate(pairs):
            if info.cancel_event.is_set():
                raise asyncio.CancelledError()

            tracker.set_pair(pair_idx)
            pair_dir = job_dir / f"pair_{pair_idx}"
            pair_dir.mkdir(exist_ok=True)

            # Get image paths
            img1_path = get_upload_path(first_id)
            img2_path = get_upload_path(last_id)
            if not img1_path or not img2_path:
                raise FileNotFoundError(f"Images not found: {first_id}, {last_id}")

            # 1. Preprocess
            await tracker.start_stage("preprocessing")
            first_frame, last_frame = await preprocess_pair(
                img1_path, img2_path, pair_dir, tech_params
            )
            await tracker.complete_stage()

            # 2. Generate video — free GPU/CPU memory first
            await tracker.start_stage("generating")
            from backend.analyzers.image_analyzer import unload_clip
            import gc

            unload_clip()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

            raw_video = pair_dir / "raw.mp4"
            await generate_video(
                first_frame=first_frame,
                last_frame=last_frame,
                prompt=prompt_text,
                output_path=raw_video,
                params=tech_params,
                progress_callback=tracker.update,
            )
            await tracker.complete_stage()

            current_video = raw_video

            # 3. Face restoration (conditional)
            if tech_params.apply_codeformer:
                await tracker.start_stage("face_restore")
                restored = pair_dir / "restored.mp4"
                current_video = await restore_faces(
                    video_path=current_video,
                    output_path=restored,
                    fidelity=tech_params.codeformer_fidelity,
                    progress_callback=tracker.update,
                )
                await tracker.complete_stage()

            # 4. Upscale (conditional)
            if tech_params.apply_upscale:
                await tracker.start_stage("upscaling")
                upscaled = pair_dir / "upscaled.mp4"
                current_video = await upscale_video(
                    video_path=current_video,
                    output_path=upscaled,
                    scale=tech_params.upscale_factor,
                    progress_callback=tracker.update,
                )
                await tracker.complete_stage()

            # 5. Frame interpolation (conditional)
            if tech_params.apply_interpolation:
                await tracker.start_stage("interpolation")
                interpolated = pair_dir / "interpolated.mp4"
                current_video = await interpolate_video(
                    video_path=current_video,
                    output_path=interpolated,
                    progress_callback=tracker.update,
                )
                await tracker.complete_stage()

            segment_videos.append(current_video)

        # 6. Concatenate if multiple pairs
        if total_pairs > 1:
            await tracker.start_stage("concatenation")
            final_path = job_dir / "final.mp4"
            await concatenate_videos(segment_videos, final_path)
            await tracker.complete_stage()
        else:
            final_path = job_dir / "final.mp4"
            import shutil
            shutil.copy2(str(segment_videos[0]), str(final_path))

        # Done
        info.status = JobStatus.COMPLETED
        info.video_path = final_path
        info.progress = 100.0
        await self._broadcast_status(info)

        # Cleanup intermediate files
        cleanup_job_dir(job_id, keep_final=True)
        logger.info("Job %s completed: %s", job_id, final_path)

    async def _on_progress(self, info: _JobInfo, progress: JobProgress):
        info.progress = progress.progress
        info.status = progress.status
        await ws_manager.broadcast(info.job_id, progress)

    async def _broadcast_status(self, info: _JobInfo):
        progress = JobProgress(
            job_id=info.job_id,
            status=info.status,
            progress=info.progress,
            message=info.error or "",
        )
        await ws_manager.broadcast(info.job_id, progress)


job_queue = JobQueue()
