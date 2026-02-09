from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.config import (
    FRONTEND_DIR,
    MAX_UPLOAD_SIZE_MB,
    OUTPUT_DIR,
    STATIC_DIR,
    UPLOAD_DIR,
)
from backend.job_queue import job_queue
from backend.models import (
    JobRequest,
    JobResponse,
    SystemConfig,
    SystemMonitor,
    VideoLength,
    VideoQuality,
)
from backend.utils.files import (
    delete_upload,
    get_image_info,
    get_video_path,
    list_uploads,
    save_upload,
)
from backend.utils.gpu import detect_gpu_tier, get_gpu_monitor_data, get_system_ram
from backend.ws_manager import ws_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class _QuietMonitorFilter(logging.Filter):
    """Suppress access log noise from /api/system/monitor polling."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "/api/system/monitor" in msg or "/api/system/health" in msg:
            return False
        return True


logging.getLogger("uvicorn.access").addFilter(_QuietMonitorFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    await job_queue.start()
    logger.info("Application started")
    yield
    await job_queue.stop()
    logger.info("Application stopped")


app = FastAPI(
    title="Wan 2.1 Video Studio",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Frontend ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(404, "Frontend not found")
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/monitor", response_class=HTMLResponse)
async def serve_monitor():
    page = FRONTEND_DIR / "monitor.html"
    if not page.exists():
        raise HTTPException(404, "Monitor page not found")
    return HTMLResponse(page.read_text(encoding="utf-8"))


# ── Image upload endpoints ───────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(413, f"File too large: {f.filename}")
        image_id, path = save_upload(data, f.filename or "image.jpg")
        info = get_image_info(image_id)
        results.append(info)
    return {"images": results}


@app.get("/api/images")
async def list_images():
    return {"images": list_uploads()}


@app.delete("/api/images/{image_id}")
async def delete_image(image_id: str):
    if not delete_upload(image_id):
        raise HTTPException(404, "Image not found")
    return {"ok": True}


# ── Job endpoints ────────────────────────────────────────────────────────────

@app.post("/api/jobs", response_model=JobResponse)
async def create_job(request: JobRequest):
    job_id = await job_queue.submit(request)
    resp = job_queue.get_job(job_id)
    if not resp:
        raise HTTPException(500, "Failed to create job")
    return resp


@app.get("/api/jobs")
async def list_jobs():
    return {"jobs": job_queue.list_jobs()}


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    resp = job_queue.get_job(job_id)
    if not resp:
        raise HTTPException(404, "Job not found")
    return resp


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    if not job_queue.cancel_job(job_id):
        raise HTTPException(404, "Job not found or already finished")
    return {"ok": True}


# ── Video endpoints ──────────────────────────────────────────────────────────

@app.get("/api/videos/{job_id}")
async def stream_video(job_id: str):
    path = get_video_path(job_id)
    if not path:
        raise HTTPException(404, "Video not found")
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@app.get("/api/videos/{job_id}/download")
async def download_video(job_id: str):
    path = get_video_path(job_id)
    if not path:
        raise HTTPException(404, "Video not found")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"video_{job_id}.mp4",
    )


# ── System endpoints ─────────────────────────────────────────────────────────

@app.get("/api/system/config", response_model=SystemConfig)
async def get_system_config():
    tier_name, profile = detect_gpu_tier()
    max_frames = profile["max_frames"]

    lengths = [VideoLength.SHORT.value]
    if max_frames >= 49:
        lengths.append(VideoLength.MEDIUM.value)
    if max_frames >= 81:
        lengths.append(VideoLength.LONG.value)

    qualities = [VideoQuality.STANDARD.value]
    if profile.get("vram_gb", 0) >= 14:
        qualities.append(VideoQuality.HD.value)

    return SystemConfig(
        gpu_name=tier_name,
        gpu_tier=tier_name,
        vram_gb=profile.get("vram_gb", 0),
        max_frames=max_frames,
        safe_resolution=profile["safe_res"],
        available_lengths=lengths,
        available_qualities=qualities,
    )


@app.get("/api/system/monitor", response_model=SystemMonitor)
async def get_system_monitor():
    gpu_data = get_gpu_monitor_data()
    ram_data = get_system_ram()
    active = job_queue.get_active_job_info()

    stage_labels = {
        "preprocessing": "전처리 중",
        "generating": "영상 생성 중",
        "postprocessing": "후처리 중",
    }

    return SystemMonitor(
        **gpu_data,
        **ram_data,
        active_job_id=active["job_id"] if active else None,
        active_job_stage=stage_labels.get(active["status"], active["status"]) if active else "",
        active_job_progress=active["progress"] if active else 0.0,
    )


@app.get("/api/system/health")
async def health_check():
    return {"status": "ok"}


# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await ws_manager.connect(job_id, websocket)
    try:
        while True:
            # Keep connection alive; client can send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(job_id, websocket)
    except Exception:
        ws_manager.disconnect(job_id, websocket)
