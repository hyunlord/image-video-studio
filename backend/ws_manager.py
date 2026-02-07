from __future__ import annotations

import json
import logging
from typing import Dict, Set

from fastapi import WebSocket

from backend.models import JobProgress

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage WebSocket connections per job for real-time progress."""

    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        if job_id not in self._connections:
            self._connections[job_id] = set()
        self._connections[job_id].add(websocket)
        logger.info("WS connected: job=%s (total=%d)", job_id, len(self._connections[job_id]))

    def disconnect(self, job_id: str, websocket: WebSocket):
        conns = self._connections.get(job_id)
        if conns:
            conns.discard(websocket)
            if not conns:
                del self._connections[job_id]
        logger.info("WS disconnected: job=%s", job_id)

    async def broadcast(self, job_id: str, progress: JobProgress):
        """Send progress update to all clients watching this job."""
        conns = self._connections.get(job_id)
        if not conns:
            return

        data = json.dumps(progress.model_dump(), ensure_ascii=False, default=str)
        dead: set[WebSocket] = set()

        for ws in conns:
            try:
                await ws.send_text(data)
            except Exception:
                dead.add(ws)

        for ws in dead:
            conns.discard(ws)
        if not conns:
            self._connections.pop(job_id, None)

    def has_listeners(self, job_id: str) -> bool:
        return bool(self._connections.get(job_id))


ws_manager = ConnectionManager()
