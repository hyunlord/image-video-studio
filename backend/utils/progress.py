from __future__ import annotations

import time
from typing import Callable, Awaitable, Optional

from backend.models import JobProgress, JobStatus


class ProgressTracker:
    """Track multi-stage pipeline progress and broadcast via callback."""

    STAGE_WEIGHTS = {
        "preprocessing": 0.05,
        "generating": 0.70,
        "face_restore": 0.08,
        "upscaling": 0.07,
        "interpolation": 0.05,
        "concatenation": 0.05,
    }

    STAGE_LABELS = {
        "preprocessing": "이미지 전처리 중...",
        "generating": "영상 생성 중...",
        "face_restore": "얼굴 복원 중...",
        "upscaling": "해상도 향상 중...",
        "interpolation": "프레임 보간 중...",
        "concatenation": "영상 결합 중...",
    }

    def __init__(
        self,
        job_id: str,
        total_pairs: int,
        callback: Callable[[JobProgress], Awaitable[None]],
        active_stages: Optional[list[str]] = None,
    ):
        self.job_id = job_id
        self.total_pairs = total_pairs
        self.callback = callback
        self.current_pair = 0

        # Recompute weights for only active stages
        if active_stages:
            active_w = {s: self.STAGE_WEIGHTS[s] for s in active_stages if s in self.STAGE_WEIGHTS}
        else:
            active_w = dict(self.STAGE_WEIGHTS)
        total_w = sum(active_w.values()) or 1.0
        self.weights = {s: w / total_w for s, w in active_w.items()}

        self._stage = ""
        self._stage_progress = 0.0  # 0-1 within current stage
        self._stage_start = 0.0
        self._completed_stages: set[str] = set()

    def set_pair(self, pair_index: int):
        self.current_pair = pair_index
        self._completed_stages.clear()

    async def start_stage(self, stage: str):
        self._stage = stage
        self._stage_progress = 0.0
        self._stage_start = time.monotonic()
        await self._emit()

    async def update(self, fraction: float):
        """Update current stage progress (0.0 to 1.0)."""
        self._stage_progress = min(max(fraction, 0.0), 1.0)
        await self._emit()

    async def complete_stage(self):
        self._stage_progress = 1.0
        self._completed_stages.add(self._stage)
        await self._emit()

    def _overall_progress(self) -> float:
        done = sum(self.weights.get(s, 0) for s in self._completed_stages)
        current = self.weights.get(self._stage, 0) * self._stage_progress
        pair_progress = done + current  # 0-1 for current pair

        if self.total_pairs <= 1:
            return pair_progress * 100

        completed_pairs = self.current_pair
        return ((completed_pairs + pair_progress) / self.total_pairs) * 100

    def _estimate_eta(self) -> Optional[float]:
        if self._stage_progress <= 0.01:
            return None
        elapsed = time.monotonic() - self._stage_start
        remaining_fraction = 1.0 - self._stage_progress
        return (elapsed / self._stage_progress) * remaining_fraction

    async def _emit(self):
        progress = JobProgress(
            job_id=self.job_id,
            status=self._map_status(),
            stage=self.STAGE_LABELS.get(self._stage, self._stage),
            progress=round(self._overall_progress(), 1),
            current_pair=self.current_pair + 1,
            total_pairs=self.total_pairs,
            eta_seconds=self._estimate_eta(),
            message=f"페어 {self.current_pair + 1}/{self.total_pairs}" if self.total_pairs > 1 else "",
        )
        await self.callback(progress)

    def _map_status(self) -> JobStatus:
        if self._stage == "preprocessing":
            return JobStatus.PREPROCESSING
        elif self._stage == "generating":
            return JobStatus.GENERATING
        else:
            return JobStatus.POSTPROCESSING
