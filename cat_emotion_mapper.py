"""Approximate HUD emotion metrics for cat faces from bbox geometry."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from emotion_mapper import _clamp01, _clamp100


@dataclass
class CatFrameHistory:
    """Short temporal buffer for one cat track."""

    max_len: int = 5
    bboxes: deque = field(default_factory=lambda: deque(maxlen=5))
    centers: deque = field(default_factory=lambda: deque(maxlen=5))
    eye_openness: deque = field(default_factory=lambda: deque(maxlen=5))
    areas: deque = field(default_factory=lambda: deque(maxlen=5))

    def push(
        self,
        bbox: tuple[int, int, int, int],
        center: tuple[int, int],
        eye_open: float,
    ) -> None:
        x, y, w, h = bbox
        self.bboxes.append(bbox)
        self.centers.append(center)
        self.eye_openness.append(eye_open)
        self.areas.append(w * h)


@dataclass
class CatEmotionMapper:
    """Derive 0-100 metrics from bbox and estimated eye region."""

    _histories: dict[str, CatFrameHistory] = field(default_factory=dict)

    def get_history(self, track_key: str) -> CatFrameHistory:
        if track_key not in self._histories:
            self._histories[track_key] = CatFrameHistory()
        return self._histories[track_key]

    def prune_stale(self, active_keys: set[str]) -> None:
        stale = [k for k in self._histories if k not in active_keys]
        for k in stale:
            del self._histories[k]

    def compute_eye_openness(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> float:
        """Laplacian variance in the upper eye band of the face bbox."""
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        if w < 8 or h < 8:
            return 0.3

        eye_y1 = y + int(h * 0.25)
        eye_y2 = y + int(h * 0.45)
        roi = frame[
            max(0, eye_y1) : min(fh, eye_y2),
            max(0, x) : min(fw, x + w),
        ]
        if roi.size == 0:
            return 0.3

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(lap.var())
        # Normalize typical range ~0-800 into 0-1
        return _clamp01(variance / 600.0)

    def raw_metrics(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        center: tuple[int, int],
        history: CatFrameHistory | None = None,
    ) -> dict[str, float]:
        """Compute instantaneous cat metrics before EMA smoothing."""
        x, y, w, h = bbox
        eye_open = self.compute_eye_openness(frame, bbox)
        aspect = h / max(w, 1)
        area = w * h

        stability = 1.0
        jitter = 0.0
        area_delta = 0.0
        eye_drop = 0.0
        slow_blink = 0.0

        if history and len(history.centers) >= 2:
            centers = list(history.centers)
            dists = [
                math.dist(centers[i], centers[i - 1])
                for i in range(1, len(centers))
            ]
            avg_dist = sum(dists) / len(dists)
            face_diag = math.hypot(w, h)
            jitter = _clamp01(avg_dist / max(face_diag * 0.08, 1.0))
            stability = 1.0 - jitter

            if len(history.areas) >= 2:
                prev_area = history.areas[-1]
                if prev_area > 0:
                    area_delta = _clamp01((area - prev_area) / prev_area)

            if len(history.eye_openness) >= 2:
                prev_eye = history.eye_openness[-1]
                eye_drop = _clamp01(max(0.0, prev_eye - eye_open) * 3.0)
                if 0.15 < eye_open < 0.55 and prev_eye > eye_open:
                    slow_blink = _clamp01(
                        (prev_eye - eye_open) * 2.0 + (1.0 - jitter) * 0.3
                    )

        # Relaxed aspect ratio for cats ~0.85-1.15
        relaxed = 1.0 - min(abs(aspect - 1.0) / 0.35, 1.0)

        joy = _clamp100(
            (stability * 0.45 + relaxed * 0.35 + (1.0 - jitter) * 0.2) * 85.0
            + 10.0
        )

        happiness = _clamp100(
            (joy * 0.5 + slow_blink * 0.35 + relaxed * 0.15)
        )

        fear = _clamp100(
            (area_delta * 0.35 + jitter * 0.35 + eye_open * 0.3) * 95.0
        )

        focus = _clamp100(
            (stability * 0.4 + eye_open * 0.35 + (1.0 - area_delta) * 0.25)
            * 90.0
            + 15.0
        )

        drowsy = _clamp100(
            ((1.0 - eye_open) * 0.5 + eye_drop * 0.35 + slow_blink * 0.15)
            * 100.0
        )

        if history is not None:
            history.push(bbox, center, eye_open)

        return {
            "JOY": joy,
            "HAPPINESS": happiness,
            "FEAR": fear,
            "FOCUS": focus,
            "DROWSY": drowsy,
        }
