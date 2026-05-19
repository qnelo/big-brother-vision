"""Map MediaPipe blendshapes and landmarks to HUD emotion metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

Species = Literal["human", "cat"]

# MediaPipe Face Mesh eye indices (478-landmark model, stable subset)
_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)

METRIC_KEYS = ("JOY", "HAPPINESS", "FEAR", "FOCUS", "DROWSY")


@dataclass
class EmotionMetrics:
    """Smoothed emotion values 0-100 for HUD display."""

    joy: float = 0.0
    happiness: float = 0.0
    fear: float = 0.0
    focus: float = 0.0
    drowsy: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "JOY": self.joy,
            "HAPPINESS": self.happiness,
            "FEAR": self.fear,
            "FOCUS": self.focus,
            "DROWSY": self.drowsy,
        }

    def update_ema(self, raw: dict[str, float], alpha: float = 0.2) -> None:
        """Exponential moving average toward raw values."""
        self.joy = self._ema(self.joy, raw["JOY"], alpha)
        self.happiness = self._ema(self.happiness, raw["HAPPINESS"], alpha)
        self.fear = self._ema(self.fear, raw["FEAR"], alpha)
        self.focus = self._ema(self.focus, raw["FOCUS"], alpha)
        self.drowsy = self._ema(self.drowsy, raw["DROWSY"], alpha)

    @staticmethod
    def _ema(prev: float, new: float, alpha: float) -> float:
        return prev * (1.0 - alpha) + new * alpha


@dataclass
class EmotionMapper:
    """Convert blendshape scores and landmarks to 0-100 metrics."""

    smoothing_alpha: float = 0.22

    def blendshapes_to_dict(self, classifications) -> dict[str, float]:
        """Build name -> score from MediaPipe Classifications."""
        result: dict[str, float] = {}
        if classifications is None:
            return result
        for cat in classifications:
            result[cat.category_name] = cat.score
        return result

    def compute_ear(
        self, landmarks, indices: tuple[int, ...], frame_w: int, frame_h: int
    ) -> float:
        """Eye aspect ratio from normalized landmarks."""
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            pts.append((lm.x * frame_w, lm.y * frame_h))
        if len(pts) < 6:
            return 0.3
        # Vertical distances / horizontal distance
        v1 = math.dist(pts[1], pts[5])
        v2 = math.dist(pts[2], pts[4])
        h = math.dist(pts[0], pts[3])
        if h < 1e-6:
            return 0.3
        return (v1 + v2) / (2.0 * h)

    def raw_metrics(
        self,
        blendshapes: dict[str, float],
        landmarks,
        frame_w: int,
        frame_h: int,
    ) -> dict[str, float]:
        """Compute instantaneous metrics before smoothing."""

        def bs(name: str) -> float:
            return blendshapes.get(name, 0.0)

        smile = (bs("mouthSmileLeft") + bs("mouthSmileRight")) / 2.0
        cheek = (bs("cheekSquintLeft") + bs("cheekSquintRight")) / 2.0
        brow_down = (bs("browDownLeft") + bs("browDownRight")) / 2.0
        brow_inner = bs("browInnerUp")
        eye_wide = (bs("eyeWideLeft") + bs("eyeWideRight")) / 2.0
        blink = (bs("eyeBlinkLeft") + bs("eyeBlinkRight")) / 2.0
        look_away = (
            abs(bs("eyeLookInLeft") - bs("eyeLookOutLeft"))
            + abs(bs("eyeLookInRight") - bs("eyeLookOutRight"))
        ) / 2.0
        jaw_open = bs("jawOpen")
        mouth_frown = (bs("mouthFrownLeft") + bs("mouthFrownRight")) / 2.0

        joy = _clamp100((smile * 0.55 + cheek * 0.45) * 115.0)

        happiness = _clamp100(
            (smile * 0.4 + cheek * 0.35 - brow_down * 0.2 - mouth_frown * 0.25)
            * 120.0
            + 15.0
        )

        fear = _clamp100(
            (brow_inner * 0.45 + eye_wide * 0.35 + jaw_open * 0.2) * 110.0
        )

        focus = _clamp100(
            (brow_down * 0.35 + (1.0 - look_away) * 0.35 - jaw_open * 0.15)
            * 95.0
            + 20.0
        )

        ear = 0.3
        if landmarks is not None and len(landmarks) > max(_RIGHT_EYE):
            ear_l = self.compute_ear(landmarks, _LEFT_EYE, frame_w, frame_h)
            ear_r = self.compute_ear(landmarks, _RIGHT_EYE, frame_w, frame_h)
            ear = (ear_l + ear_r) / 2.0

        # Low EAR and high blink => drowsy
        ear_factor = _clamp01((0.28 - ear) / 0.12)
        drowsy = _clamp100((blink * 0.55 + ear_factor * 0.45) * 105.0)

        return {
            "JOY": joy,
            "HAPPINESS": happiness,
            "FEAR": fear,
            "FOCUS": focus,
            "DROWSY": drowsy,
        }


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _clamp100(v: float) -> float:
    return max(0.0, min(100.0, v))


@dataclass
class FaceDetection:
    """Single face detection output for one frame."""

    bbox: tuple[int, int, int, int]  # x, y, w, h full-res
    center: tuple[int, int]
    species: Species = "human"
    landmarks: list | None = None
    blendshapes: dict[str, float] = field(default_factory=dict)
    metrics: EmotionMetrics = field(default_factory=EmotionMetrics)
