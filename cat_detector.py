"""Cat face detection via OpenCV Haar cascade."""

from __future__ import annotations

import cv2
import numpy as np

from emotion_mapper import EmotionMetrics, FaceDetection

_CASCADE_NAME = "haarcascade_frontalcatface_extended.xml"


class CatDetector:
    """Detect cat faces using OpenCV's built-in Haar cascade."""

    def __init__(self, detect_width: int = 320):
        cascade_path = cv2.data.haarcascades + _CASCADE_NAME
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load cat cascade: {cascade_path}")
        self._detect_width = detect_width
        self._gray_buffer: np.ndarray | None = None

    def detect(
        self,
        frame: np.ndarray,
        max_cats: int = 2,
    ) -> list[FaceDetection]:
        """Return cat face detections scaled to full frame resolution."""
        full_h, full_w = frame.shape[:2]
        detect_w = self._detect_width
        detect_h = max(1, int(full_h * detect_w / full_w))
        scale_x = full_w / detect_w
        scale_y = full_h / detect_h

        small = cv2.resize(
            frame, (detect_w, detect_h), interpolation=cv2.INTER_LINEAR
        )
        if (
            self._gray_buffer is None
            or self._gray_buffer.shape[0] != detect_h
            or self._gray_buffer.shape[1] != detect_w
        ):
            self._gray_buffer = np.empty((detect_h, detect_w), dtype=np.uint8)
        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY, dst=self._gray_buffer)

        faces = self.cascade.detectMultiScale(
            self._gray_buffer,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        detections: list[FaceDetection] = []
        for x, y, w, h in faces[:max_cats]:
            fx = int(x * scale_x)
            fy = int(y * scale_y)
            fw = int(w * scale_x)
            fh = int(h * scale_y)
            detections.append(
                FaceDetection(
                    bbox=(fx, fy, fw, fh),
                    center=(fx + fw // 2, fy + fh // 2),
                    species="cat",
                )
            )

        return detections
