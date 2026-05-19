"""Multi-face tracking with stable SUBJ-IDs via IoU matching."""

from __future__ import annotations

from dataclasses import dataclass, field

from emotion_mapper import EmotionMetrics, FaceDetection


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackedFace:
    """A face track with persistent subject ID."""

    subject_id: str
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    metrics: EmotionMetrics
    frames_lost: int = 0
    locked: bool = True

    def label(self) -> str:
        status = "LOCK" if self.locked and self.frames_lost == 0 else "SCAN"
        return f"{self.subject_id} | {status}"


@dataclass
class FaceTracker:
    """Assign and maintain SUBJ-NNN IDs across frames."""

    max_lost_frames: int = 15
    iou_threshold: float = 0.25
    _next_id: int = 1
    _tracks: list[TrackedFace] = field(default_factory=list)

    def update(self, detections: list[FaceDetection]) -> list[TrackedFace]:
        """Match detections to tracks; return active tracked faces."""
        matched_det: set[int] = set()
        matched_track: set[int] = set()

        # Greedy IoU matching
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self._tracks):
            for di, det in enumerate(detections):
                iou = _iou(track.bbox, det.bbox)
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)

        for _iou_score, ti, di in pairs:
            if ti in matched_track or di in matched_det:
                continue
            matched_track.add(ti)
            matched_det.add(di)
            track = self._tracks[ti]
            det = detections[di]
            track.bbox = det.bbox
            track.center = det.center
            track.metrics.update_ema(det.metrics.as_dict(), alpha=0.22)
            track.frames_lost = 0
            track.locked = True

        # Unmatched tracks age out
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track:
                track.frames_lost += 1
                track.locked = track.frames_lost < 3

        # New tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            sid = f"SUBJ-{self._next_id:03d}"
            self._next_id += 1
            metrics = EmotionMetrics()
            metrics.update_ema(det.metrics.as_dict(), alpha=1.0)
            self._tracks.append(
                TrackedFace(
                    subject_id=sid,
                    bbox=det.bbox,
                    center=det.center,
                    metrics=metrics,
                    frames_lost=0,
                    locked=True,
                )
            )

        self._tracks = [
            t for t in self._tracks if t.frames_lost <= self.max_lost_frames
        ]
        return [t for t in self._tracks if t.frames_lost == 0]
