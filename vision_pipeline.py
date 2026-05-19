"""Big Brother Vision processing pipeline."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from cat_detector import CatDetector
from cat_emotion_mapper import CatEmotionMapper
from emotion_mapper import EmotionMapper, EmotionMetrics, FaceDetection
from face_tracker import FaceTracker, TrackedFace, _iou
from hud_renderer import HudRenderer

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)
MODEL_FILENAME = "face_landmarker.task"


class VisionPipeline:
    """Face landmarker, emotion mapping, tracking, and HUD rendering."""

    def __init__(
        self,
        max_faces: int = 4,
        max_cats: int = 2,
        detect_cats: bool = True,
        detect_every_n_frames: int = 1,
        hud_color: str = "green",
        hud_visible: bool = True,
        assets_dir: Path | None = None,
    ):
        self.max_faces = max_faces
        self.max_cats = max_cats
        self.detect_cats = detect_cats
        self._detect_every_n = max(1, detect_every_n_frames)
        self.hud_visible = hud_visible
        self._process_frame_count = 0
        self._video_timestamp_ms = 0

        project_root = Path(__file__).parent
        self.assets_dir = assets_dir or (project_root / "assets")
        self.models_dir = self.assets_dir / "models"
        self.font_path = (
            self.assets_dir / "fonts" / "ShareTechMono-Regular.ttf"
        )

        self.emotion_mapper = EmotionMapper()
        self.cat_emotion_mapper = CatEmotionMapper()
        self.face_tracker = FaceTracker()
        self.hud_renderer = HudRenderer(
            color_mode=hud_color,
            font_path=self.font_path,
        )

        self._cat_detector: CatDetector | None = None
        if detect_cats:
            self._cat_detector = CatDetector()

        model_path = self._ensure_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=max_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )

        self._detect_width = 320
        self._detection_frame: np.ndarray | None = None
        self._detection_rgb: np.ndarray | None = None
        self._last_detections: list[FaceDetection] = []
        self._last_tracks: list[TrackedFace] = []

    def _ensure_model(self) -> Path:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.models_dir / MODEL_FILENAME
        if not model_path.exists():
            print(f"Downloading model {MODEL_FILENAME}...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print(f"Model downloaded: {model_path}")
        return model_path

    def set_hud_visible(self, visible: bool) -> None:
        self.hud_visible = visible

    def set_hud_color(self, color: str) -> None:
        self.hud_renderer.set_color_mode(color)

    def _bbox_from_landmarks(
        self, landmarks, frame_w: int, frame_h: int, padding: float = 0.12
    ) -> tuple[int, int, int, int]:
        xs = [lm.x * frame_w for lm in landmarks]
        ys = [lm.y * frame_h for lm in landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max_x - min_x
        h = max_y - min_y
        pad_x = w * padding
        pad_y = h * padding
        x = int(max(0, min_x - pad_x))
        y = int(max(0, min_y - pad_y))
        x2 = int(min(frame_w, max_x + pad_x))
        y2 = int(min(frame_h, max_y + pad_y))
        return x, y, x2 - x, y2 - y

    def _history_key_for_cat(self, det: FaceDetection) -> str:
        best_key = f"new-{det.center[0]}-{det.center[1]}"
        best_iou = 0.0
        for track in self._last_tracks:
            if track.species != "cat":
                continue
            iou = _iou(track.bbox, det.bbox)
            if iou >= 0.25 and iou > best_iou:
                best_iou = iou
                best_key = track.subject_id
        return best_key

    def _apply_cat_metrics(
        self, frame: np.ndarray, detections: list[FaceDetection]
    ) -> None:
        for det in detections:
            if det.species != "cat":
                continue
            history_key = self._history_key_for_cat(det)
            history = self.cat_emotion_mapper.get_history(history_key)
            raw = self.cat_emotion_mapper.raw_metrics(
                frame, det.bbox, det.center, history
            )
            metrics = EmotionMetrics()
            metrics.update_ema(raw, alpha=1.0)
            det.metrics = metrics

    def _detect_humans(self, frame: np.ndarray) -> list[FaceDetection]:
        full_h, full_w = frame.shape[:2]
        detect_w = self._detect_width
        detect_h = max(1, int(full_h * detect_w / full_w))

        if (
            self._detection_frame is None
            or self._detection_frame.shape[0] != detect_h
            or self._detection_frame.shape[1] != detect_w
        ):
            self._detection_frame = np.empty(
                (detect_h, detect_w, 3), dtype=np.uint8
            )
            self._detection_rgb = np.empty(
                (detect_h, detect_w, 3), dtype=np.uint8
            )

        cv2.resize(
            frame,
            (detect_w, detect_h),
            dst=self._detection_frame,
            interpolation=cv2.INTER_LINEAR,
        )
        cv2.cvtColor(
            self._detection_frame,
            cv2.COLOR_BGR2RGB,
            dst=self._detection_rgb,
        )
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=self._detection_rgb
        )

        self._video_timestamp_ms += 33 * self._detect_every_n
        result = self.landmarker.detect_for_video(
            mp_image, self._video_timestamp_ms
        )

        detections: list[FaceDetection] = []
        if not result.face_landmarks:
            return detections

        blend_list = result.face_blendshapes or []

        for fi, face_lms in enumerate(result.face_landmarks):
            bbox = self._bbox_from_landmarks(face_lms, full_w, full_h)
            x, y, w, h = bbox
            cx = x + w // 2
            cy = y + h // 2

            bs_dict: dict[str, float] = {}
            if fi < len(blend_list) and blend_list[fi] is not None:
                bs_dict = self.emotion_mapper.blendshapes_to_dict(
                    blend_list[fi]
                )

            raw = self.emotion_mapper.raw_metrics(
                bs_dict, face_lms, full_w, full_h
            )
            metrics = EmotionMetrics()
            metrics.update_ema(raw, alpha=1.0)

            detections.append(
                FaceDetection(
                    bbox=bbox,
                    center=(cx, cy),
                    species="human",
                    landmarks=face_lms,
                    blendshapes=bs_dict,
                    metrics=metrics,
                )
            )

        return detections

    def _filter_cats_overlapping_humans(
        self,
        humans: list[FaceDetection],
        cats: list[FaceDetection],
        iou_threshold: float = 0.35,
    ) -> list[FaceDetection]:
        """Drop cat detections that overlap a human face (Haar false positives)."""
        kept: list[FaceDetection] = []
        for cat in cats:
            overlaps_human = any(
                _iou(cat.bbox, human.bbox) >= iou_threshold for human in humans
            )
            if not overlaps_human:
                kept.append(cat)
        return kept

    def _detect_subjects(self, frame: np.ndarray) -> list[FaceDetection]:
        detections = self._detect_humans(frame)

        if self.detect_cats and self._cat_detector is not None:
            cat_detections = self._cat_detector.detect(frame, self.max_cats)
            cat_detections = self._filter_cats_overlapping_humans(
                detections, cat_detections
            )
            self._apply_cat_metrics(frame, cat_detections)
            detections.extend(cat_detections)

        return detections

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run detection/tracking and optionally draw HUD."""
        self._process_frame_count += 1
        run_detect = (
            self._process_frame_count % self._detect_every_n == 0
            or not self._last_detections
        )

        if run_detect:
            self._last_detections = self._detect_subjects(frame)

        tracks = self.face_tracker.update(self._last_detections)
        self._last_tracks = tracks

        active_cat_keys = {
            t.subject_id for t in tracks if t.species == "cat"
        }
        self.cat_emotion_mapper.prune_stale(active_cat_keys)

        if not self.hud_visible:
            return frame

        return self.hud_renderer.render(frame, tracks)

    def close(self) -> None:
        if hasattr(self, "landmarker"):
            self.landmarker.close()

    def __del__(self) -> None:
        self.close()
