"""Virtual background replacement using selfie segmentation.

Replaces the camera background with one of the ``wallX.jpg`` images stored in
the assets directory, keeping the segmented person in the foreground.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Landscape (144x256) selfie segmenter: optimized for 16:9 video on CPU.
SEGMENTER_FILENAME = "selfie_segmenter_landscape.tflite"
SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter_landscape/float16/latest/"
    "selfie_segmenter_landscape.tflite"
)
_LEGACY_SEGMENTER_FILENAME = "selfie_segmenter.tflite"
_STATE_FILENAME = ".last_bg"
_WALL_PATTERN = re.compile(r"wall(\d+)\.jpg$", re.IGNORECASE)
# Width used to run segmentation; the mask is upscaled back to the frame size.
_SEG_WIDTH = 256


class BackgroundReplacer:
    """Segment the foreground person and composite a wall background."""

    def __init__(
        self,
        assets_dir: Path,
        model_path: Path | None = None,
        segment_every_n: int = 1,
    ) -> None:
        self.assets_dir = assets_dir
        self._state_path = assets_dir / _STATE_FILENAME
        self._segment_every_n = max(1, segment_every_n)
        self._frame_index = 0

        model = model_path or self._ensure_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model))
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_category_mask=False,
            output_confidence_masks=True,
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(
            options
        )
        self._video_timestamp_ms = 0

        # Ordered list of (wall_number, path) sorted by number.
        self._walls = self._discover_walls()
        self._bg_cache: dict[tuple[int, int, int], np.ndarray] = {}

        self._seg_frame: np.ndarray | None = None
        self._seg_rgb: np.ndarray | None = None
        self._mask_small: np.ndarray | None = None
        self._mask_full: np.ndarray | None = None
        self._inv_mask_full: np.ndarray | None = None
        self._composite: np.ndarray | None = None

        # 0 means disabled (no background replacement).
        self.active_number = 0
        self.set_by_number(self._load_state())

    def _ensure_model(self) -> Path:
        """Download the landscape segmenter, falling back to the legacy one."""
        models_dir = self.assets_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / SEGMENTER_FILENAME
        if model_path.exists():
            return model_path

        try:
            print(f"Downloading model {SEGMENTER_FILENAME}...")
            urllib.request.urlretrieve(SEGMENTER_URL, model_path)
            print(f"Model downloaded: {model_path}")
            return model_path
        except Exception as e:
            model_path.unlink(missing_ok=True)
            legacy = self.assets_dir / _LEGACY_SEGMENTER_FILENAME
            if legacy.exists():
                print(f"Download failed ({e}); using legacy model {legacy}")
                return legacy
            raise

    def _discover_walls(self) -> list[tuple[int, Path]]:
        walls: list[tuple[int, Path]] = []
        for path in self.assets_dir.glob("wall*.jpg"):
            match = _WALL_PATTERN.search(path.name)
            if match:
                walls.append((int(match.group(1)), path))
        walls.sort(key=lambda item: item[0])
        return walls

    @property
    def available_numbers(self) -> list[int]:
        return [number for number, _ in self._walls]

    def has_backgrounds(self) -> bool:
        return bool(self._walls)

    def _load_state(self) -> int:
        try:
            value = int(self._state_path.read_text().strip())
        except (OSError, ValueError):
            return 0
        return value

    def _save_state(self) -> None:
        try:
            self._state_path.write_text(str(self.active_number))
        except OSError:
            pass

    def set_by_number(self, number: int) -> None:
        """Activate a specific wall number (0 disables replacement)."""
        if number != 0 and number not in self.available_numbers:
            return
        self.active_number = number
        self._save_state()

    def cycle(self) -> int:
        """Advance to the next background (wrapping through 'off')."""
        order = [0] + self.available_numbers
        try:
            idx = order.index(self.active_number)
        except ValueError:
            idx = 0
        self.active_number = order[(idx + 1) % len(order)]
        self._save_state()
        return self.active_number

    def active_label(self) -> str:
        if self.active_number == 0:
            return "OFF"
        return f"wall{self.active_number}"

    def _wall_path(self, number: int) -> Path | None:
        for wall_number, path in self._walls:
            if wall_number == number:
                return path
        return None

    def _background_for(
        self, number: int, width: int, height: int
    ) -> np.ndarray | None:
        cache_key = (number, width, height)
        cached = self._bg_cache.get(cache_key)
        if cached is not None:
            return cached

        path = self._wall_path(number)
        if path is None:
            return None
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return None

        resized = self._cover_resize(image, width, height)
        self._bg_cache[cache_key] = resized
        return resized

    @staticmethod
    def _cover_resize(
        image: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """Resize keeping aspect ratio and center-crop to fill the frame."""
        src_h, src_w = image.shape[:2]
        scale = max(width / src_w, height / src_h)
        new_w = max(width, int(round(src_w * scale)))
        new_h = max(height, int(round(src_h * scale)))
        resized = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        x0 = (new_w - width) // 2
        y0 = (new_h - height) // 2
        return resized[y0:y0 + height, x0:x0 + width]

    def _ensure_buffers(self, full_w: int, full_h: int) -> bool:
        """(Re)allocate reusable buffers; return True if reallocated."""
        if (
            self._mask_full is not None
            and self._mask_full.shape[0] == full_h
            and self._mask_full.shape[1] == full_w
        ):
            return False
        self._mask_full = np.ones((full_h, full_w), dtype=np.float32)
        self._inv_mask_full = np.zeros((full_h, full_w), dtype=np.float32)
        self._composite = np.empty((full_h, full_w, 3), dtype=np.uint8)
        return True

    def _update_mask(self, frame: np.ndarray) -> None:
        """Run segmentation and refresh the full-resolution alpha masks."""
        full_h, full_w = frame.shape[:2]
        seg_w = _SEG_WIDTH
        seg_h = max(1, int(full_h * seg_w / full_w))

        if (
            self._seg_frame is None
            or self._seg_frame.shape[0] != seg_h
            or self._seg_frame.shape[1] != seg_w
        ):
            self._seg_frame = np.empty((seg_h, seg_w, 3), dtype=np.uint8)
            self._seg_rgb = np.empty((seg_h, seg_w, 3), dtype=np.uint8)
            self._mask_small = np.empty((seg_h, seg_w), dtype=np.float32)

        cv2.resize(
            frame,
            (seg_w, seg_h),
            dst=self._seg_frame,
            interpolation=cv2.INTER_LINEAR,
        )
        cv2.cvtColor(self._seg_frame, cv2.COLOR_BGR2RGB, dst=self._seg_rgb)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=self._seg_rgb
        )

        self._video_timestamp_ms += 33 * self._segment_every_n
        result = self.segmenter.segment_for_video(
            mp_image, self._video_timestamp_ms
        )
        if not result.confidence_masks:
            self._mask_full.fill(1.0)
            self._inv_mask_full.fill(0.0)
            return

        mask = np.asarray(result.confidence_masks[0].numpy_view())
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Feather the small mask before upscaling (much cheaper than
        # blurring at full resolution; bilinear upscale smooths further).
        cv2.GaussianBlur(mask, (0, 0), sigmaX=1.0, dst=self._mask_small)
        np.clip(self._mask_small, 0.0, 1.0, out=self._mask_small)
        cv2.resize(
            self._mask_small,
            (full_w, full_h),
            dst=self._mask_full,
            interpolation=cv2.INTER_LINEAR,
        )
        np.subtract(1.0, self._mask_full, out=self._inv_mask_full)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Composite the active wall behind the segmented foreground.

        Returns an internal buffer reused across frames; callers may draw
        on it but must not keep references across iterations.
        """
        if self.active_number == 0 or not self._walls:
            return frame

        full_h, full_w = frame.shape[:2]
        background = self._background_for(self.active_number, full_w, full_h)
        if background is None:
            return frame

        reallocated = self._ensure_buffers(full_w, full_h)
        self._frame_index += 1
        if reallocated or self._frame_index % self._segment_every_n == 0:
            self._update_mask(frame)

        cv2.blendLinear(
            frame,
            background,
            self._mask_full,
            self._inv_mask_full,
            dst=self._composite,
        )
        return self._composite

    def close(self) -> None:
        segmenter = getattr(self, "segmenter", None)
        if segmenter is not None:
            try:
                segmenter.close()
            except Exception:
                pass
