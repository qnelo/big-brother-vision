"""
Face Overlay Module - The Laughing Man Virtual Camera

This module provides the FaceOverlay class that handles face detection using
MediaPipe and overlays a logo image (static PNG or animated APNG) on faces.
"""

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


class FaceOverlay:
    """Handles face detection and logo overlay using MediaPipe."""

    # Model URL for MediaPipe face detection
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

    # Model URL for MediaPipe selfie segmentation
    SEGMENTATION_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite"

    def __init__(
        self,
        logo_path: str,
        min_detection_confidence: float = 0.5,
        enable_background: bool = True,
        detect_every_n_frames: int = 1,
    ):
        """
        Initialize FaceOverlay with MediaPipe face detection and segmentation.

        Args:
            logo_path: Path to the logo image (must have alpha channel)
            min_detection_confidence: Minimum confidence for face detection
                (0.0-1.0)
            enable_background: Enable virtual background/segmentation
            detect_every_n_frames: Run face detection every N frames
                (1=every frame, 2=half the calls)
        """
        self.enable_background = enable_background
        self._detect_every_n = max(1, detect_every_n_frames)

        # Download models if they don't exist
        self.model_path = self._ensure_model(
            self.MODEL_URL, "blaze_face_short_range.tflite"
        )

        if self.enable_background:
            self.segmentation_model_path = self._ensure_model(
                self.SEGMENTATION_MODEL_URL,
                "selfie_segmenter.tflite",
            )

        # MediaPipe Face Detection (VIDEO mode for temporal consistency)
        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(self.model_path)
        )
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        self.face_detector = mp.tasks.vision.FaceDetector.create_from_options(
            options
        )

        # Detection at reduced resolution: target width for the small frame
        self._detect_width = 320
        self._detection_frame = None  # BGR buffer, (detect_h, detect_w, 3)
        self._detection_rgb = None  # RGB buffer for MediaPipe
        self._video_timestamp_ms = 0  # Monotonic timestamp for VIDEO mode

        # Initialize MediaPipe Image Segmenter (only if enabled)
        if self.enable_background:
            seg_base_options = mp.tasks.BaseOptions(
                model_asset_path=str(self.segmentation_model_path)
            )
            seg_options = mp.tasks.vision.ImageSegmenterOptions(
                base_options=seg_base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                output_category_mask=True,
            )
            self.segmenter = (
                mp.tasks.vision.ImageSegmenter.create_from_options(seg_options)
            )

        # Load logo (APNG animation or static PNG with alpha channel)
        self.original_logo = None
        self.original_frames = None
        self.animation_durations = None
        self.animation_loop_count = 0
        self.current_frame_index = 0
        self._load_logo_from_path(logo_path)

        # Background management (only if enabled)
        if self.enable_background:
            self.background_images = self._find_background_images()
            self.current_bg_index = self._load_last_bg_index()
            self.current_background = self._load_next_background()
            self.resized_background = None
            # Segmentation at reduced resolution + reusable buffers
            self._seg_scale = 0.5  # segment at half size
            self._seg_frame = None
            self._seg_rgb = None
            self._mask_full = None
            self._seg_timestamp_ms = 0
            self._frame_float = None
            self._bg_float = None
            self._mask_3ch = None
        else:
            self.background_images = []
            self.current_bg_index = -1
            self.current_background = None
            self.resized_background = None

        # Cache for resized logos: (logo_size,) static, (logo_size, idx) APNG
        self.logo_cache = {}
        self.last_face_size = None

        # Smoothing state
        self.prev_bbox = None
        # Increased from 0.4 for much smoother movement
        self.smoothing_factor = 0.8
        self.jitter_threshold = (
            2  # Ignore movements smaller than this (pixels)
        )

        # Overlay visibility (can be toggled at runtime to show/hide logo)
        self.overlay_visible = True

        # Reuse last bbox when not running detection this frame
        self._last_face_bbox = None
        self._process_frame_count = 0

    def _ensure_model(self, url: str, filename: str) -> Path:
        """
        Ensure a model is downloaded and available.

        Args:
            url: URL to download the model from
            filename: Name of the model file

        Returns:
            Path to the model file
        """
        # Store model in assets directory
        project_root = Path(__file__).parent
        assets_dir = project_root / "assets"
        assets_dir.mkdir(exist_ok=True)

        model_path = assets_dir / filename

        if not model_path.exists():
            print(f"📥 Downloading model {filename}...")
            urllib.request.urlretrieve(url, model_path)
            print(f"✓ Model downloaded: {model_path}")

        return model_path

    def _load_logo_from_path(self, logo_path: str) -> None:
        """
        Load logo from path as APNG (if multiple frames) or static PNG.
        Sets self.original_frames + animation_* for APNG, or
        self.original_logo for static.
        """
        imreadanimation = getattr(cv2, "imreadanimation", None)
        if imreadanimation is not None:
            try:
                success, animation = imreadanimation(logo_path)
                if (
                    success
                    and hasattr(animation, "frames")
                    and len(animation.frames) > 1
                ):
                    frames = [np.array(f) for f in animation.frames]
                    for i, f in enumerate(frames):
                        if f.shape[2] != 4:
                            raise ValueError(
                                f"APNG frame {i} must have 4 channels "
                                f"(alpha), got {f.shape[2]}"
                            )
                    self.original_frames = frames
                    self.animation_durations = getattr(
                        animation, "durations", [0] * len(frames)
                    )
                    self.animation_loop_count = getattr(
                        animation, "loop_count", 0
                    )
                    self.current_frame_index = 0
                    return
            except Exception:
                pass
        # Fallback: static PNG
        img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load logo from {logo_path}")
        if img.shape[2] != 4:
            raise ValueError("Logo must have an alpha channel (RGBA/BGRA)")
        self.original_logo = img

    def _find_background_images(self) -> list[Path]:
        """Find all wall{n}.jpg images in assets directory."""
        project_root = Path(__file__).parent
        assets_dir = project_root / "assets"

        images = []
        if assets_dir.exists():
            # Find all files matching wall{n}.jpg
            for file_path in assets_dir.glob("wall*.jpg"):
                images.append(file_path)

        # Sort by name (naturally)
        images.sort(key=lambda p: p.name)
        return images

    def _load_last_bg_index(self) -> int:
        """Load the index of the last used background."""
        project_root = Path(__file__).parent
        config_path = project_root / "assets" / ".last_bg"

        if config_path.exists():
            try:
                with open(config_path) as f:
                    return int(f.read().strip())
            except (OSError, ValueError):
                pass

        return -1  # Default to -1 so next is 0

    def _save_bg_index(self, index: int):
        """Save the current background index."""
        project_root = Path(__file__).parent
        assets_dir = project_root / "assets"
        assets_dir.mkdir(exist_ok=True)
        config_path = assets_dir / ".last_bg"

        try:
            with open(config_path, "w") as f:
                f.write(str(index))
        except OSError as e:
            print(f"⚠️ Could not save background index: {e}")

    def _load_next_background(self) -> np.ndarray | None:
        """Load the next background in the sequence."""
        if not self.background_images:
            return None

        # Calculate next index
        next_index = (self.current_bg_index + 1) % len(self.background_images)
        self.current_bg_index = next_index
        self._save_bg_index(next_index)

        image_path = self.background_images[next_index]
        print(f"🖼️ Loading background: {image_path.name}")

        bg_image = cv2.imread(str(image_path))
        if bg_image is None:
            print(f"⚠️ Failed to load background: {image_path}")
            return None

        return bg_image

    def detect_face(
        self, frame: np.ndarray
    ) -> tuple[int, int, int, int] | None:
        """
        Detect a face in the frame and return its bounding box (full-res).
        Runs the detector on a downscaled frame; bbox is scaled back.
        """
        full_h, full_w = frame.shape[:2]
        detect_w = self._detect_width
        detect_h = int(full_h * detect_w / full_w)
        if detect_h < 1:
            detect_h = 1

        # Reuse or allocate buffers for reduced-resolution detection
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
            self._detection_frame, cv2.COLOR_BGR2RGB, dst=self._detection_rgb
        )
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=self._detection_rgb
        )

        # VIDEO mode requires monotonic timestamp in ms (~30 fps)
        self._video_timestamp_ms += 33 * self._detect_every_n
        detection_result = self.face_detector.detect_for_video(
            mp_image, self._video_timestamp_ms
        )

        if not detection_result.detections:
            self.prev_bbox = None
            return None

        detection = detection_result.detections[0]
        bbox = detection.bounding_box
        scale_x = full_w / detect_w
        scale_y = full_h / detect_h

        # Scale bbox from small frame to full resolution
        x = int(bbox.origin_x * scale_x)
        y = int(bbox.origin_y * scale_y)
        width = int(bbox.width * scale_x)
        height = int(bbox.height * scale_y)

        # Apply smoothing in full-resolution coordinates
        if self.prev_bbox is not None:
            prev_x, prev_y, prev_w, prev_h = self.prev_bbox
            dx = abs(x - prev_x)
            dy = abs(y - prev_y)
            dw = abs(width - prev_w)
            dh = abs(height - prev_h)
            if (
                dx < self.jitter_threshold
                and dy < self.jitter_threshold
                and dw < self.jitter_threshold
                and dh < self.jitter_threshold
            ):
                x, y, width, height = prev_x, prev_y, prev_w, prev_h
            else:
                x = int(
                    prev_x * self.smoothing_factor
                    + x * (1 - self.smoothing_factor)
                )
                y = int(
                    prev_y * self.smoothing_factor
                    + y * (1 - self.smoothing_factor)
                )
                width = int(
                    prev_w * self.smoothing_factor
                    + width * (1 - self.smoothing_factor)
                )
                height = int(
                    prev_h * self.smoothing_factor
                    + height * (1 - self.smoothing_factor)
                )

        self.prev_bbox = (x, y, width, height)
        return (x, y, width, height)

    def _overlay_image_alpha(
        self, background: np.ndarray, overlay: np.ndarray, x: int, y: int
    ) -> np.ndarray:
        """
        Overlay an RGBA image on a BGR background using alpha blending.

        Args:
            background: BGR background image
            overlay: RGBA overlay image
            x, y: Top-left corner position for overlay

        Returns:
            BGR image with overlay applied
        """
        overlay_h, overlay_w = overlay.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # Ensure the overlay fits within the background
        if x >= bg_w or y >= bg_h:
            return background

        # Clip overlay to fit within background bounds
        x1, y1 = max(0, x), max(0, y)
        x2 = min(bg_w, x + overlay_w)
        y2 = min(bg_h, y + overlay_h)

        # Adjust overlay if it starts outside the frame
        overlay_x1 = max(0, -x)
        overlay_y1 = max(0, -y)
        overlay_x2 = overlay_x1 + (x2 - x1)
        overlay_y2 = overlay_y1 + (y2 - y1)

        # Extract the region of interest
        roi = background[y1:y2, x1:x2]
        overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

        # Separate the color and alpha channels
        overlay_bgr = overlay_crop[:, :, :3]
        overlay_alpha = overlay_crop[:, :, 3:4] / 255.0

        # Blend the images
        blended = (
            overlay_bgr * overlay_alpha + roi * (1 - overlay_alpha)
        ).astype(np.uint8)

        # Update the background
        background[y1:y2, x1:x2] = blended

        return background

    def overlay_logo(
        self, frame: np.ndarray, face_bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Overlay the logo (static or APNG animation) on the detected face.

        Args:
            frame: Input frame (BGR format)
            face_bbox: Face bounding box (x, y, width, height)

        Returns:
            Frame with logo overlay
        """
        x, y, width, height = face_bbox

        # Calculate face size (use the larger dimension to ensure coverage)
        face_size = max(width, height)
        logo_size = int(face_size * 1.55)
        logo_size = (logo_size // 2) * 2

        # Get current frame (APNG or static)
        if self.original_frames is not None:
            frame_index = self.current_frame_index % len(self.original_frames)
            current_logo = self.original_frames[frame_index]
            cache_key = (logo_size, frame_index)
            self.current_frame_index += 1
        else:
            current_logo = self.original_logo
            cache_key = logo_size

        # Resize with cache (cap size to limit memory)
        if len(self.logo_cache) > 45:
            self.logo_cache.clear()
        if cache_key not in self.logo_cache:
            self.logo_cache[cache_key] = cv2.resize(
                current_logo,
                (logo_size, logo_size),
                interpolation=cv2.INTER_LINEAR,
            )
        resized_logo = self.logo_cache[cache_key]

        # Calculate center position
        center_x = x + width // 2
        center_y = y + height // 2
        y_offset = int(height * 0.15)
        overlay_x = center_x - logo_size // 2
        overlay_y = center_y - logo_size // 2 - y_offset

        result = self._overlay_image_alpha(
            frame, resized_logo, overlay_x, overlay_y
        )
        return result

    def _segment_and_replace_background(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment the person and replace the background.
        Runs at reduced resolution for performance; mask is upscaled.
        """
        if self.current_background is None:
            return frame

        height, width = frame.shape[:2]
        seg_w = max(1, int(width * self._seg_scale))
        seg_h = max(1, int(height * self._seg_scale))

        # Resize background if needed (cache it)
        if (
            self.resized_background is None
            or self.resized_background.shape[0] != height
            or self.resized_background.shape[1] != width
        ):
            self.resized_background = cv2.resize(
                self.current_background, (width, height)
            )

        # Allocate or reuse reduced-resolution buffers for segmentation
        if (
            self._seg_frame is None
            or self._seg_frame.shape[0] != seg_h
            or self._seg_frame.shape[1] != seg_w
        ):
            self._seg_frame = np.empty((seg_h, seg_w, 3), dtype=np.uint8)
            self._seg_rgb = np.empty((seg_h, seg_w, 3), dtype=np.uint8)
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

        self._seg_timestamp_ms += 33
        segmentation_result = self.segmenter.segment_for_video(
            mp_image, self._seg_timestamp_ms
        )
        mask_np = segmentation_result.category_mask.numpy_view()

        # Binary mask: background = 1, person = 0
        bg_mask_small = (mask_np > 0.1).astype(np.float32)
        bg_mask_small = cv2.GaussianBlur(bg_mask_small, (3, 3), 0)

        # Upscale mask to full resolution and reuse buffer
        if (
            self._mask_full is None
            or self._mask_full.shape[0] != height
            or self._mask_full.shape[1] != width
        ):
            self._mask_full = np.empty((height, width), dtype=np.float32)
            self._mask_3ch = np.empty((height, width, 1), dtype=np.float32)
            self._frame_float = np.empty((height, width, 3), dtype=np.float32)
            self._bg_float = np.empty((height, width, 3), dtype=np.float32)
        cv2.resize(
            bg_mask_small,
            (width, height),
            dst=self._mask_full,
            interpolation=cv2.INTER_LINEAR,
        )
        self._mask_3ch[:, :, 0] = self._mask_full

        self._frame_float[:] = frame
        self._bg_float[:] = self.resized_background
        # Composite: person = frame, background = wall
        output = (
            self._frame_float * (1.0 - self._mask_3ch)
            + self._bg_float * self._mask_3ch
        )
        return output.astype(np.uint8)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame: segment, replace background, detect face, apply logo.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Processed frame with background and logo (or without if hidden).
        """
        frame_with_bg = frame

        # 1. Segment and replace background (if enabled)
        if self.enable_background:
            frame_with_bg = self._segment_and_replace_background(frame)

        # 2. Detect face and overlay logo only when overlay is visible
        if self.overlay_visible:
            self._process_frame_count += 1
            if (
                self._process_frame_count % self._detect_every_n == 0
                or self._last_face_bbox is None
            ):
                self._last_face_bbox = self.detect_face(frame)
            face_bbox = self._last_face_bbox
            if face_bbox is not None:
                frame_with_bg = self.overlay_logo(frame_with_bg, face_bbox)

        return frame_with_bg

    def set_overlay_visible(self, visible: bool) -> None:
        """
        Show or hide the logo overlay at runtime.

        Args:
            visible: True to draw logo on face, False for camera only
                (and optional background).
        """
        self.overlay_visible = visible
        if not visible:
            self._last_face_bbox = None

    def set_logo(self, logo_path: str):
        """
        Change the logo image at runtime (supports static PNG and APNG).

        Args:
            logo_path: Path to the new logo image
        """
        self.original_logo = None
        self.original_frames = None
        self.animation_durations = None
        self.animation_loop_count = 0
        self.current_frame_index = 0
        try:
            self._load_logo_from_path(logo_path)
        except ValueError as e:
            print(f"❌ {e}")
            return
        self.logo_cache.clear()
        name = Path(logo_path).name
        if self.original_frames is not None:
            n = len(self.original_frames)
            print(f"✓ Logo updated to {name} ({n} frames)")
        else:
            print(f"✓ Logo updated to: {name}")

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, "face_detector"):
            self.face_detector.close()
        if hasattr(self, "segmenter"):
            self.segmenter.close()
