"""
Face Overlay Module - The Laughing Man Virtual Camera

This module provides the FaceOverlay class that handles face detection using MediaPipe
and overlays a logo image (static PNG or animated APNG) on detected faces.
"""

import cv2
import mediapipe as mp
import numpy as np

import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, List


class FaceOverlay:
    """Handles face detection and logo overlay using MediaPipe."""
    
    # Model URL for MediaPipe face detection
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    
    # Model URL for MediaPipe selfie segmentation
    SEGMENTATION_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite"
    
    def __init__(self, logo_path: str, min_detection_confidence: float = 0.5, enable_background: bool = True):
        """
        Initialize the FaceOverlay with MediaPipe face detection and segmentation.
        
        Args:
            logo_path: Path to the logo image (must have alpha channel)
            min_detection_confidence: Minimum confidence for face detection (0.0-1.0)
            enable_background: Whether to enable virtual background/segmentation
        """
        self.enable_background = enable_background
        
        # Download models if they don't exist
        self.model_path = self._ensure_model(self.MODEL_URL, "blaze_face_short_range.tflite")
        
        if self.enable_background:
            self.segmentation_model_path = self._ensure_model(self.SEGMENTATION_MODEL_URL, "selfie_segmenter.tflite")
        
        # Initialize MediaPipe Face Detection
        base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence
        )
        self.face_detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        
        # Initialize MediaPipe Image Segmenter (only if enabled)
        if self.enable_background:
            seg_base_options = mp.tasks.BaseOptions(model_asset_path=str(self.segmentation_model_path))
            seg_options = mp.tasks.vision.ImageSegmenterOptions(
                base_options=seg_base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                output_category_mask=True
            )
            self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(seg_options)
        
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
        else:
            self.background_images = []
            self.current_bg_index = -1
            self.current_background = None
            self.resized_background = None

        # Cache for resized logos: (logo_size,) for static, (logo_size, frame_index) for APNG
        self.logo_cache = {}
        self.last_face_size = None
        
        # Smoothing state
        self.prev_bbox = None
        self.smoothing_factor = 0.8  # Increased from 0.4 for much smoother movement
        self.jitter_threshold = 2    # Ignore movements smaller than this (pixels)

        # Overlay visibility (can be toggled at runtime to show/hide logo)
        self.overlay_visible = True
    
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
        Sets self.original_frames + animation_* for APNG, or self.original_logo for static.
        """
        imreadanimation = getattr(cv2, "imreadanimation", None)
        if imreadanimation is not None:
            try:
                success, animation = imreadanimation(logo_path)
                if success and hasattr(animation, "frames") and len(animation.frames) > 1:
                    frames = [np.array(f) for f in animation.frames]
                    for i, f in enumerate(frames):
                        if f.shape[2] != 4:
                            raise ValueError(
                                f"APNG frame {i} must have 4 channels (alpha), got {f.shape[2]}"
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

    def _find_background_images(self) -> List[Path]:
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
                with open(config_path, "r") as f:
                    return int(f.read().strip())
            except (ValueError, IOError):
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
        except IOError as e:
            print(f"⚠️ Could not save background index: {e}")

    def _load_next_background(self) -> Optional[np.ndarray]:
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
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect a face in the frame and return its bounding box.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (x, y, width, height) if face detected, None otherwise
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process the frame
        detection_result = self.face_detector.detect(mp_image)
        
        if not detection_result.detections:
            self.prev_bbox = None  # Reset smoothing if lost face
            return None
        
        # Get the first detected face
        detection = detection_result.detections[0]
        bbox = detection.bounding_box
        
        # Convert to absolute coordinates
        x = bbox.origin_x
        y = bbox.origin_y
        width = bbox.width
        height = bbox.height
        
        # Apply smoothing if we have a previous detection
        if self.prev_bbox is not None:
            prev_x, prev_y, prev_w, prev_h = self.prev_bbox
            
            # check if the change is significant enough (anti-jitter)
            dx = abs(x - prev_x)
            dy = abs(y - prev_y)
            dw = abs(width - prev_w)
            dh = abs(height - prev_h)
            
            if dx < self.jitter_threshold and dy < self.jitter_threshold and \
               dw < self.jitter_threshold and dh < self.jitter_threshold:
                # If change is very small, keep previous box to avoid micro-jitter
                x, y, width, height = prev_x, prev_y, prev_w, prev_h
            else:
                # Smooth coordinates using exponential moving average
                x = int(prev_x * self.smoothing_factor + x * (1 - self.smoothing_factor))
                y = int(prev_y * self.smoothing_factor + y * (1 - self.smoothing_factor))
                width = int(prev_w * self.smoothing_factor + width * (1 - self.smoothing_factor))
                height = int(prev_h * self.smoothing_factor + height * (1 - self.smoothing_factor))
        
        self.prev_bbox = (x, y, width, height)
        
        return (x, y, width, height)
    
    def _overlay_image_alpha(
        self,
        background: np.ndarray,
        overlay: np.ndarray,
        x: int,
        y: int
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
        blended = (overlay_bgr * overlay_alpha + roi * (1 - overlay_alpha)).astype(np.uint8)
        
        # Update the background
        background[y1:y2, x1:x2] = blended
        
        return background
    
    def overlay_logo(
        self,
        frame: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
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

        # Resize with cache
        if len(self.logo_cache) > 60:
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

        result = self._overlay_image_alpha(frame, resized_logo, overlay_x, overlay_y)
        return result
    
    def _segment_and_replace_background(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment the person and replace the background.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Frame with background replaced
        """
        if self.current_background is None:
            return frame
            
        height, width = frame.shape[:2]
        
        # Resize background if needed (cache it)
        if self.resized_background is None or \
           self.resized_background.shape[0] != height or \
           self.resized_background.shape[1] != width:
            self.resized_background = cv2.resize(self.current_background, (width, height))
            
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run segmentation
        # We use current time in ms as timestamp
        timestamp_ms = int(time.time() * 1000)
        segmentation_result = self.segmenter.segment_for_video(mp_image, timestamp_ms)
        
        # Get category mask
        category_mask = segmentation_result.category_mask
        
        # Convert to numpy array
        mask_np = category_mask.numpy_view()
        
        # Create binary mask
        # Based on user report, the previous logic (mask > 0.1 -> person) resulted in "wall on person".
        # This implies that (mask > 0.1) actually identified the area where we wanted the WALL 
        # (or the composition was flipped).
        # Previous: frame * mask + bg * (1-mask)
        # If mask was "Wall on Person", then mask was 0 on Person and 1 on Bg? 
        # No, if mask was 0 on person, then term 2 (bg * 1) would show wall on person.
        # So "mask > 0.1" is likely 0 for person and 1 for background (255).
        # We want: Frame on Person (mask=0), Wall on Background (mask=1).
        # So: frame * (1 - mask) + bg * mask
        
        # Let's define bg_mask clearly
        # Assuming 255 is background and 0 is person
        bg_mask = (mask_np > 0.1).astype(np.float32)
        
        # Apply slight blur to mask for softer edges
        # Blur before expanding dims to avoid OpenCV error with 3D input
        bg_mask = cv2.GaussianBlur(bg_mask, (5, 5), 0)
        
        # Expand dimensions for broadcasting (H, W) -> (H, W, 1)
        bg_mask = np.expand_dims(bg_mask, axis=-1)
            
        # Composite
        # We want: 
        # - Where bg_mask is 1 (Background): Show Wall (bg_float)
        # - Where bg_mask is 0 (Person): Show Camera (frame_float)
        frame_float = frame.astype(np.float32)
        bg_float = self.resized_background.astype(np.float32)
        
        output = frame_float * (1.0 - bg_mask) + bg_float * bg_mask
        
        return output.astype(np.uint8)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame: segment person, replace background, detect face, and apply logo overlay.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Processed frame with background replacement and logo overlay (or without overlay if overlay_visible is False)
        """
        frame_with_bg = frame
        
        # 1. Segment and replace background (if enabled)
        if self.enable_background:
            frame_with_bg = self._segment_and_replace_background(frame)
        
        # 2. Detect face and overlay logo only when overlay is visible
        if self.overlay_visible:
            face_bbox = self.detect_face(frame)
            if face_bbox is not None:
                # 3. Overlay logo
                frame_with_bg = self.overlay_logo(frame_with_bg, face_bbox)
        
        return frame_with_bg

    def set_overlay_visible(self, visible: bool) -> None:
        """
        Show or hide the logo overlay at runtime.
        
        Args:
            visible: True to draw the logo on the face, False to show only the camera (and optional background).
        """
        self.overlay_visible = visible
    
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
            print(f"✓ Logo updated to {name} ({len(self.original_frames)} frames)")
        else:
            print(f"✓ Logo updated to: {name}")

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_detector'):
            self.face_detector.close()
        if hasattr(self, 'segmenter'):
            self.segmenter.close()

