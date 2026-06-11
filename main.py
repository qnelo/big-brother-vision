#!/usr/bin/env python3
"""
Big Brother Vision - Virtual Surveillance Camera

Real-time multi-face tracking with big brother HUD overlays for Linux
virtual camera (v4l2loopback + pyvirtualcam).
"""

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyvirtualcam

from vision_pipeline import VisionPipeline

ASSETS_DIR = Path(__file__).parent / "assets"
CAMERA_DEVICE = "/dev/video0"
VIRTUAL_DEVICE = "/dev/video10"
TARGET_FPS = 30


class FrameGrabber:
    """Read camera frames on a dedicated thread, keeping only the newest.

    Prevents latency buildup from stale frames queued inside the V4L2
    capture buffer when processing is slower than the camera frame rate.
    """

    def __init__(self, camera: cv2.VideoCapture) -> None:
        self._camera = camera
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="frame-grabber", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def running(self) -> bool:
        return self._running

    def _loop(self) -> None:
        while self._running:
            ret, frame = self._camera.read()
            if not ret:
                with self._cond:
                    self._running = False
                    self._cond.notify_all()
                return
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def latest(
        self, last_seq: int, timeout: float = 1.0
    ) -> tuple[np.ndarray | None, int]:
        """Block until a frame newer than ``last_seq`` arrives."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq > last_seq or not self._running,
                timeout=timeout,
            )
            return self._frame, self._seq


class BigBrotherCamera:
    """Main application for Big Brother Vision virtual camera."""

    def __init__(self) -> None:
        self.camera = None
        self.virtual_cam = None
        self.pipeline: VisionPipeline | None = None
        self.grabber: FrameGrabber | None = None
        self.running = True
        self.show_preview = False
        self.detect_every = 2
        self.seg_every = 2
        self.max_faces = 2
        self.max_cats = 2
        self.detect_cats = True
        self.hud_color = "amber"
        self.hud_visible = True
        self.background: int | None = None

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        print("\nShutting down gracefully...")
        self.running = False

    def initialize_camera(self) -> bool:
        camera_devices = ["/dev/video0", "/dev/video1", "/dev/video2"]

        for device in camera_devices:
            print(f"Trying camera: {device}...")
            try:
                camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
                if not camera.isOpened():
                    camera.release()
                    continue

                camera.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                )
                camera.set(cv2.CAP_PROP_FPS, 30)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                # Keep at most one frame queued to avoid stale-frame lag.
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(camera.get(cv2.CAP_PROP_FPS))

                if width > 0 and height > 0:
                    self.camera = camera
                    print(f"Camera: {device} ({width}x{height} @ {fps}fps)")
                    return True
                camera.release()
            except Exception as e:
                print(f"{device} failed: {e}")

        print("Failed to open any camera device")
        return False

    def initialize_virtual_camera(self) -> bool:
        print(f"Initializing virtual camera: {VIRTUAL_DEVICE}...")
        try:
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.virtual_cam = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=TARGET_FPS,
                fmt=pyvirtualcam.PixelFormat.BGR,
                device=VIRTUAL_DEVICE,
            )
            print(f"Virtual camera: {self.virtual_cam.device}")
            return True
        except Exception as e:
            print(f"Failed to initialize virtual camera: {e}")
            print(
                "Hint: sudo modprobe v4l2loopback devices=1 video_nr=10 "
                "card_label='Big-Brother-Vision-Cam' exclusive_caps=1"
            )
            return False

    def initialize_pipeline(self) -> bool:
        print("Initializing face landmarker...")
        try:
            self.pipeline = VisionPipeline(
                max_faces=self.max_faces,
                max_cats=self.max_cats,
                detect_cats=self.detect_cats,
                detect_every_n_frames=max(1, self.detect_every),
                segment_every_n_frames=max(1, self.seg_every),
                hud_color=self.hud_color,
                hud_visible=self.hud_visible,
                assets_dir=ASSETS_DIR,
                background=self.background,
            )
            cat_info = (
                f", max_cats={self.max_cats}"
                if self.detect_cats
                else ", cats=disabled"
            )
            print(
                f"Face landmarker ready (max_faces={self.max_faces}{cat_info})"
            )
            return True
        except Exception as e:
            print(f"Failed to initialize pipeline: {e}")
            return False

    def run(self) -> int:
        if not self.initialize_camera():
            return 1

        time.sleep(1.0)

        if not self.initialize_virtual_camera():
            return 1

        if not self.initialize_pipeline():
            return 1

        print("\n" + "=" * 60)
        print("Big Brother Vision is running")
        print("=" * 60)
        print(f"Virtual camera: {VIRTUAL_DEVICE}")
        print("Select 'Big-Brother-Vision-Cam' in your video app")
        if self.show_preview:
            print("Keys: h=HUD | g=green | a=amber | b=background | q=quit")
        else:
            print("Preview disabled (use --preview for window + hotkeys)")
        print("=" * 60 + "\n")

        if self.show_preview:
            cv2.namedWindow("Big Brother Vision", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Big Brother Vision", 640, 360)

        self.grabber = FrameGrabber(self.camera)
        self.grabber.start()

        frame_count = 0
        preview_interval = 2
        last_fps = time.time()
        seq = 0
        process_ms = 0.0
        send_ms = 0.0

        try:
            while self.running:
                frame, new_seq = self.grabber.latest(seq)
                if not self.grabber.running and new_seq == seq:
                    print("Failed to capture frame")
                    break
                if frame is None or new_seq == seq:
                    continue
                seq = new_seq

                t0 = time.perf_counter()
                processed = self.pipeline.process_frame(frame)
                t1 = time.perf_counter()
                self.virtual_cam.send(processed)
                t2 = time.perf_counter()
                process_ms += (t1 - t0) * 1000.0
                send_ms += (t2 - t1) * 1000.0

                frame_count += 1
                now = time.time()
                if now - last_fps >= 3.0:
                    fps = frame_count / (now - last_fps)
                    t = self.pipeline.timings
                    print(
                        f"FPS: {fps:.1f} | "
                        f"process {process_ms / frame_count:.1f}ms "
                        f"(bg {t['bg_ms']:.1f} hud {t['hud_ms']:.1f}) | "
                        f"detect(async) {t['detect_ms']:.1f}ms | "
                        f"send {send_ms / frame_count:.1f}ms"
                    )
                    frame_count = 0
                    process_ms = 0.0
                    send_ms = 0.0
                    last_fps = now

                if self.show_preview:
                    if frame_count % preview_interval == 0:
                        cv2.imshow("Big Brother Vision", processed)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.running = False
                    elif key == ord("h"):
                        self.hud_visible = not self.hud_visible
                        self.pipeline.set_hud_visible(self.hud_visible)
                        state = "ON" if self.hud_visible else "OFF"
                        print(f"HUD overlay: {state}")
                    elif key == ord("g"):
                        self.hud_color = "green"
                        self.pipeline.set_hud_color("green")
                        print("HUD color: green")
                    elif key == ord("a"):
                        self.hud_color = "amber"
                        self.pipeline.set_hud_color("amber")
                        print("HUD color: amber")
                    elif key == ord("b"):
                        label = self.pipeline.cycle_background()
                        print(f"Background: {label}")

        except Exception as e:
            print(f"Error during processing: {e}")
            return 1
        finally:
            self.cleanup()

        return 0

    def cleanup(self) -> None:
        print("\nCleaning up...")
        if self.grabber is not None:
            self.grabber.stop()
        if self.camera is not None:
            self.camera.release()
        if self.virtual_cam is not None:
            self.virtual_cam.close()
        if self.pipeline is not None:
            self.pipeline.close()
        cv2.destroyAllWindows()
        print("Goodbye!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Big Brother Vision - Virtual Surveillance Camera"
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=4,
        metavar="N",
        help="Maximum faces to track (default: 4)",
    )
    parser.add_argument(
        "--max-cats",
        type=int,
        default=2,
        metavar="N",
        help="Maximum cat faces to track (default: 2)",
    )
    parser.add_argument(
        "--no-cats",
        action="store_true",
        help="Disable cat face detection",
    )
    parser.add_argument(
        "--hud-color",
        choices=("green", "amber"),
        default="amber",
        help="HUD color palette (default: amber)",
    )
    parser.add_argument(
        "--no-hud-overlay",
        action="store_true",
        help="Disable HUD overlay (raw camera feed)",
    )
    parser.add_argument(
        "--background",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Replace the background with assets/wallN.jpg "
            "(0 disables; default: remember last used)"
        ),
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview window with keyboard controls (higher CPU)",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=2,
        metavar="N",
        help="Run face landmarker every N frames (default: 2)",
    )
    parser.add_argument(
        "--seg-every",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Run background segmentation every N frames, reusing the "
            "previous mask in between (default: 2)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = BigBrotherCamera()
    app.max_faces = max(1, args.max_faces)
    app.max_cats = max(0, args.max_cats)
    app.detect_cats = not args.no_cats
    app.hud_color = args.hud_color
    app.hud_visible = not args.no_hud_overlay
    app.background = args.background
    app.show_preview = args.preview
    app.detect_every = max(1, args.detect_every)
    app.seg_every = max(1, args.seg_every)
    sys.exit(app.run())


if __name__ == "__main__":
    main()
