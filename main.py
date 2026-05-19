#!/usr/bin/env python3
"""
Big Brother Vision - Virtual Surveillance Camera

Real-time multi-face tracking with fighter-jet HUD overlays for Linux
virtual camera (v4l2loopback + pyvirtualcam).
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
import pyvirtualcam

from vision_pipeline import VisionPipeline

ASSETS_DIR = Path(__file__).parent / "assets"
CAMERA_DEVICE = "/dev/video0"
VIRTUAL_DEVICE = "/dev/video10"
TARGET_FPS = 30


class BigBrotherCamera:
    """Main application for Big Brother Vision virtual camera."""

    def __init__(self) -> None:
        self.camera = None
        self.virtual_cam = None
        self.pipeline: VisionPipeline | None = None
        self.running = True
        self.show_preview = True
        self.detect_every = 1
        self.max_faces = 4
        self.max_cats = 2
        self.detect_cats = True
        self.hud_color = "green"
        self.hud_visible = True

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
                hud_color=self.hud_color,
                hud_visible=self.hud_visible,
                assets_dir=ASSETS_DIR,
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
        print("Keys: h=HUD | g=green | a=amber | q=quit")
        if not self.show_preview:
            print("Preview disabled (--no-preview)")
        print("=" * 60 + "\n")

        if self.show_preview:
            cv2.namedWindow("Big Brother Vision", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Big Brother Vision", 640, 360)

        frame_count = 0
        preview_interval = 2
        last_fps = time.time()

        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                processed = self.pipeline.process_frame(frame)
                self.virtual_cam.send(processed)

                frame_count += 1
                now = time.time()
                if now - last_fps >= 3.0:
                    fps = frame_count / (now - last_fps)
                    print(f"FPS: {fps:.1f}")
                    frame_count = 0
                    last_fps = now

                self.virtual_cam.sleep_until_next_frame()

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

        except Exception as e:
            print(f"Error during processing: {e}")
            return 1
        finally:
            self.cleanup()

        return 0

    def cleanup(self) -> None:
        print("\nCleaning up...")
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
        default="green",
        help="HUD color palette (default: green)",
    )
    parser.add_argument(
        "--no-hud-overlay",
        action="store_true",
        help="Disable HUD overlay (raw camera feed)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="No preview window (lower CPU)",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=1,
        metavar="N",
        help="Run face landmarker every N frames",
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
    app.show_preview = not args.no_preview
    app.detect_every = max(1, args.detect_every)
    sys.exit(app.run())


if __name__ == "__main__":
    main()
