#!/usr/bin/env python3
"""
The Laughing Man Virtual Camera

A virtual camera filter that overlays the iconic "Laughing Man" logo from
Ghost in the Shell on your face in real-time, compatible with Google Meet and
other video conferencing applications.

Author: qnelo
License: MIT
"""

import signal
import sys
import time
from pathlib import Path

import cv2
import pyvirtualcam

from face_overlay import FaceOverlay


# Configuration
ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PNG_PATH = ASSETS_DIR / "laughing_man_video_transparent.png"
LOGO_WHITE_PNG_PATH = ASSETS_DIR / "laughing_man_video_white.png"
CAMERA_DEVICE = "/dev/video0"  # Physical camera
VIRTUAL_DEVICE = "/dev/video10"  # Virtual camera (must match v4l2loopback device)
TARGET_FPS = 30


class LaughingManCamera:
    """Main application class for the Laughing Man virtual camera."""
    
    def __init__(self):
        self.camera = None
        self.virtual_cam = None
        self.face_overlay = None
        self.running = True
        self.enable_background = True  # Default to True
        self.use_white_logo = True  # Default to white logo
        self.overlay_visible = True  # Show logo overlay by default; toggle with 'f'
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n🛑 Shutting down gracefully...")
        self.running = False
    
    def verify_logo(self) -> bool:
        """
        Verify that the logo file exists.
        
        Returns:
            True if logo exists, False otherwise
        """
        if not LOGO_PNG_PATH.exists():
            print(f"❌ Logo file not found: {LOGO_PNG_PATH}")
            print(f"💡 The logo should be included with the project in the assets/ directory")
            return False
        
        print(f"✓ Logo found: {LOGO_PNG_PATH}")
        return True
    
    def initialize_camera(self) -> bool:
        """
        Initialize the physical camera.
        
        Returns:
            True if successful, False otherwise
        """
        # Try multiple camera devices in order
        camera_devices = ["/dev/video0", "/dev/video1", "/dev/video2"]
        
        for device in camera_devices:
            print(f"📷 Trying camera: {device}...")
            
            try:
                # Open with V4L2 backend
                camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
                
                if camera.isOpened():
                    # Force MJPG format to get higher FPS
                    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    
                    # Get actual camera properties
                    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(camera.get(cv2.CAP_PROP_FPS))
                    format_code = int(camera.get(cv2.CAP_PROP_FOURCC))
                    format_str = "".join([chr((format_code >> 8 * i) & 0xFF) for i in range(4)])
                    
                    # Check if we got valid properties
                    if width > 0 and height > 0:
                        self.camera = camera
                        print(f"✓ Camera initialized: {device} ({width}x{height} @ {fps}fps, {format_str})")
                        return True
                    else:
                        camera.release()
                        print(f"⚠️ {device} opened but returned invalid properties")
                else:
                    if camera is not None:
                        camera.release()
                    
            except Exception as e:
                print(f"⚠️ {device} failed: {e}")
                continue
        
        print(f"❌ Failed to open any camera device")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Check which process is using the camera:")
        print(f"      lsof /dev/video0")
        print(f"   2. Close the application (usually Chrome, Firefox, Zoom, etc.)")
        print(f"   3. Or kill the process:")
        print(f"      kill -9 <PID>")
        print(f"\n   Common culprits:")
        print(f"   - Chrome/Chromium with an open video call")
        print(f"   - Google Meet/Zoom/Teams")
        print(f"   - Another camera application")
        return False
    
    def initialize_virtual_camera(self) -> bool:
        """
        Initialize the virtual camera using pyvirtualcam.
        
        Returns:
            True if successful, False otherwise
        """
        print(f"🎥 Initializing virtual camera: {VIRTUAL_DEVICE}...")
        
        try:
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.virtual_cam = pyvirtualcam.Camera(
                width=width,
                height=height,
                fps=TARGET_FPS,
                fmt=pyvirtualcam.PixelFormat.BGR,
                device=VIRTUAL_DEVICE
            )
            
            print(f"✓ Virtual camera initialized: {self.virtual_cam.device}")
            print(f"  📹 Device: {VIRTUAL_DEVICE}")
            print(f"  📐 Resolution: {width}x{height}")
            print(f"  🎬 FPS: {TARGET_FPS}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize virtual camera: {e}")
            print(f"\n💡 Hint: Make sure v4l2loopback is loaded:")
            print(f"   sudo modprobe v4l2loopback devices=1 video_nr=10 card_label='Laughing-Man-Cam' exclusive_caps=1")
            return False
    
    def initialize_face_overlay(self) -> bool:
        """
        Initialize the face overlay processor.
        
        Returns:
            True if successful, False otherwise
        """
        print(f"🎭 Initializing face detection...")
        
        try:
            initial_logo = LOGO_WHITE_PNG_PATH if self.use_white_logo else LOGO_PNG_PATH
            print(f"🖼️ Using initial logo: {initial_logo.name}")
            
            self.face_overlay = FaceOverlay(
                logo_path=str(initial_logo),
                min_detection_confidence=0.5,
                enable_background=self.enable_background
            )
            
            print(f"✓ Face detection initialized (Background: {'Enabled' if self.enable_background else 'Disabled'})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize face overlay: {e}")
            return False
    
    def run(self) -> int:
        """
        Main application loop.
        
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        # Verify logo exists
        if not self.verify_logo():
            return 1
        
        if not self.initialize_camera():
            return 1
            
        # Allow camera to warm up
        time.sleep(2.0)
        
        # Initialize virtual camera
        if not self.initialize_virtual_camera():
            return 1
        
        # Initialize face overlay
        if not self.initialize_face_overlay():
            return 1
        
        print("\n" + "="*60)
        print("🎉 The Laughing Man Camera is now running!")
        print("="*60)
        print(f"📹 Virtual camera is available at: {VIRTUAL_DEVICE}")
        print(f"💡 In Google Meet, select 'Laughing-Man-Cam' as your camera")
        print(f"⌨️  Press 't' or SPACE to toggle logo color")
        print(f"⌨️  Press 'f' to show/hide overlay (no logo)")
        print(f"🛑 Press 'q' or Ctrl+C to stop")
        print("="*60 + "\n")
        
        # Create a window for input capture
        cv2.namedWindow("Laughing Man Control", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Laughing Man Control", 360, 202)
        
        frame_count = 0
        start_time = time.time()
        fps_display_interval = 3.0  # Display FPS every 3 seconds
        last_fps_display = start_time
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.camera.read()
                
                if not ret:
                    print("⚠️ Failed to capture frame from camera")
                    break
                
                # Process frame with face overlay
                processed_frame = self.face_overlay.process_frame(frame)
                
                # Send to virtual camera
                self.virtual_cam.send(processed_frame)
                
                # Calculate and display FPS periodically
                frame_count += 1
                current_time = time.time()
                elapsed = current_time - last_fps_display
                
                if elapsed >= fps_display_interval:
                    fps = frame_count / elapsed
                    print(f"📊 FPS: {fps:.1f}")
                    frame_count = 0
                    last_fps_display = current_time
                
                # Small delay to achieve target FPS
                self.virtual_cam.sleep_until_next_frame()

                # Show preview and capture input
                cv2.imshow("Laughing Man Control", processed_frame)
                key = cv2.waitKey(1) & 0xFF
                
                # Check for 'q' to quit
                if key == ord('q'):
                    self.running = False
                    
                # Check for 't' or space to toggle logo
                elif key == ord('t') or key == 32:  # 32 is space
                    self.use_white_logo = not self.use_white_logo
                    new_logo = LOGO_WHITE_PNG_PATH if self.use_white_logo else LOGO_PNG_PATH
                    print(f"🔄 Toggling logo to: {new_logo.name}")
                    self.face_overlay.set_logo(str(new_logo))

                # Check for 'f' to toggle overlay on/off (show or hide logo completely)
                elif key == ord('f'):
                    self.overlay_visible = not self.overlay_visible
                    self.face_overlay.set_overlay_visible(self.overlay_visible)
                    print(f"🔄 Overlay: {'ON' if self.overlay_visible else 'OFF'}")
        
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            return 1
        
        finally:
            self.cleanup()
        
        return 0
    
    def cleanup(self):
        """Release resources."""
        print("\n🧹 Cleaning up resources...")
        
        if self.camera is not None:
            self.camera.release()
            print("✓ Camera released")
        
        if self.virtual_cam is not None:
            self.virtual_cam.close()
            print("✓ Virtual camera closed")
        
        
        # Close any open windows
        cv2.destroyAllWindows()
        print("✓ Windows closed")
        
        print("👋 Goodbye!")


import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="The Laughing Man Virtual Camera")
    parser.add_argument("--no-background", action="store_true", help="Disable virtual background and segmentation")
    return parser.parse_args()

def main():
    """Entry point for the application."""
    args = parse_args()
    
    app = LaughingManCamera()
    
    # We need to pass the background setting to the app/overlay
    # Since LaughingManCamera initializes FaceOverlay internally, 
    # we should likely pass arguments to LaughingManCamera or handle it there.
    # Let's modify LaughingManCamera.__init__ to accept args or config.
    app.enable_background = not args.no_background
    
    sys.exit(app.run())


if __name__ == "__main__":
    main()
