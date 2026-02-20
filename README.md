# The Laughing Man Virtual Camera 🎭

<div align="center">

![Ghost in the Shell](https://img.shields.io/badge/Inspired_by-Ghost_in_the_Shell-blueviolet)
![Python 3.10+](https://shields.io/badge/Python-3.10+-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

*Overlay the iconic "The Laughing Man" logo over your face in real-time for video calls* 👤➡️🎭

</div>

## 📖 Description

This project creates a virtual camera filter that detects your face in real-time and overlays the rotating "The Laughing Man" logo from Ghost in the Shell: Stand Alone Complex. The output is sent to a virtual camera compatible with Google Meet, Zoom, and other video conferencing applications.

## ✨ Features

- 🎯 **Face detection** with MediaPipe (high precision and performance)
- 🔄 **Continuous rotation** of the logo (faithful to the anime)
- 🎨 **Alpha blending** for perfect transparency
- 📹 **Virtual camera** compatible with video calling applications
- ⚡ **Optimized** for consistent 30 FPS
- 🎭 **Automatic download** of the logo from the official source

## 🔧 System Requirements

- **OS**: Linux (Ubuntu/Debian recommended)
- **Python**: 3.10 or higher
- **Hardware**: V4L2-compatible webcam

### System Dependencies

```bash
sudo apt update
sudo apt install -y v4l2loopback-dkms v4l2loopback-utils libcairo2-dev
```

## 🚀 Installation

For detailed step-by-step instructions, see [INSTALL.md](INSTALL.md).

### Quick Installation

```bash
# 1. Clone the repository
cd /home/qnelo/develop/personal/thelaughing-man

# 2. Load the v4l2loopback module
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Laughing-Man-Cam" exclusive_caps=1

# 3. Install with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv pip install -e .

# 4. Run
python main.py
```

## 💻 Usage

1. **Start the virtual camera**:
   ```bash
   ./start.sh
   ```
   Or manually:
   ```bash
   source .venv/bin/activate
   python main.py
   ```

2. **Configure in Google Meet**:
   - Open Google Meet in your browser
   - Go to Settings → Video
   - Select **"Laughing-Man-Cam"** as your camera
   - Done! The filter will be applied automatically

3. **Keyboard shortcuts** (focus on the "Laughing Man Control" window):
   - **`t`** or **Space**: Toggle logo style (e.g. white / transparent)
   - **`f`**: Show or hide the overlay (camera only, or camera + logo on face)
   - **`q`** or **Ctrl+C**: Quit the application

4. **Stop the application**:
   - Press `q` in the control window, or `Ctrl+C` in the terminal

## 📊 Performance

The system is optimized to maintain a constant 30 FPS:
- ✅ Face detection using MediaPipe's lightweight model
- ✅ Resized logo caching
- ✅ Optimized alpha blending with NumPy
- ✅ Efficient rotation with OpenCV

## 🔧 Advanced Configuration

### Change the virtual camera device

Edit `main.py` and modify:
```python
VIRTUAL_DEVICE = "/dev/video10"  # Change the number if necessary
```

### Adjust detection confidence

In `main.py`, modify:
```python
self.face_overlay = FaceOverlay(
    logo_path=str(LOGO_PNG_PATH),
    min_detection_confidence=0.5  # Range: 0.0 - 1.0
)
```

## 🐛 Troubleshooting

### Error: "Failed to open camera"
- Verify that your webcam is connected: `ls /dev/video*`
- Try another device: `CAMERA_DEVICE = "/dev/video1"`

### Error: "Failed to initialize virtual camera"
- Verify that v4l2loopback is loaded: `lsmod | grep v4l2loopback`
- Reload the module: `sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Laughing-Man-Cam" exclusive_caps=1`

### Face detection is slow
- Reduce your webcam resolution
- Increase `min_detection_confidence` to 0.6 or 0.7

### Google Meet doesn't show the virtual camera
- Verify that the device exists: `v4l2-ctl --list-devices`
- Restart the browser after starting the script
- Grant camera permissions to the browser

## 📄 License

MIT License - See LICENSE file for more details

## 🙏 Credits

- **Logo**: Ghost in the Shell: Stand Alone Complex
- **Face Detection**: [MediaPipe](https://mediapipe.dev/)
- **Virtual Camera**: [pyvirtualcam](https://github.com/letmaik/pyvirtualcam)
- **Inspiration**: The iconic "Laughing Man" scene 🎭

## 🤝 Contributions

Contributions are welcome! Please:
1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">
Made with ❤️ and Python 🐍
</div>
