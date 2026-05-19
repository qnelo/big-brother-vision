# Installation Guide - Big Brother Vision

Step-by-step setup for the virtual surveillance camera on Linux.

## Prerequisites

- **OS**: Linux (Ubuntu 20.04+, Debian 11+, or similar)
- **Python**: 3.10 or higher
- **Webcam**: V4L2-compatible
- **Permissions**: Sudo access for system packages

## Step-by-Step Installation

### 1. System Packages

```bash
sudo apt update
sudo apt install -y v4l2loopback-dkms v4l2loopback-utils
```

- `v4l2loopback-dkms`: Kernel module for virtual camera devices
- `v4l2loopback-utils`: Utilities to manage v4l2loopback

### 2. Load v4l2loopback

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 \
  card_label="Big-Brother-Vision-Cam" exclusive_caps=1
```

| Parameter | Meaning |
|-----------|---------|
| `devices=1` | One virtual device |
| `video_nr=10` | Device path `/dev/video10` |
| `card_label` | Name shown in Meet/Zoom |
| `exclusive_caps=1` | Browser compatibility |

Verify:

```bash
lsmod | grep v4l2loopback
v4l2-ctl --list-devices
```

### 3. User Permissions

```bash
sudo usermod -aG video $USER
```

Log out and back in, then verify:

```bash
groups | grep video
```

### 4. Optional: Load Module at Boot

```bash
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo "options v4l2loopback devices=1 video_nr=10 card_label='Big-Brother-Vision-Cam' exclusive_caps=1" | \
  sudo tee /etc/modprobe.d/v4l2loopback.conf
```

### 5. Python Environment

#### Using uv (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

cd big-brother-vision
uv venv
source .venv/bin/activate
uv pip install -e .
```

#### Using venv

```bash
cd big-brother-vision
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Verify:

```bash
python -c "import mediapipe, cv2, pyvirtualcam; print('OK')"
```

## First Run

```bash
source .venv/bin/activate
python main.py
```

On first run, the app downloads `face_landmarker.task` (~10 MB) into `assets/models/`.

Expected output:

```
Downloading model face_landmarker.task...
Camera: /dev/video0 (1280x720 @ 30fps)
Virtual camera initialized: /dev/video10
Face landmarker ready (max_faces=4)

Big Brother Vision is running
Select 'Big-Brother-Vision-Cam' in your video app
```

## Google Meet

1. Open Meet in Chrome or Firefox
2. Settings → Video → **Big-Brother-Vision-Cam**
3. Restart the browser if the device does not appear

## Daily Use

```bash
./start.sh
```

Or with options:

```bash
python main.py --detect-every 2 --max-faces 2 --hud-color amber
```

## Troubleshooting

### Module not found

```bash
sudo apt remove v4l2loopback-dkms
sudo apt install v4l2loopback-dkms
```

### /dev/video10 busy or missing

```bash
ls /dev/video*
```

Use another `video_nr` in modprobe and update `VIRTUAL_DEVICE` in `main.py`.

### Permission denied on /dev/video*

```bash
sudo usermod -aG video $USER
# log out and back in
```

### mediapipe install fails

```bash
sudo apt install -y python3-dev build-essential
```

### Model download fails

Download manually:

```bash
mkdir -p assets/models
curl -L -o assets/models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### Test virtual camera

```bash
ffplay /dev/video10
```

## Resources

- [v4l2loopback](https://github.com/umlaeute/v4l2loopback)
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam)
