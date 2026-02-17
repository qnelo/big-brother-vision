# Installation Guide - The Laughing Man Virtual Camera

This guide will walk you through the installation and configuration process of the virtual camera filter step by step.

## 📋 Prerequisites

- **Operating System**: Linux (Ubuntu 20.04+, Debian 11+, or similar)
- **Python**: 3.10 or higher
- **Webcam**: V4L2-compatible
- **Permissions**: Sudo access to install system packages

## 🔧 Step-by-Step Installation

### 1. Update System and Install Dependencies

```bash
sudo apt update
sudo apt install -y v4l2loopback-dkms v4l2loopback-utils libcairo2-dev
```

**What does each package install?**
- `v4l2loopback-dkms`: Kernel module to create virtual camera devices
- `v4l2loopback-utils`: Utilities to manage v4l2loopback
- `libcairo2-dev`: Library needed to convert SVG to PNG

### 2. Load the v4l2loopback Module

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Laughing-Man-Cam" exclusive_caps=1
```

**Parameters explained:**
- `devices=1`: Create a single virtual device
- `video_nr=10`: Device number (`/dev/video10`)
- `card_label="Laughing-Man-Cam"`: Name that will appear in Google Meet
- `exclusive_caps=1`: Necessary for compatibility with modern browsers

**Verify that the module loaded correctly:**
```bash
# Should show "v4l2loopback"
lsmod | grep v4l2loopback

# Should show "Laughing-Man-Cam" as /dev/video10
v4l2-ctl --list-devices
```

### 3. Configure User Permissions

Add your user to the `video` group to access camera devices:

```bash
sudo usermod -aG video $USER
```

**⚠️ IMPORTANT**: You must log out and log back in for the changes to take effect.

Verify that you are in the group:
```bash
groups | grep video
```

### 4. [OPTIONAL] Configure Module Persistence

If you want the v4l2loopback module to load automatically at system startup:

```bash
# Load module at startup
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf

# Configure module options
echo "options v4l2loopback devices=1 video_nr=10 card_label='Laughing-Man-Cam' exclusive_caps=1" | sudo tee /etc/modprobe.d/v4l2loopback.conf
```

**Verify configuration:**
```bash
cat /etc/modules-load.d/v4l2loopback.conf
cat /etc/modprobe.d/v4l2loopback.conf
```

### 5. Install Python Dependencies

#### Option A: Using `uv` (⚡ RECOMMENDED)

`uv` is significantly faster than traditional pip and handles dependencies better.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell to use uv
source ~/.bashrc  # or ~/.zshrc if using zsh

# Navigate to project directory
cd /home/qnelo/develop/personal/thelaughing-man

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install the project and its dependencies
uv pip install -e .
```

#### Option B: Using traditional `venv`

```bash
# Navigate to project directory
cd /home/qnelo/develop/personal/thelaughing-man

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Update pip
pip install --upgrade pip

# Install the project and its dependencies
pip install -e .
```

**Verify installation:**
```bash
python -c "import mediapipe, cv2, pyvirtualcam, cairosvg; print('✓ All dependencies installed correctly')"
```

## 🚀 First Run

```bash
# Make sure the virtual environment is activated
source .venv/bin/activate

# Run the script
python main.py
```

**Expected output:**
```
📥 Downloading logo from https://static.wikia.nocookie.net/...
✓ Logo downloaded successfully: assets/laughing_man.svg
🔄 Converting SVG to PNG...
✓ Logo converted to PNG: assets/laughing_man.png
📷 Initializing camera: /dev/video0...
✓ Camera initialized: 640x480 @ 30fps
🎥 Initializing virtual camera: /dev/video10...
✓ Virtual camera initialized: /dev/video10
🎭 Initializing face detection...
✓ Face detection initialized

============================================================
🎉 The Laughing Man Camera is now running!
============================================================
📹 Virtual camera is available at: /dev/video10
💡 In Google Meet, select 'Laughing-Man-Cam' as your camera
🛑 Press Ctrl+C to stop
============================================================

📊 FPS: 29.8
```

## 🎥 Configure in Google Meet

1. Open Google Meet in your preferred browser (Chrome/Firefox)
2. Start or join a meeting
3. Click on the three dots (⋮) → **Settings**
4. Go to the **Video** tab
5. Under "Camera", select **"Laughing-Man-Cam"**
6. You should see the filter applied in real-time!

## 🔄 Daily Use

### Start the virtual camera

```bash
cd /home/qnelo/develop/personal/thelaughing-man
source .venv/bin/activate
python main.py
```

### Stop the virtual camera

Press `Ctrl+C` in the terminal where it's running.

### Convenience script (optional)

You can create a script to start quickly:

```bash
#!/bin/bash
# Save as ~/start-laughing-man.sh

cd /home/qnelo/develop/personal/thelaughing-man
source .venv/bin/activate
python main.py
```

Hacerlo ejecutable:
```bash
chmod +x ~/start-laughing-man.sh
```

Run:
```bash
~/start-laughing-man.sh
```

## 🐛 Troubleshooting

### The v4l2loopback module won't load

**Error**: `modprobe: FATAL: Module v4l2loopback not found`

**Solution**:
```bash
# Reinstall v4l2loopback-dkms
sudo apt remove v4l2loopback-dkms
sudo apt install v4l2loopback-dkms
```

### /dev/video10 is not created

**Check existing devices**:
```bash
ls /dev/video*
```

If `/dev/video10` already exists, change the number:
```bash
sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="Laughing-Man-Cam" exclusive_caps=1
```

And update `VIRTUAL_DEVICE` in `main.py`.

### Permission errors when accessing /dev/videoX

**Error**: `Permission denied`

**Solution**:
```bash
# Check permissions
ls -l /dev/video*

# Add user to video group
sudo usermod -aG video $USER

# Log out and log back in
```

### Python dependencies won't install

**Error related to `cairosvg`**:

Make sure `libcairo2-dev` is installed:
```bash
sudo apt install -y libcairo2-dev pkg-config
```

**Error related to `mediapipe`**:

MediaPipe may require additional dependencies:
```bash
sudo apt install -y python3-dev build-essential
```

### Camera is not detected in Google Meet

1. **Verify that the script is running** and there are no errors
2. **Reload the page** in Google Meet
3. **Check camera permissions** in the browser
4. **Test the virtual camera** with another tool first:
   ```bash
   ffplay /dev/video10
   ```

## 📚 Additional Resources

- [v4l2loopback Documentation](https://github.com/umlaeute/v4l2loopback)
- [MediaPipe Documentation](https://mediapipe.dev/)
- [pyvirtualcam Guide](https://github.com/letmaik/pyvirtualcam)
- [uv Documentation](https://github.com/astral-sh/uv)

## ❓ Need Help?

If you encounter any issues not covered in this guide, open an issue in the project repository.

---

Enjoy your virtual camera with style! 🎭✨
