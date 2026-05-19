# Big Brother Vision – Virtual Surveillance Camera

![Python 3.10+](https://shields.io/badge/Python-3.10+-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

*Real-time face tracking with big brother HUD overlays and emotion telemetry for video calls*

## Description

Big Brother Vision turns your webcam into a **virtual surveillance feed** for Linux. It detects every visible human face (`SUBJ-001`, …) and cat face (`CAT-001`, …), and draws a **big brother HUD** with live emotion telemetry, estimated age, and **LOYALTY TO BIG BROTHER**—a derived “adherence” score from positive affect and low fear.

The output is exposed as a V4L2 virtual camera compatible with Google Meet, Zoom, and other apps.

> **Disclaimer:** Metrics are approximate visual indicators. Human metrics use facial blendshapes; cat metrics use bbox and eye-region heuristics. For entertainment only—not psychological or medical measurements.

## Features

- Multi-face detection with MediaPipe Face Landmarker
- Cat face detection with OpenCV Haar cascade (`CAT-NNN` subject IDs)
- Real-time emotion telemetry from blendshapes and eye aspect ratio (EAR)
- **Estimated age** for humans (heuristic from face geometry + expression cues, smoothed per track)
- **LOYALTY TO BIG BROTHER** bar—computed from JOY, HAPPINESS, FOCUS, and inverse FEAR; cats receive a loyalty bonus so they typically score higher than humans
- Cat metrics from bbox geometry and estimated eye openness
- Big brother HUD: corner brackets, crosshair, large gauge bars with **full metric labels**, subject IDs, per-face side panel
- LOYALTY block styled separately (darker accent color, multi-line label, extra spacing below other metrics)
- Green / amber color palettes (toggle at runtime)
- Virtual camera at 30 FPS with optional detection throttling
- Automatic download of the Face Landmarker model on first run

## System Requirements

- **OS**: Linux (Ubuntu/Debian recommended)
- **Python**: 3.10 or higher
- **Hardware**: V4L2-compatible webcam

### System Dependencies

```bash
sudo apt update
sudo apt install -y v4l2loopback-dkms v4l2loopback-utils
```

## Quick Start

See [INSTALL.md](INSTALL.md) for full setup.

```bash
git clone https://github.com/your-user/big-brother-vision.git
cd big-brother-vision

sudo modprobe v4l2loopback devices=1 video_nr=10 \
  card_label="Big-Brother-Vision-Cam" exclusive_caps=1

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install -e .

./start.sh
```

## Usage

1. Start the app: `./start.sh` or `python main.py`
2. In Google Meet → Settings → Video → select **Big-Brother-Vision-Cam**
3. Keyboard shortcuts (preview window focused):
   - **`h`**: Show / hide HUD overlay
   - **`g`**: Green HUD palette
   - **`a`**: Amber HUD palette
   - **`q`** or **Ctrl+C**: Quit

### Command Line Options

| Flag | Description |
|------|-------------|
| `--max-faces N` | Maximum human faces to track (default: 2) |
| `--max-cats N` | Maximum cat faces to track (default: 2) |
| `--no-cats` | Disable cat face detection |
| `--hud-color green\|amber` | Initial HUD color |
| `--no-hud-overlay` | Raw camera feed only |
| `--no-preview` | No preview window (lower CPU) |
| `--detect-every N` | Run landmarker every N frames |

Example for weaker hardware:

```bash
python main.py --detect-every 2 --max-faces 2
```

## What the Metrics Mean

Each tracked subject gets a **side panel** next to their bounding box with:

1. **AGE** — humans only (`AGE: 034`); cats show `AGE: --`
2. Six **gauge bars** (0–100) with full English labels (no truncation)

| HUD Label | Approximate signal |
|-----------|-------------------|
| **JOY** | Smile and cheek activation |
| **HAPPINESS** | Overall positive expression |
| **FEAR** | Raised inner brows, wide eyes |
| **FOCUS** | Brow tension, neutral mouth, steady gaze |
| **DROWSY** | Eye closure (blink + low EAR) |
| **LOYALTY TO BIG BROTHER** | Derived score (not a direct blendshape): higher when JOY, HAPPINESS, and FOCUS are high and FEAR is low |

### LOYALTY TO BIG BROTHER

Computed each frame from the smoothed emotion values:

```
LOYALTY = 25%·JOY + 30%·HAPPINESS + 25%·FOCUS + 20%·(100 − FEAR)
```

- Displayed on **three lines** (`LOYALTY` / `TO BIG` / `BROTHER`) in a **darker accent color** than the standard HUD green (or darker gold in amber mode)
- Rendered **below** the five emotion bars with extra vertical spacing
- **Cats** get +28 points and a minimum of 72, so feline subjects almost always appear more loyal than humans with the same underlying emotions

### Age (humans only)

- Estimated from **face proportions** (forehead–chin height vs. cheek width) plus light adjustment from expression blendshapes
- Clamped to **16–85**; smoothed with a slower EMA than emotions so the number does not jitter
- Entertainment only—not real age detection

Values are smoothed (0–100%) and meant to look convincing on stream—not to be scientifically accurate.

### Cat faces

Cat metrics are **less precise** than human metrics because OpenCV Haar cascade only provides a bounding box (no facial landmarks). They are derived from:

- Estimated eye openness (Laplacian variance in the upper face region)
- Bbox stability and motion between frames
- Approximate slow-blink and alertness heuristics

Cats use the same HUD panel (including LOYALTY TO BIG BROTHER with the cat bonus) but **no age estimate**. Cat detection works best with frontal, well-lit faces. Profile views and very small cats may not be detected.

## Performance

- Face landmarker runs at reduced width (~320px)
- Cat Haar cascade uses the same downscaled frame
- Use `--detect-every 2` to halve inference cost
- Limit faces with `--max-faces 2` or `--max-cats 1` on laptops

## Troubleshooting

### Camera won't open
- Check devices: `ls /dev/video*`
- Free the camera: `lsof /dev/video0`

### Virtual camera missing
- Load module: `lsmod | grep v4l2loopback`
- Reload: `sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Big-Brother-Vision-Cam" exclusive_caps=1`

### Meet doesn't list the camera
- Restart the browser after starting the app
- Verify: `v4l2-ctl --list-devices`

## License

MIT License — see [LICENSE](LICENSE).

## Credits

- [MediaPipe](https://mediapipe.dev/) — Face Landmarker & blendshapes
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) — Virtual camera
- [OpenCV](https://opencv.org/) — Capture and rendering
