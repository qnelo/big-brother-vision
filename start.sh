#!/bin/bash
# Quick start script for Big Brother Vision Virtual Camera

set -e

CAMERA_DEV="/dev/video0"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Big Brother Vision - Launcher${NC}"
echo "================================================"

echo -e "\n${YELLOW}Checking v4l2loopback module...${NC}"
if ! lsmod | grep -q v4l2loopback; then
    echo "Module not loaded. Loading now (requires sudo)..."
    if sudo modprobe v4l2loopback devices=1 video_nr=10 \
        card_label="Big-Brother-Vision-Cam" exclusive_caps=1; then
        echo -e "${GREEN}Module loaded successfully${NC}"
    else
        echo -e "${RED}Failed to load v4l2loopback module${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}v4l2loopback module already loaded${NC}"
fi

echo -e "\n${YELLOW}Checking camera availability ($CAMERA_DEV)...${NC}"
BUSY_PID=$(lsof -t "$CAMERA_DEV" 2>/dev/null || true)

if [ -n "$BUSY_PID" ]; then
    PROCESS_NAME=$(ps -p "$BUSY_PID" -o comm=)
    echo -e "${RED}Camera in use by: $PROCESS_NAME (PID: $BUSY_PID)${NC}"
    read -p "Terminate this process? (y/N): " choice
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        kill -9 "$BUSY_PID"
        sleep 1
        echo -e "${GREEN}Camera is now free.${NC}"
    else
        echo -e "${RED}Cannot proceed while camera is in use.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}Camera is free.${NC}"
fi

echo -e "\n${GREEN}Starting Big Brother Vision...${NC}"
echo "================================================"
echo -e "${BLUE}Controls: h = HUD | g = green | a = amber | q = quit${NC}"
echo "================================================"

if command -v uv &> /dev/null; then
    uv run python main.py "$@"
else
    echo "uv not found, running with python3..."
    if [ -d ".venv" ]; then
        # shellcheck source=/dev/null
        source .venv/bin/activate
    fi
    python3 main.py "$@"
fi
