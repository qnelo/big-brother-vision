#!/bin/bash
# Quick start script for The Laughing Man Virtual Camera
# This script ensures the environment is ready and the camera is free

set -e  # Exit on error

CAMERA_DEV="/dev/video0"
VIRTUAL_DEV="/dev/video10"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎭 The Laughing Man Virtual Camera - Launcher${NC}"
echo "================================================"

# 1. Check/Load v4l2loopback module
echo -e "\n${YELLOW}📦 Checking v4l2loopback module...${NC}"
if ! lsmod | grep -q v4l2loopback; then
    echo "Module not loaded. Loading now (requires sudo)..."
    if sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Laughing-Man-Cam" exclusive_caps=1; then
        echo -e "${GREEN}✓ Module loaded successfully${NC}"
    else
        echo -e "${RED}❌ Failed to load v4l2loopback module${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ v4l2loopback module already loaded${NC}"
fi

# 2. Check camera availability
echo -e "\n${YELLOW}🔍 Checking camera availability ($CAMERA_DEV)...${NC}"
BUSY_PID=$(lsof -t $CAMERA_DEV 2>/dev/null || true)

if [ -n "$BUSY_PID" ]; then
    PROCESS_NAME=$(ps -p $BUSY_PID -o comm=)
    echo -e "${RED}⚠️  Camera is currently IN USE by process: $PROCESS_NAME (PID: $BUSY_PID)${NC}"
    echo "The camera must be free for this application to work."
    
    read -p "❓ Do you want to terminate this process automatically? (y/N): " choice
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        echo "🔪 Killing process $BUSY_PID..."
        kill -9 $BUSY_PID
        sleep 1
        echo -e "${GREEN}✓ Process terminated. Camera is now free.${NC}"
    else
        echo -e "${RED}❌ Cannot proceed while camera is in use. Please close the other application manually.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Camera is free.${NC}"
fi

# 3. Run the application
echo -e "\n${GREEN}🚀 Starting application...${NC}"
echo "================================================"

# Ask user if they want to enable the virtual background
EXTRA_ARGS=""
read -p "❓ Enable virtual background (person segmentation)? (y/N): " bg_choice
if [[ "$bg_choice" =~ ^[Yy]$ ]]; then
    # User explicitly requested background
    echo "Creating camera WITH virtual background..."
    EXTRA_ARGS=""
else
    # Default is no background
    echo "Creating camera WITHOUT virtual background..."
    EXTRA_ARGS="--no-background"
fi

if command -v uv &> /dev/null; then
    # Use uv if available (recommended)
    uv run python main.py $EXTRA_ARGS
else
    # Fallback for systems without uv
    echo "uv not found, running with python3..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
    python3 main.py $EXTRA_ARGS
fi
