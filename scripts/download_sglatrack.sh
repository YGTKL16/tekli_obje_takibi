#!/usr/bin/env bash
# Download or bootstrap the vendored SGLATrack repository layout.
set -euo pipefail

MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"

SGLATRACK_DIR="$MODELS_DIR/SGLATrack"
CKPT_DIR="$SGLATRACK_DIR/checkpoints"

if [ -d "$SGLATRACK_DIR/.git" ] || [ -d "$SGLATRACK_DIR/lib" ]; then
    echo "[INFO] SGLATrack already exists at $SGLATRACK_DIR"
else
    echo "[1/2] Cloning SGLATrack..."
    git clone --depth 1 https://github.com/GXNU-ZhongLab/SGLATrack.git "$SGLATRACK_DIR"
fi

mkdir -p "$CKPT_DIR"

echo "[2/2] Checkpoint setup..."
echo "  Download 'sglatrack_ep0297.pth.tar' manually from the authors' Google Drive"
echo "  and place it here:"
echo "    $CKPT_DIR/sglatrack_ep0297.pth.tar"
echo "  Project docs: docs/sglatrack/README.md"

echo ""
echo "=== SGLATrack bootstrap complete ==="
