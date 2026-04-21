#!/bin/bash
# Setup SGLATrack environment and download pretrained weights.
#
# Usage:
#   bash scripts/setup_sglatrack.sh              # Full setup (env + weights)
#   bash scripts/setup_sglatrack.sh --weights-only  # Just download weights

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SGLA_DIR="$PROJECT_ROOT/models/SGLATrack"
CKPT_DIR="$SGLA_DIR/checkpoints"

echo "=== SGLATrack Setup ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# -------------------------------------------------------
# 1. Check if SGLATrack repo exists
# -------------------------------------------------------
if [ ! -d "$SGLA_DIR/lib" ]; then
    echo "[1/3] Cloning SGLATrack repo..."
    git clone https://github.com/GXNU-ZhongLab/SGLATrack.git "$SGLA_DIR"
else
    echo "[1/3] SGLATrack repo found at $SGLA_DIR"
fi

# -------------------------------------------------------
# 2. Install Python dependencies (if not --weights-only)
# -------------------------------------------------------
if [ "$1" != "--weights-only" ]; then
    echo ""
    echo "[2/3] Installing Python dependencies..."
    pip install --quiet torch torchvision easydict PyYAML opencv-python timm 2>/dev/null || {
        echo "[WARN] pip install failed — install manually:"
        echo "  pip install torch torchvision easydict PyYAML opencv-python timm"
    }
else
    echo "[2/3] Skipping dependency install (--weights-only)"
fi

# -------------------------------------------------------
# 3. Download pretrained weights
# -------------------------------------------------------
echo ""
mkdir -p "$CKPT_DIR"
CKPT_FILE="$CKPT_DIR/sglatrack_ep0297.pth.tar"

if [ -f "$CKPT_FILE" ]; then
    echo "[3/3] Checkpoint already exists: $CKPT_FILE"
else
    echo "[3/3] Downloading pretrained weights..."
    echo ""
    echo "  SGLATrack weights are hosted on Google Drive."
    echo "  The authors provide them at:"
    echo "    https://drive.google.com/drive/folders/1k8mYFHSqmj8dT_EoZ_QKNkCiDvWNR4I"
    echo ""
    echo "  Please download 'sglatrack_ep0297.pth.tar' (DeiT-distilled model)"
    echo "  and place it in:"
    echo "    $CKPT_DIR/"
    echo ""
    echo "  Or use gdown if available:"
    echo "    pip install gdown"
    echo "    gdown --fuzzy 'https://drive.google.com/file/d/<FILE_ID>/view' -O $CKPT_FILE"
    echo ""

    # Try gdown if available
    if command -v gdown &> /dev/null; then
        echo "  gdown found, attempting download..."
        echo "  (If this fails, please download manually from the Google Drive link above)"
        # The file ID needs to be found from the Google Drive folder
        # For now, prompt user
        echo ""
        echo "  [ACTION REQUIRED] Download the checkpoint manually and place at:"
        echo "    $CKPT_FILE"
    else
        echo "  [ACTION REQUIRED] Download the checkpoint manually and place at:"
        echo "    $CKPT_FILE"
    fi
fi

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo ""
echo "=== Setup Summary ==="
echo "SGLATrack repo:  $SGLA_DIR"
echo "Checkpoint path: $CKPT_FILE"
[ -f "$CKPT_FILE" ] && echo "Checkpoint:      FOUND" || echo "Checkpoint:      MISSING (download required)"
echo ""
echo "To run inference on competition data:"
echo "  python scripts/run_competition.py --split train"
echo ""
echo "To evaluate locally:"
echo "  python scripts/evaluate_local.py"
