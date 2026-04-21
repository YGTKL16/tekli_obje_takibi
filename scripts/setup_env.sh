#!/usr/bin/env bash
# Setup script for object_track Distrobox container (Ubuntu 22.04)
# Usage: distrobox enter object_track -- bash scripts/setup_env.sh
set -euo pipefail

echo "=== Tracker Environment Setup ==="

# Avoid re-running if already done
MARKER="$HOME/.tracker_env_ready"

export DEBIAN_FRONTEND=noninteractive

echo "[1/6] System packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential cmake pkg-config git wget curl \
    python3-dev python3-pip python3-venv \
    libeigen3-dev \
    libopencv-dev python3-opencv \
    valgrind cppcheck

echo "[2/6] Pybind11..."
pip3 install --user pybind11[global] 2>/dev/null || pip3 install pybind11[global]

echo "[3/6] Python packages..."
pip3 install --user numpy pyyaml pytest

echo "[4/6] Checking NVIDIA/CUDA..."
if command -v nvidia-smi &>/dev/null; then
    echo "  GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    # PyTorch with CUDA
    pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
        echo "  [WARN] PyTorch CUDA install failed. Install manually."
else
    echo "  No NVIDIA GPU found. Installing CPU-only PyTorch."
    pip3 install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "[5/6] Verifying Eigen..."
pkg-config --modversion eigen3 && echo "  Eigen OK" || echo "  [WARN] Eigen pkg-config not found"

echo "[6/6] Test build..."
cd "$(dirname "$0")/.."
cmake -B build -DBUILD_TESTS=ON 2>&1 | tail -5
cmake --build build -j"$(nproc)" 2>&1 | tail -5
echo "  Build OK"

touch "$MARKER"
echo ""
echo "=== Setup complete ==="
echo "Run tests: cd build && ctest --output-on-failure"
echo "Python:    cd python && pytest"
