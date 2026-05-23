#!/bin/bash
set -e

echo "[INFO] Updating apt packages..."
sudo apt update

echo "[INFO] Installing system dependencies..."
sudo apt install -y \
    ffmpeg \
    colmap \
    gcc-12 \
    g++-12 \
    git \
    ninja-build \
    curl \
    build-essential

# ============================================================
# NODE.JS (LATEST STABLE)
# ============================================================

echo "[INFO] Installing latest Node.js..."

if ! command -v n >/dev/null 2>&1; then
    sudo npm cache clean -f || true
    sudo npm install -g n
fi

sudo n stable

export PATH="/usr/local/bin:$PATH"

echo "[INFO] Node version:"
node -v

echo "[INFO] npm version:"
npm -v

# ============================================================
# CUDA + COMPILERS
# ============================================================

export CUDA_HOME=/usr/local/cuda-12.8
export CC=gcc-12
export CXX=g++-12

# ============================================================
# PYTHON
# ============================================================

echo "[INFO] Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo "[INFO] Installing PyTorch CUDA 12.8..."
pip install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo "[INFO] Installing Python requirements..."
pip install -r requirements.txt

# ============================================================
# GAUSSIAN SPLATTING
# ============================================================

mkdir -p third_party

if [ ! -d "third_party/gaussian-splatting" ]; then

    echo "[INFO] Cloning Gaussian Splatting repo..."

    git clone \
        https://github.com/graphdeco-inria/gaussian-splatting \
        third_party/gaussian-splatting

else

    echo "[INFO] Gaussian Splatting repo already exists."

fi

cd third_party/gaussian-splatting

echo "[INFO] Updating submodules..."
git submodule update --init --recursive

echo "[INFO] Installing diff-gaussian-rasterization..."
pip install --no-build-isolation \
    submodules/diff-gaussian-rasterization

echo "[INFO] Installing simple-knn..."
pip install --no-build-isolation \
    submodules/simple-knn

cd ../..

# ============================================================
# VIEWER
# ============================================================

if [ -d "viewer" ]; then

    echo "[INFO] Installing viewer npm dependencies..."

    cd viewer

    npm install

    cd ..

else

    echo "[WARN] viewer/ directory not found. Skipping viewer setup."

fi

echo "[OK] Setup complete."