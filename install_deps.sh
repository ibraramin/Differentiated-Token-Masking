#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "Thesis Pipeline  Dependency  Installer"
echo "============================================"

echo ""
echo "[1/3] Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "[2/3] Installing pipeline requirements..."
pip install -r requirements.txt

echo ""
echo "[3/3] Downloading SpaCy English model (en_core_web_sm)..."
python -m spacy download en_core_web_sm

echo ""
echo "============================================"
echo "All dependencies installed successfully."
echo "============================================"
