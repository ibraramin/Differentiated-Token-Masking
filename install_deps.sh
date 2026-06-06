#!/usr/bin/env bash
set -euo pipefail

echo "==============================================================================="
echo "  Differentiated Token Masking — Dependency Installer"
echo "==============================================================================="

echo ""
echo "[1/3] Installing PyTorch (CUDA 11.8 — compatible with any NVIDIA driver >=450)..."
python3 -m pip install --break-system-packages torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "[2/3] Installing pipeline dependencies..."
python3 -m pip install --break-system-packages \
  spacy datasets peft trl accelerate lm_eval transformers

echo ""
echo "[3/3] Downloading SpaCy English model (en_core_web_sm)..."
python3 -m spacy download en_core_web_sm

echo ""
echo "==============================================================================="
echo "  All dependencies installed successfully."
echo "  Verify:  python3 -c 'import torch; assert torch.cuda.is_available()'"
echo "==============================================================================="
