#!/usr/bin/env bash
# QFormer-RevMark environment setup
set -e

cd "$(dirname "$0")"

echo "===> Creating venv (Python 3.12)..."
python3 -m venv .venv
source .venv/bin/activate

echo "===> Upgrading pip/wheel/setuptools..."
pip install --upgrade pip wheel setuptools

echo "===> Installing PyTorch + CUDA 12.1 build..."
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

echo "===> Installing project requirements..."
pip install -r requirements.txt

echo "===> Cloning quaternion PyTorch ops (Parcollet QNN)..."
mkdir -p third_party
if [ ! -d third_party/Quaternion-Neural-Networks ]; then
    git clone --depth 1 https://github.com/TParcollet/Quaternion-Neural-Networks.git third_party/Quaternion-Neural-Networks
fi

echo "===> Verification..."
python -c "import torch; print(f\"torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}\")"

echo ""
echo "===> Setup complete. Activate with: source .venv/bin/activate"
