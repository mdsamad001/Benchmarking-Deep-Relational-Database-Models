#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

USE_VENV=false
if [[ "${1:-}" == "--venv" ]]; then
    USE_VENV=true
fi

if $USE_VENV; then
    echo "[1/4] Creating virtualenv in .venv/ ..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "       Activated: $(which python3)"
else
    echo "[1/4] Using system Python: $(which python3)"
fi

echo "[2/4] Upgrading pip and installing core dependencies ..."
pip install --upgrade pip setuptools wheel

CUDA_TAG="cpu"
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed -E 's/.*release ([0-9]+\.[0-9]+).*/\1/')
    CUDA_TAG="cu$(echo $CUDA_VER | tr -d '.')"
    echo "       Detected CUDA $CUDA_VER -> $CUDA_TAG"
else
    echo "       No nvcc found — installing CPU-only PyTorch"
fi

echo "[3/4] Installing PyTorch + PyG ..."
if [[ "$CUDA_TAG" == "cpu" ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
echo "       PyG wheel index: $PYG_URL"
pip install torch_geometric torch_scatter torch_sparse -f "$PYG_URL" 2>/dev/null \
    || pip install torch_geometric torch_scatter

echo "[4/4] Installing remaining dependencies ..."
pip install -r requirements.txt

echo ""
echo "================================================================"
echo " Verification"
echo "================================================================"
python3 -c "
import torch
print(f'  torch:            {torch.__version__}')
print(f'  CUDA available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device:      {torch.cuda.get_device_name(0)}')

import torch_geometric
print(f'  torch_geometric:  {torch_geometric.__version__}')

import accelerate
print(f'  accelerate:       {accelerate.__version__}')

import datasets
print(f'  datasets:         {datasets.__version__}')

import torchmetrics
print(f'  torchmetrics:     {torchmetrics.__version__}')
"
echo ""
echo "[DONE] Environment ready."

if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
    echo ""
    echo "NOTE: No accelerate config found.  Creating a single-GPU default."
    echo "      Run 'accelerate config' to customise for multi-GPU."
    mkdir -p ~/.cache/huggingface/accelerate
    cat > ~/.cache/huggingface/accelerate/default_config.yaml <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 1
use_cpu: false
EOF
fi
