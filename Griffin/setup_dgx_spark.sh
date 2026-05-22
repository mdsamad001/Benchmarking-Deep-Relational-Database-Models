#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PY="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

echo "============================================="
echo " Griffin — DGX Spark GB10 Setup"
echo "============================================="
echo ""

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "[WARN] Expected aarch64, got ${ARCH}."
    echo "       This script is designed for DGX Spark (ARM Grace CPU)."
    echo "       Proceeding anyway — PyTorch index URL may need adjustment."
fi

if [ -d "$VENV_DIR" ]; then
    echo "[INFO] Virtual environment already exists at ${VENV_DIR}"
    echo "       To recreate: rm -rf ${VENV_DIR} && ./setup_dgx_spark.sh"
else
    echo "[1/5] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"

echo "[2/5] Upgrading pip/setuptools/wheel..."
"$PIP" install --upgrade pip setuptools wheel 2>&1 | tail -1

echo "[3/5] Installing PyTorch (cu130 index for aarch64)..."
if [ "$ARCH" = "aarch64" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu130"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
fi

"$PIP" install torch torchvision --index-url "$TORCH_INDEX" 2>&1 | tail -3

echo "[4/5] Installing remaining dependencies..."
"$PIP" install -r "${SCRIPT_DIR}/requirements_dgx_spark.txt" 2>&1 | tail -3

echo "[5/5] Installing accelerate config (single-GPU)..."
ACCEL_DIR="${HOME}/.cache/huggingface/accelerate"
mkdir -p "$ACCEL_DIR"
cp "${SCRIPT_DIR}/hconfig_dgx_spark.yaml" "${ACCEL_DIR}/default_config.yaml"
echo "       Wrote ${ACCEL_DIR}/default_config.yaml"

echo ""
echo "============================================="
echo " Sanity checks"
echo "============================================="

"$PY" - <<'PYEOF'
import sys, torch

print(f"Python:       {sys.version.split()[0]}")
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA avail:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"GPU mem:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    x = torch.randn(100, 64, device="cuda")
    idx = torch.randint(0, 10, (100,), device="cuda")
    out = torch.zeros(10, 64, device="cuda")
    out.scatter_add_(0, idx.unsqueeze(-1).expand_as(x), x)
    out2 = torch.full((10, 64), float('-inf'), device="cuda")
    out2.scatter_reduce_(0, idx.unsqueeze(-1).expand_as(x), x, reduce='amax', include_self=False)
    print(f"Scatter ops:  OK (scatter_add_ + scatter_reduce_ on GPU)")
else:
    print("[ERROR] CUDA not available — check driver / CUDA toolkit")
    sys.exit(1)

from hmodel import GriffinMod
m = GriffinMod(hiddim=512, num_mp=1, use_rev=True, use_gate=True).cuda()
print(f"GriffinMod:   OK ({sum(p.numel() for p in m.parameters()):,} params)")
print()
print("All checks passed.  Ready to train.")
PYEOF

echo ""
echo "============================================="
echo " Setup complete"
echo "============================================="
echo ""
echo "To activate the environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To run a task (example):"
echo "  ./run_task.sh <DATASET_DIR> <TASK_NAME> --hop 2 --fanout 10 30"
echo ""
