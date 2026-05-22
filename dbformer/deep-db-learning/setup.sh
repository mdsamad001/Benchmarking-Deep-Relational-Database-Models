#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="dbformer"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================================="
echo "  DBFormer + RelBench — DGX Spark GB10 setup"
echo "  Project: $PROJECT_DIR"
echo "  Conda env: $ENV_NAME"
echo "=================================================================="

CONDA_BASE=""
for candidate in \
    "$HOME/miniconda3" \
    "$HOME/miniforge3" \
    "$HOME/anaconda3" \
    "/opt/conda" \
    "/opt/miniconda3" \
    "/opt/miniforge3"; do
    if [ -f "$candidate/bin/conda" ]; then
        CONDA_BASE="$candidate"
        break
    fi
done

if [ -z "$CONDA_BASE" ]; then
    echo "ERROR: conda not found. Expected at ~/miniconda3 or similar."
    echo "       Install Miniforge: https://github.com/conda-forge/miniforge"
    exit 1
fi

echo ""
echo "[0] Found conda at: $CONDA_BASE"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda --version

echo ""
echo "[1] Creating conda environment '$ENV_NAME' with Python 3.13..."

if conda env list | grep -q "^$ENV_NAME "; then
    echo "    Environment '$ENV_NAME' already exists — skipping creation."
    echo "    To rebuild from scratch: conda env remove -n $ENV_NAME"
else
    conda create -y -n "$ENV_NAME" python=3.13
    echo "    Created."
fi

conda activate "$ENV_NAME"
echo "    Active Python: $(python --version)"
echo "    Active pip:    $(pip --version)"

echo ""
echo "[2] Installing numpy, pandas, scipy, scikit-learn, matplotlib via conda-forge..."
conda install -y -c conda-forge \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    tqdm \
    attrs

echo ""
echo "[3] Installing PyTorch 2.12.0 (CUDA 13.0)..."
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130

python - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "CUDA not available! Check driver / torch build."
print(f"    torch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}")
PYEOF

echo ""
echo "[4] Installing PyTorch Geometric 2.7.0..."
pip install torch_geometric==2.7.0

echo "    Trying PyG C++ extensions for torch-2.12.0+cu130..."
pip install \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.12.0+cu130.html \
    --quiet 2>/dev/null \
|| {
    echo "    Prebuilt PyG C++ wheels not available — using pure-Python fallback."
    echo "    (This is fully correct, just slightly slower for large graphs.)"
}

echo ""
echo "[5] Installing pytorch_frame 0.2.5..."
pip install pytorch_frame==0.2.5

echo ""
echo "[6] Installing relbench 1.1.0..."
pip install relbench==1.1.0

echo ""
echo "[7] Installing lightning 2.6.1..."
pip install lightning==2.6.1

echo ""
echo "[8] Installing remaining dependencies..."
pip install \
    sqlalchemy>=2.0.0 \
    inflect>=6.0.0 \
    unidecode>=1.3.6 \
    simple-parsing>=0.1.5 \
    transformers>=4.44.0

echo ""
echo "[9] Installing db_transformer package (editable)..."
pip install -e "$PROJECT_DIR"

echo ""
echo "[10] Verifying installation..."
python - <<'PYEOF'
import sys

failures = []

def chk(label, fn, expect=None):
    try:
        result = fn()
        ok = (expect is None) or str(result).startswith(expect)
        marker = "OK  " if ok else "WARN"
        print(f"  [{marker}] {label:<28} {result}")
        if not ok:
            failures.append(label)
    except Exception as e:
        print(f"  [FAIL] {label:<28} {e}")
        failures.append(label)

import torch
chk("torch",              lambda: torch.__version__,        "2.12")
import torch_geometric
chk("torch_geometric",    lambda: torch_geometric.__version__, "2.7")
import torch_frame
chk("torch_frame",        lambda: torch_frame.__version__,  "0.2")
import relbench
chk("relbench",           lambda: relbench.__version__,     "1.")
import lightning
chk("lightning",          lambda: lightning.__version__,    "2.")
import numpy
chk("numpy",              lambda: numpy.__version__)
import pandas
chk("pandas",             lambda: pandas.__version__)
import sqlalchemy
chk("sqlalchemy",         lambda: sqlalchemy.__version__)
import sklearn
chk("scikit-learn",       lambda: sklearn.__version__)
import simple_parsing
chk("simple_parsing",     lambda: "ok")
import inflect
chk("inflect",            lambda: "ok")
from unidecode import unidecode
chk("unidecode",          lambda: "ok")
import db_transformer
chk("db_transformer",     lambda: "ok (editable install)")

cuda = torch.cuda.is_available()
print(f"\n  [{'OK  ' if cuda else 'FAIL'}] CUDA available:           {cuda}")
if cuda:
    print(f"         GPU    : {torch.cuda.get_device_name(0)}")
    print(f"         CUDA   : {torch.version.cuda}")
else:
    failures.append("CUDA")

print()
if not failures:
    print("  All checks passed. Ready to train.")
else:
    print(f"  Issues with: {failures}")
    sys.exit(1)
PYEOF

echo ""
echo "=================================================================="
echo "  Setup complete!"
echo ""
echo "  EVERY TIME you open a new terminal, activate the environment:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Then run from $PROJECT_DIR:"
echo ""
echo "    # Binary classification:"
echo "    python main_relbench.py rel-f1 driver-top3 \\"
echo "        --epochs 100 --lr 1e-3 --batch-size 64 --cuda"
echo ""
echo "    # Regression:"
echo "    python main_relbench.py rel-amazon user-ltv \\"
echo "        --epochs 100 --lr 1e-3 --batch-size 128 --cuda"
echo "=================================================================="