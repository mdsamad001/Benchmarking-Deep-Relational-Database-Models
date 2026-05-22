#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi

if [ $
    echo "Usage: $0 <DATASET_DIR> <TASK_NAME> [--hop N] [--fanout F1 F2 ...] [...]"
    echo ""
    echo "Positional:"
    echo "  DATASET_DIR   Path to Griffin-format dataset (contains metanode.yaml etc.)"
    echo "  TASK_NAME     Task name from metatask.yaml (e.g. rel-f1-driver-position)"
    echo ""
    echo "Common options (passed to hmaintask_combine.py):"
    echo "  --hop N            Number of graph hops (default: 2)"
    echo "  --fanout F1 [F2]   Per-hop fanout (default: 10 per hop)"
    echo "  --hiddim D         Hidden dimension (default: 512, must match floatenc-D.pt)"
    echo "  --num_mp N         Number of message-passing layers (default: 4)"
    echo "  --batchsize B      Batch size (default: 256)"
    echo "  --lr LR            Learning rate (default: 1e-4)"
    echo "  --wd WD            Weight decay (default: 1e-4)"
    echo "  --maxepoch E       Max epochs (default: 100)"
    echo "  --seed S [S2 ...]  Seed(s) (default: 0)"
    echo "  --fewshotfanout F  Few-shot fanout (default: 3)"
    echo "  --use_rev BOOL     Use reverse message passing (default: True)"
    echo "  --use_gate BOOL    Use gating (default: False)"
    exit 1
fi

DATASET="$1"
TASK="$2"
shift 2

DB_NAME=$(basename "$DATASET")
LOGNAME="scratch_${TASK}"
LOGDIR="${SCRIPT_DIR}/logs/${LOGNAME}"
SAVEPATH="${SCRIPT_DIR}/checkpoints/${LOGNAME}"

mkdir -p "$LOGDIR" "$SAVEPATH"

echo "============================================="
echo " Griffin — Training: ${TASK}"
echo " Dataset:  ${DATASET}"
echo " Logs:     ${LOGDIR}"
echo " Checkpts: ${SAVEPATH}"
echo "============================================="
echo ""

cd "$SCRIPT_DIR"

accelerate launch \
    --config_file "${SCRIPT_DIR}/hconfig_dgx_spark.yaml" \
    hmaintask_combine.py \
    "$DATASET" \
    "$LOGDIR" \
    "$LOGNAME" \
    --tasks "$TASK" \
    --savepath "$SAVEPATH" \
    --hiddim 512 \
    --num_mp 4 \
    --lr 1e-4 \
    --wd 1e-4 \
    --maxepoch 100 \
    --eval_per_epoch 1 \
    --use_rev True \
    --use_gate False \
    --fewshotfanout 3 \
    --seed 0 \
    "$@"

echo ""
echo "Done.  Results in: ${SCRIPT_DIR}/results/${DB_NAME}/${TASK}/"
