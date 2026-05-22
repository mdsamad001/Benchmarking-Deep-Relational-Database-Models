#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $
    echo "Usage: $0 <DATASET_DIR> <TASK_NAME>"
    exit 1
fi

DATASET="$1"
TASK="$2"

echo "============================================="
echo " Hop sweep: ${TASK}"
echo " Dataset:   ${DATASET}"
echo "============================================="

for HOP in 0 1 2; do
    echo ""
    echo "========== HOP = ${HOP} =========="
    echo ""

    if [ "$HOP" -eq 0 ]; then
        FANOUT_ARGS=""
    elif [ "$HOP" -eq 1 ]; then
        FANOUT_ARGS="--fanout 10"
    else
        FANOUT_ARGS="--fanout 10 10"
    fi

    "${SCRIPT_DIR}/run_task.sh" "$DATASET" "$TASK" \
        --hop "$HOP" \
        $FANOUT_ARGS \
        --maxepoch 100 \
        --seed 0
done

echo ""
echo "============================================="
echo " Hop sweep complete for ${TASK}"
echo "============================================="
