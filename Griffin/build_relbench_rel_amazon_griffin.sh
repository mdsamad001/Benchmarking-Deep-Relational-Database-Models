#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/Griffin"
DB="rel-amazon"

OUT_BASE="$REPO_DIR/datasets/relbench_build/$DB"
RDB_DIR="$OUT_BASE/rdb"
RAW_DIR="$OUT_BASE/raw_for_griffin"
SINGLE_DIR="$OUT_BASE/single_griffin"
GRAPH_DIR="$OUT_BASE/r2n_griffin"
GRIFFIN_DIR="$REPO_DIR/datasets/${DB}-griffin"

cd "$REPO_DIR"

echo "[1/6] Checkout Griffin processing branch that has the RelBench conversion pipeline"
git fetch --all
git checkout processing_data

echo "[2/6] Install python deps in container (minimal set)"
pip install -U pip
pip install relbench tab2graph huggingface_hub datasets pyarrow pandas numpy pyyaml tqdm

echo "[3/6] Convert RelBench -> 4DBInfer(RDB) format"
mkdir -p "$RDB_DIR"
python convert_relbench_to_dbinfer.py --dataset "$DB" --out "$RDB_DIR"

echo "[4/6] Build graph inputs with tab2graph (3 steps)"
mkdir -p "$RAW_DIR" "$SINGLE_DIR" "$GRAPH_DIR"

python -m tab2graph.main preprocess "$RDB_DIR" transform "$RAW_DIR" \
  -c configs/transform/raw_for_griffin.yaml

python -m tab2graph.main preprocess "$RAW_DIR" transform "$SINGLE_DIR" \
  -c configs/transform/generate_griffin_feature_separate_num.yaml

python -m tab2graph.main construct-graph "$SINGLE_DIR" r2n-griffin "$GRAPH_DIR"

echo "[5/6] Switch back to Griffin main branch to run dataconverters"
git checkout main-public

echo "[6/6] Convert graph -> Griffin dataset folder (node/edge/task + meta yamls)"
rm -rf "$GRIFFIN_DIR"
mkdir -p "$GRIFFIN_DIR"

python dataconverter.py     "$GRAPH_DIR" "$GRIFFIN_DIR"
python dataconverteredge.py "$GRAPH_DIR" "$GRIFFIN_DIR"
python dataconvertertask.py "$GRAPH_DIR" "$GRIFFIN_DIR"
python dataconverterpost.py "$GRAPH_DIR"

echo ""
echo "[DONE] Griffin dataset created at:"
echo "  $GRIFFIN_DIR"
echo ""
echo "Sanity check tasks:"
python - <<'PY'
import yaml, os
p = "datasets/rel-amazon-griffin/metatask.yaml"
m = yaml.safe_load(open(p))
print("Tasks:", list(m.keys())[:20])
print("Has item-churn?", "item-churn" in m)
print("Has item-ltv?", "item-ltv" in m)
PY