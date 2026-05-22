# Griffin — Relational Graph Learning on Heterogeneous Databases

Griffin is a relational GNN framework that operates on heterogeneous graph
databases in [RelBench](https://relbench.stanford.edu/) format.  It builds
multi-hop subgraphs around each target entity, encodes node/edge features
with a learned float-embedding layer, and trains a relational message-passing
network for node-level regression and classification tasks.

This repository contains the full training pipeline, pre-trained feature
encoders (`floatenc-512.pt` / `floatdec-512.pt`), and all setup scripts for
running on an **NVIDIA DGX Spark GB10** (ARM Grace + Blackwell).

---

## Hardware Requirements

| Component | Spec |
|-----------|------|
| Platform | NVIDIA DGX Spark GB10 |
| CPU | 20-core ARM Grace (aarch64) |
| GPU | NVIDIA GB10 Blackwell, sm_121 |
| Memory | 128 GB unified LPDDR5X (CPU+GPU via NVLink-C2C) |
| OS | DGX OS — Ubuntu 24.04 LTS |
| CUDA | 13.0+ |
| Python | 3.12 or 3.13 |

> **x86 note:** The setup script detects the architecture and falls back to a
> cu124 PyTorch wheel automatically if you are on an x86 machine.

---

## Repository Structure

```
Griffin/
│
│── Core training
│   ├── hmaintask_combine.py          Main training entry point (multi-seed, all tasks)
│   ├── hmaintask_downsample_absolute_eval_sample.py
│   │                                 Alt trainer — adds downsampling for large datasets
│   ├── hmaintask_completion.py       Self-supervised pretraining (feature-completion)
│   └── run_one_db.py                 Convenience wrapper: filter + launch single DB
│
│── Model and data
│   ├── hmodel.py                     GriffinMod: relational MPNN + cross-attention
│   ├── hdataset.py                   Graph class: loads HF datasets, multi-hop subgraph
│   ├── hloaderwrapper.py             Batch construction, fanout sampling, DataLoader wraps
│   ├── hFloatEmb.py                  Float encoder (floatenc-512.pt) and decoder
│   ├── metric.py                     Evaluation metrics (MSE, RMSE, MAE, AUC, HR@1)
│   ├── floatenc-512.pt               Pre-trained float feature encoder (hiddim=512)
│   └── floatdec-512.pt               Pre-trained float feature decoder (hiddim=512)
│
│── Data pipeline  (only needed to rebuild a dataset from scratch)
│   ├── dataconverter.py              Node features → HuggingFace datasets
│   ├── dataconverteredge.py          Adjacency lists → HF datasets
│   ├── dataconvertertask.py          Task labels + splits → HF datasets
│   ├── dataconverterpost.py          Post-processing (edge name embeddings)
│   ├── combine_dataset.py            Merge two Griffin-format datasets
│   ├── make_textemb.py               Text → sentence embedding
│   ├── make_textemb_from_ids.py      Variant that takes pre-computed IDs
│   └── build_relbench_rel_amazon_griffin.sh   End-to-end data build example
│
│── Setup and runners
│   ├── setup_dgx_spark.sh            One-time environment setup (DGX Spark)
│   ├── run_task.sh                   Run a single training job
│   ├── run_hop_sweep.sh              Run hop=0/1/2 comparison sweep
│   ├── hconfig_dgx_spark.yaml        Accelerate config — single GPU
│   ├── hconfig_8gpu.yaml             Accelerate config — 8-GPU multi-process
│   └── requirements_dgx_spark.txt    Pip requirements (torch installed separately)
│
└── task_names.yaml                   Grouped list of all benchmark tasks
```

---

## Setup on DGX Spark GB10

### Step 1 — Clone and enter the repository

```bash
git clone <your-repo-url> Griffin
cd Griffin
```

### Step 2 — One-time environment setup

```bash
chmod +x setup_dgx_spark.sh run_task.sh run_hop_sweep.sh
./setup_dgx_spark.sh
```

What the script does:

1. Creates a Python virtual environment at `.venv/`
2. Installs PyTorch from the `cu130` aarch64 wheel index (required for sm_121 Blackwell)
3. Installs all remaining dependencies from `requirements_dgx_spark.txt`
4. Writes the single-GPU accelerate config to `~/.cache/huggingface/accelerate/default_config.yaml`
5. Runs a sanity check — expected output:

```
=============================================
 Sanity checks
=============================================
Python:       3.13.x
PyTorch:      2.12.0+cu130
CUDA avail:   True
GPU:          NVIDIA GB10
GPU mem:      128.0 GB
Scatter ops:  OK (scatter_add_ + scatter_reduce_ on GPU)
GriffinMod:   OK (25,952,769 params)

All checks passed.  Ready to train.
```

> **PyTorch CUDA capability warning:** You may see a message like
> `UserWarning: CUDA capability 12.1 is not yet tested`.
> This is a known cosmetic warning (PyTorch PR #164590) and is safe to ignore.
> Training and inference work correctly on GB10.

### Step 3 — Activate the environment

```bash
source .venv/bin/activate
```

Add this line to your `~/.bashrc` or run it at the start of every session.

---

## Preparing a Dataset

Griffin expects a dataset directory with the following structure — produced
by the data pipeline scripts:

```
datasets/filtered/<db_name>/
├── metanode.yaml          node-type metadata (feature names, is_target flag)
├── metatask.yaml          task definitions (metric, label column, splits)
├── metaedge.yaml          edge-type metadata (src/dst node types)
├── node/
│   └── <node_type>/       one HuggingFace dataset folder per node type
│       ├── dataset_info.json
│       └── data-*.arrow
├── edge/
│   └── <edge_name>/       one HF dataset per edge type
└── task/
    └── <task_name>/       one HF dataset per task
```

### Building a new dataset from a RelBench database

Use `build_relbench_rel_amazon_griffin.sh` as a template.  The pipeline has
six stages — see the script for the exact commands:

```
Stage 1: git checkout processing_data branch
Stage 2: pip install relbench tab2graph
Stage 3: convert_relbench_to_dbinfer.py  →  RDB format
Stage 4a: tab2graph preprocess (raw)
Stage 4b: tab2graph preprocess (Griffin features)
Stage 4c: tab2graph construct-graph
Stage 5: git checkout main-public
Stage 6: dataconverter.py / dataconverteredge.py / dataconvertertask.py
```

> Data pipeline dependencies (`pqdm`, `pandas`, `sentence-transformers`, `tqdm`)
> are commented out in `requirements_dgx_spark.txt` and only needed for this
> stage. Uncomment them if rebuilding a dataset.

---

## Running Training

### Single task

```bash
./run_task.sh <DATASET_DIR> <TASK_NAME> [OPTIONS]
```

**Examples:**

```bash
# F1 driver position — hop=2, fanout 10 at hop-1 and 30 at hop-2
./run_task.sh datasets/filtered/rel-f1_only rel-f1-driver-position \
    --hop 2 --fanout 10 30

# Amazon item-churn — hop=2, uniform fanout 10 at each hop
./run_task.sh datasets/filtered/rel-amazon_only rel-amazon-item-churn \
    --hop 2 --fanout 10 10 --maxepoch 100

# StackExchange — multiple seeds
./run_task.sh datasets/filtered/stackexchange stackexchange-churn \
    --hop 2 --fanout 10 10 --seed 0 1 2

# Zero-hop baseline (no graph neighbors, feature-only)
./run_task.sh datasets/filtered/rel-f1_only rel-f1-driver-position \
    --hop 0
```

### Hop sweep (hop = 0, 1, 2 run sequentially)

```bash
./run_hop_sweep.sh <DATASET_DIR> <TASK_NAME>

# Example
./run_hop_sweep.sh datasets/filtered/rel-f1_only rel-f1-driver-position
```

### Calling the training script directly

```bash
accelerate launch \
    --config_file hconfig_dgx_spark.yaml \
    hmaintask_combine.py \
    <DATASET_DIR> \
    <LOG_DIR> \
    <RUN_NAME> \
    --tasks <TASK_NAME> \
    --savepath <CHECKPOINT_DIR> \
    [OPTIONS]
```

---

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `dataset` | *(required)* | Path to Griffin-format dataset directory |
| `logdir` | *(required)* | Directory for loss CSVs and plots |
| `logname` | *(required)* | Run name prefix for output files |
| `--tasks` | `ALLTASK` | Space-separated task names to train on. Use `ALLTASK` to train on all tasks in the dataset simultaneously |
| `--hop` | `2` | Number of graph hops for subgraph extraction |
| `--fanout` | `[10] * hop` | Per-hop neighbor fanout. Pass one value per hop, e.g. `--fanout 10 30` for hop=2 |
| `--fewshotfanout` | `0` | Additional few-shot neighbors sampled for the target node |
| `--hiddim` | `256` | Hidden dimension. **Must be `512`** to use the shipped `floatenc-512.pt` |
| `--num_mp` | `4` | Number of relational message-passing layers |
| `--use_rev` | `True` | Add reverse (destination→source) message-passing edges |
| `--use_gate` | `True` | Gated combination of forward/reverse messages |
| `--batchsize` | `512` | Training batch size |
| `--eval_batchsize` | same as batchsize | Batch size used during evaluation |
| `--lr` | `3e-4` | AdamW learning rate |
| `--wd` | `4e-4` | AdamW weight decay |
| `--maxepoch` | `10` | Number of training epochs |
| `--eval_per_epoch` | `3` | Log per-task validation metrics every N epochs |
| `--seed` | `[0]` | One or more random seeds: `--seed 0 1 2` runs three independent trials |
| `--savepath` | `None` | Directory to save best-checkpoint per seed |
| `--loadpath` | `None` | Path to an existing checkpoint to warm-start from |

> **`--hiddim 512` is required** when using the pre-trained encoders shipped
> with this repo.  `run_task.sh` always passes `--hiddim 512`.

---

## Outputs

Training produces the following files:

```
logs/<run_name>/
└── losses_<db>_<task>_seed<N>.csv      train_loss, valid_loss, valid_metric per epoch
    losses_<db>_<task>_seed<N>.png      loss curve plot

checkpoints/<run_name>/
└── seed_<N>/
    └── best_checkpoint/
        ├── model.safetensors           best-validation-epoch weights
        └── ...

results/<db>/<task>/<hop>/
└── seed_<N>/
    ├── results.json                    full config + train losses + best/test metrics
    └── losses_<...>.csv                same as logs/
```

`results.json` schema:

```json
{
  "task": "rel-f1-driver-position",
  "config": { "hop": 2, "fanout": [10, 30], "hiddim": 512, ... },
  "best_val_metric": 0.387,
  "best_val_epoch": 14,
  "test_metric": 0.412,
  "epoch_train_losses": [1.49, 0.99, ...],
  "epoch_val_losses":   [0.51, 0.50, ...],
  "epoch_val_metrics":  [0.44, 0.43, ...]
}
```

---

## Metrics

Each task has one metric defined in `metatask.yaml`.

| Metric name | Type | Better |
|-------------|------|--------|
| `mse` | Mean squared error | Lower |
| `rmse` | Root mean squared error | Lower |
| `mae` | Mean absolute error | Lower |
| `auc` / `retrieval_auroc` | Multi-class AUROC | Higher |
| `hr@1` | Hit-rate at rank 1 | Higher |
| `retrieval_logloss` | Cross-entropy | Lower |

All metrics are reported as raw positive values in their natural units.
The training script uses `metric_higher_is_better()` internally to select the
best checkpoint in the correct direction.

---

## Available Tasks

Tasks are grouped in `task_names.yaml`:

**commerce-1:** `diginetica-downsample-ctr`, `rel-hm-item-sales`, `rel-hm-user-churn`, `retailrocket-cvr`, `seznam-charge`, `seznam-prepay`

**commerce-2:** `amazon-churn`, `amazon-rating`, `outbrain-small-ctr`, `rel-avito-ad-ctr`, `rel-avito-user-clicks`, `rel-avito-user-visits`

**others-1:** `rel-f1-driver-dnf`, `rel-f1-driver-position`, `rel-f1-driver-top3`, `stackexchange-churn`, `stackexchange-upvote`, `virus-wnv-pred`

**others-2:** `airbnb-destination`, `rel-trial-site-success`, `rel-trial-study-adverse`, `rel-trial-study-outcome`, `talkingdata-demo-pred`, `telstra-severity`

**single-table:** 50+ tabular regression benchmarks (anime ratings, car prices, wine scores, etc.) — see `task_names.yaml` for the full list.

---

## Self-Supervised Pretraining

`hmaintask_completion.py` implements a feature-completion pretraining objective:
one feature column is masked per sample and predicted from the node's graph
neighborhood.  Run it before fine-tuning to warm-start the encoder:

```bash
accelerate launch \
    --config_file hconfig_dgx_spark.yaml \
    hmaintask_completion.py \
    <DATASET_DIR> \
    <LOG_DIR> \
    <RUN_NAME> \
    --hop 2 --fanout 10 10 --hiddim 512
```

Then pass the resulting checkpoint to the main training script via
`--loadpath <checkpoint_dir>`.

---

## Known Platform Notes

### `torch_geometric` removed

PyG's C++ extensions (`torch-scatter`, `torch-sparse`, `torch-cluster`,
`torch-spline-conv`) have no pre-built `aarch64` wheels and fail to compile on
DGX Spark.  Griffin only used `MeanAggregation` and `MaxAggregation` from PyG.
These are reimplemented in `hmodel.py` using native `torch.scatter_add_` and
`torch.scatter_reduce_` — available in all PyTorch ≥ 2.0 releases.

### `torch.unique` on 2D tensors crashes on Blackwell (sm_121)

Calling `torch.unique` on a 2D tensor routes through a merge-sort CUDA kernel
that produces `cudaErrorIllegalAddress` on Blackwell with PyTorch 2.12 + CUDA
13.  The `RMPNN.forward` method in `hmodel.py` encodes `(src_node, edge_type)`
as a single `int64` key so that `torch.unique` operates only on a 1D tensor,
avoiding the broken kernel path entirely.

### Tensorboard is optional

If `tensorboard` is not installed, the training script logs to CSV/PNG only
and suppresses the accelerate `UserWarning` automatically.  To enable
TensorBoard logging: `pip install tensorboard` then re-run training.

---

## Troubleshooting

**`cudaErrorIllegalAddress` in RMPNN.forward**
Your `hmodel.py` is older than the fix.  Replace it with the patched version
that uses the 1D-key approach for `torch.unique`.

**Metrics are negative (e.g. `-0.44` for MSE)**
Your `metric.py` is the pre-fix version that negated regression scores.
Replace `metric.py` and `hmaintask_combine.py` with the patched versions.

**`AttributeError: 'CudaDeviceProperties' object has no attribute 'total_mem'`**
Replace `total_mem` with `total_memory` in `setup_dgx_spark.sh` line 102.

**`ImportError: cannot import name 'METRIC_LOWER_IS_BETTER' from 'metric'`**
Replace `hmaintask_combine.py` with the patched version — the import was
updated to `from metric import compute_metric, metric_higher_is_better`.

**`torch-scatter` / `torch-sparse` build failure**
These are not needed.  `torch_geometric` has been removed from this codebase.
Do not install PyG packages — run `pip uninstall torch-scatter torch-sparse
torch-cluster torch-spline-conv torch-geometric` if they were installed.

**`UserWarning: log_with=tensorboard was passed but no supported trackers installed`**
Install tensorboard (`pip install tensorboard`) or ignore it — training proceeds
correctly without it.

---

## File-by-File Change Log

See `CHANGES.md` for a precise description of every bug that was fixed,
which line was changed, and why.