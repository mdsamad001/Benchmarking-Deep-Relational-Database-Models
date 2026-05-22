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
│   ├── hmaintask_combine.py         
│   ├── hmaintask_completion.py      
│   └── run_one_db.py                
│
│── Model and data
│   ├── hmodel.py                     
│   ├── hdataset.py                  
│   ├── hloaderwrapper.py             
│   ├── hFloatEmb.py                 
│   ├── metric.py                    
│   ├── floatenc-512.pt              
│   └── floatdec-512.pt              
│
│── Data pipeline 
│   ├── dataconverter.py              
│   ├── dataconverteredge.py          
│   ├── dataconvertertask.py          
│   ├── dataconverterpost.py         
│   ├── combine_dataset.py            
│   ├── make_textemb.py              
│   ├── make_textemb_from_ids.py      
│   └── build_relbench_rel_amazon_griffin.sh   
│
│── Setup and runners
│   ├── setup_dgx_spark.sh            
│   ├── run_task.sh                  
│   ├── run_hop_sweep.sh              
│   ├── hconfig_dgx_spark.yaml        
│   ├── hconfig_8gpu.yaml             
│   └── requirements_dgx_spark.txt    
│
└── task_names.yaml                   
```

---

## Setup on DGX Spark GB10

### Step 1 — Clone and enter the repository

```bash
git clone https://github.com/mdsamad001/Benchmarking-Deep-Relational-Database-Models.git
cd Benchmarking-Deep-Relational-Database-Models/Griffin
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
# F1 driver position — hop=2
./run_task.sh datasets/filtered/rel-f1_only rel-f1-driver-position \
    --hop 2 

# StackExchange — multiple seeds
./run_task.sh datasets/filtered/stackexchange stackexchange-churn \
    --hop 2 --seed 0 1 2

# Zero-hop
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
  "config": { "hop": 2, ... },
  "best_val_metric": 0.387,
  "best_val_epoch": 14,
  "test_metric": 0.412,
  "epoch_train_losses": [1.49, 0.99, ...],
  "epoch_val_losses":   [0.51, 0.50, ...],
  "epoch_val_metrics":  [0.44, 0.43, ...]
}
```

---
