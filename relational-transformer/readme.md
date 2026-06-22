
# Relational Transformer

## Repository Structure

```
relational-transformer/
├── rt/
│   ├── data.py              
│   ├── model.py             
│   ├── hop_control.py        
│   ├── main.py              
│   ├── tasks.py             
│   ├── embed.py             
│   └── griffin_float_embedder.py
├── rustler/
│   └── src/
│       ├── fly.rs            
│       ├── pre.rs           
│       ├── common.rs         
│       └── main.rs 
├── scripts/
│   ├── example_finetune.py         
│   ├── example_pretrain.py
│   ├── example_contd_pretrain.py
│   └── download_relbench.py
├── pretrained_weights/
│   └── float_embedder.pt   
├── pretrain_float.py
├── pyproject.toml
└── environment.yml
```

---

## Setup

### Requirements

- Linux aarch64 (Grace Blackwell) or linux-x86\_64
- CUDA 13.0 driver / CUDA 12.8+ toolkit
- Python 3.12
- Rust 1.82+

### 1. Create environment

```bash
conda env create -f environment.yml
conda activate rt
```

### 2. Install PyTorch

```bash
pip install torch torchvision \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. Install the rt package

```bash
pip install -e .
```

### 4. Build rustler

```bash
cd rustler
pip install -e . --no-build-isolation
cargo build --release
cd ..
```

Verify the Python extension has all required methods:
```bash
python -c "from rustler import Sampler; print([m for m in dir(Sampler) if not m.startswith('__')])"
# Expected: ['batch_py', 'len_py', 'p2f_neighbors_py', 'shuffle_py']
```

---

## Data Preparation

###  Download preprocessed data from HuggingFace

Skips the preprocessing step entirely. The Stanford team hosts fully preprocessed graph data:

```bash
mkdir -p ~/scratch
ln -s ~/.cache/relbench ~/scratch/relbench

hf download hvag976/relational-transformer \
  --repo-type dataset \
  --local-dir ~/scratch/pre \
  --local-dir-use-symlinks False
```

Then compute text embeddings (still required, not included in the HuggingFace download):

```bash
for db in rel-f1 rel-amazon rel-hm rel-stack rel-trial rel-event rel-avito; do
  python -m rt.embed $db
done
```
---

## Running Experiments

### Fine-tuning

```bash
python scripts/example_finetune.py \
  --dataset rel-f1 \
  --task driver-top3 \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4 \
  --max_hop 2 \
  --seq_len 512 \
  --seed 0 \
  --results_root results/
```

## Hop Control

| Parameter | Description |
|---|---|
| `max_hop -1` | No hop control (full RT default behavior) |
| `max_hop 0` | Seed node only — no relational neighbors |
| `max_hop 1` | 1-hop neighbors |
| `max_hop 2` | 2-hop neighbors |

---

## Metrics

| Task Type | Training Loss | Validation Metric | Test Metric |
|---|---|---|---|
| Classification | cross-entropy | AUROC | AUROC |
| Regression | MSE | MSE | MSE |

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{ranjan2026relationaltransformer,
  title={{Relational Transformer:} Toward Zero-Shot Foundation Models for Relational Data},
  author={Ranjan, Rishabh and Hudovernik, Valter and Znidar, Mark and Kanatsoulis, Charilaos
          and Upendra, Roshan and Mohammadi, Mahmoud and Meyer, Joe and Palczewski, Tom
          and Guestrin, Carlos and Leskovec, Jure},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

---
