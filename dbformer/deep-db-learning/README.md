# DBFormer — Transformers Meet Relational Databases

---

## Environment Setup — DGX Spark GB10

> **Hardware:** NVIDIA Grace Blackwell GB10, aarch64, CUDA 13.0, Python 3.13

### Step 1 — Run the setup script

```bash
cd deep-db-learning
bash setup.sh
```

This script:
1. Detects conda installation
2. Creates a conda environment named `dbformer` (Python 3.13)
3. Installs numpy, pandas, scipy, scikit-learn via conda (aarch64 pre-built binaries)
4. Installs PyTorch 2.12.0 from the CUDA 13.0 wheel index
5. Installs PyG, pytorch_frame, relbench, lightning via pip
6. Installs the `db_transformer` package in editable mode
7. Runs a verification check

> **Important package name note:**  
> The project imports `torch_frame`, which is installed by the PyPI package
> `pytorch_frame` (0.x).  
> Do **not** install `torch-frame` (1.x) — that is an unrelated mmengine package
> that crashes on import with `No module named 'tensorboard'`.

### Step 2 — Activate the environment

```bash
conda activate dbformer
```

Add this to `~/.bashrc` to activate automatically on login:

```bash
echo "conda activate dbformer" >> ~/.bashrc
```

### Step 3 — Verify

```bash
python - <<'EOF'
import torch, torch_geometric, torch_frame, relbench, lightning
print(f"torch          {torch.__version__}")
print(f"torch_geometric {torch_geometric.__version__}")
print(f"torch_frame    {torch_frame.__version__}")
print(f"relbench       {relbench.__version__}")
print(f"lightning      {lightning.__version__}")
print(f"CUDA:          {torch.cuda.is_available()}")
print(f"GPU:           {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
EOF
```

Expected:
```
torch          2.12.0+cu130
torch_geometric 2.7.0
torch_frame    0.2.5
relbench       1.1.0
lightning      2.6.1
CUDA:          True
GPU:           NVIDIA GB10
```

---

## Running a Task

> Always activate the environment first: `conda activate dbformer`

### Quick start

```bash
cd deep-db-learning

### Controlling hop

```bash
# 2-hop neighbourhood
python main_relbench.py rel-f1 driver-top3 --cuda \
    --num-hops 2

# 0-hop
python main_relbench.py rel-f1 driver-top3 --cuda \
    --num-hops 0
```

### Larger model

```bash
python main_relbench.py rel-stack user-engagement --cuda \
    --epochs 150 --lr 5e-4 --batch-size 128 \
    --num-hops 2 \
    --model_config.dim 128
```

---

## Project Structure

```
deep-db-learning/
├── main_relbench.py                                       
├── relbench_dataset.py                                   
├── db_transformer/
│   ├── nn/
│   │   ├── models/
│   │   │   └── blueprint.py  
│   │   ├── conv/
│   │   │   └── mean_add.py   
│   │   ├── embedder/
│   │   │   └── db_embedder.py 
│   │   └── layers/           
│   ├── data/                 
│   └── schema/               
├── setup.sh                  
├── requirements.txt          
├── pyproject.toml            
├── main.py                   
└── experiments/              
```

---

## Citation

```bibtex
@misc{peleška2024transformersmeetrelationaldatabases,
      title={Transformers Meet Relational Databases},
      author={Jakub Peleška and Gustav Šír},
      year={2024},
      eprint={2412.05218},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.05218},
}
```
