# Benchmarking Deep Relational Database Models

A comparative benchmarking suite for deep learning models that operate directly on **relational databases**. This repository evaluates three state-of-the-art frameworks, **Griffin**, **DBFormer**, and **Relational Transformer** on the [RelBench](https://relbench.stanford.edu/) benchmark, a collection of real-world relational database tasks.

---

## Repository Structure

```
Benchmarking-Deep-Relational-Database-Models/
├── Griffin/                    
├── dbformer/deep-db-learning/ 
└── relational-transformer/   
```

Each folder is self-contained with its own environment, setup script, and detailed README.

---

## Benchmark: RelBench

All three models are evaluated on [RelBench](https://relbench.stanford.edu/) tasks, which include real-world databases such as:

| Dataset | Domain |
|---|---|
| `rel-f1` | Formula 1 racing |
| `rel-amazon` | E-commerce product reviews |
| `rel-stack` | StackExchange Q&A |
| `rel-trial` | Clinical trials |
| `rel-avito` | Russian classifieds |

Tasks span both **classification** (evaluated by AUROC) and **regression** (evaluated by MSE).

---

## Hardware

All models are tested and optimised for the **NVIDIA DGX Spark GB10** (ARM Grace + Blackwell, 128 GB unified memory, CUDA 13.0). Setup scripts auto-detect architecture and fall back to x86/cu124 wheels where needed.

| Requirement | Version |
|---|---|
| CUDA | 12.8 |
| Python | 3.13 |
| PyTorch | 2.12.0+cu130 |
| Rust | 1.82 |

---

## Quick Start

Clone the repository, then follow the README inside the model folder you want to run:

```bash
git clone https://github.com/FuadBinAkhter/Benchmarking-Deep-Relational-Database-Models.git
cd Benchmarking-Deep-Relational-Database-Models

# Griffin
cd Griffin && ./setup_dgx_spark.sh

# DBFormer
cd dbformer/deep-db-learning && bash setup.sh

# Relational Transformer
cd relational-transformer && conda env create -f environment.yml
```

---
