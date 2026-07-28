# VRPTW + QUBO-CIM Framework
## Hybrid Quantum-Classical Optimization for Vehicle Routing with Time Windows

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

This repository contains the official implementation of the paper:

> **"A Hybrid Quantum-Classical Framework for the Vehicle Routing Problem with Time Windows: QUBO Formulation and Coherent Ising Machine Optimization"**
>
> Submitted to *Computers & Operations Research*, 2026.

---

## 📋 Overview

We propose a hybrid quantum-classical framework for solving the Vehicle Routing Problem with Time Windows (VRPTW) by integrating:

1. **Multi-Dimensional Scaling (MDS)** for customer clustering
2. **Quadratic Unconstrained Binary Optimization (QUBO)** formulation for TSPTW subproblems
3. **Coherent Ising Machine (CIM)** optimization via Kaiwu SDK
4. **Enhanced Simulated Annealing (ESA)** for global coordination

---

## 🗂️ Repository Structure

```
.
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── data/
│   └── instance_generator.py       # VRPTW instance generation
├── src/
│   ├── mds_clustering.py           # MDS-based customer clustering
│   ├── qubo_construction.py        # QUBO formulation for TSPTW
│   ├── cim_solver.py               # CIM optimizer (Kaiwu SDK + simulation mode)
│   ├── esa_solver.py               # Enhanced Simulated Annealing
│   ├── classical_baselines.py      # SA, 2-opt, greedy baselines
│   └── solomon_benchmark.py        # Solomon 100-customer evaluation
├── results/
│   └── generate_all_tables.py      # Reproduce all paper tables & figures
└── manuscript/
    └── supplementary_materials.pdf # Additional experimental details
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Reproduce All Paper Results

```bash
# Generate all 10 tables and 2 figures from the paper (~2 minutes)
python results/generate_all_tables.py
```

This produces:
- `results/table_2_15customer_tsp.txt` through `results/table_10_statistical.txt`
- `results/fig_esa_convergence.png`
- `results/fig_mds_clustering.png`

### 3. Run Individual Experiments

```bash
# 15-customer TSPTW validation
python src/cim_solver.py --instance-size 15

# 50-customer VRPTW full comparison (CIM vs SA vs LKH-3 vs OR-Tools)
python src/esa_solver.py --instance-size 50 --compare-all

# Solomon 100-customer benchmark (all 56 instances)
python src/solomon_benchmark.py --classes C1,C2,R1,R2,RC1,RC2
```

### 4. Using Real CIM Hardware (Kaiwu SDK)

```bash
# Install Kaiwu SDK (requires Boson Quantum account)
pip install kaiwu-sdk

# Set your API credentials
export KAIWU_API_KEY="your_api_key"
export KAIWU_API_SECRET="your_api_secret"

# Run with real CIM hardware
python src/cim_solver.py --backend cim --instance-size 50
```

> **Note:** Real CIM hardware access requires a Boson Quantum Technology account.
> Register at: https://kaiwu.qboson.com/
>
> If Kaiwu SDK is unavailable, the code falls back to a simulated CIM mode
> (NumPy-based coherent-state-inspired annealing) that validates algorithm correctness.

---

## 📊 Key Results (Quick Reference)

| Metric | Result |
|--------|--------|
| 15-customer: CIM vs DP | **8,814× speedup**, exact optimum |
| 50-customer: CIM vs SA | **66.2% objective improvement** |
| 50-customer: gap to LKH-3 | **2.4%** (8 vehicles) |
| Solomon-100: C-type avg gap | **3.1%** |
| Solomon-100: overall avg gap | **8.3%** |
| CIM vs Gurobi (m=25 QUBO) | **156× speedup** |
| 50-customer total runtime | **28 seconds** |

---

## 🔧 Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
kaiwu-sdk>=1.2.0  # Optional: for real CIM hardware
```

---

## 📝 Reproducibility Notes

- **Hardware:** All experiments were run on Intel Core i9-13900K (24 cores, 3.0 GHz, 64 GB RAM)
  with Kaiwu SDK v1.2.0 on the Boson Quantum 550-qubit Wuyue cloud platform.
- **Random seeds:** All experiments use fixed random seeds (`seed=42` for primary runs)
  to ensure reproducibility. Statistical results (Table 10) use seeds 1-30.
- **Classical baselines:** LKH-3 v3.0.8 and OR-Tools v9.8 were compiled with default flags.
- **Simulated CIM:** When Kaiwu SDK hardware is unavailable, a NumPy-based coherent-state-
  inspired annealing simulator is used. This validates algorithm correctness but does not
  reproduce the quantum speedup of real CIM hardware. Real hardware results are reported
  in the paper (Tables 2, 3, 7).

---

## 📄 Citation

```bibtex
@article{author2026hybrid,
  title={A Hybrid Quantum-Classical Framework for the Vehicle Routing Problem
         with Time Windows: QUBO Formulation and Coherent Ising Machine Optimization},
  author={First Author and Second Author},
  journal={Computers \& Operations Research},
  year={2026},
  note={Under review}
}
```

---

## 📧 Contact

- Corresponding author: email@xxx.edu.cn
- Code issues: Open a GitHub Issue in this repository

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

The data and code in this repository are provided to ensure full reproducibility
of the experimental results reported in the associated paper, in compliance with
the *Computers & Operations Research* reproducibility policy.
