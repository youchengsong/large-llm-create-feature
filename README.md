# KnowFeat

**Knowledge-Guided Feature Engineering via LLM Agents**

## Overview

KnowFeat is an LLM Agent framework that injects structured domain knowledge into automated feature engineering and subjects every candidate to a three-stage verification pipeline. Each accepted feature carries a provenance card tracing its design from knowledge assets through reasoning chains to verification outcomes.

### Core Pipeline

```
Domain Knowledge Base                    LLM Agent (ReAct)
├── Schema Metadata                      ├── explore_data
├── Risk Indicators (38)                 ├── generate_feature (+ Provenance Card)
├── Detection Rules (15)                 └── execute_sandbox
├── Expert Opinions (12)                         ↓
└── Court Document Evidence (309)        Three-Stage Verification
                                         ├── L1: Code Execution Sandbox
                                         ├── L2: Noise-calibrated IV/KS
                                         └── L3: XGBoost Δ-metric Evaluation
                                                 ↓
                                         Accepted Features + Provenance Cards
```

### Key Results

- **Avg. Rank 1st** (2.3/7 methods) across 12 public benchmarks under leakage-free evaluation
- **+11.6 pp AUC** peak gain on telecom churn dataset
- **Provenance cards** cut per-feature compliance review from 2-3 hours to 15-30 minutes
- Validated on SimECNY (200K digital currency transactions) and Elliptic (Bitcoin AML)

## Project Structure

```
knowfeat/                  # Core framework
├── agent/                 # LLM Agent with ReAct tools
├── verification/          # L1/L2/L3 verification pipeline
├── knowledge/             # Knowledge injection module
└── provenance/            # Provenance card generation

scripts/                   # Evaluation and analysis
├── evaluate_holdout.py    # Frozen-test holdout evaluation
├── evaluate_cross_model.py # Cross-model (XGB/LGBM/CatBoost) validation
└── gen_fig2_avg_rank.py   # Generate avg rank figure

data/                      # Public datasets (via OpenML)
inputs/                    # Domain knowledge bases
outputs/                   # Results and feature matrices
├── eval_holdout/          # Leakage-free holdout results
└── splits/                # Frozen train/test splits

paper/                     # LaTeX source
ModelSet/                  # 11 AML baseline models
```

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run KnowFeat

```bash
python3 -m knowfeat.main
```

### Reproduce Paper Results

```bash
# Holdout evaluation (frozen 20% test set)
python scripts/evaluate_holdout.py --dataset credit_g --baseline
python scripts/evaluate_holdout.py --dataset heart \
  --features-csv outputs/heart/knowfeat_features.csv --method KnowFeat

# Cross-model validation
python scripts/evaluate_cross_model.py

# Generate figures
python scripts/gen_fig2_avg_rank.py
```

### Run Tests

```bash
PYTHONPATH=. pytest -q
```

## Datasets

- **12 public benchmarks**: credit_g, diabetes, blood_transfusion, heart, bank_marketing, breast_w, adult, wine_quality, default_credit, churn, kc2, jm1 (via [OpenML](https://www.openml.org/))
- **SimECNY**: Synthetic digital currency AML benchmark (200K transactions, available on [HuggingFace](https://huggingface.co/datasets/anonymous-7219/knowfeat))
- **Elliptic**: Real-world Bitcoin AML dataset (Weber et al., 2019)

## License

MIT
