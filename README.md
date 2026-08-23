# KnowFeat

**Distilling Domain Expert Knowledge into Verified Features via LLM Agents for Digital Yuan Anti-Money Laundering**

ICDM 2026 Applied Track

## Overview

KnowFeat is an LLM Agent framework that automatically distills unstructured domain expert knowledge into statistically verified ML features for anti-money laundering detection.

### Two-Stage Knowledge Injection

- **Stage 1**: Crypto currency general knowledge (BABD-13, 148 baseline features) + 5 crypto laundering cases
- **Stage 2**: Digital Yuan specific knowledge (bank expert insights + 309 judicial documents + 38 risk indicators)

### Core Pipeline

```
Unstructured Knowledge → Key-clue Extractor → Structured Risk Factors
                                                      ↓
                                              Feature Agent (ReAct)
                                              ├── explore_data
                                              ├── generate_feature (+ Provenance Card)
                                              └── execute_sandbox
                                                      ↓
                                              Three-level Verification
                                              ├── L1: Sandbox Execution
                                              ├── L2: Noise-calibrated IV/KS
                                              └── L3: Cross-validated Delta-Recall
                                                      ↓
                                              Accepted Feature Set
```

## Project Structure

```
docs/
├── KnowFeat_design.md        # System design reference (working doc)
outputs/
├── system_architecture_v4.html # Technical architecture (presentation)
ModelSet/                       # 11 AML models + unified runner
├── runner.py                  # Central dispatcher
inputs/
├── domain_knowledge/          # Expert interviews, risk indicators, rules
├── SimECNY/                   # Digital Yuan simulation data
├── baseline/                  # Baseline feature engineering code
data/                          # Digital Yuan transaction data
裁判文书docx/                    # 309 judicial documents
```

## Quick Start

### Install

```bash
pip install -r requirements.txt
# PyTorch: install separately per your CUDA version
# https://pytorch.org/get-started/locally/
```

### Deploy to GPU Server

```bash
# Default (uses env vars DEPLOY_HOST/DEPLOY_PORT or built-in defaults)
./deploy.sh

# Custom server
./deploy.sh --host <host> --port <port>

# Preview what would be synced
./deploy.sh --dry-run

# Only sync code (skip pip install)
./deploy.sh --sync-only
```

### Run Experiments

```bash
# Stage 2 (Digital Yuan, data available)
nohup python3 -m knowfeat.main > logs/stage2.log 2>&1 &

# Monitor progress
tail -f logs/stage2.log
```

Public-baseline generation uses the frozen split cached under
`outputs/splits/<dataset>_split.json`. AutoFeat, CAAFE, FeatLLM, OpenFE, and
KnowFeat fit or select features on the development partition only; generated
transformations are then applied in original row order for holdout scoring.
Post-hoc pruning is available for dedicated experiments through
`cfg.search.enable_posthoc_selection` or
`scripts/run_knowfeat_public.py --enable-posthoc-selection` and is disabled by
default.

Run final frozen-test evaluation and reproduce the paper summaries with:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate_holdout.py \
  --dataset credit_g --baseline
PYTHONPATH=. .venv/bin/python scripts/analyze_public_results.py
PYTHONPATH=. .venv/bin/python scripts/evaluate_reduced_baseline.py
PYTHONPATH=. .venv/bin/python scripts/gen_fig2_avg_rank.py
```

The exact manuscript inputs are versioned in `outputs/eval_holdout/`.
`paper/results/public_holdout_analysis.json` contains the average ranks and
Holm-corrected paired Wilcoxon tests. The reduced-baseline analysis explicitly
excludes identifiers and labels before model fitting.

Compile the paper from `paper/` with a standard TeX Live installation:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

### Run Tests

```bash
PYTHONPATH=. .venv/bin/pytest -q
```

## Documentation

- **System Design**: `docs/KnowFeat_design.md`
- **Architecture (visual)**: `outputs/system_architecture_v4.html`
- **ModelSet Guide**: `ModelSet/README.md`
