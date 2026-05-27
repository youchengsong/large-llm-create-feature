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

## Documentation

- **System Design**: `docs/KnowFeat_design.md`
- **Architecture (visual)**: `outputs/system_architecture_v4.html`
- **ModelSet Guide**: `ModelSet/README.md`
