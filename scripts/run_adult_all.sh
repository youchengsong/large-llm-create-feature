#!/bin/bash
# Run all baselines + KnowFeat on adult dataset (Server C)
set -e
cd /root/knowfeat
PY=/root/miniconda3/bin/python3.12

echo "=========================================="
echo "Adult Dataset: All Methods"
echo "=========================================="

# 1. Download dataset
echo "--- Downloading adult dataset ---"
$PY scripts/download_public_datasets.py --datasets adult

# 2. No-FE baseline
echo "--- No-FE ---"
$PY scripts/evaluate.py --dataset adult --method No-FE --baseline

# 3. KnowFeat
echo "--- KnowFeat ---"
$PY scripts/run_knowfeat_public.py --dataset adult --max-rounds 8
$PY scripts/evaluate.py --dataset adult --method KnowFeat --features-csv outputs/adult/knowfeat_features.csv

# 4. OpenFE
echo "--- OpenFE ---"
$PY scripts/run_openfe.py --dataset adult 2>&1 || echo "OpenFE failed"
if [ -f outputs/openfe/adult_features.csv ]; then
    $PY scripts/evaluate.py --dataset adult --method OpenFE --features-csv outputs/openfe/adult_features.csv
fi

# 5. CAAFE
echo "--- CAAFE ---"
$PY scripts/run_caafe.py --dataset adult 2>&1 || echo "CAAFE failed"
if [ -f outputs/caafe/adult_features.csv ]; then
    $PY scripts/evaluate.py --dataset adult --method CAAFE --features-csv outputs/caafe/adult_features.csv
fi

# 6. FeatLLM
echo "--- FeatLLM ---"
$PY scripts/run_featllm.py --dataset adult 2>&1 || echo "FeatLLM failed"
if [ -f outputs/featllm/adult_features.csv ]; then
    $PY scripts/evaluate.py --dataset adult --method FeatLLM --features-csv outputs/featllm/adult_features.csv
fi

# 7. AutoFeat
echo "--- AutoFeat ---"
$PY scripts/run_autofeat.py --dataset adult 2>&1 || echo "AutoFeat failed"
if [ -f outputs/autofeat/adult_features.csv ]; then
    $PY scripts/evaluate.py --dataset adult --method AutoFeat --features-csv outputs/autofeat/adult_features.csv
fi

echo ""
echo "Adult 全部实验完成!"
