#!/bin/bash
# SimECNY spw=1 experiments on Server A
# Phase 1: Quick eval — re-evaluate all methods with spw=1 using existing feature CSVs
# Phase 2: Complete — re-run KnowFeat pipeline with L3 spw=1

set -e
cd /root/knowfeat
PY=/root/miniconda3/bin/python3.12

echo "=========================================="
echo "Phase 1: Quick eval with spw=1"
echo "=========================================="

# No-FE baseline
echo "--- No-FE spw=1 ---"
$PY scripts/evaluate.py --dataset simecny --method No-FE_spw1 --spw 1 --baseline

# OpenFE
echo "--- OpenFE spw=1 ---"
$PY scripts/evaluate.py --dataset simecny --method OpenFE_spw1 --features-csv outputs/openfe/simecny_features.csv --spw 1

# CAAFE
echo "--- CAAFE spw=1 ---"
$PY scripts/evaluate.py --dataset simecny --method CAAFE_spw1 --features-csv outputs/caafe/simecny_features.csv --spw 1

# FeatLLM
echo "--- FeatLLM spw=1 ---"
$PY scripts/evaluate.py --dataset simecny --method FeatLLM_spw1 --features-csv outputs/featllm/simecny_features.csv --spw 1

# KnowFeat (existing features, spw=1 eval only)
echo "--- KnowFeat spw=1 (eval only) ---"
$PY scripts/simecny_knowfeat_to_csv.py || echo "WARN: knowfeat CSV conversion skipped"
if [ -f outputs/simecny_knowfeat_features.csv ]; then
    $PY scripts/evaluate.py --dataset simecny --method KnowFeat_spw1_evalonly --features-csv outputs/simecny_knowfeat_features.csv --spw 1
else
    echo "SKIP: no KnowFeat SimECNY feature CSV found"
fi

echo ""
echo "Phase 1 完成!"
echo ""

echo "=========================================="
echo "Phase 2: KnowFeat pipeline with L3 spw=1"
echo "=========================================="

echo "--- KnowFeat SimECNY with spw=1 L3 ---"
$PY -m knowfeat.main --stage stage2 --spw 1

# Convert features to CSV
$PY scripts/simecny_knowfeat_to_csv.py --features-dir outputs/ --output outputs/simecny_knowfeat_spw1_features.csv

# Evaluate
$PY scripts/evaluate.py --dataset simecny --method KnowFeat_spw1 --features-csv outputs/simecny_knowfeat_spw1_features.csv --spw 1

echo ""
echo "Phase 2 完成!"
echo "All spw=1 experiments done."
