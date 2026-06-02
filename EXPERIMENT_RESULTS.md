# KnowFeat 实验结果汇总

> 最后更新: 2026-06-02
> 仅包含已验证有效的结果，待跑/无效结果标记为 —

---

## 一、主表格：公开数据集 AUC 对比

评估设置：XGBoost, 80/20 split × 5 seeds, 主指标 AUC

| 数据集 | No-FE | OpenFE | AutoFeat | CAAFE | FeatLLM | KnowFeat |
|--------|-------|--------|----------|-------|---------|----------|
| credit_g | 0.7670±0.0205 | 0.7833±0.0305 | 0.7540±0.0173 | 0.7596±0.0232 | 0.7604±0.0216 | **0.7854±0.0136** |
| diabetes | 0.8027±0.0332 | **0.8252±0.0338** | 0.8089±0.0271 | 0.8032±0.0297 | 0.8096±0.0320 | 0.8123±0.0339 |
| blood_transfusion | **0.6824±0.0429** | 0.6654±0.0402 | 0.6824±0.0429 | 0.6779±0.0429 | 0.6805±0.0443 | 0.6727±0.0596 |
| heart | 0.9320±0.0068 | **0.9417±0.0110** | 0.9340±0.0032 | 0.9302±0.0055 | 0.9297±0.0039 | 0.9379±0.0063 |
| bank_marketing | 0.9330±0.0022 | **0.9374±0.0024** | T/O | 0.9334±0.0018 | 0.9337±0.0020 | 0.9327±0.0028 |
| breast_w | 0.9926±0.0051 | 0.9944±0.0050 | 0.9936±0.0039 | 0.9924±0.0061 | 0.9924±0.0053 | **0.9938±0.0047** |

> T/O = 超时未完成

---

## 二、主表格：SimECNY (AML) 指标对比

评估设置：XGBoost, scale_pos_weight=auto, 80/20 split × 5 seeds

| 方法 | n_features | Recall | Precision | F1 | AUC |
|------|-----------|--------|-----------|-----|-----|
| No-FE | — | — | — | — | — |
| OpenFE | — | — | — | — | — |
| AutoFeat | — | — | — | — | — |
| CAAFE | — | — | — | — | — |
| FeatLLM | — | — | — | — | — |
| KnowFeat | — | — | — | — | — |

> SimECNY: 200K 交易, 95 基线特征（正样本数待 baseline_cache 重建后确认）
> 旧结果因 baseline_cache.parquet 行对齐错误已全部废弃，需重新评估
> 旧结果备份在 outputs/eval_results_old_wrong_parquet/

---

## 三、生成特征数量

### 公开数据集

| 数据集 | No-FE | OpenFE | AutoFeat | CAAFE | FeatLLM | KnowFeat |
|--------|-------|--------|----------|-------|---------|----------|
| credit_g | 20 | 50 | 23 | 24 | 280 | 46 |
| diabetes | 8 | 38 | 13 | 9 | 152 | 23 |
| blood_transfusion | 4 | 34 | 4 | 7 | 120 | 8 |
| heart | 11 | 41 | 21 | 14 | 98 | 31 |
| bank_marketing | 16 | 46 | T/O | 21 | 132 | 39 |
| breast_w | 9 | 39 | 20 | 10 | 107 | 16 |

### SimECNY

| 方法 | n_features |
|------|-----------|
| No-FE | 95 (基线) |
| OpenFE | 125 (+30) |
| FeatLLM | 1126 (+1031) |
| KnowFeat | — |

---

## 四、消融实验：知识层级对 KnowFeat 的影响 (AUC)

评估设置：同主表格

| 数据集 | schema_only | indicators | rules | expert | full (KnowFeat) |
|--------|-------------|------------|-------|--------|-----------------|
| credit_g | 0.7952±0.0130 | — | — | — | 0.7854±0.0136 |
| diabetes | — | — | — | — | 0.8123±0.0339 |
| blood_transfusion | — | — | — | — | 0.6727±0.0596 |
| heart | 0.9304±0.0054 | — | — | — | 0.9379±0.0063 |
| bank_marketing | 0.9340±0.0028 | — | — | — | 0.9327±0.0028 |
| breast_w | 0.9978±0.0014 | — | — | — | 0.9938±0.0047 |

> indicators / rules / expert 消融实验待跑

---

## 五、实验运行状态

| 服务器 | 已完成 | 待跑 |
|--------|--------|------|
| A | breast_w full ✅, breast_w schema_only ✅, simecny full ✅ | breast_w 3消融, simecny 4消融 |
| B | credit_g full ✅, credit_g schema_only ✅ | credit_g 3消融, blood_transfusion 全部(full + 4消融) |
| C | heart full ✅, heart schema_only ✅ | heart 3消融, diabetes 全部(full + 4消融) |
| D | bank_marketing full ✅, bank_marketing schema_only ✅ | bank_marketing 3消融 |

> 所有服务器空闲，baseline_cache.parquet 已更新(200K, 4484正样本)
> SimECNY 评估待 baseline_cache 重建后重跑（No-FE, OpenFE, FeatLLM, KnowFeat）
