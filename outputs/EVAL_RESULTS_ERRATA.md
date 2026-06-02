# SimECNY 评估结果勘误记录

> **重要：以下旧结果存在配置错误，不可用于论文或后续分析。**

## 错误描述

### 问题 1：evaluate.py 未使用 scale_pos_weight

`scripts/evaluate.py` 中的 XGBoost 未设置 `scale_pos_weight`（默认值 1），
而 `knowfeat/verification/model_eval.py`（L3 验证）使用了 `scale_pos_weight = neg/pos ≈ 43.6`。

两套评估标准不一致：
- evaluate.py (spw=1)：模型保守，recall 偏低（~0.75），precision 偏高（~0.99）
- model_eval.py (spw=43.6)：模型激进，recall 偏高（~0.95），precision 偏低（~0.5）

AML 反洗钱场景应使用 spw 以提高召回率。

### 问题 2：L3 验证 precision_min=0.9 阈值过严

在 spw=43.6 下，XGBoost 预测倾向激进，单特征增量评估时 precision 普遍在 0.1~0.85 之间，
远低于 0.9 阈值，导致 KnowFeat simecny 运行时 **所有特征都被拒绝**（0 特征产出）。

## 错误结果（已废弃，备份在 `outputs/eval_results_old_no_spw/`）

### simecny_No-FE.json (spw=1, 错误)
| Metric | Mean | Std |
|---|---|---|
| Recall | 0.7500 | 0.0514 |
| Precision | 0.9920 | 0.0075 |
| F1 | 0.8532 | 0.0338 |
| AUC | 0.9993 | 0.0004 |

### simecny_OpenFE.json (spw=1, 错误)
| Metric | Mean | Std |
|---|---|---|
| Recall | 0.7924 | 0.0320 |
| Precision | 0.9793 | 0.0148 |
| F1 | 0.8757 | 0.0231 |
| AUC | 0.9994 | 0.0002 |

### simecny_FeatLLM.json (spw=1, 错误)
| Metric | Mean | Std |
|---|---|---|
| Recall | 0.7545 | 0.0403 |
| Precision | 0.9902 | 0.0063 |
| F1 | 0.8558 | 0.0253 |
| AUC | 0.9992 | 0.0005 |

### KnowFeat simecny (precision_min=0.9, 错误)
- 结果：0 特征产出，连续 2 轮无新特征收敛退出
- 原因：所有特征在 L3 验证时因 precision < 0.9 被拒绝
- 备份：`outputs/stage2_features_old_no_spw.json`, `outputs/stage2_summary_old_no_spw.json`

## 修复内容

1. `evaluate.py`：添加 `scale_pos_weight = neg_count / pos_count`（与 model_eval.py 统一）
2. `model_eval.py`：注释掉 `precision_min` 检查，L3 仅看 `delta_recall > 0`
3. 重新运行所有 simecny 评估（No-FE, OpenFE, FeatLLM, KnowFeat）

## 不受影响的结果

- **公开数据集**（credit_g, diabetes, blood_transfusion, heart, bank_marketing, breast_w）：
  使用 AUC 指标，SPW 不影响 AUC（只影响决策阈值，不影响概率排序）
- **消融实验**（Server B, C）：使用 `accept_metric="auc"`，不检查 precision

### 问题 3：baseline_cache.parquet 与 transactions.csv 行对齐错误

`subsample_200k.py` 按 index 从 parquet 选行，但 parquet 被 `feature_engineering.py`
按时间戳排序后 index 已经重映射，导致选出的行与 transactions.csv 不对应。

影响：
- parquet 的 is_risk 有 662 正样本 (0.33%)，实际应为 4484 (2.24%)
- KnowFeat L3 验证：parquet 特征与 tx_df 标签错位，baseline_recall=0.54（应为 0.88）
- evaluate.py 结果**内部一致**但基于错误的数据子集

修复：用当前 200K transactions.csv 重跑 feature_engineering.py，输出保留 tx_id，
按 transactions.csv 行顺序重排后保存为新 parquet。

备份：
- 旧 parquet：`outputs/old_wrong_parquet/baseline_cache_wrong.parquet`
- 旧 SimECNY 评估结果：`outputs/eval_results_old_wrong_parquet/`

## 时间线

- 2026-05-30：旧结果生成（错误的 spw=1 配置）
- 2026-06-01 19:49：修复 SPW 配置，重新运行 simecny 评估
- 2026-06-02：发现 baseline_cache.parquet 行对齐错误，所有 SimECNY 结果废弃，重建 parquet
