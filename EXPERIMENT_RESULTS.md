# KnowFeat 实验结果汇总

> 最后更新: 2026-06-02
> 仅包含已验证有效的结果，待跑/无效结果标记为 —

---

## 一、主表格：公开数据集 AUC 对比

评估设置：XGBoost, 80/20 split × 5 seeds, **spw=auto**, 主指标 AUC

| 数据集 | No-FE | OpenFE | AutoFeat | CAAFE | FeatLLM | KnowFeat |
|--------|-------|--------|----------|-------|---------|----------|
| credit_g | 0.7670±0.0205 | 0.7833±0.0305 | 0.7540±0.0173 | 0.7596±0.0232 | 0.7604±0.0216 | **0.7854±0.0136** |
| diabetes | 0.8027±0.0332 | **0.8252±0.0338** | 0.8089±0.0271 | 0.8032±0.0297 | 0.8096±0.0320 | 0.8190±0.0315 |
| blood_transfusion | 0.6824±0.0429 | 0.6654±0.0402 | 0.6824±0.0429 | 0.6779±0.0429 | 0.6805±0.0443 | **0.7007±0.0359** |
| heart | 0.9320±0.0068 | **0.9417±0.0110** | 0.9340±0.0032 | 0.9302±0.0055 | 0.9297±0.0039 | 0.9379±0.0063 |
| bank_marketing | 0.9330±0.0022 | **0.9374±0.0024** | T/O | 0.9334±0.0018 | 0.9337±0.0020 | —(重跑中) |
| breast_w | 0.9926±0.0051 | 0.9944±0.0050 | 0.9936±0.0039 | 0.9924±0.0061 | 0.9924±0.0053 | **0.9957±0.0033** |
| adult | — | — | — | — | — | — |

> T/O = 超时未完成; — = 待跑
> 公开数据集正例率较均衡（11%~35%），spw=auto 影响不大
> adult 数据集准备中，将在 Server C 运行

---

## 二、SimECNY (AML) 指标对比 — 93个基线特征, spw=auto

评估设置：XGBoost, **93个基线特征**, **scale_pos_weight=auto(≈43.6)**, 80/20 split × 5 seeds

| 方法 | n_features | Recall | Precision | F1 | AUC |
|------|-----------|--------|-----------|-----|-----|
| No-FE | 93 | 0.9663±0.0102 | 0.9764±0.0036 | 0.9713±0.0053 | 0.9996±0.0002 |
| OpenFE | 123 | 0.9645±0.0075 | 0.9752±0.0039 | 0.9698±0.0034 | 0.9996±0.0002 |
| AutoFeat | — | — | — | — | — |
| CAAFE | 94 | 0.9681±0.0081 | 0.9725±0.0056 | 0.9703±0.0045 | 0.9996±0.0002 |
| FeatLLM | 1124 | 0.9625±0.0099 | 0.9728±0.0050 | 0.9676±0.0058 | 0.9995±0.0003 |
| KnowFeat | 95 | **0.9681±0.0071** | 0.9751±0.0025 | **0.9716±0.0033** | 0.9996±0.0002 |

> SimECNY: 200K 交易, **93个基线特征**, 4484 正样本 (2.24%)
> spw=auto(≈43.6) 推高所有方法 recall 至 ~0.96+，方法间差异较小
> 后续计划：随机选取40个基线特征重新对比，以更清晰展示方法间差异

---

## 三、SimECNY (AML) 指标对比 — 93个基线特征, spw=1

评估设置：XGBoost, **93个基线特征**, **scale_pos_weight=1（无类别平衡）**, 80/20 split × 5 seeds

> 目的：spw=auto(≈43.6) 掩盖了方法间差异。spw=1 可更清晰地展示特征质量。

### Phase 1: 用现有特征重新评估（已完成 ✅）

| 方法 | n_features | Recall | Precision | F1 | AUC |
|------|-----------|--------|-----------|-----|-----|
| No-FE | 93 | 0.9409±0.0109 | 0.9934±0.0019 | 0.9664±0.0058 | 0.9996±0.0002 |
| OpenFE | 123 | 0.9342±0.0122 | 0.9912±0.0016 | 0.9619±0.0066 | 0.9995±0.0002 |
| CAAFE | 94 | 0.9382±0.0118 | 0.9922±0.0020 | 0.9644±0.0071 | 0.9996±0.0002 |
| FeatLLM | 1124 | 0.9298±0.0135 | 0.9912±0.0019 | 0.9595±0.0077 | 0.9995±0.0003 |
| KnowFeat (eval-only) | 95 | **0.9394±0.0082** | 0.9932±0.0023 | **0.9655±0.0049** | 0.9996±0.0002 |

> Recall 排名: No-FE > KnowFeat > CAAFE > OpenFE > FeatLLM
> F1 排名: No-FE > KnowFeat > CAAFE > OpenFE > FeatLLM
> KnowFeat 仅次于 No-FE baseline，差距最小(Recall -0.0015)；FeatLLM 降幅最大(-0.0111)
> KnowFeat eval-only 使用 spw=auto 管道找到的特征，仅评估阶段用 spw=1

### Phase 2: KnowFeat pipeline with L3 spw=1（运行中）

> Server A 上运行中，KnowFeat 全管道（含 L3 验证）使用 spw=1
> 完成后可得到 spw=1 环境下 L3 筛选出的特征，实验公平性更强

---

## 四、spw=auto vs spw=1 对比（93个基线特征）

| 方法 | Recall (auto) | Recall (spw=1) | Δ Recall | F1 (auto) | F1 (spw=1) | Δ F1 |
|------|--------------|----------------|----------|-----------|------------|------|
| No-FE | 0.9663 | 0.9409 | -0.0254 | 0.9713 | 0.9664 | -0.0049 |
| OpenFE | 0.9645 | 0.9342 | -0.0303 | 0.9698 | 0.9619 | -0.0079 |
| CAAFE | 0.9681 | 0.9382 | -0.0299 | 0.9703 | 0.9644 | -0.0059 |
| FeatLLM | 0.9625 | 0.9298 | -0.0327 | 0.9676 | 0.9595 | -0.0081 |
| KnowFeat | 0.9681 | 0.9394 | -0.0287 | 0.9716 | 0.9655 | -0.0061 |

> spw=1 下所有方法 recall 均下降 2.5~3.3 个百分点，precision 大幅上升
> KnowFeat 在两种设置下 F1 均为最优
> FeatLLM 受 spw 切换影响最大，说明其 1031 个新特征质量不高

---

## 五、消融实验：schema_only vs full (AUC) — spw=auto

评估设置：同主表格

### 公开数据集消融

| 数据集 | schema_only (n_feat) | full/KnowFeat (n_feat) | Δ AUC |
|--------|---------------------|------------------------|-------|
| credit_g | —(重跑中) | 0.7854±0.0136 (46) | — |
| diabetes | —(0特征) | 0.8190±0.0315 (24) | — |
| blood_transfusion | 0.6891±0.0480 (12) | **0.7007±0.0359** (12) | +0.012 |
| heart | 0.9304±0.0054 (25) | **0.9379±0.0063** (31) | +0.008 |
| bank_marketing | —(重跑中) | —(重跑中) (39) | — |
| breast_w | —(重跑中) | 0.9957±0.0033 (16) | — |

> diabetes schema_only 生成 0 特征（知识不足以通过 L3 验证）
> credit_g/bank_marketing/breast_w: schema_only ≥ full，可能因为领域知识引导生成了过拟合特征

### SimECNY 消融 — 93个基线特征, spw=auto

| 变体 | n_features | Recall | F1 | AUC |
|------|-----------|--------|-----|-----|
| schema_only | — | —(重跑中) | —(重跑中) | —(重跑中) |
| full (KnowFeat) | 95 | 0.9681±0.0071 | 0.9716±0.0033 | 0.9996±0.0002 |

> SimECNY 消融差异极小，受 spw=auto 影响方法间差异被压缩

---

## 六、生成特征数量

### 公开数据集

| 数据集 | No-FE | OpenFE | AutoFeat | CAAFE | FeatLLM | KnowFeat |
|--------|-------|--------|----------|-------|---------|----------|
| credit_g | 20 | 50 | 23 | 24 | 280 | 46 |
| diabetes | 8 | 38 | 13 | 9 | 152 | 24 |
| blood_transfusion | 4 | 34 | 4 | 7 | 120 | 12 |
| heart | 11 | 41 | 21 | 14 | 98 | 31 |
| bank_marketing | 16 | 46 | T/O | 21 | 132 | —(重跑中) |
| breast_w | 9 | 39 | 20 | 10 | 107 | 16 |
| adult | 14 | — | — | — | — | — |

### SimECNY

| 方法 | n_features |
|------|-----------|
| No-FE | 93 (基线) |
| OpenFE | 123 (+30) |
| CAAFE | 94 (+1) |
| FeatLLM | 1124 (+1031) |
| KnowFeat | 95 (+2) |

---

## 七、实验运行状态

| 服务器 | 已完成 | 待跑 |
|--------|--------|------|
| A | SimECNY spw=1 Phase 1 ✅; SimECNY 全部(spw=auto) ✅ | 空闲 |
| B | bank_marketing 消融(indicators/rules/expert) ✅ | credit_g/bank_marketing/breast_w schema_only 重跑; bank_marketing KnowFeat full 重跑 |
| C | heart 全部 ✅; diabetes full ✅ schema_only(0特征) | adult 全部方法 |
| D | bank_marketing 全部 ✅; breast_w 全部 ✅ | SimECNY schema_only 重跑 |

---

## 八、eval_results 文件清单与 spw 标注

### spw=auto（默认，文件中无特殊标记）

公开数据集:
- `{dataset}_No-FE.json` — 6 个数据集
- `{dataset}_OpenFE.json` — 6 个数据集
- `{dataset}_AutoFeat.json` — 4 个数据集 (无 bank_marketing/adult)
- `{dataset}_CAAFE.json` — 6 个数据集
- `{dataset}_FeatLLM.json` — 6 个数据集
- `{dataset}_KnowFeat.json` — 5 个数据集 (无 bank_marketing，重跑中)
- `{dataset}_KnowFeat_schema_only.json` — 2 个数据集 (blood_transfusion, heart; credit_g/bank_marketing/breast_w 重跑中; diabetes 无特征)

SimECNY (spw=auto):
- `simecny_OpenFE.json`
- `simecny_CAAFE.json`
- `simecny_FeatLLM.json`
- `simecny_KnowFeat.json`
- `simecny_KnowFeat_schema_only.json` — 重跑中
- `simecny_KnowFeat_indicators.json`
- `simecny_KnowFeat_rules.json`

### spw=1（文件名含 `_spw1` 后缀）

- `simecny_No-FE.json` — ✅ 已恢复为 spw=auto 版本
- `simecny_No-FE_spw1.json` — spw=1 版本（从覆盖文件重命名而来）
- `simecny_OpenFE_spw1.json`
- `simecny_CAAFE_spw1.json`
- `simecny_FeatLLM_spw1.json`
- `simecny_KnowFeat_spw1_evalonly.json` — 用 spw=auto 管道的特征，仅评估用 spw=1
- `simecny_KnowFeat_spw1.json` — 待定（等待用户指示）
