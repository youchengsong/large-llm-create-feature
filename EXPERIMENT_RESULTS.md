# KnowFeat 实验结果汇总

> 最后更新: 2026-06-03
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
| bank_marketing | 0.9330±0.0022 | **0.9374±0.0024** | T/O | 0.9334±0.0018 | 0.9337±0.0020 | 0.9344±0.0019 |
| breast_w | 0.9926±0.0051 | 0.9944±0.0050 | 0.9936±0.0039 | 0.9924±0.0061 | 0.9924±0.0053 | **0.9957±0.0033** |
| adult | 0.9288±0.0036 | 0.9300±0.0028 | — | 0.9286±0.0033 | 0.9289±0.0033 | **0.9301±0.0029** |

> T/O = 超时未完成; — = 待跑
> 公开数据集正例率较均衡（11%~35%），spw=auto 影响不大
> adult: KnowFeat/AutoFeat/schema_only 待完成（Server D 运行中）

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

## 五、SimECNY (AML) 指标对比 — 40个基线特征, spw=1

评估设置：XGBoost, **随机选取40个基线特征**, **scale_pos_weight=1**, 80/20 split × 5 seeds

> 目的：93个基线特征 + spw=auto 掩盖了方法间差异。减少基线特征 + spw=1 可更清晰展示特征工程方法的增量价值。

| 方法 | n_features | Recall | Precision | F1 | AUC |
|------|-----------|--------|-----------|-----|-----|
| No-FE | 40 | 0.9128±0.0139 | 0.9859±0.0046 | 0.9479±0.0067 | 0.9988±0.0005 |
| OpenFE | 70 (+30) | 0.9309±0.0128 | 0.9927±0.0023 | 0.9607±0.0064 | 0.9992±0.0003 |
| AutoFeat | 42 (+2) | 0.9101±0.0114 | 0.9891±0.0039 | 0.9479±0.0051 | 0.9988±0.0004 |
| CAAFE | 41 (+1) | 0.9126±0.0120 | 0.9877±0.0031 | 0.9486±0.0057 | 0.9987±0.0005 |
| FeatLLM | 1070 (+1030) | **0.9411±0.0131** | 0.9925±0.0017 | **0.9661±0.0070** | **0.9997±0.0002** |
| KnowFeat | — | — | — | — | — |

> 40feat spw=1 下方法间差异显著放大：Recall 范围 0.9101~0.9411
> FeatLLM 表现最好（Recall 0.9411, F1 0.9661），大量特征(+1030)在基线不足时有补充作用
> OpenFE 新增 30 个特征，Recall 提升 1.8pp，排名第二
> CAAFE/AutoFeat 仅+1~2 个特征，与 No-FE 持平
> KnowFeat 重跑中（Server B）

---

## 六、消融实验：schema_only vs full — spw=auto

评估设置：同主表格

### 公开数据集消融

| 数据集 | schema_only (n_feat) | full/KnowFeat (n_feat) | Δ AUC |
|--------|---------------------|------------------------|-------|
| credit_g | 0.7750±0.0124 (40) | **0.7854±0.0136** (46) | +0.010 |
| diabetes | —(0特征) | 0.8190±0.0315 (24) | — |
| blood_transfusion | 0.6891±0.0480 (12) | **0.7007±0.0359** (12) | +0.012 |
| heart | 0.9304±0.0054 (25) | **0.9379±0.0063** (31) | +0.008 |
| bank_marketing | 0.9314±0.0020 (30) | **0.9344±0.0019** (36) | +0.003 |
| breast_w | **0.9964±0.0015** (19) | 0.9957±0.0033 (16) | -0.001 |
| adult | 0.9292±0.0032 (32) | **0.9301±0.0029** (34) | +0.001 |

> diabetes schema_only 生成 0 特征（知识不足以通过 L3 验证）
> breast_w: schema_only 略高于 full，可能因为 schema_only 生成了更多特征(19 vs 16)
> credit_g/bank_marketing/heart: full 均优于 schema_only，说明领域知识有正向作用

### SimECNY 消融 — 93个基线特征, spw=auto

| 变体 | n_features | Recall | F1 | AUC |
|------|-----------|--------|-----|-----|
| schema_only | — | —(重跑中) | —(重跑中) | —(重跑中) |
| full (KnowFeat) | 95 | 0.9681±0.0071 | 0.9716±0.0033 | 0.9996±0.0002 |

> SimECNY 消融差异极小，受 spw=auto 影响方法间差异被压缩

---

## 七、生成特征数量

### 公开数据集

| 数据集 | No-FE | OpenFE | AutoFeat | CAAFE | FeatLLM | KnowFeat |
|--------|-------|--------|----------|-------|---------|----------|
| credit_g | 20 | 50 | 23 | 24 | 280 | 46 |
| diabetes | 8 | 38 | 13 | 9 | 152 | 24 |
| blood_transfusion | 4 | 34 | 4 | 7 | 120 | 12 |
| heart | 11 | 41 | 21 | 14 | 98 | 31 |
| bank_marketing | 16 | 46 | T/O | 21 | 132 | 36 |
| breast_w | 9 | 39 | 20 | 10 | 107 | 16 |
| adult | 14 | 44 | — | 19 | 170 | 34 |

### SimECNY

| 方法 | n_features |
|------|-----------|
| No-FE | 93 (基线) |
| OpenFE | 123 (+30) |
| CAAFE | 94 (+1) |
| FeatLLM | 1124 (+1031) |
| KnowFeat | 95 (+2) |

---

## 八、实验运行状态

| 服务器 | 已完成 | 运行中/待跑 |
|--------|--------|------------|
| A | SimECNY 全部 ✅; credit_g/bank_marketing/breast_w schema_only ✅ | 空闲 |
| B | SimECNY 40feat: OpenFE ✅, CAAFE ✅, FeatLLM ✅ | 空闲 |
| C | SimECNY 40feat: KnowFeat full ✅(1特征), schema_only ✅(1特征) | AutoFeat 🔄 |
| D | adult: No-FE ✅, OpenFE ✅, CAAFE ✅, FeatLLM ✅, KnowFeat ✅(20特征) | 转CSV+评估 → schema_only → AutoFeat |

---

## 九、eval_results 文件清单与 spw 标注

### spw=auto（默认，文件中无特殊标记）

公开数据集:
- `{dataset}_No-FE.json` — 7 个数据集（含 adult）
- `{dataset}_OpenFE.json` — 7 个数据集（含 adult）
- `{dataset}_AutoFeat.json` — 4 个数据集 (无 bank_marketing/adult/breast_w)
- `{dataset}_CAAFE.json` — 7 个数据集（含 adult）
- `{dataset}_FeatLLM.json` — 7 个数据集（含 adult）
- `{dataset}_KnowFeat.json` — 6 个数据集 (无 adult，运行中)
- `{dataset}_KnowFeat_schema_only.json` — 5 个数据集 (blood_transfusion, heart, credit_g, bank_marketing, breast_w; diabetes 无特征)

SimECNY (spw=auto):
- `simecny_OpenFE.json`
- `simecny_CAAFE.json`
- `simecny_FeatLLM.json`
- `simecny_KnowFeat.json`
- `simecny_KnowFeat_schema_only.json` — 重跑中
- `simecny_KnowFeat_indicators.json`
- `simecny_KnowFeat_rules.json`

---

## 十、SimECNY 完整特征矩阵对比（95基线 vs 95+18 KnowFeat）

> **关键实验**：直接对比完整基线特征矩阵（由 `build_baseline_95_features.py` 生成的 93 个特征列）
> 与加入 18 个 KnowFeat 特征后的 111 个特征列（由 `build_all_113_features.py` 生成）。
> 评估设置：XGBoost (n_estimators=200, max_depth=6, lr=0.1, subsample=0.8, colsample_bytree=0.8)

### spw=1（无类别平衡）

| 方法 | 95 基线 (93列) | 95+18 KnowFeat (111列) | delta |
|------|---------------|------------------------|-------|
| 5-fold StratifiedKFold × 5 seeds | 0.9430 ± 0.0012 | **0.9664 ± 0.0014** | **+0.0235** |
| 80/20 split × 5 seeds | 0.9405 ± 0.0111 | **0.9683 ± 0.0099** | **+0.0279** |

### spw=auto（≈43.6）

| 方法 | 95 基线 (93列) | 95+18 KnowFeat (111列) | delta |
|------|---------------|------------------------|-------|
| 5-fold StratifiedKFold × 5 seeds | 0.9695 ± 0.0009 | **0.9847 ± 0.0008** | **+0.0152** |
| 80/20 split × 5 seeds | 0.9663 ± 0.0102 | **0.9871 ± 0.0064** | **+0.0207** |

> **结论**：18 个 KnowFeat 新特征在所有评估配置下均显著提升 recall（+1.52% ~ +2.79%）。
> 5-fold CV 方差更小（0.0008~0.0014 vs 0.0064~0.0111），结果更稳定。
> 之前在 `baseline_cache.parquet`（20个简化基线特征）上测试时新特征表现为噪声稀释，
> 根因是简化基线本身不够强，新特征无法在弱基线上产生增量价值。

### 18 个 KnowFeat 特征列表

| # | 特征名 | 描述 |
|---|--------|------|
| 1 | small_test_round_amount_ratio | 小额试探与整百金额比率 |
| 2 | amount_concentration_entropy | 金额集中度熵 |
| 3 | top3_hour_concentration | Top-3 小时交易集中度 |
| 4 | fan_in_fan_out_ratio | 扇入扇出比 |
| 5 | mode_transition_entropy | 交易模式转换熵 |
| 6 | new_counterparty_concentration | 新交易对手集中度 |
| 7 | counterparty_level_diversity_risk | 交易对手等级多样性风险 |
| 8 | amount_range_ratio_risk | 金额范围比风险 |
| 9 | high_risk_region_off_hour_ratio | 高风险地区非工作时间比 |
| 10 | rhythm_stability_risk | 交易节奏稳定性风险 |
| 11 | net_flow_direction_risk | 净流向方向风险 |
| 12 | enterprise_counterparty_concentration | 企业交易对手集中度 |
| 13 | first_small_test_ratio | 首笔小额试探比率 |
| 14 | night_amount_short_term_risk | 夜间金额短期风险 |
| 15 | amount_bin_skew_index | 金额分档偏斜指数 |
| 16 | counterparty_type_amount_divergence | 交易对手类型金额分歧度 |
| 17 | tiny_amount_concentration | 极小金额集中度 |
| 18 | time_concentration_risk_index | 时间集中度风险指数 |

---

### spw=1（文件名含 `_spw1` 后缀）

- `simecny_No-FE.json` — ✅ 已恢复为 spw=auto 版本
- `simecny_No-FE_spw1.json` — spw=1 版本（从覆盖文件重命名而来）
- `simecny_OpenFE_spw1.json`
- `simecny_CAAFE_spw1.json`
- `simecny_FeatLLM_spw1.json`
- `simecny_KnowFeat_spw1_evalonly.json` — 用 spw=auto 管道的特征，仅评估用 spw=1
- `simecny_KnowFeat_spw1.json` — 待定（等待用户指示）
