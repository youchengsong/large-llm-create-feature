# 实验结果汇总

> 最后更新：2026-05-29

## 1. 数据集概况

| 项目 | 值 |
|------|-----|
| 数据集 | BABD-13 (SimECNY 数字人民币仿真) |
| 总样本数 | 1,384,512 |
| 正样本 (is_laundering=1) | 4,484 (0.32%) |
| 负样本 | 1,380,028 |
| 原始特征数 | 106 列 (含 1 标签列) |
| 全量基线特征 | 105 个 |
| 精简基线特征 | 95 个 (排除 10 个低质量特征) |

### 排除的 10 个低质量特征

`Min_Received`, `Min_Sent`, `S_src_counterparty_index`, `Timestamp_Hour`, `S_dst_counterparty_index`, `1H_frequency_From`, `1H_frequency_To`, `wallets_per_cert_From`, `min_inactive_period_From`, `Max_Sent`

---

## 2. 基线实验：scale_pos_weight 参数调优

XGBoost 参数：`n_estimators=200, max_depth=6, learning_rate=0.1, 5-fold StratifiedKFold CV`

### 105 特征（全量基线）

| scale_pos_weight | recall | precision | f1 | auc |
|-----------------|--------|-----------|------|------|
| 19.0 | 0.9681±0.0051 | 0.9828±0.0060 | 0.9754±0.0033 | 1.0000±0.0000 |
| 39.0 | 0.9728±0.0050 | 0.9787±0.0057 | 0.9757±0.0040 | 1.0000±0.0000 |
| **59.0** | **0.9739±0.0045** | **0.9781±0.0064** | **0.9760±0.0034** | **1.0000±0.0000** |
| 99.0 | 0.9759±0.0048 | 0.9714±0.0073 | 0.9736±0.0034 | 1.0000±0.0000 |
| 307.8 | 0.9855 | 0.8620 | 0.9196 | — |

### 95 特征（精简基线）

| scale_pos_weight | recall | precision | f1 | precision 余量 (vs 0.9) |
|-----------------|--------|-----------|------|----------------------|
| 19.0 | 0.9407±0.0109 | 0.9641±0.0067 | 0.9522±0.0079 | +0.064 |
| 39.0 | 0.9494±0.0070 | 0.9475±0.0042 | 0.9484±0.0048 | +0.048 |
| **59.0** | **0.9538±0.0057** | **0.9402±0.0044** | **0.9470±0.0036** | **+0.040** |
| 99.0 | 0.9603±0.0076 | 0.9152±0.0052 | 0.9372±0.0053 | +0.015 |
| 307.8 | 0.9726 | 0.5372 | 0.6921 | -0.363 |

### 105 vs 95 特征对比 (scale_pos_weight=59)

| 指标 | 105 特征 | 95 特征 | Δ |
|------|---------|---------|-----|
| recall | 0.9739 | 0.9538 | -0.0201 |
| precision | 0.9781 | 0.9402 | -0.0379 |
| f1 | 0.9760 | 0.9470 | -0.0290 |
| auc | 1.0000 | 0.9998 | -0.0002 |

**结论**：选择 `scale_pos_weight=59`，precision 余量 +0.040，足够容纳新特征引入的波动。99 时余量仅 +0.015，风险过高。

---

## 3. Stage 2 pipeline 运行结果

### 运行概况

| 项目 | 值 |
|------|-----|
| 启动时间 | 2026-05-28 22:52 |
| 完成时间 | 2026-05-29 08:20 |
| 总耗时 | 34,065.8 秒 (~9.5 小时) |
| 总轮数 | 7 轮 |
| 接受特征数 | 18 / 30 |
| 候选池剩余 | 0 |
| 拒绝数 | 25 |
| 收敛原因 | 连续 2 轮无新特征 |

### 模型参数

| 参数 | 值 |
|------|-----|
| scale_pos_weight | 59.0 |
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| eval_metric | logloss |

### 噪声校准阈值

| 指标 | p95 值 | fallback | 最终阈值 |
|------|--------|----------|---------|
| IV | 0.0038 | 0.0200 | 0.0200 |
| KS | 0.0204 | 0.0200 | 0.0204 |

### 每轮统计

| 轮次 | 时间 | 金额频次 | 时间地域 | 行为序列 | 主体关联 | 前向选择 | 新增 | 累计 |
|------|------|---------|---------|---------|---------|---------|------|------|
| R1 | 22:52→00:21 | +2 | +1 | +2 | +1 | 1/6 接受 | **+7** | 7/30 |
| R2 | 00:21→01:40 | +1 | +1 | +2 | +1 | 1/4 接受 | **+6** | 13/30 |
| R3 | 01:40→03:04 | 0 | +1 | 0 | 0 | 0/6 | **+1** | 14/30 |
| R4 | 03:04→04:26 | +1 | 0 | 0 | +1 | 1/5 接受 | **+3** | 17/30 |
| R5 | 04:26→05:36 | 0 | +1 | 0 | 0 | 0/5 | **+1** | 18/30 |
| R6 | 05:36→06:45 | 0 | 0 | 0 | 0 | 0/5 | 0 | 18/30 |
| R7 | 06:45→08:20 | 0 | 0 | 0 | 0 | 0/9 | 0 | 18/30 |

### 维度统计

| 维度 | 已生成 | 已接受 | 候选池 | 拒绝 | SHAP 贡献 |
|------|--------|--------|--------|------|----------|
| 金额频次 | 12 | 4 | 7 | 1 | **4.7411** |
| 时间地域 | 17 | 4 | 5 | 7 | 1.0031 |
| 行为序列 | 25 | 4 | 12 | 9 | 0.4635 |
| 主体关联 | 30 | 3 | 16 | 8 | 0.2688 |

### Recall 提升轨迹 (95 基线特征 + KnowFeat 新特征)

| # | 特征名 | recall | precision | delta_recall | IV | KS | 冗余度 |
|---|--------|--------|-----------|-------------|------|------|--------|
| baseline | 95 个精简基线特征 | 0.9538 | 0.9402 | — | — | — | — |
| 1 | small_test_round_amount_ratio | 0.9590 | 0.9407 | +0.0051 | 2.5893 | 0.7451 | 0.0000 |
| 2 | amount_concentration_entropy | 0.9614 | 0.9387 | +0.0025 | 6.0102 | 0.8683 | 0.2597 |
| 3 | top3_hour_concentration | 0.9625 | 0.9453 | +0.0011 | 3.4737 | 0.7376 | 0.1508 |
| 4 | fan_in_fan_out_ratio | 0.9639 | 0.9489 | +0.0013 | 0.1470 | 0.3535 | 0.1786 |
| 5 | mode_transition_entropy | 0.9663 | 0.9517 | +0.0025 | 2.4628 | 0.4628 | 0.2412 |
| 6 | new_counterparty_concentration | 0.9668 | 0.9500 | +0.0004 | 5.6824 | 0.8590 | 0.4502 |
| 7 | counterparty_level_diversity_risk (前向选择) | 0.9677 | 0.9524 | +0.0009 | 1.1905 | 0.5921 | 0.3808 |
| 8 | amount_range_ratio_risk | 0.9677 | 0.9513 | +0.0000 | 1.4108 | 0.3064 | 0.4601 |
| 9 | high_risk_region_off_hour_ratio | 0.9681 | 0.9508 | +0.0004 | 0.9922 | 0.3967 | 0.1667 |
| 10 | rhythm_stability_risk | 0.9692 | 0.9560 | +0.0011 | 3.2052 | 0.7364 | 0.4370 |
| 11 | net_flow_direction_risk | 0.9699 | 0.9567 | +0.0007 | 0.4483 | 0.2728 | 0.4204 |
| 12 | enterprise_counterparty_concentration | 0.9699 | 0.9559 | +0.0000 | 0.3595 | 0.2035 | 0.2714 |
| 13 | first_small_test_ratio (前向选择) | 0.9710 | 0.9574 | +0.0011 | 1.7636 | 0.4947 | 0.4019 |
| 14 | night_amount_short_term_risk | 0.9721 | 0.9601 | +0.0011 | 1.0930 | 0.2319 | 0.2619 |
| 15 | amount_bin_skew_index | 0.9730 | 0.9627 | +0.0009 | 4.8824 | 0.7570 | 0.6065 |
| 16 | counterparty_type_amount_divergence | 0.9735 | 0.9602 | +0.0004 | 0.9489 | 0.3341 | 0.1175 |
| 17 | tiny_amount_concentration (前向选择) | 0.9735 | 0.9600 | +0.0000 | 4.8056 | 0.7494 | 0.3599 |
| 18 | time_concentration_risk_index | 0.9737 | 0.9621 | +0.0002 | 4.1398 | 0.8058 | 0.6881 |

### SHAP Top 10 特征重要性

| 排名 | 特征名 | SHAP 值 | 维度 |
|------|--------|---------|------|
| 1 | amount_concentration_entropy | 3.5310 | 金额频次 |
| 2 | amount_bin_skew_index | 0.7828 | 金额频次 |
| 3 | time_concentration_risk_index | 0.6590 | 时间地域 |
| 4 | activetime_To | 0.4010 | 基线 |
| 5 | rhythm_stability_risk | 0.3378 | 行为序列 |
| 6 | counts_1H_From | 0.2602 | 基线 |
| 7 | transaction_frequency_To | 0.2586 | 基线 |
| 8 | wallet_type_From | 0.2352 | 基线 |
| 9 | send_amount_concentration | 0.2269 | 基线 |
| 10 | amount_range_ratio_risk | 0.2221 | 金额频次 |

### 最终效果对比

| 配置 | 特征数 | recall | precision | f1 |
|------|--------|--------|-----------|------|
| 全量基线 (105) | 105 | 0.9739 | 0.9781 | 0.9760 |
| 精简基线 (95) | 95 | 0.9538 | 0.9402 | 0.9470 |
| **精简基线 + 17 KnowFeat** | **112** | **0.9536** | **0.9408** | **0.9472** |

> 注：night_amount_short_term_risk 因 timestamp 列类型问题未加载（17/18 成功），实际特征数为 112。
> 上表 recall/precision/f1 为独立 5-fold CV 结果（非增量 delta），与 Recall 提升轨迹表中逐步累加的 recall 值不同。

### CR@K 对比 (Capture Rate at Top K%)

| 配置 | CR@0.1% | CR@0.2% | CR@0.3% | CR@0.5% | CR@1% | CR@3% | CR@5% |
|------|---------|---------|---------|---------|-------|-------|-------|
| 全量基线 (105) | 0.3078±0.0001 | 0.6166±0.0003 | 0.9228±0.0018 | 0.9982±0.0018 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 |
| 精简基线 (95) | 0.3078±0.0001 | 0.6166±0.0003 | 0.9135±0.0056 | 0.9842±0.0038 | 0.9951±0.0024 | 0.9978±0.0014 | 0.9993±0.0005 |
| **精简+17 KnowFeat (112)** | **0.3078±0.0001** | **0.6166±0.0003** | **0.9126±0.0055** | **0.9824±0.0060** | **0.9940±0.0038** | **0.9978±0.0019** | **0.9993±0.0009** |

> **分析**：CR@0.1%=0.3078 和 CR@0.2%=0.6166 在三组实验中完全一致，说明排名最高的 ~1385 和 ~2769 条交易的排序已由基线特征决定。CR@0.3% 开始出现差异，这是因为正样本率为 0.32%，在 CR@0.3% 附近模型需要精确区分正负样本边界。KnowFeat 特征在 CR@3%/CR@5% 与精简基线持平（0.9978/0.9993），未引入负面影响。

---

## 3.1 已接受特征详细信息

### #1 small_test_round_amount_ratio

- **维度**：金额频次
- **L2 验证**：IV=2.5893, KS=0.7451, 缺失率=0.00%, 冗余度=0.0000
- **L3 验证**：recall 0.9538→0.9590 (+0.0051), precision=0.9407
- **解释**：组合特征，同时捕捉小额测试交易比例高（账户可用性测试，RULE_002）和整数金额交易比例高（规避监管，RULE_015）。异常账户 small_tx_ratio 是正常的 6 倍，round_100_ratio 是正常的 3.64 倍。
- **知识来源**：RULE_002（休眠账户先小额测试后大额交易）、RULE_015（网络赌博100元整数倍）、裁判文书、邮储银行专家意见

<details>
<summary>特征代码</summary>

```python
def feature_small_test_round_amount_ratio(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()
    src_features = tx.groupby('src').agg(
        tx_count=('amount', 'count'),
        small_tx_count=('amount_abs', lambda x: (x <= 5).sum()),
        round_100_count=('amount_abs', lambda x: (x % 100 == 0).sum()),
    ).reset_index()
    src_features['small_tx_ratio'] = src_features['small_tx_count'] / (src_features['tx_count'] + 1e-10)
    src_features['round_100_ratio'] = src_features['round_100_count'] / (src_features['tx_count'] + 1e-10)
    src_features['small_test_round_amount_ratio'] = src_features['small_tx_ratio'] * (1 + src_features['round_100_ratio'])
    result = src_features[['src', 'small_test_round_amount_ratio']].copy()
    result.columns = ['wallet_id', 'small_test_round_amount_ratio']
    return result
```

</details>

### #2 amount_concentration_entropy

- **维度**：金额频次
- **L2 验证**：IV=6.0102, KS=0.8683, 缺失率=0.00%, 冗余度=0.2597
- **L3 验证**：recall 0.9590→0.9614 (+0.0025), precision=0.9387
- **解释**：捕捉洗钱账户金额模式单一。Top3 金额占比（异常 59.8% vs 正常 33.5%，1.78 倍）× (1 - 归一化熵)，量化金额集中度。
- **知识来源**：RULE_012（信用卡套现整数金额占比≥0.8）、RULE_015（网络赌博100元整数倍）、邮储银行专家意见

<details>
<summary>特征代码</summary>

```python
def feature_amount_concentration_entropy(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()
    def calc_top3_ratio(x):
        counts = x.value_counts()
        if len(counts) == 0: return 0.0
        return counts.head(3).sum() / len(x)
    def calc_entropy(x):
        counts = x.value_counts()
        probs = counts / len(x)
        return -np.sum(probs * np.log2(probs + 1e-10))
    src_features = tx.groupby('src')['amount_abs'].agg([
        ('amount_top3_ratio', calc_top3_ratio),
        ('amount_entropy', calc_entropy),
        ('tx_count', 'count')
    ]).reset_index()
    max_entropy = np.log2(src_features['tx_count'].clip(lower=2))
    src_features['norm_entropy'] = src_features['amount_entropy'] / (max_entropy + 1e-10)
    src_features['amount_concentration_entropy'] = src_features['amount_top3_ratio'] * (1 - src_features['norm_entropy'].clip(0, 1))
    result = src_features[['src', 'amount_concentration_entropy']].copy()
    result.columns = ['wallet_id', 'amount_concentration_entropy']
    return result
```

</details>

### #3 top3_hour_concentration

- **维度**：时间地域
- **L2 验证**：IV=3.4737, KS=0.7376, 缺失率=0.00%, 冗余度=0.1508
- **L3 验证**：recall 0.9614→0.9625 (+0.0011), precision=0.9453
- **解释**：交易最集中的 3 个小时的交易占比。异常钱包 0.7632 vs 正常 0.6183，差异 23%。
- **知识来源**：RULE_011（毒品交易凌晨交易）、RULE_015（网络赌博深夜交易）、线上跑分凌晨操作

<details>
<summary>特征代码</summary>

```python
def feature_top3_hour_concentration(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    tx = tx_df.copy()
    tx['hour'] = tx['hour'].astype(int)
    def calc_top3_hour_ratio(group):
        counts = group.value_counts()
        if len(counts) == 0: return 0.0
        return counts.head(3).sum() / len(group)
    result = tx.groupby('src')['hour'].apply(calc_top3_hour_ratio).reset_index()
    result.columns = ['wallet_id', 'top3_hour_concentration']
    return result
```

</details>

### #4 fan_in_fan_out_ratio

- **维度**：行为序列
- **L2 验证**：IV=0.1470, KS=0.3535, 缺失率=0.00%, 冗余度=0.1786
- **L3 验证**：recall 0.9625→0.9639 (+0.0013), precision=0.9489
- **解释**：fan_in 占比 / fan_out 占比 × (1 + 异常模式总占比)。异常 src 的 fan_in 占比 12.94%（正常 1.40%，9.2 倍差异）。
- **知识来源**：RULE_006（地下钱庄分散转入集中转出）、RULE_015（网络赌博分散转入分散转出）、裁判文书

<details>
<summary>特征代码</summary>

```python
def feature_fan_in_fan_out_ratio(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    tx = tx_df.copy()
    def calc_mode_ratios(group):
        total = len(group)
        return pd.Series({
            'fan_in_ratio': (group == 'fan_in').sum() / total,
            'fan_out_ratio': (group == 'fan_out').sum() / total,
            'many_to_many_ratio': (group == 'many_to_many').sum() / total,
            'one_to_many_ratio': (group == 'one_to_many').sum() / total,
        })
    mode_stats = tx.groupby('src')['transaction_mode'].apply(calc_mode_ratios).unstack()
    mode_stats['fan_in_fan_out_ratio'] = mode_stats['fan_in_ratio'] / (mode_stats['fan_out_ratio'] + 1e-10)
    mode_stats['abnormal_mode_ratio'] = mode_stats['fan_in_ratio'] + mode_stats['many_to_many_ratio'] + mode_stats['one_to_many_ratio']
    result_val = mode_stats['fan_in_fan_out_ratio'] * (1 + mode_stats['abnormal_mode_ratio'])
    result = pd.DataFrame({'wallet_id': mode_stats.index, 'fan_in_fan_out_ratio': result_val.values})
    return result
```

</details>

### #5 mode_transition_entropy

- **维度**：行为序列
- **L2 验证**：IV=2.4628, KS=0.4628, 缺失率=0.00%, 冗余度=0.2412
- **L3 验证**：recall 0.9639→0.9663 (+0.0025), precision=0.9517
- **解释**：交易模式归一化熵的补数 (1-熵)。异常 src 使用 2.19 种模式（正常 2.99），模式熵 0.8755（正常 1.0112），模式更单一。
- **知识来源**：邮储银行专家意见（线上跑分钱包互转、线下跑分ATM取现、商户收款码跑分）

<details>
<summary>特征代码</summary>

```python
def feature_mode_transition_entropy(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    tx = tx_df.copy()
    def calc_normalized_entropy(group):
        counts = group.value_counts()
        probs = counts / len(group)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        return entropy / max_entropy
    src_entropy = tx.groupby('src')['transaction_mode'].apply(calc_normalized_entropy).reset_index()
    src_entropy.columns = ['wallet_id', 'mode_transition_entropy']
    src_entropy['mode_transition_entropy'] = 1.0 - src_entropy['mode_transition_entropy']
    return src_entropy
```

</details>

### #6 new_counterparty_concentration

- **维度**：主体关联
- **L2 验证**：IV=5.6824, KS=0.8590, 缺失率=0.00%, 冗余度=0.4502
- **L3 验证**：recall 0.9663→0.9668 (+0.0004), precision=0.9500
- **解释**：新交易对手比例 / (交易对手数量 + 1)。异常钱包新对手比例 0.878（正常 0.748），对手数 6.91（正常 13.88）。
- **知识来源**：裁判文书（多个钱包接收不明资金）、邮储专家（跑分团伙短期交易）

<details>
<summary>特征代码</summary>

```python
def feature_new_counterparty_concentration(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    pair_counts = tx_df.groupby(['src', 'dst']).size().reset_index()
    pair_counts.columns = ['src', 'dst', 'pair_count']
    src_stats = pair_counts.groupby('src').agg(
        total_counterparties=('dst', 'count'),
        new_counterparties=('pair_count', lambda x: (x == 1).sum())
    ).reset_index()
    src_stats['new_counterparty_ratio'] = src_stats['new_counterparties'] / (src_stats['total_counterparties'] + 1e-10)
    src_stats['new_counterparty_concentration'] = src_stats['new_counterparty_ratio'] / (src_stats['total_counterparties'] + 1.0)
    result = src_stats[['src', 'new_counterparty_concentration']].copy()
    result.columns = ['wallet_id', 'new_counterparty_concentration']
    return result
```

</details>

### #7 counterparty_level_diversity_risk (前向选择)

- **维度**：主体关联
- **L2 验证**：IV=1.1905, KS=0.5921, 缺失率=0.00%, 冗余度=0.3808
- **L3 验证**：recall 0.9668→0.9677 (+0.0009), precision=0.9524
- **解释**：交易对手钱包等级多样性 × 自身等级组合。异常钱包 dst_level_diversity 2.61（正常 3.25），src_wallet_level 1.91（正常 2.36）。
- **知识来源**：邮储专家（线下跑分四类匿名钱包ATM取现、线上跑分匿名→实名→银行链）、RULE_001（四类账户大额取现）

<details>
<summary>特征代码</summary>

```python
def feature_counterparty_level_diversity_risk(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    level_diversity = tx_df.groupby('src')['dst_wallet_level'].nunique().reset_index()
    level_diversity.columns = ['wallet_id', 'dst_level_diversity']
    src_self_level = tx_df.groupby('src')['src_wallet_level'].first().reset_index()
    src_self_level.columns = ['wallet_id', 'self_wallet_level']
    result = level_diversity.merge(src_self_level, on='wallet_id', how='left')
    diversity_risk = 1.0 - result['dst_level_diversity'] / 4.0
    level_risk = 1.0 - result['self_wallet_level'] / 4.0
    result['counterparty_level_diversity_risk'] = diversity_risk * (1.0 + level_risk)
    return result[['wallet_id', 'counterparty_level_diversity_risk']]
```

</details>

### #8 amount_range_ratio_risk

- **维度**：金额频次
- **L2 验证**：IV=1.4108, KS=0.3064, 缺失率=0.00%, 冗余度=0.4601
- **L3 验证**：recall 0.9677→0.9677 (+0.0000), precision=0.9513
- **解释**：最大正金额 / 最小正金额，log 变换。异常钱包 amount_range_ratio 均值 10273.8（正常 1691.5，6.07 倍）。
- **知识来源**：RULE_002（休眠账户先小额测试再大额交易）、裁判文书（跑分团伙验卡测试）、邮储专家（线上跑分金额跨度大）

<details>
<summary>特征代码</summary>

```python
def feature_amount_range_ratio_risk(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    tx = tx_df.copy()
    tx_pos = tx[tx['amount'] > 0].copy()
    if len(tx_pos) == 0:
        return pd.DataFrame({'wallet_id': tx_df['src'].unique(), 'amount_range_ratio_risk': 0.0})
    range_stats = tx_pos.groupby('src').agg(
        max_amount=('amount', 'max'), min_positive=('amount', 'min'), tx_count=('amount', 'count'),
    ).reset_index()
    range_stats['amount_range_ratio'] = range_stats['max_amount'] / (range_stats['min_positive'] + 1e-10)
    range_stats['amount_range_ratio_risk'] = np.log1p(range_stats['amount_range_ratio'])
    range_stats.loc[range_stats['tx_count'] < 2, 'amount_range_ratio_risk'] = 0.0
    result = range_stats[['src', 'amount_range_ratio_risk']].copy()
    result.columns = ['wallet_id', 'amount_range_ratio_risk']
    return result
```

</details>

### #9 high_risk_region_off_hour_ratio

- **维度**：时间地域
- **L2 验证**：IV=0.9922, KS=0.3967, 缺失率=0.00%, 冗余度=0.1667
- **L3 验证**：recall 0.9677→0.9681 (+0.0004), precision=0.9508
- **解释**：高风险地区(西藏 540000 / 新疆 650000) × 非工作时间(18:00-7:59)交易占比。异常钱包 69.9% 来自高风险地区，组合后差异 43 倍。
- **知识来源**：RULE_014（恐怖融资特征）、高风险地区分析

<details>
<summary>特征代码</summary>

```python
def feature_high_risk_region_off_hour_ratio(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    high_risk_regions = [540000, 650000]
    open_df = open_df.copy()
    open_df['is_high_risk_region'] = open_df['region_code'].isin(high_risk_regions).astype(int)
    tx = tx_df.copy()
    tx['is_off_hour'] = ((tx['hour'] >= 18) | (tx['hour'] < 8)).astype(int)
    tx_with_region = tx.merge(open_df[['wallet_id', 'is_high_risk_region']],
                               left_on='src', right_on='wallet_id', how='left')
    src_stats = tx_with_region.groupby('src').agg(
        total_tx=('tx_id', 'count'), off_hour_tx=('is_off_hour', 'sum'), high_risk_flag=('is_high_risk_region', 'first')
    ).reset_index()
    src_stats['off_hour_ratio'] = src_stats['off_hour_tx'] / (src_stats['total_tx'] + 1e-10)
    src_stats['high_risk_region_off_hour_ratio'] = src_stats['high_risk_flag'] * src_stats['off_hour_ratio']
    result = src_stats[['src', 'high_risk_region_off_hour_ratio']].copy()
    result.columns = ['wallet_id', 'high_risk_region_off_hour_ratio']
    return result
```

</details>

### #10 rhythm_stability_risk

- **维度**：行为序列
- **L2 验证**：IV=3.2052, KS=0.7364, 缺失率=0.00%, 冗余度=0.4370
- **L3 验证**：recall 0.9681→0.9692 (+0.0011), precision=0.9560
- **解释**：长间隔比例(>300s) × 节奏稳定性 1/(1+CV)。异常钱包间隔 CV 1.08（正常 2.28），长间隔比例 0.47（正常 0.25），组合差异 2.4 倍。
- **知识来源**：RULE_006（分散转入集中转出）、邮储专家（跑分车队规律取现）

<details>
<summary>特征代码</summary>

```python
def feature_rhythm_stability_risk(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    src_stats = tx_df.groupby('src').agg(
        interval_mean=('interval', 'mean'), interval_std=('interval', 'std'),
        long_ratio=('interval', lambda x: (x > 300).mean()),
    ).reset_index()
    src_stats['interval_cv'] = src_stats['interval_std'] / (src_stats['interval_mean'] + 1e-10)
    src_stats['rhythm_stability'] = 1.0 / (1.0 + src_stats['interval_cv'])
    src_stats['rhythm_stability_risk'] = src_stats['long_ratio'] * src_stats['rhythm_stability']
    result = src_stats[['src', 'rhythm_stability_risk']].copy()
    result.columns = ['wallet_id', 'rhythm_stability_risk']
    return result
```

</details>

### #11 net_flow_direction_risk

- **维度**：行为序列
- **L2 验证**：IV=0.4483, KS=0.2728, 缺失率=0.00%, 冗余度=0.4204
- **L3 验证**：recall 0.9692→0.9699 (+0.0007), precision=0.9567
- **解释**：净流量绝对值 × (1 - 方向切换比例)。异常钱包净流量比例 0.87（正常 0.57），方向切换比例 0.21（正常 0.33）。
- **知识来源**：RULE_006（分散转入集中转出）、RULE_008（诈骗资金快进快出）、邮储专家（线上跑分链式转账）

<details>
<summary>特征代码</summary>

```python
def feature_net_flow_direction_risk(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    tx_src = tx_df[['src', 'timestamp', 'amount']].copy()
    tx_src.columns = ['wallet_id', 'timestamp', 'amount']
    tx_src['direction'] = -1
    tx_dst = tx_df[['dst', 'timestamp', 'amount']].copy()
    tx_dst.columns = ['wallet_id', 'timestamp', 'amount']
    tx_dst['direction'] = 1
    all_tx = pd.concat([tx_src, tx_dst]).sort_values(['wallet_id', 'timestamp'])
    def calc_flow_pattern(group):
        if len(group) < 2:
            return pd.Series({'net_flow_ratio': 0.0, 'switch_ratio': 0.0})
        directions = group['direction'].values
        amounts = group['amount'].abs().values * directions
        total_in = amounts[amounts > 0].sum()
        total_out = abs(amounts[amounts < 0].sum())
        total = total_in + total_out
        net_flow_ratio = (total_in - total_out) / total if total > 0 else 0.0
        switches = (directions[1:] != directions[:-1]).sum()
        switch_ratio = switches / (len(directions) - 1)
        return pd.Series({'net_flow_ratio': net_flow_ratio, 'switch_ratio': switch_ratio})
    wallet_counts = pd.concat([tx_df.groupby('src').size(), tx_df.groupby('dst').size()]).groupby(level=0).sum()
    active = wallet_counts[wallet_counts >= 2].index
    all_tx_active = all_tx[all_tx['wallet_id'].isin(active)]
    flow_pattern = all_tx_active.groupby('wallet_id').apply(calc_flow_pattern).reset_index()
    flow_pattern['net_flow_abs'] = flow_pattern['net_flow_ratio'].abs()
    flow_pattern['net_flow_direction_risk'] = flow_pattern['net_flow_abs'] * (1.0 - flow_pattern['switch_ratio'])
    return flow_pattern[['wallet_id', 'net_flow_direction_risk']].copy()
```

</details>

### #12 enterprise_counterparty_concentration

- **维度**：主体关联
- **L2 验证**：IV=0.3595, KS=0.2035, 缺失率=0.00%, 冗余度=0.2714
- **L3 验证**：recall 0.9699→0.9699 (+0.0000), precision=0.9559
- **解释**：企业钱包(wallet_type=2)交易对手占比 × (1 - 类型熵归一化)。异常 src 企业占比 0.5427（正常 0.4328），类型熵 1.0111（正常 1.2509）。
- **知识来源**：邮储专家（线上跑分链式转账）、裁判文书（多银行账户资金转移通道）、RULE_004（商户账户异常交易）

<details>
<summary>特征代码</summary>

```python
def feature_enterprise_counterparty_concentration(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    wallet_type_map = open_df[['wallet_id', 'wallet_type']].set_index('wallet_id')['wallet_type'].to_dict()
    tx = tx_df[['src', 'dst']].copy()
    tx['dst_wtype'] = tx['dst'].map(wallet_type_map)
    tx = tx.dropna(subset=['dst_wtype'])
    def calc_enterprise_concentration(group):
        counts = group.value_counts()
        total = len(group)
        enterprise_ratio = counts.get(2, 0) / total
        probs = counts / total
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
        norm_entropy = entropy / max_entropy
        return enterprise_ratio * (1 - norm_entropy)
    result = tx.groupby('src')['dst_wtype'].apply(calc_enterprise_concentration).reset_index()
    result.columns = ['wallet_id', 'enterprise_counterparty_concentration']
    return result
```

</details>

### #13 first_small_test_ratio (前向选择)

- **维度**：行为序列
- **L2 验证**：IV=1.7636, KS=0.4947, 缺失率=0.00%, 冗余度=0.4019
- **L3 验证**：recall 0.9699→0.9710 (+0.0011), precision=0.9574
- **解释**：首笔交易是否为小额测试（<=5元）。异常钱包首笔小额比例 41.5%（正常 4.7%，8.8 倍差异）。
- **知识来源**：RULE_002（沉睡期异常先小额测试）、邮储专家（验卡测试）、裁判文书（测试交易）

<details>
<summary>特征代码</summary>

```python
def feature_first_small_test_ratio(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    tx_sorted = tx_df.sort_values(['src', 'timestamp'])
    first_tx = tx_sorted.groupby('src').first().reset_index()
    first_tx['amount_abs'] = first_tx['amount'].abs()
    first_tx['first_small_test'] = (first_tx['amount_abs'] <= 5).astype(int)
    result = first_tx[['src', 'first_small_test']].copy()
    result.columns = ['wallet_id', 'first_small_test_ratio']
    return result
```

</details>

### #14 night_amount_short_term_risk

- **维度**：时间地域
- **L2 验证**：IV=1.0930, KS=0.2319, 缺失率=0.00%, 冗余度=0.2619
- **L3 验证**：recall 0.9710→0.9721 (+0.0011), precision=0.9601
- **解释**：深夜(22:00-5:59)交易金额占比 × 时间跨度风险 1/(天数+1)。异常钱包深夜金额占比 0.118（正常 0.053，2.2 倍），组合后差异 6.16 倍。
- **知识来源**：邮储专家（线上跑分凌晨操作）、RULE_015（网络赌博深夜交易）、裁判文书（夜间洗钱活动）

<details>
<summary>特征代码</summary>

```python
def feature_night_amount_short_term_risk(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    tx = tx_df.copy()
    tx['is_deep_night'] = (tx['hour'] >= 22) | (tx['hour'] < 6)
    tx['amount_abs'] = tx['amount'].abs()
    deep_night_amt = tx[tx['is_deep_night']].groupby('src')['amount_abs'].sum().reset_index()
    deep_night_amt.columns = ['src', 'deep_night_amount']
    total_amt = tx.groupby('src')['amount_abs'].sum().reset_index()
    total_amt.columns = ['src', 'total_amount']
    amt_ratio = total_amt.merge(deep_night_amt, on='src', how='left').fillna(0)
    amt_ratio['deep_night_amt_ratio'] = amt_ratio['deep_night_amount'] / (amt_ratio['total_amount'] + 1e-10)
    time_span = tx.groupby('src')['timestamp'].agg(['min', 'max']).reset_index()
    time_span['time_span_days'] = (time_span['max'] - time_span['min']).dt.days
    result = amt_ratio.merge(time_span[['src', 'time_span_days']], on='src')
    result['night_amount_short_term_risk'] = result['deep_night_amt_ratio'] * (1.0 / (result['time_span_days'] + 1.0))
    result = result[['src', 'night_amount_short_term_risk']].copy()
    result.columns = ['wallet_id', 'night_amount_short_term_risk']
    return result
```

</details>

### #15 amount_bin_skew_index

- **维度**：金额频次
- **L2 验证**：IV=4.8824, KS=0.7570, 缺失率=0.00%, 冗余度=0.6065
- **L3 验证**：recall 0.9721→0.9730 (+0.0009), precision=0.9627
- **解释**：极小额(<=1元)占比 / (中等金额(1000-50000)占比 + 大额(>=50000)占比 + 0.01)。异常钱包极小额占比 0.158（正常 0.009，18.49 倍），呈"L 型"偏态。
- **知识来源**：RULE_002（休眠账户小额测试）、邮储专家（跑分团伙小额测试）、裁判文书

<details>
<summary>特征代码</summary>

```python
def feature_amount_bin_skew_index(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()
    src_tiny = tx.groupby('src')['amount_abs'].apply(lambda x: (x <= 1).mean()).reset_index()
    src_tiny.columns = ['src', 'ultra_tiny_ratio']
    src_mid = tx.groupby('src')['amount_abs'].apply(lambda x: ((x >= 1000) & (x <= 50000)).mean()).reset_index()
    src_mid.columns = ['src', 'mid_amount_ratio']
    src_large = tx.groupby('src')['amount_abs'].apply(lambda x: (x >= 50000).mean()).reset_index()
    src_large.columns = ['src', 'large_amount_ratio']
    result = src_tiny.merge(src_mid, on='src').merge(src_large, on='src')
    result['amount_bin_skew_index'] = result['ultra_tiny_ratio'] / (result['mid_amount_ratio'] + result['large_amount_ratio'] + 0.01)
    result = result[['src', 'amount_bin_skew_index']].copy()
    result.columns = ['wallet_id', 'amount_bin_skew_index']
    return result
```

</details>

### #16 counterparty_type_amount_divergence

- **维度**：主体关联 + 金额频次
- **L2 验证**：IV=0.9489, KS=0.3341, 缺失率=0.00%, 冗余度=0.1175
- **L3 验证**：recall 0.9730→0.9735 (+0.0004), precision=0.9602
- **解释**：向个人钱包(type=1)金额占比 - 向商户钱包(type=3)金额占比。异常钱包背离度 0.2097（正常 0.1344，1.56 倍）。
- **知识来源**：邮储专家（线上跑分匿名→实名→银行链式转账）、裁判文书（多银行账户资金转移通道）

<details>
<summary>特征代码</summary>

```python
def feature_counterparty_type_amount_divergence(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    wallet_type = open_df[['wallet_id', 'wallet_type']].set_index('wallet_id')
    tx = tx_df.copy()
    tx['dst_wallet_type'] = tx['dst'].map(wallet_type['wallet_type'])
    tx_pos = tx[tx['amount'] > 0].copy()
    if len(tx_pos) == 0:
        return pd.DataFrame({'wallet_id': tx_df['src'].unique(), 'counterparty_type_amount_divergence': 0.0})
    total_amt = tx_pos.groupby('src')['amount'].sum().reset_index()
    total_amt.columns = ['wallet_id', 'total_amount']
    type1_amt = tx_pos[tx_pos['dst_wallet_type'] == 1].groupby('src')['amount'].sum().reset_index()
    type1_amt.columns = ['wallet_id', 'amount_to_type1']
    type3_amt = tx_pos[tx_pos['dst_wallet_type'] == 3].groupby('src')['amount'].sum().reset_index()
    type3_amt.columns = ['wallet_id', 'amount_to_type3']
    merged = pd.merge(total_amt, type1_amt, on='wallet_id', how='left').fillna(0)
    merged = pd.merge(merged, type3_amt, on='wallet_id', how='left').fillna(0)
    merged['type1_amount_ratio'] = merged['amount_to_type1'] / (merged['total_amount'] + 1e-10)
    merged['type3_amount_ratio'] = merged['amount_to_type3'] / (merged['total_amount'] + 1e-10)
    merged['counterparty_type_amount_divergence'] = merged['type1_amount_ratio'] - merged['type3_amount_ratio']
    return merged[['wallet_id', 'counterparty_type_amount_divergence']].copy()
```

</details>

### #17 tiny_amount_concentration (前向选择)

- **维度**：金额频次
- **L2 验证**：IV=4.8056, KS=0.7494, 缺失率=0.00%, 冗余度=0.3599
- **L3 验证**：recall 0.9735→0.9735 (+0.0000), precision=0.9600
- **解释**：金额<=1元交易中，最常见金额值的出现占比。异常钱包 tiny_amount_concentration 均值 0.1885（正常 0.0487，3.87 倍）。
- **知识来源**：RULE_002（休眠账户小额测试）、邮储专家（跑分团伙小额测试确认可用性）

<details>
<summary>特征代码</summary>

```python
def feature_tiny_amount_concentration(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    tx = tx_df.copy()
    tx['amount_abs'] = tx['amount'].abs()
    def calc_tiny_concentration(group):
        tiny = group[group <= 1]
        if len(tiny) == 0: return 0.0
        value_counts = tiny.value_counts()
        if len(value_counts) == 0: return 0.0
        return value_counts.iloc[0] / len(tiny)
    result = tx.groupby('src')['amount_abs'].apply(calc_tiny_concentration).reset_index()
    result.columns = ['wallet_id', 'tiny_amount_concentration']
    return result
```

</details>

### #18 time_concentration_risk_index

- **维度**：时间地域
- **L2 验证**：IV=4.1398, KS=0.8058, 缺失率=0.00%, 冗余度=0.6881
- **L3 验证**：recall 0.9735→0.9737 (+0.0002), precision=0.9621
- **解释**：最大小时交易占比 / 活跃小时数。异常钱包最大小时占比 0.515（正常 0.292，1.76 倍），活跃小时数 3.86（正常 8.45，0.46 倍），组合差异 3.17 倍。
- **知识来源**：邮储专家（跑分团伙集中时段交易）、RULE_006（分散转入集中转出）、裁判文书（短时间内完成大量交易）

<details>
<summary>特征代码</summary>

```python
def feature_time_concentration_risk_index(open_df, tx_df, close_df):
    import pandas as pd
    import numpy as np
    tx = tx_df.copy()
    hour_counts = tx.groupby(['src', 'hour']).size().reset_index(name='count')
    src_total = tx.groupby('src').size().reset_index(name='total')
    hour_pct = hour_counts.merge(src_total, on='src')
    hour_pct['pct'] = hour_pct['count'] / hour_pct['total']
    max_hour_pct = hour_pct.groupby('src')['pct'].max().reset_index()
    max_hour_pct.columns = ['src', 'max_hour_pct']
    active_hours = tx.groupby('src')['hour'].nunique().reset_index()
    active_hours.columns = ['src', 'active_hours']
    result = max_hour_pct.merge(active_hours, on='src')
    result['time_concentration_risk_index'] = result['max_hour_pct'] / (result['active_hours'] + 1e-10)
    result = result[['src', 'time_concentration_risk_index']].copy()
    result.columns = ['wallet_id', 'time_concentration_risk_index']
    return result
```

</details>

---

## 4. Stage 1 知识获取

### 知识来源 (13 个)

| 编号 | 来源 | 类型 |
|------|------|------|
| 01 | 洪湖抢红包虚拟币洗钱案 | 裁判文书 |
| 02 | 陈某枝比特币洗钱案 | 裁判文书 |
| 03 | 最高检虚拟货币洗钱典型案例 | 司法文件 |
| 04 | 上海跑分洗钱案 | 裁判文书 |
| 05 | 江苏检察院虚拟货币案例 | 司法文件 |
| 06 | 衡阳虚拟货币洗钱案 | 裁判文书 |
| 07 | 北京虚拟货币案 | 裁判文书 |
| 08 | 四川虚拟货币案 | 裁判文书 |
| 09 | 反洗钱法 (2025修订) | 法规 |
| 10 | 八部门虚拟货币通知 | 监管文件 |
| 11 | 广东虚拟货币团体标准 | 行业标准 |
| 12 | 慢雾 2024 区块链安全报告 | 行业报告 |
| 13 | Beosin 2025 Web3 安全报告 | 行业报告 |

### Key-clue Extraction 结果

- 提取风险因子总数：**88 个**
- 维度分布：金额频次、时间地域、行为序列、主体关联
- 压缩后 prompt 长度：6,018 字符（原始 417K → 25K 压缩）

> Stage 1 pipeline（特征生成）尚未运行。

---

## 5. Bug 修复记录

| 问题 | 原因 | 修复 | commit |
|------|------|------|--------|
| L3 precision ~0.55 | scale_pos_weight=307.8 (neg/pos) | 固定为 59.0 | 2afeb02 |
| IV=0.0 稀疏特征 | pd.qcut 对多零值特征分箱失败 | 零/非零 fallback 分箱 | 2afeb02 |
| DeepSeek 503 不重试 | InternalServerError 不在重试列表 | 加入 RETRYABLE_ERRORS | 2afeb02 |
