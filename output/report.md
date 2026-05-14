# LLM 特征提取验证报告

> 生成时间: 2026-05-14 06:17:22

---

## 1. 搜索过程摘要

- **总轮数**: 4
- **通过验证的特征总数**: 2
- **收敛**: 是（末轮无新特征）

### 各岛屿贡献

| 岛屿（维度） | 通过特征数 |
|---|---|
| 主体关联 | 2 |
| 金额频次 | 0 |
| 时间地域 | 0 |
| 行为序列 | 0 |

### 各轮详情

| 轮次 | 时间 | 新增特征 | 各岛屿候选/通过 |
|---|---|---|---|
| 1 | 2026-05-14 00:39:44 | 1 | 金额频次: 3/0; 时间地域: 3/0; 行为序列: 3/0; 主体关联: 3/1 |
| 2 | 2026-05-14 00:44:59 | 1 | 金额频次: 3/0; 时间地域: 3/0; 行为序列: 3/0; 主体关联: 3/1 |
| 3 | 2026-05-14 00:51:39 | 0 | 金额频次: 3/0; 时间地域: 3/0; 行为序列: 3/0; 主体关联: 3/0 |
| 4 | 2026-05-14 00:59:03 | 0 | 金额频次: 3/0; 时间地域: 2/0; 行为序列: 3/0; 主体关联: 3/0 |

---

## 2. 特征字典

- **特征总数**: 2
- **维度分布**: 主体关联: 2

| 排名 | 特征名 | 维度 | 综合得分 | MI | Cramér's V | ΔPR-AUC | SHAP排名 |
|---|---|---|---|---|---|---|---|
| 1 | multi_cert_phone_ratio | 主体关联 | 0.5000 | 0.0126 | 0.0743 | 0.0093 | None |
| 2 | cert_phone_multi_wallet_count | 主体关联 | 0.4250 | 0.0353 | 0.0835 | 0.0051 | None |

### 特征详情

#### 1. multi_cert_phone_ratio

- **维度**: 主体关联
- **综合得分**: 0.5000
- **说明**: 根据薅羊毛模式，批量注册4类钱包时，同一身份证/手机号会关联多个钱包。通过计算每个身份证关联的钱包数量与每个手机号关联的钱包数量的比值，可以识别出使用同一身份信息批量注册多个钱包的可疑行为。

```python
cert_phone = open_df[['wallet_id', 'wallet_open_cert', 'wallet_open_tel']].dropna(subset=['wallet_open_cert', 'wallet_open_tel'])

# 计算每个身份证关联的钱包数
cert_counts = cert_phone.groupby('wallet_open_cert')['wallet_id'].nunique().reset_index()
cert_counts.columns = ['wallet_open_cert', 'cert_wallet_count']

# 计算每个手机号关联的钱包数
phone_counts = cert_phone.groupby('wallet_open_tel')['wallet_id'].nunique().reset_index()
phone_counts.columns = ['wallet_open_tel', 'phone_wallet_count']

# 合并到原数据
merged = cert_phone.merge(cert_counts, on='wallet_open_cert', how='left')
merged = merged.merge(phone_counts, on='wallet_open_tel', how='left')

# 计算比值，避免除零
merged['cert_phone_ratio'] = (merged['cert_wallet_count'] / merged['phone_wallet_count'].clip(lower=1)).fillna(0)

result = merged[['wallet_id', 'cert_phone_ratio']]
```

#### 2. cert_phone_multi_wallet_count

- **维度**: 主体关联
- **综合得分**: 0.4250
- **说明**: 在薅羊毛洗钱模式中，犯罪分子常使用同一身份证或手机号批量注册多个4类钱包。统计每个钱包的开户证件(open_cert)和手机号(open_tel)关联的钱包数量，可以识别出批量开户的异常账户。

```python
import pandas as pd
import numpy as np

# 统计每个开户证件关联的钱包数
cert_counts = open_df.groupby('wallet_open_cert')['wallet_id'].nunique().reset_index()
cert_counts.columns = ['wallet_open_cert', 'cert_wallet_count']

# 统计每个手机号关联的钱包数
tel_counts = open_df.groupby('wallet_open_tel')['wallet_id'].nunique().reset_index()
tel_counts.columns = ['wallet_open_tel', 'tel_wallet_count']

# 合并到开户表
result = open_df[['wallet_id', 'wallet_open_cert', 'wallet_open_tel']].copy()
result = result.merge(cert_counts, on='wallet_open_cert', how='left')
result = result.merge(tel_counts, on='wallet_open_tel', how='left')

# 取两个计数的最大值作为最终特征
result['cert_phone_multi_wallet_count'] = result[['cert_wallet_count', 'tel_wallet_count']].max(axis=1)
result = result[['wallet_id', 'cert_phone_multi_wallet_count']]
result['cert_phone_multi_wallet_count'] = result['cert_phone_multi_wallet_count'].fillna(1)

result = result
```

---

## 3. 增量贡献实验结果

### PR-AUC 对比

| 实验 | PR-AUC (均值 ± 标准差) |
|---|---|
| 实验1: 仅人工基线特征 | 0.0000 ± 0.0000 |
| 实验2: 人工 + LLM 特征 | 0.0363 ± 0.0251 |
| 实验3: 仅 LLM 特征 | 0.0363 ± 0.0251 |
| 实验4: 人工 + 随机对照 | 0.0009 ± 0.0004 |

### 关键指标

- **增量 ΔPR-AUC（实验2 - 实验1）**: 0.0363
- **LLM 独立 PR-AUC**: 0.0363
- **随机对照 ΔPR-AUC（实验4 - 实验1）**: 0.0009
- **最关键特征（消融）**: cert_phone_multi_wallet_count_dst

### 消融实验（逐个移除 LLM 特征）

| 特征 | 移除后 PR-AUC | 贡献（ΔPR-AUC） |
|---|---|---|
| cert_phone_multi_wallet_count_dst | 0.0134 | 0.0229 |
| cert_phone_ratio_src | 0.0478 | -0.0115 |
| cert_phone_ratio_dst | 0.0466 | -0.0103 |
| cert_phone_multi_wallet_count_src | 0.0365 | -0.0002 |

---

## 4. L4 稳定性验证

- **Nogueira 稳定性指数**: 0.0000
- **Bootstrap 轮数**: 10
- **阈值 > 0.8**: 未通过 ✗

### 特征出现频率（Top-20 排名中）

| 特征 | 出现次数 | 出现率 |
|---|---|---|
| cert_phone_multi_wallet_count_src | 10/10 | 100.00% |
| cert_phone_ratio_src | 10/10 | 100.00% |
| cert_phone_ratio_dst | 10/10 | 100.00% |
| cert_phone_multi_wallet_count_dst | 10/10 | 100.00% |

---

## 5. 可视化图表

### SHAP 特征重要性

![SHAP 特征重要性](/home/ubuntu/Projects/大模型提取特征/output/plots/shap_summary.png)

### 实验 PR-AUC 对比

![实验 PR-AUC 对比](/home/ubuntu/Projects/大模型提取特征/output/plots/experiment_comparison.png)

### 消融实验热力图

![消融实验热力图](/home/ubuntu/Projects/大模型提取特征/output/plots/ablation_heatmap.png)

### 稳定性 Bootstrap 分布

![稳定性 Bootstrap 分布](/home/ubuntu/Projects/大模型提取特征/output/plots/stability_distribution.png)

### 岛屿贡献分布

![岛屿贡献分布](/home/ubuntu/Projects/大模型提取特征/output/plots/island_contribution.png)
