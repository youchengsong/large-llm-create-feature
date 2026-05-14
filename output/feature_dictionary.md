# 特征字典

生成时间: 2026-05-14T06:17:17.808483

通过验证的特征总数: **2**

## 维度分布

| 维度 | 特征数量 |
|------|---------|
| 主体关联 | 2 |

---

## 特征详情

### 1. multi_cert_phone_ratio

- **维度**: 主体关联
- **综合评分**: 0.5000

#### 可解释性说明

根据薅羊毛模式，批量注册4类钱包时，同一身份证/手机号会关联多个钱包。通过计算每个身份证关联的钱包数量与每个手机号关联的钱包数量的比值，可以识别出使用同一身份信息批量注册多个钱包的可疑行为。

#### 统计指标

| 指标 | 值 |
|------|-----|
| 互信息 (MI) | 0.012577 |
| Cramér's V | 0.074314 |
| 冗余度 | 0.000000 |
| Bootstrap p 值 | 4.82e-13 |

#### 模型指标

| 指标 | 值 |
|------|-----|
| ΔPR-AUC | 0.009323 |
| SHAP 排名 | N/A |
| 平均 |SHAP| | N/A |

#### Python 代码

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

---

### 2. cert_phone_multi_wallet_count

- **维度**: 主体关联
- **综合评分**: 0.4250

#### 可解释性说明

在薅羊毛洗钱模式中，犯罪分子常使用同一身份证或手机号批量注册多个4类钱包。统计每个钱包的开户证件(open_cert)和手机号(open_tel)关联的钱包数量，可以识别出批量开户的异常账户。

#### 统计指标

| 指标 | 值 |
|------|-----|
| 互信息 (MI) | 0.035266 |
| Cramér's V | 0.083493 |
| 冗余度 | 0.029568 |
| Bootstrap p 值 | 1.18e-47 |

#### 模型指标

| 指标 | 值 |
|------|-----|
| ΔPR-AUC | 0.005142 |
| SHAP 排名 | N/A |
| 平均 |SHAP| | N/A |

#### Python 代码

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
