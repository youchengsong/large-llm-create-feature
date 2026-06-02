# 候选 Baseline 和公开数据集调研

> 调研日期: 2026-06-02
> 目的: 为 ICDM 2026 KnowFeat 论文补充 baseline 和数据集

---

## 一、候选 Baseline 方法

### 已有 Baseline
No-FE, OpenFE (ICML 2023), AutoFeat (2019), CAAFE (NeurIPS 2023), FeatLLM (ICML 2024)

### Tier 1: 强烈建议添加（LLM-based，直接竞品）

| 方法 | 会议/年份 | 核心思想 | 代码 | 为什么重要 |
|------|----------|---------|------|-----------|
| **OCTree** | NeurIPS 2024 | LLM推理 + 决策树反馈迭代优化特征生成规则 | [GitHub](https://github.com/jaehyun513/OCTree) | 最新顶会LLM-FE方法，审稿人会期望对比 |
| **LLM-FE** | arXiv 2025 (投稿中) | 将FE建模为程序搜索，LLM+进化优化 | [GitHub](https://github.com/nikhilsab/LLMFE) | 与我们最相似的竞品，覆盖 21+ 数据集，对比了CAAFE/FeatLLM/OpenFE |

### Tier 2: 建议添加（传统AutoFE，增加对比广度）

| 方法 | 会议/年份 | 核心思想 | 代码 | 备注 |
|------|----------|---------|------|------|
| **DIFER** | AutoML-Conf 2022 | 梯度可微的特征工程（encoder-predictor-decoder） | [GitHub](https://github.com/PasaLab/DIFER) | 代表不同范式（梯度 vs LLM） |
| **NFS** | ICDM 2019 | RNN控制器+RL搜索特征变换 | [GitHub](https://github.com/TjuJianyu/NFS) | 投稿 ICDM，审稿人可能熟悉 |
| **FETCH** | IJCAI 2023 | RL + 特征预评估模型 | 代码不确定 | OpenFE 论文中作为 baseline |

### Tier 3: 可选（经典方法，引用高但代码难跑）

| 方法 | 会议/年份 | 核心思想 | 代码 | 备注 |
|------|----------|---------|------|------|
| **AutoCross** | KDD 2019 | Beam search 搜索高阶交叉特征 | 部分: [GitHub](https://github.com/sfu-db/FeatAug) | OpenFE baseline |
| **FCTree** | ~2019 | 决策树分裂节点做特征交叉 | 无公开代码 | OpenFE baseline，无法复现 |
| **SAFE** | arXiv 2020 | 分阶段特征生成-选择-组合 | 无官方代码 | OpenFE baseline，工业场景 |
| **ExploreKit** | ICDM 2016 | 元学习预测候选特征有效性 | [GitHub](https://github.com/giladkatz/ExploreKit) (Java) | 经典但 Java 实现，难集成 |
| **LFE** | IJCAI 2017 | 神经网络从历史FE经验学习 | 无代码 | 无法复现 |

### Tier 4: 最新工作（2025-2026，参考）

| 方法 | 时间 | 核心思想 | 备注 |
|------|------|---------|------|
| **FEBP** | ICLR 2025 | LLM + 逆波兰表达式迭代生成特征 | 代码状态不确定 |
| **SCOPE-FE** | arXiv 2026.04 | 算子探测 + 特征聚类缩小搜索空间 | 代码未发布 |
| **CoFEH** | arXiv 2026.02 | LLM + 贝叶斯超参优化 | 太新 |

---

## 二、候选公开数据集

### 各论文使用的数据集汇总

| 数据集 | CAAFE | FeatLLM | OpenFE | LLM-FE | OCTree | KnowFeat(ours) | 推荐度 |
|--------|:-----:|:-------:|:------:|:------:|:------:|:--------------:|:------:|
| credit_g | ✓ | ✓ | | ✓ | | ✓ | 已有 |
| diabetes | ✓ | ✓ | | | | ✓ | 已有 |
| blood_transfusion | | ✓ | | ✓ | | ✓ | 已有 |
| heart | | ✓ | | ✓ | | ✓ | 已有 |
| bank_marketing | | ✓ | | ✓ | ✓ | ✓ | 已有 |
| breast_w | ✓ | | | ✓ | | ✓ | 已有 |
| **adult** | | ✓ | | ✓ | | | ⭐⭐⭐ |
| **car** | | ✓ | | ✓ | | | ⭐⭐ |
| **balance-scale** | ✓ | | | ✓ | | | ⭐⭐ |
| **cmc** | ✓ | | | ✓ | | | ⭐⭐ |
| **tic-tac-toe** | ✓ | | | ✓ | | | ⭐ |
| **eucalyptus** | ✓ | | | ✓ | | | ⭐ |
| **pc1** | ✓ | | | ✓ | | | ⭐ |
| communities | | ✓ | | ✓ | | | ⭐ |
| myocardial | | ✓ | | ✓ | | | ⭐ |
| vehicle | | | ✓ | ✓ | | | ⭐ |
| covertype | | | ✓ | | ✓ | | ⭐ |

### 推荐添加的数据集（优先级排序）

#### 优先级 1: 强烈建议

| 数据集 | OpenML ID | 样本数 | 特征数 | 任务 | 出现论文 | 理由 |
|--------|-----------|--------|--------|------|---------|------|
| **adult** | 1590 | 48,842 | 14 | 二分类 | FeatLLM, LLM-FE | 标准benchmark，大规模，所有人都认 |

#### 优先级 2: 建议添加

| 数据集 | OpenML ID | 样本数 | 特征数 | 任务 | 出现论文 | 理由 |
|--------|-----------|--------|--------|------|---------|------|
| **car** | — | 1,728 | 6 | 多分类(4) | FeatLLM, LLM-FE | 小数据集跑得快，多分类任务丰富实验 |
| **balance-scale** | 11 | 625 | 4 | 多分类(3) | CAAFE, LLM-FE | 覆盖CAAFE数据集，特征极少 |
| **cmc** | 23 | 1,473 | 9 | 多分类(3) | CAAFE, LLM-FE | 覆盖CAAFE数据集 |

#### 优先级 3: 可选

| 数据集 | OpenML ID | 样本数 | 特征数 | 任务 | 出现论文 |
|--------|-----------|--------|--------|------|---------|
| tic-tac-toe | 50 | 958 | 9 | 二分类 | CAAFE, LLM-FE |
| eucalyptus | 188 | 736 | 19 | 多分类(5) | CAAFE, LLM-FE |
| pc1 | 1068 | 1,109 | 21 | 二分类 | CAAFE, LLM-FE |
| communities | — | 1,994 | 103 | 二分类 | FeatLLM, LLM-FE |
| myocardial | — | 1,700 | 111 | 二分类 | FeatLLM, LLM-FE |

---

## 三、建议方案

### 最低成本方案（时间紧）

- **Baseline**: 只加 **OCTree**（NeurIPS 2024，代码可用，1个就够）
- **数据集**: 只加 **adult**（标准 benchmark，大家都用）
- 总新增工作量: 1 baseline × 8 datasets + 现有 6 baselines × 1 dataset = 14 次实验

### 推荐方案（平衡覆盖和工作量）

- **Baseline**: 加 **OCTree** + **LLM-FE**
- **数据集**: 加 **adult** + **car**
- 总新增工作量: 2 baselines × 9 datasets + 现有 6 baselines × 2 datasets = 30 次实验

### 完整方案（时间充裕）

- **Baseline**: 加 **OCTree** + **LLM-FE** + **DIFER**
- **数据集**: 加 **adult** + **car** + **balance-scale** + **cmc**
- 总新增工作量较大，但覆盖了所有主要竞品和数据集

---

## 四、各方法论文引用

```
OCTree:   Kim et al., "OCTree: Optimized Feature Generation for Tabular Data via LLMs with Decision Tree Reasoning", NeurIPS 2024
LLM-FE:  Sab et al., "LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers", arXiv 2025
DIFER:   Zhu et al., "DIFER: Differentiable Automated Feature Engineering", AutoML-Conf 2022
NFS:     Chen et al., "Neural Feature Search: A Neural Architecture for Automated Feature Engineering", ICDM 2019
FETCH:   Zhang et al., "FETCH: Feature Extraction for Tabular Data via Conditional Hugging", IJCAI 2023
FEBP:    "Automated Feature Engineering by Prompting", ICLR 2025
```
