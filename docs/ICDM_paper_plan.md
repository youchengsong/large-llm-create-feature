# ICDM 2026 投稿计划

> 最后更新：2026-05-20

## 1. 会议信息

| 项目 | 详情 |
|------|------|
| 会议 | IEEE International Conference on Data Mining (ICDM 2026) |
| Abstract 截止 | 2026-05-30 (AoE) |
| Paper 截止 | 2026-06-06 (AoE) |
| 格式 | IEEE 2-column，10 页（含参考文献） |
| 目标 Track | **Applied Track**（单盲审稿，侧重实际应用） |

## 2. 核心 Idea

**将非结构化领域专家知识通过 LLM Agent 自动蒸馏为统计可验证的机器学习特征。**

聚焦场景：数字人民币反洗钱。

### 2.1 当前系统

- **领域知识注入**：邮储银行专家访谈（4 种洗钱模式）+ 309 份裁判文书（9 个子维度）→ 结构化后注入 Agent system prompt
- **SHAP 反馈闭环**：每轮计算 SHAP 重要性，反馈给 Agent 指导下一轮生成方向
- **候选缓冲池 + Forward Selection**：生成候选特征 → IV/KS 统计验证 → 前向选择去冗余
- **噪声基线校准**：100 个随机噪声特征的 IV/KS p95 分位数作为验证阈值
- **多轮迭代**：多 round × 多 island × 多 turn，带收敛检测

### 2.2 Novelty 诊断

现状：每个组件单独看都有 prior work。

| 组件 | 已有工作 | 我们的区别 |
|------|----------|------------|
| LLM 生成特征 | CAAFE (2023) | 他们也是让 LLM 写 pandas 代码 |
| SHAP 反馈 | DS-Agent (2024) | 他们也用模型反馈指导 LLM |
| 领域知识注入 | RAG 类工作 | 我们是结构化注入 prompt |
| 统计验证筛选 | OpenFE 的增量评估 | 我们用 IV/KS + 噪声校准 |
| 多轮迭代 | Agent loop 标准范式 | — |

**审稿人最可能的质疑**："This is essentially CAAFE + domain knowledge + SHAP feedback. The contribution is primarily engineering. What is the fundamental insight?"

### 2.3 中稿概率（当前状态）

| 目标 | 概率 | 原因 |
|------|------|------|
| ICDM Applied Track | 30-40% | 应用价值明确，但 novelty 偏 system/engineering |
| ICDM Research Track | 10-15% | 缺理论贡献 |
| 二区期刊 (ESWA, KBS) | 50-60% | 对 novelty 要求低，看重完整性 |

## 3. Novelty 增强方案（A + B）

### 3.1 方案 A：改 Framing 为「领域知识蒸馏」（纯写作，不改代码）

**当前叙事**（弱）：LLM 帮你做 feature engineering

**改后叙事**（强）：将领域专家知识通过 LLM 蒸馏为可计算的特征表示

| | 当前 | 改后 |
|---|------|------|
| 问题定义 | 如何用 LLM 自动生成特征 | 如何将非结构化领域知识自动转化为统计可验证的特征 |
| 核心矛盾 | 人工特征工程慢 | 领域知识与特征之间存在语义鸿沟 |
| 贡献定位 | 一个自动特征工程系统 | 一种知识蒸馏框架：非结构化专家知识 → ML-ready 特征 |
| 与 CAAFE 的区分 | 加了反馈和领域知识 | CAAFE 是数据驱动，我们是知识驱动+统计验证 |

这个改动的本质：不是改系统，是改论文讲的故事。同样的系统，换个问题定义，novelty 感知完全不同。

### 3.2 方案 B：领域知识引导的特征空间剪枝（改代码，核心 algorithmic contribution）

**这是补 novelty 的关键一步。** 给系统加一个审稿人一眼能看到的 algorithmic contribution。

#### 核心思想

在 LLM 生成候选特征之前，用领域知识构建一个 **「特征模板图」(Feature Template Graph)**，约束 LLM 的生成空间。

#### 具体设计

```
步骤 1: 构建特征模板图 G = (V, E)
  - 节点 V = 数据集的所有字段（如：交易金额、交易时间、账户类型...）
  - 边 E = 领域知识中共现的字段对
    - 来源 A（专家访谈）：提到的字段组合 → 加边
    - 来源 B（裁判文书）：判决书中涉及的字段组合 → 加边
    - 来源 C（检测规则）：每条规则涉及的字段 → 两两加边
  - 边权重 = 该组合在领域知识中出现的频次

步骤 2: 约束 LLM 生成
  - LLM 生成特征时，prompt 中注入当前建议探索的字段组合（从图中采样）
  - 高权重边 → 优先探索
  - 已探索过的边 → 降低优先级
  - 未连接的字段对 → 不建议但不禁止（允许 LLM 发现意外组合）

步骤 3: 动态更新图
  - 每轮结束后，通过验证的特征涉及的字段对 → 加强边权重
  - 被淘汰的特征涉及的字段对 → 减弱边权重
  - SHAP 高贡献的字段 → 以该字段为中心扩展邻域探索
```

#### 为什么这个有 novelty

1. **可量化的搜索空间缩减**：假设数据集有 n 个字段，无约束的 k 阶特征组合是 O(n^k)；图约束下是 O(|E| * k)，这是可以写公式证明的
2. **领域知识的结构化利用**：不是把知识当文本塞进 prompt（所有 RAG 都这么做），而是把知识转化为搜索空间的拓扑结构
3. **自适应**：图权重随验证结果动态更新，形成"知识引导 + 数据反馈"的双重约束
4. **与 CAAFE/DS-Agent 的明确区分**：他们没有对生成空间的结构化约束，是无引导的随机探索

#### 实现量估算

- 构建初始图：解析现有 domain_knowledge.py 中的领域知识，提取字段共现关系 → 约 100-200 行代码
- 图采样 + prompt 注入：每轮从图中采样推荐字段组合，注入 Agent prompt → 约 50-100 行代码
- 动态更新：每轮根据验证结果更新边权重 → 约 50 行代码
- **总计约 200-350 行代码，不影响现有架构**

#### 论文中怎么写

这个机制可以占 Section 4 的核心位置（约 1.5-2 页），包含：
- 图的形式化定义
- 搜索空间缩减的理论分析
- 动态更新算法伪代码
- 消融实验：有图 vs 无图的对比

## 4. 实验设计

### 4.1 策略：单数据集 + 领域深绑定 + 多 baseline

**决策**：不补公开数据集，聚焦数字人民币反洗钱单一场景。

理由：
- 核心卖点就是深度领域绑定（专家访谈 + 裁判文书），换数据集会稀释优势
- Applied Track 接受 domain-specific 研究
- 审稿人质疑泛化性时的回应："本文聚焦数字人民币反洗钱这一高价值监管场景，框架具有通用性（领域知识模块可替换），跨领域验证留作后续工作"

### 4.2 Baseline 对比

| 方法 | 出处 | 核心思路 | 对比意义 |
|------|------|----------|----------|
| **OpenFE** | NeurIPS 2022 | 枚举 + 增量评估 | 数据驱动 SOTA |
| **DIFER** | ICLR 2024 | 强化学习搜索特征变换 | RL-based SOTA |
| **AutoFeat** | 经典 | 多项式展开 + 选择 | 经典 baseline |
| **CAAFE** | 2023 | LLM 单轮生成 | 最直接对比（无反馈、无领域知识） |
| **手工特征** | 领域专家 | 人工设计 | 证明 Agent 接近/超越人类专家 |
| **Random** | — | 随机特征 | 下界 |

如果某些 baseline 跑赢了我们：强调"我们在保持领域可解释性的同时接近 SOTA 数据驱动方法的性能，且无需人工特征设计"。

### 4.3 消融实验

| 配置 | 说明 | 验证什么 |
|------|------|----------|
| Full Agent | 完整系统 | 上界 |
| - Domain Knowledge | 去掉领域知识注入 | 领域知识的贡献 |
| - SHAP Feedback | 去掉 SHAP 反馈 | 反馈闭环的贡献 |
| - Feature Template Graph | 去掉特征模板图（方案 B） | 搜索空间剪枝的贡献 |
| - Candidate Pool + FS | 去掉候选池和前向选择 | 特征筛选的贡献 |
| - Knowledge & SHAP | 同时去掉知识和反馈 | 退化为 CAAFE-like |

### 4.4 其他实验

- **多模型验证**：XGBoost / LightGBM / RandomForest / LogisticRegression，证明特征模型无关
- **稳定性分析**：同配置多次运行，报告指标均值和标准差
- **收敛性分析**：round vs cumulative metric improvement 曲线
- **Case Study**：展示 top-5 生成特征的代码、领域含义、与哪条领域知识对齐

## 5. 优先级与执行顺序

| 顺序 | 行动 | 性质 | 影响 |
|------|------|------|------|
| 1 | 打磨系统本身 | 代码 | 基础 |
| 2 | 实现方案 B（特征模板图） | 代码，~300 行 | **核心 novelty 来源** |
| 3 | 跑全部实验（含方案 B 的消融） | 实验 | 数据支撑 |
| 4 | 方案 A（改 framing） | 写作 | 提升 novelty 感知 |
| 5 | 补强 baseline 对比 | 代码 + 实验 | 实验说服力 |
| 6 | 写论文 | 写作 | — |

### 做完 A + B 后的中稿概率

| 目标 | 概率 | 变化 |
|------|------|------|
| ICDM Applied Track | **40-50%** | ↑ from 30-40% |
| ICDM Research Track | 20-25% | ↑ from 10-15% |
| 二区期刊 | 60-70% | ↑ from 50-60% |

提升原因：方案 B 给了一个明确的 algorithmic contribution（特征模板图 + 搜索空间剪枝），不再是纯 engineering；方案 A 让审稿人第一眼就理解 novelty 在哪。

## 6. 论文结构草案

```
1. Introduction
   - 数字人民币反洗钱的挑战
   - 核心矛盾：领域专家知识丰富但难以转化为可计算特征
   - 本文贡献：(1) 领域知识蒸馏框架 (2) 特征模板图剪枝 (3) 数字人民币场景验证

2. Related Work
   - 自动特征工程：AutoFeat, OpenFE, DIFER
   - LLM for Data Science：CAAFE, DS-Agent
   - 反洗钱/欺诈检测

3. Problem Formulation
   - 形式化：领域知识 → 特征的映射问题
   - 特征空间搜索的复杂度分析

4. Methodology
   4.1 System Overview（架构图）
   4.2 Domain Knowledge Structuring（专家访谈 + 裁判文书的结构化）
   4.3 Feature Template Graph（特征模板图构建与动态更新）  ← 方案 B，核心贡献
   4.4 LLM Agent with SHAP Feedback Loop
   4.5 Noise-Calibrated Statistical Validation
   4.6 Forward Selection with Candidate Pool

5. Experiments
   5.1 Dataset & Setup
   5.2 Comparison with Baselines
   5.3 Ablation Studies（含特征模板图的消融）
   5.4 Multi-model Validation
   5.5 Stability & Convergence Analysis
   5.6 Case Study

6. Conclusion & Future Work
```

## 7. 相关文献

- Hollmann et al., "CAAFE: LLM-based Feature Engineering for Tabular Data" (2023)
- Zhang et al., "OpenFE: Automated Feature Generation with Expert-level Performance" (NeurIPS 2022)
- Zhu et al., "DIFER: Differentiable Automated Feature Engineering" (ICLR 2024)
- DS-Agent: "Automated Data Science by Empowering LLMs" (2024)
- SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (NeurIPS 2017)

## 8. 备注

- 实验代码已就绪：`scripts/run_paper_experiments.py`，`src/evaluation/multi_model.py`
- 方案 B 的消融需要在实验脚本中新增一个配置（no_template_graph）
- lightgbm 依赖可能需要在服务器上安装
