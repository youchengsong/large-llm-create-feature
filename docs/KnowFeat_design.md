# KnowFeat 系统设计方案（工作参考文档）

> 最后更新：2026-05-27
> 本文件为开发参考，v4.html 为导师展示版本。

## 1. 论文信息

| 项目 | 详情 |
|------|------|
| Title | KnowFeat: Distilling Domain Expert Knowledge into Verified Features via LLM Agents for Digital Yuan Anti-Money Laundering |
| 会议 | ICDM 2026 Applied Track（单盲审稿，10 页 IEEE 2-column） |
| Deadline | 2026-06-07 (AoE) |
| 核心贡献 | 将非结构化领域专家知识通过 LLM Agent 蒸馏为统计可验证的 ML 特征 |

## 2. 系统总览

### 2.1 两阶段知识注入

```
Stage 1: 加密货币通用特征生成
  - 数据: BABD-13（148 个地址级特征作为基线）
  - 知识: 5 个加密货币洗钱案例 → Key-clue Extractor → 结构化风险因子
  - 输出: 有效特征集 A

Stage 2: 数字人民币专属特征生成
  - 数据: SimECNY（数字人民币模拟交易）
  - 知识: 银行专家意见 + 309 份裁判文书 → Key-clue Extractor → 结构化风险因子
  - 基线: 原始特征 + Feature Set A（直接作为已有列拼入数据集）
  - 输出: 有效特征集 B
```

### 2.2 核心设计理念

- **Recall 优先**：所有验证以 Recall 提升为必要条件，Precision 只设下限（≥ 90%）
- **动态基线**：每接受一个特征，有效特征集立即更新，后续评估基于最新状态
- **知识结构化**：非结构化文档经 Key-clue Extractor 转为结构化风险因子
- **特征可溯源**：每个特征自带 Provenance Card，天然具备可解释性

## 3. Key-clue Extractor

独立的 LLM 预处理步骤，将非结构化领域文档转化为结构化风险因子列表。

### 输出 Schema

```json
{
  "source": "湖南衡阳9.15案",
  "risk_factors": [
    {
      "pattern": "多层级虚拟币交易网络分散资金流向",
      "dimension": ["主体关联", "行为序列"],
      "observable_signals": ["短时间内资金经过3层以上地址", "每层地址存活时间<24h"],
      "data_hint": "可从 fan_out_depth, address_lifetime 构造"
    }
  ]
}
```

### 字段说明

| 字段 | 说明 |
|------|------|
| pattern | 自然语言描述的洗钱模式 |
| dimension | 归属维度（可多选） |
| observable_signals | 可观测的数据信号 |
| data_hint | 提示可能用到的原始字段 |

### 知识来源

- **Stage 1**: 5 个加密货币洗钱新闻案例（湖南衡阳 400 亿案、北京 8 亿案、洪湖抢红包案、四川 USDT+黄金案、陈某枝比特币案）
- **Stage 2**: 银行风控专家意见（4 种数币洗钱模式）+ 309 份裁判文书（9 个子维度）

## 4. Feature Agent

### 4.1 架构：ReAct 范式

Agent 在每一步先思考 → 调用工具 → 根据观测决定下一步。通过 Function Calling 与环境交互。

### 4.2 工具箱

| 工具 | 作用 |
|------|------|
| explore_data | 在 DataFrame 上执行查询，探索数据分布 |
| generate_feature | 提交特征代码+名称+维度+溯源信息 |
| execute_sandbox | 在隔离沙盒中执行代码，触发 L1→L2→L3 验证 |

### 4.3 generate_feature 参数

| 参数 | 说明 |
|------|------|
| feature_name | 特征名 |
| dimension | 所属维度（支持多维度标签） |
| code | Python 代码，签名 `def feature_xxx(open_df, tx_df, close_df) → pd.DataFrame` |
| explanation | 特征含义解释 |
| reasoning | 设计推理过程 |
| knowledge_source | 知识溯源（来自哪个案例/规则/专家意见） |
| data_fields_used | 使用的原始数据字段列表 |

**Provenance Card** = explanation + reasoning + knowledge_source + data_fields_used，自动组成特征的溯源卡片。

### 4.4 四维度分组

| 维度 | 关注点 |
|------|--------|
| 金额频次 | 交易金额分布、频率异常 |
| 时间地域 | 时间模式、地域分布 |
| 行为序列 | 交易模式序列、行为链 |
| 主体关联 | 账户关系、网络结构 |

特征允许声明多维度标签（跨维度交叉特征），SHAP 覆盖率统计时同时计入多个维度。

## 5. 三级验证

### L1 沙盒执行
- E2B 隔离沙盒中运行 Agent 提交的代码
- 检查：是否可执行、输出 shape 是否正确、是否超时（30s）
- 失败 → Agent 自动修复（最多 2 次）

### L2 统计质量（噪声校准）
- 生成 100 个随机噪声特征，取 IV 和 KS 的 p95 分位数作为动态阈值
- 通过条件：IV > 阈值 AND KS > 阈值 AND 冗余度 < 0.7 AND 缺失率 < 30%
- 未通过 → 淘汰

### L3 模型效果（交叉验证 delta-recall）
- 5-fold StratifiedKFold CV
- 通过条件：Delta Recall > 0 AND Precision ≥ 90%
- 通过 → 加入有效特征集
- 未通过 → 进入候选特征集（等待前向选择评估组合效应）

### 前向选择（Forward Selection）
- 每轮末尾，从候选特征集中贪心搜索组合
- 逐个评估候选：加入后 Delta Recall > 0 → 接受
- 解决"单特征增量不够但组合有效"的问题

## 6. 搜索策略

### SHAP 反馈循环
- 每轮前向选择后计算 SHAP 重要性
- 按维度贡献的反比分配下一轮预算（薄弱维度获更多名额）

### 失败记忆（FailureMemory）
- 结构化记录每个失败特征的名称、代码、失败原因
- 每轮注入 prompt，避免重复犯错

### 跨维度知识传播（migrate_best）
- 高分特征的信息传播到其他维度组的 prompt 中
- 作为灵感来源，可能激发跨维度交叉特征

### 收敛终止条件
| 条件 | 判定标准 |
|------|----------|
| 收敛终止 | 连续 2 轮无新特征被接受 |
| 目标达成 | 已接受特征数达上限（默认 30） |
| 轮次上限 | 完成 10 轮 |

## 7. 评价指标

### 7.1 核心指标

| 指标 | 角色 | 说明 |
|------|------|------|
| **Recall** | 核心指标 | 所有实验对比以 Recall 提升为主要依据 |
| **Precision** | 约束条件 | 要求 ≥ 90%，防止以过高误报换召回 |
| **Top-K Capture Rate** | 业务指标 | CR@K = (Top K% 中的洗钱数) / (总洗钱数)，K = 1%, 3%, 5% |
| **F1** | 辅助参考 | 综合参考 |

### 7.2 Wallet_type 分场景评估

按 `src_wallet_type` / `dst_wallet_type` 切分测试集：

| wallet_type | 值 | 典型洗钱模式 |
|-------------|-----|-------------|
| Corporate wallet | 1 | 对公账户大额转出、虚假贸易 |
| Personal wallet | 2 | 跑分、小额分散、夜间高频 |
| Merchant wallet | 3 | 收款码跑分、虚假流水、佣金抽成 |

### 7.3 自动模型选择与训练

特征生成完成后，系统根据数据特性从 ModelSet 的 11 个候选模型中自动选择最优模型：
- 候选：catboost, ga2m, scorecard, acxgb, wav_xgb, ensemble, lgbm, resample_boost, gbdt_fm, dnn_res, mlp
- 选择依据：数据规模、特征类型、类别不平衡程度等
- 使用 5-fold StratifiedKFold CV 评估各模型表现，自动选取最优模型进行训练和推理

## 8. 实验设计

### 8.1 核心实验

| 实验 | 内容 | 目的 |
|------|------|------|
| Recall + CR@K 对比 | baseline vs +KnowFeat，多模型 | 主表：证明特征有效 |
| Wallet_type 分场景 | 三种钱包类型分别计算指标 | 场景差异分析 |
| 自动模型选择 | 11 个候选模型中自动选最优 | 端到端自动化能力 |

### 8.2 消融实验

| 配置 | 验证什么 |
|------|----------|
| Full KnowFeat | 上界 |
| − Key-clue Extractor | 知识结构化的贡献（直接用原文 vs 结构化风险因子） |
| − SHAP Feedback | 反馈闭环的贡献 |
| − Stage 1 (only Stage 2) | 两阶段递进的贡献 |
| − Candidate Pool + FS | 前向选择的贡献 |

### 8.3 Baseline 对比

| 方法 | 出处 | 对比意义 |
|------|------|----------|
| OpenFE | NeurIPS 2022 | 数据驱动 SOTA |
| CAAFE | 2023 | LLM 单轮生成，最直接对比 |
| 手工特征 | 领域专家 | 证明接近/超越人工 |
| Random | — | 下界 |

## 9. 数据资产

### Stage 1
- **BABD-13**：Bitcoin Address Behavior Dataset，148 个地址级特征（Kaggle）
- **领域知识**：5 个加密货币洗钱新闻案例

### Stage 2
- **SimECNY**：数字人民币模拟交易数据（`inputs/SimECNY/`，`data/`）
  - open_df: 100,000 钱包
  - tx_df: 1,384,512 笔交易（正样本率 0.32%）
  - close_df: 500 条销户
- **领域知识**：
  - 银行专家意见（`inputs/domain_knowledge/邮储专家访谈_数币洗钱模式.docx`）
  - 309 份裁判文书（`裁判文书docx/`）
  - 38 风险指标（`inputs/domain_knowledge/风险分类表_38指标.xlsx`）
  - 15 条检测规则（`inputs/domain_knowledge/检测规则_15条.xlsx`）

## 10. 论文结构草案

```
1. Introduction
   - 数字人民币反洗钱挑战
   - 核心矛盾：领域知识丰富但难以转化为可计算特征
   - 贡献：(1) 知识蒸馏框架 (2) Key-clue Extractor (3) 数字人民币验证

2. Related Work
   - 自动特征工程：OpenFE, CAAFE
   - LLM for Data Science：DS-Agent
   - 反洗钱/欺诈检测

3. Problem Formulation

4. Methodology
   4.1 System Overview
   4.2 Key-clue Extractor（知识结构化）
   4.3 Feature Agent（ReAct + 4 维度 + 多维度标签）
   4.4 Three-level Verification（噪声校准 + delta-recall）
   4.5 SHAP Feedback & Search Strategy
   4.6 Two-stage Knowledge Injection

5. Experiments
   5.1 Dataset & Setup
   5.2 Main Results（Recall + CR@K，多模型）
   5.3 Wallet_type 分场景分析
   5.4 Ablation Studies
   5.5 Case Study（Provenance Card 示例）

6. Conclusion & Future Work
```

## 11. 关键决策记录

| 决策 | 结论 | 理由 |
|------|------|------|
| 特征模板图（方案 B） | 不做 | 时间不够，Applied Track 不要求硬 algorithmic contribution |
| Stage 3（模型选择训练） | 纳入评估方法（9.3） | 自动选择最优模型，体现端到端能力；不单独作为方法论 Stage |
| 跨维度组合 | 方案 1（多维度标签） | 轻量实现，消融风险低；方案 2 增量 novelty 有限 |
| AUC | 不作为核心指标 | Recall 是核心，Precision ≥ 90% 是约束，Top-K 是业务指标 |
| Feature Set A 迁移 | 直接拼为已有列 | 简单直接，Stage 2 视为基线特征 |
