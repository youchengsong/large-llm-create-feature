# KnowFeat 实验设计方案

> 目标会议：ICDM Applied Track
> 核心叙事：Knowledge-Traceable Feature Engineering — 每个特征都能追溯到具体领域知识，并有结构化因果推理链

---

## 一、数据集

### 1.1 AML 数据集（核心贡献，完整领域知识）

| 数据集 | 规模 | 特征数 | 任务 | 知识来源 |
|---|---|---|---|---|
| **数字人民币** | 138 万交易 | 95 基线特征 | 二分类（is_risk） | 38 指标 + 15 规则 + 26 段专家意见 + 309 份裁判文书 + 8 交易模式 |
| **BABD-13** | 54.4 万地址 | 149 特征 | 二分类（is_illicit） | 加密货币洗钱案例风险因子 |

### 1.2 公开数据集（泛化性验证，轻量领域知识）

| 数据集 | OpenML ID | 规模 | 特征数 | 任务 | 出现论文数 |
|---|---|---|---|---|---|
| **credit-g** | 31 | 1,000 | 20 | 二分类（信贷违约） | 4（CAAFE, LLM-FE, FeatLLM, FeRG-LLM） |
| **diabetes** | 37 | 768 | 8 | 二分类（糖尿病诊断） | 4（CAAFE, LLM-FE, FeatLLM, FeRG-LLM） |
| **blood-transfusion** | 1464 | 748 | 4 | 二分类（献血预测） | 3（LLM-FE, FeatLLM, FeRG-LLM） |
| **heart** | Kaggle | 918 | 11 | 二分类（心脏病诊断） | 3（LLM-FE, FeatLLM, FeRG-LLM） |
| **bank-marketing** | 1461 | 45,211 | 16 | 二分类（营销响应） | 3（FeatLLM, FeRG-LLM, OCTree） |
| **breast-w** | 15 | 699 | 9 | 二分类（乳腺癌诊断） | 2+（CAAFE, LLM-FE） |

公开数据集选择标准：出现在 3 篇以上 AutoFE 论文中，确保结果可对比。

---

## 二、Baseline 方法

| 方法 | 类型 | 代码来源 | 依赖 |
|---|---|---|---|
| **No-FE** | 原始特征 + XGBoost | — | — |
| **OpenFE** | 经典 AutoFE | `pip install openfe` | 无 API 需求 |
| **AutoFeat** | 经典 AutoFE | `pip install autofeat` | 无 API 需求 |
| **CAAFE** | LLM-based | github.com/noahho/CAAFE | OpenAI API |
| **FeatLLM** | LLM-based | github.com/Sungwon-Han/FeatLLM | OpenAI API |

备选：LLM-FE（github.com/nikhilsab/LLMFE），代码可用但需要 OpenAI API。

---

## 三、评估设置

### 3.1 评估协议

- **划分方式**：80/20 train/test split
- **重复次数**：5 次不同随机种子
- **下游模型**：XGBoost（主），LightGBM（辅助验证）
- **显著性检验**：Welch's t-test (p=0.05)

### 3.2 评估指标

| 指标 | 用途 |
|---|---|
| **Recall** | AML 场景核心指标（漏检代价远高于误报） |
| **Precision** | 约束条件（≥90%） |
| **F1** | 综合指标 |
| **AUC** | 公开数据集主指标（与其他论文对齐） |
| **Accuracy** | 公开数据集辅助指标 |

---

## 四、实验列表

### Exp 1：主性能对比（Performance Comparison）

**目的**：证明 KnowFeat 在 AML 场景和公开数据集上的性能竞争力。

**设置**：
- 数据集：全部 8 个（2 AML + 6 公开）
- 方法：No-FE, OpenFE, AutoFeat, CAAFE, FeatLLM, KnowFeat
- AML 数据集指标：Recall, Precision, F1, AUC
- 公开数据集指标：AUC, Accuracy

**结果呈现**：两张表
- Table 1: AML 数据集性能对比（突出 Recall 提升）
- Table 2: 公开数据集性能对比（证明泛化性）

### Exp 2：知识消融（Knowledge Ablation）

**目的**：量化每种领域知识资产的边际贡献。

**设置**：
- 数据集：数字人民币（知识最丰富的数据集）
- 配置：

| 配置 | 输入知识 |
|---|---|
| KnowFeat-Full | 全部知识（指标+规则+专家+裁判文书+交易模式） |
| KnowFeat-Rules | 仅检测规则（RULE_001~015） |
| KnowFeat-Indicators | 仅风险指标（IND_01~38） |
| KnowFeat-Expert | 仅专家意见（EXP_01~26） |
| KnowFeat-Court | 仅裁判文书风险因子 |
| KnowFeat-Schema | 仅数据 Schema（无领域知识） |

- 指标：Recall, F1, 生成特征通过率

**结果呈现**：
- Table 3: 各知识配置的性能对比
- Figure 1: 条形图展示各知识类型的边际贡献

### Exp 3：可解释性评估 — 溯源完整性（Traceability Metrics）

**目的**：定量对比各方法生成特征的可解释性。

**设置**：
- 数据集：数字人民币
- 方法：CAAFE, FeatLLM, KnowFeat
- 每个方法生成 20 个特征

**指标**：

| 指标 | 定义 |
|---|---|
| **知识引用率** (Knowledge Citation Rate) | 特征解释中明确引用了具体知识来源的特征占比 |
| **知识可验证率** (Knowledge Verifiability Rate) | 引用的知识能追溯到原始内容的比例 |
| **推理步骤覆盖率** (Reasoning Coverage) | 计算步骤中有"为什么"解释的步骤占比 |
| **字段溯源率** (Field Traceability Rate) | 代码中使用的数据字段在解释中被提及的比例 |

**结果呈现**：
- Table 4: 溯源完整性指标对比（KnowFeat 应全面领先）

### Exp 4：可解释性评估 — 专家打分（Expert Evaluation）

**目的**：通过领域专家的人工评估验证可解释性质量。

**设置**：
- 评估者：3-5 名反洗钱/风控领域专家
- 样本：每个方法（CAAFE, FeatLLM, KnowFeat）各 10-15 个特征
- 匿名化：去除方法名，随机打乱顺序
- 评分量表：1-5 分

**评分维度**：

| 维度 | 问题 |
|---|---|
| **领域合理性** (Domain Relevance) | 该特征在反洗钱场景中是否有意义？（1=完全没有，5=非常有意义） |
| **解释充分性** (Explanation Sufficiency) | 解释是否让你理解这个特征为什么这样设计？（1=完全不理解，5=完全理解） |
| **可审计性** (Auditability) | 你能否验证这个特征的设计依据是否正确？（1=完全无法验证，5=可以完全验证） |

**结果呈现**：
- Table 5: 专家评分均值和标准差
- Figure 2: 雷达图对比三个方法在三个维度的得分

### Exp 5：Case Study（定性展示）

**目的**：直观展示 KnowFeat 溯源卡片 vs 其他方法解释的差异。

**设置**：
- 挑选 2-3 个有代表性的特征（如"休眠账户激活风险"、"分散转入集中转出比率"）
- 并排展示：

```
┌─────────────────────────────────────┐
│ CAAFE 的解释                         │
│ # 代码注释里的一句话解释              │
├─────────────────────────────────────┤
│ FeatLLM 的解释                       │
│ 一句因果描述                          │
├─────────────────────────────────────┤
│ KnowFeat 的溯源卡片                  │
│ 知识溯源: [RULE_002] 内容...         │
│           [IND_05] 内容...           │
│ 计算步骤: 步骤1→步骤2→步骤3         │
│ 设计推理: 因为...所以...             │
└─────────────────────────────────────┘
```

**结果呈现**：
- Figure 3: 溯源卡片对比图（论文中占 1/3~1/2 页）

### Exp 6：效率对比（Efficiency）

**目的**：评估各方法的计算成本。

**设置**：
- 数据集：数字人民币（最大数据集）
- 指标：总运行时间、LLM API 调用次数、API 费用、生成特征数/通过特征数

**结果呈现**：
- Table 6: 效率对比

---

## 五、论文结构（草案）

1. **Introduction** — 金融合规要求可解释性 → 现有 AutoFE 缺乏知识溯源 → KnowFeat
2. **Related Work** — AutoFE (OpenFE, CAAFE) + XAI in finance + LLM for tabular data
3. **System Design** — 知识资产体系 + LLM Agent + 三级验证 + 溯源卡片
4. **Experiments**
   - 4.1 Performance Comparison (Exp 1)
   - 4.2 Knowledge Ablation (Exp 2)
   - 4.3 Traceability Evaluation (Exp 3 + 4)
   - 4.4 Case Study (Exp 5)
   - 4.5 Efficiency (Exp 6)
5. **Discussion** — 局限性 + 未来工作
6. **Conclusion**

---

## 六、公开数据集领域知识采集计划

每个公开数据集需要采集 10-15 条结构化领域知识，格式与数字人民币一致：

- **风险指标/诊断指标**：该领域的关键判断指标（类似 IND_01~38）
- **检测规则/诊断规则**：领域公认的判断规则（类似 RULE_001~015）
- **专家共识/临床指南**：权威来源的专业意见

知识来源：医学指南（WHO, ADA）、金融监管文件、学术综述、数据集原始论文。

| 数据集 | 领域 | 知识来源方向 |
|---|---|---|
| credit-g | 信贷风控 | 巴塞尔协议、信用评分卡规则、银行风控指标 |
| diabetes | 糖尿病诊断 | ADA 诊断指南、Pima Indian 研究背景 |
| blood-transfusion | 献血行为 | RFM 模型、献血行为研究 |
| heart | 心脏病诊断 | AHA/ESC 心血管风险指南、Framingham 风险因子 |
| bank-marketing | 银行营销 | 客户响应模型、RFM、生命周期价值 |
| breast-w | 乳腺癌诊断 | FNA 细胞学标准、BI-RADS 分类、WHO 指南 |
