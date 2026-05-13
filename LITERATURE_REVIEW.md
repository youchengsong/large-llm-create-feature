# 文献综述：LLM 提取/生成高质量特征

> 调研目标：通过 LLM 获取更丰富、高质量、可解释的特征，用于下游机器学习任务

---

## 一、核心论文

### 1. LLM-based Feature Generation from Text for Interpretable ML

| 项目 | 内容 |
|------|------|
| 作者 | Vojtech Balek, Lukas Sykora, Vilem Sklenak, Tomas Kliegr |
| 发表 | Machine Learning (Springer), Volume 114, 2025 |
| arXiv | https://arxiv.org/abs/2409.07132 |
| 代码 | https://github.com/vojtech-balek/llm-features |

**核心思想**：用 LLM 从文本中提取少量（~62 个）语义明确的可解释特征，替代高维嵌入（768 维），在保持接近 SOTA 性能的同时大幅提升可解释性。

**两种工作流**：
1. **用户指定特征**：领域专家定义特征名称（如"方法论严谨性"、"新颖性"），LLM 计算每条文本对应的特征值
2. **自动发现特征**：LLM 从数据集中随机抽取 40 行样本，自主识别相关特征，再为全量数据生成特征值

**技术细节**：
- 模型：Llama2 13B（本地推理）/ GPT-4o（特征发现）/ GPT-4o-mini（特征生成）
- 设置 `do_sample=False` 确保输出一致性
- 用卡方检验（2500 次 bootstrap）验证特征与目标的显著相关性
- Cramer's V 衡量效应大小（0.191–0.865，中等到强相关）
- SHAP 分析 + Action Rules 挖掘可解释规则

**关键结果**：

| 数据集 | LLM 特征 | 嵌入基线 | 组合最佳 |
|--------|----------|----------|----------|
| CORD-19（3000 篇论文）| 59.7%（62 维）| 62.5%（SciBERT 768 维）| 66.5% |
| 仇恨言论 | 少数类 recall 54% | TF-IDF recall 14% | — |
| 食品危害 | 93% | — | — |

**成本**：本地 GPU 处理 3000 条仅 $0.15，全部 5 个数据集 API 总成本约 $12.07

**人工评估**：41 名参与者评 100 个特征，27 个评分 4.0–5.0（明确相关），仅 4 个低于 2.0

**对我们的启发**：
- **特征设计**：可以先用领域知识定义候选特征，再让 LLM 批量打分
- **特征发现**：让 LLM 看少量样本后自动发现潜在特征，降低对领域专家的依赖
- **成本可控**：小模型+本地推理的成本极低，适合大规模数据
- **验证流程**：卡方检验 + Cramer's V 的统计验证方法值得复用

---

### 2. LLM-FE: LLM-based Feature Engineering as Evolutionary Optimizers

| 项目 | 内容 |
|------|------|
| 作者 | Nikhil Abhyankar, Parshin Shojaee, Chandan K. Reddy |
| 发表 | ICLR 2026 提交（withdrawn） |
| OpenReview | https://openreview.net/forum?id=JhJPJtau8B |
| 代码 | https://github.com/nikhilsab/llmfe |

**核心思想**：将特征工程形式化为**程序搜索问题**，结合进化搜索策略与 LLM 的领域知识和代码生成能力。

**方法**：
- LLM 生成特征转换的 Python 代码（而非直接生成特征值）
- 进化搜索策略迭代优化特征组合
- 不依赖数据集的自然语言描述

**对我们的启发**：
- 让 LLM 生成**特征转换代码**而非直接打分，更灵活、可复用
- 进化搜索思路可用于自动发现最优特征组合

---

### 3. Exploring the ICL Capabilities of LLMs for Money Laundering Detection

| 项目 | 内容 |
|------|------|
| 作者 | Erfan Pirmorad |
| 发表 | IEEE ICDMW 2025 |
| arXiv | https://arxiv.org/abs/2507.14785 |

**核心思想**：将金融图谱中的子图序列化为文本，用 LLM 的 in-context learning 能力直接推理洗钱可疑性。

**方法**：
1. 从金融知识图谱提取目标交易的 k-hop 邻域子图
2. 将图结构转为结构化文本（保留拓扑关系、节点属性、边属性）
3. 提供 8 种洗钱模式示例 + 4 个正常交易示例（few-shot）
4. LLM 评估可疑程度并生成推理理由

**关键结果**：

| 洗钱模式 | 精确率 | 召回率 |
|---------|--------|--------|
| Fan-out | 75.1% | 80.3% |
| Fan-in | 73.8% | 81.5% |
| Gather-scatter | 66.2% | 70.7% |
| Random/Bipartite/Stack | 51–55% | 53–58% |

**总体准确率**：63.7%（使用 GPT-4o，无微调）

**对我们的启发**：
- **图结构序列化**：将复杂的图/网络数据转为文本让 LLM 处理是可行的
- **Few-shot 模式匹配**：提供典型模式示例可显著提升检测效果
- **局限性**：复杂模式检测效果下降明显，纯 ICL 方式在生产环境中性能不够

---

### 4. RiskTagger: LLM-based Agent for Web3 Crypto Laundering Annotation

| 项目 | 内容 |
|------|------|
| 作者 | Dan Lin 等 |
| 发表 | ACM WWW 2026 |
| arXiv | https://arxiv.org/abs/2510.17848 |

**核心思想**：三模块 Agent 架构，端到端自动标注加密货币洗钱行为。

**三个核心模块**：

1. **Key-clue Extractor**（关键线索提取器）
   - 两阶段提取：先分块摘要，再全局合并
   - 从安全报告提取资金来源、金额、可疑账户
   - 准确率：100%

2. **Laundering Tracer**（洗钱追踪器）
   - 基于 FATF 红旗指标，从四个维度分析：交易模式、资金流向、关联地址、时序行为
   - Chain-of-Thought + Self-reflection 机制
   - 与专家一致率：84.1%

3. **Dataset Explainer**（数据集解释器）
   - 生成审计友好报告，标注交易类型
   - 信息覆盖率：90%

**对我们的启发**：
- **多维度特征提取框架**：交易模式、资金流向、关联地址、时序行为四个维度可直接借鉴
- **两阶段提取**：先分块后合并，适合处理长文本/大量数据
- **Self-reflection**：出现矛盾证据时触发重新评估，提升标注质量

---

### 5. "Regulation In, Regulation Out" — Agentic LLM for AML

未能找到该论文的公开版本，可能尚未公开或为行业白皮书。最接近的替代工作：

- **Co-Investigator AI** (arXiv: 2509.08380)：多 Agent 框架，包含规划/犯罪类型检测/情报收集/合规验证 Agent
- **Agentic LLM for Adverse Media Screening** (arXiv: 2602.23373)：LLM + RAG 自动搜索、检索、计算 Adverse Media Index

---

## 二、相关重要论文

### A. LLM 驱动的特征工程

#### CAAFE: Context-Aware Automated Feature Engineering (NeurIPS 2023)
- **开创性工作**：首次系统性将 LLM 引入自动化特征工程
- LLM 根据数据集描述迭代生成语义有意义的新特征，同时输出 Python 代码和特征解释
- 14 个数据集中 11 个获得提升，平均 ROC AUC 0.798 → 0.822
- 代码：https://github.com/noahho/CAAFE

#### OCTree: Optimized Feature Generation via LLMs with Decision Tree Reasoning (NeurIPS 2024)
- 用决策树推理引导 LLM 生成特征工程规则
- 将实验结果翻译为自然语言反馈给 LLM 迭代优化
- 无需数据集语言描述（克服 CAAFE 的限制）
- 代码：https://github.com/jaehyun513/OCTree

### B. LLM + 金融数据

#### FinFRE-RAG: Understanding Structured Financial Data with LLMs (ACL 2026)
- 两阶段：重要性引导的特征降维 → 序列化为自然语言 → RAG + ICL 做欺诈检测
- 在 4 个公开欺诈数据集上显著优于直接提示，与强表格基线竞争

### C. 综述

#### Large Language Models on Tabular Data: A Survey (2024)
- arXiv: 2402.17944
- 全面综述 LLM 在表格数据上的应用：序列化方法、表格操作、提示工程、端到端系统

---

## 三、技术方法分类

| 方法类别 | 代表论文 | 核心思路 | 适用场景 |
|---------|---------|---------|---------|
| **文本→可解释特征** | Balek 2025 | LLM 直接从文本提取语义特征值 | 文本分类、NLP |
| **表格→特征代码** | CAAFE, OCTree, LLM-FE | LLM 生成特征转换代码/规则 | 表格数据预测 |
| **图→文本推理** | Pirmorad 2025 | 子图序列化 + ICL 推理 | 金融图谱分析 |
| **多 Agent 系统** | RiskTagger | 多模块 Agent 协同 | AML 合规、犯罪调查 |
| **结构化数据→NL** | FinFRE-RAG | 特征降维 + 序列化 + RAG | 欺诈检测 |

---

## 四、关键发现与建议

### 趋势

1. **可解释性与性能的权衡正在缩小**：62 个 LLM 特征可接近 768 维嵌入的性能
2. **从单一调用到 Agent 架构**：RiskTagger 等代表了多模块 Agent 系统的趋势
3. **成本可控**：本地小模型推理成本极低（$0.15/3000 条），具备实际部署可行性
4. **图结构 + LLM 是新方向**：序列化后 ICL 可行但性能有限，需要结合其他方法

### 对本项目的建议

1. **优先参考 Balek 2025 的方法**：
   - 两种工作流（专家指定 + 自动发现）灵活实用
   - 统计验证流程（卡方检验 + Cramer's V）可直接复用
   - 成本效益最优

2. **结合 CAAFE/OCTree 的代码生成思路**：
   - 让 LLM 生成特征转换代码，比直接打分更灵活
   - 迭代优化机制可持续改进特征质量

3. **借鉴 RiskTagger 的多维度分析框架**：
   - 从多个维度（模式、流向、关联、时序）系统性提取特征
   - Chain-of-Thought + Self-reflection 提升质量

4. **建立特征质量验证体系**：
   - 统计显著性检验（卡方/互信息）
   - 效应大小衡量（Cramer's V）
   - 人工评估（抽样验证语义合理性）
   - 下游任务性能对比

---

## 参考文献

1. Balek et al. "LLM-based feature generation from text for interpretable ML." Machine Learning, 2025.
2. Abhyankar et al. "LLM-FE: Automated Feature Engineering with LLMs as Evolutionary Optimizers." ICLR 2026 submission.
3. Pirmorad. "Exploring the ICL Capabilities of LLMs for Money Laundering Detection." IEEE ICDMW, 2025.
4. Lin et al. "RiskTagger: An LLM-based Agent for Automatic Annotation of Web3 Crypto Laundering." WWW 2026.
5. Hollmann et al. "CAAFE: Context-Aware Automated Feature Engineering." NeurIPS 2023.
6. Nam et al. "OCTree: Optimized Feature Generation via LLMs with Decision Tree Reasoning." NeurIPS 2024.
7. Tan et al. "Understanding Structured Financial Data with LLMs." ACL 2026.
8. Fang et al. "Large Language Models on Tabular Data: A Survey." arXiv 2402.17944.
9. Naik et al. "Co-Investigator AI: Agentic AI for AML Compliance." arXiv 2509.08380.
10. Chernakov et al. "An Agentic LLM Framework for Adverse Media Screening in AML." arXiv 2602.23373.
