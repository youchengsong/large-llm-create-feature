# KnowFeat 实现计划

> 目标：在本机完成系统代码，部署到服务器运行实验
> Deadline: 2026-06-07 (ICDM)
> 设计参考：docs/KnowFeat_design.md

## 项目结构（目标）

```
knowfeat/
├── config.py                  # 全局配置（API key, 模型, 阈值, 路径）
├── main.py                    # 主入口：两阶段编排
│
├── extractor/                 # Key-clue Extractor
│   └── key_clue_extractor.py  # 非结构化文档 → 结构化风险因子 JSON
│
├── agent/                     # Feature Agent
│   ├── react_agent.py         # ReAct 主循环（Thought → Action → Observation）
│   └── tools.py               # 3 个工具：explore_data, generate_feature, execute_sandbox
│
├── verification/              # 三级验证
│   ├── sandbox.py             # L1: 隔离执行特征代码（subprocess）
│   ├── statistics.py          # L2: IV/KS 计算 + 噪声校准阈值
│   └── model_eval.py          # L3: XGBoost 5-fold CV delta-recall
│
├── search/                    # 搜索策略
│   ├── dimension_manager.py   # 4 维度分组 + 预算分配
│   ├── shap_feedback.py       # SHAP 分析 + 维度预算动态调整
│   ├── failure_memory.py      # 失败记忆
│   ├── cross_dim.py           # 跨维度知识传播（migrate_best）
│   └── forward_selection.py   # 前向选择（候选池组合评估）
│
├── data/                      # 数据加载与处理
│   ├── loader.py              # Stage 2 数据加载（open_df, tx_df, close_df）
│   ├── babd13.py              # Stage 1 BABD-13 数据加载
│   └── schema_stats.py        # 自动生成 Schema + 分布差异统计
│
├── prompt/                    # Prompt 构建
│   ├── builder.py             # System Prompt 组装（知识 + Schema + SHAP + 失败记忆）
│   └── domain_knowledge.py    # 领域知识加载（风险指标/检测规则/交易模式）
│
├── llm/                       # LLM 接口
│   └── client.py              # DeepSeek/OpenAI 兼容接口 + Function Calling
│
├── model_select/              # 自动模型选择
│   └── auto_select.py         # 从 ModelSet 11 模型中自动选最优
│
└── output/                    # 输出
    ├── provenance.py          # Provenance Card 生成
    └── report.py              # 实验报告生成
```

## 实现步骤

### Phase 1: 基础设施（无需 LLM 调用即可测试）

- [x] **1.1 config.py** — 全局配置
  - API key/base_url/model
  - 文件路径（数据、知识、输出）
  - 验证阈值（IV/KS 噪声校准参数、冗余度阈值、Precision 下限）
  - 搜索参数（轮次上限、特征上限、维度预算）

- [x] **1.2 data/loader.py** — Stage 2 数据加载
  - 加载 data/accounts.csv → open_df
  - 加载 data/transactions.csv → tx_df
  - 加载 data/close.csv → close_df
  - 加载 outputs/baseline_cache.parquet → baseline_features
  - 复用 inputs/baseline/feature_engineering.py 的逻辑

- [x] **1.3 data/schema_stats.py** — Schema + 分布差异自动生成
  - 自动提取字段名/类型/取值范围
  - 按 is_risk 分组计算正常 vs 异常分布差异（均值/中位数/比例）
  - 输出格式化文本供 prompt 注入

- [x] **1.4 verification/sandbox.py** — L1 沙盒执行
  - subprocess 隔离执行 Agent 生成的 Python 代码
  - 超时 30s，捕获异常
  - 验证输出 DataFrame 格式（必须含 wallet_id 或 tx_id 列）

- [x] **1.5 verification/statistics.py** — L2 统计验证
  - IV 值计算（WoE 分箱）
  - KS 统计量计算
  - 噪声校准：生成 100 个随机特征，取 IV/KS 的 p95 作为动态阈值
  - 冗余度检测（与已有特征的最大相关系数）
  - 缺失率计算

- [x] **1.6 verification/model_eval.py** — L3 模型效果验证
  - XGBoost 5-fold StratifiedKFold CV
  - 计算 delta-recall, delta-F1, precision
  - 通过条件：delta_recall > 0 AND precision >= 0.90

- [x] **1.7 search/forward_selection.py** — 前向选择
  - 从候选池中贪心搜索：逐个加入，delta_recall > 0 → 接受
  - 每轮末尾执行

### Phase 2: LLM 交互层

- [ ] **2.1 llm/client.py** — LLM 接口
  - OpenAI 兼容接口（DeepSeek / GPT / 其他）
  - Function Calling 支持（tools 定义 + 响应解析）
  - 重试 + 错误处理

- [ ] **2.2 prompt/domain_knowledge.py** — 领域知识加载
  - 解析 风险分类表_38指标.xlsx → 4 维度 × 8 风险类文本
  - 解析 检测规则_15条.xlsx → 规则文本 + Python 代码示例
  - 硬编码 8 种交易模式描述
  - 加载专家意见文本

- [ ] **2.3 prompt/builder.py** — System Prompt 组装
  - Stage 1 prompt：BABD-13 schema + 风险因子 + 分布差异
  - Stage 2 prompt：SimECNY schema + 全部领域知识 + 基线特征列表
  - 动态注入：SHAP 反馈 + 失败记忆 + 跨维度高分示例
  - 每轮任务指令（当前轮次、维度、预算、候选数）

- [ ] **2.4 agent/tools.py** — 3 个 Function Calling 工具
  - explore_data：接收 Python 代码，在 DataFrame 上执行，返回结果文本
  - generate_feature：接收 name/dimension/code/explanation/reasoning/knowledge_source/data_fields_used，注册特征
  - execute_sandbox：调用 L1→L2→L3 验证流水线，返回完整结果

- [ ] **2.5 agent/react_agent.py** — ReAct 主循环
  - 循环：发送 prompt → 解析 LLM 响应 → 执行 tool call → 拼接 observation → 继续
  - 单轮终止：预算用完或 LLM 声明完成
  - 代码修复：L1 失败后自动修复（最多 2 次）

### Phase 3: 搜索策略 + 编排

- [ ] **3.1 search/dimension_manager.py** — 维度管理
  - 4 维度定义 + 初始均匀预算
  - 多维度标签支持
  - 每轮预算根据 SHAP 反馈动态更新

- [ ] **3.2 search/shap_feedback.py** — SHAP 反馈 + 特征打分
  - 前向选择后对当前模型计算 SHAP 值
  - 按维度聚合 SHAP 贡献
  - 按贡献反比分配下一轮预算
  - 特征复合打分（用于排序和跨维度传播优先级，不影响接受判定）：
    30% SHAP rank(1/rank) + 20% IV + 20% (1-redundancy) + 30% Delta Recall

- [ ] **3.3 search/failure_memory.py** — 失败记忆
  - 记录每个失败特征：名称、代码、失败原因、轮次
  - 每轮注入最近 N 条到 prompt

- [ ] **3.4 search/cross_dim.py** — 跨维度传播
  - 高分特征信息（名称、代码、指标、SHAP rank）传播到其他维度
  - 每轮更新传播列表

- [ ] **3.5 main.py** — 主编排
  - Stage 2 完整流程：
    1. 加载数据 + 基线特征
    2. 生成噪声校准阈值
    3. 每轮循环：4 维度 Agent 探索 → 前向选择 → SHAP → 更新预算
    4. 收敛检测（连续 2 轮无新特征 / 达上限 / 10 轮）
    5. 输出结果
  - Stage 1 同样流程（数据和知识不同）

### Phase 4: Key-clue Extractor

- [ ] **4.1 extractor/key_clue_extractor.py**
  - 输入：非结构化文本（新闻案例 / 裁判文书摘要）
  - 调用 LLM，prompt 要求输出结构化 JSON（pattern/dimension/signals/data_hint）
  - 批量处理多个文档
  - 输出保存为 JSON 文件供后续 prompt 注入

### Phase 5: 可解释性 + 模型选择

- [ ] **5.1 output/provenance.py** — Provenance Card
  - 从 generate_feature 的参数自动生成溯源卡片
  - 格式：feature_name / description / knowledge_source / data_fields / dimension / risk_logic

- [ ] **5.2 model_select/auto_select.py** — 自动模型选择
  - 调用 ModelSet/runner.py 全量跑 11 个模型（5-fold StratifiedKFold CV）
  - 解析 results/global_results.csv，汇总各模型指标
  - 分析数据特征（样本量、不平衡比、特征维度、特征类型分布）与模型表现的关联
  - 按 Recall（Precision ≥ 90% 约束）自动推荐最优模型
  - 输出：模型对比表 + 推荐理由（数据特征 → 模型适配分析）

- [ ] **5.3 output/report.py** — 实验报告
  - 生成 Recall / CR@K / wallet_type 分场景结果表
  - 消融实验对比表
  - 导出 CSV + HTML

### Phase 6: 部署 + 运行实验

- [ ] **6.1 requirements.txt** — 依赖清单
- [ ] **6.2 部署到服务器** — scp 代码到服务器，安装依赖
- [ ] **6.3 Stage 2 实验** — 先跑 Stage 2（数据已有）
- [ ] **6.4 Stage 1 实验** — 下载 BABD-13，跑 Stage 1
- [ ] **6.5 消融实验** — 依次关闭各组件
- [ ] **6.6 ModelSet 评估** — 11 模型对比

## 优先级说明

**Phase 1-3 是核心**，完成后 Stage 2 就能端到端跑通。
Phase 4 (Key-clue Extractor) 可以先手动准备 JSON，代码后补。
Phase 5-6 是实验阶段。

先做 Phase 1 → 2 → 3 → 5.2 → 6，确保能跑出实验数据。
Key-clue Extractor 和报告生成可以并行补。

## Stage 1 领域知识来源（完整清单）

### 一、新闻案例（5 个）
1. 湖南衡阳 "9.15" 特大虚拟币洗钱案（2025）— 400 亿元，多层级虚拟币交易网络 + 跑分模式
2. 北京 8 亿元虚拟币地下钱庄案（2024）— 跨境转移职务侵占赃款
3. 洪湖 "抢红包" 虚拟币洗钱案（2026）— "抢红包"模式转移 USDT
4. 四川 "USDT + 黄金" 跨境洗钱案（2024）— 虚拟币→黄金→现金闭环
5. 陈某枝比特币洗钱案（最高检典型案例，2021）— 购买比特币掩饰犯罪所得

### 二、官方监管标准（2 个）
1. 《反洗钱法（2024修订版）》— 虚拟货币纳入反洗钱监管，穿透式监测
2. 《关于进一步防范和处置虚拟货币等相关风险的通知》（银发〔2026〕42号）— 最新监管文件

### 三、行业报告（3 个）
1. Beosin《2025年虚拟资产反洗钱合规年度报告》— 全网交易统计，主流洗钱模式
2. 慢雾科技《2024区块链安全与反洗钱年度报告》— 跑分平台、多层转账团伙行为特征
3. 《区块链异常金融交易行为分类及特征指标》团体标准（T/GZDFA 001-2025）— 28 项异常指标及量化阈值

### 四、执法分析报告（3 个）
1. 最高检《惩治洗钱犯罪典型案例》— 6 起典型案例（含虚拟货币）
2. 上海警方《严打跑分洗钱》— 批量账户控制、高频小额、夜间集中转账特征
3. 江苏省检察院《涉虚拟货币犯罪案件办理》— 取证方向、行为定性要点

> 以上 13 个来源均需通过 Key-clue Extractor 转化为结构化风险因子 JSON，注入 Stage 1 的 Feature Agent prompt。
> 原始文件保存在 inputs/stage1_knowledge/，结构化输出保存在 inputs/stage1_knowledge/risk_factors/。
