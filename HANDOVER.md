# KnowFeat 项目移交文档

> 最后更新：2026-05-30
> 原负责人：宋有丞 (youchengsong)
> Git 仓库：git@github.com:youchengsong/large-llm-create-feature.git

---

## 目录

1. [项目概述](#1-项目概述)
2. [环境与部署](#2-环境与部署)
3. [目录结构总览](#3-目录结构总览)
4. [核心模块详解 (knowfeat/)](#4-核心模块详解)
5. [数据说明](#5-数据说明)
6. [运行方式](#6-运行方式)
7. [已有实验结果](#7-已有实验结果)
8. [辅助脚本 (scripts/)](#8-辅助脚本)
9. [测试](#9-测试)
10. [ModelSet 模型库](#10-modelset-模型库)
11. [未完成工作](#11-未完成工作)
12. [常见问题与踩坑记录](#12-常见问题与踩坑记录)
13. [关键配置参数速查](#13-关键配置参数速查)

---

## 1. 项目概述

### 1.1 KnowFeat 是什么

KnowFeat（Knowledge-guided Feature Engineering）是一个 **基于大语言模型 + 领域知识的自动化特征工程系统**，专门面向金融反洗钱场景。

核心思路：将非结构化的领域知识（裁判文书、法规、专家意见、行业报告）通过 LLM 转化为结构化风险因子，再由 LLM Agent 基于这些知识自动生成特征代码，经过三级验证（沙盒执行 → 统计验证 → 模型验证）后保留有效特征。

### 1.2 两个阶段（Stage）

| | Stage 1 | Stage 2 |
|---|---------|---------|
| 数据集 | BABD-13（比特币地址行为数据集） | SimECNY（数字人民币仿真数据） |
| 数据结构 | 单表：544,462 个地址 × 149 个预计算特征 | 三表：accounts（开户）+ transactions（交易）+ close（销户） |
| 标签 | 13 类 → 二分类（illicit/licit），正样本率 25.76% | 二分类（is_risk），正样本率 0.32% |
| 特征函数签名 | `def feature_xxx(df)` | `def feature_xxx(open_df, tx_df, close_df)` |
| 特征输出 | DataFrame with columns `[account, feature_name]` | DataFrame with columns `[wallet_id 或 tx_id, feature_name]` |
| 当前进展 | 正在运行（Round 4/5，已接受 6 个特征） | 已完成（7 轮，接受 18 个特征） |

### 1.3 系统流水线

```
知识源文档                    数据
    │                        │
    ▼                        ▼
┌──────────────┐    ┌──────────────┐
│ Key-clue     │    │ Data Loader  │
│ Extractor    │    │ + Schema     │
│ (LLM 提取)   │    │ Stats        │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────┐
│         Prompt Builder          │
│  (角色 + 知识 + Schema +        │
│   SHAP反馈 + 失败记忆)           │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│         ReAct Agent             │
│  (Thought → Action → Observe)   │
│  工具: explore_data,            │
│        generate_feature,        │
│        execute_sandbox          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│       三级验证 Pipeline          │
│  L1: 沙盒执行（子进程隔离）       │
│  L2: 统计验证（IV/KS/冗余度）    │
│  L3: 模型验证（XGBoost delta-R） │
└──────────────┬──────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
   [L3 通过]     [L3 未通过]
   直接接受       进入候选池
                     │
                     ▼
              ┌─────────────┐
              │ Forward      │
              │ Selection    │
              │ (贪心前向选择) │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ SHAP        │
              │ Feedback    │
              │ + 预算重分配  │
              └──────┬──────┘
                     │
                     ▼
              下一轮 Round
```

### 1.4 四个搜索维度

系统每轮按 4 个维度轮流生成特征：

| 维度 | 中文名 | 含义 |
|------|--------|------|
| 金额频次 | Amount/Frequency | 交易金额分布、频率模式 |
| 时间地域 | Time/Geography | 交易时间模式、地理特征 |
| 行为序列 | Behavior Sequence | 交易序列模式、状态转换 |
| 主体关联 | Entity Association | 交易对手关系、网络结构 |

---

## 2. 环境与部署

### 2.1 本地开发环境

- Python 3.14（项目根目录有 `.venv/`）
- 操作系统：Ubuntu Linux
- 依赖安装：`pip install -r requirements.txt`（尚未创建，见下方手动安装）

### 2.2 核心依赖

```
pandas>=2.0
numpy
scikit-learn
xgboost
shap
openai          # LLM 客户端（兼容 DeepSeek API）
python-docx     # 读取 .docx 专家意见
openpyxl        # 读取 .xlsx 知识表
pyarrow         # parquet 读写
httpx           # HTTP 客户端（被 openai 使用）
```

### 2.3 GPU 服务器（已关闭，仅供参考）

| 项 | 值 |
|----|----|
| 平台 | SeetaCloud (autodl) |
| 主机 | connect.westd.seetacloud.com |
| 端口 | 15720 |
| GPU | RTX 4080 SUPER |
| Python | /root/miniconda3/bin/python |
| 数据路径 | /root/autodl-tmp/archive/BABD-13.csv |
| 代码路径 | /root/大模型提取特征/ |

### 2.4 LLM API

| 项 | 值 |
|----|----|
| Provider | DeepSeek |
| Base URL | https://api.deepseek.com/v1 |
| Model | deepseek-chat |
| API Key 环境变量 | `KNOWFEAT_API_KEY` |

设置方式：
```bash
export KNOWFEAT_API_KEY="sk-your-key-here"
```

### 2.5 部署到远程服务器

```bash
./deploy.sh                    # 默认连接配置的服务器
./deploy.sh --sync-only        # 仅同步代码不装依赖
./deploy.sh --host HOST --port PORT  # 指定服务器
```

---

## 3. 目录结构总览

```
大模型提取特征/
├── CLAUDE.md                    # Claude Code 自动化工作流配置
├── HANDOVER.md                  # 本文档
├── README.md                    # 项目说明
├── RESULT.md                    # 实验结果汇总（非常详细）
├── todo.md                      # 实验 TODO 列表
├── deploy.sh                    # 服务器部署脚本
├── docs/
│   └── KnowFeat_design.md      # 系统设计文档
│
├── knowfeat/                    # 核心代码包
│   ├── __init__.py
│   ├── config.py                # 全局配置（路径、阈值、参数）
│   ├── main.py                  # 主入口 & 编排循环
│   ├── agent/                   # LLM Agent
│   │   ├── react_agent.py       # ReAct 循环主逻辑
│   │   └── tools.py             # 3 个工具定义 + ToolContext
│   ├── data/                    # 数据加载
│   │   ├── loader.py            # Stage 1/2 数据加载器
│   │   └── schema_stats.py      # 数据 Schema 描述生成
│   ├── extractor/               # 知识提取
│   │   └── key_clue_extractor.py# Key-clue 提取器（LLM 文档→JSON）
│   ├── llm/                     # LLM 客户端
│   │   └── client.py            # OpenAI 兼容客户端 + 重试
│   ├── prompt/                  # Prompt 构建
│   │   ├── builder.py           # System Prompt 组装
│   │   └── domain_knowledge.py  # 领域知识加载与格式化
│   ├── search/                  # 搜索策略
│   │   ├── cross_dim.py         # 跨维度知识传播
│   │   ├── dimension_manager.py # 维度预算管理
│   │   ├── failure_memory.py    # 失败记忆
│   │   ├── forward_selection.py # 前向特征选择
│   │   └── shap_feedback.py     # SHAP 反馈计算
│   ├── verification/            # 三级验证
│   │   ├── sandbox.py           # L1 沙盒执行
│   │   ├── statistics.py        # L2 统计验证
│   │   └── model_eval.py        # L3 模型验证
│   ├── output/                  # 输出
│   │   ├── provenance.py        # Provenance Card 生成
│   │   └── report.py            # 实验报告生成
│   └── model_select/            # 模型选择
│       └── auto_select.py       # 自动模型选择（11 个模型）
│
├── inputs/                      # 输入数据 & 知识源
│   ├── SimECNY/                 # SimECNY 仿真数据生成脚本
│   │   ├── open/                # 开户数据生成
│   │   ├── transaction/         # 交易数据生成
│   │   └── close/               # 销户数据生成
│   ├── baseline/                # 基线特征工程
│   │   └── feature_engineering.py # 原始 105 特征生成脚本
│   ├── domain_knowledge/        # 领域知识源文件
│   │   ├── 风险分类表_38指标.xlsx
│   │   ├── 检测规则_15条.xlsx
│   │   └── 邮储专家访谈_数币洗钱模式.docx
│   ├── stage1_knowledge/        # Stage 1 知识源（13 个文档 + 提取结果）
│   │   ├── *.md                 # 13 个案例/法规/报告 Markdown
│   │   └── risk_factors/*.json  # LLM 提取的结构化风险因子
│   └── stage2_knowledge/        # Stage 2 知识源（裁判文书提取结果）
│       └── risk_factors/*.json  # 309 个裁判文书提取的风险因子
│
├── outputs/                     # 实验输出
│   ├── stage2_features.json     # Stage 2 已接受特征（代码 + 元数据）
│   ├── stage2_summary.json      # Stage 2 运行摘要
│   ├── stage2_provenance.json   # Stage 2 特征溯源卡
│   ├── stage2_provenance.md     # Stage 2 特征溯源卡（Markdown）
│   ├── babd13_baseline.json     # BABD-13 基线实验结果
│   ├── baseline_compare.json    # 105 vs 95 特征对比
│   ├── baseline_cache.parquet   # 预计算特征矩阵（~335MB，git ignored）
│   ├── baseline_feature_importance.csv # 基线特征重要性
│   ├── crk_results.json         # CR@K 实验结果
│   ├── crk_results_exp3.json    # CR@K 实验 3 结果
│   ├── features/                # 18 个已接受特征的独立 .py 文件
│   └── single_agent_run/        # 早期单 Agent 测试运行输出
│
├── scripts/                     # 辅助脚本
│   ├── babd13_baseline.py       # BABD-13 基线评估
│   ├── baseline_eval_compare.py # 105 vs 95 对比评估
│   ├── build_all_113_features.py# 构建 95 基线 + 18 KnowFeat 特征
│   ├── build_baseline_95_features.py # 构建 95 基线特征
│   ├── compute_crk.py           # 计算 CR@K
│   ├── compute_crk_exp3.py      # 计算 CR@K（实验 3）
│   └── run_stage1_extraction.py # 运行 Stage 1 知识提取
│
├── logs/                        # 运行日志
│   └── stage1_run.log           # Stage 1 KnowFeat 运行日志
│
├── tests/                       # 单元测试
│   ├── test_*.py                # 17 个测试文件
│
└── ModelSet/                    # 模型库（11 个候选模型）
    ├── runner.py                # 模型运行器
    ├── catboost_std.py          # CatBoost
    ├── lgbm_std.py              # LightGBM
    ├── xgboost_model.py         # XGBoost
    ├── ...                      # 其他 8 个模型
    └── requirements.txt         # 模型库依赖
```

---

## 4. 核心模块详解

### 4.1 config.py — 全局配置

**位置**：`knowfeat/config.py`

所有配置集中管理，分为 4 个子配置 + 1 个顶层配置：

```python
KnowFeatConfig
├── LLMConfig          # LLM 连接（API key、model、temperature 等）
├── PathConfig         # 所有文件路径（数据、知识、输出）
├── VerificationConfig # L1/L2/L3 验证阈值
└── SearchConfig       # 搜索策略参数
```

**重要参数：**

| 参数 | 位置 | 默认值 | 含义 |
|------|------|--------|------|
| `api_key` | LLMConfig | 环境变量 `KNOWFEAT_API_KEY` | DeepSeek API 密钥 |
| `babd13_csv` | PathConfig | `/root/autodl-tmp/archive/BABD-13.csv` | BABD-13 数据路径 |
| `max_rounds` | SearchConfig | 10 | 最大搜索轮数 |
| `max_features` | SearchConfig | 30 | 最大接受特征数 |
| `convergence_rounds` | SearchConfig | 2 | 连续 N 轮无新特征则停止 |
| `initial_budget_per_dim` | SearchConfig | 3 | 每维度每轮初始预算 |
| `delta_recall_min` | VerificationConfig | 0.0 | L3 通过阈值（必须 > 0） |
| `precision_min` | VerificationConfig | 0.90 | L3 precision 下限 |
| `redundancy_max` | VerificationConfig | 0.7 | L2 最大冗余度 |
| `cv_folds` | VerificationConfig | 5 | L3 交叉验证折数 |

**工厂函数：**
```python
cfg = load_config(stage="stage1", max_rounds=5)
```

---

### 4.2 main.py — 主编排

**位置**：`knowfeat/main.py`

整个系统的入口和编排逻辑。

**核心函数 `run_stage()`** 的执行流程：

```
1. 初始化 LLM Client、DimensionManager、FailureMemory、CrossDimPropagator
2. 噪声校准（100 个随机特征 → 计算 IV/KS p95 作为阈值）
3. for round in range(max_rounds):
    3.1 for dimension in [金额频次, 时间地域, 行为序列, 主体关联]:
        - 构建 prompt（含 SHAP 反馈 + 失败记忆 + 跨维度示例）
        - 调用 ReactAgent.run_round()
        - 记录结果到 DimensionManager 和 FailureMemory
    3.2 前向选择：从候选池中贪心选择能提升 recall 的特征
    3.3 SHAP 反馈：计算所有已接受特征的 SHAP 重要性
    3.4 预算重分配：SHAP 贡献低的维度获得更多预算
    3.5 收敛检查：
        - 达到 max_features → 停止
        - 连续 convergence_rounds 轮无新特征 → 停止
4. 保存结果（features JSON + summary + provenance cards）
```

**CLI 用法：**
```bash
# Stage 2（SimECNY）
KNOWFEAT_API_KEY=sk-xxx python -m knowfeat.main --stage stage2

# Stage 1（BABD-13）
KNOWFEAT_API_KEY=sk-xxx python -m knowfeat.main --stage stage1 --max-rounds 5

# 安静模式
python -m knowfeat.main --stage stage1 --quiet
```

**输出文件：**
- `outputs/stage{1,2}_features.json` — 已接受特征的代码和元数据
- `outputs/stage{1,2}_summary.json` — 运行概况统计
- `outputs/stage{1,2}_provenance.json` — 特征溯源卡
- `outputs/stage{1,2}_provenance.md` — 特征溯源卡（可读版）

---

### 4.3 agent/react_agent.py — ReAct Agent

**位置**：`knowfeat/agent/react_agent.py`

实现 ReAct（Reasoning + Acting）循环。每一步：

```
1. 将当前对话历史发给 LLM
2. LLM 返回：
   - 纯文本思考（Thought） → 记录但不执行
   - 工具调用（Action） → 执行并返回 Observation
3. 将 Observation 追加到对话历史
4. 重复，直到：
   - 达到维度预算（generated >= budget）
   - Agent 宣布完成
   - 达到 MAX_STEPS_PER_ROUND (100)
```

**L1 自动修复机制：**

当 `execute_sandbox` 返回 L1 失败时，Agent 会自动注入修复提示（含错误信息和修复建议），让 LLM 修改代码重试，最多修复 2 次。

---

### 4.4 agent/tools.py — 工具定义 & ToolContext

**位置**：`knowfeat/agent/tools.py`

这是最核心的文件之一，定义了 LLM Agent 可用的 3 个工具：

#### 工具 1: explore_data

**用途**：在数据上执行探索性代码（只读），查看数据分布、统计信息等。

**输入参数**：
```json
{
  "code": "print(df.describe())"  // Python 代码
}
```

**Stage 1** 注入变量：`df`（BABD-13 完整 DataFrame）
**Stage 2** 注入变量：`open_df`, `tx_df`, `close_df`, `baseline_features`

**执行方式**：在 fork 的子进程中运行，超时 30 秒，输出截断为 4000 字符。

**返回**：代码执行的 stdout 输出（字符串）。

#### 工具 2: generate_feature

**用途**：注册一个新特征（只注册元数据，不执行代码）。

**输入参数**：
```json
{
  "feature_name": "small_test_ratio",
  "code": "def feature_small_test_ratio(df): ...",
  "explanation": "小额测试交易占比...",
  "reasoning": "基于 RULE_002...",
  "knowledge_source": "RULE_002, 邮储专家",
  "data_fields_used": ["amount", "src"],
  "dimension": ["金额频次"]
}
```

**返回**：注册确认 JSON，包含特征名和维度。

#### 工具 3: execute_sandbox

**用途**：触发已注册特征的 L1→L2→L3 验证 Pipeline。

**输入参数**：
```json
{
  "feature_name": "small_test_ratio"
}
```

**验证 Pipeline（`_run_pipeline`）**：

```
L1 沙盒执行（sandbox.py）
  ├── 成功 → 提取特征列，对齐标签
  │          ├── L2 统计验证（statistics.py）
  │          │    ├── 通过 → L3 模型验证（model_eval.py）
  │          │    │          ├── 通过 → 直接接受，加入 baseline_X
  │          │    │          └── 未通过 → 加入候选池
  │          │    └── 未通过 → 拒绝
  │          └── 对齐失败 → 拒绝
  └── 失败 → 返回错误信息（Agent 会尝试自动修复）
```

**返回**：JSON 格式的完整验证结果，包含每一级的指标。

#### ToolContext 状态管理

```python
tool_ctx = ToolContext(data, cfg, noise_thresholds, baseline_X)

# 运行时状态：
tool_ctx.registered_features   # dict: 已注册但未验证的特征
tool_ctx.accepted_features     # dict: L3 通过的特征
tool_ctx.candidate_pool        # list: L2 通过但 L3 未通过的特征
tool_ctx.rejected_features     # dict: 被拒绝的特征
tool_ctx.baseline_X            # pd.DataFrame: 当前基线特征矩阵（动态增长）
```

---

### 4.5 data/loader.py — 数据加载

**位置**：`knowfeat/data/loader.py`

#### Stage 2 加载

```python
from knowfeat.data import load_stage2, get_label_series

data = load_stage2(cfg=cfg)  # 返回 Stage2Data
# data.open_df    — 开户表
# data.tx_df      — 交易表（按 timestamp 排序）
# data.close_df   — 销户表
# data.baseline_features — 基线特征矩阵 (parquet)

labels = get_label_series(data)  # data.tx_df["is_risk"]
```

#### Stage 1 加载

```python
from knowfeat.data import load_stage1, get_label_series_stage1

data = load_stage1(cfg=cfg)  # 返回 Stage1Data
# data.df             — 完整 DataFrame (544462 × 151)
# data.baseline_features — 149 个特征列
# data.label_col      — "is_illicit"
# data.id_col         — "account"
# data.feature_cols   — 特征列名列表

labels = get_label_series_stage1(data)  # data.df["is_illicit"]
```

**关键处理：**
- SW 列编码：`SA` → 0, `WA` → 1
- 13 类 → 二分类：`BABD13_ILLICIT_LABELS = {0, 2, 6, 7, 8, 9, 11}`
- 排除的 10 个基线特征：`BASELINE_EXCLUDE_FEATURES`（数据泄漏/过拟合风险）

---

### 4.6 data/schema_stats.py — Schema 描述生成

**位置**：`knowfeat/data/schema_stats.py`

为 Prompt 生成数据描述文本，包含：
- 每张表的列名、类型、取值范围、缺失率
- 正常 vs 异常样本的分布差异（按差异幅度排序）
- 表间 JOIN 覆盖率（Stage 2）
- 基线特征列表

```python
# Stage 2
schema_text = generate_all_schema_text(data, label_col="is_risk")

# Stage 1
schema_text = generate_all_schema_text_stage1(data)
```

---

### 4.7 extractor/key_clue_extractor.py — 知识提取

**位置**：`knowfeat/extractor/key_clue_extractor.py`

将非结构化文档（.md/.txt/.docx）通过 LLM 转化为结构化 JSON 风险因子。

**输入**：知识源文档（裁判文书、法规、报告等）
**输出**：每个文档一个 JSON 文件，格式如下：

```json
{
  "source": "陈某枝比特币洗钱案",
  "risk_factors": [
    {
      "pattern": "比特币场外交易OTC规避KYC",
      "dimension": ["主体关联", "金额频次"],
      "observable_signals": ["使用多个平台分散交易", "单笔金额不超过5万"],
      "data_hint": "检查交易对手多样性和金额阈值"
    }
  ]
}
```

**运行方式**：
```bash
# 首次运行时 main.py 会自动调用（如果缓存不存在）
# 也可手动运行：
KNOWFEAT_API_KEY=sk-xxx python scripts/run_stage1_extraction.py
```

**缓存位置**：
- Stage 1: `inputs/stage1_knowledge/risk_factors/*.json`（13 个文件）
- Stage 2: `inputs/stage2_knowledge/risk_factors/*.json`（309 个文件）

---

### 4.8 llm/client.py — LLM 客户端

**位置**：`knowfeat/llm/client.py`

基于 OpenAI Python SDK 的封装，支持：
- Function Calling（工具调用）
- 指数退避重试（2^n 秒，最大 30 秒）
- 可重试错误类型：RateLimitError、APITimeoutError、APIConnectionError、InternalServerError

```python
from knowfeat.llm.client import LLMClient, ToolSpec

client = LLMClient(cfg.llm)

# 简单问答
answer = client.ask("什么是洗钱？", system="你是反洗钱专家")

# 带工具的对话
response = client.chat(messages, tools=[tool_spec1, tool_spec2])
if response.has_tool_calls:
    for tc in response.tool_calls:
        result = execute_tool(tc.name, tc.arguments)
        messages.append(client.make_tool_result_message(tc.id, result))
```

---

### 4.9 prompt/builder.py — Prompt 构建

**位置**：`knowfeat/prompt/builder.py`

组装 System Prompt 的 7 个部分：

```
1. 角色定义（_ROLE_STAGE1 或 _ROLE_STAGE2）
   - 你是谁、任务是什么、特征函数签名要求
2. 领域知识（风险指标 + 检测规则 + 交易模式 + 专家意见）
3. 数据 Schema（各表结构 + 分布差异）
4. 验证规则（L1/L2/L3 阈值说明）
5. SHAP 反馈（上一轮各维度贡献 + 重要特征排名）
6. 失败记忆（最近 N 个失败特征的代码和原因）
7. 跨维度示例（其他维度高分特征的代码和指标）
```

**调用方式**：
```python
system_prompt, user_message = build_full_prompt(
    cfg, domain_knowledge, data=data, schema_text=schema_text,
    shap_feedback=shap_fb, failure_records=failures,
    cross_dim_examples=examples,
    round_num=2, dimension="金额频次", budget=3,
    n_accepted=6, n_candidates=10,
)
```

---

### 4.10 prompt/domain_knowledge.py — 领域知识

**位置**：`knowfeat/prompt/domain_knowledge.py`

加载和格式化 5 类领域知识：

| 知识类型 | 来源文件 | 数量 |
|----------|---------|------|
| 风险指标 | `风险分类表_38指标.xlsx` | 38 个指标 |
| 检测规则 | `检测规则_15条.xlsx` | 15 条规则（含 Python hints） |
| 交易模式 | 硬编码在代码中 | 8 个模式（3 正常 + 5 异常） |
| 专家意见 | `邮储专家访谈_数币洗钱模式.docx` | 2190 字符 |
| 风险因子 | `risk_factors/*.json` | Stage 1: 13 文件 / Stage 2: 309 文件 |

**压缩策略**：裁判文书的 309 个 JSON 文件（原始 ~400K 字符）会被聚合去重压缩到 ~6K 字符，以适应 Prompt 长度。

```python
from knowfeat.prompt.domain_knowledge import load_domain_knowledge, format_domain_knowledge

dk = load_domain_knowledge(cfg)
text = format_domain_knowledge(dk, stage="stage1")  # 压缩后的知识文本
```

---

### 4.11 search/ — 搜索策略模块

#### dimension_manager.py — 维度预算管理

```python
dm = DimensionManager(cfg)
dm.get_budget("金额频次")  # 返回当前预算
dm.record_round("金额频次", generated=3, accepted=1, candidates=2)
dm.update_budgets_from_shap({"金额频次": 4.7, "时间地域": 1.0, ...})
# → SHAP 贡献低的维度获得更多预算（反比分配）
```

#### failure_memory.py — 失败记忆

```python
fm = FailureMemory()
fm.record_failure("bad_feature", code="...", reason="IV=0.001 < 0.02", round_num=1, dimension="金额频次")
recent = fm.get_recent(10)  # 最近 10 个失败记录 → 注入 Prompt
```

#### forward_selection.py — 前向选择

L3 未通过但 L2 通过的特征进入候选池。每轮结束后，贪心地逐个尝试加入：

```python
result = forward_select(baseline_X, label, candidates, cfg=cfg)
# result.accepted: 通过的候选特征
# result.rejected: 未通过的候选特征
```

#### shap_feedback.py — SHAP 反馈

计算所有已接受特征的 Tree SHAP 重要性，按维度聚合：

```python
feedback = analyze_shap_feedback(X, label, feature_dimensions, ...)
# feedback.dimension_contributions: {"金额频次": 4.74, "时间地域": 1.00, ...}
# feedback.feature_scores: 按 composite_score 排序的特征列表
```

#### cross_dim.py — 跨维度传播

将高分特征（来自其他维度）的代码和指标注入当前维度的 Prompt，作为灵感：

```python
propagator = CrossDimPropagator(cfg)
propagator.register_feature("feature_a", ["金额频次"], code="...")
examples = propagator.get_examples_for_dimension("时间地域", top_k=5)
```

---

### 4.12 verification/ — 三级验证

#### L1: sandbox.py — 沙盒执行

在隔离子进程中执行 LLM 生成的特征代码。

**流程**：
1. 将输入数据 pickle 到临时文件
2. 生成 runner 脚本（注入用户代码）
3. subprocess 执行，超时 30 秒
4. 加载 pickle 输出，验证格式（必须是 DataFrame，含 ID 列）

```python
result = run_sandbox(code, open_df, tx_df, close_df, timeout=30)  # Stage 2
result = run_sandbox_stage1(code, df, timeout=30)                   # Stage 1
# result.success: bool
# result.df: 输出 DataFrame
# result.error: 错误信息
```

#### L2: statistics.py — 统计验证

检查 4 个指标：

| 指标 | 含义 | 通过条件 |
|------|------|---------|
| IV (Information Value) | 特征区分度 | > noise_threshold（噪声校准动态阈值） |
| KS (Kolmogorov-Smirnov) | 分布差异 | > noise_threshold |
| 缺失率 | NaN 比例 | < 0.3 (30%) |
| 冗余度 | 与已有特征最大相关系数 | < 0.7 |

**噪声校准**：生成 100 个随机高斯特征，计算 IV/KS 的 p95 作为阈值（防止随机噪声通过）。

```python
thresholds = calibrate_noise_thresholds(label, n_samples=len(label))
result = validate_feature(feature_series, label, existing_features, thresholds)
```

#### L3: model_eval.py — 模型验证

XGBoost 5-fold StratifiedKFold CV，比较加入新特征前后的 delta-recall。

```python
result = evaluate_feature(baseline_X, new_feature_series, label, cfg=cfg)
# result.passed: delta_recall > 0 且 precision >= 0.90
# result.baseline_recall, result.new_recall, result.delta_recall
```

**XGBoost 参数**：
```python
XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=neg_count/pos_count,  # 动态计算
    eval_metric="logloss"
)
```

---

### 4.13 output/ — 输出模块

#### provenance.py — 特征溯源卡

每个接受的特征生成一张"溯源卡"，记录：
- 特征名、维度、解释、推理过程
- 知识来源（对应哪些 RULE/专家意见/裁判文书）
- 使用的数据字段
- 代码 SHA256 哈希

```python
cards = build_cards_from_accepted(accepted_features)
save_provenance(cards, output_dir, stage="stage2")
```

#### report.py — 实验报告

生成评估报告，支持：
- 整体 recall/precision/f1/CR@K
- 按 wallet_type 分场景评估
- 消融实验对比
- 导出 CSV + HTML + JSON

---

### 4.14 model_select/auto_select.py — 模型选择

自动运行 11 个候选模型，选择满足 Precision >= 90% 且 Recall 最高的模型。

**11 个模型**：CatBoost、GA2M、Scorecard、AC-XGB、WAV-XGB、Ensemble、LightGBM、Resample-Boost、GBDT-FM、DNN-Res、MLP

```python
result = run_model_selection("train.csv", "test.csv", output_dir="outputs/model_select")
print(result.recommended_model)  # "catboost"
```

---

## 5. 数据说明

### 5.1 SimECNY（Stage 2）

| 表 | 文件 | 行数 | 关键列 |
|----|------|------|--------|
| accounts | `data/accounts.csv` | ~50K | wallet_id, wallet_type, region_code, wallet_open_date |
| transactions | `data/transactions.csv` | ~1.38M | tx_id, src, dst, amount, timestamp, hour, is_risk, transaction_mode |
| close | `data/close.csv` | ~10K | wallet_id, zs_timestamp |
| baseline_features | `outputs/baseline_cache.parquet` | ~1.38M | 106 列（95 有效特征 + 10 排除 + 1 标签） |

**数据生成**：`inputs/SimECNY/` 下有完整的仿真数据生成脚本。

**注意**：原始 CSV 数据文件（transactions.csv 约 1.5GB）被 .gitignore 排除，需要从服务器获取或重新生成。

### 5.2 BABD-13（Stage 1）

| 项 | 值 |
|----|----|
| 文件 | `/root/autodl-tmp/archive/BABD-13.csv` |
| 行数 | 544,462 个地址 |
| 列数 | 151（account + label + 149 features） |
| 特征类型 | PAI（地址属性）、PDI（度特征）、PTI（交易模式）、CI（关联特征）、S（统计特征）、SW |
| 标签 | 13 类 → 二分类（Illicit: 25.76%） |

**公开数据集**：https://www.kaggle.com/datasets/babd-13

### 5.3 领域知识

| 类型 | 数量 | 路径 |
|------|------|------|
| 风险指标 | 38 个 | `inputs/domain_knowledge/风险分类表_38指标.xlsx` |
| 检测规则 | 15 条 | `inputs/domain_knowledge/检测规则_15条.xlsx` |
| 专家意见 | 1 份 | `inputs/domain_knowledge/邮储专家访谈_数币洗钱模式.docx` |
| Stage 1 案例/报告 | 13 个 | `inputs/stage1_knowledge/*.md` |
| Stage 1 风险因子 | 13 个 JSON | `inputs/stage1_knowledge/risk_factors/*.json` |
| Stage 2 裁判文书风险因子 | 309 个 JSON | `inputs/stage2_knowledge/risk_factors/*.json` |

---

## 6. 运行方式

### 6.1 完整运行 KnowFeat

```bash
# 设置 API Key
export KNOWFEAT_API_KEY="sk-your-deepseek-key"

# 设置 PYTHONPATH
export PYTHONPATH=/path/to/大模型提取特征

# Stage 2（SimECNY，需要三表数据 + baseline_cache.parquet）
python -m knowfeat.main --stage stage2 --max-rounds 10

# Stage 1（BABD-13，需要 BABD-13.csv）
python -m knowfeat.main --stage stage1 --max-rounds 5

# 后台运行（推荐在服务器上使用 nohup）
nohup python -u -m knowfeat.main --stage stage1 --max-rounds 5 > stage1_run.log 2>&1 &
```

**重要**：`python -u` 是必须的（unbuffered），否则 nohup 日志不会实时刷新。

### 6.2 运行基线评估

```bash
# SimECNY 基线对比（105 vs 95 特征）
python scripts/baseline_eval_compare.py

# BABD-13 基线评估
python scripts/babd13_baseline.py

# 计算 CR@K
python scripts/compute_crk.py
```

### 6.3 构建特征矩阵

```bash
# 构建 95 基线特征
python scripts/build_baseline_95_features.py \
  data/transactions.csv data/accounts.csv data/close.csv \
  outputs/baseline_95.csv

# 构建 95 + 18 KnowFeat = 113 特征
python scripts/build_all_113_features.py \
  data/transactions.csv data/accounts.csv data/close.csv \
  outputs/all_113.csv
```

### 6.4 运行测试

```bash
cd /path/to/大模型提取特征
python -m pytest tests/ -v
```

---

## 7. 已有实验结果

详细结果见 `RESULT.md`，摘要如下：

### 7.1 Stage 2（SimECNY）— 已完成

| 项目 | 值 |
|------|-----|
| 运行时间 | 9.5 小时（7 轮） |
| 接受特征 | 18 / 30 |
| 收敛原因 | 连续 2 轮无新特征 |
| Baseline recall (95 特征) | 0.9538 |
| KnowFeat recall (95+18=113) | 0.9737 |
| delta recall | **+0.0199** |
| Precision | 0.9402 → 0.9621 (+0.0219) |

**18 个已接受特征**（代码在 `outputs/features/*.py`）：

| # | 特征名 | 维度 | delta_recall |
|---|--------|------|-------------|
| 1 | small_test_round_amount_ratio | 金额频次 | +0.0051 |
| 2 | amount_concentration_entropy | 金额频次 | +0.0025 |
| 3 | top3_hour_concentration | 时间地域 | +0.0011 |
| 4 | fan_in_fan_out_ratio | 行为序列 | +0.0013 |
| 5 | mode_transition_entropy | 行为序列 | +0.0025 |
| 6 | new_counterparty_concentration | 主体关联 | +0.0004 |
| 7 | counterparty_level_diversity_risk | 主体关联 | +0.0009 |
| 8 | amount_range_ratio_risk | 金额频次 | +0.0000 |
| 9 | high_risk_region_off_hour_ratio | 时间地域 | +0.0004 |
| 10 | rhythm_stability_risk | 行为序列 | +0.0011 |
| 11 | net_flow_direction_risk | 行为序列 | +0.0007 |
| 12 | enterprise_counterparty_concentration | 主体关联 | +0.0000 |
| 13 | first_small_test_ratio | 行为序列 | +0.0011 |
| 14 | night_amount_short_term_risk | 时间地域 | +0.0011 |
| 15 | amount_bin_skew_index | 金额频次 | +0.0009 |
| 16 | counterparty_type_amount_divergence | 主体关联 | +0.0004 |
| 17 | tiny_amount_concentration | 金额频次 | +0.0000 |
| 18 | time_concentration_risk_index | 时间地域 | +0.0002 |

### 7.2 Stage 1（BABD-13）— 进行中

| 项目 | 值 |
|------|-----|
| Baseline recall (149 特征) | 0.9814 |
| 当前 recall | 0.9828 |
| 已接受特征 | 6 个 |
| 当前状态 | Round 4/5 运行中（日志在 `logs/stage1_run.log`） |

**注意**：因为服务器即将关闭，Stage 1 运行可能未完成。最终结果需要重新运行。

---

## 8. 辅助脚本

| 脚本 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `babd13_baseline.py` | BABD-13 基线评估 | BABD-13.csv | babd13_baseline.json |
| `baseline_eval_compare.py` | 105 vs 95 特征对比 | baseline_cache.parquet | baseline_compare.json |
| `build_all_113_features.py` | 构建 113 特征矩阵 | 3 个 CSV | 特征 CSV |
| `build_baseline_95_features.py` | 构建 95 基线特征 | 3 个 CSV | 特征 CSV |
| `compute_crk.py` | 计算 3 组 CR@K | parquet + features.json + CSV | crk_results.json |
| `compute_crk_exp3.py` | 计算实验 3 CR@K | 同上 | crk_results_exp3.json |
| `run_stage1_extraction.py` | Stage 1 知识提取 | stage1_knowledge/*.md | risk_factors/*.json |

---

## 9. 测试

项目有 17 个测试文件，覆盖所有核心模块：

```
tests/
├── test_auto_select.py       # 模型选择
├── test_builder.py           # Prompt 构建
├── test_cross_dim.py         # 跨维度传播
├── test_dimension_manager.py # 维度管理
├── test_domain_knowledge.py  # 领域知识加载
├── test_failure_memory.py    # 失败记忆
├── test_forward_selection.py # 前向选择
├── test_key_clue_extractor.py# 知识提取
├── test_llm_client.py        # LLM 客户端
├── test_main.py              # 主入口
├── test_model_eval.py        # 模型评估
├── test_provenance.py        # 溯源卡
├── test_react_agent.py       # ReAct Agent
├── test_report.py            # 报告生成
├── test_sandbox.py           # 沙盒执行
├── test_shap_feedback.py     # SHAP 反馈
├── test_statistics.py        # 统计验证
└── test_tools.py             # 工具执行
```

运行：`python -m pytest tests/ -v`

---

## 10. ModelSet 模型库

`ModelSet/` 目录包含 11 个候选 ML 模型的标准化实现，用于模型选择实验。

| 模型 | 文件 | 说明 |
|------|------|------|
| CatBoost | catboost_std.py | 类别特征原生支持 |
| LightGBM | lgbm_std.py | 高效梯度提升 |
| XGBoost | xgboost_model.py | 项目主力模型 |
| AC-XGB | acxgb_std.py | Adversarial Calibrated XGBoost |
| WAV-XGB | wav_xgb_std.py | Wavelet + XGBoost |
| GA2M | ga2m_std.py | 可解释加性模型 |
| Scorecard | scorecard_std.py | 评分卡模型 |
| DNN-Res | dnn_res_std.py | 残差深度网络 |
| GBDT-FM | gbdt_fm_std.py | GBDT + FM 组合 |
| Resample-Boost | resample_boost_std.py | 重采样增强 |
| Ensemble | ensemble_std.py | 多模型集成 |

统一接口通过 `runner.py` 调用：
```bash
cd ModelSet
python runner.py --train train.csv --test test.csv --output results.csv --models all
```

---

## 11. 未完成工作

详见 `todo.md`，主要待办：

### 11.1 实验

| 任务 | 状态 | 说明 |
|------|------|------|
| Stage 1 完整运行 | 进行中（可能因关机中断） | 需重新运行 |
| Baseline 对比 | 未开始 | OpenFE、CAAFE、AutoFeat、Featuretools |
| 多模型验证 | 未开始 | LightGBM、LogisticRegression |
| Ablation 实验 | 未开始 | 去领域知识/去验证/去反馈等 8 组 |
| 知识迁移实验 | 未开始 | Stage 1 知识用于 Stage 2 |
| 敏感性分析 | 未开始 | IV/KS 阈值、budget、temperature 等 |

### 11.2 已知 Bug

| 问题 | 状态 | 说明 |
|------|------|------|
| night_amount_short_term_risk | 未修复 | timestamp 列类型问题（str vs datetime），导致该特征在独立评估时加载失败。Stage 2 已接受的 18 个特征中实际只有 17 个能正常加载 |

### 11.3 论文

- 目标 venue: ICDM 2026 Applied Track（6月6日截止，已决定放弃本轮）
- 设计文档：`docs/KnowFeat_design.md`
- 实验结果：`RESULT.md`

---

## 12. 常见问题与踩坑记录

### Q1: 为什么排除了 10 个基线特征？

这 10 个特征存在数据泄漏或过拟合风险（如 `Timestamp_Hour` 直接泄漏时间信息，`Min_Received`/`Min_Sent` 对极端值过敏）。排除后基线 recall 从 0.9739 降到 0.9538，但 KnowFeat 在此基础上能恢复并超过（0.9737）。

### Q2: Stage 1 的 SW 列是什么？

BABD-13 的 SW 列是分类特征（SA=某地址类型, WA=某钱包类型）。原始值是字符串，必须编码为 0/1 整数，否则 XGBoost 会报错。代码中使用 `pd.api.types.is_numeric_dtype()` 检查，因为 pandas Arrow backend 的字符串类型 `dtype != object`。

### Q3: scale_pos_weight 怎么设置的？

**动态计算**：`neg_count / pos_count`。BABD-13 ≈ 2.88，SimECNY ≈ 59.0。之前硬编码为 59.0 导致 BABD-13 结果错误，已修复为动态计算。

### Q4: nohup 运行日志为什么看不到输出？

必须使用 `python -u`（unbuffered stdout），否则 stdout 缓冲导致日志文件不实时更新。

### Q5: 服务器上 git pull 失败？

服务器没有 SSH key，无法通过 git 从 GitHub 拉代码。解决方案：使用 `rsync` 或 `scp` 直接同步文件（`deploy.sh` 已封装）。

### Q6: 为什么 BABD-13 上提升这么小？

BABD-13 已有 149 个精心设计的特征，baseline recall 高达 0.9814。在此基础上提升极其困难，+0.0014 已经是实质性贡献。而 SimECNY 只有 95 个基线特征，提升空间大得多（+0.0199）。

### Q7: CR@K 在 BABD-13 上为什么很低？

BABD-13 正样本率 25.76%，top 5% 样本最多只能覆盖 ~19.4% 的正样本。CR@K 本身就不适合高正样本率数据集，BABD-13 应该以 recall/precision/F1 为主要指标。

---

## 13. 关键配置参数速查

### 验证阈值

| 验证级别 | 参数 | 默认值 | 文件位置 |
|---------|------|--------|---------|
| L1 | sandbox_timeout_sec | 30 | config.py:VerificationConfig |
| L1 | sandbox_max_retries | 2 | config.py:VerificationConfig |
| L2 | noise_n_random_features | 100 | config.py:VerificationConfig |
| L2 | redundancy_max | 0.7 | config.py:VerificationConfig |
| L2 | missing_rate_max | 0.3 | config.py:VerificationConfig |
| L3 | cv_folds | 5 | config.py:VerificationConfig |
| L3 | delta_recall_min | 0.0 | config.py:VerificationConfig |
| L3 | precision_min | 0.90 | config.py:VerificationConfig |

### 搜索参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| max_rounds | 10 | 最大搜索轮数 |
| max_features | 30 | 最大接受特征数 |
| convergence_rounds | 2 | 连续无新特征停止轮数 |
| initial_budget_per_dim | 3 | 每维度每轮初始特征预算 |
| failure_memory_inject_n | 10 | 注入 Prompt 的失败记录数 |
| cross_dim_top_k | 5 | 跨维度示例数 |

### XGBoost 参数（固定，在 model_eval.py 中）

| 参数 | 值 |
|------|-----|
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| scale_pos_weight | 动态 (neg/pos) |
| eval_metric | logloss |
| random_state | 42 |

### LLM 参数

| 参数 | 默认值 |
|------|--------|
| model | deepseek-chat |
| temperature | 0.4 |
| max_tokens | 4096 |
| timeout | 120s |
| max_retries | 3 |

---

## 附录 A: 文件依赖关系图

```
main.py
├── config.py
├── data/loader.py ← config.py
├── data/schema_stats.py ← data/loader.py
├── extractor/key_clue_extractor.py ← config.py, llm/client.py
├── llm/client.py ← config.py
├── prompt/builder.py ← config.py, data/loader.py, data/schema_stats.py, prompt/domain_knowledge.py
├── prompt/domain_knowledge.py ← config.py
├── agent/react_agent.py ← agent/tools.py, config.py, llm/client.py, prompt/builder.py
├── agent/tools.py ← config.py, data/loader.py, llm/client.py, search/forward_selection.py,
│                     verification/model_eval.py, verification/sandbox.py, verification/statistics.py
├── search/cross_dim.py ← config.py, prompt/builder.py, search/shap_feedback.py
├── search/dimension_manager.py ← config.py
├── search/failure_memory.py
├── search/forward_selection.py ← config.py, verification/model_eval.py
├── search/shap_feedback.py ← config.py
├── verification/model_eval.py ← config.py
├── verification/sandbox.py ← config.py
├── verification/statistics.py ← config.py
├── output/provenance.py
└── output/report.py
```

---

> 如有疑问，请联系原负责人或查看 `docs/KnowFeat_design.md` 获取更多设计细节。
