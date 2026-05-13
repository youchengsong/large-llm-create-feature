# Phase 0-2 实现 TODO

> 按顺序逐项实现，每完成一项打勾。代码统一放在 `src/` 目录下。
> 参考方案：`LLM_FEATURE_PLAN.md`

---

## Phase 0：数据生成

### 0.1 环境准备
- [x] 确认依赖已安装：pandas, numpy, networkx, xgboost, shap, scikit-learn, scipy, openpyxl
- [x] 创建项目代码目录结构：`src/`, `src/data/`, `src/prompt/`, `src/search/`, `src/evaluation/`, `src/llm/`, `data/`（存放生成的 CSV）

### 0.2 运行 SimECNY 生成三表数据
- [x] 运行开户模块：`inputs/SimECNY/open/open_wallet_distribution_based.py` → 输出 `data/accounts.csv`（100K 钱包，16 列）
- [ ] 运行图生成：`inputs/SimECNY/transaction/generate_static_edges.py` → 输出边表文件
- [ ] 运行交易生成：`inputs/SimECNY/transaction/simplified_main.py` → 输出 `data/transactions.csv`
- [ ] 修复销户模块硬编码路径（`inputs/SimECNY/close/generate_wallet_close_data.py` 第 9-10 行的 `/home/lwy/...` → 相对路径），运行 → 输出 `data/close.csv`

### 0.3 数据验证
- [ ] 写 `src/data/validate_data.py`，检查：
  - 三表字段名与 LLM_FEATURE_PLAN.md §1.3 一致
  - `is_risk` 正样本比例 ~5%
  - `wallet_id` 在三表间的关联完整性（交易表的 src/dst 能匹配到开户表）
  - 打印各表行数、列数、缺失值统计

---

## Phase 1：基础设施搭建

### 1.1 数据加载模块
- [ ] 写 `src/data/loader.py`：
  - `load_all_tables(data_dir) -> (open_df, tx_df, close_df)` 加载三表
  - `extract_schema(df, table_name) -> str` 提取 schema 文本（字段名、类型、取值范围、统计摘要）供 prompt 用
  - `extract_label(tx_df) -> Series` 提取 `is_risk` 标签列

### 1.2 三表关联模块
- [ ] 写 `src/data/merge.py`：
  - `merge_feature_to_transactions(feature_df, tx_df) -> DataFrame`
  - 逻辑：如果 feature_df 含 `tx_id` 列 → 直接 merge on tx_id；如果含 `wallet_id` 列 → 分别 merge 到 src 和 dst 两侧，列名加 `_src` / `_dst` 后缀
  - 参考 LLM_FEATURE_PLAN.md §3.4 的代码

### 1.3 人工基线特征提取
- [ ] 写 `src/data/baseline.py`：
  - `run_baseline(open_df, tx_df, close_df) -> DataFrame` 调用 `inputs/baseline/feature_engineering.py` 中的 `feature_eng()` 生成人工基线特征矩阵
  - `get_baseline_feature_names() -> list[str]` 返回 40+ 人工特征名称列表（写入 prompt 告诉 LLM 不要重复）
  - `compute_baseline_prauc(X_baseline, y) -> float` 用 XGBoost 5 折分层 CV 计算基线 PR-AUC

### 1.4 领域知识结构化
- [ ] 写 `src/prompt/domain_knowledge.py`：
  - `load_transaction_patterns() -> str` 返回 8 种交易模式说明文本（3 正常 + 5 风险，从 LLM_FEATURE_PLAN.md §1.5 提取，硬编码即可）
  - `load_expert_interview() -> str` 返回 4 类洗钱模式压缩文本
  - `load_risk_indicators() -> str` 读取 `inputs/domain_knowledge/风险分类表_38指标.xlsx`，提取 38 个指标为压缩表格文本
  - `load_detection_rules() -> list[dict]` 读取 `inputs/domain_knowledge/检测规则_15条.xlsx`，返回 15 条规则
  - `rules_to_python_examples(rules, n=2) -> str` 将规则转为 Python 函数示例文本（冷启动用）

### 1.5 Prompt 组装
- [ ] 写 `src/prompt/builder.py`：
  - `build_system_prompt(schema_text, transaction_patterns, expert_knowledge, risk_indicators, baseline_features, constraints) -> str` 组装完整 System Prompt（~5100 token），参考 LLM_FEATURE_PLAN.md §4.3 模板
  - `build_user_prompt_cold_start(island_name, island_fields, seed_examples) -> str` 第 1 轮 User Prompt（无 SHAP，含维度引导 + 种子示例）
  - `build_user_prompt(island_name, island_fields, shap_feedback, top_features, failed_features) -> str` 第 2+ 轮 User Prompt（含维度引导 + SHAP + 示例 + 失败记录）
  - 每个函数末尾打印 token 数估算（按 1 中文字 ≈ 2 token 粗估）

### 1.6 特征执行沙盒
- [ ] 写 `src/evaluation/sandbox.py`：
  - `ALLOWED_MODULES = {'pandas', 'numpy', 'scipy', 'sklearn'}` 安全白名单
  - `check_code_safety(code: str) -> (bool, str)` AST 解析检查是否有禁止的 import/调用（os, sys, subprocess, open, exec, eval, 网络请求）
  - `execute_feature_code(code: str, open_df, tx_df, close_df, timeout=30) -> (DataFrame | None, str | None)` 在沙盒中执行代码，返回 (结果DataFrame, 错误信息)，超时 30s
  - `validate_output(feature_df: DataFrame) -> (bool, str)` 检查输出：含 wallet_id 或 tx_id 列、非空、非全 0、非全 1、NaN < 50%

### 1.7 特征评估模块
- [ ] 写 `src/evaluation/metrics.py`：
  - `compute_mutual_info(feature, label) -> float` 互信息
  - `compute_redundancy(feature, existing_features) -> float` 最大相关系数 |corr|
  - `compute_cramers_v(feature, label) -> float` Cramer's V
  - `compute_bootstrap_chi2(feature, label, n_resamples=2500) -> float` Bootstrap 卡方检验 p 值
  - `compute_delta_prauc(X_existing, new_feature, y) -> float` XGBoost 5 折 CV 增量 PR-AUC
  - `compute_shap_rank(X_all, y, feature_name) -> int` 新特征在完整模型中的 SHAP 重要性排名
  - `normalize_score(value, all_values) -> float` min-max 归一化到 [0,1]

- [ ] 写 `src/evaluation/validator.py`：
  - `run_l1_validation(code, open_df, tx_df, close_df) -> (DataFrame | None, str)` L1 代码验证，返回执行结果或错误
  - `run_l2_validation(feature, label, existing_features) -> (bool, dict)` L2 统计验证（MI > 0.01, |corr| < 0.9, Cramer's V > 0.05, bootstrap p < 0.05），返回通过/拒绝 + 各指标值
  - `run_l3_validation(X_existing, new_feature, y, baseline_prauc) -> (bool, dict)` L3 模型验证（ΔPR-AUC ≥ 0.005），返回通过/拒绝 + 指标
  - `compute_composite_score(shap_rank, mi, redundancy, delta_prauc, all_scores) -> float` 归一化后加权打分：0.35*SHAP + 0.25*MI + 0.25*(1-冗余度) + 0.15*ΔPR-AUC

### 1.8 LLM 可插拔接口
- [ ] 写 `src/llm/interface.py`：
  - `class LLMInterface` 基类，`generate(system_prompt, user_prompt) -> str` 抽象方法
  - `class OpenAILLM(LLMInterface)` GPT 系列（gpt-4o, gpt-4o-mini 等），用 openai SDK
  - `class GeminiLLM(LLMInterface)` Google Gemini（gemini-2.5-flash, gemini-2.5-pro 等），用 google-genai SDK
  - `class DeepSeekLLM(LLMInterface)` DeepSeek（deepseek-chat, deepseek-coder 等），用 OpenAI 兼容接口，base_url=https://api.deepseek.com
  - `class LocalLLM(LLMInterface)` 开源模型（Qwen/Llama/Mistral 等），通过 OpenAI 兼容接口调用本地 vllm/ollama 服务，支持自定义 base_url + model_name
  - `class MockLLM(LLMInterface)` 测试用 mock，返回预定义的 JSON 输出
  - 所有实现类统一接口：`__init__(model_name, api_key, base_url=None, **kwargs)`
- [ ] 写 `src/llm/parser.py`：
  - `parse_llm_output(raw_text) -> list[dict]` 从 LLM 原始输出中提取 JSON 数组，每个 dict 含 feature_name, code, explanation, domain_source 等字段
  - 处理常见的 LLM 输出格式问题（markdown 代码块包裹、多余文本、JSON 不完整等）
  - `extract_json_from_text(text) -> str` 从混杂文本中提取 JSON 部分（正则匹配 `[...]` 或 ````json...```）

### 1.9 配置文件
- [ ] 写 `src/config.py`：集中管理所有超参数
  - 岛屿配置：4 个维度岛屿的名称、探索方向描述、重点字段列表
  - 搜索参数：m=4, b=3, k=2, T=10, tau（温度）
  - 验证阈值：MI > 0.01, |corr| < 0.9, Cramer's V > 0.05, ΔPR-AUC ≥ 0.005, bootstrap p < 0.05
  - 打分权重：SHAP 0.35, MI 0.25, 冗余度 0.25, ΔPR-AUC 0.15
  - XGBoost 参数：scale_pos_weight=19, eval_metric='aucpr'
  - LLM 配置：provider（openai/gemini/deepseek/local）、model_name、api_key（从环境变量读取）、base_url（local 模式必填）、temperature、max_tokens、超时等
  - 路径配置：数据目录、输出目录

---

## Phase 2：核心搜索循环

### 2.1 岛屿数据结构
- [ ] 写 `src/search/island.py`：
  - `class Island`：name, dimension, fields, memory(成功特征列表), 每个特征含 {name, code, explanation, score, shap_rank, mi, redundancy, delta_prauc}
  - `add_feature(feature_dict)` 加入记忆库
  - `boltzmann_sample(k=2, tau=1.0) -> list[dict]` Boltzmann 采样选 k 个 in-context 示例，P_i = exp(score_i/τ) / Σexp(score/τ)
  - `class IslandManager`：管理 4 个维度岛屿，提供 `get_all_islands()`, `migrate_best(feature_dict)` 迁移最优特征到所有岛屿
- [ ] 写 `src/search/memory.py`：
  - `class FailureMemory`：全局失败记忆库，记录 {feature_name, code, failure_reason}
  - `add_failure(name, code, reason)`, `get_recent(n=5) -> list` 获取最近 n 条失败记录（供 prompt 用）
  - `to_prompt_text() -> str` 格式化为 prompt 文本

### 2.2 SHAP 反馈生成
- [ ] 写 `src/search/shap_feedback.py`：
  - `generate_shap_feedback(X_all, y) -> dict` 训练 XGBoost → 计算 SHAP → 返回 {top_10: [(name, importance)], weak_dimensions: [str]}
  - `shap_feedback_to_prompt(feedback_dict, island_dimensions) -> str` 将 SHAP 反馈转为 prompt 文本，标注哪些维度（对应哪些岛屿）是弱势

### 2.3 搜索循环主程序
- [ ] 写 `src/search/engine.py`：
  - `class SearchEngine`：核心搜索引擎
  - `__init__(config, llm, open_df, tx_df, close_df, X_baseline, y, baseline_prauc)` 初始化
  - `run_single_island_step(island, system_prompt, shap_feedback) -> list[dict]` 单个岛屿一轮：Boltzmann 采样 → 构建 prompt → 调用 LLM → 解析输出 → 返回候选列表
  - `evaluate_candidates(candidates) -> list[dict]` 对候选列表逐个执行 L1→L2→L3 验证，返回通过验证的特征（含 score）
  - `run_round(round_num) -> list[dict]` 一轮完整迭代：SHAP 分析 → 4 岛屿各生成 → 评估 → 记忆更新 → 迁移
  - `check_convergence() -> bool` 连续 2 轮无新特征通过 → True
  - `run(max_rounds=10) -> list[dict]` 主循环，返回最终接受的全部特征列表
  - 每轮打印日志：轮次、各岛屿生成数、通过验证数、当前总特征数、当前 PR-AUC

### 2.4 错误重试机制
- [ ] 在 `src/search/engine.py` 的 `run_single_island_step` 中实现：
  - LLM 生成的代码 L1 失败时，将错误信息反馈给 LLM 重试（最多 2 次）
  - 构建 retry prompt：`"你上一次生成的代码执行失败：\n{error_msg}\n请修正并重新生成。"`

### 2.5 主入口
- [ ] 写 `src/main.py`：
  - 加载数据 → 运行基线 → 打印基线 PR-AUC
  - 组装 System Prompt
  - 初始化 LLM、SearchEngine
  - 运行搜索循环
  - 输出最终特征列表到 `output/features.json`（每个特征含 name, code, explanation, domain_source, score, shap_rank, mi, delta_prauc）
  - 输出搜索日志到 `output/search_log.json`（每轮的详细记录）

### 2.6 单元测试
- [ ] 写 `tests/test_sandbox.py`：测试安全白名单（禁止 os/sys）、超时、输出格式检查
- [ ] 写 `tests/test_metrics.py`：用简单数据测试 MI、冗余度、Cramer's V、ΔPR-AUC 计算
- [ ] 写 `tests/test_island.py`：测试 Boltzmann 采样、迁移逻辑
- [ ] 写 `tests/test_parser.py`：测试 LLM 输出解析（正常 JSON、markdown 包裹、格式错误等）
- [ ] 用 MockLLM 跑一次完整流程（2 轮），验证 pipeline 端到端可运行

---

## 完成标准

所有 Phase 0-2 任务完成后，应该能做到：
1. `python src/main.py` 一键运行完整搜索流程（用 MockLLM 测试）
2. 替换为真实 LLM 后，能产出经过 L1-L3 验证的特征列表
3. 每个特征有完整的 code + explanation + 评估指标
4. 搜索日志记录每轮每岛屿的详细过程
