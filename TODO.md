# Phase 0-3 实现 TODO

> 按顺序逐项实现，每完成一项打勾。代码统一放在 `src/` 目录下。
> 参考方案：`LLM_FEATURE_PLAN.md`
> 注意：开发阶段使用 1K 钱包数据集（`conf_wallet_distribution_based_dev.json`），后续切换 100K。

---

## Phase 0：数据生成（已手动完成）

- [x] 0.1 环境准备：依赖安装 + 目录结构创建
- [x] 0.2 开户数据生成：`data/accounts.csv`（1K 钱包，16 列）
- [x] 0.3 图生成：`inputs/SimECNY/transaction/wallet_static_edges_distribution_based.csv`
- [x] 0.4 交易数据生成：`data/transactions.csv`
- [x] 0.5 销户数据生成：`data/close.csv`
- [x] 0.6 数据验证：`src/data/validate_data.py`

---

## Phase 1：基础设施搭建

- [x] 1.1 写 `src/config.py`：集中管理所有超参数。包含：岛屿配置（4 个维度岛屿的名称、探索方向描述、重点字段列表）；搜索参数（m=4, b=3, k=2, T=10, tau 温度）；验证阈值（MI > 0.01, |corr| < 0.9, Cramer's V > 0.05, ΔPR-AUC >= 0.005, bootstrap p < 0.05）；打分权重（SHAP 0.35, MI 0.25, 冗余度 0.25, ΔPR-AUC 0.15）；XGBoost 参数（scale_pos_weight=19, eval_metric='aucpr'）；LLM 配置（provider: openai/gemini/deepseek/local, model_name, api_key 从环境变量读取, base_url, temperature, max_tokens, 超时）；路径配置（数据目录、输出目录）

- [x] 1.2 写 `src/data/loader.py`：数据加载模块。`load_all_tables(data_dir) -> (open_df, tx_df, close_df)` 加载三表；`extract_schema(df, table_name) -> str` 提取 schema 文本（字段名、类型、取值范围、统计摘要）供 prompt 用；`extract_label(tx_df) -> Series` 提取 `is_risk` 标签列

- [x] 1.3 写 `src/data/merge.py`：三表关联模块。`merge_feature_to_transactions(feature_df, tx_df) -> DataFrame`，逻辑：如果 feature_df 含 `tx_id` 列则直接 merge on tx_id；如果含 `wallet_id` 列则分别 merge 到 src 和 dst 两侧，列名加 `_src` / `_dst` 后缀。参考 LLM_FEATURE_PLAN.md §3.4 的代码

- [x] 1.4 写 `src/data/baseline.py`：人工基线特征提取。`run_baseline(open_df, tx_df, close_df) -> DataFrame` 调用 `inputs/baseline/feature_engineering.py` 中的 `feature_eng()` 生成人工基线特征矩阵；`get_baseline_feature_names() -> list[str]` 返回 40+ 人工特征名称列表；`compute_baseline_prauc(X_baseline, y) -> float` 用 XGBoost 5 折分层 CV 计算基线 PR-AUC

- [x] 1.5 写 `src/prompt/domain_knowledge.py`：领域知识结构化。`load_transaction_patterns() -> str` 返回 8 种交易模式说明文本（3 正常 + 5 风险，从 LLM_FEATURE_PLAN.md §1.5 提取，硬编码即可）；`load_expert_interview() -> str` 返回 4 类洗钱模式压缩文本；`load_risk_indicators() -> str` 读取 `inputs/domain_knowledge/风险分类表_38指标.xlsx` 提取 38 个指标为压缩表格文本；`load_detection_rules() -> list[dict]` 读取 `inputs/domain_knowledge/检测规则_15条.xlsx` 返回 15 条规则；`rules_to_python_examples(rules, n=2) -> str` 将规则转为 Python 函数示例文本

- [x] 1.6 写 `src/prompt/builder.py`：Prompt 组装。`build_system_prompt(schema_text, transaction_patterns, expert_knowledge, risk_indicators, baseline_features, constraints) -> str` 组装完整 System Prompt（~5100 token），参考 LLM_FEATURE_PLAN.md §4.3 模板；`build_user_prompt_cold_start(island_name, island_fields, seed_examples) -> str` 第 1 轮 User Prompt；`build_user_prompt(island_name, island_fields, shap_feedback, top_features, failed_features) -> str` 第 2+ 轮 User Prompt；每个函数末尾打印 token 数估算

- [x] 1.7 写 `src/evaluation/sandbox.py`：特征执行沙盒。`ALLOWED_MODULES = {'pandas', 'numpy', 'scipy', 'sklearn'}` 安全白名单；`check_code_safety(code: str) -> (bool, str)` AST 解析检查禁止的 import/调用（os, sys, subprocess, open, exec, eval, 网络请求）；`execute_feature_code(code: str, open_df, tx_df, close_df, timeout=30) -> (DataFrame | None, str | None)` 沙盒执行代码；`validate_output(feature_df: DataFrame) -> (bool, str)` 检查输出格式

- [x] 1.8 写 `src/evaluation/metrics.py`：特征评估指标。`compute_mutual_info(feature, label) -> float`；`compute_redundancy(feature, existing_features) -> float` 最大 |corr|；`compute_cramers_v(feature, label) -> float`；`compute_bootstrap_chi2(feature, label, n_resamples=2500) -> float`；`compute_delta_prauc(X_existing, new_feature, y) -> float` XGBoost 5 折 CV 增量 PR-AUC；`compute_shap_rank(X_all, y, feature_name) -> int`；`normalize_score(value, all_values) -> float`

- [x] 1.9 写 `src/evaluation/validator.py`：特征验证流水线。`run_l1_validation(code, open_df, tx_df, close_df) -> (DataFrame | None, str)` L1 代码验证；`run_l2_validation(feature, label, existing_features) -> (bool, dict)` L2 统计验证；`run_l3_validation(X_existing, new_feature, y, baseline_prauc) -> (bool, dict)` L3 模型验证；`compute_composite_score(shap_rank, mi, redundancy, delta_prauc, all_scores) -> float` 加权打分

- [x] 1.10 写 `src/llm/interface.py`：LLM 可插拔接口。`class LLMInterface` 基类 + `generate(system_prompt, user_prompt) -> str`；`class OpenAILLM` GPT 系列用 openai SDK；`class GeminiLLM` 用 google-genai SDK；`class DeepSeekLLM` 用 OpenAI 兼容接口 base_url=https://api.deepseek.com；`class LocalLLM` 开源模型通过 OpenAI 兼容接口；`class MockLLM` 测试用 mock；统一 `__init__(model_name, api_key, base_url=None, **kwargs)`

- [x] 1.11 写 `src/llm/parser.py`：LLM 输出解析。`parse_llm_output(raw_text) -> list[dict]` 提取 JSON 数组；处理 markdown 代码块包裹、多余文本、JSON 不完整等；`extract_json_from_text(text) -> str` 正则匹配提取 JSON

---

## Phase 2：核心搜索循环

- [x] 2.1 写 `src/search/island.py`：岛屿数据结构。`class Island` 含 name, dimension, fields, memory；`add_feature(feature_dict)`；`boltzmann_sample(k=2, tau=1.0) -> list[dict]` Boltzmann 采样；`class IslandManager` 管理 4 个维度岛屿 + `migrate_best(feature_dict)` 迁移

- [x] 2.2 写 `src/search/memory.py`：失败记忆。`class FailureMemory` 记录 {feature_name, code, failure_reason}；`add_failure(name, code, reason)`；`get_recent(n=5) -> list`；`to_prompt_text() -> str`

- [x] 2.3 写 `src/search/shap_feedback.py`：SHAP 反馈生成。`generate_shap_feedback(X_all, y) -> dict` 训练 XGBoost + SHAP；`shap_feedback_to_prompt(feedback_dict, island_dimensions) -> str` 转 prompt 文本

- [x] 2.4 写 `src/search/engine.py`：搜索引擎主程序。`class SearchEngine`；`run_single_island_step` 单岛屿一轮（含 L1 失败时 LLM 重试最多 2 次）；`evaluate_candidates` L1→L2→L3 验证；`run_round` 一轮完整迭代；`check_convergence` 连续 2 轮无新特征；`run(max_rounds=10)` 主循环

- [x] 2.5 写 `src/main.py`：主入口。加载数据 → 运行基线 → 组装 System Prompt → 初始化 LLM + SearchEngine → 运行搜索循环 → 输出 `output/features.json` + `output/search_log.json`

- [x] 2.6 写 `tests/test_sandbox.py`：测试安全白名单、超时、输出格式检查

- [x] 2.7 写 `tests/test_metrics.py`：用简单数据测试 MI、冗余度、Cramer's V、ΔPR-AUC

- [x] 2.8 写 `tests/test_island.py`：测试 Boltzmann 采样、迁移逻辑

- [x] 2.9 写 `tests/test_parser.py`：测试 LLM 输出解析

- [x] 2.10 用 MockLLM 跑一次完整流程（2 轮），验证 pipeline 端到端可运行

---

---

## Phase 3：最终验证与产出

- [x] 3.1 写 `src/evaluation/stability.py`：L4 稳定性验证。`compute_nogueira_stability(selection_sets, n_features) -> float` 计算 Nogueira 稳定性指数；`run_l4_stability(X_final, y, n_bootstrap=10, top_k=20) -> dict` 10 次 bootstrap 训练 XGBoost + SHAP，收集 top-20 特征排名，计算稳定性指数（阈值 > 0.8）。返回 `{stability_index, per_round_rankings, pass: bool}`

- [x] 3.2 写 `src/evaluation/experiments.py`：5 组增量贡献实验。`run_ablation_experiments(X_baseline, X_llm, y, baseline_feature_names, llm_feature_names) -> dict` 实现 5 组对比实验：
  - 实验 1（基线）：仅人工特征 → XGBoost 5 折 CV → PR-AUC_baseline
  - 实验 2（增量）：人工 + LLM 特征 → PR-AUC_augmented
  - 实验 3（独立）：仅 LLM 特征 → PR-AUC_llm_only
  - 实验 4（随机对照）：人工 + 等量随机特征 → PR-AUC_random（排除过拟合）
  - 实验 5（消融）：逐个移除 LLM 特征 → 识别最关键特征
  - 报告指标：增量 PR-AUC、独立 PR-AUC、随机对照差、各特征 SHAP 排名、消融贡献
  - 所有实验使用 5 折分层 CV，报告均值 ± 标准差

- [x] 3.3 写 `src/report/feature_dictionary.py`：特征字典生成。`generate_feature_dictionary(features_data, X_final, y) -> dict` 为每个通过验证的特征生成完整字典条目：Python 代码 + 可解释性说明 + 领域知识来源 + SHAP 排名 + 统计指标（MI, Cramer's V, Delta PR-AUC）；`export_feature_dictionary(dictionary, output_path)` 导出为 JSON + 可读 Markdown 格式

- [x] 3.4 写 `src/report/visualization.py`：结果可视化。生成以下图表：
  - SHAP summary plot（特征重要性）
  - 5 组实验 PR-AUC 对比柱状图（含误差线）
  - 消融实验热力图
  - 特征稳定性 bootstrap 分布图
  - 各维度岛屿贡献饼图（每个岛产出多少通过验证的特征）
  - 输出到 `output/plots/` 目录

- [x] 3.5 写 `src/report/generator.py`：验证报告生成。`generate_report(stability_results, experiment_results, feature_dictionary, search_log) -> str` 生成完整验证报告（Markdown），包含：
  - 搜索过程摘要（轮数、各岛屿贡献、收敛情况）
  - 特征字典表格
  - 5 组实验结果及分析
  - L4 稳定性验证结果
  - 嵌入可视化图表引用
  - 导出到 `output/report.md`

- [x] 3.6 写 `src/phase3_main.py`：Phase 3 主入口。加载 Phase 2 产出的 `output/features.json` → 重建特征矩阵 → 运行 L4 稳定性验证 → 运行 5 组增量实验 → 生成特征字典 → 生成可视化 → 生成验证报告。支持 `--features-json`、`--data-dir`、`--output-dir` 参数

- [x] 3.7 写 `tests/test_stability.py`：测试 Nogueira 稳定性指数计算、bootstrap 逻辑

- [x] 3.8 写 `tests/test_experiments.py`：测试 5 组实验流程（用小数据集验证正确性）

- [ ] 3.9 用现有的 2 个通过特征跑一次完整 Phase 3 流程，验证端到端可运行

---

## 完成标准

所有 Phase 0-3 任务完成后，应该能做到：
1. `python src/main.py` 一键运行完整搜索流程（Phase 0-2）
2. `python src/phase3_main.py` 一键运行最终验证与产出（Phase 3）
3. 替换为真实 LLM 后，能产出经过 L1-L4 四级验证的特征列表
4. 每个特征有完整的 code + explanation + 评估指标 + 稳定性验证
5. 搜索日志记录每轮每岛屿的详细过程
6. 输出完整的验证报告、特征字典和可视化图表
