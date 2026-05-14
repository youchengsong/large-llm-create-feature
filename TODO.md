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

- [x] 1.1 `src/config.py`：集中管理超参数（岛屿配置、搜索参数、验证阈值、打分权重、XGBoost 参数、LLM 配置、缓刑池配置）
- [x] 1.2 `src/data/loader.py`：数据加载 + schema 提取 + 分布差异提取 + 三表关联统计
- [x] 1.3 `src/data/merge.py`：三表关联（tx_id 直接 merge / wallet_id 分 src/dst 两侧 merge）
- [x] 1.4 `src/data/baseline.py`：人工基线特征提取 + 基线代码示例注入
- [x] 1.5 `src/prompt/domain_knowledge.py`：领域知识结构化（交易模式 + 专家访谈 + 风险指标 + 检测规则）
- [x] 1.6 `src/prompt/builder.py`：Prompt 组装（含 CoT 推理、分布差异、关联统计、基线代码、数据探索说明）
- [x] 1.7 `src/evaluation/sandbox.py`：代码安全沙盒（AST 检查 + 子进程隔离 + 30s 超时）
- [x] 1.8 `src/evaluation/metrics.py`：特征评估指标（MI、冗余度、Cramer's V、Bootstrap χ²、Delta PR-AUC、SHAP、CV PR-AUC）
- [x] 1.9 `src/evaluation/validator.py`：L1-L3 验证流水线 + 综合打分 + 反向淘汰函数
- [x] 1.10 `src/llm/interface.py`：LLM 可插拔接口（OpenAI / Gemini / DeepSeek / Local / Mock）
- [x] 1.11 `src/llm/parser.py`：LLM 输出解析（特征 JSON + 查询动作分离）

---

## Phase 2：核心搜索循环

- [x] 2.1 `src/search/island.py`：岛屿数据结构 + Boltzmann 采样 + 迁移
- [x] 2.2 `src/search/memory.py`：失败记忆
- [x] 2.3 `src/search/shap_feedback.py`：SHAP 反馈 + 动态岛屿候选预算分配
- [x] 2.4 `src/search/pending_pool.py`：缓刑池管理（age/consecutive_rejects/容量控制/淘汰）
- [x] 2.5 `src/data/stats_query.py`：按需统计查询沙盒（方案 B）
- [x] 2.6 `src/search/engine.py`：搜索引擎主程序
  - 单岛屿步骤 + L1 失败重试
  - 多轮岛内对话（intra_island_turns=3）
  - 按需数据查询执行
  - L2 通过 / L3 未通过 → 进缓刑池
  - 每轮末缓刑池反向淘汰评估
  - 通过 L3 后生成可解释性报告
  - SHAP 动态预算分配
- [x] 2.7 `src/main.py`：主入口（加载数据 → 基线 → Prompt → 搜索 → 输出）
- [x] 2.8 测试套件：sandbox / metrics / island / parser / e2e_mock / pending_pool（171 个测试）

---

## Phase 3：最终验证与产出

- [x] 3.1 `src/evaluation/stability.py`：L4 稳定性验证（Nogueira 指数，10 次 Bootstrap）
- [x] 3.2 `src/evaluation/experiments.py`：5 组增量贡献实验
- [x] 3.3 `src/report/feature_dictionary.py`：特征字典生成
- [x] 3.4 `src/report/visualization.py`：结果可视化（SHAP plot、实验柱状图、热力图等）
- [x] 3.5 `src/report/generator.py`：验证报告生成
- [x] 3.6 `src/phase3_main.py`：Phase 3 主入口
- [x] 3.7 `tests/test_stability.py` + `tests/test_experiments.py`

---

## 完成标准

所有 Phase 0-3 任务完成后，应该能做到：
1. `python src/main.py` 一键运行完整搜索流程（Phase 0-2）
2. `python src/phase3_main.py` 一键运行最终验证与产出（Phase 3）
3. 替换为真实 LLM 后，能产出经过 L1-L4 四级验证的特征列表
4. 特征有两条接受路径：直通（L3 通过）和缓刑池（反向淘汰通过）
5. 每个特征有完整的 code + explanation + 评估指标 + 可解释性报告
6. 搜索日志记录每轮每岛屿的详细过程（含缓刑池状态）
7. 输出完整的验证报告、特征字典和可视化图表
