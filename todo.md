# ICDM 2026 论文实验 TODO

> KnowFeat: Knowledge-Traceable Feature Engineering
>
> 实验设计参考：`docs/experiment_design.md`
>
> 最后更新：2026-05-30

---

## 数据集总览

| # | 数据集 | 规模 | 特征数 | 类型 | 状态 |
|---|--------|------|--------|------|------|
| 1 | **数字人民币** | 138万交易 | 95 | AML核心 | ✅ Stage2已完成 |
| 2 | credit-g | 1,000 | 20 | 公开 | ⬜ 待运行 |
| 3 | diabetes | 768 | 8 | 公开 | ⬜ 待运行 |
| 4 | blood-transfusion | 748 | 4 | 公开 | ⬜ 待运行 |
| 5 | heart | 918 | 11 | 公开 | ⬜ 待运行 |
| 6 | bank-marketing | 45,211 | 16 | 公开 | ⬜ 待运行 |
| 7 | breast-w | 699 | 9 | 公开 | ⬜ 待运行 |

领域知识文件：`inputs/public_datasets/*/domain_knowledge.json`（✅ 6个已创建）

---

## Phase 0: 基础设施准备

> 在跑任何实验之前，必须先完成

- [x] **P0.1** 公开数据集通用适配器 ✅ commit 8a1e8eb
  - `knowfeat/config.py`: 新增 `PublicDatasetConfig`, `stage="generic"`
  - `knowfeat/data/loader.py`: 新增 `load_public_dataset()` (CSV → Stage1Data)
  - `knowfeat/prompt/domain_knowledge.py`: 新增 JSON 知识加载器 + 格式化 + 注册表
  - `knowfeat/prompt/builder.py`: 新增通用角色模板 `_build_role_generic()`
  - `knowfeat/agent/tools.py`: 新增 `build_generic_tool_specs()`, `row_id` 支持
  - `knowfeat/verification/sandbox.py`: `REQUIRED_ID_COLS` 加入 `row_id`
  - `scripts/run_knowfeat_public.py`: 便捷入口，6 个数据集配置齐全

- [x] **P0.2** 公开数据集下载脚本 ✅ commit 8a1e8eb
  - `scripts/download_public_datasets.py`: OpenML 5个 + Kaggle heart
  - 统一存放到 `data/public/` 目录，CSV 格式

- [ ] **P0.2b** 执行数据集下载
  - 运行: `.venv/bin/python scripts/download_public_datasets.py`
  - 验证: 确认 `data/public/` 下有 6 个 CSV 文件（credit_g.csv, diabetes.csv, blood_transfusion.csv, heart.csv, bank_marketing.csv, breast_w.csv）
  - 验证: 每个 CSV 行数与数据集总览表一致
  - 完成后在数据集总览表中更新状态为「✅ 已下载」

- [x] **P0.3** 统一评估框架 ✅ commit 8a1e8eb
  - `scripts/evaluate.py`: 80/20 split × 5 seeds × XGBoost
  - 数字人民币指标：Recall, Precision, F1, AUC
  - 公开数据集指标：AUC, Accuracy
  - 输出标准化 JSON 到 `outputs/eval_results/`
  - 显著性检验：Welch's t-test (p=0.05)

- [ ] **P0.4** Baseline 方法安装与适配
  - **P0.4a** 经典 AutoFE baseline:
    - 运行: `.venv/bin/pip install openfe autofeat`
    - 验证: `python -c "import openfe; import autofeat; print('OK')"`
  - **P0.4b** CAAFE 适配 DeepSeek API:
    - Clone: `git clone https://github.com/noahho/CAAFE.git baselines/CAAFE`
    - 找到 CAAFE 中调用 `openai.OpenAI()` 的地方，添加 `base_url` 和 `api_key` 参数
    - base_url 用环境变量 `KNOWFEAT_BASE_URL`（默认 `https://api.deepseek.com/v1`）
    - api_key 用环境变量 `KNOWFEAT_API_KEY`
    - model 改为 `deepseek-chat`
    - 写一个 wrapper 脚本 `scripts/run_caafe.py`，接受 --dataset 参数，输出特征到 CSV
  - **P0.4c** FeatLLM 适配 DeepSeek API:
    - Clone: `git clone https://github.com/Sungwon-Han/FeatLLM.git baselines/FeatLLM`
    - FeatLLM 使用旧版 `openai.ChatCompletion.create()` API，需升级为现代 SDK:
      - `openai.ChatCompletion.create(...)` → `client.chat.completions.create(...)`
      - 初始化 `client = openai.OpenAI(api_key=..., base_url=...)`
    - model 改为 `deepseek-chat`
    - 写一个 wrapper 脚本 `scripts/run_featllm.py`，接受 --dataset 参数，输出特征到 CSV
  - **P0.4d** 验证所有 baseline 能跑通:
    - 用 breast-w（最小数据集）测试每个 baseline
    - 确认 OpenFE、AutoFeat 可直接运行
    - 确认 CAAFE、FeatLLM 可通过 DeepSeek API 运行
    - 如果某个 baseline 无法适配，记录原因并在实验中标注

---

## Phase 1: Exp 1 — 主性能对比 (Performance Comparison)

> 目标：7个数据集 × 6种方法 → 两张性能对比表
> 评估命令: `.venv/bin/python scripts/evaluate.py --dataset <name> --method <method> [--features-csv <path>]`
> 结果存放: `outputs/eval_results/<dataset>_<method>.json`

### 1A: 数字人民币上的 Baseline

- [ ] **T1.1** No-FE baseline — 用 `scripts/evaluate.py --dataset simecny --baseline`
  - 已有参考数据: recall=0.9538，需用新的 80/20×5seeds 评估框架重新跑
- [ ] **T1.2** OpenFE baseline — 在 95 特征上运行 OpenFE 自动搜索 top 30 特征
- [ ] **T1.3** AutoFeat baseline — 在 95 特征上运行 AutoFeat 多项式+L1 选择
- [ ] **T1.4** CAAFE baseline — 用 `scripts/run_caafe.py --dataset simecny` 生成特征
- [ ] **T1.5** FeatLLM baseline — 用 `scripts/run_featllm.py --dataset simecny` 生成特征
- [ ] **T1.6** KnowFeat — 已有参考数据: recall=0.9737, 18/30特征被接受，需用新评估框架重新跑

### 1B: 6个公开数据集上的全部方法

> 依赖 P0.2b（数据已下载）和 P0.4（baseline 已安装）
> 每个数据集对 6 种方法分别运行评估脚本

- [ ] **T1.7** credit-g: No-FE, OpenFE, AutoFeat, CAAFE, FeatLLM, KnowFeat
- [ ] **T1.8** diabetes: No-FE, OpenFE, AutoFeat, CAAFE, FeatLLM, KnowFeat
- [ ] **T1.9** blood-transfusion: No-FE, OpenFE, AutoFeat, CAAFE, FeatLLM, KnowFeat
- [ ] **T1.10** heart: No-FE, OpenFE, AutoFeat, CAAFE, FeatLLM, KnowFeat
- [ ] **T1.11** bank-marketing: No-FE, OpenFE, AutoFeat, CAAFE, FeatLLM, KnowFeat
- [ ] **T1.12** breast-w: No-FE, OpenFE, AutoFeat, CAAFE, FeatLLM, KnowFeat

### 1C: 结果汇总

- [ ] **T1.13** 从 `outputs/eval_results/*.json` 汇总生成 Table 1（数字人民币）和 Table 2（公开数据集）

---

## Phase 2: Exp 2 — 知识消融 (Knowledge Ablation)

> 仅在数字人民币上运行，量化每种知识资产的边际贡献
> 结果存放: `outputs/eval_results/simecny_KnowFeat-*.json` + `outputs/ablation/`

### T2.0 实现知识过滤机制

- [ ] **T2.0** 在 `knowfeat/prompt/domain_knowledge.py` 的 `format_domain_knowledge()` 中添加 `knowledge_filter` 参数
  - 函数签名: `format_domain_knowledge(dk, stage="stage2", knowledge_filter=None)`
  - `knowledge_filter` 取值: `"rules"`, `"indicators"`, `"expert"`, `"court"`, `"schema_only"`, `None`(=全部)
  - 实现方式: 根据 filter 值只格式化对应知识类型的内容
    - `"rules"`: 只输出 `dk.detection_rules`（RULE_001~015）
    - `"indicators"`: 只输出 `dk.risk_indicators`（IND_01~38）
    - `"expert"`: 只输出 `dk.expert_opinion`（EXP_01~26）
    - `"court"`: 只输出 `dk.court_doc_risk_factors`（RF_S2_*）
    - `"schema_only"`: 返回空字符串（不输出任何知识，仅保留 schema）
    - `None`: 保持原有行为，输出全部知识
  - 在 `build_system_prompt()` 中透传 `knowledge_filter` 参数
  - 在 `knowfeat/main.py` 的 CLI 添加 `--knowledge-filter` 参数
  - 运行测试: `.venv/bin/python -m pytest tests/ -x`

### T2.1~T2.6 运行消融实验

> 每个配置运行 KnowFeat Stage2，生成特征后用 `evaluate.py` 评估
> 输出目录: `outputs/ablation/<config_name>/`（含 stage2_features.json, stage2_provenance.json, stage2_summary.json）

- [ ] **T2.1** KnowFeat-Full（全部知识）
  - 运行: `.venv/bin/python -m knowfeat.main --stage stage2 --max-features 30`
  - 复制输出到: `outputs/ablation/full/`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method KnowFeat-Full --features-csv outputs/ablation/full/stage2_features.csv`
  - 结果: `outputs/eval_results/simecny_KnowFeat-Full.json`

- [ ] **T2.2** KnowFeat-Rules（仅检测规则）
  - 运行: `.venv/bin/python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter rules`
  - 复制输出到: `outputs/ablation/rules/`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method KnowFeat-Rules --features-csv outputs/ablation/rules/stage2_features.csv`
  - 结果: `outputs/eval_results/simecny_KnowFeat-Rules.json`

- [ ] **T2.3** KnowFeat-Indicators（仅风险指标）
  - 运行: `.venv/bin/python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter indicators`
  - 复制输出到: `outputs/ablation/indicators/`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method KnowFeat-Indicators --features-csv outputs/ablation/indicators/stage2_features.csv`
  - 结果: `outputs/eval_results/simecny_KnowFeat-Indicators.json`

- [ ] **T2.4** KnowFeat-Expert（仅专家意见）
  - 运行: `.venv/bin/python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter expert`
  - 复制输出到: `outputs/ablation/expert/`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method KnowFeat-Expert --features-csv outputs/ablation/expert/stage2_features.csv`
  - 结果: `outputs/eval_results/simecny_KnowFeat-Expert.json`

- [ ] **T2.5** KnowFeat-Court（仅裁判文书风险因子）
  - 运行: `.venv/bin/python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter court`
  - 复制输出到: `outputs/ablation/court/`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method KnowFeat-Court --features-csv outputs/ablation/court/stage2_features.csv`
  - 结果: `outputs/eval_results/simecny_KnowFeat-Court.json`

- [ ] **T2.6** KnowFeat-Schema（仅数据 Schema，无领域知识）
  - 运行: `.venv/bin/python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter schema_only`
  - 复制输出到: `outputs/ablation/schema_only/`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method KnowFeat-Schema --features-csv outputs/ablation/schema_only/stage2_features.csv`
  - 结果: `outputs/eval_results/simecny_KnowFeat-Schema.json`

### T2.7 汇总

- [ ] **T2.7** 从 `outputs/eval_results/simecny_KnowFeat-*.json` 汇总生成结果
  - 写脚本 `scripts/gen_table3.py`，读取 6 个 JSON，输出:
    - `outputs/tables/table3_ablation.csv`（Table 3: 各知识配置的 Recall/F1/AUC + 通过率）
    - `outputs/figures/figure1_ablation.png`（Figure 1: 条形图，各知识类型的边际贡献 = Full - 去掉该类型后的性能差）
  - 通过率计算: 从 `outputs/ablation/<config>/stage2_summary.json` 读取 `total_accepted / (total_accepted + total_rejected)`

---

## Phase 3: Exp 3 — 溯源完整性评估 (Traceability Metrics)

> 仅在数字人民币上，对比 CAAFE / FeatLLM / KnowFeat 的可解释性
> 依赖: Phase 1 中 CAAFE、FeatLLM、KnowFeat 的运行产物
> 结果存放: `outputs/eval_results/traceability_metrics.json` + `outputs/tables/table4_traceability.csv`

### T3.1 收集溯源数据

- [ ] **T3.1** 收集三个方法各 20 个特征的解释/溯源信息
  - KnowFeat: 从 `outputs/stage2_provenance.json` 提取（已有字段: explanation, reasoning, knowledge_source, data_fields_used, knowledge_ids, computation_steps）
  - CAAFE: 从 `baselines/CAAFE/` 运行输出中提取代码注释和特征描述，保存到 `outputs/traceability/caafe_explanations.json`
  - FeatLLM: 从 `baselines/FeatLLM/` 运行输出中提取因果描述，保存到 `outputs/traceability/featllm_explanations.json`
  - 统一格式: `[{"name": "feat_name", "explanation": "...", "code": "...", "knowledge_refs": [...], "reasoning_steps": [...], "data_fields": [...]}]`

### T3.2 实现溯源指标计算脚本

- [ ] **T3.2** 编写 `scripts/compute_traceability.py`
  - 输入: `--knowfeat-provenance outputs/stage2_provenance.json --caafe-explanations outputs/traceability/caafe_explanations.json --featllm-explanations outputs/traceability/featllm_explanations.json`
  - 计算 4 个指标:
    - **知识引用率** (Knowledge Citation Rate): 特征解释中包含具体知识 ID（如 RULE_*, IND_*, EXP_*）的特征占比
    - **知识可验证率** (Knowledge Verifiability Rate): 引用的知识 ID 能在 `knowledge_registry` 中找到原文的比例
    - **推理步骤覆盖率** (Reasoning Coverage): `computation_steps` 或 `reasoning` 中有 "因为/because/由于" 等因果词的步骤占比
    - **字段溯源率** (Field Traceability Rate): 代码中引用的 DataFrame 列名在 `data_fields_used` 中出现的比例
  - 输出: `outputs/eval_results/traceability_metrics.json`
    ```json
    {
      "KnowFeat": {"citation_rate": 0.95, "verifiability_rate": 0.90, "reasoning_coverage": 0.85, "field_traceability": 0.92},
      "CAAFE": {"citation_rate": 0.10, ...},
      "FeatLLM": {"citation_rate": 0.15, ...}
    }
    ```

### T3.3 生成表格

- [ ] **T3.3** 从 `outputs/eval_results/traceability_metrics.json` 生成 Table 4
  - 在 `scripts/compute_traceability.py` 中加入 `--output-table` 参数
  - 输出: `outputs/tables/table4_traceability.csv`（3行×4列: 方法 × 4个溯源指标）

---

## Phase 4: Exp 4 — 专家打分 (Expert Evaluation)

> 人工评估，需要找 3-5 名反洗钱专家
> 结果存放: `outputs/expert_eval/`

### T4.1 准备评估材料

- [ ] **T4.1** 编写 `scripts/prepare_expert_eval.py`，自动生成评估材料
  - 输入: 三个方法的特征及解释
    - KnowFeat: `outputs/stage2_provenance.json`（取前 15 个特征）
    - CAAFE: `outputs/traceability/caafe_explanations.json`（取前 15 个特征）
    - FeatLLM: `outputs/traceability/featllm_explanations.json`（取前 15 个特征）
  - 处理:
    - 匿名化: 去除方法名，替换为 "方法A/B/C"（随机分配）
    - 随机打乱所有特征顺序（固定 seed=42 保证可复现）
    - 生成评分表模板（Excel/CSV），每行一个特征，列: 特征ID, 特征名, 解释内容, 领域合理性(1-5), 解释充分性(1-5), 可审计性(1-5)
  - 输出:
    - `outputs/expert_eval/eval_form.xlsx`（评分表，发给专家填写）
    - `outputs/expert_eval/method_mapping.json`（方法A/B/C → 真实方法名映射，评分完成后用于汇总）
    - `outputs/expert_eval/eval_materials.md`（Markdown 格式评估材料，备用）

### T4.2 专家评分（人工任务）

- [ ] **T4.2** 联系 3-5 名反洗钱专家，发送 `outputs/expert_eval/eval_form.xlsx`
  - ⚠️ 这是人工任务，loop 模式跳过
  - 收回后将评分结果放到 `outputs/expert_eval/scores/expert_1.csv`, `expert_2.csv`, ...

### T4.3 汇总评分

- [ ] **T4.3** 编写 `scripts/summarize_expert_eval.py`
  - 输入: `outputs/expert_eval/scores/expert_*.csv` + `outputs/expert_eval/method_mapping.json`
  - 计算: 按方法分组，计算三个维度的均值和标准差
  - 输出:
    - `outputs/tables/table5_expert_eval.csv`（Table 5: 3方法 × 3维度的 mean±std）
    - `outputs/figures/figure2_radar.png`（Figure 2: 雷达图，matplotlib 绘制）

---

## Phase 5: Exp 5 — Case Study

> 定性展示溯源卡片 vs 其他方法解释的差异
> 依赖: Phase 1 中三个方法的运行产物
> 结果存放: `outputs/case_study/`

- [ ] **T5.1** 编写 `scripts/gen_case_study.py`，自动提取对比材料
  - 从 `outputs/stage2_provenance.json` 中挑选 2-3 个代表性特征:
    - 标准: knowledge_ids 数量多（≥3）、computation_steps 步骤完整、explanation 长度适中
    - 候选: "休眠账户激活风险"、"分散转入集中转出比率" 等（按 knowledge_ids 数量排序取 top 3）
  - 从 CAAFE/FeatLLM 输出中找到功能最接近的特征（按特征名或 explanation 语义匹配）
  - 输出:
    - `outputs/case_study/case_study.json`（结构化数据: 每个 case 包含三个方法的特征名、代码、解释、溯源信息）
    - `outputs/case_study/case_study.md`（Markdown 格式并排对比，可直接用于论文）

- [ ] **T5.2** 制作 Figure 3（并排对比图）
  - 在 `scripts/gen_case_study.py` 中加入 `--plot` 参数
  - 用 matplotlib 或手动排版生成对比图
  - 输出: `outputs/figures/figure3_case_study.png`
  - 格式参考 `docs/experiment_design.md` 中的对比框图

---

## Phase 6: Exp 6 — 效率对比 (Efficiency)

> 在数字人民币上对比各方法的计算成本
> 结果存放: `outputs/tables/table6_efficiency.csv`

- [ ] **T6.1** 编写 `scripts/gen_efficiency_table.py`，汇总各方法的运行指标
  - 数据来源:
    - KnowFeat: 从 `outputs/stage2_summary.json` 读取 `elapsed_sec`, `total_accepted`, `total_rejected`, `total_candidates`
    - CAAFE: 从 `baselines/CAAFE/` 运行日志中提取（运行时用 `time` 命令计时）
    - FeatLLM: 从 `baselines/FeatLLM/` 运行日志中提取
    - OpenFE/AutoFeat: 从运行日志中提取（无 API 调用）
  - 收集指标:
    - 总运行时间（秒）
    - LLM API 调用次数（仅 CAAFE, FeatLLM, KnowFeat）
    - API 费用估算（调用次数 × 单价，DeepSeek: ~¥0.001/1K tokens）
    - 生成特征数（total_candidates）
    - 通过特征数（total_accepted）
    - 通过率（total_accepted / total_candidates）
  - 输出: `outputs/tables/table6_efficiency.csv`
  - 格式: 6行（方法）× 6列（上述指标）

- [ ] **T6.2** 如果运行时间数据不完整，补充运行缺失方法并计时
  - 对每个 LLM-based 方法，运行时在命令前加 `time`
  - 将计时结果写入 `outputs/efficiency/<method>_timing.txt`

---

## 执行顺序与依赖

> **Loop 模式执行规则**：从上到下找到第一个未完成的 `- [ ]` 任务，执行它。
> 完成后勾选 `- [x]`，commit + push，然后继续下一个任务。

```
Phase 0 (基础设施) — 剩余: P0.2b 下载数据, P0.4 Baseline 安装
  ├── P0.2b 执行下载 ─┐
  └── P0.4 Baseline ──┼── Phase 1 (主性能对比)
                       │
                       ├── Phase 2 (知识消融, 可并行)
                       ├── Phase 3 (溯源指标, 可并行)
                       ├── Phase 5 (Case Study, 可并行)
                       └── Phase 6 (效率, 可并行)
                             │
                       Phase 4 (专家打分, 人工, 最后)
```

**关键路径**: P0.2b → P0.4 → Phase 1 → Phase 2~6

**环境信息**:
- Python: `.venv/bin/python`
- DeepSeek API Key: 环境变量 `KNOWFEAT_API_KEY`
- DeepSeek Base URL: `https://api.deepseek.com/v1`
- DeepSeek Model: `deepseek-chat`
