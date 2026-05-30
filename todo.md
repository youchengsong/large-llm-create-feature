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

- [ ] **T2.1** KnowFeat-Full（全部知识）— ✅ 已有数据
- [ ] **T2.2** KnowFeat-Rules（仅 RULE_001~015）
- [ ] **T2.3** KnowFeat-Indicators（仅 IND_01~38）
- [ ] **T2.4** KnowFeat-Expert（仅 EXP_01~26）
- [ ] **T2.5** KnowFeat-Court（仅裁判文书风险因子）
- [ ] **T2.6** KnowFeat-Schema（仅数据 Schema，无领域知识）
- [ ] **T2.7** 生成 Table 3 + Figure 1（各知识类型边际贡献条形图）

实现方式：在 `prompt/builder.py` 中控制 `format_domain_knowledge()` 只加载指定知识类型。

---

## Phase 3: Exp 3 — 溯源完整性评估 (Traceability Metrics)

> 仅在数字人民币上，对比 CAAFE / FeatLLM / KnowFeat 的可解释性

- [ ] **T3.1** 每个方法生成 20 个特征，收集其解释/溯源信息
  - KnowFeat: 从 provenance card 提取
  - CAAFE: 从代码注释提取
  - FeatLLM: 从因果描述提取

- [ ] **T3.2** 实现 4 个溯源指标的自动计算脚本
  - 知识引用率 (Knowledge Citation Rate)
  - 知识可验证率 (Knowledge Verifiability Rate)
  - 推理步骤覆盖率 (Reasoning Coverage)
  - 字段溯源率 (Field Traceability Rate)

- [ ] **T3.3** 生成 Table 4（溯源完整性指标对比）

---

## Phase 4: Exp 4 — 专家打分 (Expert Evaluation)

> 人工评估，需要找 3-5 名反洗钱专家

- [ ] **T4.1** 准备评估材料
  - 从 CAAFE, FeatLLM, KnowFeat 各抽取 10-15 个特征
  - 匿名化（去除方法名，随机打乱）
  - 制作评分表（领域合理性 / 解释充分性 / 可审计性，1-5分）

- [ ] **T4.2** 联系专家进行评分（人工任务）

- [ ] **T4.3** 汇总评分，生成 Table 5 + Figure 2（雷达图）

---

## Phase 5: Exp 5 — Case Study

> 定性展示溯源卡片 vs 其他方法解释的差异

- [ ] **T5.1** 挑选 2-3 个代表性特征（如"休眠账户激活风险"、"分散转入集中转出比率"）
- [ ] **T5.2** 收集 CAAFE 和 FeatLLM 对同类特征的解释
- [ ] **T5.3** 制作并排对比图（Figure 3），展示溯源卡片的完整性优势

---

## Phase 6: Exp 6 — 效率对比 (Efficiency)

> 在数字人民币上对比各方法的计算成本

- [ ] **T6.1** 记录各方法的运行指标
  - 总运行时间
  - LLM API 调用次数（仅 LLM-based 方法）
  - API 费用估算
  - 生成特征数 / 通过特征数

- [ ] **T6.2** 生成 Table 6（效率对比表）

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
