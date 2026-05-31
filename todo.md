# ICDM 2026 论文实验 TODO

> KnowFeat: Knowledge-Traceable Feature Engineering
>
> 实验设计参考：`docs/experiment_design.md`
>
> 最后更新：2026-05-31

---

## 数据集总览

| # | 数据集 | 规模 | 特征数 | 类型 | 状态 |
|---|--------|------|--------|------|------|
| 1 | **数字人民币** | 20万交易(子集) | 95 | AML核心 | ✅ 已采样 |
| 2 | credit-g | 1,000 | 20 | 公开 | ✅ 已下载 |
| 3 | diabetes | 768 | 8 | 公开 | ✅ 已下载 |
| 4 | blood-transfusion | 748 | 4 | 公开 | ✅ 已下载 |
| 5 | heart | 918 | 11 | 公开 | ✅ 已下载 |
| 6 | bank-marketing | 45,211 | 16 | 公开 | ✅ 已下载 |
| 7 | breast-w | 683 | 9 | 公开 | ✅ 已下载 |

领域知识文件：`inputs/public_datasets/*/domain_knowledge.json`（✅ 6个已创建）

---

## Phase 0: 基础设施准备

- [x] **P0.1** 公开数据集通用适配器 ✅ commit 8a1e8eb
- [x] **P0.2** 公开数据集下载脚本 ✅ commit 8a1e8eb
- [x] **P0.2b** 执行数据集下载 ✅ 6个CSV已存放于 `data/public/`
- [x] **P0.3** 统一评估框架 ✅ commit 8a1e8eb
- [x] **P0.4** Baseline 方法安装与适配 ✅
  - OpenFE/AutoFeat: `pip install openfe autofeat` ✅
  - CAAFE: `baselines/CAAFE/` 已 clone，API 已改为 DeepSeek，wrapper `scripts/run_caafe.py` ✅
  - FeatLLM: `baselines/FeatLLM/` 已 clone，API 已升级为新版 SDK，wrapper `scripts/run_featllm.py` ✅
  - OpenFE wrapper: `scripts/run_openfe.py` ✅
  - AutoFeat wrapper: `scripts/run_autofeat.py` ✅
- [x] **P0.5** L3 验证指标适配 ✅
  - AML 数据集: delta-recall + precision≥0.90
  - 公开数据集: delta-AUC（自动按 `accept_metric` 切换）
- [x] **P0.6** 知识过滤机制 (原 T2.0) ✅
  - `format_domain_knowledge(dk, stage, knowledge_filter=None)` 支持 rules/indicators/expert/court/schema_only
  - CLI: `python -m knowfeat.main --knowledge-filter <type>`
  - 全链路透传: main.py → run_stage → ReactAgent → build_full_prompt → build_system_prompt → format_domain_knowledge
- [x] **P0.7** 公开数据集 Agent 入口 ✅
  - `scripts/run_knowfeat_public.py` 已接通 `run_stage` + `save_results`
- [x] **P0.8** 数字人民币数据采样 ✅
  - 200K 交易子集（全部 4,484 异常 + 195,516 正常随机采样）
  - 对应 accounts/close 子表 + baseline_cache.parquet 已同步
  - 原始数据备份于 `data/full_backup/`

---

## Phase 0.9: 服务器部署准备

> 3 台 AutoDL 服务器并行跑实验，共享 1 个 DeepSeek API 账号
> **由本机 Claude Code 通过 SSH 统一控制 3 台服务器**

### 服务器配置建议

- **GPU**: 最便宜的即可（如 RTX 2080Ti），实验不需要 GPU
- **CPU**: ≥8 核（XGBoost/OpenFE 吃多核）
- **内存**: ≥16GB
- **选区**: 网络好的区（需调 DeepSeek API）

### SSH 连接信息（用户提供后填入）

```
机器A (simecny): ssh <user>@<host_a> -p <port_a>
机器B (公开数据集前3): ssh <user>@<host_b> -p <port_b>
机器C (公开数据集后3): ssh <user>@<host_c> -p <port_c>
```

### 部署流程（loop 模式通过 SSH 在每台机器上执行）

对每台机器依次执行：

```bash
# 通过 SSH 执行以下命令
# 1. clone 代码
git clone git@github.com:youchengsong/large-llm-create-feature.git
cd large-llm-create-feature

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
pip install openfe autofeat openml
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. 设环境变量（写入 ~/.bashrc）
echo 'export KNOWFEAT_API_KEY="<key>"' >> ~/.bashrc
echo 'export KNOWFEAT_BASE_URL="https://api.deepseek.com/v1"' >> ~/.bashrc
source ~/.bashrc

# 5. 创建日志目录
mkdir -p logs
```

### 任务分配

> 3 台机器，LLM 方法串行（避免 API 限速），CPU 方法可并行
> 所有实验命令通过 `ssh <机器> "cd large-llm-create-feature && nohup ... &"` 远程启动
> 用 `ssh <机器> "tail -f large-llm-create-feature/logs/xxx.log"` 检查进度

- [x] **P0.9a** 生成 `requirements.txt` ✅
  - 已生成精简版 `requirements.txt`（直接依赖）+ `requirements-lock.txt`（完整 freeze）

- [ ] **P0.9b** 部署 3 台服务器环境（通过 SSH）
  - 用户提供 SSH 信息后，逐台 SSH 执行部署脚本
  - 每台验证: `ssh <机器> "cd large-llm-create-feature && .venv/bin/python -c 'import openfe; import autofeat; import xgboost; import openai; print(\"OK\")'"`
  - 每台验证: `ssh <机器> "cd large-llm-create-feature && .venv/bin/python scripts/evaluate.py --dataset breast_w --method No-FE --baseline"`（应输出 AUC≈0.99）

---

## Phase 1: Exp 1 — 主性能对比 (Performance Comparison)

> 目标：7个数据集 × 6种方法 → 两张性能对比表
> **执行方式**: 所有命令通过 SSH 远程执行，格式: `ssh <机器X> "cd large-llm-create-feature && <命令>"`
> 评估命令: `.venv/bin/python scripts/evaluate.py --dataset <name> --method <method> [--features-csv <path>]`
> 结果存放: `outputs/eval_results/<dataset>_<method>.json`
> 结果回收: 实验完成后用 `scp` 将 3 台机器的 `outputs/eval_results/*.json` 拉回本机汇总

### 机器 A: 数字人民币 (simecny) — 全部 6 方法

> 预计耗时: ~6-8h（LLM 方法串行）

- [ ] **T1.1** No-FE baseline
  - 运行: `nohup .venv/bin/python scripts/evaluate.py --dataset simecny --method No-FE --baseline > logs/t1.1.log 2>&1 &`
- [ ] **T1.2** OpenFE baseline
  - 生成特征: `nohup .venv/bin/python scripts/run_openfe.py --dataset simecny --n-features 30 > logs/t1.2_gen.log 2>&1 &`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method OpenFE --features-csv outputs/openfe/simecny_features.csv`
- [ ] **T1.3** AutoFeat baseline
  - 生成特征: `nohup .venv/bin/python scripts/run_autofeat.py --dataset simecny > logs/t1.3_gen.log 2>&1 &`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method AutoFeat --features-csv outputs/autofeat/simecny_features.csv`
- [ ] **T1.4** CAAFE baseline（需 API）
  - 生成特征: `nohup .venv/bin/python scripts/run_caafe.py --dataset simecny --iterations 10 > logs/t1.4_gen.log 2>&1 &`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method CAAFE --features-csv outputs/caafe/simecny_features.csv`
- [ ] **T1.5** FeatLLM baseline（需 API）
  - 生成特征: `nohup .venv/bin/python scripts/run_featllm.py --dataset simecny > logs/t1.5_gen.log 2>&1 &`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method FeatLLM --features-csv outputs/featllm/simecny_features.csv`
- [ ] **T1.6** KnowFeat（需 API）
  - 运行: `nohup .venv/bin/python -m knowfeat.main --stage stage2 --max-features 30 > logs/t1.6.log 2>&1 &`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method KnowFeat --features-csv outputs/stage2_features.csv`

### 机器 B: 公开数据集前 3 个

> 预计耗时: ~4-5h
> 每个数据集跑 6 种方法，LLM 方法串行，CPU 方法可后台并行

- [ ] **T1.7** credit-g 全部 6 方法
  - No-FE: `.venv/bin/python scripts/evaluate.py --dataset credit_g --method No-FE --baseline`
  - OpenFE: `.venv/bin/python scripts/run_openfe.py --dataset credit_g && .venv/bin/python scripts/evaluate.py --dataset credit_g --method OpenFE --features-csv outputs/openfe/credit_g_features.csv`
  - AutoFeat: `.venv/bin/python scripts/run_autofeat.py --dataset credit_g && .venv/bin/python scripts/evaluate.py --dataset credit_g --method AutoFeat --features-csv outputs/autofeat/credit_g_features.csv`
  - CAAFE: `.venv/bin/python scripts/run_caafe.py --dataset credit_g && .venv/bin/python scripts/evaluate.py --dataset credit_g --method CAAFE --features-csv outputs/caafe/credit_g_features.csv`
  - FeatLLM: `.venv/bin/python scripts/run_featllm.py --dataset credit_g && .venv/bin/python scripts/evaluate.py --dataset credit_g --method FeatLLM --features-csv outputs/featllm/credit_g_features.csv`
  - KnowFeat: `.venv/bin/python scripts/run_knowfeat_public.py --dataset credit_g && .venv/bin/python scripts/evaluate.py --dataset credit_g --method KnowFeat --features-csv outputs/credit_g/stage2_features.csv`
- [ ] **T1.8** diabetes 全部 6 方法（同上格式，替换 dataset 名）
- [ ] **T1.9** blood-transfusion 全部 6 方法

### 机器 C: 公开数据集后 3 个

> 预计耗时: ~4-5h

- [ ] **T1.10** heart 全部 6 方法
- [ ] **T1.11** bank-marketing 全部 6 方法
- [ ] **T1.12** breast-w 全部 6 方法

### 结果汇总（任意机器，所有实验完成后）

- [ ] **T1.13** 汇总生成 Table 1 和 Table 2
  - 将 3 台机器的 `outputs/eval_results/*.json` 合并到一台机器
  - 编写 `scripts/gen_table1_table2.py` 读取所有 JSON 生成:
    - `outputs/tables/table1_simecny.csv`（数字人民币: 6方法 × Recall/Precision/F1/AUC）
    - `outputs/tables/table2_public.csv`（公开数据集: 6方法 × 6数据集 × AUC/Accuracy）

---

## Phase 2: Exp 2 — 知识消融 (Knowledge Ablation)

> 仅在数字人民币上运行，量化每种知识资产的边际贡献
> **在机器 A 上串行执行（Phase 1 simecny 完成后）**
> 结果存放: `outputs/eval_results/simecny_KnowFeat-*.json` + `outputs/ablation/`

- [ ] **T2.1** KnowFeat-Full（全部知识）
  - 运行: `nohup .venv/bin/python -m knowfeat.main --stage stage2 --max-features 30 > logs/t2.1.log 2>&1 &`
  - 复制输出到: `outputs/ablation/full/`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method KnowFeat-Full --features-csv outputs/ablation/full/stage2_features.csv`

- [ ] **T2.2** KnowFeat-Rules（仅检测规则）
  - 运行: `nohup .venv/bin/python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter rules > logs/t2.2.log 2>&1 &`
  - 复制输出到: `outputs/ablation/rules/`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method KnowFeat-Rules --features-csv outputs/ablation/rules/stage2_features.csv`

- [ ] **T2.3** KnowFeat-Indicators（仅风险指标）
  - 运行: `nohup .venv/bin/python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter indicators > logs/t2.3.log 2>&1 &`
  - 复制输出到: `outputs/ablation/indicators/`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method KnowFeat-Indicators --features-csv outputs/ablation/indicators/stage2_features.csv`

- [ ] **T2.4** KnowFeat-Expert（仅专家意见）
  - 运行: `nohup .venv/bin/python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter expert > logs/t2.4.log 2>&1 &`
  - 复制输出到: `outputs/ablation/expert/`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method KnowFeat-Expert --features-csv outputs/ablation/expert/stage2_features.csv`

- [ ] **T2.5** KnowFeat-Court（仅裁判文书风险因子）
  - 运行: `nohup .venv/bin/python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter court > logs/t2.5.log 2>&1 &`
  - 复制输出到: `outputs/ablation/court/`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method KnowFeat-Court --features-csv outputs/ablation/court/stage2_features.csv`

- [ ] **T2.6** KnowFeat-Schema（仅数据 Schema，无领域知识）
  - 运行: `nohup .venv/bin/python -m knowfeat.main --stage stage2 --max-features 30 --knowledge-filter schema_only > logs/t2.6.log 2>&1 &`
  - 复制输出到: `outputs/ablation/schema_only/`
  - 评估: `.venv/bin/python scripts/evaluate.py --dataset simecny --method KnowFeat-Schema --features-csv outputs/ablation/schema_only/stage2_features.csv`

- [ ] **T2.7** 汇总消融结果
  - 编写 `scripts/gen_table3.py` 生成:
    - `outputs/tables/table3_ablation.csv`（Table 3: 6配置 × Recall/F1/AUC/通过率）
    - `outputs/figures/figure1_ablation.png`（条形图: 各知识类型的边际贡献）

---

## Phase 3: Exp 3 — 溯源完整性评估 (Traceability Metrics)

> 仅在数字人民币上，对比 CAAFE / FeatLLM / KnowFeat 的可解释性
> 依赖: Phase 1 中 CAAFE、FeatLLM、KnowFeat 在 simecny 上的运行产物
> **在机器 A 上执行（Phase 2 完成后或并行）**

- [ ] **T3.1** 收集三个方法各 20 个特征的解释/溯源信息
  - KnowFeat: 从 `outputs/stage2_provenance.json` 提取
  - CAAFE: 从 `outputs/caafe/` 运行输出中提取
  - FeatLLM: 从 `outputs/featllm/` 运行输出中提取

- [ ] **T3.2** 编写 `scripts/compute_traceability.py` 计算 4 个溯源指标
  - 知识引用率、知识可验证率、推理步骤覆盖率、字段溯源率
  - 输出: `outputs/eval_results/traceability_metrics.json`

- [ ] **T3.3** 生成 Table 4: `outputs/tables/table4_traceability.csv`

---

## Phase 4: Exp 4 — 专家打分 (Expert Evaluation)

> ⚠️ 人工任务，loop 模式跳过

- [ ] **T4.1** 编写 `scripts/prepare_expert_eval.py` 生成匿名化评估材料
- [ ] **T4.2** 联系 3-5 名专家评分（人工）
- [ ] **T4.3** 编写 `scripts/summarize_expert_eval.py` 汇总评分 → Table 5 + Figure 2 雷达图

---

## Phase 5: Exp 5 — Case Study

> 依赖: Phase 1 中三个方法的运行产物

- [ ] **T5.1** 编写 `scripts/gen_case_study.py` 提取 2-3 个代表性特征并排对比
- [ ] **T5.2** 生成 Figure 3

---

## Phase 6: Exp 6 — 效率对比 (Efficiency)

> 在数字人民币上对比各方法的计算成本

- [ ] **T6.1** 编写 `scripts/gen_efficiency_table.py` 汇总运行指标
  - 总运行时间、API 调用次数、API 费用、生成/通过特征数、通过率
  - 输出: `outputs/tables/table6_efficiency.csv`

---

## 执行顺序与依赖

> **Loop 模式执行规则**：从上到下找到第一个未完成的 `- [ ]` 任务，执行它。
> 完成后勾选 `- [x]`，commit + push，然后继续下一个任务。

```
Phase 0 (基础设施) ✅ 全部完成
  │
  ├── Phase 0.9 (服务器部署) ← 当前步骤
  │
  ├── Phase 1 (主性能对比) ← 3台机器并行
  │     机器A: simecny × 6方法 (~6-8h)
  │     机器B: credit_g + diabetes + blood_transfusion × 6方法 (~4-5h)
  │     机器C: heart + bank_marketing + breast_w × 6方法 (~4-5h)
  │
  ├── Phase 2 (知识消融) ← 机器A，Phase 1 之后串行 (~6h)
  │
  ├── Phase 3 (溯源指标) ← 机器A，可与 Phase 2 并行
  ├── Phase 5 (Case Study) ← 任意机器
  └── Phase 6 (效率) ← 任意机器
        │
  Phase 4 (专家打分) ← 人工，最后
```

**环境信息**:
- Python: `.venv/bin/python`
- DeepSeek API Key: 环境变量 `KNOWFEAT_API_KEY`
- DeepSeek Base URL: `https://api.deepseek.com/v1`
- DeepSeek Model: `deepseek-chat`
- 1 个 DeepSeek 账号，3 台机器共用（实际并发低，不会限速）
