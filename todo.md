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

- [ ] **P0.1** 公开数据集通用适配器
  - 当前 KnowFeat 硬编码为数字人民币三表结构(open_df, tx_df, close_df)
  - 需要开发通用单表适配器：输入一个 DataFrame + target列名 → 适配到 KnowFeat pipeline
  - 涉及文件：`knowfeat/config.py`, `knowfeat/prompt/builder.py`, `knowfeat/agent/tools.py`
  - 公开数据集的 prompt 模板需要通用化（不能写死"数字人民币反洗钱"）
  - 每个数据集从 `domain_knowledge.json` 加载知识，格式化注入 prompt

- [ ] **P0.2** 公开数据集下载脚本
  - 从 OpenML 下载 5 个数据集（credit-g #31, diabetes #37, blood-transfusion #1464, bank-marketing #1461, breast-w #15）
  - heart 从 Kaggle 下载
  - 统一存放到 `data/public/` 目录，CSV 格式

- [ ] **P0.3** 统一评估框架
  - 实现统一的评估脚本：80/20 split × 5 seeds × XGBoost
  - 数字人民币指标：Recall, Precision, F1, AUC
  - 公开数据集指标：AUC, Accuracy
  - 输出标准化 JSON 结果文件
  - 显著性检验：Welch's t-test (p=0.05)

- [ ] **P0.4** Baseline 方法安装与适配
  - `pip install openfe autofeat`
  - CAAFE: clone github.com/noahho/CAAFE，确认能否用 DeepSeek API 替代 OpenAI
  - FeatLLM: clone github.com/Sungwon-Han/FeatLLM，确认 API 兼容性
  - 如果 LLM-based baseline 必须用 OpenAI API，记录成本估算

---

## Phase 1: Exp 1 — 主性能对比 (Performance Comparison)

> 目标：7个数据集 × 6种方法 → 两张性能对比表

### 1A: 数字人民币上的 Baseline

- [ ] **T1.1** No-FE baseline（原始95特征 + XGBoost）— ✅ 已有数据: recall=0.9538
- [ ] **T1.2** OpenFE baseline（95特征上自动搜索，top 30 特征）
- [ ] **T1.3** AutoFeat baseline（95特征上多项式+L1选择）
- [ ] **T1.4** CAAFE baseline（LLM单轮生成，DeepSeek API）
- [ ] **T1.5** FeatLLM baseline（LLM模板生成）
- [ ] **T1.6** KnowFeat — ✅ 已有数据: recall=0.9737, 18/30特征被接受

### 1B: 6个公开数据集上的全部方法

> 依赖 P0.1（适配器）和 P0.2（数据）

- [ ] **T1.7** credit-g: No-FE, OpenFE, AutoFeat, CAAFE, FeatLLM, KnowFeat
- [ ] **T1.8** diabetes: No-FE, OpenFE, AutoFeat, CAAFE, FeatLLM, KnowFeat
- [ ] **T1.9** blood-transfusion: No-FE, OpenFE, AutoFeat, CAAFE, FeatLLM, KnowFeat
- [ ] **T1.10** heart: No-FE, OpenFE, AutoFeat, CAAFE, FeatLLM, KnowFeat
- [ ] **T1.11** bank-marketing: No-FE, OpenFE, AutoFeat, CAAFE, FeatLLM, KnowFeat
- [ ] **T1.12** breast-w: No-FE, OpenFE, AutoFeat, CAAFE, FeatLLM, KnowFeat

### 1C: 结果汇总

- [ ] **T1.13** 生成 Table 1（数字人民币性能对比）和 Table 2（公开数据集性能对比）

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

```
Phase 0 (基础设施)
  ├── P0.1 适配器 ──┐
  ├── P0.2 数据下载 ─┤
  ├── P0.3 评估框架 ─┼── Phase 1 (主性能对比)
  └── P0.4 Baseline ─┘       │
                              ├── Phase 2 (知识消融, 可并行)
                              ├── Phase 3 (溯源指标, 可并行)
                              ├── Phase 5 (Case Study, 可并行)
                              └── Phase 6 (效率, 可并行)
                                    │
                              Phase 4 (专家打分, 人工, 最后)
```

**预计工作量**：
- Phase 0: 2-3 天（主要是适配器开发）
- Phase 1: 3-4 天（7个数据集 × 6种方法 = 42 次运行）
- Phase 2: 2 天（5 次消融运行，数字人民币）
- Phase 3: 1 天（指标计算脚本 + 收集解释）
- Phase 4: 1 周+（依赖专家排期）
- Phase 5: 0.5 天
- Phase 6: 0.5 天

**总计约 2 周（不含专家评审等待时间）**
