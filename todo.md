# ICDM 2026 论文实验 TODO

> KnowFeat: Distilling Domain Expert Knowledge into Verified Features via LLM Agents for Digital Yuan Anti-Money Laundering
>
> 设计参考：`docs/KnowFeat_design.md`
>
> 最后更新：2026-05-29

---

## 一、实验总览

| 类别 | 实验数 | 优先级 | 状态 |
|------|--------|--------|------|
| 主实验 (Main Results) | 5 | P0 | 部分完成 |
| Baseline 对比 | 8 | P0 | 未开始 |
| 消融实验 (Ablation) | 8 | P0 | 未开始 |
| 知识迁移实验 | 3 | P1 | 未开始 |
| 效率与成本分析 | 2 | P1 | 部分完成 |
| 敏感性分析 | 4 | P2 | 未开始 |
| Case Study | 2 | P2 | 未开始 |

---

## 二、数据集

### Dataset 1: SimECNY (数字人民币仿真)

| 项目 | 值 |
|------|-----|
| 样本数 | 1,384,512 笔交易 |
| 正样本 | 4,484 (0.32%) |
| 基线特征 | 95 个 (精简) / 105 个 (全量) |
| 数据结构 | 三表 (accounts, transactions, close) |
| Stage | Stage 2 |

### Dataset 2: BABD-13 (Bitcoin Address Benchmark)

| 项目 | 值 |
|------|-----|
| 样本数 | 544,462 个地址 |
| 原始标签 | 13 类 |
| 二分类映射 | Illicit (~140K) vs Licit (~404K) |
| 基线特征 | 148 个 (PAI/PDI/PTI/CI/LSI) |
| 数据结构 | 单表 (地址级预计算特征) |
| Stage | Stage 1 |

**BABD-13 二分类映射规则**：
- **Illicit (洗钱/非法)**：Blackmail, Darknet Market, Gambling, Gov Criminal Blacklist, Money Laundering, Ponzi Scheme, Tumbler
- **Licit (合法)**：Centralized Exchange, Cyber-Security Service, P2P Financial, Infrastructure/Service, Mining Pool, Individual Wallet

---

## 三、详细实验计划

---

### 实验 1：主实验 (Main Results)

> 目标：展示 KnowFeat 在两个不同域、不同数据结构的 AML 数据集上的效果

#### 1.1 Stage 2 — SimECNY 完整运行 ✅ 已完成

- **配置**: 95 基线特征, scale_pos_weight=59, max_rounds=10, max_features=30
- **结果**: 18/30 特征被接受, recall 0.9538→0.9737 (+2.09%), precision 0.9402→0.9621
- **数据位置**: `outputs/stage2_summary.json`, `outputs/stage2_features.json`

#### 1.2 Stage 1 — BABD-13 完整运行 ⬜ 未开始

- **配置**: 148 基线特征, max_rounds=5, max_features=30
- **前置条件**: 
  - [ ] 上传 BABD-13 数据到 `/root/autodl-tmp/babd13/`
  - [ ] 实现 13→binary 标签映射
  - [ ] 适配 Stage 1 数据加载器 (单表 → `def feature_xxx(df)`)
  - [ ] 确认 scale_pos_weight (需根据 BABD-13 正负比例计算)
- **需要记录**: 每轮接受特征数、总耗时、最终 recall/precision/f1/auc
- **输出文件**: `outputs/stage1_summary.json`, `outputs/stage1_features.json`

#### 1.3 知识迁移后 Stage 2 再运行 ⬜ 未开始

- **配置**: 在 1.2 完成后，提取 Stage 1 生成的新知识，注入 Stage 2 知识库
- **运行**: Stage 2 以 95+18 基线特征 + 新知识，再跑 3-5 轮
- **对比指标**: 迁移前 vs 迁移后的 recall/precision 增益

#### 1.4 Wallet_type 分场景评估 ⬜ 未开始

- **方法**: 按 `src_wallet_type` 切分测试集，分别计算 Recall/Precision
- **分组**:
  - Corporate wallet (1): 对公账户大额转出、虚假贸易
  - Personal wallet (2): 跑分、小额分散、夜间高频
  - Merchant wallet (3): 收款码跑分、虚假流水、佣金抽成
- **对比**: Baseline vs KnowFeat 在各场景下的差异
- **目的**: 分析 KnowFeat 在不同洗钱场景下的提升是否均匀

#### 1.5 ModelSet 11 模型自动选择 ⬜ 未开始

- **方法**: 使用 `ModelSet/runner.py` 跑全部 11 个候选模型
- **候选模型**: catboost, ga2m, scorecard, acxgb, wav_xgb, ensemble, lgbm, resample_boost, gbdt_fm, dnn_res, mlp
- **评估**: 5-fold StratifiedKFold CV，按 Recall (Precision≥90%) 排序
- **输出**: 模型对比表 + 数据特征→模型适配分析
- **目的**: 验证端到端自动化能力，XGBoost 不一定是最优

#### 1.6 最终对比表 ⬜ 待汇总

在 RESULT.md 中汇总两个数据集的最终结果:

| Dataset | Baseline Features | +KnowFeat | Δrecall | Δprecision | #New Features |
|---------|------------------|-----------|---------|------------|--------------|
| SimECNY | 95 | 95+18 | +2.09% | +2.19% | 18 |
| BABD-13 | 148 | 148+? | ? | ? | ? |

---

### 实验 2：Baseline 对比实验

> 目标：与现有自动特征工程方法和纯 LLM 方法对比，证明 KnowFeat 的优势

所有 baseline 使用相同的 XGBoost 参数和 5-fold CV 评估。

#### 2.1 Manual Features (手工基线) ✅ 已有数据

- SimECNY: 95 特征 → recall=0.9538, precision=0.9402
- SimECNY: 105 特征 → recall=0.9739, precision=0.9781
- BABD-13: 148 特征 → 待跑

#### 2.2 OpenFE ⬜ 未开始

- **工具**: https://github.com/IIIS-Li-Group/OpenFE
- **方法**: 在基线特征上自动搜索交叉/变换特征
- **配置**: 
  - 输入: 基线特征表 + 标签
  - 生成特征数上限: 30 (与 KnowFeat 对齐)
  - 用同样的 XGBoost 评估
- **两个数据集都要跑**
- **安装**: `pip install openfe`
- **实现步骤**:
  ```python
  from openfe import OpenFE, transform
  ofe = OpenFE()
  features = ofe.fit(data=X_train, label=y_train, n_jobs=4)
  X_train_new, X_test_new = transform(X_train, X_test, features, n_jobs=4)
  ```

#### 2.3 Featuretools ⬜ 未开始

- **工具**: https://github.com/alteryx/featuretools
- **方法**: Deep Feature Synthesis (DFS)，自动生成聚合/变换特征
- **配置**:
  - SimECNY: 定义 accounts/transactions/close 三表的 EntitySet
  - BABD-13: 单表，使用变换原语 (transform primitives)
  - 生成后筛选 top 30 特征 (按 feature importance)
- **安装**: `pip install featuretools`

#### 2.4 AutoFeat ⬜ 未开始

- **工具**: https://github.com/cod3licious/autofeat
- **方法**: 多项式特征 + L1 选择
- **配置**:
  - `feateng_steps=3` (三步变换)
  - 在基线特征上生成候选，自动 L1 筛选
- **安装**: `pip install autofeat`

#### 2.5 CAAFE (Context-Aware Automated Feature Engineering) ⬜ 未开始

- **出处**: NeurIPS 2023 Workshop — LLM 单轮生成特征，最直接对比
- **方法**: LLM 根据数据描述和上下文一次性生成特征代码
- **配置**:
  - 使用同样的 DeepSeek-Chat LLM
  - 生成特征数上限: 30
  - 无多轮迭代、无验证反馈
- **两个数据集都要跑**
- **安装**: `pip install caafe`
- **目的**: 证明 KnowFeat 的多轮迭代 + 验证反馈 + 知识注入优于 LLM 单轮生成

#### 2.6 Random Features (随机特征下界) ⬜ 未开始

- **方法**: 随机生成 30 个特征（随机选择字段 + 随机聚合函数）
- **配置**:
  - 与 KnowFeat 相同的 L2/L3 验证
  - 多次运行取平均 (3 次)
- **目的**: 作为下界，证明其他方法确实优于随机

#### 2.7 LLM-only (无知识注入) ⬜ 未开始

- **方法**: 使用 KnowFeat 框架但 **移除所有领域知识注入**
  - 不加载裁判文书风险因子 (risk_factors)
  - 不加载风险分类表 / 检测规则 / 专家访谈
  - Prompt 中只给数据 schema，不给领域上下文
- **配置**: 
  - 同样的 ReactAgent + 三级验证 + 前向选择
  - max_rounds=10, max_features=30
- **目的**: 证明知识注入的重要性
- **实现**: 在 `prompt/builder.py` 中设置 `domain_knowledge=None` 和 `risk_factors=[]`
- **注意**: 与消融实验 3.1 复用

#### 2.8 Rule-based Feature Templates (基于规则的特征模板) ⬜ 未开始

- **方法**: 从知识库中的 88 个风险因子 → 直接用规则映射为特征
  - 每个 risk_factor 的 `data_hint` → 硬编码聚合 (mean/max/count/ratio)
  - 不使用 LLM，纯规则模板
- **配置**:
  - 遍历所有 risk_factors JSON
  - 对每个 `data_hint` 中的字段生成简单统计特征
  - L2/L3 验证筛选
- **目的**: 证明 LLM 的创造力超过简单规则映射
- **输出**: 规则模板生成的特征数 + recall/precision

---

### 实验 3：消融实验 (Ablation Study)

> 目标：验证 KnowFeat 各组件的独立贡献
>
> **基准**: Stage 2 完整运行结果 (18 特征, recall=0.9737)
>
> 每个消融实验移除一个组件，其余保持不变，对比最终 recall 和接受特征数。

#### 3.1 w/o Knowledge Injection (无知识注入) ⬜ 未开始

- **移除**: Key-clue Extractor 输出 + domain_knowledge 全部清空
- **保留**: ReactAgent, 三级验证, 前向选择, SHAP, 失败记忆, 跨维度, 维度管理
- **改动点**: `prompt/builder.py` 中 `domain_knowledge=None`, `risk_factors=[]`
- **预期**: 生成特征质量下降，接受率降低
- **注意**: 这个和 Baseline 2.5 (LLM-only) 是同一个实验，可以复用

#### 3.2 w/o Three-Level Verification (无三级验证) ⬜ 未开始

- **移除**: L1 沙盒验证 + L2 统计验证，直接用 L3 模型验证
- **保留**: 其他全部组件
- **改动点**: `verification/` 模块，跳过 L1 和 L2，代码能跑就直接进 L3
- **预期**: 更多低质量特征进入 L3，增加计算开销；可能引入噪声特征

#### 3.3 w/o Noise Calibration (无噪声校准) ⬜ 未开始

- **移除**: 噪声校准阈值计算，改用固定阈值 (IV>0.02, KS>0.05)
- **保留**: 三级验证结构 (L1+L2+L3) + 其他组件
- **改动点**: `verification/statistics.py` 中 `calibrate_noise_thresholds()` 返回固定值
- **预期**: 固定阈值可能过松或过紧，不如数据自适应

#### 3.4 w/o SHAP Feedback (无 SHAP 反馈) ⬜ 未开始

- **移除**: 每轮结束后的 SHAP 分析 + 反馈注入 prompt
- **保留**: 其他全部组件
- **改动点**: `search/shap_feedback.py` 中 `analyze_shap_feedback()` 返回空
- **预期**: Agent 失去对已有特征重要性分布的感知，生成特征重复度上升

#### 3.5 w/o Forward Selection (无前向选择) ⬜ 未开始

- **移除**: 前向选择 (`forward_select()`)，L2 通过的特征直接进入接受池
- **保留**: 其他全部组件
- **改动点**: `search/forward_selection.py` 中跳过前向选择，全部接受
- **预期**: 冗余特征增加，模型性能可能下降或持平

#### 3.6 w/o Failure Memory (无失败记忆) ⬜ 未开始

- **移除**: 失败记忆注入 (`FailureMemory`)
- **保留**: 其他全部组件
- **改动点**: `search/failure_memory.py` 中 `get_top_failures()` 返回空列表
- **预期**: Agent 重复犯相同错误，生成效率降低

#### 3.7 w/o Cross-Dimension Propagation (无跨维度传播) ⬜ 未开始

- **移除**: 跨维度传播 (`CrossDimPropagator`)
- **保留**: 其他全部组件
- **改动点**: `search/cross_dim.py` 中 `propagate()` 返回空列表
- **预期**: 维度间信息隔离，后期生成质量下降

#### 3.8 w/o Dimension Management (无维度管理) ⬜ 未开始

- **移除**: 四维度结构 (`DimensionManager`)，改为单一池
- **保留**: 其他全部组件
- **改动点**: `search/dimension_manager.py` 中合并为单维度
- **预期**: 特征分布不均匀，某些维度被忽略

---

### 实验 4：知识迁移实验

> 目标：验证跨域知识迁移的有效性

#### 4.1 Stage 1 → Stage 2 知识迁移 ⬜ 未开始

- **流程**:
  1. Stage 1 在 BABD-13 上生成新特征
  2. 提取新特征的 provenance (知识来源、设计逻辑、维度归属)
  3. 将 provenance 转化为结构化风险因子 JSON
  4. 注入 Stage 2 的知识库 (`inputs/stage2_knowledge/risk_factors/`)
  5. Stage 2 以 95+18 已有特征 + 新知识再运行 3-5 轮
- **对比**: 迁移前 Stage 2 (18特征) vs 迁移后 Stage 2 (18+?特征)
- **指标**: recall 增益、新接受特征数、知识转化率 (迁移知识→实际特征的比例)

#### 4.2 Stage 2 → Stage 1 反向迁移 ⬜ 未开始 (可选)

- **流程**: Stage 2 的 18 个特征的知识 → 反向注入 Stage 1
- **目的**: 验证知识迁移的双向性
- **优先级**: P2，时间不够可省略

#### 4.3 无迁移对照组 ⬜ 未开始

- **方法**: Stage 2 不注入 Stage 1 知识，直接从 18 特征开始再跑 3-5 轮
- **目的**: 与 4.1 对照，证明迁移的知识确实带来增益（而非单纯多跑几轮的效果）

---

### 实验 5：效率与成本分析

> 目标：展示 KnowFeat 的计算成本和效率

#### 5.1 LLM 调用统计 ⬜ 未开始

从日志中提取以下数据：

| 指标 | Stage 1 | Stage 2 |
|------|---------|---------|
| 总 LLM 调用次数 | ? | ? |
| 总输入 token 数 | ? | ? |
| 总输出 token 数 | ? | ? |
| API 费用估算 ($) | ? | ? |
| 平均每特征 LLM 调用数 | ? | ? |
| 平均每特征生成时间 (s) | ? | ? |

#### 5.2 时间分解 ⬜ 部分完成

Stage 2 总耗时 34,065.8s，需要分解为：

| 阶段 | 耗时 | 占比 |
|------|------|------|
| LLM 生成代码 | ? | ? |
| L1 沙盒验证 | ? | ? |
| L2 统计验证 | ? | ? |
| L3 模型验证 (5-fold CV) | ? | ? |
| SHAP 分析 | ? | ? |
| 前向选择 | ? | ? |
| 其他 (数据加载等) | ? | ? |

---

### 实验 6：敏感性分析 (Sensitivity Analysis)

> 目标：关键超参数对性能的影响

#### 6.1 scale_pos_weight 敏感性 ✅ 部分完成

- 已有数据: 19, 39, 59, 99, 307.8 在 95/105 特征上的结果
- **还需**: 在 KnowFeat 生成的 95+18 特征上用不同 scale_pos_weight 跑
- **目的**: 证明 scale_pos_weight=59 的选择合理性

#### 6.2 max_rounds 敏感性 ⬜ 未开始

- **方法**: 从已有 Stage 2 运行日志中分析
  - 绘制 recall vs. round 曲线
  - 绘制 累计接受特征数 vs. round 曲线
  - 分析边际收益递减点
- **不需要重新跑实验**，从已有日志/结果中提取即可

#### 6.3 redundancy_max 阈值敏感性 ⬜ 未开始

- **配置**: redundancy_max = {0.5, 0.6, 0.7, 0.8, 0.9}
- **方法**: 在已生成的候选特征上回测不同冗余阈值的筛选结果
- **目的**: 找到冗余/多样性的最佳平衡

#### 6.4 不同 LLM 的对比 ⬜ 未开始

- **配置**: 
  - DeepSeek-Chat (当前使用)
  - GPT-4o
  - Claude Sonnet (如果 API 可用)
  - Qwen-72B
- **方法**: 各 LLM 跑 3 轮 Stage 2，对比生成特征质量 (接受率、L2 通过率)
- **目的**: 验证框架对 LLM 的鲁棒性
- **注意**: 这个实验成本较高，优先级 P2

---

### 实验 7：Case Study

> 目标：定性分析 KnowFeat 的能力

#### 7.1 特征生成过程可视化 ⬜ 未开始

- 选取 2-3 个有代表性的特征，展示完整生成过程：
  - 知识来源 → Prompt → LLM 生成 → L1/L2/L3 验证 → SHAP 反馈
  - 展示失败记忆如何帮助 Agent 修正
  - 展示跨维度传播如何启发新特征

#### 7.2 知识迁移案例 ⬜ 未开始

- 选取 Stage 1 → Stage 2 迁移成功的案例：
  - BABD-13 上发现的风险模式
  - 如何转化为 SimECNY 上的具体特征
  - 转化前后的特征对比

---

## 四、实验执行顺序

> 按照依赖关系和优先级排列

### Phase 1: 数据准备 (当前阶段)

- [ ] **T1.1** 上传 BABD-13 数据到 `/root/autodl-tmp/babd13/`
- [ ] **T1.2** 实现 BABD-13 标签映射 (13-class → binary)
- [ ] **T1.3** 适配 Stage 1 数据加载器
- [ ] **T1.4** 在 BABD-13 上跑基线 (148 特征, 确定 scale_pos_weight)
- [ ] **T1.5** 安装 baseline 工具: `pip install openfe featuretools autofeat caafe`

### Phase 2: 主实验 + Baseline (P0)

- [ ] **T2.1** Stage 1 完整运行 (BABD-13, max 5 rounds) — 依赖 T1.1-T1.4
- [ ] **T2.2** OpenFE baseline (SimECNY) — 依赖 T1.5
- [ ] **T2.3** OpenFE baseline (BABD-13) — 依赖 T1.5, T2.1
- [ ] **T2.4** Featuretools baseline (SimECNY) — 依赖 T1.5
- [ ] **T2.5** Featuretools baseline (BABD-13) — 依赖 T1.5, T2.1
- [ ] **T2.6** AutoFeat baseline (SimECNY) — 依赖 T1.5
- [ ] **T2.7** AutoFeat baseline (BABD-13) — 依赖 T1.5, T2.1
- [ ] **T2.8** CAAFE baseline (SimECNY) — 依赖 T1.5
- [ ] **T2.9** CAAFE baseline (BABD-13) — 依赖 T1.5, T2.1
- [ ] **T2.10** Random Features baseline (SimECNY, 3 runs)
- [ ] **T2.11** LLM-only baseline (SimECNY, 无知识注入) — 可以并行
- [ ] **T2.12** Rule-based template baseline (SimECNY)
- [ ] **T2.13** Wallet_type 分场景评估 (SimECNY, baseline vs KnowFeat)
- [ ] **T2.14** ModelSet 11 模型对比 (SimECNY, 95+18 特征)

### Phase 3: 消融实验 (P0)

> 全部在 SimECNY 上运行，与 Stage 2 已有结果对比

- [ ] **T3.1** w/o Knowledge Injection — 与 T2.8 复用
- [ ] **T3.2** w/o Three-Level Verification
- [ ] **T3.3** w/o Noise Calibration
- [ ] **T3.4** w/o SHAP Feedback
- [ ] **T3.5** w/o Forward Selection
- [ ] **T3.6** w/o Failure Memory
- [ ] **T3.7** w/o Cross-Dimension Propagation
- [ ] **T3.8** w/o Dimension Management

### Phase 4: 知识迁移 (P1)

- [ ] **T4.1** 提取 Stage 1 知识 → 结构化 JSON — 依赖 T2.1
- [ ] **T4.2** Stage 2 迁移后再运行 (3-5 rounds) — 依赖 T4.1
- [ ] **T4.3** Stage 2 无迁移对照组 (3-5 rounds) — 与 T4.2 并行

### Phase 5: 分析实验 (P1-P2)

- [ ] **T5.1** LLM 调用统计 (从日志提取)
- [ ] **T5.2** 时间分解分析 (从日志提取)
- [ ] **T5.3** max_rounds 敏感性 (从已有日志分析)
- [ ] **T5.4** scale_pos_weight 在 95+18 特征上的敏感性
- [ ] **T5.5** redundancy_max 阈值回测
- [ ] **T5.6** 不同 LLM 对比 (可选, 成本高)

### Phase 6: Case Study + 可视化 (P2)

- [ ] **T6.1** 特征生成过程可视化 (2-3 个案例)
- [ ] **T6.2** 知识迁移案例分析

---

## 五、实验配置统一规范

### XGBoost 参数 (所有实验统一)

```python
xgb_params = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": 59.0,  # SimECNY; BABD-13 需单独计算
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}
```

### 评估指标 (所有实验统一)

- **主指标**: Recall (TPR) — AML 场景下漏报成本远高于误报
- **约束指标**: Precision ≥ 0.90
- **业务指标**: CR@K (Top-K Capture Rate) = Top K% 样本中的洗钱数 / 总洗钱数, K = 1%, 3%, 5%
- **报告指标**: Recall, Precision, F1, AUC-ROC, CR@1%, CR@3%, CR@5%
- **评估方法**: 5-fold StratifiedKFold CV, report mean±std

### KnowFeat 默认参数

```python
knowfeat_params = {
    "max_rounds": 10,
    "max_features": 30,
    "convergence_rounds": 2,
    "dimensions": ["金额频次", "时间地域", "行为序列", "主体关联"],
    "initial_budget_per_dim": 3,
    "noise_percentile": 95.0,
    "iv_min_fallback": 0.02,
    "ks_min_fallback": 0.02,
    "redundancy_max": 0.7,
    "precision_min": 0.90,
    "delta_recall_min": 0.0,
}
```

---

## 六、预期论文表格

### Table 1: Main Results — 两个数据集上的性能

| Method | SimECNY Recall | SimECNY Prec. | SimECNY F1 | SimECNY CR@3% | BABD-13 Recall | BABD-13 Prec. | BABD-13 F1 |
|--------|---------------|--------------|------------|--------------|---------------|--------------|------------|
| Baseline (Manual) | 0.9538 | 0.9402 | 0.9470 | ? | ? | ? | ? |
| Random | ? | ? | ? | ? | ? | ? | ? |
| OpenFE | ? | ? | ? | ? | ? | ? | ? |
| Featuretools | ? | ? | ? | ? | ? | ? | ? |
| AutoFeat | ? | ? | ? | ? | ? | ? | ? |
| CAAFE | ? | ? | ? | ? | ? | ? | ? |
| LLM-only | ? | ? | ? | ? | ? | ? | ? |
| Rule-based | ? | ? | ? | ? | ? | ? | ? |
| **KnowFeat** | **0.9737** | **0.9621** | **0.9679** | ? | ? | ? | ? |
| KnowFeat+Transfer | ? | ? | ? | ? | — | — | — |

### Table 1b: Wallet_type 分场景 (SimECNY)

| wallet_type | Baseline Recall | KnowFeat Recall | Δrecall |
|-------------|----------------|----------------|---------|
| Corporate (1) | ? | ? | ? |
| Personal (2) | ? | ? | ? |
| Merchant (3) | ? | ? | ? |

### Table 2: Ablation Study (SimECNY)

| Variant | Recall | Precision | #Accepted | Δrecall vs Full |
|---------|--------|-----------|-----------|----------------|
| Full KnowFeat | 0.9737 | 0.9621 | 18 | — |
| w/o Knowledge | ? | ? | ? | ? |
| w/o 3-Level Verify | ? | ? | ? | ? |
| w/o Noise Calib. | ? | ? | ? | ? |
| w/o SHAP Feedback | ? | ? | ? | ? |
| w/o Forward Select | ? | ? | ? | ? |
| w/o Failure Memory | ? | ? | ? | ? |
| w/o Cross-Dim | ? | ? | ? | ? |
| w/o Dim Mgmt | ? | ? | ? | ? |

### Table 3: Knowledge Transfer

| Setting | Recall | Precision | #New Features | Δrecall |
|---------|--------|-----------|--------------|---------|
| Stage 2 (before transfer) | 0.9737 | 0.9621 | 18 | — |
| Stage 2 (after transfer) | ? | ? | 18+? | ? |
| Stage 2 (no transfer, more rounds) | ? | ? | 18+? | ? |

### Table 4: ModelSet 11 模型对比

| Model | Recall | Precision | F1 | CR@3% |
|-------|--------|-----------|------|-------|
| XGBoost | 0.9737 | 0.9621 | 0.9679 | ? |
| CatBoost | ? | ? | ? | ? |
| LightGBM | ? | ? | ? | ? |
| ... | ... | ... | ... | ... |

### Figure 1: Recall vs. Round 曲线 (收敛分析)
### Figure 2: 各组件消融对比柱状图
### Figure 3: 知识迁移流程图 + 效果
### Figure 4: SHAP feature importance 对比 (baseline vs KnowFeat)
### Figure 5: 各维度特征分布饼图
### Figure 6: Wallet_type 分场景 Recall 对比柱状图
### Figure 7: CR@K 曲线 (K=1%~10%)

---

## 七、估算资源和时间

| 实验 | 预计耗时 | LLM 调用 | 备注 |
|------|---------|---------|------|
| Stage 1 (BABD-13) | ~8h | ~200 calls | 5 rounds |
| 每个消融实验 | ~8-10h | ~200 calls | 与完整运行相同配置 |
| 8 个消融实验合计 | ~3-4 天 | ~1600 calls | 可并行但受 GPU 限制 |
| CAAFE | ~2h | ~50 calls | LLM 单轮生成 |
| OpenFE/Featuretools/AutoFeat | 各 ~1h | 0 | 纯计算，无 LLM |
| Random baseline (3 runs) | ~0.5h | 0 | 纯计算 |
| Wallet_type 分场景 | ~0.5h | 0 | 复用已有模型，切分评估 |
| ModelSet 11 模型 | ~4h | 0 | 11×5-fold CV |
| LLM-only baseline | ~8h | ~200 calls | 与消融 3.1 复用 |
| 知识迁移实验 | ~5h | ~100 calls | 3-5 rounds |
| 不同 LLM 对比 | ~12h × 3 | ~600 calls | 可选 |
| 分析/统计/可视化 | ~2h | 0 | 纯数据处理 |

**总计最小估算**: ~7-8 天 (串行), ~3-4 天 (部分并行)

---

## 八、当前进度追踪

- [x] Stage 2 完整运行 (18 特征)
- [x] 基线调优 (scale_pos_weight 5 档)
- [x] Stage 1 知识获取 (13 文档 → 88 risk factors)
- [x] RESULT.md 文件维护
- [ ] BABD-13 数据下载和上传 ← **当前阶段**
- [ ] BABD-13 标签映射和适配
- [ ] Stage 1 运行
- [ ] Baseline 对比实验
- [ ] 消融实验
- [ ] 知识迁移实验
- [ ] 论文图表制作
