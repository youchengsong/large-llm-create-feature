# LLM 自动特征工程方案：数字人民币反洗钱

> **核心思路**：构建一个系统，让 LLM 自动发现新特征并生成可执行代码，同时为每个特征提供领域依据的可解释性说明。人不去设计特征，LLM 是特征工程师。

---

## 一、问题定义

### 1.1 我们要解决什么问题

我们有一个数字人民币（e-CNY）反洗钱检测系统，当前已有人工设计的 40+ 个特征（见 `inputs/baseline/feature_engineering.py`）。这些特征存在已知缺口（多个数据字段未利用、时间窗口单一等），但人工逐一设计新特征效率低、容易遗漏。

**目标**：用 LLM 自动发现人工未覆盖的新特征，每个新特征必须：
1. 生成可直接执行的 Python 代码（不是文字描述或逻辑表达式）
2. 附带可解释性说明（为什么创建这个特征、对应什么反洗钱领域知识）
3. 经过验证确认有效（对下游模型有增量贡献）

### 1.2 预测目标

**交易级二分类**：每行 = 一笔交易，标签 = `is_risk`（0 或 1）。

依据：人工基线 `feature_engineering.py` 以交易表为主表，将 `is_risk` 作为 label（代码第 151 行：`df['is_laundering'] = df['is_risk']`）。开户表的 `is_abnormal` 和销户表的 `is_zs_laundering` 是辅助信息，不是预测目标。

### 1.3 数据来源

数据来自 SimECNY 模拟器（`inputs/SimECNY/`），包含三个关联表：

**开户表**（16 字段，每行 = 一个钱包）：

| 字段名 | 类型 | 说明 | 取值范围 |
|--------|------|------|---------|
| wallet_id | string | 钱包唯一标识 | 14位数字 |
| wallet_type | int | 钱包类型 | 1=企业, 2=个人, 3=商户 |
| wallet_level | int | 钱包等级 | 1-4类（1类限额最高） |
| wallet_open_cert | string | 开户身份证号 | 18位，4类钱包为NULL |
| wallet_open_cert_expire | string | 证件过期日期 | YYYY-MM-DD |
| wallet_open_date | string | 开户日期 | YYYY-MM-DD |
| wallet_open_timestamp | string | 开户精确时间 | YYYY-MM-DD HH:MM:SS |
| wallet_open_tel | string | 手机号 | 11位 |
| realname_auth_time | string | 实名认证时间 | YYYY-MM-DD HH:MM:SS，4类为NULL |
| accociated_bank_account | string | 关联银行卡号 | 完整卡号，3/4类为NULL |
| bank_account_number | string | 银行编号 | 前4位 |
| init_balance | float | 初始余额 | 0-N（CNY） |
| region_code | string | 开户地区 | 6位行政区划码 |
| is_abnormal | bool | 是否异常账户 | True/False（~5%异常率） |
| open_device | string | 开户设备 | MAC地址，格式 XX:XX:XX:XX:XX:XX |
| open_ip | string | 开户IP | IPv4地址 |

**交易表**（20 字段，每行 = 一笔交易）：

| 字段名 | 类型 | 说明 | 取值范围 |
|--------|------|------|---------|
| tx_id | string | 交易唯一标识 | |
| timestamp | string | 交易时间 | YYYY-MM-DD HH:MM:SS |
| src | string | 发送方 wallet_id | |
| dst | string | 接收方 wallet_id | |
| amount | float | 交易金额 | 0.1 - 100,000（CNY） |
| transaction_motif | string | 交易动机类型 | 7种：single_transaction, normal_small_high_freq, regular_large_low_freq, small_amount_testing, merchant_laundering, class4_laundering, online_laundering |
| motif_id | string | 动机组ID | 关联同一批次的交易 |
| transaction_mode | string | 交易结构模式 | 7种：single, forward, fan_out, fan_in, gather_scatter, scatter_gather, many_to_many |
| is_risk | string | **预测标签** | 0=正常, 1=风险（~5%风险率） |
| is_src_victim | int | 发送方是否受害者 | 0/1 |
| src_bank_account_number | string | 发送方银行编号 | 前4位 |
| dst_bank_account_number | string | 接收方银行编号 | 前4位 |
| src_wallet_level | int | 发送方钱包等级 | 1-4 |
| dst_wallet_level | int | 接收方钱包等级 | 1-4 |
| src_device | string | 发送方设备 | MAC地址 |
| src_ip | string | 发送方IP | IPv4 |
| dst_device | string | 接收方设备 | MAC地址 |
| dst_ip | string | 接收方IP | IPv4 |
| interval | float | 与关联交易的时间间隔 | 秒 |
| hour | int | 交易小时 | 0-23 |

**销户表**（6 字段，每行 = 一次销户）：

| 字段名 | 类型 | 说明 | 取值范围 |
|--------|------|------|---------|
| wallet_id | string | 钱包标识 | |
| zs_id | string | 销户流水号 | ZS000001... |
| zs_channel | string | 销户渠道 | mobile_app, web, counter |
| zs_timestamp | string | 销户时间 | YYYY-MM-DD HH:MM:SS |
| wallet_open_tel | string | 手机号 | 11位 |
| is_zs_laundering | int | 是否洗钱销户 | 0/1 |

### 1.4 三表关联关系

```
交易表（主表，每行一笔交易）
  ├── src (wallet_id) ──→ 开户表.wallet_id → 获取发送方的账户属性
  │                   ──→ 销户表.wallet_id → 获取发送方的销户信息
  ├── dst (wallet_id) ──→ 开户表.wallet_id → 获取接收方的账户属性
  │                   ──→ 销户表.wallet_id → 获取接收方的销户信息
  └── is_risk (label)
```

关联键是 `wallet_id`，交易表的 `src` 和 `dst` 字段都是 wallet_id。一个钱包可以出现在多笔交易中（作为 src 或 dst），所以是一对多关系。

### 1.5 SimECNY 模拟器中编程的异常模式

模拟器在生成数据时，按以下规则生成异常行为（这些信息**不告诉 LLM**，用于我们自己验证 LLM 是否能发现这些模式）：

| 异常类型 | 编程特征 | 配置参数 |
|---------|---------|---------|
| **身份证复用** | 异常钱包 90% 共享身份证（vs 正常 5%），组大小 2-10 | `cert_reuse_rate: 0.9` |
| **手机号复用** | 异常钱包 90% 共享手机号（vs 正常 2%） | `tel_reuse_rate: 0.9` |
| **高风险地区** | 异常钱包 70% 在西藏/新疆（vs 正常 2%） | `region_code: 540000, 650000` |
| **实名延迟** | 异常钱包认证延迟 20-30 天（vs 正常当天） | `auth_delay: 20-30 days` |
| **夜间开户** | 异常钱包 50% 夜间开户（vs 正常 1%） | `night_open: 22-5时` |
| **小额试探** | 金额 0.1-1.0 元，高频短间隔 | `small_amount_testing` |
| **商户洗钱** | 金额 1K-100K，many_to_many 模式 80% | `merchant_laundering` |
| **4类钱包洗钱** | 金额 100-2K，90% 夜间（0-6时, 22-23时） | `class4_laundering` |
| **线上跑分** | 金额 10-1K，80% 夜间（20-1时） | `online_laundering` |
| **快速销户** | 60% 风险钱包在最后交易 1-2 天内注销 | `fast_closure: 0.6` |

---

## 二、参考文献与借鉴方式

本方案参考 6 篇论文，每篇借鉴特定组件。下面详细说明每篇论文的核心算法、我们具体借鉴什么、怎么用、为什么这样组合。

### 2.1 LLM-FE — 搜索引擎骨架

> Abhyankar et al. "LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers." ICLR 2026 Workshop. arXiv:2503.14434

**论文做了什么**：

LLM-FE 把 LLM 当作进化优化器——维护多个"岛屿"（独立种群），每个岛屿保存一批特征转换程序。每轮从岛屿中选出高分程序作为 in-context 示例给 LLM，让 LLM 生成新程序。新程序在数据上执行，评估分数，再放回岛屿。

**核心算法**（我们借鉴的部分）：

```
算法：岛屿模型进化搜索
参数：m=3 个岛屿，b=3 个候选/岛屿/轮，k=2 个 in-context 示例，τ=温度参数

初始化：每个岛屿 Island_j = {} (空集)

FOR round = 1 to T:
    选择一个岛屿 j（轮流或随机）
    
    # Boltzmann 采样选 in-context 示例
    IF Island_j 非空:
        对 Island_j 中每个程序 i，计算选择概率：
            P_i = exp(score_i / τ) / Σ_all exp(score_all / τ)
        从 P_i 分布中采样 k=2 个程序作为示例
    ELSE:
        示例 = 空（冷启动）
    
    # LLM 生成新候选
    prompt = System_Prompt + 示例 + "请生成 b=3 个新特征转换"
    candidates = LLM(prompt)  → 得到 b 个 Python 函数
    
    # 执行和评估
    FOR each candidate c in candidates:
        执行 c 在训练数据上 → 生成新特征列
        训练下游模型（XGBoost）→ 在验证集上评估 → score_c
    
    # 更新岛屿
    将 (candidate, score) 加入 Island_j
    相同 score 的程序聚类合并（去重）
```

**我们怎么用**：
- 完整采用岛屿模型和 Boltzmann 采样机制
- 修改评估方式：原论文用 Accuracy，我们改为 PR-AUC（因为 AML 数据不平衡）
- 扩展到多表：原论文仅支持单表输入，我们的 LLM 函数接收三个 DataFrame
- 增强领域知识：原论文仅靠特征名，我们注入结构化专家知识

**论文实验数据**：
- 27 个 OpenML 数据集上排名 1.54（第一）
- 消融实验：去掉领域知识（特征名匿名化），准确率从 0.848 降至 0.626（-26%）
- 消融实验：去掉进化精炼（无迭代反馈），准确率降至 0.587（-31%）
- 模型验证：Llama-3.1-8B 和 GPT-3.5-Turbo 均有效，框架对底层 LLM 不敏感

### 2.2 MedFeat — SHAP 导航系统

> "MedFeat: Model-Aware and Explainability-Driven Feature Engineering with LLMs." arXiv:2603.02221, 2026

**论文做了什么**：

MedFeat 在每轮特征生成后计算 SHAP 特征重要性，用 SHAP 值引导 LLM 下一轮的搜索方向。高重要性特征方向被更频繁探索，低重要性方向被抑制。

**核心算法**（我们借鉴的部分）：

```
算法：SHAP 引导的特征生成

FOR round t = 1 to T:
    # Step 1: 计算 SHAP 重要性
    model_t = 训练 XGBoost(当前特征集, y_train)
    shap_values = TreeExplainer(model_t).shap_values(X_val)
    
    对每个特征 i：
        s_t_i = mean(|shap_values[:, i]|)      # 全局平均绝对 SHAP 值
    
    归一化：s̃_t_i = s_t_i / Σ_all s_t_all     # 转为概率分布
    
    # Step 2: 构建 prompt 反馈
    shap_feedback = {
        "top_10_features": 按 s̃_t_i 排序的前10个特征及其重要性,
        "weak_dimensions": SHAP 值低的特征类别（提示 LLM 这些方向需要探索）,
        "model_type": "XGBoost"  # 模型感知：告诉 LLM 下游模型类型
    }
    
    # Step 3: LLM 生成
    candidates = LLM(System_Prompt + shap_feedback + memory)
    
    # Step 4: 特征接受条件
    FOR each candidate k:
        加入 candidate_k 到特征集，重新训练模型
        L_t_k = 验证集上的 loss
    
    k* = argmin_k(L_t_k)                       # 选最佳候选
    IF L_t_k* >= L_t_baseline + β (β=0.01):    # 必须超过基线 + 容忍度
        接受 candidate_k*，更新特征集和 baseline
    ELSE:
        拒绝，记录原因到失败记忆库
    
    # Step 5: 记忆更新
    IF 接受: 加入成功记忆库（用于 in-context 示例）
    IF 拒绝: 加入失败记忆库（防止重复生成）
```

**我们怎么用**：
- 完整采用 SHAP 引导机制：每轮计算 SHAP，反馈给 LLM
- 采用接受条件 β=0.01：只接受有真实增量的特征
- 采用失败记忆：记录拒绝原因，避免重复
- 采用模型感知：告诉 LLM 下游模型是 XGBoost，引导生成树模型友好的特征
- 采用隐私保护：只发送 schema 和统计摘要给 LLM，不发原始数据
- **不采用**它的搜索方式（单线迭代太慢），用 LLM-FE 的岛屿并行替代

**论文实验数据**：
- 消融：去掉 SHAP 引导，F1 下降 9-26%（SHAP 引导是最关键组件）
- 消融：去掉模型感知，F1 最高下降 35.9%

### 2.3 CAAFE — 代码安全与错误反馈

> Hollmann et al. "Large Language Models for Automated Data Science: Introducing CAAFE." NeurIPS 2023. arXiv:2305.03403

**论文做了什么**：

CAAFE 让 LLM 生成 Python 代码来创建新特征，通过安全白名单和错误反馈循环保证代码可靠执行。

**我们借鉴的具体机制**：

```
代码安全白名单（借鉴 CAAFE）：
  允许的模块: pandas, numpy, scipy.stats, sklearn.preprocessing
  允许的操作: 算术运算, groupby, merge, apply, 条件判断
  禁止的操作: os, sys, subprocess, open(), exec(), eval(), 网络请求
  
错误反馈循环（借鉴 CAAFE）：
  IF 代码执行失败:
      error_msg = 捕获的异常信息（类型 + 行号 + 消息）
      retry_prompt = f"你上一次生成的代码执行失败：\n{error_msg}\n请修正并重新生成。"
      corrected = LLM(retry_prompt)
      最多重试 2 次
  
输出格式（借鉴 CAAFE）：
  每个特征必须附带文本解释，说明：
  - 为什么创建这个特征
  - 它捕捉了什么反洗钱模式
  - 对应哪个领域知识来源
```

**论文实验数据**：14 个数据集中 11 个获得提升，平均 ROC AUC +2.4%

### 2.4 OCTree — 可解释性增强（可选）

> Nam et al. "OCTree: Optimized Feature Generation via LLMs with Decision Tree Reasoning." NeurIPS 2024. arXiv:2406.08527

**论文做了什么**：

训练一个浅层决策树 CART(max_depth=4)，将决策树的分裂规则转为自然语言反馈给 LLM，让 LLM 理解"什么阈值和条件组合对分类有帮助"。

**我们可选借鉴的部分**：

```
决策树反馈（可选增强，在 SHAP 反馈之外额外提供）：
  cart = DecisionTreeClassifier(max_depth=4)
  cart.fit(X_train_augmented, y_train)
  rules = 提取决策树规则为自然语言
  
  例如：
  "当 amount > 5000 且 hour ∈ [22,23,0,1,2,3] 且 transaction_mode = fan_out 时，
   is_risk 概率为 78%。这说明大额夜间扇出交易是强风险信号。"
  
  → 将 rules 加入 User Prompt，作为 SHAP 反馈的补充
```

**论文实验数据**：加入决策树反馈后，改进幅度从 1.4-17.2% 提升到 2.6-20.1%

**为什么可选**：SHAP 反馈已经提供了特征方向信息，决策树反馈是额外的语义增强，实现优先级低于核心流程。

### 2.5 CoFEH — 后续升级方向

> Xu et al. "CoFEH: LLM-driven Feature Engineering Empowered by Collaborative Bayesian HPO." ASPLOS 2026. arXiv:2602.09851

**论文做了什么**：

用 MCTS（蒙特卡洛树搜索）替代进化搜索，同时联合优化特征工程和模型超参数。

**我们暂不采用但关注的**：
- MCTS 全局搜索（比进化搜索理论上更强，但实现复杂度高得多）
- 特征工程 + 超参数调优联合优化
- MAB-PUCB 动态调度（早期偏重特征探索，后期偏重超参数）

**暂不采用原因**：2026 年 2 月刚发布，实现复杂度最高（MCTS + 贝叶斯优化 + 动态调度），作为 V2 升级方向。

**论文实验数据**：排名 1.75（仅次于 LLM-FE 的 1.54），消融显示去掉联合优化后排名降至 3.75

### 2.6 Balek 2025 — 统计验证方法

> Balek et al. "LLM-based Feature Generation from Text for Interpretable ML." Machine Learning, 2025. arXiv:2409.07132

**论文做了什么**：

用 Bootstrap 卡方检验验证 LLM 生成特征与标签的统计显著性，用 Cramer's V 衡量效应大小，用人工评估验证特征语义合理性。

**我们借鉴的具体方法**：

```
Bootstrap 卡方检验（验证特征有效性）：
  FOR i = 1 to 2500:           # 2500 次重采样
      sample = 有放回抽样(feature, label)
      chi2_i, p_i = chi2_contingency(crosstab(sample_feature, sample_label))
  
  p_value = mean(所有 p_i)
  IF p_value < 0.05: 特征与标签显著相关

Cramer's V（衡量效应大小，不受样本量影响）：
  V = sqrt(χ² / (N × (min(r,c) - 1)))
  
  解释：V < 0.10 可忽略, 0.10-0.30 弱, 0.30-0.50 中等, > 0.50 强

人工评估协议：
  随机抽取 N 个 LLM 生成的特征
  评估者对每个特征打分 1-5（1=无关, 5=高度相关）
  报告平均分和分布
```

**论文实验数据**：62 个 LLM 特征的 Cramer's V 为 0.191-0.865（中等到强关联），41 名评估者中 27/100 特征评分 4.0-5.0

**关键发现：五篇论文（LLM-FE, CAAFE, MedFeat, OCTree, CoFEH）无一使用统计检验。** 我们用 Balek 的方法填补这个缺口。

### 2.7 各组件如何组合

```
[1] 领域知识（我们独有）
    - 专家访谈 + 200+识别点 + 38风险指标 + 监管规则
            |
            | 注入 System Prompt
            v
[2] 进化搜索引擎（来自 LLM-FE, 见2.1）
    - 岛屿1 --> LLM --> 候选特征代码 --+
    - 岛屿2 --> LLM --> 候选特征代码 --+--> 汇总
    - 岛屿3 --> LLM --> 候选特征代码 --+
    - Boltzmann 采样选 in-context 示例
    - 失败记忆: 拒绝原因（来自 MedFeat, 见2.2）
            |
            v
[3] 代码执行沙盒（来自 CAAFE, 见2.3）
    - 安全白名单 -> 语法检查 -> 执行 -> 错误反馈给 LLM
            |
            v
[4] SHAP 导航系统（来自 MedFeat, 见2.2）
    - 1. 训练 XGBoost（含新特征）
    - 2. 计算 SHAP 重要性 -> 反馈给 LLM
    - 3. 接受条件: deltaPR-AUC >= beta (0.01)
    - 4. 反馈给搜索引擎: 哪些方向有价值、哪些冗余
            |
            v
[5] 统计验证层（来自 Balek, 见2.6）
    - Bootstrap 卡方检验 + Cramer's V + 互信息
```

**组合逻辑**：
- **LLM-FE** 提供搜索效率（多岛屿并行），但搜索方向盲目
- **MedFeat** 提供搜索方向（SHAP 引导），但搜索效率低（单线迭代）
- 两者结合：**高效且有方向的搜索**
- **CAAFE** 提供代码执行的安全保障和错误反馈
- **Balek** 提供超越所有论文的统计验证（五篇论文无一使用统计检验）
- **我们的领域知识**是所有论文都没有的独特信号源

### 2.8 方法全景对比

"平均排名"指各方法在公共 OpenML benchmark 数据集上的平均名次（所有数据集上排序后取均值，越小越好，1.0 = 每个数据集都第一）。MedFeat 使用不同的数据集，无可比排名。

| 方法 | 发表 | 搜索策略 | 评估指标 | 可解释性 | 领域知识 | 平均排名（↓越小越好） |
|------|------|---------|---------|---------|---------|------|
| CAAFE | NeurIPS'23 | 贪心迭代 | ROC AUC | 中 | 仅数据集描述 | 3.82 |
| OCTree | NeurIPS'24 | 决策树引导 | 错误率 | 强 | 列名语义 | ~2.5 |
| **LLM-FE** | ICLR'26 WS | **岛屿进化** | Accuracy | 中 | 特征名+描述 | **1.54** |
| FeatLLM | arXiv'24 | 集成bagging | 错误率 | 强 | 任务描述+样本 | 3.82 |
| **MedFeat** | arXiv'26 | **SHAP引导** | AUC+F1 | **最强** | SHAP+模型感知 | N/A（不同数据集） |
| CoFEH | ASPLOS'26 | MCTS+BO | 错误率 | 中 | prompt+记忆 | **1.75** |
| **本方案** | — | **岛屿进化+SHAP** | **PR-AUC** | **强** | **专家知识+200+识别点** | — |

---

## 三、输入文件与数据准备

### 3.1 文件清单（全部已入库 `inputs/`）

| 文件 | 路径 | 内容 | 在系统中的用途 |
|------|------|------|-------------|
| 开户模拟器 | `inputs/SimECNY/open/` | 生成 100K 钱包数据的 Python 代码 | 运行生成 accounts.csv |
| 交易模拟器 | `inputs/SimECNY/transaction/` | 生成交易流水的 Python 代码 | 运行生成 transactions.csv |
| 销户模拟器 | `inputs/SimECNY/close/` | 生成销户数据的 Python 代码 | 运行生成 close.csv |
| 度分布文件 | `inputs/SimECNY/transaction/wallet_static_edges_in_out_freq.csv` | 交易网络的度分布 | 生成交易图结构 |
| 邮储专家访谈 | `inputs/domain_knowledge/邮储专家访谈_数币洗钱模式.docx` | 4 类洗钱模式的专家描述 | 结构化后注入 System Prompt |
| 可疑交易识别点 | `inputs/domain_knowledge/可疑交易识别点_DATA_SOURCE.pdf` | 12 类 200+ 识别点 | 压缩后注入 System Prompt |
| 风险分类表 | `inputs/domain_knowledge/风险分类表_38指标.xlsx` | 38 个风险指标 | 完整写入 System Prompt |
| 检测规则 | `inputs/domain_knowledge/检测规则_15条.xlsx` | 15 条种子规则 | 转为 Python 示例用于冷启动 |
| 人工特征基线 | `inputs/baseline/feature_engineering.py` | 40+ 人工特征代码 | 提取特征清单告诉 LLM |
| 失败规则 | `inputs/baseline/failed_rules_458.json` | 458 条失败规则 | 反面教材写入约束 |
| 背景论文 | `inputs/domain_knowledge/数字人民币交易模拟论文.pdf` | 研究背景 | 参考（不注入 prompt） |

### 3.2 数据生成流程

SimECNY 是模拟器代码，需要先运行生成实际 CSV 数据：

```
Step 1: 运行开户模块
  cd inputs/SimECNY/open/
  python3 open_wallet_distribution_based.py
  → 输出: accounts.csv（100,000 行，16 列，~5% is_abnormal=True）

Step 2: 运行图生成
  cd inputs/SimECNY/transaction/
  python3 generate_static_edges.py
  → 输入: accounts.csv + wallet_static_edges_in_out_freq.csv
  → 输出: wallet_static_edges_distribution_based.csv（交易网络边表）

Step 3: 运行交易生成
  cd inputs/SimECNY/transaction/
  python3 simplified_main.py
  → 输入: accounts.csv + 边表 + transaction_config.json
  → 输出: wallet_temporal_transactions_*.csv（交易流水，~20列）

Step 4: 修复并运行销户模块
  需要修改 inputs/SimECNY/close/generate_wallet_close_data.py：
    第9行: '/home/lwy/.../accounts.csv' → '../open/accounts.csv'（相对路径）
    第10行: '/home/lwy/.../transactions.csv' → '../transaction/wallet_temporal_transactions_*.csv'
  python3 generate_wallet_close_data.py
  → 输出: new_close_3.csv（销户记录，6列）

Step 5: 验证数据完整性
  检查: 字段名是否与 §1.3 一致
  检查: is_risk 的正样本比例（预期 ~5%）
  检查: wallet_id 在三表间的关联完整性
```

依赖库：`pandas`, `numpy`, `networkx`（交易生成需要）

### 3.3 领域知识结构化

领域知识需要压缩后注入 LLM prompt。压缩策略：

| 来源 | 原始量 | 压缩方式 | 压缩后 token |
|------|--------|---------|------------|
| 专家访谈 | ~3000 字 | 按4类洗钱模式提取关键特征描述 | ~800 |
| 200+ 识别点 | ~10000 字 | 按 12 类分组，每类保留 top-3 最相关的识别点 | ~1000 |
| 38 个指标 | ~1500 字 | 保持表格形式，字段名+说明 | ~600 |
| 15 条规则 | ~500 字 | 转为 Python 函数示例（冷启动用） | ~400 |

**结构化示例**（专家访谈压缩后）：

```
## 数字人民币洗钱模式（来自邮储银行专家访谈）

### 模式1：薅羊毛
特征：批量注册 4 类钱包，同一身份证/手机号关联多个钱包，利用红包/优惠活动

### 模式2：线下跑分
特征：个人钱包收到大额转入后立即转出，交易对手固定，金额接近限额

### 模式3：线上跑分
特征：凌晨高频交易(20时-1时)，分散转出(fan_out)模式，金额10-1000元

### 模式4：商户收款码跑分
特征：商户钱包大量收款后转至个人钱包，many_to_many模式，佣金约0.5-1%
```

### 3.4 三表关联模块的具体实现

LLM 生成的特征函数支持两种返回粒度，系统自动处理：

```python
def merge_feature_to_transactions(feature_df, tx_df):
    """将 LLM 生成的特征合并到交易表"""
    
    if 'tx_id' in feature_df.columns:
        # 交易级特征：直接 merge
        return tx_df.merge(feature_df, on='tx_id', how='left')
    
    elif 'wallet_id' in feature_df.columns:
        # 钱包级特征：分别 merge 到 src 和 dst
        feature_name = [c for c in feature_df.columns if c != 'wallet_id'][0]
        
        # src 侧
        src_df = feature_df.rename(columns={
            'wallet_id': 'src', 
            feature_name: f'{feature_name}_src'
        })
        tx_df = tx_df.merge(src_df, on='src', how='left')
        
        # dst 侧
        dst_df = feature_df.rename(columns={
            'wallet_id': 'dst',
            feature_name: f'{feature_name}_dst'
        })
        tx_df = tx_df.merge(dst_df, on='dst', how='left')
        
        return tx_df
```

---

## 四、系统架构详细设计

### 4.1 总体流程

```
====================================================================
  Phase 0: 输入准备（一次性）
====================================================================
  1. 运行 SimECNY 生成三表 CSV 数据
  2. 从数据中提取 schema + 统计摘要（不发原始数据给 LLM）
  3. 结构化领域知识（专家访谈/识别点/指标 -> 压缩文本）
  4. 提取人工基线特征清单（从 feature_engineering.py）
  5. 转换 15 条种子规则为 Python 函数示例
  6. 组装 System Prompt
====================================================================
            |
            v
====================================================================
  Phase 1-2: 进化搜索循环（核心）
====================================================================

  FOR round = 1 to T (最多 10 轮):

  --- Step 1: 生成候选特征 ---
      3 个岛屿各自:
      - Boltzmann 采样选 k=2 个 in-context 示例
      - 构建 User Prompt（SHAP反馈 + 示例 + 失败记录）
      - 调用 LLM 生成 b=3 个候选特征
      -> 本轮共 3x3 = 9 个候选特征
            |
            v
  --- Step 2: 代码执行与安全检查 ---
      对每个候选:
      - 安全白名单检查（禁止 os/sys/网络等）
      - 在沙盒中执行（超时 30s, 内存 2GB）
      - 检查输出: 非空、非全0、非全1、无 NaN 超过 50%
      - 失败的: 错误信息反馈给 LLM 重试（最多 2 次）
      - 通过的: merge 到交易表（自动处理 wallet/tx 级别）
            |
            v
  --- Step 3: 特征质量评估 ---
      对每个执行成功的候选特征:

      L2 统计验证:
      - 互信息 MI(feature, label) > 0.01
      - 与每个已有特征的 |corr| < 0.9（非冗余）
      - Cramer's V > 0.05（对分类特征）

      L3 模型验证:
      - 加入特征集, 训练 XGBoost, 5折分层CV
      - 计算 deltaPR-AUC = PR-AUC_new - PR-AUC_baseline
      - 接受条件: deltaPR-AUC >= 0.005

      SHAP 分析:
      - 计算新特征在完整模型中的 SHAP 重要性排名

      综合打分:
      score = 0.35*SHAP_rank + 0.25*MI + 0.25*(1-冗余度) + 0.15*deltaPR-AUC
            |
            v
  --- Step 4: 记忆更新与反馈 ---
      接受的特征（score > 阈值 且 deltaPR-AUC >= 0.005）:
      - 加入对应岛屿的成功记忆库
      - 更新全局特征集和 baseline PR-AUC

      拒绝的特征:
      - 记录失败原因（执行错误/无区分度/冗余/无增量）
      - 加入失败记忆库（下轮告诉 LLM 避免）

      岛屿间迁移:
      - 本轮最高分的特征复制到所有岛屿的记忆库

      SHAP 反馈（给下轮 LLM）:
      - 当前 SHAP top-10 特征及重要性
      - 弱势维度提示（哪些方向还需探索）
            |
            v
  收敛判断: 连续 2 轮无新特征通过接受条件 -> 停止

====================================================================
            |
            v
====================================================================
  Phase 3: 最终验证与产出
====================================================================
  1. L4 稳定性验证: 10 次 bootstrap, Nogueira 稳定性 > 0.8
  2. 增量贡献实验: 5 组对比实验（详见 6.4）
  3. 生成特征字典（每个特征: 代码+解释+SHAP排名+统计指标）
====================================================================
```

### 4.2 LLM 输出格式

每个候选特征，LLM 需要输出以下 JSON 结构：

```json
{
    "feature_name": "night_fan_out_ratio",
    
    "granularity": "wallet",
    
    "rationale": "邮储专家指出线上跑分常在凌晨进行，且呈现'分散转出'模式。夜间fan_out交易的占比可以捕捉这种特定洗钱行为。该特征组合了时间维度(night)和结构维度(fan_out)，是已有特征(Timestamp_Hour和Out_Degree)无法直接表达的交叉模式。",
    
    "knowledge_source": "邮储专家访谈-线上跑分模式 + 可疑交易识别点1-1",
    
    "code": "def compute_feature(open_df, tx_df, close_df):\n    tx = tx_df.copy()\n    tx['is_night'] = tx['hour'].isin([22,23,0,1,2,3,4,5])\n    tx['is_fan_out'] = (tx['transaction_mode'] == 'fan_out')\n    tx['is_night_fan_out'] = tx['is_night'] & tx['is_fan_out']\n    stats = tx.groupby('src').agg(\n        total_tx=('tx_id', 'count'),\n        night_fan_out_count=('is_night_fan_out', 'sum')\n    ).reset_index()\n    stats['night_fan_out_ratio'] = stats['night_fan_out_count'] / stats['total_tx']\n    return stats[['src', 'night_fan_out_ratio']].rename(columns={'src': 'wallet_id'})"
}
```

字段说明：
- `feature_name`：LLM 命名的特征名，需具有描述性
- `granularity`：`"wallet"` 或 `"transaction"`，决定系统如何 merge
- `rationale`：可解释性说明，必须引用具体的领域知识
- `knowledge_source`：引用的领域知识来源（专家访谈/识别点/风险指标的具体条目）
- `code`：可直接执行的 Python 函数，签名固定为 `compute_feature(open_df, tx_df, close_df)`

### 4.3 Prompt 详细设计

#### System Prompt（固定，~3900 token）

```
你是数字人民币反洗钱领域的特征工程师。你的任务是为反洗钱检测模型生成新特征。

## 输出要求
每个特征必须以 JSON 格式输出，包含以下字段：
- feature_name: 特征名称（英文，有描述性）
- granularity: "wallet" 或 "transaction"
- rationale: 为什么创建这个特征（必须引用下方的领域知识）
- knowledge_source: 引用的具体领域知识条目
- code: Python 函数，签名为 def compute_feature(open_df, tx_df, close_df)

函数必须返回 DataFrame，包含：
  - wallet 级特征：[wallet_id, feature_value] 两列
  - transaction 级特征：[tx_id, feature_value] 两列

## 数据结构

### 开户表 open_df（每行一个钱包，16字段）
- wallet_id (string): 14位钱包ID
- wallet_type (int): 1=企业, 2=个人, 3=商户
- wallet_level (int): 1-4类
- wallet_open_cert (string): 身份证号，4类为NULL
- wallet_open_cert_expire (string): 证件过期日期
- wallet_open_date (string): 开户日期 YYYY-MM-DD
- wallet_open_timestamp (string): 开户时间 YYYY-MM-DD HH:MM:SS
- wallet_open_tel (string): 手机号
- realname_auth_time (string): 实名认证时间
- accociated_bank_account (string): 关联银行卡号
- bank_account_number (string): 银行编号前4位
- init_balance (float): 初始余额
- region_code (string): 6位行政区划码
- is_abnormal (bool): 是否异常（仅用于训练集标签，不能作为特征使用）
- open_device (string): MAC地址
- open_ip (string): IPv4地址

### 交易表 tx_df（每行一笔交易，20字段）
- tx_id (string): 交易ID
- timestamp (string): 交易时间 YYYY-MM-DD HH:MM:SS
- src (string): 发送方 wallet_id
- dst (string): 接收方 wallet_id
- amount (float): 金额（元）
- transaction_motif (string): 动机类型，7种: single_transaction, normal_small_high_freq, regular_large_low_freq, small_amount_testing, merchant_laundering, class4_laundering, online_laundering
- motif_id (string): 动机组ID
- transaction_mode (string): 结构模式，7种: single, forward, fan_out, fan_in, gather_scatter, scatter_gather, many_to_many
- is_risk (string): 风险标签 0/1（不能作为特征使用）
- is_src_victim (int): 发送方是否受害者（不能作为特征使用）
- src/dst_bank_account_number (string): 银行编号
- src/dst_wallet_level (int): 钱包等级
- src/dst_device (string): 设备MAC
- src/dst_ip (string): IP地址
- interval (float): 时间间隔（秒）
- hour (int): 交易小时 0-23

### 销户表 close_df（每行一次销户，6字段）
- wallet_id (string): 钱包ID
- zs_id (string): 销户流水号
- zs_channel (string): 销户渠道 mobile_app/web/counter
- zs_timestamp (string): 销户时间
- wallet_open_tel (string): 手机号
- is_zs_laundering (int): 是否洗钱销户（不能作为特征使用）

### 三表关联
- 交易表的 src 和 dst 字段是 wallet_id，关联到开户表和销户表
- 一个钱包可以在多笔交易中出现

## 领域知识

### 数字人民币洗钱模式（邮储银行专家访谈）
1. 薅羊毛：批量注册4类钱包，同身份证/手机号关联多钱包，利用红包活动
2. 线下跑分：个人钱包收大额后立即转出，交易对手固定，金额接近限额
3. 线上跑分：凌晨高频交易(20时-1时)，分散转出(fan_out)，金额10-1000元
4. 商户收款码跑分：商户钱包大量收款后转个人钱包，many_to_many模式，佣金0.5-1%

### 风险指标（38个，按维度摘要）
{此处放入 38 个指标的压缩表格}

### 可疑交易识别点（12类，每类 top-3）
{此处放入压缩后的 36 条核心识别点}

## 已有特征（不要重复）
以下 40+ 特征已经由人工实现：
- 交易对统计: same_transactionaccount, transaction_count_directed/Undirected, timeinterval, density
- 集中度: send/receive_concentration, send/receive_amount_concentration
- 1H频率: counts_1H_From/To, 1H_frequency_From/To
- 图度: Out/In_Degree, From_To_Ratio, unique_outdegree/indegree
- 金额: Total/Max/Min/Mean/Std Sent/Received, volatility, z_score, large_ratio
- 时间: Timestamp_Hour, activetime, transaction_frequency, inactive_period
- S指标: S_history_count, S_geo_risk, S_time_risk
- 开户: open_hour, is_night_hour, auth_delay_days, wallets_per_cert

## 约束
1. 代码必须在 30 秒内执行完毕
2. 只能使用 pandas, numpy, scipy.stats, sklearn.preprocessing
3. 不要引用数据中不存在的字段
4. 不要使用标签列（is_risk, is_abnormal, is_zs_laundering, is_src_victim）作为特征
5. 特征值应有区分度（不能全0或全1，NaN不超过50%）
6. 只生成 Python 代码，不要生成一阶逻辑表达式
```

#### User Prompt（每轮动态，~1500 token）

```
第 {N} 轮迭代

## 上轮最佳特征（记忆库中的高分示例）
{按 score 排序的 top-3 特征，每个包含: 名称, 代码, SHAP排名, 互信息, score}

## 上轮 SHAP 分析
当前模型最重要的 10 个特征：
{特征名: SHAP重要性（归一化）}

以下维度还很欠缺（SHAP 重要性低或无覆盖）：
{列出 SHAP 分析发现的弱势维度}

## 上轮失败的特征（不要重复类似的）
{特征名: 失败原因}

## 请生成 3 个新特征
要求每个特征探索不同的维度。输出 JSON 数组格式。
```

#### 冷启动（第 1 轮的 User Prompt）

```
第 1 轮迭代（首轮）

## 种子示例
以下是 2 个已有的检测规则转换为 Python 代码的示例，展示期望的代码风格：

示例1:
{从 15 条种子规则中选择的 Python 函数示例}

示例2:
{另一个示例}

## 请生成 3 个新特征
参考上面的领域知识（尤其是4类洗钱模式），从你认为最有价值的角度出发。
要求每个特征探索不同的维度。输出 JSON 数组格式。
```

### 4.4 SHAP 与岛屿进化的融合细节

SHAP 引导和 Boltzmann 采样在不同层面起作用：

```
每一轮迭代的执行流程：

1. SHAP 分析（全局，所有岛屿共享）
   model = XGBoost(当前全部特征)
   shap_values = TreeExplainer(model).shap_values(X_val)
   shap_feedback = 生成 top-10 特征 + 弱势维度文本

2. 对每个岛屿 j ∈ {1, 2, 3}：
   
   a) Boltzmann 采样选 in-context 示例（岛屿独立）
      对 Island_j.memory 中每个特征 i：
          P_i = exp(score_i / τ) / Σ exp(score_all / τ)
      示例 = 从 P_i 分布中采样 k=2 个特征
   
   b) 构建 User Prompt
      = shap_feedback（全局共享）
      + 示例（岛屿独立的 Boltzmann 采样结果）
      + 失败记忆（全局共享）
   
   c) 调用 LLM 生成 b=3 个候选

3. 岛屿间迁移
   best_feature = 本轮所有 9 个候选中得分最高的
   FOR each Island_j:
       IF best_feature 不在 Island_j.memory 中:
           加入 Island_j.memory
```

**为什么这样设计**：
- SHAP 全局共享：所有岛屿都知道"哪个方向有价值"，避免浪费搜索资源
- Boltzmann 岛屿独立：各岛屿看到不同的 in-context 示例，保持多样性
- 迁移最优：好特征传播到所有岛屿，提高整体搜索质量

### 4.5 LLM 模块（可插拔）

LLM 作为可插拔模块，接口定义：

```python
class LLMInterface:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """输入 prompt，输出包含 JSON 的文本"""
        raise NotImplementedError

class OpenAILLM(LLMInterface):
    """API 模式：GPT-4o / GPT-3.5-Turbo"""
    
class LocalLLM(LLMInterface):
    """本地模式：Qwen2.5-Coder-32B / Llama-3.1-8B 等"""
```

LLM-FE 论文验证了 Llama-3.1-8B 和 GPT-3.5-Turbo 均有效，框架对底层模型不敏感。搜索智能来自进化框架（Boltzmann 采样 + SHAP 引导），不是来自 LLM 本身。具体模型根据实际可用资源确定。

### 4.6 Prompt 长度管理

| 层级 | 内容 | token 估算 | 说明 |
|------|------|-----------|------|
| **必须** | Schema + 三表关联 + 约束 + 输出格式 | ~1500 | 不可压缩 |
| **核心** | 4 类洗钱模式（专家访谈压缩版） | ~800 | 4 段，每段 ~200 token |
| **重要** | 38 个风险指标（表格） | ~600 | 压缩为 名称+说明 两列 |
| **补充** | 36 条核心识别点（每类 top-3） | ~1000 | 可根据 context 长度调整数量 |
| System 小计 | | ~3900 | |
| **动态** | SHAP反馈 + 示例 + 失败记录 | ~1500 | User Prompt |
| **总计** | | **~5400** | 8K context 的模型可容纳 |

如果模型 context ≥ 32K（如 GPT-4o），可以放入更多识别点原文和更多 in-context 示例。

---

## 五、现状分析

### 5.1 人工特征工程基线（feature_engineering.py，~40 个特征）

| 类别 | 特征 | 已覆盖 |
|------|------|--------|
| 交易对 | same_transactionaccount, transaction_count_directed/Undirected, timeinterval, density | 是 |
| 集中度 | send/receive_concentration, send/receive_amount_concentration | 是 |
| 短期频率 | counts_1H_From/To, 1H_frequency_From/To | 是（仅1H窗口） |
| 图度 | Out/In_Degree, From_To_Ratio, unique_outdegree/indegree | 是（仅简单度） |
| 金额统计 | Total/Max/Min/Mean/Std Sent/Received, volatility, z_score, large_ratio | 是 |
| 时间 | Timestamp_Hour, activetime, transaction_frequency, inactive_period | 是 |
| S指标 | S_history_count, S_geo_risk, S_time_risk | 是 |
| 开户特征 | open_hour, is_night_hour, auth_delay_days, wallets_per_cert 等 | 是 |

### 5.2 人工特征的关键缺口

| 缺口 | 具体说明 | 涉及的未用字段 |
|------|---------|-------------|
| **transaction_motif/mode 完全未用** | 7种交易动机 + 7种结构模式，是区分正常/异常交易的直接信号 | transaction_motif, transaction_mode |
| **设备/IP 交易级特征未用** | 同设备多钱包、IP 跳变等是洗钱强信号 | src/dst_device, src/dst_ip |
| **图拓扑高阶特征缺失** | 无中心性、环路检测、聚集系数、PageRank | 需从 src/dst 构建图计算 |
| **销户模块完全未实现** | 快速销户是洗钱关键信号（模拟器编程了 60% 快速销户模式） | close_df 全部字段 |
| **领域知识特征缺失** | 无整数金额检测、限额利用率、跨行交易模式 | amount, wallet_level |
| **行为序列模式缺失** | 无沉睡-爆发、小额试探后大额转移等时序模式 | timestamp, amount 时序 |
| **时间窗口单一** | 仅 1H 窗口，无日/周/月维度 | timestamp |
| **对手风险传播缺失** | 不知道交易对手的风险水平 | 需跨交易聚合 |

**重要**：以上缺口是我们自己分析的，**不告诉 LLM**。我们通过领域知识注入和 SHAP 引导让 LLM 自己发现这些方向。如果 LLM 发现了我们没预料到的方向，那才是 LLM 特征工程的真正价值。

### 5.3 之前 458 条规则的失败教训

失败的根本原因和本方案的应对：

| 失败原因 | 详细说明 | 本方案如何避免 |
|---------|---------|-------------|
| 引用不存在的字段 | 如 `daily_limit_breach`，数据中无此字段 | Prompt 中列出全部实际字段名和类型 |
| 输出格式无法执行 | 一阶逻辑表达式，不是代码 | 要求输出 Python 函数，立即在数据上执行验证 |
| 无执行验证 | 规则从未在数据上运行过 | 每个候选特征都在沙盒中执行，失败则反馈修正 |

---

## 六、特征有效性验证方法

> 详细分析见 [VALIDATION_METHODS.md](./VALIDATION_METHODS.md)

### 6.1 四级验证框架

每个 LLM 生成的候选特征经过四级验证，逐级过滤。只有通过全部四级的特征才进入最终特征集。

#### L1: 代码验证（来自 CAAFE §2.3）

| 检查项 | 通过条件 | 失败处理 |
|--------|---------|---------|
| 安全白名单 | 仅使用 pandas/numpy/scipy/sklearn | 直接拒绝 |
| 执行成功 | 30s 内无异常退出 | 错误信息反馈 LLM 重试（最多2次） |
| 输出格式 | 返回 DataFrame，含 wallet_id 或 tx_id 列 | 反馈格式要求 |
| 数据质量 | 非全 0、非全 1、NaN < 50% | 拒绝并记录 |

#### L2: 统计验证（来自 Balek §2.6）

```python
# 互信息检验
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(feature.values.reshape(-1,1), label, random_state=42)[0]
assert mi > 0.01, "特征与标签的互信息过低"

# 冗余度检验（与所有已有特征）
for existing in existing_features:
    corr = abs(feature.corr(existing))
    assert corr < 0.9, f"与已有特征 {existing.name} 相关系数 {corr:.3f} 过高"

# Cramer's V（对分类特征）
from scipy.stats import chi2_contingency
contingency = pd.crosstab(feature_binned, label)
chi2, p, dof, _ = chi2_contingency(contingency)
V = np.sqrt(chi2 / (len(feature) * (min(contingency.shape) - 1)))
assert V > 0.05, "Cramer's V 过低"

# Bootstrap 卡方检验（2500次）
p_values = []
for _ in range(2500):
    idx = np.random.choice(len(feature), size=len(feature), replace=True)
    ct = pd.crosstab(feature_binned.iloc[idx], label.iloc[idx])
    _, p, _, _ = chi2_contingency(ct)
    p_values.append(p)
assert np.mean(p_values) < 0.05, "Bootstrap卡方检验不显著"
```

#### L3: 模型验证（来自 MedFeat §2.2）

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import xgboost as xgb

# 基线 PR-AUC（不含新特征）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_scores = []
for train_idx, val_idx in cv.split(X_existing, y):
    model = xgb.XGBClassifier(scale_pos_weight=19, eval_metric='aucpr')
    model.fit(X_existing.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict_proba(X_existing.iloc[val_idx])[:, 1]
    baseline_scores.append(average_precision_score(y.iloc[val_idx], pred))
baseline_pr_auc = np.mean(baseline_scores)

# 增量 PR-AUC（含新特征）
X_augmented = pd.concat([X_existing, new_feature], axis=1)
augmented_scores = []
for train_idx, val_idx in cv.split(X_augmented, y):
    model = xgb.XGBClassifier(scale_pos_weight=19, eval_metric='aucpr')
    model.fit(X_augmented.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict_proba(X_augmented.iloc[val_idx])[:, 1]
    augmented_scores.append(average_precision_score(y.iloc[val_idx], pred))
augmented_pr_auc = np.mean(augmented_scores)

delta = augmented_pr_auc - baseline_pr_auc
assert delta >= 0.005, f"PR-AUC 增量 {delta:.4f} 不足"

# XGBoost 参数说明：
# scale_pos_weight=19: 因为正样本~5%，负:正 ≈ 19:1
# eval_metric='aucpr': 用 PR-AUC 作为评估指标
```

#### L4: 稳定性验证（通用最佳实践）

```python
# 在最终特征集确定后执行（不在搜索循环中，成本太高）

from sklearn.utils import resample

stability_results = []
for _ in range(10):
    X_boot, y_boot = resample(X_final, y, stratify=y, random_state=None)
    model = xgb.XGBClassifier(scale_pos_weight=19)
    model.fit(X_boot, y_boot)
    shap_values = shap.TreeExplainer(model).shap_values(X_boot)
    importance_rank = np.argsort(-np.mean(np.abs(shap_values), axis=0))
    stability_results.append(importance_rank[:20])  # top-20 特征排名

# Nogueira 稳定性指数
# 衡量 10 次 bootstrap 中 top-20 特征排名的一致性
stability_index = compute_nogueira_stability(stability_results)
assert stability_index > 0.8, "特征选择不够稳定"
```

### 6.2 为什么用 PR-AUC 而非 ROC-AUC

AML 数据正样本 ~5%，ROC-AUC 会过度乐观：

```
ROC-AUC 高 + PR-AUC 低 → 善于排除正常交易，但不擅长找出可疑交易
ROC-AUC 高 + PR-AUC 高 → 模型真正有效
不平衡越严重，两者差距越大
```

五篇参考论文的评估指标及其问题：

| 论文 | 评估指标 | 对 AML 场景的问题 |
|------|---------|----------------|
| LLM-FE | Accuracy | 全预测负可达 95%，完全无用 |
| CAAFE | ROC AUC | 过度乐观，FPR 分母太大 |
| MedFeat | AUC + F1 | AUC 同上，F1 依赖阈值选择 |
| OCTree | 错误率 | 同 Accuracy 问题 |
| CoFEH | 错误率 | 同 Accuracy 问题 |

### 6.3 最终增量贡献实验设计

在全部搜索完成后，运行 5 组对比实验：

```
实验 1（基线）：    40个人工特征 → XGBoost → PR-AUC_baseline
实验 2（增量）：    40个人工 + N个LLM特征 → XGBoost → PR-AUC_augmented  
实验 3（独立）：    仅N个LLM特征 → XGBoost → PR-AUC_llm_only
实验 4（随机对照）：40个人工 + N个随机特征 → XGBoost → PR-AUC_random
实验 5（消融）：    逐个移除LLM特征，观察PR-AUC变化 → 识别最关键的特征

报告指标：
- 增量 PR-AUC = PR-AUC_augmented - PR-AUC_baseline（应 > 0）
- 独立 PR-AUC = PR-AUC_llm_only（LLM 特征独立的预测能力）
- 随机对照 = PR-AUC_random - PR-AUC_baseline（应 ≈ 0，排除过拟合）
- 各 LLM 特征的 SHAP 排名（在完整模型中）
- 每个 LLM 特征的增量贡献（消融实验）

所有实验使用 5 折分层 CV，报告均值 ± 标准差
```

### 6.4 与我们已有验证方法的对比

后续提供已有验证方法后，进行对比分析，取长补短。

---

## 七、实现路线图

### Phase 0：数据生成（0.5 天）

- [ ] 安装依赖：`pip install pandas numpy networkx`
- [ ] 运行开户模块 → accounts.csv
- [ ] 运行图生成 → 交易网络边文件
- [ ] 运行交易生成 → 交易流水 CSV
- [ ] 修复销户模块硬编码路径，运行 → 销户 CSV
- [ ] 验证：字段名一致、is_risk ~5%、三表 wallet_id 关联完整

### Phase 1：基础设施搭建（2 天）

- [ ] 数据加载模块（读取三表，生成 schema + 统计摘要）
- [ ] 三表关联模块（wallet/tx 级特征自动 merge，见 §3.4 代码）
- [ ] 领域知识结构化（分层压缩，见 §3.3）
- [ ] 种子规则转换（15 条检测规则 → Python 函数示例）
- [ ] System Prompt 组装（见 §4.3 模板）
- [ ] 特征执行沙盒（安全白名单 + 超时 + 内存限制 + 错误捕获）
- [ ] 特征评估模块（MI + 冗余度 + Cramer's V + PR-AUC + SHAP）
- [ ] LLM 调用接口（可插拔，见 §4.5）

### Phase 2：核心搜索循环（3 天）

- [ ] 岛屿数据结构（3 个独立记忆库 + 全局失败记忆）
- [ ] Boltzmann 采样（从记忆库选 in-context 示例）
- [ ] User Prompt 动态生成（SHAP 反馈 + 示例 + 失败记录）
- [ ] 搜索循环主程序（3 岛屿 × 3 候选/轮 × ~10 轮）
- [ ] 冷启动处理（第 1 轮用种子规则示例）
- [ ] 岛屿间迁移（最优特征广播）
- [ ] 收敛判断（连续 2 轮无新特征通过 → 停止）

### Phase 3：验证与产出（2 天）

- [ ] L4 稳定性验证（10 次 bootstrap）
- [ ] 5 组增量贡献实验
- [ ] 特征字典生成（每个特征：代码 + 解释 + SHAP + 统计指标）
- [ ] 结果报告和可视化

---

## 八、预期产出

1. **自动发现的特征集**：经过四级验证的高质量特征（预计 15-30 个通过全部验证）
2. **特征字典**：每个特征包含 Python 代码 + 可解释性说明 + 领域知识来源 + SHAP 排名 + 统计指标（MI, Cramer's V）
3. **增量贡献报告**：LLM 特征 vs 人工特征的 PR-AUC / F1 / Recall@Precision 对比
4. **验证报告**：四级验证结果 + 5 组实验数据 + 稳定性分析
5. **可复用框架**：LLM 自动特征工程 pipeline 代码（可插拔 LLM，可换数据集）

---

## 九、核心创新点（论文贡献）

1. **领域知识增强的 LLM 特征工程**：首次将结构化专家知识（邮储访谈 + 200+ 识别点 + 监管规则）系统性注入 LLM 特征生成流程。现有方法（LLM-FE 仅靠特征名、MedFeat 无领域知识、CAAFE 仅靠数据集描述）都缺乏深度领域知识注入。
2. **多表关联特征发现**：首次将 LLM 特征工程扩展到多表关系型数据（开户 + 交易 + 销户三表关联）。所有 6 篇参考论文均仅支持单表输入。
3. **进化搜索 + SHAP 引导融合**：结合 LLM-FE 的多岛屿搜索效率和 MedFeat 的 SHAP 方向引导，在已有 40+ 人工特征基线上发现增量贡献。
4. **四级特征验证框架**：代码验证 → 统计验证 → 模型验证 → 稳定性验证，超越现有论文（无一使用统计检验）的验证深度。
5. **金融领域可解释特征**：每个 LLM 生成的特征附带领域依据说明和知识来源引用，满足金融监管的可解释性要求。

---

## 参考文献

1. Abhyankar et al. "LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers." ICLR 2026 Workshop. arXiv:2503.14434
2. "MedFeat: Model-Aware and Explainability-Driven Feature Engineering with LLMs." arXiv:2603.02221, 2026
3. Hollmann et al. "Large Language Models for Automated Data Science: Introducing CAAFE." NeurIPS 2023. arXiv:2305.03403
4. Nam et al. "OCTree: Optimized Feature Generation via LLMs with Decision Tree Reasoning." NeurIPS 2024. arXiv:2406.08527
5. Xu et al. "CoFEH: LLM-driven Feature Engineering Empowered by Collaborative Bayesian HPO." ASPLOS 2026. arXiv:2602.09851
6. Balek et al. "LLM-based Feature Generation from Text for Interpretable ML." Machine Learning, 2025. arXiv:2409.07132
