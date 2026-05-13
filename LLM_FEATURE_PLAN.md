# LLM 特征生成方案：数字人民币反洗钱

> 目标：利用 LLM 在 SimECNY 现有特征基础上生成更多**有效、可解释**的特征

---

## 一、现状分析

### 1.1 现有数据结构（SimECNY）

**开户模块（16 字段）**：wallet_id, wallet_type(1企业/2个人/3商户), wallet_level(1-4类), wallet_open_cert(身份证), wallet_open_cert_expire, wallet_open_date, wallet_open_timestamp, wallet_open_tel, realname_auth_time, accociated_bank_account, bank_account_number, init_balance, region_code, is_abnormal, open_device(MAC), open_ip

**交易模块（20 字段）**：tx_id, timestamp, src, dst, amount, transaction_motif, motif_id, transaction_mode, is_risk, is_src_victim, src/dst_bank_account_number, src/dst_wallet_level, src/dst_device, src/dst_ip, interval, hour

**销户模块（6 字段）**：wallet_id, zs_id, zs_channel, zs_timestamp, wallet_open_tel, is_zs_laundering

### 1.2 人工特征工程已覆盖的特征（feature_engineering.py）

现有人工特征工程（`feature_engineering.py`）已实现约 **40+ 个特征**，分类如下：

**A. 交易对特征（Expanding Window）**：
- `same_transactionaccount`（自交易标记）
- `transaction_count_directed` / `transaction_count_Undirected`（账户对交易次数）
- `accounttransaction_timeinterval`（账户对首尾交易间隔）
- `transaction_density_Undirected`（交易密度 = 次数/间隔）

**B. 交易集中度特征**：
- `send_concentration` / `receive_concentration`（按次数的交易集中度）
- `send_amount_concentration` / `receive_amount_concentration`（按金额的交易集中度）

**C. 短期频率特征（Rolling Window）**：
- `counts_1H_From` / `counts_1H_To`（1小时内交易次数）
- `1H_frequency_From` / `1H_frequency_To`（1小时频率 / 总频率）

**D. 图度特征**：
- `Out_Degree` / `In_Degree`（出入度，expanding cumcount）
- `From_To_Ratio`（出入度比）
- `unique_outdegree` / `unique_indegree`（唯一交易对手数）
- `unique_From_To_Ratio`

**E. 金额统计特征**：
- `Total/Max/Min/Mean/Std_Sent` + `Total/Max/Min/Mean/Std_Received`
- `volatility_Sent` / `volatility_Received`（变异系数）
- `Receive_Send_Ratio` / `Difference`（收付款比/差额）
- `avg_max_sent_ratio` / `avg_max_receive_ratio`
- `max_sent_receive_ratio` / `volatility_ratio`
- `z_score_paid` / `z_score_received`（Z-score 异常检测）
- `large_transaction_ratio`（大额异常交易占比）

**F. 时间与活跃特征**：
- `Timestamp_Hour`（交易小时）
- `activetime`（活跃时间跨度）
- `transaction_frequency`（交易频率）
- `avg_payment_time_diff` / `avg_receipt_time_diff`（平均付/收款间隔）
- `avg/max/min_inactive_period`（非活跃期统计）
- `last_direction` / `last_amount`（最后交易方向/金额）

**G. 静态S指标**：
- `S_history_count`（同手机号历史钱包数）
- `S_geo_risk_202308`（高风险地区标记）
- `S_time_risk_202308`（夜间开户标记）

**H. 账户开立特征（外部文件）**：
- `open_hour`, `is_work_hour`, `is_night_hour`
- `auth_delay_days`, `wallets_per_cert`, `days_to_expire_from_current`
- `tel_reuse_count`, `is_high_risk_region`, `log_balance`

### 1.3 人工特征工程的关键缺口（LLM 应该填补的空白）

| 缺口类别 | 具体缺失 | 原因 |
|---------|---------|------|
| **交易模式特征** | `transaction_motif`（7种模式）和 `transaction_mode`（7种结构）完全未使用 | 原始代码未利用这两个字段 |
| **设备/IP 关联特征** | 交易级的 `src_device`/`dst_device`/`src_ip`/`dst_ip` 未使用 | 只在 S 指标中用了开户设备 |
| **图拓扑高阶特征** | 无中心性、环路、聚集系数、PageRank、连通分量等 | 仅用了简单度特征 |
| **跨模块生命周期特征** | 销户模块完全未使用 | 代码中 close_path 参数存在但未实现 |
| **领域驱动的组合特征** | 无整数金额检测、限额利用率、跨行模式、毒品价格倍数等 | 纯统计方法，未注入领域知识 |
| **行为序列模式** | 无沉睡-测试-爆发、小额试探后大额等时序模式 | 只有时间间隔统计，无序列模式识别 |
| **多时间窗口** | 仅1小时窗口，无日/周/月维度 | 固定1H窗口 |
| **交易对手风险传播** | 无对手风险评分、邻居异常率等 | 只有自身统计 |
| **金额特殊模式** | 无接近报告阈值、金额聚类、尾数分析等 | 只有基础统计 |

### 1.4 之前 458 条规则生成的问题

1. 458 条一阶逻辑规则**全部无法转换执行**（precision/recall/F1 = 0）
2. **根本原因**：规则引用了数据中不存在的字段（如 `daily_limit_breach`、`high_freq_night_tx` 等）
3. 输出格式为自然语言一阶逻辑表达式，无法映射到实际代码

### 1.5 领域知识来源

| 来源 | 内容 |
|------|------|
| 邮储银行专家访谈 | 4类数币洗钱：薅羊毛、线下跑分、线上跑分、商户收款码跑分 |
| 可疑交易识别点对照表 | 12类可疑交易（地下钱庄、腐败、毒品、走私、诈骗等）200+识别点 |
| 风险分类表 | 38个数币风险指标，4大维度 |
| 大额交易报告管理办法 | 法定报告阈值（个人5万/50万，企业200万） |

---

## 二、方案设计

### 核心思路

借鉴 **Balek 2025**（LLM 直接从数据提取可解释特征值）和 **CAAFE**（LLM 生成特征转换代码）两种方法，设计**两阶段特征生成流程**：

```
阶段一：LLM 基于领域知识 → 设计特征定义（Feature Schema）
阶段二：LLM 生成 Python 代码 → 从原始数据计算特征值
```

### 2.1 阶段一：LLM 领域驱动的特征设计

**输入给 LLM 的 Prompt 包含**：
1. SimECNY 的完整数据 schema（3 个模块所有字段）
2. 领域知识上下文（邮储专家访谈要点、可疑交易识别点、风险分类表）
3. 少量样本数据（正常 + 异常各 10-20 条）
4. 已知的洗钱模式描述

**要求 LLM 输出**：
```json
{
  "feature_name": "night_opening_ratio",
  "description": "钱包在夜间（22:00-06:00）开户的比例",
  "rationale": "邮储专家指出不法分子常在凌晨开立大量四类钱包，夜间开户是典型异常信号",
  "knowledge_source": "邮储银行专家访谈-取现模式",
  "data_fields_used": ["wallet_open_timestamp"],
  "computation_logic": "提取开户时间的小时数，判断是否在22-06区间",
  "expected_risk_correlation": "正相关：夜间开户比例越高越可疑",
  "python_code": "...",
  "feature_type": "wallet_level"  // wallet_level | tx_level | network_level | cross_module
}
```

### 2.2 阶段二：代码生成与验证

LLM 为每个特征生成**可直接执行的 Python 函数**，输入为 DataFrame，输出为新特征列。

**关键约束**（避免重蹈之前 458 条规则全部失败的覆辙）：
1. 只使用数据中**实际存在的字段名**
2. 代码必须通过语法检查和单元测试
3. 每个特征附带计算样例，可验证输出合理性

---

## 三、LLM 增量特征体系（人工特征未覆盖的空白区域）

> 以下特征是人工 `feature_engineering.py` **未实现的**，是 LLM 应该增量贡献的部分。
> 标记 `[NEW]` 表示全新特征，`[ENHANCE]` 表示对已有特征的增强。

### 3.1 交易模式特征（利用 transaction_motif / transaction_mode）

**现状**：人工代码完全未使用 `transaction_motif`（7种）和 `transaction_mode`（7种）这两个关键字段。

| 特征 | 计算方式 | 领域依据 |
|------|---------|---------|
| `[NEW] motif_risk_score` | 交易 motif 映射到风险分数（class4_laundering=1.0, merchant_laundering=0.9, ...） | 邮储：四类取现/商户跑分/线上跑分是主要风险模式 |
| `[NEW] risk_motif_ratio` | 账户的风险 motif 交易占比 | 高风险模式集中度 |
| `[NEW] mode_fan_out_ratio` | fan_out 模式交易占比 | 识别点：分散转出是地下钱庄特征 |
| `[NEW] mode_fan_in_ratio` | fan_in 模式交易占比 | 识别点：集中转入是洗钱特征 |
| `[NEW] motif_diversity` | 使用的 motif 种类数 / 总种类数 | 混合多种模式规避监管 |
| `[NEW] motif_transition_entropy` | 连续交易 motif 变化的信息熵 | 模式跳变频繁说明异常 |

### 3.2 设备/IP 关联特征（交易级）

**现状**：人工代码仅在 S 指标中用了开户设备，交易级的 4 个设备/IP 字段完全未使用。

| 特征 | 计算方式 | 领域依据 |
|------|---------|---------|
| `[NEW] src_dst_same_device` | src_device == dst_device | 同一设备控制收发双方，典型洗钱自操作 |
| `[NEW] src_dst_same_ip` | src_ip == dst_ip | 同一网络下不同钱包互转 |
| `[NEW] device_wallet_count` | 该设备关联的钱包数量 | 邮储：同一设备开大量钱包是薅羊毛特征 |
| `[NEW] ip_wallet_count` | 该 IP 关联的钱包数量 | 邮储：同一IP地址大量钱包 |
| `[NEW] device_change_rate` | 账户使用不同设备交易的频率 | 频繁换设备可能是多人共用账户 |
| `[NEW] src_device_mismatch_open` | 交易设备与开户设备是否不同 | 账户可能被他人使用 |

### 3.3 领域知识驱动的组合特征

**现状**：人工代码是纯统计方法，未注入任何反洗钱领域知识。

| 特征 | 计算方式 | 领域依据 |
|------|---------|---------|
| `[NEW] is_round_amount` | amount 是否为整数或 100 的倍数 | 识别点：赌博/套现以整数交易为主 |
| `[NEW] round_amount_ratio` | 整数金额交易占比 | 识别点：赌博商户"短期内频繁发生大量刷卡交易，以整数交易为主" |
| `[NEW] amount_to_limit_ratio` | amount / 钱包等级对应限额 | 接近限额的交易更可疑，邮储：限额比较低（跑分5000元以下） |
| `[NEW] is_cross_bank` | src_bank_account_number != dst_bank_account_number | 识别点：跨行交易频繁规避单行监控 |
| `[NEW] cross_bank_ratio` | 跨行交易占比 | 地下钱庄：跨银行交易频繁 |
| `[NEW] is_cross_level` | src_wallet_level != dst_wallet_level | 不同等级钱包间转账可能是层级洗钱 |
| `[NEW] amount_drug_multiple` | amount 是否为毒品价格整数倍（100/150/200元） | 识别点：汇款或银行卡单向流入的资金为毒品黑市价格整数倍 |
| `[NEW] is_rapid_transfer` | interval < 60 秒 | 邮储：资金快进快出，网银交易"快进快出，1分钟左右完成" |
| `[NEW] near_report_threshold` | 金额接近大额报告阈值（个人5万/50万） | 识别点：规避金额刻意低于报告标准 |
| `[NEW] night_tx_ratio` | 夜间（22:00-06:00）交易占比 | 识别点：通过ATM交易的时间多为午夜或凌晨 |
| `[NEW] class4_high_utilization` | 四类钱包且日交易金额接近5000元限额 | 邮储：充分利用匿名四类钱包额度进行跑分 |

### 3.4 行为序列模式特征

**现状**：人工代码只有时间间隔统计（mean/max/min），无任何序列模式识别。

| 特征 | 计算方式 | 领域依据 |
|------|---------|---------|
| `[NEW] dormancy_then_burst` | 是否存在 ≥7天无交易后突然日均>10笔 | 邮储："开户后3个月以上无交易，突然启动时先小额测试，随后大额频繁" |
| `[NEW] small_test_before_large` | 大额交易（>1000）前24小时内是否有小额测试（<100） | 邮储："先使用小额交易测试常用业务（如网银转账、ATM提现等）" |
| `[NEW] amount_escalation` | 交易金额是否呈递增趋势（线性回归斜率） | 逐步试探+升级是典型异常行为 |
| `[NEW] burst_frequency` | 是否存在突发高频交易（1小时内>10笔） | 邮储：跑分日交易量达限额上限 |
| `[NEW] periodic_pattern` | 交易间隔的规律性（标准差/均值） | 识别点：传销"资金交易具有一定的周期性（一般以一周为周期）" |
| `[NEW] tx_interval_cv` | 交易间隔变异系数 | 极低CV说明机器人操作（固定间隔） |

### 3.5 图拓扑高阶特征

**现状**：人工代码仅有简单度特征（in/out degree + unique degree），无任何图算法特征。

| 特征 | 计算方式 | 领域依据 |
|------|---------|---------|
| `[NEW] betweenness_centrality` | 介数中心性（NetworkX） | 中转账户在资金流通中的桥接作用 |
| `[NEW] pagerank` | PageRank 分数 | 资金流向重要性，类似"资金热度" |
| `[NEW] is_in_cycle` | 是否处于交易环路中 | 识别点：资金循环，"资金循环呈现深圳公司-异地公司-异地个人-深圳个人" |
| `[NEW] shortest_cycle_length` | 所属最短环路长度 | 短环路（2-3跳）更可疑，直接回流 |
| `[NEW] clustering_coefficient` | 聚集系数 | 密集子图可能是洗钱团伙网络 |
| `[NEW] component_size` | 所属弱连通分量大小 | 大型交易网络更可疑 |
| `[NEW] local_abnormal_ratio` | 1-hop 邻居中异常账户占比 | 近朱者赤：与高风险主体直接关联 |
| `[NEW] shared_counterparty_with_abnormal` | 与已知异常账户共享的交易对手数 | 洗钱网络重叠度 |

### 3.6 跨模块生命周期特征

**现状**：人工代码中 `close_path` 参数存在但销户模块完全未使用。

| 特征 | 计算方式 | 领域依据 |
|------|---------|---------|
| `[NEW] wallet_lifetime_days` | zs_timestamp - wallet_open_timestamp | 短生命周期可疑（跑分钱包用完即弃） |
| `[NEW] pre_close_tx_spike` | 销户前7天交易量 / 历史周均交易量 | 销户前突然活跃 |
| `[NEW] pre_close_amount_spike` | 销户前7天金额 / 历史周均金额 | 销户前突然大额交易 |
| `[NEW] close_channel_mismatch` | 销户渠道(zs_channel)与开户方式是否不一致 | 柜台开户但APP销户可疑 |
| `[NEW] same_phone_close_count` | 同一手机号的历史销户次数 | 邮储：反复开销户是跑分特征 |
| `[NEW] open_to_first_tx_hours` | 开户到首笔交易的时间差 | 邮储："开户后立即转账或汇款" |
| `[NEW] first_tx_is_large` | 首笔交易是否为大额（>P90） | 正常用户通常先小额尝试 |
| `[NEW] balance_at_close` | 销户时余额是否接近0 | 识别点：账户不留余额 |

### 3.7 多时间窗口特征

**现状**：人工代码仅有 1 小时窗口 `counts_1H_From/To`，缺少更长时间维度。

| 特征 | 计算方式 | 领域依据 |
|------|---------|---------|
| `[ENHANCE] counts_24H_From/To` | 24小时内交易次数 | 日维度行为模式 |
| `[ENHANCE] counts_7D_From/To` | 7天内交易次数 | 周维度行为模式 |
| `[NEW] daily_amount_max` | 单日最大交易金额 | 规则：日交易百万为商户异常 |
| `[NEW] daily_tx_count_max` | 单日最大交易笔数 | 规则：>20笔/天为地下钱庄特征 |
| `[NEW] weekly_pattern_stability` | 每周交易量的稳定性（周间方差） | 传销周期性 vs 正常波动 |

---

## 四、LLM 特征生成流程

### 4.1 总体架构

```
┌─────────────────────────────────────────────────────┐
│                   输入准备                            │
│  ┌─────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │数据Schema│  │领域知识库│  │样本数据(正常+异常)│   │
│  └────┬────┘  └────┬─────┘  └────────┬─────────┘   │
│       └────────────┼─────────────────┘              │
│                    ▼                                 │
│  ┌─────────────────────────────────────────┐        │
│  │  Step 1: LLM 特征方案生成               │        │
│  │  - 输入: schema + 领域知识 + 样本        │        │
│  │  - 输出: 特征定义 JSON (名称/解释/依据)  │        │
│  └────────────────┬────────────────────────┘        │
│                   ▼                                  │
│  ┌─────────────────────────────────────────┐        │
│  │  Step 2: LLM 代码生成                   │        │
│  │  - 输入: 特征定义 + 数据Schema           │        │
│  │  - 输出: Python 函数 (可直接执行)        │        │
│  └────────────────┬────────────────────────┘        │
│                   ▼                                  │
│  ┌─────────────────────────────────────────┐        │
│  │  Step 3: 自动验证                        │        │
│  │  - 语法检查                              │        │
│  │  - 在样本数据上试运行                     │        │
│  │  - 输出分布检查 (全0/全1/NaN 比例)       │        │
│  └────────────────┬────────────────────────┘        │
│                   ▼                                  │
│  ┌─────────────────────────────────────────┐        │
│  │  Step 4: 统计验证                        │        │
│  │  - 与标签的相关性检验 (卡方/互信息)      │        │
│  │  - 效应大小 (Cramer's V / 点二列相关)    │        │
│  │  - 特征间冗余检测 (相关系数矩阵)         │        │
│  └────────────────┬────────────────────────┘        │
│                   ▼                                  │
│  ┌─────────────────────────────────────────┐        │
│  │  Step 5: 迭代优化                        │        │
│  │  - 将验证结果反馈给 LLM                  │        │
│  │  - LLM 修正/增强/删除特征                │        │
│  │  - 重复 Step 2-5                         │        │
│  └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
```

### 4.2 Prompt 设计策略

**System Prompt**（固定）：
```
你是一名数字人民币反洗钱领域的资深数据科学家。
你的任务是基于现有数据字段和领域知识，设计新的特征。

关键约束：
1. 只使用以下数据字段（附完整字段列表）
2. 每个特征必须附带：领域依据、计算逻辑、可执行的Python代码
3. 代码输入为 pandas DataFrame，输出为新列
4. 特征值应有区分度（不能全部相同值）
```

**User Prompt**（迭代）：
```
第 N 轮迭代

领域知识上下文：
{邮储专家访谈要点}
{可疑交易识别点（重点类型）}

数据样本：
{正常钱包样本 10 条}
{异常钱包样本 10 条}

上一轮特征验证结果（如有）：
{特征名: 卡方p值, Cramer's V, 信息增益}
{需要改进的特征}
{需要删除的冗余特征}

请生成/优化 10 个新特征。
```

### 4.3 关键设计决策

| 决策点 | 选择 | 理由 |
|-------|------|------|
| LLM 输出形式 | Python 代码 | 之前一阶逻辑表达式全部无法执行，代码可直接验证 |
| 特征粒度 | 钱包级别（最终） | 目标是判断钱包是否异常，交易级特征需聚合 |
| 验证方式 | 统计检验 + 下游模型 | 双重验证确保有效性 |
| 迭代策略 | 反馈驱动 | 将验证结果回传 LLM，类似 CAAFE 的迭代优化 |
| 领域知识注入 | Prompt 内嵌 | 将识别点和专家知识直接放入 prompt |

---

## 五、与现有方法的区别

### 5.1 vs 之前的 458 条规则

| 方面 | 之前的方法 | 本方案 |
|------|----------|-------|
| 输出格式 | 一阶逻辑表达式 | Python 可执行代码 |
| 字段映射 | 引用不存在的字段 | 严格约束为实际字段 |
| 验证 | 事后验证（全部失败） | 边生成边验证 |
| 迭代 | 一次性生成 | 多轮迭代优化 |
| 领域知识 | 有限 | 系统性注入4类知识来源 |

### 5.2 vs 文献中的方法

| 方面 | Balek 2025 | CAAFE | 本方案 |
|------|-----------|-------|-------|
| 数据类型 | 文本 | 表格 | 表格+图 |
| 特征形式 | LLM直接打分 | 代码生成 | **代码生成 + 领域知识驱动** |
| 领域知识 | 无 | 数据集描述 | **专家访谈+监管规则+案例** |
| 验证 | 卡方检验 | 下游模型 | **统计检验+下游模型双重验证** |
| 可解释性 | 特征名 | 代码注释 | **领域依据+知识来源+可读描述** |

---

## 六、实现路线图

### Phase 1：基础设施（1-2 天）
- [ ] 搭建数据加载和预处理 pipeline
- [ ] 实现特征计算框架（输入 DataFrame，输出新特征）
- [ ] 实现统计验证模块（卡方、Cramer's V、互信息）
- [ ] 搭建 LLM API 调用接口

### Phase 2：特征生成（2-3 天）
- [ ] 准备 Prompt 模板和领域知识文档
- [ ] 运行 LLM 生成第一批特征（约 50 个）
- [ ] 代码验证 + 数据运行测试
- [ ] 统计验证 + 筛选有效特征

### Phase 3：迭代优化（2-3 天）
- [ ] 将验证结果反馈 LLM，迭代 3-5 轮
- [ ] 特征去冗余（相关系数 > 0.9 的去除）
- [ ] 下游模型测试（XGBoost/LightGBM + SHAP）

### Phase 4：文档与可解释性（1 天）
- [ ] 生成特征字典（每个特征的完整文档）
- [ ] SHAP 分析报告
- [ ] 与原始特征的性能对比报告

---

## 七、预期产出

1. **特征集**：30-60 个经过验证的高质量特征（含 Python 计算代码）
2. **特征字典**：每个特征附带领域依据、知识来源、可解释性描述
3. **验证报告**：统计显著性检验结果 + 下游模型性能对比
4. **可复用框架**：LLM 特征生成 pipeline，可应用于新数据集

---

## 八、核心创新点

1. **领域知识系统性注入**：将邮储专家经验、银行业识别点、监管法规系统性编码到 LLM prompt 中，而非仅靠 LLM 自身知识
2. **代码直接可执行**：避免之前一阶逻辑表达式无法落地的问题
3. **多层次特征体系**：从单字段到跨模块到图结构，5 个层次全覆盖
4. **闭环验证反馈**：统计验证结果驱动 LLM 迭代优化，类似 CAAFE 但增加了领域知识维度
5. **双重可解释性**：特征本身可解释（有领域依据）+ 模型预测可解释（SHAP）
