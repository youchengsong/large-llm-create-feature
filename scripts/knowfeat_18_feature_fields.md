# KnowFeat 18 个新特征 — 使用字段说明

> 每个特征函数签名: `feature_xxx(open_df, tx_df, close_df)`
> - `open_df` = accounts.csv (账户开立表)
> - `tx_df` = transactions.csv (交易表)
> - `close_df` = close.csv (注销表, 当前 18 个特征均未使用)

---

## 汇总：数据表依赖

| # | 特征名 | 维度 | open_df | tx_df | close_df |
|---|--------|------|---------|-------|----------|
| 1 | small_test_round_amount_ratio | 金额频次 | - | src, amount | - |
| 2 | amount_concentration_entropy | 金额频次 | - | src, amount | - |
| 3 | top3_hour_concentration | 时间地域 | - | src, hour | - |
| 4 | fan_in_fan_out_ratio | 行为序列 | - | src, transaction_mode | - |
| 5 | mode_transition_entropy | 行为序列 | - | src, transaction_mode | - |
| 6 | new_counterparty_concentration | 主体关联 | - | src, dst | - |
| 7 | counterparty_level_diversity_risk | 主体关联 | - | src, dst_wallet_level, src_wallet_level | - |
| 8 | amount_range_ratio_risk | 金额频次 | - | src, amount | - |
| 9 | high_risk_region_off_hour_ratio | 时间地域 | wallet_id, region_code | src, hour, tx_id | - |
| 10 | rhythm_stability_risk | 行为序列 | - | src, interval | - |
| 11 | net_flow_direction_risk | 行为序列 | - | src, dst, timestamp, amount | - |
| 12 | enterprise_counterparty_concentration | 主体关联 | wallet_id, wallet_type | src, dst | - |
| 13 | first_small_test_ratio | 行为序列 | - | src, timestamp, amount | - |
| 14 | night_amount_short_term_risk | 时间地域 | - | src, hour, amount, timestamp | - |
| 15 | amount_bin_skew_index | 金额频次 | - | src, amount | - |
| 16 | counterparty_type_amount_divergence | 主体关联+金额 | wallet_id, wallet_type | src, dst, amount | - |
| 17 | tiny_amount_concentration | 金额频次 | - | src, amount | - |
| 18 | time_concentration_risk_index | 时间地域 | - | src, hour | - |

---

## 各特征详细字段说明

### #1 small_test_round_amount_ratio (金额频次)

**使用字段:**
- `tx_df.src` — 按发送方钱包聚合
- `tx_df.amount` — 取绝对值后判断: <=5 为小额测试, %100==0 为整数金额

**计算逻辑:** 小额测试比例 × (1 + 整数金额比例)

---

### #2 amount_concentration_entropy (金额频次)

**使用字段:**
- `tx_df.src` — 按发送方聚合
- `tx_df.amount` — 取绝对值, 计算 Top3 金额频率占比和金额分布熵

**计算逻辑:** Top3金额占比 × (1 - 归一化熵)

---

### #3 top3_hour_concentration (时间地域)

**使用字段:**
- `tx_df.src` — 按发送方聚合
- `tx_df.hour` — 交易小时 (0-23), 统计 Top3 小时的交易占比

**计算逻辑:** 交易最集中的 3 个小时的交易笔数占总交易笔数的比例

---

### #4 fan_in_fan_out_ratio (行为序列)

**使用字段:**
- `tx_df.src` — 按发送方聚合
- `tx_df.transaction_mode` — 交易模式类别 (fan_in, fan_out, many_to_many, one_to_many 等)

**计算逻辑:** fan_in占比 / fan_out占比 × (1 + 异常模式总占比)

---

### #5 mode_transition_entropy (行为序列)

**使用字段:**
- `tx_df.src` — 按发送方聚合
- `tx_df.transaction_mode` — 交易模式类别, 计算模式分布的归一化熵

**计算逻辑:** 1 - 归一化熵 (值越高表示模式越单一, 越可疑)

---

### #6 new_counterparty_concentration (主体关联)

**使用字段:**
- `tx_df.src` — 发送方 (按 src 聚合)
- `tx_df.dst` — 接收方 (统计交易对手数和新交易对手数)

**计算逻辑:** 新交易对手比例 (只出现1次的对手) / (交易对手总数 + 1)

---

### #7 counterparty_level_diversity_risk (主体关联)

**使用字段:**
- `tx_df.src` — 按发送方聚合
- `tx_df.dst_wallet_level` — 接收方钱包等级 (1-4), 计算 nunique 多样性
- `tx_df.src_wallet_level` — 发送方自身钱包等级

**计算逻辑:** (1 - 对手等级多样性/4) × (1 + (1 - 自身等级/4))

---

### #8 amount_range_ratio_risk (金额频次)

**使用字段:**
- `tx_df.src` — 按发送方聚合
- `tx_df.amount` — 筛选正金额, 计算 max/min 比值

**计算逻辑:** log(1 + 最大正金额 / 最小正金额), 交易次数<2 置为 0

---

### #9 high_risk_region_off_hour_ratio (时间地域)

**使用字段:**
- `open_df.wallet_id` — 账户 ID, 用于 join
- `open_df.region_code` — 地区代码, 判断是否高风险地区 (540000=西藏, 650000=新疆)
- `tx_df.src` — 按发送方聚合
- `tx_df.hour` — 判断非工作时间 (18:00-7:59)
- `tx_df.tx_id` — 计算交易总数 (count)

**计算逻辑:** 高风险地区标志 (0/1) × 非工作时间交易占比

---

### #10 rhythm_stability_risk (行为序列)

**使用字段:**
- `tx_df.src` — 按发送方聚合
- `tx_df.interval` — 交易间隔 (秒), 计算均值/标准差/长间隔(>300s)比例

**计算逻辑:** 长间隔比例 × 1/(1+CV), 其中 CV = std/mean

---

### #11 net_flow_direction_risk (行为序列)

**使用字段:**
- `tx_df.src` — 出金方 (direction=-1)
- `tx_df.dst` — 入金方 (direction=+1)
- `tx_df.timestamp` — 交易时间, 用于排序
- `tx_df.amount` — 交易金额, 计算净流量

**计算逻辑:** |净流量比例| × (1 - 方向切换比例). 同时考虑 src 和 dst 双视角。

---

### #12 enterprise_counterparty_concentration (主体关联)

**使用字段:**
- `open_df.wallet_id` — 账户 ID, 构建 wallet_type 映射表
- `open_df.wallet_type` — 钱包类型 (1=个人, 2=企业, 3=商户), 映射到 dst
- `tx_df.src` — 按发送方聚合
- `tx_df.dst` — 查找对手的钱包类型

**计算逻辑:** 企业钱包(type=2)交易对手占比 × (1 - 对手类型归一化熵)

---

### #13 first_small_test_ratio (行为序列)

**使用字段:**
- `tx_df.src` — 按发送方聚合
- `tx_df.timestamp` — 排序确定首笔交易
- `tx_df.amount` — 首笔交易金额, 判断 |amount| <= 5

**计算逻辑:** 首笔交易是否为小额测试 (<=5元), 二值特征 (0/1)

---

### #14 night_amount_short_term_risk (时间地域)

**使用字段:**
- `tx_df.src` — 按发送方聚合
- `tx_df.hour` — 判断深夜时段 (22:00-5:59)
- `tx_df.amount` — 取绝对值, 计算深夜交易金额占比
- `tx_df.timestamp` — 计算交易时间跨度 (首末交易天数差)

**计算逻辑:** 深夜金额占比 × 1/(时间跨度天数 + 1)

**注意:** 此特征在运行时可能因 timestamp 列类型为 str 导致 `.dt.days` 报错。需确保 timestamp 列已转为 datetime。

---

### #15 amount_bin_skew_index (金额频次)

**使用字段:**
- `tx_df.src` — 按发送方聚合
- `tx_df.amount` — 取绝对值, 分三个区间统计:
  - 极小额: <= 1 元
  - 中等金额: 1,000 ~ 50,000 元
  - 大额: >= 50,000 元

**计算逻辑:** 极小额占比 / (中等金额占比 + 大额占比 + 0.01)

---

### #16 counterparty_type_amount_divergence (主体关联 + 金额频次)

**使用字段:**
- `open_df.wallet_id` — 账户 ID, 构建 wallet_type 映射表
- `open_df.wallet_type` — 钱包类型, 映射到 dst
- `tx_df.src` — 按发送方聚合
- `tx_df.dst` — 查找对手钱包类型
- `tx_df.amount` — 筛选正金额, 按对手类型分别汇总

**计算逻辑:** 向个人钱包(type=1)金额占比 - 向商户钱包(type=3)金额占比

---

### #17 tiny_amount_concentration (金额频次)

**使用字段:**
- `tx_df.src` — 按发送方聚合
- `tx_df.amount` — 取绝对值, 筛选 <= 1 元的交易, 统计最常见金额值的出现占比

**计算逻辑:** 极小额交易(<=1元)中, 最常见金额的出现频率

---

### #18 time_concentration_risk_index (时间地域)

**使用字段:**
- `tx_df.src` — 按发送方聚合
- `tx_df.hour` — 交易小时, 计算每小时交易占比和活跃小时数

**计算逻辑:** 最大小时交易占比 / 活跃小时数

---

## 字段频率统计

### tx_df 字段使用频率

| 字段 | 使用次数 | 使用该字段的特征 |
|------|---------|----------------|
| `src` | 18 | 全部 18 个特征 |
| `amount` | 11 | #1,#2,#8,#11,#13,#14,#15,#16,#17 及部分间接使用 |
| `hour` | 5 | #3,#9,#14,#18 及 top3_hour |
| `dst` | 5 | #6,#11,#12,#16 及 enterprise |
| `transaction_mode` | 2 | #4,#5 |
| `timestamp` | 3 | #11,#13,#14 |
| `interval` | 1 | #10 |
| `dst_wallet_level` | 1 | #7 |
| `src_wallet_level` | 1 | #7 |
| `tx_id` | 1 | #9 (count) |

### open_df 字段使用频率

| 字段 | 使用次数 | 使用该字段的特征 |
|------|---------|----------------|
| `wallet_id` | 3 | #9,#12,#16 (用于 join) |
| `wallet_type` | 2 | #12,#16 |
| `region_code` | 1 | #9 |

### close_df

未被任何特征使用 (函数签名保留该参数以保持接口一致)。
