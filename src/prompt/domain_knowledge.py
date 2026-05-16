"""领域知识结构化模块，将专家知识、交易模式、风险指标、检测规则转为 prompt 文本。"""

import pandas as pd
from pathlib import Path


def load_transaction_patterns() -> str:
    """返回 8 种交易模式说明文本（3 正常 + 5 风险）。"""
    return """## 数据集交易模式说明

### 正常模式

1. 单笔交易（single_transaction）— 正常交易
   - 时间间隔：交易频率适中，间隔具有一定随机性
   - 交易模式：以单笔交易为主，少量存在转发行为
   - 时间偏好：更倾向于白天和晚间活跃时段，较少出现在凌晨
   - 生成逻辑：从源钱包随机选择目标钱包进行单次转账
   - 适用钱包：适用于所有钱包类型

2. 正常小额高频交易（normal_small_high_freq）— 正常交易
   - 金额特征：金额较小，以日常消费场景为主
   - 时间间隔：交易较频繁，间隔时间较短且随机
   - 交易模式：包含扇入、扇出、多对一、一对多、多对多等多种模式
   - 时间偏好：集中于日常消费高峰时段，较少出现在凌晨
   - 生成逻辑：模拟日常消费与零钱流转行为，支持多种复杂交易结构
   - 适用钱包：适用于所有钱包类型

3. 常规大额低频交易（regular_large_low_freq）— 正常交易
   - 金额特征：金额相对较高
   - 时间间隔：交易频率较低，间隔时间较长
   - 交易模式：以单笔交易为主，部分存在一对多转账
   - 时间偏好：偏向工作时段发生，较少出现在深夜
   - 生成逻辑：模拟企业转账、工资发放等大额资金流动场景
   - 适用钱包：适用于所有钱包类型，更偏向商户钱包

### 风险模式

4. 小额测试交易（small_amount_testing）— 风险交易
   - 金额特征：金额极小，具有试探性特征
   - 时间间隔：交易非常频繁，短时间内连续发生
   - 交易模式：以扇出为主，部分存在扇入行为
   - 时间偏好：主要集中在深夜或非活跃时段
   - 生成逻辑：模拟长期休眠账户突然发起的小额测试交易
   - 适用钱包：适用于所有钱包类型
   - 特殊条件：大量交易来源于长期休眠账户

5. 商户洗钱（merchant_laundering）— 风险交易
   - 金额特征：金额跨度较大，存在明显资金流转特征
   - 时间间隔：交易较为频繁
   - 交易模式：以一对多交易为主，同时伴随复杂多对多交易
   - 时间偏好：主要集中在商户正常经营时段
   - 生成逻辑：模拟商户钱包利用经营交易进行异常资金清洗
   - 适用钱包：仅适用于商户钱包
   - 特殊条件：具有较强跨行交易偏好，同时偏向同地区交易，并伴随一定佣金抽成特征

6. 四类钱包洗钱（class4_laundering）— 风险交易
   - 金额特征：金额相对较高
   - 时间间隔：交易频率高，资金快速流转
   - 交易模式：包含多对一、一对多以及多对多等复杂模式
   - 时间偏好：主要发生在深夜或低活跃时段
   - 生成逻辑：模拟四类钱包利用限额特征进行资金清洗
   - 适用钱包：仅适用于四类钱包
   - 特殊条件：存在明显的日累计额度控制特征

7. 在线洗钱（online_laundering）— 风险交易
   - 金额特征：金额中等偏低，交易分散
   - 时间间隔：交易频繁，连续性较强
   - 交易模式：包含扇入、扇出、多对一、一对多及多对多等多种复杂结构
   - 时间偏好：更偏向夜间及深夜时段
   - 生成逻辑：模拟在线钱包环境下的异常资金清洗行为
   - 适用钱包：适用于所有钱包类型
   - 特殊条件：整体交易次数明显偏多

8. 受害者洗钱（victim_laundering）— 风险交易
   - 交易模式：包含扇入、多对一、一对多及多对多等复杂资金流转模式
   - 生成逻辑：上游用户存在正常日常交易行为，并进一步演化为异常资金流转，主要包括：
     - 诱导操作：交易设备与IP环境与开户环境基本一致
     - 盗用操作：交易行为与历史环境存在一定偏离或异常
   - 适用钱包：适用于所有钱包类型"""


def load_expert_interview() -> str:
    """返回洗钱模式知识文本，来自专家访谈 + 309份裁判文书提取。"""
    return """## 数字人民币洗钱模式（来自专家访谈）

### 模式1：薅羊毛
特征：批量注册4类钱包，同一身份证/手机号关联多个钱包，利用红包/优惠活动

### 模式2：线下跑分
特征：个人钱包收到大额转入后立即转出，交易对手固定，金额接近限额

### 模式3：线上跑分
特征：凌晨高频交易(20时-1时)，分散转出(fan_out)模式，金额10-1000元

### 模式4：商户收款码跑分
特征：商户钱包大量收款后转至个人钱包，many_to_many模式，佣金约0.5-1%

## 真实案例洗钱特征（来自309份数字人民币相关裁判文书全文阅读）

### 一、资金转移链路模式（7种）
1. 银行卡→数字人民币钱包→转出（最主流，85%案件）：资金先入银行卡，再通过数字人民币钱包转至上线指定账户
2. 多渠道并行分散转出：同时使用数字人民币+微信+支付宝+ATM取现（5种方式并用），增加追踪难度
3. 数字人民币+ATM取现组合：部分资金通过钱包转出，无法转出的通过ATM取现
4. 虚拟货币兑换：数字人民币钱包收款后兑换USDT（通过Gate.io/火币等平台），再通过虚拟货币交易转移
5. 多跳数字人民币链路：资金经过4+个数字人民币账户逐级转移后汇聚到一个结算账户（增加追踪难度，部分跳转数据警方无法获取）
6. 数字人民币→礼品卡→二手平台→虚拟货币：四步商品转换链（京东E卡→闲鱼变现→火币买USDT→上线地址）
7. 数字人民币→实物资产：通过数字人民币二维码在商场购买黄金等实物，直接脱离电子审计链

### 二、典型操作流程（跑分模式，61/309案件明确提及）
1. 通过Telegram/微信/QQ/抖音招募卡主，承诺佣金（或以"办贷款"为幌子诱骗）
2. 见面后收取卡主的银行卡+身份证+手机（三件套，72/309案件）
3. 在卡主手机上下载数字人民币APP+云闪付+手机银行并绑定银行卡（多APP同时安装）
4. 通过Telegram群接收上线指令，等待资金入账
5. 资金到账后，由操作人员或配合卡主刷脸(128/309)/指纹验证完成转账
6. 多渠道转出：数字人民币钱包转账、跨行转账、微信零钱通、ATM取现、信用卡还款
7. 转出完成后按比例结算佣金（现金/转账/积分）

### 三、佣金/提成特征（可作为异常金额模式识别依据）
- 常见比例：1%-5%（最集中在2%-4%）
- 按笔计算：150-500元/万元（多层级时逐级递减：250→200→150→20元/万）
- 固定日薪：300-500元/天（底层操作员）
- 固定单次：500-7600元/次（卡主/工具人）
- 积分制：1元=1积分，积分可兑换现金（如303万积分兑现2.5万元）
- 特征表现：转出金额约等于转入金额×(1-佣金率)，差额为固定比例
- USDT套利：以高于市场价约2%出售USDT，数字人民币钱包收取差价

### 四、关键行为特征（可转化为特征工程线索）
1. 当日即转即取（58/309）：资金入账后极短时间内全部转出，最快75秒完成出账
2. 刷脸认证频繁（128/309）：短时间内多次人脸验证/指纹验证=多次转账操作
3. 验卡行为（13/309）：正式跑分前有小额测试交易确认银行卡/钱包可用
4. 账户冻结后换卡（77/309）：一张卡被冻结后立即切换到新卡继续，或注销账户取回余额
5. 多卡轮换操作（16/309）：同一人名下多张银行卡轮流使用
6. 拆分多次转出（41/309）：大额资金拆分为多笔小额分散转出（平均2218元/笔）
7. 跨行转账偏好：资金入账和转出倾向于不同银行，增加追踪难度
8. 多银行钱包注册：同一人在7+家银行同时开通数字人民币钱包（邮储/建设/中行/兴业/招商/交通/工商）
9. 亲属身份注册：用配偶/父母/兄弟姐妹身份开通钱包，单人控制5-36个钱包
10. 单卡绑定多钱包：一张银行卡绑定9个数字人民币账户作为结算通道
11. 风控绕过：账户被冻结后用营业执照等虚假材料欺骗银行解冻

### 五、时间特征
- 犯罪操作集中在下午(14-17时)最多，其次上午和晚间
- 夜班集中操作（22:00-04:00）是大额跑分的标准模式，单晚可达100万+
- 凌晨(0-5时)也有显著活动（14%），对应在线洗钱模式
- 资金到账后通常在数小时内完成全部转移（当日清模式）
- 单次操作窗口：2-8小时完成全部资金流转
- 钱包活跃周期：平均2-28天后被冻结或主动废弃

### 六、组织分工特征
- 上线/群主：通过Telegram发布任务、指定收款账户
- 操作员：负责手机转账操作（日薪300-500元）
- 取现员：负责ATM取现并交接现金
- 卡主/工具人：提供银行卡和刷脸配合（500-7600元/次）
- 中介/介绍人：负责招募卡主，每介绍一人获得固定报酬（300元/万或1000元/人）
- 平台运营：通过"跑跑"APP/"CX支付"等平台分配任务和管理钱包

### 七、规模化钱包网络特征（可转化为图特征）
- 单人钱包数量：5-36个（博盈平台案件典型6-17个）
- 组织化网络钱包总量：60-394个
- 单人资金流水：36万-1434万元
- 组织总流水：470万-2839万元
- 最大单钱包流水：2.94亿元（农业银行数字人民币钱包）
- 单人单日流水：21.4万元（9个钱包同日操作）

### 八、商户/企业钱包洗钱特征
- 珠宝店：6天内141笔交易收入199万+，兑换实物黄金
- 茶庄：注册商户二维码+数字人民币钱包作为收款前端，被冻结后伪造材料解冻
- 副食品店：将店铺数字人民币二维码拍照发给上线，由受害人直接扫码付款
- 超市：犯罪分子指挥买家用数字人民币购买烟酒，店主将数字人民币全部提现（总计107万元）
- 酒业公司：公司对公账户+数字人民币钱包联动，2个月收入157万+

### 九、"博盈"平台USDT套利模式（2024年高发）
- 在"博盈"平台注册QQ邮箱账号，绑定多个数字人民币钱包收款
- 通过Telegram购买USDT，在平台上以高于市场价约2%出售
- 买家通过数字人民币钱包付款，卖家赚取差价
- 单人月流水36万-278万元，利润1500-25000元
- 为防冻结，用亲属身份注册钱包并定期轮换"""


def load_risk_indicators(
    filepath: str = "inputs/domain_knowledge/风险分类表_38指标.xlsx",
) -> str:
    """读取风险分类表，提取 38 个指标为压缩表格文本。"""
    df = pd.read_excel(filepath)

    col_seq = df.iloc[0].tolist()  # 序号, 维度, 风险类, 数据类, 方向
    header_names = [str(c).strip() for c in col_seq]
    data = df.iloc[1:].copy()
    data.columns = header_names

    data = data[data["序号"].notna()].reset_index(drop=True)

    data["维度"] = data["维度"].ffill()
    data["风险类"] = data["风险类"].ffill()

    lines = ["## 风险分类指标（38个）", ""]
    current_dim = None
    current_risk = None

    for _, row in data.iterrows():
        dim = str(row["维度"]).strip()
        risk_cls = str(row["风险类"]).strip()
        data_cls = str(row["数据类"]).strip() if pd.notna(row["数据类"]) else ""
        direction = str(row["方向"]).strip() if pd.notna(row["方向"]) else ""

        if dim != current_dim:
            lines.append(f"### {dim}")
            current_dim = dim
            current_risk = None

        if risk_cls != current_risk:
            lines.append(f"**{risk_cls}**")
            current_risk = risk_cls

        entry = f"- {data_cls}"
        if direction:
            entry += f"：{direction}"
        lines.append(entry)

    return "\n".join(lines)


def load_detection_rules(
    filepath: str = "inputs/domain_knowledge/检测规则_15条.xlsx",
) -> list[dict]:
    """读取检测规则 Excel，返回 15 条规则的字典列表。"""
    df = pd.read_excel(filepath)

    df = df[df["规则编号"].notna() & df["规则编号"].astype(str).str.startswith("RULE")].copy()

    rules = []
    for _, row in df.iterrows():
        rules.append(
            {
                "id": str(row["规则编号"]).strip(),
                "description": str(row["自然语言描述"]).strip(),
                "logic": str(row["一阶逻辑表达式"]).strip(),
                "confidence": str(row["置信度"]).strip(),
                "type": str(row["规则类型"]).strip(),
                "category": str(row["异常模式类别"]).strip(),
            }
        )
    return rules


def rules_to_python_examples(rules: list[dict], n: int = 2) -> str:
    """将前 n 条规则转为 Python 函数示例文本，供 LLM 参考。"""
    examples = []

    if len(rules) >= 1:
        examples.append(
            '''def feature_dormant_account_activation(open_df, tx_df, close_df):
    """RULE_002: 休眠账户突然激活 — 开户后>=90天无交易后突然出现交易"""
    import pandas as pd
    open_df = open_df.copy()
    open_df["wallet_open_date"] = pd.to_datetime(open_df["wallet_open_date"])
    tx_df_tmp = tx_df[["src", "timestamp"]].copy()
    tx_df_tmp["timestamp"] = pd.to_datetime(tx_df_tmp["timestamp"])
    first_tx = tx_df_tmp.groupby("src")["timestamp"].min().reset_index()
    first_tx.columns = ["wallet_id", "first_tx_time"]
    merged = open_df[["wallet_id", "wallet_open_date"]].merge(first_tx, on="wallet_id", how="left")
    merged["dormant_days"] = (merged["first_tx_time"] - merged["wallet_open_date"]).dt.days
    merged["is_dormant_activated"] = (merged["dormant_days"] >= 90).astype(int)
    return merged[["wallet_id", "is_dormant_activated"]]'''
        )

    if len(rules) >= 2:
        examples.append(
            '''def feature_merchant_abnormal_volume(open_df, tx_df, close_df):
    """RULE_004: 商户账户异常交易量 — 商户钱包短时间内大额交易"""
    import pandas as pd
    merchant_wallets = open_df[open_df["wallet_type"] == 3]["wallet_id"]
    merchant_tx = tx_df[tx_df["dst"].isin(merchant_wallets)].copy()
    merchant_tx["timestamp"] = pd.to_datetime(merchant_tx["timestamp"])
    merchant_tx["date"] = merchant_tx["timestamp"].dt.date
    daily = merchant_tx.groupby(["dst", "date"])["amount"].agg(["sum", "count"]).reset_index()
    daily.columns = ["wallet_id", "date", "daily_amount", "daily_count"]
    stats = daily.groupby("wallet_id").agg(
        max_daily_amount=("daily_amount", "max"),
        mean_daily_amount=("daily_amount", "mean"),
    ).reset_index()
    stats["merchant_volume_ratio"] = stats["max_daily_amount"] / stats["mean_daily_amount"].clip(lower=1)
    return stats[["wallet_id", "merchant_volume_ratio"]]'''
        )

    result_examples = examples[:n]
    lines = ["## 检测规则 Python 实现示例", ""]
    for i, code in enumerate(result_examples, 1):
        lines.append(f"### 示例 {i}")
        lines.append("```python")
        lines.append(code)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
