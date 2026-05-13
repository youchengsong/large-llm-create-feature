"""领域知识结构化模块，将专家知识、交易模式、风险指标、检测规则转为 prompt 文本。"""

import pandas as pd
from pathlib import Path


def load_transaction_patterns() -> str:
    """返回 8 种交易模式说明文本（3 正常 + 5 风险），硬编码自 LLM_FEATURE_PLAN.md §1.5。"""
    return """## 数据集交易模式说明（浦发提供）

### 正常模式
1. single_transaction: 金额1-5000元，高峰时段(9-11,14-16,19-21点)，70%单笔30%转发
2. normal_small_high_freq: 金额1-100元，日常消费，各模式各20%
3. regular_large_low_freq: 金额100-1000元，间隔1H-3天，偏向商户钱包

### 风险模式
4. small_amount_testing: 金额0.01-0.1元，深夜(0-5,22-23点)，休眠账户(>=90天)突然激活
5. merchant_laundering: 金额10-1000元，仅商户钱包，跨行90%，同地区80%，佣金7-9%
6. class4_laundering: 金额100-2000元，深夜(0-1,23点)，仅4类钱包，日限额5000
7. online_laundering: 金额10-1000元，深夜(20-1点)，交易次数多
8. victim_laundering: 诱导(设备IP一致)/盗用(设备IP不一致)两种子模式"""


def load_expert_interview() -> str:
    """返回 4 类洗钱模式压缩文本，来自邮储专家访谈。"""
    return """## 数字人民币洗钱模式（来自专家访谈）

### 模式1：薅羊毛
特征：批量注册4类钱包，同一身份证/手机号关联多个钱包，利用红包/优惠活动

### 模式2：线下跑分
特征：个人钱包收到大额转入后立即转出，交易对手固定，金额接近限额

### 模式3：线上跑分
特征：凌晨高频交易(20时-1时)，分散转出(fan_out)模式，金额10-1000元

### 模式4：商户收款码跑分
特征：商户钱包大量收款后转至个人钱包，many_to_many模式，佣金约0.5-1%"""


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
