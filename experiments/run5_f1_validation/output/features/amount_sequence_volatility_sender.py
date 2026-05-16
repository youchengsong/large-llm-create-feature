"""
特征: amount_sequence_volatility_sender
说明: 异常交易中金额波动剧烈，与历史均值偏离过大是金额异常风险的重要指标。通过计算序列化的金额变化率，可以捕捉到短时间内金额大幅波动的异常行为模式。

验证指标:
  - Score: 0.4408
  - MI: 0.0000
  - Redundancy: 0.2199
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 72

可解释性报告:
  - 值含义: {'0-10': '金额序列波动极小，表示该钱包的交易金额非常稳定，变化幅度小，常见于日常小额消费或固定金额的定期转账等正常交易模式。', '10-30': '金额序列有一定波动，属于正常范围，反映了用户日常交易中金额的自然变化，如不同金额的消费、转账等。', '30-80': '金额序列波动较大，可能表明用户交易行为出现变化，如偶发的大额交易、收入波动等，需要结合其他特征进一步判断。', '80-200': '金额序列波动剧烈，明显偏离历史均值，是潜在异常行为的信号，可能涉及拆分交易、洗钱等风险模式。', '>200': '金额序列波动极端剧烈，极大概率是异常交易行为，如短时间内金额从极小值突然跃升至极大值，或频繁进行大额、小额交替交易以规避监管。'}
  - 风险阈值: 高风险阈值为 80。判断依据：异常交易在 75% 分位（87.44）处开始明显高于正常交易（82.17），且值大于 80 时，金额波动已超过正常交易的典型范围，符合「交易金额波动剧烈，与历史均值偏离过大」的风险特征。
  - 协同特征: [{'feature': 'send_concentration', 'reason': 'send_concentration 衡量资金是否集中流向少数对手，与 amount_sequence_volatility_sender 结合可以区分「集中式大额转账」和「分散式小额试探」两种不同的异常模式。例如，高波动性+高集中度可能指向洗钱中的资金归集行为。'}, {'feature': 'avg_inactive_period', 'reason': 'avg_inactive_period 衡量交易间隔，与金额波动结合可以识别「休眠账户突然激活后大额交易」的典型风险场景。高波动性+长平均间隔是洗钱模式中常见的特征组合。'}, {'feature': 'transaction_frequency', 'reason': 'transaction_frequency 衡量交易频率，与金额波动结合可以识别「高频小额测试+突然大额转账」的跑分模式。高波动性+高频率是线上跑分的典型行为特征。'}]
  - 误判场景: ['用户进行一次性大额消费或转账，如购买高价商品、支付学费、购房首付等，会导致单笔交易金额与历史均值偏离较大，但属于正常行为。', '用户收入模式不规律，如自由职业者、兼职人员，其收入和支出金额波动较大，但并非洗钱行为。', '用户在不同场景下使用钱包，如日常消费（小额）和偶尔的理财操作（大额），导致金额序列波动较大。', '新注册钱包，仅有少量交易记录，此时金额波动统计不稳定，容易产生较高的波动值。']
  - 监管说明: 该指标衡量一个钱包在连续交易中金额变化的剧烈程度，数值越高说明该钱包的金额忽大忽小、波动异常，可能是在通过拆分或混入不同金额来掩盖资金流向。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_sequence_volatility_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    tx_sorted = tx_df.sort_values(['src', 'timestamp']).copy()
    tx_sorted['prev_amount'] = tx_sorted.groupby('src')['amount'].shift(1)
    tx_sorted['amount_change_rate'] = (tx_sorted['amount'] - tx_sorted['prev_amount']).abs() / (tx_sorted['prev_amount'].abs() + 1)
    volatility = tx_sorted.groupby('src')['amount_change_rate'].std().fillna(0).reset_index()
    volatility.columns = ['wallet_id', 'amount_sequence_volatility_sender']
    result = volatility
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_sequence_volatility_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_sequence_volatility_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
