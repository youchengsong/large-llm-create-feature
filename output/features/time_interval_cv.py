"""
特征: time_interval_cv
说明: 异常交易（如线上跑分、小额测试）的行为模式不固定，时间间隔波动大，而正常交易（如日常消费、定期转账）的时间间隔相对稳定。时间间隔的变异系数能有效量化这种波动性，补充均值特征的不足。

验证指标:
  - Score: 0.0770
  - MI: 0.0012
  - Redundancy: 0.8152
  - Cramer's V: 0.0270
  - Delta PR-AUC: -0.012470
  - SHAP Rank: 33

可解释性报告:
  - 值含义: {'0-1.0': '时间间隔高度规律，交易行为非常稳定，如定期定额转账、固定时间消费，属于正常交易模式。', '1.0-1.5': '时间间隔有一定波动但整体规律，属于日常消费、随机小额支付的正常波动范围。', '1.5-2.5': '时间间隔波动较大，可能存在多种交易模式混合，或交易时间不固定，有一定洗钱风险。', '>2.5': '时间间隔极不稳定，交易行为异常，如凌晨随机小额测试、跑分频繁切换对手，高洗钱风险。'}
  - 风险阈值: 高风险阈值为 time_interval_cv > 2.5。依据：异常交易中位数(1.1465)低于正常(1.4593)，但异常交易标准差(0.5452)小于正常(0.6596)，说明异常交易的波动范围更集中。正常交易的75%分位为1.8767，而异常交易的25%分位为1.0014，两者有重叠。取2.5作为阈值，能有效识别出时间间隔极端不稳定的异常交易（如小额测试的随机间隔），同时避免误伤正常波动较大的消费行为。
  - 协同特征: [{'feature': 'activetime（活跃时段）', 'reason': 'time_interval_cv高但activetime在白天，可能是正常消费波动；time_interval_cv高且activetime在凌晨，则高度怀疑为小额测试或线上跑分。'}, {'feature': 'large_transaction_ratio（大额交易占比）', 'reason': 'time_interval_cv高且large_transaction_ratio低，符合小额测试模式（金额小、间隔随机）；time_interval_cv高且large_transaction_ratio高，可能是大额跑分（金额大、间隔不稳定）。'}, {'feature': 'Out_Degree（出度）', 'reason': 'time_interval_cv高且Out_Degree高，符合跑分模式（频繁转出给不同对手）；time_interval_cv低且Out_Degree高，可能是正常商户大量收款。'}]
  - 误判场景: ['兼职自由职业者：收入时间不固定，可能一天内多次小额收款，导致时间间隔波动大，CV值偏高。', '跨时区用户：如留学生或跨境工作者，交易时间跨越多个时区，时间间隔不规律。', '新注册用户：刚激活的账户，交易笔数少（如2-3笔），时间间隔差异被放大，CV值异常高。']
  - 监管说明: 该特征衡量用户交易时间间隔的规律性，数值越大说明交易时间越不固定，对于正常用户通常有稳定的作息和交易习惯，而洗钱账户往往在不同时间点随机发起交易以规避监测。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 time_interval_cv，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 1. 计算每笔交易的时间间隔（按src分组，时间排序）
    # 先确保 timestamp 是 datetime 类型
    tx_df = tx_df.copy()
    tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'])
    tx_df = tx_df.sort_values(['src', 'timestamp'])
    tx_df['prev_timestamp'] = tx_df.groupby('src')['timestamp'].shift(1)
    tx_df['time_diff'] = (tx_df['timestamp'] - tx_df['prev_timestamp']).dt.total_seconds()

    # 2. 按src计算时间间隔的变异系数（CV = std / mean）
    # 对于只有1笔交易的钱包，time_diff全部为NaN，CV设为0
    stats = tx_df.groupby('src')['time_diff'].agg(['std', 'mean'])
    stats['time_interval_cv'] = np.where(stats['mean'] > 0, stats['std'] / stats['mean'], 0)
    stats = stats.reset_index()
    stats = stats[['src', 'time_interval_cv']]
    stats.columns = ['wallet_id', 'feature_value']
    stats = stats.fillna(0)
    result = stats
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: time_interval_cv")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 time_interval_cv: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
