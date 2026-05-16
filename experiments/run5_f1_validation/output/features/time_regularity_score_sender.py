"""
特征: time_regularity_score_sender
说明: 正常模式中的regular_large_low_freq有固定时间间隔(1H-3天)，而洗钱模式(如线上跑分)的交易时间更随机。时间间隔的变异系数能区分规律性交易和随机交易。

验证指标:
  - Score: 0.3205
  - MI: 0.0000
  - Redundancy: 0.6671
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 94

可解释性报告:
  - 值含义: {'0-1.0': '交易时间间隔非常规律（如定期定额转账），通常为正常用户（如工资发放、定期缴费），但需警惕极低值可能为自动化交易', '1.0-2.5': '时间间隔有一定波动但整体规律，属于正常用户常见范围，涵盖regular_large_low_freq模式的正常交易', '2.5-4.0': '时间间隔波动较大，交易时间趋于随机，可能为洗钱特征（线上跑分、薅羊毛等），异常交易中位数(2.54)落在此区间', '>4.0': '交易时间高度随机，间隔极不规律，强烈暗示洗钱行为（如分散转出、凌晨突发交易），需要重点关注'}
  - 风险阈值: 2.8
  - 协同特征: [{'feature': 'transaction_frequency_From', 'reason': '高频交易+高时间随机性=典型的线上跑分模式（凌晨高频转出），两者同时高时风险显著提升'}, {'feature': 'is_night_hour_From', 'reason': '凌晨交易+时间间隔随机=符合online_laundering和small_amount_testing特征，可精准识别深夜异常活动'}, {'feature': 'send_concentration', 'reason': '高时间随机性+低交易对手集中度（分散转出）=典型的fan_out洗钱模式，可结合Out_Degree判断'}, {'feature': 'avg_inactive_period_From', 'reason': '长休眠期后突然活跃+时间间隔随机=休眠账户激活型洗钱（small_amount_testing模式）'}]
  - 误判场景: ['自由职业者/临时工：收入来源不固定，交易时间间隔天然随机，可能误判为高风险', '刚激活的休眠账户：正常用户重新使用账户时，初期交易时间间隔不规律，但后续会趋于正常', '跨时区用户：由于时差导致本地时间看似凌晨交易，实际是对方正常工作时间', '紧急大额支出（如医疗、维修）：突发性交易导致时间间隔异常，但后续无持续异常行为']
  - 监管说明: 该特征衡量用户交易时间间隔的规律程度——正常用户（如定期缴费、工资发放）的交易间隔相对固定，而洗钱者为了快速转移资金，交易时间往往毫无规律可言。值越高，说明用户的交易时间越像随机打乱，越需要人工核查其交易背景。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 time_regularity_score_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    tx_sorted = tx_df.sort_values(['src', 'timestamp']).copy()
    tx_sorted['prev_timestamp'] = tx_sorted.groupby('src')['timestamp'].shift(1)
    tx_sorted['time_diff'] = (pd.to_datetime(tx_sorted['timestamp']) - pd.to_datetime(tx_sorted['prev_timestamp'])).dt.total_seconds()
    tx_sorted = tx_sorted.dropna(subset=['time_diff'])
    interval_stats = tx_sorted.groupby('src')['time_diff'].agg(['mean', 'std']).reset_index()
    interval_stats.columns = ['wallet_id', 'mean_interval', 'std_interval']
    interval_stats['time_regularity_score_sender'] = interval_stats['std_interval'] / interval_stats['mean_interval'].clip(lower=1)
    interval_stats['time_regularity_score_sender'] = interval_stats['time_regularity_score_sender'].fillna(0)
    result = interval_stats[['wallet_id', 'time_regularity_score_sender']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: time_regularity_score_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 time_regularity_score_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
