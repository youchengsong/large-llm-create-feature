"""
特征: consecutive_tx_time_std
说明: 异常交易往往在凌晨时段突发交易，时间间隔不规律；正常交易（如 regular_large_low_freq）有固定的时间间隔（1H-3天）。计算连续交易时间间隔的标准差可以量化时间间隔的波动性，异常钱包的波动性通常更大。

验证指标:
  - Score: 0.6533
  - MI: 0.0012
  - Redundancy: 0.6201
  - Cramer's V: 0.0401
  - Delta PR-AUC: -0.018126
  - SHAP Rank: 6

可解释性报告:

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 consecutive_tx_time_std，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按钱包和时间排序
    tx_sorted = tx_df.sort_values(['src', 'timestamp'])
    tx_sorted['prev_timestamp'] = tx_sorted.groupby('src')['timestamp'].shift(1)
    tx_sorted['prev_timestamp'] = pd.to_datetime(tx_sorted['prev_timestamp'])
    tx_sorted['timestamp_dt'] = pd.to_datetime(tx_sorted['timestamp'])
    tx_sorted['time_diff'] = (tx_sorted['timestamp_dt'] - tx_sorted['prev_timestamp']).dt.total_seconds()

    # 计算每个钱包连续交易时间间隔的标准差
    time_std = tx_sorted.groupby('src')['time_diff'].std().reset_index()
    time_std.columns = ['wallet_id', 'consecutive_tx_time_std']
    time_std['consecutive_tx_time_std'] = time_std['consecutive_tx_time_std'].fillna(0)
    result = time_std
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: consecutive_tx_time_std")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 consecutive_tx_time_std: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
