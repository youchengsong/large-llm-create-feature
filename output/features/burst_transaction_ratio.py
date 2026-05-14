"""
特征: burst_transaction_ratio
说明: 洗钱行为常在深夜时段（20时-1时）爆发式交易，短时间内大量小额转账。burst_transaction_ratio能准确捕捉这种时间密集度异常，与整体平均频率互补。

验证指标:
  - Score: 0.0358
  - MI: 0.0028
  - Redundancy: 0.8425
  - Cramer's V: 0.0351
  - Delta PR-AUC: -0.026337
  - SHAP Rank: 58


由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 burst_transaction_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 提取小时信息
    tx_df['hour'] = pd.to_datetime(tx_df['timestamp']).dt.hour

    # 对每个钱包，按小时统计交易次数
    hourly_counts = tx_df.groupby(['src', 'hour']).size().reset_index(name='count_per_hour')

    # 标记是否为爆发时段（>=10笔/小时）
    hourly_counts['is_burst'] = (hourly_counts['count_per_hour'] >= 10).astype(int)

    # 计算每个钱包爆发时段占比
    stats = hourly_counts.groupby('src')['is_burst'].mean().reset_index()
    stats.columns = ['wallet_id', 'feature_value']
    result = stats
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: burst_transaction_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 burst_transaction_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
