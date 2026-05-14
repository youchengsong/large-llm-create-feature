"""
特征: day_night_tx_ratio_change
说明: 休眠账户突然激活并在深夜进行测试性交易（small_amount_testing）是典型的洗钱前兆行为。通过计算白天/夜间交易比例的变化率，可以捕捉钱包行为模式的突变，这是行为序列异常的重要指标。

验证指标:
  - Score: 0.2216
  - MI: 0.0011
  - Redundancy: 0.6419
  - Cramer's V: 0.0177
  - Delta PR-AUC: -0.012493
  - SHAP Rank: 19


由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 day_night_tx_ratio_change，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    # 提取小时信息
    tx_df['hour'] = pd.to_datetime(tx_df['timestamp']).dt.hour

    # 定义白天(9-21点)和夜间(22-8点)
    def is_day(hour):
        return 9 <= hour <= 21

    def is_night(hour):
        return hour >= 22 or hour <= 8

    # 计算每个钱包在白天和夜间的交易数量
    tx_df['is_day'] = tx_df['hour'].apply(is_day)
    tx_df['is_night'] = tx_df['hour'].apply(is_night)

    # 按钱包分组计算白天和夜间交易数
    day_count = tx_df[tx_df['is_day']].groupby('src').size().reset_index(name='day_count')
    night_count = tx_df[tx_df['is_night']].groupby('src').size().reset_index(name='night_count')

    # 合并
    ratio_change = day_count.merge(night_count, on='src', how='outer').fillna(0)

    # 计算比例变化：如果某钱包只有白天交易，ratio_change=1；只有夜间交易，ratio_change=-1
    ratio_change['day_night_ratio'] = ratio_change['day_count'] / (ratio_change['day_count'] + ratio_change['night_count'] + 1e-10)

    # 计算变化率：将比例与全局平均比例比较
    global_day_ratio = ratio_change['day_count'].sum() / (ratio_change['day_count'].sum() + ratio_change['night_count'].sum() + 1e-10)
    ratio_change['feature_value'] = np.abs(ratio_change['day_night_ratio'] - global_day_ratio)

    result = ratio_change[['src', 'feature_value']].rename(columns={'src': 'wallet_id'})
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: day_night_tx_ratio_change")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 day_night_tx_ratio_change: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
