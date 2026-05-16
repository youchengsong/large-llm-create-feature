"""
特征: night_tx_ratio_src
说明: 线上跑分（模式3）在凌晨20时-1时高频交易，class4_laundering（模式6）在深夜0-1时、23时交易，small_amount_testing（模式4）在深夜0-5时、22-23时交易。深夜交易比例是识别这些洗钱模式的关键指标。

验证指标:
  - Score: 0.1897
  - MI: 0.0000
  - Redundancy: 0.6819
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 114


由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 night_tx_ratio_src，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 定义深夜时段（23时-5时，包含23时和5时）
    def is_deep_night(h):
        return (h >= 23) | (h < 6)

    # 标记每笔交易是否为深夜交易
    tx_df['is_deep_night'] = tx_df['hour'].apply(is_deep_night)

    # 计算每个钱包的深夜交易比例
    night_tx = tx_df.groupby('src')['is_deep_night'].agg(['sum', 'count']).reset_index()
    night_tx['night_tx_ratio_src'] = night_tx['sum'] / night_tx['count'].clip(lower=1)

    result = night_tx[['src', 'night_tx_ratio_src']].rename(columns={'src': 'wallet_id'})
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: night_tx_ratio_src")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 night_tx_ratio_src: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
