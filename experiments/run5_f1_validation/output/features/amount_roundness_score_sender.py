"""
特征: amount_roundness_score_sender
说明: 洗钱行为中，交易金额常呈现'整数'特征（如10000、5000、2000等）或接近钱包限额（如4类钱包日限额5000）。通过分析金额的整数程度，可识别异常的资金安排模式。

验证指标:
  - Score: 0.1967
  - MI: 0.0000
  - Redundancy: 0.3108
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 129


由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_roundness_score_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每笔交易的金额整数度
    tx_df['amount_abs'] = tx_df['amount'].abs()

    # 整数度指标1：金额是否是100的整数倍（接近整数）
    tx_df['is_round_hundred'] = (tx_df['amount_abs'] % 100 == 0).astype(float)

    # 整数度指标2：金额是否是1000的整数倍
    tx_df['is_round_thousand'] = (tx_df['amount_abs'] % 1000 == 0).astype(float)

    # 整数度指标3：金额与整数的最小距离（归一化）
    tx_df['roundness_score'] = 1 - np.minimum(
        tx_df['amount_abs'] % 100, 
        100 - (tx_df['amount_abs'] % 100)
    ) / 50  # 0-1之间，越接近1越整数

    # 整数度指标4：金额是否接近常见限额（5000, 10000, 20000, 50000）
    limits = [5000, 10000, 20000, 50000]
    for limit in limits:
        col_name = f'close_to_{limit}'
        tx_df[col_name] = (np.abs(tx_df['amount_abs'] - limit) / limit < 0.05).astype(float)

    tx_df['is_close_to_limit'] = tx_df[[f'close_to_{l}' for l in limits]].max(axis=1)

    # 综合整数度得分
    tx_df['combined_roundness'] = (
        tx_df['roundness_score'] * 0.3 +
        tx_df['is_round_hundred']
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_roundness_score_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_roundness_score_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
