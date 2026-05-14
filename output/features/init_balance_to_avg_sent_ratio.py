"""
特征: init_balance_to_avg_sent_ratio
说明: 休眠账户突然激活并进行小额测试交易是典型的风险模式，初始余额与交易金额的严重不匹配可以识别出这类账户。正常账户的初始余额和交易金额通常有一定关联，而异常账户可能初始余额很高但只进行小额测试。

验证指标:
  - Score: 0.2269
  - MI: 0.0011
  - Redundancy: 0.3579
  - Cramer's V: 0.0141
  - Delta PR-AUC: -0.013704
  - SHAP Rank: 40


由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 init_balance_to_avg_sent_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每个钱包的平均发送金额
    avg_sent = tx_df.groupby('src')['amount'].mean().reset_index()
    avg_sent.columns = ['wallet_id', 'avg_sent']

    # 合并初始余额
    result = open_df[['wallet_id', 'init_balance']].merge(avg_sent, on='wallet_id', how='left')
    result['avg_sent'] = result['avg_sent'].fillna(0)

    # 计算比例，避免除零
    result['feature_value'] = np.where(result['avg_sent'] > 0, result['init_balance'] / result['avg_sent'], 0)
    result = result[['wallet_id', 'feature_value']].copy()
    result = result.drop_duplicates(subset='wallet_id').reset_index(drop=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: init_balance_to_avg_sent_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 init_balance_to_avg_sent_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
