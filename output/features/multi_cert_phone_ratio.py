"""
特征: multi_cert_phone_ratio
说明: 根据薅羊毛模式，批量注册4类钱包时，同一身份证/手机号会关联多个钱包。通过计算每个身份证关联的钱包数量与每个手机号关联的钱包数量的比值，可以识别出使用同一身份信息批量注册多个钱包的可疑行为。

验证指标:
  - Score: 0.5000
  - MI: 0.0126
  - Redundancy: 0.0000
  - Cramer's V: 0.0743
  - Delta PR-AUC: 0.009323
  - SHAP Rank: 1

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 multi_cert_phone_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    cert_phone = open_df[['wallet_id', 'wallet_open_cert', 'wallet_open_tel']].dropna(subset=['wallet_open_cert', 'wallet_open_tel'])

    # 计算每个身份证关联的钱包数
    cert_counts = cert_phone.groupby('wallet_open_cert')['wallet_id'].nunique().reset_index()
    cert_counts.columns = ['wallet_open_cert', 'cert_wallet_count']

    # 计算每个手机号关联的钱包数
    phone_counts = cert_phone.groupby('wallet_open_tel')['wallet_id'].nunique().reset_index()
    phone_counts.columns = ['wallet_open_tel', 'phone_wallet_count']

    # 合并到原数据
    merged = cert_phone.merge(cert_counts, on='wallet_open_cert', how='left')
    merged = merged.merge(phone_counts, on='wallet_open_tel', how='left')

    # 计算比值，避免除零
    merged['cert_phone_ratio'] = (merged['cert_wallet_count'] / merged['phone_wallet_count'].clip(lower=1)).fillna(0)

    result = merged[['wallet_id', 'cert_phone_ratio']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: multi_cert_phone_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 multi_cert_phone_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
