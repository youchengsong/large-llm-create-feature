"""
特征: cert_phone_multi_wallet_count
说明: 在薅羊毛洗钱模式中，犯罪分子常使用同一身份证或手机号批量注册多个4类钱包。统计每个钱包的开户证件(open_cert)和手机号(open_tel)关联的钱包数量，可以识别出批量开户的异常账户。

验证指标:
  - Score: 0.4250
  - MI: 0.0353
  - Redundancy: 0.0296
  - Cramer's V: 0.0835
  - Delta PR-AUC: 0.005142
  - SHAP Rank: 1

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 cert_phone_multi_wallet_count，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 统计每个开户证件关联的钱包数
    cert_counts = open_df.groupby('wallet_open_cert')['wallet_id'].nunique().reset_index()
    cert_counts.columns = ['wallet_open_cert', 'cert_wallet_count']

    # 统计每个手机号关联的钱包数
    tel_counts = open_df.groupby('wallet_open_tel')['wallet_id'].nunique().reset_index()
    tel_counts.columns = ['wallet_open_tel', 'tel_wallet_count']

    # 合并到开户表
    result = open_df[['wallet_id', 'wallet_open_cert', 'wallet_open_tel']].copy()
    result = result.merge(cert_counts, on='wallet_open_cert', how='left')
    result = result.merge(tel_counts, on='wallet_open_tel', how='left')

    # 取两个计数的最大值作为最终特征
    result['cert_phone_multi_wallet_count'] = result[['cert_wallet_count', 'tel_wallet_count']].max(axis=1)
    result = result[['wallet_id', 'cert_phone_multi_wallet_count']]
    result['cert_phone_multi_wallet_count'] = result['cert_phone_multi_wallet_count'].fillna(1)

    result = result
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: cert_phone_multi_wallet_count")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 cert_phone_multi_wallet_count: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
