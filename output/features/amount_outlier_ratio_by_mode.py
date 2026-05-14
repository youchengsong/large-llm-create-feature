"""
特征: amount_outlier_ratio_by_mode
说明: 异常交易中small_amount_testing金额极低(0.01-0.1元)，而正常交易金额较高。通过比较同一钱包在不同模式下的金额分布，可以识别出在正常模式下金额异常偏低的钱包。

验证指标:
  - Score: 0.7063
  - MI: 0.0011
  - Redundancy: 0.2626
  - Cramer's V: 0.0792
  - Delta PR-AUC: 0.053376
  - SHAP Rank: 6

可解释性报告:
  - 值含义: {'0': '钱包在非正常模式下没有出现金额异常的交易，或没有非正常模式交易', '(0, 0.5)': '钱包在部分非正常模式下出现金额异常，但比例不高，可能属于正常用户的偶尔异常行为', '[0.5, 1)': '钱包在超过一半的非正常模式交易中金额异常，存在洗钱嫌疑', '1': '钱包在所有非正常模式交易中金额都异常，强烈指向洗钱行为'}
  - 风险阈值: 0.5
  - 协同特征: [{'feature': 'activetime', 'reason': 'amount_outlier_ratio_by_mode识别金额异常，activetime识别交易时间异常（如凌晨交易），两者结合可覆盖洗钱行为的两个关键维度：金额和时间异常。'}, {'feature': 'In_Degree', 'reason': '金额异常的钱包常伴随fan_in模式（异常中22.2% vs 正常1.5%），In_Degree高说明接收大量交易，结合金额异常可识别跑分钱包。'}, {'feature': 'From_To_Ratio', 'reason': '洗钱钱包通常转出远多于转入（fan_out模式），From_To_Ratio高，结合金额异常可确认是主动转出异常金额而非被动接收。'}]
  - 误判场景: [{'scenario': '正常用户偶尔进行小额测试交易', 'explanation': '用户首次使用数字人民币或更换设备时，可能进行0.01元测试，导致在非正常模式下金额异常，但交易次数少，特征值可能偏高。'}, {'scenario': '正常商户钱包有真实小额退款或找零', 'explanation': '商户钱包可能产生小额退款交易（如0.5元），若被归类为非正常模式，会误判为金额异常。'}, {'scenario': '数据标注不准确', 'explanation': '部分正常交易被错误标注为small_amount_testing或merchant_laundering等风险模式，导致正常钱包在非正常模式下金额异常。'}]
  - 监管说明: 该特征用于检测那些在非正常交易模式（如小额测试、跑分）中，交易金额与自己在正常交易模式下的金额习惯严重不符的钱包，例如一个平时转账几百元的人突然频繁进行0.01元的测试交易。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_outlier_ratio_by_mode，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 定义正常交易模式
    normal_modes = ['single_transaction', 'normal_small_high_freq', 'regular_large_low_freq']

    # 标记交易是否属于正常模式
    tx_df['is_normal_mode'] = tx_df['transaction_motif'].isin(normal_modes)

    # 计算每个钱包在正常模式下的金额统计
    normal_stats = tx_df[tx_df['is_normal_mode']].groupby('src')['amount'].agg(['mean', 'std']).reset_index()
    normal_stats.columns = ['wallet_id', 'normal_mean', 'normal_std']
    normal_stats['normal_std'] = normal_stats['normal_std'].fillna(0).clip(lower=1)

    # 计算每个钱包在非正常模式下的金额
    abnormal_amounts = tx_df[~tx_df['is_normal_mode']].groupby('src')['amount'].agg(['mean', 'count']).reset_index()
    abnormal_amounts.columns = ['wallet_id', 'abnormal_mean', 'abnormal_count']

    # 合并
    merged = pd.merge(normal_stats, abnormal_amounts, on='wallet_id', how='outer')
    merged['abnormal_mean'] = merged['abnormal_mean'].fillna(0)
    merged['abnormal_count'] = merged['abnormal_count'].fillna(0)

    # 计算异常比例：非正常模式交易中金额低于正常均值-1倍标准差的占比
    # 先计算每个钱包每笔非正常交易是否异常
    abnormal_tx = tx_df[~tx_df['is_normal_mode']].copy()
    abnormal_tx = abnormal_tx.merge(normal_stats[['wallet_id', 'normal_mean', 'normal_std']], left_on='src', right_on='wallet_id', how='left')
    abnormal_tx['normal_mean'] = abnormal_tx['normal_mean'].fillna(abnormal_tx['amount'].mean())
    abnormal_tx['normal_std'] = abnormal_tx['normal_std'].fillna(abnormal_tx['amount'].std())
    abnormal_tx['normal_std'] = abnormal_tx['normal_std'].clip(lower=1)

    # 标记是否为金额异常（低于正常均值-1倍标准差）
    abnormal_tx['is_amount_outlier'] = (abnormal_tx['amount'] < (abnormal_tx['normal_mean'] - abnormal_tx['normal_std'])) | \
                                       (abnormal_tx['amount'] > (abnormal_tx['normal_mean'] + abnormal_tx['normal_std']))

    # 计算每个钱包的异常比例
    outlier_ratio = abnormal_tx.groupby('src')['is_amount_outlier'].mean().reset_index()
    outlier_ratio.columns = ['wallet_id', 'feature_value']

    # 填充没有非正常交易的钱包
    all_wallets = open_df[['wallet_id']].copy()
    result = all_wallets.merge(outlier_ratio, on='wallet_id', how='left')
    result['feature_value'] = result['feature_value'].fillna(0)

    result = result[['wallet_id', 'feature_value']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_outlier_ratio_by_mode")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_outlier_ratio_by_mode: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
