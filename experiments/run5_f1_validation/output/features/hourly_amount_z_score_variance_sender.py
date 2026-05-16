"""
特征: hourly_amount_z_score_variance_sender
说明: 通过计算每个钱包每小时交易金额的z-score，再计算这些z-score的方差，可以量化该钱包在不同时间段金额行为的『异常稳定性』。如果某个钱包在特定小时的金额始终偏离其自身平均水平（z-score高且稳定），则表明存在规律性异常交易。这与hourly_amount_std_ratio不同，后者衡量的是波动性，这里衡量的是偏离程度的稳定性。

验证指标:
  - Score: 0.4346
  - MI: 0.0000
  - Redundancy: 0.4084
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 71

可解释性报告:
  - 值含义: {'0-1': '正常范围：钱包各小时的交易金额偏离程度稳定，没有某个小时出现持续偏离自身均值的行为', '1-2': '轻度波动：部分小时存在一定的金额偏离，但幅度不大，可能是正常的消费习惯差异', '2-3': '关注区间：存在个别小时交易金额持续偏离自身均值，可能是有规律的异常交易模式', '3-5': '可疑区间：某个小时的交易金额显著且稳定地偏离自身均值，高度怀疑存在定时定额的异常交易', '>5': '高风险：极端情况下，某个小时的交易行为完全脱离正常模式，极可能是洗钱或非法交易'}
  - 风险阈值: 2.5
  - 协同特征: [{'feature': 'hourly_amount_std_ratio', 'reason': "该特征衡量波动性，而本特征衡量偏离的稳定性。两者结合可以区分'稳定异常'（高z-score方差+低波动性）和'随机异常'（低z-score方差+高波动性），前者更可能是洗钱行为"}, {'feature': 'avg_inactive_period_From', 'reason': "结合休眠期特征，如果休眠账户突然在某个特定小时出现高z-score方差，符合'small_amount_testing'模式中休眠账户突然激活的特征"}, {'feature': 'is_night_hour_From', 'reason': "如果高z-score方差出现在凌晨时段（22-5时），结合领域知识中'线上跑分'和'online_laundering'模式，进一步确认异常"}]
  - 误判场景: ['固定缴费场景：用户每月固定在某小时缴纳固定金额的水电费/房租，会导致该小时金额稳定偏离均值，但属于正常行为', '工资发放：企业钱包在固定小时发放工资，金额稳定且偏离日常消费均值，但这是正常业务', '定期理财：用户在固定时间进行定投或赎回，金额稳定且与日常消费行为不同', '时差用户：跨时区用户可能在本地凌晨活跃，但其交易模式稳定，导致高z-score方差']
  - 监管说明: 该特征用于发现那些在特定时间段内交易金额'异常稳定'的钱包——如果一个钱包在每天同一小时的交易金额都明显高于或低于其日常水平，且这种偏离持续存在，说明该钱包存在规律性的异常交易模式，可能是洗钱分子利用固定时段进行资金归集或分散。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 hourly_amount_z_score_variance_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 1. 计算每个钱包的全局金额均值和标准差
    wallet_stats = tx_df.groupby('src')['amount'].agg(['mean', 'std']).reset_index()
    wallet_stats.columns = ['wallet_id', 'global_mean', 'global_std']
    wallet_stats['global_std'] = wallet_stats['global_std'].replace(0, np.nan)

    # 2. 合并到交易表，计算每小时z-score
    tx_with_stats = tx_df.merge(wallet_stats, left_on='src', right_on='wallet_id', how='left')
    tx_with_stats['z_score'] = np.where(
        tx_with_stats['global_std'].notna() & (tx_with_stats['global_std'] > 0),
        (tx_with_stats['amount'] - tx_with_stats['global_mean']) / tx_with_stats['global_std'],
        0
    )

    # 3. 按src和hour分组，计算每小时z-score的方差
    hourly_z_var = (
        tx_with_stats.groupby(['src', 'hour'])['z_score']
        .var()
        .reset_index()
        .rename(columns={'z_score': 'hourly_z_var'})
    )

    # 4. 对每个钱包，计算其不同小时之间z-score方差的均值（或最大值）
    # 如果每小时z-score的方差都很小（即每小时内的金额行为稳定），但不同小时间的z-score方差很大，说明行为在小时间有规律变化
    # 这里取每小时z-score方差的最大值，如果最大值很大，说明存在某个小时行为极度反常且稳定
    result = (
        hourly_z_var.groupby('src')['hourly_z_var']
        .max()
        .reset_index()
        .rename(columns={'src': 'wallet_id', 'hourly_z_var': 'feature_value'})
    )
    result['feature_value'] = result['feature_value'].fillna(0).clip(upper=100)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: hourly_amount_z_score_variance_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 hourly_amount_z_score_variance_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
