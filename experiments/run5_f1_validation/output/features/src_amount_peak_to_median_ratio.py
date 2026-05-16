"""
特征: src_amount_peak_to_median_ratio
说明: 线上跑分和商户跑分模式中，交易金额集中在10-1000元（异常中位数更高），且因模式固定，峰值与中位数接近；正常消费有少量大额拉高均值，导致峰值与中位数差异大。峰值/中位数比值能捕捉这种分布偏态差异。

验证指标:
  - Score: 0.3441
  - MI: 0.0000
  - Redundancy: 0.4466
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 91

可解释性报告:
  - 值含义: 该特征表示钱包交易金额的峰值（最常出现的金额区间中点）与中位数的比值。值接近1表示峰值与中位数相近，说明金额分布集中、模式固定（如跑分交易）；值远大于1或小于1表示金额分布偏态，峰值与中位数差异大（如正常消费有少量大额拉高均值或大量小额拉低中位数）。正常交易中位数0.42，均值0.95，说明多数正常钱包峰值显著低于中位数（小额高频消费）；异常交易中位数0.38，均值0.81，说明异常钱包峰值更低（金额更集中在小额区间）。
  - 风险阈值: 高风险阈值建议设为<0.15或>2.0。<0.15表示峰值金额远低于中位数（如金额集中在0-500元区间但中位数被少量大额拉高），符合线上跑分（10-1000元）和商户跑分（10-1000元）中金额集中在小额区间的特征；>2.0表示峰值金额远高于中位数（如金额集中在1000-1500元区间但中位数被大量小额拉低），符合洗钱模式中目标金额集中的特征。正常交易25%分位为0.15，75%分位为0.95，因此<0.15和>0.95可作为初步筛选阈值。
  - 协同特征: 建议与以下特征配合使用：1）交易频率（transaction_frequency）：高频率+低峰值/中位数比值，符合线上跑分高频小额特征；2）fan_out模式占比（send_concentration）：高扇出+低比值，符合分散转出模式；3）凌晨交易占比（is_night_hour）：凌晨交易+低比值，符合夜间跑分特征。协同效果：组合可区分正常小额高频消费（如早高峰早餐）与跑分交易（凌晨+分散+金额集中）。
  - 误判场景: 以下正常场景可能误判：1）固定金额缴费（如每月固定房租1000元），导致峰值与中位数接近，比值接近1；2）单一定向消费（如每天买同一商品），金额分布高度集中；3）钱包刚激活且只有少量交易（如3笔金额均为500元），峰值与中位数相等。这些场景下正常用户也会显示低比值，需结合交易时间、对手多样性等特征排除。
  - 监管说明: 该指标衡量钱包的交易金额是否集中在某个固定数值附近——如果金额总是集中在某个小范围内（比如总是转100元或500元），说明交易模式固定、缺乏真实消费的随机性，可能是洗钱分子在按固定金额跑分；反之，金额高低差异大、分布分散，更符合正常用户消费习惯。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 src_amount_peak_to_median_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    # 计算每个src钱包的金额峰值（众数区间中点）和金额中位数
    # 使用直方图近似峰值（选择出现次数最多的金额区间）
    bins = np.arange(0, tx_df['amount'].max() + 500, 500)  # 500元为区间宽度

    # 为每笔交易分配区间编号
    tx_df['amount_bin_500'] = pd.cut(tx_df['amount'], bins=bins, labels=False, right=False)

    # 计算每个src的众数区间（峰值区间）
    mode_bin = tx_df.groupby('src')['amount_bin_500'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index()
    mode_bin.columns = ['wallet_id', 'mode_bin']

    # 计算每个src的金额中位数
    medians = tx_df.groupby('src')['amount'].median().reset_index()
    medians.columns = ['wallet_id', 'amount_median']

    # 合并
    stats = mode_bin.merge(medians, on='wallet_id')

    # 将区间编号转换为区间中点（近似峰值金额）
    stats['peak_amount'] = stats['mode_bin'] * 500 + 250  # 区间中点
    stats['peak_to_median_ratio'] = stats['peak_amount'] / (stats['amount_median'] + 1e-10)
    stats['peak_to_median_ratio'] = stats['peak_to_median_ratio'].clip(0, 100)  # 限制异常值

    result = stats[['wallet_id', 'peak_to_median_ratio']].fillna(1.0)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: src_amount_peak_to_median_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 src_amount_peak_to_median_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
