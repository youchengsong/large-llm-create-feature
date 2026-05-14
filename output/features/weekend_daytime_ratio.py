"""
特征: weekend_daytime_ratio
说明: 正常用户倾向于在工作日白天进行交易，洗钱账户往往在深夜或非正常时段交易，周末白天交易占比低可能反映账户使用行为异常。

验证指标:
  - Score: 0.6440
  - MI: 0.0010
  - Redundancy: 0.2395
  - Cramer's V: 0.0151
  - Delta PR-AUC: -0.025851
  - SHAP Rank: 16

可解释性报告:
  - 值含义: {'0-0.15': '异常可疑：周末白天交易占比极低，表明账户几乎不在正常时段活动，符合洗钱账户利用深夜或非工作日交易的模式', '0.15-0.19': '偏异常：周末白天交易占比较低，可能存在异常行为倾向，需结合其他特征进一步判断', '0.19-0.25': '正常范围：周末白天交易占比处于正常波动区间，覆盖大部分正常用户的交易习惯', '0.25-0.35': '偏正常但需关注：周末白天交易占比偏高，可能是特定场景（如周末集中消费型用户），需排除误判', '0.35-1.0': '高度可疑：周末白天交易占比极高，可能对应特定洗钱模式（如商户收款码跑分集中在周末白天完成交易）'}
  - 风险阈值: {'low_risk': 0.25, 'high_risk': 0.15, 'rationale': '异常交易的25%分位为0.1744，正常交易的25%分位为0.1979，两者差距明显。选择0.15作为高风险阈值，因为该值低于异常交易25%分位，能有效捕捉极端异常行为；选择0.25作为低风险阈值，因为正常交易的75%分位为0.2472，高于该值可能对应特定正常场景但仍有可疑空间'}
  - 协同特征: [{'feature': 'Timestamp_Hour', 'reason': '该特征捕捉具体交易时段分布（如凌晨占比），与weekend_daytime_ratio结合可区分是周末白天占比低但集中在工作日白天（正常）还是集中在深夜（异常）'}, {'feature': 'activetime', 'reason': 'activetime反映账户活跃时间段，与weekend_daytime_ratio联合分析可判断账户是否在正常时段活跃但周末白天占比低（正常），还是整体活跃时段偏离正常（异常）'}, {'feature': 'transaction_frequency', 'reason': '高频交易账户如果weekend_daytime_ratio偏低，可能是线上跑分模式（凌晨高频交易）；如果偏高，可能是商户收款码跑分模式（集中在周末白天交易）'}, {'feature': 'avg_inactive_period', 'reason': '结合非活跃期时长，可以区分周末白天占比低是由于账户整体不活跃（正常休眠）还是刻意避开白天时段（异常）'}]
  - 误判场景: [{'scenario': '夜班工作者', 'explanation': '上夜班的用户（如医护人员、工厂工人）主要在夜间活动，周末白天可能用于休息，导致周末白天交易占比低'}, {'scenario': '周末休息型用户', 'explanation': '部分用户周末倾向于线下消费或外出，较少使用数字人民币，导致周末白天交易占比偏低'}, {'scenario': '节假日集中交易', 'explanation': '某些特定行业（如旅游、餐饮）在节假日（非周末）交易活跃，周末白天占比自然偏低'}, {'scenario': '少量交易账户', 'explanation': '交易笔数极少的账户（如仅1-2笔交易），周末白天占比可能极端为0或1，但不一定代表异常'}]
  - 监管说明: 该特征衡量账户在周末白天（8点-20点）的交易占比，正常用户的交易时间分布均匀，而洗钱账户往往集中在深夜或非正常时段活动，导致周末白天交易占比异常偏低或偏高，可作为识别异常交易时间模式的重要辅助指标。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 weekend_daytime_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 提取日期和时间信息
    tx = tx_df.copy()
    tx['timestamp'] = pd.to_datetime(tx['timestamp'])
    tx['hour'] = tx['timestamp'].dt.hour
    tx['weekday'] = tx['timestamp'].dt.weekday  # 0=周一, 6=周日

    # 定义周末（周六、周日）
    is_weekend = tx['weekday'].isin([5, 6])

    # 定义白天时段（8点-20点）
    is_daytime = tx['hour'].between(8, 20)

    # 标记是否为周末白天交易
    tx['is_weekend_daytime'] = is_weekend & is_daytime

    # 对每个钱包，计算周末白天交易占比
    wallet_tx_count = tx.groupby('src').size().reset_index(name='total_tx')
    weekend_daytime_count = tx[tx['is_weekend_daytime']].groupby('src').size().reset_index(name='weekend_daytime_tx')

    result = wallet_tx_count.merge(weekend_daytime_count, on='src', how='left')
    result['weekend_daytime_tx'] = result['weekend_daytime_tx'].fillna(0)
    result['weekend_daytime_ratio'] = result['weekend_daytime_tx'] / result['total_tx'].clip(lower=1)

    result = result[['src', 'weekend_daytime_ratio']]
    result.columns = ['wallet_id', 'weekend_daytime_ratio']
    result['weekend_daytime_ratio'] = result['weekend_daytime_ratio'].clip(0, 1)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: weekend_daytime_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 weekend_daytime_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
