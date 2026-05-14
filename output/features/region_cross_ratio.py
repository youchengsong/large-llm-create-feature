"""
特征: region_cross_ratio
说明: 洗钱模式中，线上跑分和商户收款码跑分都涉及大量跨地域交易。异常交易中 fan_out 和 fan_in 模式占比高，这些模式天然涉及跨地域资金流动。跨region交易比例是直接衡量跨地域行为的指标。

验证指标:
  - Score: 0.5672
  - MI: 0.0009
  - Redundancy: 0.1103
  - Cramer's V: 0.0164
  - Delta PR-AUC: -0.011079
  - SHAP Rank: 18

可解释性报告:
  - 值含义: {'0-0.95': '低跨地域交易比例，表示该钱包绝大部分交易发生在同一地区，属于本地化交易模式，常见于日常消费场景。', '0.95-0.97': '正常跨地域交易比例，与正常交易中位数(0.9628)接近，表明该钱包有一定比例的跨地区交易，但仍在正常范围内。', '0.97-0.99': '较高跨地域交易比例，异常交易中位数(0.9744)落在此区间，表示该钱包跨地区交易占比偏高，需关注是否存在线上跑分或商户收款码跑分行为。', '0.99-1.0': '极高跨地域交易比例，几乎全部交易均为跨地区，与洗钱模式中跨地域资金归集或分散的特征高度吻合，需重点排查。'}
  - 风险阈值: 0.97
  - 协同特征: [{'feature': 'Out_Degree', 'reason': '跨地域交易比例高 + 出度大，符合线上跑分(分散转出fan_out)的特征，两者结合可有效识别资金分散转移行为。'}, {'feature': 'In_Degree', 'reason': '跨地域交易比例高 + 入度大，符合商户收款码跑分(大量收款后转出)的特征，可识别资金归集后再分散的洗钱模式。'}, {'feature': 'Timestamp_Hour', 'reason': '跨地域交易比例高 + 凌晨交易占比高，符合线上跑分(20时-1时)和class4_laundering(0-1,23点)的时间特征，显著提升检测精度。'}, {'feature': 'avg_inactive_period', 'reason': '跨地域交易比例高 + 休眠账户突然激活(avg_inactive_period很大)，符合small_amount_testing模式的典型特征，可识别测试性交易。'}]
  - 误判场景: ['跨省电商卖家：正常经营的电商卖家可能频繁向全国各地的客户发货收款，导致跨地域交易比例极高(接近1.0)，但交易金额和频率符合正常经营规律。', '异地务工人员：在外地工作的人员可能频繁向家乡汇款，或接收来自家乡的转账，跨地域交易比例高但属于正常资金往来。', '旅游城市商户：在旅游热点城市的商户，交易对手来自全国各地，跨地域交易比例高但属于正常商业活动。', '跨区域企业：集团型企业内部资金调拨可能涉及多个地区，跨地域交易比例高但具有真实贸易背景。']
  - 监管说明: 该特征衡量一个钱包的交易对手是否遍布全国各地——正常人的钱大多花在本地，而洗钱团伙为了躲避监管，刻意将资金在不同地区间频繁转移，导致其交易几乎全是跨省的。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 region_cross_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    # 合并region信息到交易表
    tx_with_region = tx_df.merge(open_df[['wallet_id', 'region_code']], left_on='src', right_on='wallet_id', how='left')
    tx_with_region = tx_with_region.merge(open_df[['wallet_id', 'region_code']], left_on='dst', right_on='wallet_id', how='left', suffixes=('_src', '_dst'))
    # 判断是否跨region
    tx_with_region['is_cross_region'] = (tx_with_region['region_code_src'] != tx_with_region['region_code_dst']).astype(int)
    # 计算每个src钱包的跨region交易比例
    cross_ratio = tx_with_region.groupby('src')['is_cross_region'].mean().reset_index()
    cross_ratio.columns = ['wallet_id', 'region_cross_ratio']
    # 补充没有交易的钱包
    all_wallets = open_df[['wallet_id']].copy()
    result = all_wallets.merge(cross_ratio, on='wallet_id', how='left').fillna(0)
    result = result[['wallet_id', 'region_cross_ratio']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: region_cross_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 region_cross_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
