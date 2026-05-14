"""
特征: dst_overlap_ratio
说明: 洗钱模式中，资金常通过多个账户循环流转以混淆资金来源和去向，如线上跑分模式中资金分散转出后又回流。此特征可识别这种双向交易网络中的异常循环行为。

验证指标:
  - Score: 0.1848
  - MI: 0.0011
  - Redundancy: 0.4816
  - Cramer's V: 0.0391
  - Delta PR-AUC: -0.031416
  - SHAP Rank: 17

可解释性报告:
  - 值含义: {'0-0.15': '极低重叠率，钱包的资金流向高度单向，转出对手几乎不会向该钱包转回资金，符合正常单次交易或简单支付场景', '0.15-0.28': '低重叠率，存在少量双向资金往来，可能是亲友间互转或小型商业往来，但异常交易集中在此区间（均值0.23），需结合其他特征判断', '0.28-0.39': '中等重叠率，正常交易的中位数（0.39）附近，常见于日常消费场景中偶尔的回流行为', '0.39-0.51': '较高重叠率，正常交易的上四分位区间，多见于商户与固定客户之间的双向转账', '0.51-1.0': '高重叠率，钱包的大部分转出对手同时也是其收款方，可能涉及闭环洗钱或循环交易'}
  - 风险阈值: {'low_risk_cutoff': 0.15, 'high_risk_cutoff': 0.51, 'basis': '异常交易在0.15-0.28区间高度集中（均值0.23），而正常交易在0.28-0.51区间集中（均值0.39）。低于0.15表示资金单向流动（正常），高于0.51表示异常高的双向重叠（可能为洗钱循环）。实际应用中建议以0.15和0.51为双阈值，低于0.15视为正常，高于0.51视为高风险，0.15-0.28区间需结合其他特征'}
  - 协同特征: [{'feature': 'fan_out_ratio（分散转出比例）', 'reason': '线上跑分模式同时具有高分散转出（fan_out）和低重叠率特征，组合使用可区分洗钱与正常商户行为'}, {'feature': 'avg_inactive_period（平均休眠间隔）', 'reason': '休眠账户突然激活且重叠率异常低（<0.15），可识别small_amount_testing模式中的测试交易'}, {'feature': 'large_transaction_ratio（大额交易比例）', 'reason': '高重叠率+高比例大额交易可能指向商户收款码跑分模式中的佣金回流'}]
  - 误判场景: ['亲友间频繁互转：家庭成员或朋友之间经常互相转账（如AA制消费、红包互发），重叠率可能高于0.5，但属于正常社交行为', '小型商户与固定客户：早餐摊、便利店等与常客之间既有收款也有付款（如进货），重叠率可能偏高', '平台内循环交易：电商平台或社区团购中，用户既是买家也是卖家，双向交易属于正常商业模式']
  - 监管说明: 该特征衡量的是钱包转出给谁，以及谁又转回来给这个钱包的比例——简单说就是看资金是否在同一个圈子内来回流转，洗钱账户倾向于让资金单向快速流动减少回流，而正常商户则常有双向往来。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 dst_overlap_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每个钱包的转出对手集合
    src_dst = tx_df.groupby('src')['dst'].apply(set).reset_index()
    src_dst.columns = ['wallet_id', 'out_dst_set']

    # 计算每个钱包的转入对手集合
    dst_src = tx_df.groupby('dst')['src'].apply(set).reset_index()
    dst_src.columns = ['wallet_id', 'in_src_set']

    # 合并两个集合
    overlap = src_dst.merge(dst_src, on='wallet_id', how='outer')
    overlap['out_dst_set'] = overlap['out_dst_set'].fillna('').apply(lambda x: set() if x == '' else x)
    overlap['in_src_set'] = overlap['in_src_set'].fillna('').apply(lambda x: set() if x == '' else x)

    # 计算转出对手中同时也是自己收款方的比例
    overlap['overlap_count'] = overlap.apply(lambda row: len(row['out_dst_set'] & row['in_src_set']), axis=1)
    overlap['total_out'] = overlap['out_dst_set'].apply(len)
    overlap['feature_value'] = np.where(overlap['total_out'] > 0, overlap['overlap_count'] / overlap['total_out'], 0)

    result = overlap[['wallet_id', 'feature_value']].copy()
    result = result.dropna(subset=['feature_value'])
    result['feature_value'] = result['feature_value'].fillna(0)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: dst_overlap_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 dst_overlap_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
