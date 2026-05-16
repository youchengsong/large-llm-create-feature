"""
特征: dst_region_shift_entropy_sender
说明: 洗钱者常通过跨地域分散转账规避地域监控，对手地域熵值能有效捕捉这种异常模式。与send_concentration不同，它关注地域多样性而非金额集中度。

验证指标:
  - Score: 0.0289
  - MI: 0.0000
  - Redundancy: 0.5117
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 94

可解释性报告:
  - 值含义: 该特征衡量一个钱包向其转账的所有对手方所在地域（region_code）的多样性程度，取值范围为[0, ~4.0]。值越接近0表示所有对手都集中在同一地域（多样性低），值接近2.0表示对手分布在2个左右的地域，值接近3.5~4.0表示对手广泛分布在多个不同地域（多样性高）。正常交易的均值约3.66，异常交易的均值约3.70，表明异常交易的地域多样性略高但差异不显著。
  - 风险阈值: 建议高风险阈值为≥3.90（即异常交易上四分位3.9967附近取整）。依据：异常交易的上四分位为3.9967，正常交易的上四分位为3.9843，两者几乎重叠，说明该特征在高值区域区分能力极弱。实际上，该特征的中位数和均值差异仅约0.038，远小于标准差（0.5左右），表明正常与异常交易在地域多样性上高度相似。因此该特征不适合单独设定高风险阈值，建议仅作为辅助信号使用。
  - 协同特征: 建议与以下特征配合使用：
1. `send_concentration`（金额集中度）—— 若对手地域高度分散（熵值高）但金额集中在少数对手（send_concentration高），则可能是洗钱者通过分散地域但集中金额到少数账户的异常模式。
2. `transaction_frequency`（交易频率）—— 地域熵值高且交易频率异常高的钱包，可能存在线上跑分行为。
3. `avg_inactive_period`（平均不活跃间隔）—— 地域熵值高且不活跃间隔突然缩短（休眠账户激活），符合small_amount_testing模式。
  - 误判场景: 1. 电商平台商户钱包：商户可能向全国各地的客户发货，对手地域自然分布广泛，熵值会偏高。
2. 跨区域经营的企业钱包：如物流公司、连锁零售企业，其交易对手来自多个省份，熵值正常偏高。
3. 个人钱包在旅游/出差期间：短期内向多个不同城市的商户付款，熵值会暂时升高。
4. 使用数字人民币进行线上购物：电商平台聚集了全国用户，个人钱包向多个不同地域的商家付款是正常行为。
  - 监管说明: 该特征用于识别转账对手是否散布在多个不同地区——如果一个人把钱转给很多不同地方的人，有可能是洗钱团伙在通过跨地域分散转账来规避监测，但正常做生意的商户也会有类似行为，需要结合其他特征综合判断。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 dst_region_shift_entropy_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    tx_with_region = tx_df.merge(open_df[['wallet_id', 'region_code']].rename(columns={'wallet_id': 'dst', 'region_code': 'dst_region'}), on='dst', how='left')
    tx_with_region['dst_region'] = tx_with_region['dst_region'].fillna(-1)
    region_counts = tx_with_region.groupby(['src', 'dst_region']).size().reset_index(name='count')
    src_total = region_counts.groupby('src')['count'].sum().reset_index(name='total')
    region_pct = region_counts.merge(src_total, on='src')
    region_pct['pct'] = region_pct['count'] / region_pct['total']
    entropy = region_pct.groupby('src').apply(lambda x: -np.sum(x['pct'] * np.log2(x['pct'] + 1e-10))).reset_index(name='feature_value')
    entropy.columns = ['wallet_id', 'feature_value']
    result = entropy
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: dst_region_shift_entropy_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 dst_region_shift_entropy_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
