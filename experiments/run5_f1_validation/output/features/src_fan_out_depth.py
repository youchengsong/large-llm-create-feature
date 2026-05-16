"""
特征: src_fan_out_depth
说明: 异常交易中fan_out模式占比高，且领域知识中'线上跑分'模式明确提到'分散转出(fan_out)模式'。通过计算每个钱包的对手中，有多少也表现出fan_out行为（即对手也有多个下游），可以捕捉多层分散的资金链路特征。

验证指标:
  - Score: 0.4838
  - MI: 0.0000
  - Redundancy: 0.1652
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 103

可解释性报告:
  - 值含义: 该特征表示一个钱包在资金流出时，其下游接收方（直接交易对手）作为发送方时的平均下游分散程度。具体来说：特征值约为13-14表示中等水平的资金分散深度，值越高说明资金经过该钱包后，下游钱包继续向外分散的程度越深，资金链路越长。正常交易和异常交易的中位数分别为13.68和13.53，差异很小，说明该特征在区分正常与异常方面能力有限。
  - 风险阈值: 由于正常与异常分布的差异极小（KS=0.0318），没有明显的自然分界点。建议不使用单一阈值判断，而是将特征值>15.32（异常交易的75分位）作为辅助参考阈值，此时仅有约25%的异常交易被覆盖，但同时也会有约25%的正常交易被误判。该阈值的选择依据是异常交易的上四分位数，但实际区分能力很弱。
  - 协同特征: 建议与以下特征配合使用：1) 'transaction_mode'（交易模式），直接识别fan_out模式；2) 'send_concentration'（发送集中度），高分散深度+低发送集中度更符合跑分模式；3) 'hour'（交易小时），结合凌晨交易时段可增强对线上跑分模式的识别。这些特征的组合能弥补该特征单独区分能力不足的问题。
  - 误判场景: 1) 正常的大型商户钱包，其下游可能是众多分销商或零售商，这些下游钱包也会继续向下游支付，导致较高的分散深度；2) 正常的资金归集场景，如电商平台收款后向众多供应商付款，形成多层分散结构；3) 正常的工资代发场景，企业向员工发薪，员工再向各自供应商付款。
  - 监管说明: 该指标衡量的是资金经过某个钱包后，下游钱包继续向外转账的活跃程度——如果某个钱包转出的钱，接收方也频繁转给其他人，说明资金在多个层级间快速流动，需要关注是否存在隐蔽的资金转移链条。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 src_fan_out_depth，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每个钱包作为src时的fan_out程度（下游对手数）
    src_fanout = tx_df.groupby('src')['dst'].nunique().reset_index()
    src_fanout.columns = ['wallet_id', 'fan_out_count']

    # 计算每个钱包作为dst时，其上游src的fan_out程度
    # 即：对于每笔交易，dst的上游src的fan_out_count
    # 先建立src→fan_out_count的映射
    fanout_dict = dict(zip(src_fanout['wallet_id'], src_fanout['fan_out_count']))

    # 对每笔交易，获取src的fan_out_count
    tx_df['src_fanout'] = tx_df['src'].map(fanout_dict)

    # 对每个dst，计算其所有上游src的fan_out的均值（上游分散程度）
    dst_upstream_fanout = tx_df.groupby('dst')['src_fanout'].mean().reset_index()
    dst_upstream_fanout.columns = ['wallet_id', 'avg_upstream_fanout']

    # 对每个src，计算其所有下游dst的avg_upstream_fanout的均值
    # 即：src转给dst，dst再转给其他人，dst的'上游分散度'的平均值
    # 但这里我们直接计算每个src的下游dst的'下游fanout'（即dst作为src时的fan_out_count）
    # 这需要dst→fan_out_count的映射
    dst_fanout = tx_df.groupby('dst')['src'].nunique().reset_index()
    dst_fanout.columns = ['wallet_id', 'fan_in_count']

    # 合并：dst作为src时的fan_out
    all_wallets = pd.DataFrame({'wallet_id': pd.concat([tx_df['src'], tx_df['dst']]).unique()})
    all_wallets = all_wallets.merge(src_fanout, on='wallet_id', how='left')
    all_wallets = all_wallets.merge(dst_fanout, on='wallet_id', how='left')

    # 对每个src，计算其下游dst的fan_out均值
    # 即：src转给的那些dst，它们自身作为src时的fan_out有多大
    tx_with_downstream = tx_df.merge(all_wallets[['wallet_id', 'fan_out_count']], left_on='dst', right_on='wallet_id', how='left')
    tx_with_downstream = tx_with_downstream.rename(columns={'fan_out_count': 'dst_fanout_as_src'})

    downstream_depth = tx_with_downstream.groupby('src')['dst_fanout_as_src'].mean().reset_index()
    downstream_depth.columns = ['wallet_id', 'src_fan_out_depth']
    downstream_depth['src_fan_out_depth'] = downstream_depth['src_fan_out_depth'].fillna(0)

    result = downstream_depth[['wallet_id', 'src_fan_out_depth']].copy()
    result.columns = ['wallet_id', 'feature_value']
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: src_fan_out_depth")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 src_fan_out_depth: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
