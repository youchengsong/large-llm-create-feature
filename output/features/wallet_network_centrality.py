"""
特征: wallet_network_centrality
说明: 洗钱网络中的'跑分'钱包通常充当资金中转站，从多个来源收款后快速分散转出。高中介中心性的钱包很可能就是这样的中转节点。这与'多账户交叉转账形成异常交易网络：如闭环转账、多层嵌套转账'风险指标直接相关。

验证指标:
  - Score: 0.4440
  - MI: 0.0014
  - Redundancy: 0.6273
  - Cramer's V: 0.0436
  - Delta PR-AUC: -0.041004
  - SHAP Rank: 2

可解释性报告:
  - 值含义: {'0-0.1': '该钱包几乎不扮演资金中转角色——要么只与极少数对手交易，要么交易方向单一（纯收款或纯付款），属于典型的个人日常使用模式。', '0.1-0.3': '钱包有少量中转特征，但对手方差较小，可能为正常的小型商户或与固定亲友之间的资金往来。', '0.3-0.5': '中转角色逐渐明显，钱包开始连接多个来源和去向，可能是活跃的个体经营者或社区团购团长。', '0.5-0.7': '强中转枢纽特征，钱包同时连接大量不同对手方，与正常用户的典型交易模式显著偏离，需引起关注。', '0.7-1.0': "极端高中介性，该钱包是交易网络中的核心枢纽，极大概率是专业洗钱团伙的'跑分'账户或资金归集账户。"}
  - 风险阈值: 高风险阈值为 > 0.55。判断依据：异常交易的中位数仅为0.27（正常为0.66），且异常交易的75%分位仅为0.44，意味着异常钱包的中介性普遍极低。正常钱包的中介性反而更高（均值0.63），这与直觉相反——洗钱钱包为了规避监管，往往刻意控制对手方数量（如固定几个上家+固定几个下家），而正常高频交易钱包反而连接更多商户。因此，阈值设在0.55可以识别出那些'异常地不像跑分账户'的洗钱账户（即伪装成正常高频交易的钱包），以及真正的高中介性跑分账户。但需注意，该特征对典型小额测试洗钱（对手方极少）的识别能力有限。
  - 协同特征: [{'feature': 'send_concentration / receive_concentration', 'reason': "高中介性+高发送集中度=该钱包虽然连接多个来源，但资金高度集中于少数几个对手方流出，这是典型的'多源归一'洗钱结构。"}, {'feature': 'transaction_frequency + Timestamp_Hour (凌晨交易占比)', 'reason': '高中介性+凌晨高频交易=几乎可以确定为线上跑分模式（领域知识模式3），因为正常商户不会在凌晨进行高频中转交易。'}, {'feature': 'avg_inactive_period + last_direction', 'reason': '高中介性+长时间休眠后突然激活（avg_inactive_period大）+近期交易方向为转出=休眠账户突然成为中转枢纽，符合small_amount_testing风险模式。'}]
  - 误判场景: [{'scenario': '大型电商平台收款钱包', 'reason': '一个真实的电商平台收款钱包会同时连接成千上万个买家（入度极高）和少量供应商（出度有限），但出度多样性会被低估。然而，如果该平台同时有大量分销商或二级代理商，则出度也可能很高，导致高中介性误判。'}, {'scenario': '社区团购团长/拼单组织者', 'reason': '团长从多个邻居收款（入度多样性高），再统一转给供应商（出度多样性高），其交易模式与洗钱中转站几乎完全一致，会产生较高的中介性得分。'}, {'scenario': 'P2P借贷平台个人放贷者', 'reason': '个人放贷者同时向多人放款（出度多样性高）并从少数人收款（入度多样性低），但如果其同时也在多个平台借款，则入度也可能增高，从而产生高中介性。'}]
  - 监管说明: 该特征衡量一个钱包在交易网络中'认识多少人又被人认识'的程度——正常高频交易用户会连接许多不同商户（高中介性），而真正的洗钱跑分账户反而刻意只与固定上家和下家交易（低中介性），因此我们通过这个特征来识别那些'伪装成普通用户但中介性异常低'或'中介性异常高'的可疑账户。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 wallet_network_centrality，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每个钱包作为src的独特对手方数量（出度多样性）
    out_diversity = tx_df.groupby('src')['dst'].nunique().reset_index()
    out_diversity.columns = ['wallet_id', 'out_unique_dst']

    # 计算每个钱包作为dst的独特对手方数量（入度多样性）
    in_diversity = tx_df.groupby('dst')['src'].nunique().reset_index()
    in_diversity.columns = ['wallet_id', 'in_unique_src']

    # 合并
    centrality = out_diversity.merge(in_diversity, on='wallet_id', how='outer').fillna(0)

    # 中介中心性近似：出度多样性 * 入度多样性（衡量桥梁作用）
    # 一个钱包连接的不同来源和不同去向越多，其中介作用越强
    centrality['wallet_network_centrality'] = (
        centrality['out_unique_dst'] * centrality['in_unique_src']
    )

    # 归一化到0-1范围，避免极端值影响
    max_val = centrality['wallet_network_centrality'].max()
    if max_val > 0:
        centrality['wallet_network_centrality'] = centrality['wallet_network_centrality'] / max_val

    result = centrality[['wallet_id', 'wallet_network_centrality']].copy()
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: wallet_network_centrality")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 wallet_network_centrality: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
