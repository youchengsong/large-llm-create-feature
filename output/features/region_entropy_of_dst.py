"""
特征: region_entropy_of_dst
说明: 洗钱行为常涉及跨地区分散转账以规避监管。fan_out和many_to_many模式中，资金被分散到多个不同地区的收款方。通过计算收款方地域的信息熵，可以量化这种地域分散程度。高熵值表明收款方遍布多个地区，符合洗钱特征。

验证指标:
  - Score: 0.1300
  - MI: 0.0011
  - Redundancy: 0.6666
  - Cramer's V: 0.0270
  - Delta PR-AUC: -0.004620
  - SHAP Rank: 49

可解释性报告:
  - 值含义: {'0.0-0.85': '收款方地域高度集中，可能为单一地区或少数地区，常见于正常个人转账或固定商户收款', '0.85-0.95': '收款方地域有一定分散性，但仍在可控范围，可结合其他特征判断是否为异常', '0.95-1.0': '收款方地域高度分散，覆盖多个地区，符合洗钱行为中跨区域分散资金的特征，但正常高频跨区域交易的商户也可能达到此值'}
  - 风险阈值: 高风险阈值设为 0.85 以下（低熵值）。判断依据：异常交易的均值（0.9233）和中位数（0.9424）均低于正常交易（均值0.9574，中位数0.9590），且异常交易的25%分位为0.9139，远低于正常交易的25%分位0.9501。说明洗钱行为中收款方地域分布反而更集中（熵值更低），这与原设计预期相反。低熵值（<0.85）可能对应洗钱模式中'线下跑分'或'商户收款码跑分'，资金最终汇聚到少数控制账户。
  - 协同特征: [{'feature': 'send_concentration', 'reason': "低地域熵值+高发送集中度（资金主要流向少数几个收款方），能更准确识别'线下跑分'模式中资金集中流向少数控制账户的行为。"}, {'feature': 'From_To_Ratio', 'reason': "低地域熵值+高发送/接收比（大量转出很少转入），可识别'商户收款码跑分'中商户钱包大量分散收款后集中转出的行为。"}, {'feature': 'transaction_mode', 'reason': '低地域熵值+fan_in或many_to_many模式，可识别资金最终汇聚到少数地区的洗钱行为，与正常fan_out分散转账形成对比。'}]
  - 误判场景: ['本地商户：只向同一城市或少数几个城市的供应商付款，收款方地域自然集中，熵值低。', '个人工资账户：每月固定转账给家人或房贷账户，收款方地域单一。', '新注册钱包：交易历史短，仅与少数收款方交互，地域熵值天然偏低。', '集团内部资金归集：同一企业集团下多个子公司在同一地区，资金集中汇至总部账户。']
  - 监管说明: 该特征衡量收款方地域的集中程度，洗钱行为中资金最终往往流向少数控制账户，导致收款方地域异常集中（熵值偏低），而正常交易中收款方地域分布更均匀。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 region_entropy_of_dst，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 获取交易中src->dst的映射及dst的地区信息
    # 先提取dst的wallet_id对应的region_code
    dst_region = open_df[['wallet_id', 'region_code']].dropna(subset=['region_code'])
    dst_region.columns = ['dst', 'dst_region']

    # 合并交易和dst地区信息
    tx_with_region = tx_df[['src', 'dst']].merge(dst_region, on='dst', how='left')

    # 只保留有地区信息的交易
    tx_with_region = tx_with_region.dropna(subset=['dst_region'])

    # 对每个src，计算dst地区的分布熵
    def calc_entropy(group):
        counts = group.value_counts()
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy

    region_entropy = tx_with_region.groupby('src')['dst_region'].apply(calc_entropy).reset_index()
    region_entropy.columns = ['wallet_id', 'region_entropy_of_dst']

    # 归一化到[0,1]区间（除以最大可能熵值log2(唯一地区数)）
    unique_regions = open_df['region_code'].nunique()
    max_entropy = np.log2(unique_regions) if unique_regions > 0 else 1
    region_entropy['region_entropy_of_dst'] = region_entropy['region_entropy_of_dst'] / max_entropy

    region_entropy['region_entropy_of_dst'] = region_entropy['region_entropy_of_dst'].fillna(0)

    result = region_entropy
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: region_entropy_of_dst")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 region_entropy_of_dst: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
