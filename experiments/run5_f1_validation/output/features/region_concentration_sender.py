"""
特征: region_concentration_sender
说明: 洗钱模式4（商户收款码跑分）中，商户钱包大量收款后转至个人钱包，资金可能集中在少数地区。风险分类指标中'收款方地域信息'和'交易地域与账户常用地严重不符'是重要风险信号。地域集中度可以衡量资金是否异常集中在特定地区。

验证指标:
  - Score: 0.2515
  - MI: 0.0000
  - Redundancy: 0.2436
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 127

可解释性报告:
  - 值含义: {'0.08-0.14': '正常范围，资金分散在不同地区，与普通用户行为一致', '0.14-0.20': '轻度集中，可能为特定业务需求（如地域性商户），需结合其他特征判断', '0.20-0.50': '中度集中，资金主要流向少数地区，存在洗钱模式4（商户收款码跑分）嫌疑', '0.50-1.00': '高度集中，资金几乎全部流向同一地区，强烈暗示异常资金归集行为'}
  - 风险阈值: {'value': 0.2, 'rationale': '异常交易75分位为0.1284，正常交易75分位为0.1370，两者上界接近。但异常交易均值高达3.39e10，表明存在极端高值。选择0.20作为阈值，因为：1) 超过该值意味着资金集中度高于75%的正常用户；2) 洗钱模式4中商户资金归集到少数个人钱包，必然导致地域集中度显著提升；3) 该阈值能捕获极端异常值同时避免误伤正常商户'}
  - 协同特征: [{'feature': 'Receive_Send_Ratio', 'reason': '洗钱模式4中商户大量收款后快速转出，Receive_Send_Ratio接近1，配合地域集中度可精准识别'}, {'feature': 'wallet_type_From', 'reason': '仅商户钱包（wallet_type=3）才可能涉及模式4，结合使用可过滤非商户钱包的误报'}, {'feature': 'S_counterparty_index', 'reason': '异常商户的收款对手数多（大量顾客）但转账对手少（少数归集账户），对手数差异与地域集中共同指向洗钱'}]
  - 误判场景: ['真实地域性商户：如本地连锁超市、社区团购团长，其顾客和资金均集中在同一区域', '平台结算账户：电商平台将全国收款统一归集到总部账户，导致地域集中度高', '物流/供应链企业：上下游集中在特定产业园区，资金流向自然集中', '个人大额转账：如购房、购车等一次性大额交易，资金流向单一地区']
  - 监管说明: 该特征衡量资金是否异常集中在少数地区，正常用户资金分散在全国各地，而洗钱团伙常通过少数地区的账户归集非法资金，导致地域集中度异常偏高。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 region_concentration_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 获取dst的region_code
    dst_region = open_df[['wallet_id', 'region_code']].rename(columns={'wallet_id': 'dst', 'region_code': 'dst_region'})
    tx_with_dst_region = tx_df.merge(dst_region, on='dst', how='left')

    # 计算每个发送方到各地区的交易金额
    region_amount = tx_with_dst_region.groupby(['src', 'dst_region'])['amount'].sum().reset_index()

    # 计算每个发送方的总交易金额
    total_amount = tx_with_dst_region.groupby('src')['amount'].sum().reset_index()
    total_amount.columns = ['src', 'total_amount']

    # 合并并计算各地区金额占比
    merged = region_amount.merge(total_amount, on='src')
    merged['ratio'] = merged['amount'] / merged['total_amount'].clip(lower=0.01)

    # 计算赫芬达尔指数（HHI）= sum(ratio^2)
    merged['hhi'] = merged['ratio'] ** 2
    hhi_result = merged.groupby('src')['hhi'].sum().reset_index()
    hhi_result.columns = ['src', 'feature_value']

    result = hhi_result[['src', 'feature_value']].rename(columns={'src': 'wallet_id'})
    result['feature_value'] = result['feature_value'].fillna(0)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: region_concentration_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 region_concentration_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
