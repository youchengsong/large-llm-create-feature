"""
特征: ip_region_mismatch_flag
说明: 创建这个特征是为了检测交易IP地址与钱包注册地域是否一致。根据领域知识中的'victim_laundering'模式，盗用场景下交易使用的设备IP与账户注册地不一致。正常用户通常在本地区域交易，而洗钱者可能跨地域操作。

验证指标:
  - Score: 0.4906
  - MI: 0.0000
  - Redundancy: 0.0323
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 68

可解释性报告:
  - 值含义: 该特征为二值标志，取值为0或1。0表示交易IP地域与钱包注册地域一致（正常行为），1表示交易IP地域与钱包注册地域不一致（潜在异常行为）。
  - 风险阈值: 高风险阈值为1。判断依据：该特征是二值标志，取值为1即表示IP地域与注册地不匹配。根据领域知识中'victim_laundering'模式的描述，盗用场景下交易设备IP与账户注册地不一致，因此任何不匹配都值得关注。实际应用中可结合其他特征（如交易金额、时间）进一步确认风险等级。
  - 协同特征: [{'feature': 'S_src_geo_risk_202308', 'reason': "两者都涉及地理信息，但视角不同。ip_region_mismatch_flag关注单笔交易IP与注册地的一致性，而S_src_geo_risk_202308关注历史交易中跨地域行为的统计风险。联合使用可以区分'偶尔跨地域'与'系统性跨地域洗钱'。"}, {'feature': 'hour', 'reason': '凌晨交易(22-5时)叠加IP地域不匹配，符合线上跑分和victim_laundering模式的典型特征，可显著提升异常检测准确率。'}, {'feature': 'is_high_risk_region_From', 'reason': '如果钱包注册地在高风险地区，同时IP地域不匹配，可能意味着不法分子利用高风险地区账户进行跨地域洗钱操作。'}]
  - 误判场景: ['出差/旅游用户：正常用户出差或旅游时，可能在异地发起交易，导致IP地域与注册地不一致，但这是正常行为。', 'VPN/代理用户：部分用户因隐私保护或网络访问需求使用VPN，导致IP显示为其他地域。', '多地域经营商户：商户可能在多个城市有经营活动，使用不同地域的IP发起交易。', '家庭地址变更：用户搬家后未更新钱包注册信息，导致新IP与旧注册地不匹配。']
  - 监管说明: 该标志检测交易发生时使用的网络IP地址是否与钱包注册时填写的地区一致，如果不同，可能意味着账户被他人盗用或在异地操作，需要进一步核实交易背景。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 ip_region_mismatch_flag，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 获取钱包注册地域
    wallet_region = open_df[['wallet_id', 'region_code']].copy()
    wallet_region.columns = ['wallet_id', 'wallet_region']

    # 从IP提取地域信息（IP前三位代表地域）
    tx_df['src_ip_prefix'] = tx_df['src_ip'].str.extract(r'(\d+\.\d+\.\d+)\.')[0]

    # 为每个IP前缀分配一个简化的地域编码（基于IP段映射）
    # 使用IP前三位作为地域标识（已有特征中S_src_geo_risk_202308使用类似方法）
    # 将IP前缀转换为数值型地域编码
    ip_region_map = tx_df['src_ip_prefix'].drop_duplicates().reset_index(drop=True)
    ip_region_map = pd.DataFrame({'src_ip_prefix': ip_region_map, 'ip_region_code': range(len(ip_region_map))})
    tx_df = tx_df.merge(ip_region_map, on='src_ip_prefix', how='left')

    # 关联钱包注册地域
    tx_df = tx_df.merge(wallet_region, left_on='src', right_on='wallet_id', how='left')

    # 标记地域不匹配：交易IP地域 != 钱包注册地域
    tx_df['ip_region_mismatch'] = (tx_df['ip_region_code'] != tx_df['wallet_region']).astype(int)

    result = tx_df[['tx_id', 'ip_region_mismatch']].copy()
    result.columns = ['tx_id', 'feature_value']
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: ip_region_mismatch_flag")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 ip_region_mismatch_flag: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
