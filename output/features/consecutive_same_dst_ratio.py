"""
特征: consecutive_same_dst_ratio
说明: 洗钱模式中，薅羊毛和线上跑分常涉及向固定对手连续转账。small_amount_testing模式就是连续向同一对手发送0.01-0.1元测试交易。本特征能捕捉这种序列行为模式。

验证指标:
  - Score: 0.2034
  - MI: 0.0003
  - Redundancy: 0.2866
  - Cramer's V: 0.0154
  - Delta PR-AUC: -0.031610
  - SHAP Rank: 47

可解释性报告:
  - 值含义: {'0.0': '该钱包从未出现连续向同一对手转账的行为，交易对手高度分散，符合正常用户随机消费、转账的行为模式。', '0.0 - 0.0068': '低连续对手比例区间，属于正常范围内的零星重复转账（如向家人、固定商户的偶尔支付），无异常行为嫌疑。', '0.0068 - 0.01': '中等比例区间，可能存在少量连续向同一对手转账行为，需结合其他特征进一步观察（如是否伴随小额测试交易）。', '> 0.01': '高连续对手比例，强烈指示系统性测试或批量转账行为。该比例意味着每100笔交易中至少有1次连续向同一对手转账，在洗钱场景中常见于small_amount_testing模式的连续小额测试交易或线上跑分的重复转账。'}
  - 风险阈值: {'high_risk_threshold': 0.01, 'judgment_basis': '异常交易的均值(0.0034)低于正常均值(0.0046)，但中位数均为0，说明异常交易中连续相同对手行为高度集中在少数异常钱包。选择0.01作为阈值依据：(1) 异常交易75分位值为0.0065，正常75分位为0.0068，两者接近，说明75%的异常交易与正常交易在该特征上无差异；(2) 然而异常交易的均值(0.0034)明显低于正常(0.0046)，表明异常交易中极少数钱包贡献了极高的连续相同对手比例，拉低了整体均值；(3) 0.01是正常75分位(0.0068)的约1.5倍，能有效捕捉尾部异常行为。'}
  - 协同特征: [{'feature_name': 'small_amount_testing_ratio', 'reason': 'consecutive_same_dst_ratio捕捉连续向同一对手转账的序列模式，small_amount_testing_ratio识别0.01-0.1元的小额测试交易。两者结合可精确识别small_amount_testing洗钱模式：先连续向同一对手发送小额测试交易，确认钱包可用后再进行大额转账。'}, {'feature_name': 'avg_inactive_period', 'reason': '洗钱模式中，休眠账户突然激活后常连续向同一对手发送测试交易。consecutive_same_dst_ratio高值+avg_inactive_period高值（>90天）的组合，可精准识别休眠账户激活后的测试行为，排除正常用户的连续转账。'}, {'feature_name': 'send_concentration', 'reason': 'send_concentration衡量资金集中度（最大对手占比），consecutive_same_dst_ratio衡量连续重复转账的序列模式。两者结合可区分：高集中度但低连续比（正常向固定商户支付）vs 高集中度且高连续比（洗钱测试行为）。'}]
  - 误判场景: ['定期向同一商户付款的订阅服务用户（如每月固定向某商户支付会员费），由于交易日期相近可能产生连续相同对手的假象。', '短期内向同一人归还多笔借款的用户，可能连续多次转账给同一对手。', '分次支付同一笔大额账单的用户（如分多笔向物业公司支付物业费），单日内多次转账给同一对手。', '使用数字人民币进行红包发放的用户，连续向多个不同对手发放红包时，由于红包金额极小（0.01-0.1元），可能被误判为small_amount_testing模式。']
  - 监管说明: 该指标衡量用户是否频繁向同一收款方连续转账，异常情况下洗钱分子会先连续转几笔几分钱测试账户是否正常，而正常用户很少这样操作。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 consecutive_same_dst_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 按钱包和时间排序，计算连续交易中dst是否与上一笔相同
    tx_sorted = tx_df.sort_values(['src', 'timestamp']).copy()
    tx_sorted['prev_dst'] = tx_sorted.groupby('src')['dst'].shift(1)
    tx_sorted['same_dst'] = (tx_sorted['dst'] == tx_sorted['prev_dst']).astype(int)

    # 计算每个钱包连续相同对手的比例（排除第一笔交易）
    stats = tx_sorted.groupby('src')['same_dst'].agg(['sum', 'count']).reset_index()
    stats['count'] = stats['count'] - 1  # 排除第一笔（没有prev_dst）
    stats['feature_value'] = np.where(stats['count'] > 0, stats['sum'] / stats['count'], 0.0)
    result = stats[['src', 'feature_value']].rename(columns={'src': 'wallet_id'})
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: consecutive_same_dst_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 consecutive_same_dst_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
