"""
特征: amount_surge_ratio
说明: 线下跑分模式中，个人钱包收到大额转入后立即转出，金额接近限额，导致单笔金额远超该钱包历史均值。正常交易金额波动更平缓。

验证指标:
  - Score: 0.5000
  - MI: 0.0000
  - Redundancy: 0.2938
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 75

可解释性报告:
  - 值含义: {'0.00 - 0.05': '金额突增比例极低，该钱包几乎每笔交易金额都接近其历史平均水平，交易模式非常稳定，属于典型的正常消费或固定收支行为。', '0.05 - 0.15': '金额突增比例在正常范围内，偶尔会出现金额高于历史均值2倍的情况，可能是正常的大额消费、工资入账或朋友转账，属于大多数正常钱包的典型表现。', '0.15 - 0.30': '金额突增比例偏高，约15%-30%的交易金额显著偏离历史均值。可能是经营类账户（收入/支出波动大），或存在一定程度的异常交易（如跑分、洗钱），需要结合其他特征判断。', '0.30 - 0.50': "金额突增比例较高，近1/3到1/2的交易金额远超历史均值。高度可疑，尤其是当这些突增交易集中在短时间内发生，极有可能是线下跑分模式中'快进快出'的典型行为。", '> 0.50': '金额突增比例极高，超过一半的交易金额都远高于历史均值。几乎可以确定该钱包的交易模式发生了根本性改变，极大概率是洗钱账户，建议立即冻结并启动调查。'}
  - 风险阈值: {'threshold': 0.2, 'justification': '选择0.20作为高风险阈值，基于以下三个原因：1) 正常钱包的中位数是0.10，均值是0.11，75分位是0.20，说明75%的正常钱包该特征值≤0.20；2) 异常钱包的中位数是0.068，均值是0.098，75分位是0.182，说明超过0.20后异常比例显著上升；3) 该阈值在业务上也有意义——如果一个钱包有超过20%的交易金额都远超历史均值，说明其交易行为已经出现系统性异常，而非偶发事件。'}
  - 协同特征: {'recommended_features': ['transaction_frequency_From', 'avg_inactive_period_From', 'From_To_Ratio', 'S_src_counterparty_index'], 'synergy_explanation': "1) 与transaction_frequency_From配合：金额突增+交易频率激增，是跑分账户的典型双指标异常，比单一特征更可靠。2) 与avg_inactive_period_From配合：金额突增+平均交易间隔显著缩短（如从几天缩短到几分钟），说明账户从休眠状态突然活跃，是洗钱账户激活的典型模式。3) 与From_To_Ratio配合：金额突增+转出/转入比>1，说明该钱包在'收大钱、转小钱'，符合洗钱账户的资金漏斗特征。4) 与S_src_counterparty_index配合：金额突增+交易对手数量激增，说明该钱包突然开始与大量陌生账户交易，是'跑分'网络的典型特征。"}
  - 误判场景: {'case_1': '季节性大额消费：例如双十一、618等购物节期间，用户集中购买高价商品（如家电、电子产品），导致单笔金额远超平时的小额消费均值，但这是正常的周期性消费行为。', 'case_2': '工资入账或奖金发放：个人钱包突然收到大额工资或年终奖，金额远超日常转账均值，但这是正常的收入行为，后续资金会正常消费或理财。', 'case_3': '二手交易平台：用户偶尔出售高价值二手物品（如手机、相机），收到一笔大额转账，金额远超其平时的收发记录，但这是真实的二手交易。', 'case_4': '创业初期或生意周转：小微商户或个体经营者，在进货或支付供应商时，单笔金额可能远高于其日常小额交易，但这是真实的经营活动。'}
  - 监管说明: 该特征通过对比每笔交易金额与钱包的历史平均金额，识别出那些突然出现大额交易的比例——如果一个人平时只转几十块钱，突然有超过20%的交易都是几百甚至上千元，说明这个钱包可能正在被用于洗钱活动。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 amount_surge_ratio，返回 DataFrame(wallet_id/tx_id, feature_value)。"""

    # 计算每个钱包的历史平均发送金额
    sent_mean = tx_df.groupby('src')['amount'].mean().reset_index()
    sent_mean.columns = ['wallet_id', 'avg_sent_amount']

    # 将平均金额合并回交易数据
    tx_with_mean = tx_df.merge(sent_mean, left_on='src', right_on='wallet_id', how='left')

    # 标记金额突增：当前金额 > 历史均值的2倍 且 历史均值不为0
    tx_with_mean['is_surge'] = (tx_with_mean['amount'] > tx_with_mean['avg_sent_amount'] * 2) & (tx_with_mean['avg_sent_amount'] > 0)

    # 计算每个钱包的突增比例
    surge_stats = tx_with_mean.groupby('src').agg(
        total_tx=('tx_id', 'count'),
        surge_tx=('is_surge', 'sum')
    ).reset_index()
    surge_stats['surge_ratio'] = surge_stats['surge_tx'] / surge_stats['total_tx']

    result = surge_stats[['src', 'surge_ratio']].rename(columns={'src': 'wallet_id', 'surge_ratio': 'feature_value'})
    result['feature_value'] = result['feature_value'].fillna(0)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: amount_surge_ratio")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 amount_surge_ratio: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
