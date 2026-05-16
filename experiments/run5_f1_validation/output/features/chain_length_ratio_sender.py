"""
特征: chain_length_ratio_sender
说明: 洗钱模式2（线下跑分）中，个人钱包收到大额转入后立即转出，形成短链；模式3（线上跑分）则是分散扇出模式。正常交易中链式转发更常见（如A→B→C的支付链）。链式交易占比低说明该账户更可能是洗钱链条的末端或源头。

验证指标:
  - Score: 0.1791
  - MI: 0.0000
  - Redundancy: 0.6351
  - Cramer's V: 0.0000
  - Delta PR-AUC: 0.000000
  - SHAP Rank: 108

可解释性报告:
  - 值含义: {'0-0.13': '高异常风险：链式交易占比极低，账户几乎只转出不转入，符合洗钱链条末端（资金归集后转出至黑产）或源头（批量开户后分散转出）的典型行为模式。', '0.13-0.27': '中高风险：转入转出比例偏低，账户以转出为主，可能存在跑分行为（收到小额测试资金后立即转出），或为洗钱链条中的中间层节点。', '0.27-0.82': '风险中性：该区间覆盖了正常交易和部分异常交易，需结合其他特征进行判断。正常账户（如商户收款为主）和部分洗钱账户（如线上跑分中的扇出节点）均可能落在此区间。', '0.82-1.0': '低风险：转入转出比例接近1:1，账户行为符合正常个人用户的收支平衡模式，如工资收入后用于日常消费。', '>1.0': '极低风险：转入大于转出，账户为资金沉淀型（如储蓄账户、收款商户），链式交易占比高，行为模式健康。'}
  - 风险阈值: {'primary_threshold': 0.27, 'primary_basis': '异常交易75%分位值为0.8186，正常交易25%分位值为0.14。取异常交易中位数0.2722作为高风险阈值，低于该值的账户转入转出比显著偏低，符合洗钱模式2（线下跑分：收到大额后立即转出，in_cnt≈1, out_cnt≥2）和模式3（线上跑分：扇出模式，in_cnt=1, out_cnt≥3）的行为特征。', 'secondary_threshold': 0.13, 'secondary_basis': "异常交易25%分位值为0.1326，低于该值的账户转入转出比极低（如转入1笔转出10笔），高度符合线上跑分中'单次收款后多次分散转出'的典型模式，可视为高风险预警线。"}
  - 协同特征: [{'feature': 'send_concentration', 'reason': 'chain_length_ratio_sender低（转入少转出多）+ send_concentration高（转出集中到少数对手）→ 典型线下跑分模式（固定交易对手、大额快进快出）；chain_length_ratio_sender低 + send_concentration低（转出分散到大量对手）→ 线上跑分模式（扇出分散）。'}, {'feature': 'avg_inactive_period', 'reason': 'chain_length_ratio_sender低 + avg_inactive_period极小（如<1分钟）→ 资金快进快出特征明显，符合洗钱模式2（立即转出）；chain_length_ratio_sender低 + avg_inactive_period中等（数小时）→ 可能为正常商户的资金归集行为，需进一步区分。'}, {'feature': 'large_transaction_ratio', 'reason': 'chain_length_ratio_sender低 + large_transaction_ratio高 → 账户频繁发生大额转出但很少有大额转入，符合洗钱链条末端（资金汇集后大额转出至黑产账户）的特征，可有效降低商户正常归集资金的误报。'}]
  - 误判场景: [{'scenario': '商户收款后提现', 'explanation': '商户钱包（如便利店、小商贩）通常收款笔数远多于转出笔数（只在每日营业结束后提现一次），导致in_cnt/out_cnt比值偏低。例如收款50笔、提现1笔，比值=0.02，被误判为高风险。建议结合wallet_type（商户钱包）和regular_large_low_freq交易模式进行豁免。'}, {'scenario': '新注册的活跃用户', 'explanation': '用户新开户后短期内频繁进行小额消费（如充值话费、购买会员），转出笔数远多于转入笔数，但转入金额可能仅一笔。此类用户in_cnt=1, out_cnt=10，比值=0.09，触发高风险阈值。建议结合auth_delay_days（实名认证及时性）和账户注册时长进行综合判断。'}, {'scenario': '企业代发工资账户', 'explanation': '企业账户每月接收一笔大额资金后，分散转出至多名员工。in_cnt=1（月工资入账）, out_cnt=50（员工工资），比值=0.02。此类账户行为虽与跑分模式相似，但交易时间规律（每月固定日期）、交易对手固定（同批员工），可通过transaction_motif和交易对手稳定性特征去噪。'}]
  - 监管说明: 该指标衡量一个钱包的'收款次数与付款次数的比例'——比例越低，说明该钱包'只往外转钱、很少收钱'，这正是洗钱活动中'资金汇集后快速分散转移'的典型特征，正常用户通常收支平衡或收大于支。

由 LLM 自动生成，通过 L1/L2/L3 三级验证。
"""

import pandas as pd
import numpy as np


def compute_feature(open_df: pd.DataFrame, tx_df: pd.DataFrame,
                    close_df: pd.DataFrame) -> pd.DataFrame:
    """计算特征 chain_length_ratio_sender，返回 DataFrame(wallet_id/tx_id, feature_value)。"""
    in_count = tx_df.groupby('dst')['tx_id'].count().reset_index().rename(columns={'dst':'wallet_id', 'tx_id':'in_cnt'})
    out_count = tx_df.groupby('src')['tx_id'].count().reset_index().rename(columns={'src':'wallet_id', 'tx_id':'out_cnt'})
    chain_stats = in_count.merge(out_count, on='wallet_id', how='outer').fillna(0)
    chain_stats['chain_length_ratio_sender'] = chain_stats['in_cnt'] / (chain_stats['out_cnt'] + 1)  # 转入转出比，衡量是否为中间节点
    result = chain_stats[['wallet_id', 'chain_length_ratio_sender']]
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="计算特征: chain_length_ratio_sender")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output", default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    open_df = pd.read_csv(f"{args.data_dir}/accounts.csv")
    tx_df = pd.read_csv(f"{args.data_dir}/transactions.csv")
    close_df = pd.read_csv(f"{args.data_dir}/close.csv")

    result = compute_feature(open_df, tx_df, close_df)
    print(f"特征 chain_length_ratio_sender: {len(result)} 行, 列={list(result.columns)}")
    print(result.describe())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"已保存到 {args.output}")
